#include "ulcnet_nn_stage.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"    /* skn_fill_f32, skn_deinterleave_cf32 */

struct UlcnetNnStage {
    UlcnetModel model;            /* infer == NULL: identity */
    UlcnetGrid grid;

    /* The far analysis owns the FFT handle and the window; the error analysis and the
     * synthesis borrow them and run sequentially within a hop. */
    AcStft* far_analysis;
    AcStft* err_analysis;         /* NULL unless built with err_analysis */
    AcIstft* synthesis;

    float *err_re, *err_im, *far_re, *far_im, *enh_re, *enh_im;   /* n_bins each */

    UlcnetNnDelay last_delay;
    uint64_t frame_index;

    /* Frames still straddling the last alignment boundary: identity, model not stepped.
     * Re-armed, never accumulated. */
    int reprime_frames;

    int phase;                    /* 0 idle, 1 hop begun, 2 model frame open */
    int nn_wanted;                /* the frame has not been fetched and is not reprime */
    int nn_accepted;              /* the model output replaces the error frame */
};

/* The grid's framing: unnormalised spectra, centre 0, envelope-normalised synthesis. */
#define STAGE_OLA AC_OLA_ENV_NORM

static void framing_config(AcStftConfig* sc, const UlcnetGrid* grid) {
    ac_stft_config_default(sc, grid->n_fft);
    sc->ola = STAGE_OLA;
}

/* Bytes of an analysis (is_synthesis 0) or synthesis (1) object that borrows the first
 * analysis object's FFT handle and window; those two tables are the first thing an owning
 * object carves, each in its own 16-byte slot. init() checks the sum against the borrowing
 * configuration. */
static size_t borrowed_bytes(const UlcnetGrid* grid, int is_synthesis) {
    AcStftConfig sc;
    size_t own, tables;
    framing_config(&sc, grid);
    own = is_synthesis ? ac_istft_get_mem_size(&sc) : ac_stft_get_mem_size(&sc);
    tables = ck_add_size(ck_align16_size(fft_get_mem_size(grid->n_fft)),
                         ck_field_size(0, (size_t)grid->n_fft, sizeof(float)));
    return MEM_SIZE_INVALID(tables) || own <= tables ? 0 : own - tables;
}

/* The six spectra of n_bins values, back to back. */
static size_t spectra_bytes(const UlcnetGrid* grid) {
    return ck_field_size(0, 6u * (size_t)grid->n_bins, sizeof(float));
}

size_t ulcnet_nn_stage_mem_size(const UlcnetGrid* grid, int err_analysis) {
    AcStftConfig sc;
    size_t far_b, ana_b, syn_b, total;
    if (!grid || ulcnet_grid_validate(grid) != ULCNET_GRID_OK) return 0;
    framing_config(&sc, grid);
    far_b = ac_stft_get_mem_size(&sc);
    ana_b = borrowed_bytes(grid, 0);
    syn_b = borrowed_bytes(grid, 1);
    if (far_b == 0 || ana_b == 0 || syn_b == 0) return 0;
    total = ck_align16_size(sizeof(UlcnetNnStage));
    total = ck_add_size(total, spectra_bytes(grid));
    total = ck_add_size(total, ck_align16_size(far_b));
    if (err_analysis) total = ck_add_size(total, ck_align16_size(ana_b));
    total = ck_add_size(total, ck_align16_size(syn_b));
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

UlcnetNnStage* ulcnet_nn_stage_init(void* mem, size_t bytes, const UlcnetGrid* grid,
                                    int err_analysis) {
    UlcnetNnStage* s;
    AcStftConfig sc;
    uint8_t* cur;
    size_t total = ulcnet_nn_stage_mem_size(grid, err_analysis);
    size_t far_b, ana_b, syn_b;
    if (!mem || !MEM_IS_ALIGNED16(mem) || total == 0 || bytes < total) return NULL;
    /* The framing objects initialise their own blocks. */
    memset(mem, 0, ALIGN16(sizeof(*s)) + spectra_bytes(grid));
    s = (UlcnetNnStage*)mem;
    s->grid = *grid;
    cur = (uint8_t*)mem + ALIGN16(sizeof(*s));
    s->err_re = (float*)cur;
    s->err_im = s->err_re + grid->n_bins;
    s->far_re = s->err_im + grid->n_bins;
    s->far_im = s->far_re + grid->n_bins;
    s->enh_re = s->far_im + grid->n_bins;
    s->enh_im = s->enh_re + grid->n_bins;
    cur += spectra_bytes(grid);

    framing_config(&sc, grid);
    far_b = ac_stft_get_mem_size(&sc);
    s->far_analysis = ac_stft_init(cur, far_b, &sc);
    if (!s->far_analysis) return NULL;
    cur += ALIGN16(far_b);

    /* Borrow the far analysis's handle and window. */
    if (ac_stft_get_config(s->far_analysis, &sc) != 0) return NULL;
    sc.ola = STAGE_OLA;
    ana_b = ac_stft_get_mem_size(&sc);
    syn_b = ac_istft_get_mem_size(&sc);
    if (ana_b != borrowed_bytes(grid, 0) || syn_b != borrowed_bytes(grid, 1)) return NULL;
    if (err_analysis) {
        s->err_analysis = ac_stft_init(cur, ana_b, &sc);
        if (!s->err_analysis) return NULL;
        cur += ALIGN16(ana_b);
    }
    s->synthesis = ac_istft_init(cur, syn_b, &sc);
    if (!s->synthesis) return NULL;
    return s;
}

const UlcnetGrid* ulcnet_nn_stage_grid(const UlcnetNnStage* s) {
    return s ? &s->grid : NULL;
}

int ulcnet_nn_stage_set_model(UlcnetNnStage* s, const UlcnetModel* model) {
    if (!s) return -1;
    if (model) {
        /* A model that infers must publish a descriptor: a shape disagreement between the graph
         * and the stage's grid is undetectable downstream. */
        if (model->infer && !model->io_descriptor) return -1;
        if (model->io_descriptor &&
            ulcnet_model_io_descriptor_validate(&s->grid, model->io_descriptor) != 0)
            return -1;
        s->model = *model;
    } else {
        memset(&s->model, 0, sizeof(s->model));
    }
    return 0;
}

void ulcnet_nn_stage_reset(UlcnetNnStage* s) {
    if (!s) return;
    ac_stft_reset(s->far_analysis);
    if (s->err_analysis) ac_stft_reset(s->err_analysis);
    ac_istft_reset(s->synthesis);
    memset(&s->last_delay, 0, sizeof(s->last_delay));
    s->frame_index = 0;
    s->phase = 0;
    s->nn_wanted = 0;
    s->nn_accepted = 0;
    s->reprime_frames = 0;
    if (s->model.reset) s->model.reset(s->model.user);
}

/* Alignment bookkeeping of the hop. The first transition to a usable alignment of a mode
 * without generation events counts as a boundary too. */
static void gate_delay(UlcnetNnStage* s, UlcnetNnDelay delay) {
    if (delay.changed || (s->frame_index != 0 && !s->last_delay.solid && delay.solid)) {
        /* Flush the runtime's far attention ring and logit history; the straddling frame is
         * covered by the identity reprime armed here, whether or not a runtime is attached. */
        if (s->model.reset) s->model.reset(s->model.user);
        s->reprime_frames = ULCNET_NN_STAGE_REPRIME_FRAMES;
    }
    s->last_delay = delay;
}

static void start_frame(UlcnetNnStage* s, const float* far_hop) {
    (void)ac_stft_push_planar(s->far_analysis, far_hop, s->far_re, s->far_im);
    s->nn_wanted = 0;
    s->nn_accepted = 0;
    if (s->reprime_frames > 0) s->reprime_frames -= 1;   /* identity; model idle */
    else s->nn_wanted = 1;
    s->phase = 1;
}

static int begin_check(const UlcnetNnStage* s, const void* err_a, const void* err_b,
                       const float* far_hop) {
    if (!s || !err_a || !err_b || !far_hop) return ULCNET_NN_STAGE_INVALID_ARGUMENT;
    if (s->phase != 0) return ULCNET_NN_STAGE_SEQUENCE_ERROR;
    return ULCNET_NN_STAGE_OK;
}

int ulcnet_nn_stage_begin(UlcnetNnStage* s, const float* err_re, const float* err_im,
                          const float* far_hop, UlcnetNnDelay delay) {
    int rc = begin_check(s, err_re, err_im, far_hop);
    size_t bytes;
    if (rc != ULCNET_NN_STAGE_OK) return rc;
    bytes = (size_t)s->grid.n_bins * sizeof(float);
    gate_delay(s, delay);
    memcpy(s->err_re, err_re, bytes);
    memcpy(s->err_im, err_im, bytes);
    start_frame(s, far_hop);
    return ULCNET_NN_STAGE_OK;
}

int ulcnet_nn_stage_begin_complex(UlcnetNnStage* s, const Complex* err,
                                  const float* far_hop, UlcnetNnDelay delay) {
    int rc = begin_check(s, err, err, far_hop);
    if (rc != ULCNET_NN_STAGE_OK) return rc;
    gate_delay(s, delay);
    skn_deinterleave_cf32(err, s->err_re, s->err_im, (size_t)s->grid.n_bins);
    start_frame(s, far_hop);
    return ULCNET_NN_STAGE_OK;
}

int ulcnet_nn_stage_begin_time(UlcnetNnStage* s, const float* err_hop,
                               const float* far_hop, UlcnetNnDelay delay) {
    int rc = begin_check(s, err_hop, err_hop, far_hop);
    if (rc != ULCNET_NN_STAGE_OK) return rc;
    if (!s->err_analysis) return ULCNET_NN_STAGE_INVALID_ARGUMENT;
    gate_delay(s, delay);
    (void)ac_stft_push_planar(s->err_analysis, err_hop, s->err_re, s->err_im);
    start_frame(s, far_hop);
    return ULCNET_NN_STAGE_OK;
}

int ulcnet_nn_stage_nn_begin(UlcnetNnStage* s, UlcnetNnJob* job) {
    if (!s || !job) return ULCNET_NN_STAGE_INVALID_ARGUMENT;
    if (s->phase != 1) return ULCNET_NN_STAGE_SEQUENCE_ERROR;
    if (!s->nn_wanted) return 0;
    s->nn_wanted = 0;
    /* Full-write contract: NaN staging lets nn_end's finite guard catch a bin the model
     * left unwritten. */
    skn_fill_f32(s->enh_re, (size_t)s->grid.n_bins, NAN);
    skn_fill_f32(s->enh_im, (size_t)s->grid.n_bins, NAN);
    job->err_re = s->err_re; job->err_im = s->err_im;
    job->far_re = s->far_re; job->far_im = s->far_im;
    job->out_re = s->enh_re; job->out_im = s->enh_im;
    s->phase = 2;
    return 1;
}

int ulcnet_nn_stage_nn_end(UlcnetNnStage* s, int model_rc) {
    if (!s) return ULCNET_NN_STAGE_INVALID_ARGUMENT;
    if (s->phase != 2) return ULCNET_NN_STAGE_SEQUENCE_ERROR;
    /* Fail-open: a failed run or any non-finite bin leaves the identity. */
    s->nn_accepted = model_rc == 0 &&
                     ulcnet_frame_is_finite(s->enh_re, s->enh_im, s->grid.n_bins);
    s->phase = 1;
    return ULCNET_NN_STAGE_OK;
}

int ulcnet_nn_stage_end(UlcnetNnStage* s, float* out) {
    const float* spec_re;
    const float* spec_im;
    int written;
    if (!s || !out) return ULCNET_NN_STAGE_INVALID_ARGUMENT;
    if (s->phase != 1 && s->phase != 2) return ULCNET_NN_STAGE_SEQUENCE_ERROR;
    /* nn_accepted is set only by nn_end, so a frame left open or never fetched is the identity. */
    spec_re = s->nn_accepted ? s->enh_re : s->err_re;
    spec_im = s->nn_accepted ? s->enh_im : s->err_im;
    s->phase = 0;
    written = ac_istft_push_planar(s->synthesis, spec_re, spec_im, out);
    if (written == 0) {
        /* hop #0: the synthesis holds its first frame back. */
        memset(out, 0, (size_t)s->grid.hop * sizeof(float));
    } else if (written != s->grid.hop) {
        return ULCNET_NN_STAGE_DSP_ERROR;
    }
    s->frame_index += 1;
    return ULCNET_NN_STAGE_OK;
}

int ulcnet_nn_stage_process(UlcnetNnStage* s, float* out) {
    UlcnetNnJob job;
    /* No runtime attached: the frame stays unfetched and is the identity. */
    if (s && s->model.infer && ulcnet_nn_stage_nn_begin(s, &job) == 1)
        (void)ulcnet_nn_stage_nn_end(
            s, s->model.infer(s->model.user, job.err_re, job.err_im,
                              job.far_re, job.far_im, job.out_re, job.out_im));
    return ulcnet_nn_stage_end(s, out);
}
