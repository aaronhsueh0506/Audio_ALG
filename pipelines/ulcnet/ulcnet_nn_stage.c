#include "ulcnet_nn_stage.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"    /* skn_fill_f32, skn_deinterleave_cf32 */

_Static_assert(ULCNET_N_FFT == 512 || ULCNET_N_FFT == 1024,
               "the stage supports FFT 512 (16 kHz) or 1024 (48 kHz)");
_Static_assert(ULCNET_HOP == ULCNET_N_FFT / 2 && ULCNET_BINS == ULCNET_N_FFT / 2 + 1,
               "the compiled grid is a no-padding 50 % overlap grid");

struct UlcnetNnStage {
    UlcnetModel model;            /* infer == NULL: identity */

    /* One FFT handle and window table serve every framing object; the objects run
     * sequentially within a hop. */
    FftHandle* fft;
    AcStft* far_analysis;
    AcStft* err_analysis;         /* NULL unless built with err_analysis */
    AcIstft* synthesis;
    float window[ULCNET_N_FFT];

    float err_re[ULCNET_BINS], err_im[ULCNET_BINS];
    float far_re[ULCNET_BINS], far_im[ULCNET_BINS];
    float enh_re[ULCNET_BINS], enh_im[ULCNET_BINS];

    UlcnetNnDelay last_delay;
    uint64_t frame_index;

    /* Frames still straddling the last alignment boundary: identity, model not stepped.
     * Re-armed, never accumulated. */
    int reprime_frames;

    int phase;                    /* 0 idle, 1 hop begun, 2 model frame open */
    int nn_wanted;                /* the frame has not been fetched and is not reprime */
    int nn_accepted;              /* the model output replaces the error frame */
};

/* Bytes of an analysis (is_synthesis 0) or synthesis (1) object that borrows the shared handle
 * and window table. */
static size_t framing_bytes(int is_synthesis) {
    AcStftConfig sc;
    size_t own, shared;
    ulcnet_framing_config(&sc, NULL, NULL, 0);
    own = is_synthesis ? ac_istft_get_mem_size(&sc) : ac_stft_get_mem_size(&sc);
    shared = ALIGN16(fft_get_mem_size(ULCNET_N_FFT)) +
             ALIGN16((size_t)ULCNET_N_FFT * sizeof(float));
    return own > shared ? own - shared : 0;
}

size_t ulcnet_nn_stage_mem_size(int err_analysis) {
    size_t fft_b = fft_get_mem_size(ULCNET_N_FFT);
    size_t ana_b = framing_bytes(0);
    size_t syn_b = framing_bytes(1);
    if (fft_b == 0 || ana_b == 0 || syn_b == 0) return 0;
    return ALIGN16(sizeof(UlcnetNnStage)) + ALIGN16(fft_b) + ALIGN16(ana_b) +
           (err_analysis ? ALIGN16(ana_b) : 0) + ALIGN16(syn_b);
}

UlcnetNnStage* ulcnet_nn_stage_init(void* mem, size_t bytes, int err_analysis) {
    UlcnetNnStage* s;
    AcStftConfig sc;
    uint8_t* cur;
    size_t total = ulcnet_nn_stage_mem_size(err_analysis);
    size_t fft_b = fft_get_mem_size(ULCNET_N_FFT);
    size_t ana_b = framing_bytes(0);
    size_t syn_b = framing_bytes(1);
    if (!mem || !MEM_IS_ALIGNED16(mem) || total == 0 || bytes < total) return NULL;
    memset(mem, 0, total);
    s = (UlcnetNnStage*)mem;
    cur = (uint8_t*)mem + ALIGN16(sizeof(*s));

    s->fft = fft_init(cur, fft_b, ULCNET_N_FFT);
    if (!s->fft || fft_get_n_freqs(s->fft) != ULCNET_BINS) return NULL;
    cur += ALIGN16(fft_b);

    ac_window_build(s->window, ULCNET_N_FFT);
    ulcnet_framing_config(&sc, s->fft, s->window, 0);
    s->far_analysis = ac_stft_init(cur, ana_b, &sc);
    cur += ALIGN16(ana_b);
    if (err_analysis) {
        s->err_analysis = ac_stft_init(cur, ana_b, &sc);
        cur += ALIGN16(ana_b);
        if (!s->err_analysis) return NULL;
    }
    s->synthesis = ac_istft_init(cur, syn_b, &sc);
    if (!s->far_analysis || !s->synthesis) return NULL;
    return s;
}

int ulcnet_nn_stage_set_model(UlcnetNnStage* s, const UlcnetModel* model) {
    if (!s) return -1;
    if (model) {
        /* A model that infers must publish a descriptor: a shape disagreement between the graph
         * and the host-side rings is undetectable downstream. */
        if (model->infer && !model->io_descriptor) return -1;
        if (model->io_descriptor &&
            ulcnet_model_io_descriptor_validate(model->io_descriptor) != 0)
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
    if (rc != ULCNET_NN_STAGE_OK) return rc;
    gate_delay(s, delay);
    memcpy(s->err_re, err_re, sizeof(s->err_re));
    memcpy(s->err_im, err_im, sizeof(s->err_im));
    start_frame(s, far_hop);
    return ULCNET_NN_STAGE_OK;
}

int ulcnet_nn_stage_begin_complex(UlcnetNnStage* s, const Complex* err,
                                  const float* far_hop, UlcnetNnDelay delay) {
    int rc = begin_check(s, err, err, far_hop);
    if (rc != ULCNET_NN_STAGE_OK) return rc;
    gate_delay(s, delay);
    skn_deinterleave_cf32(err, s->err_re, s->err_im, ULCNET_BINS);
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
    skn_fill_f32(s->enh_re, ULCNET_BINS, NAN);
    skn_fill_f32(s->enh_im, ULCNET_BINS, NAN);
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
    s->nn_accepted = model_rc == 0 && ulcnet_frame_is_finite(s->enh_re, s->enh_im);
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
        memset(out, 0, (size_t)ULCNET_HOP * sizeof(float));
    } else if (written != ULCNET_HOP) {
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
