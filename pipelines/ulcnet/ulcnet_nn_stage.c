#include "ulcnet_nn_stage.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "utility/mem_align.h"
#include "utility/simd_kernel_nn.h"    /* skn_fill_f32, skn_(de)interleave_cf32 */

struct UlcnetNnStage {
    UlcnetModel model;            /* infer == NULL: identity */
    UlcnetGrid grid;

    float *enh_re, *enh_im, *far_re, *far_im, *err_re, *err_im;   /* n_bins each */

    UlcnetNnDelay last_delay;
    uint64_t frame_index;

    /* Frames still straddling the last alignment boundary: identity, model not stepped.
     * Re-armed, never accumulated. */
    int reprime_frames;

    int phase;                    /* 0 idle, 1 hop begun, 2 model frame open */
    int nn_wanted;                /* the frame has not been fetched and is not reprime */
    int nn_accepted;              /* the model output replaces the error frame */
};

/* The six planar spectra of n_bins values, back to back (the model's tensors are planar). */
static size_t spectra_bytes(const UlcnetGrid* grid) {
    return ck_field_size(0, 6u * (size_t)grid->n_bins, sizeof(float));
}

size_t ulcnet_nn_stage_mem_size(const UlcnetGrid* grid) {
    size_t total;
    if (!grid || ulcnet_grid_validate(grid) != ULCNET_GRID_OK) return 0;
    total = ck_add_size(ck_align16_size(sizeof(UlcnetNnStage)), spectra_bytes(grid));
    total = ck_align16_size(total);
    return MEM_SIZE_INVALID(total) ? 0 : total;
}

UlcnetNnStage* ulcnet_nn_stage_init(void* mem, size_t bytes, const UlcnetGrid* grid) {
    UlcnetNnStage* s;
    uint8_t* cur;
    size_t total = ulcnet_nn_stage_mem_size(grid);
    if (!mem || !MEM_IS_ALIGNED16(mem) || total == 0 || bytes < total) return NULL;
    memset(mem, 0, ALIGN16(sizeof(*s)) + spectra_bytes(grid));
    s = (UlcnetNnStage*)mem;
    s->grid = *grid;
    cur = (uint8_t*)mem + ALIGN16(sizeof(*s));
    s->enh_re = (float*)cur;
    s->enh_im = s->enh_re + grid->n_bins;
    s->far_re = s->enh_im + grid->n_bins;
    s->far_im = s->far_re + grid->n_bins;
    s->err_re = s->far_im + grid->n_bins;
    s->err_im = s->err_re + grid->n_bins;
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
            (ulcnet_model_io_descriptor_validate(&s->grid, model->io_descriptor) != 0 ||
             ulcnet_model_io_descriptor_frames(model->io_descriptor) != 1))
            return -1;
        s->model = *model;
    } else {
        memset(&s->model, 0, sizeof(s->model));
    }
    return 0;
}

void ulcnet_nn_stage_reset(UlcnetNnStage* s) {
    if (!s) return;
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

int ulcnet_nn_stage_begin_spec(UlcnetNnStage* s, const Complex* err_spec, const Complex* far_spec,
                               UlcnetNnDelay delay) {
    size_t n;
    if (!s || !err_spec || !far_spec) return ULCNET_NN_STAGE_INVALID_ARGUMENT;
    if (s->phase != 0) return ULCNET_NN_STAGE_SEQUENCE_ERROR;
    n = (size_t)s->grid.n_bins;
    gate_delay(s, delay);
    skn_deinterleave_cf32(err_spec, s->err_re, s->err_im, n);
    skn_deinterleave_cf32(far_spec, s->far_re, s->far_im, n);
    s->nn_wanted = 0;
    s->nn_accepted = 0;
    if (s->reprime_frames > 0) s->reprime_frames -= 1;   /* identity; model idle */
    else s->nn_wanted = 1;
    s->phase = 1;
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

int ulcnet_nn_stage_end_spec(UlcnetNnStage* s, Complex* out_spec) {
    if (!s || !out_spec) return ULCNET_NN_STAGE_INVALID_ARGUMENT;
    if (s->phase != 1 && s->phase != 2) return ULCNET_NN_STAGE_SEQUENCE_ERROR;
    /* nn_accepted is set only by nn_end, so a frame left open or never fetched is the identity. */
    if (s->nn_accepted) skn_interleave_cf32(s->enh_re, s->enh_im, out_spec, (size_t)s->grid.n_bins);
    else skn_interleave_cf32(s->err_re, s->err_im, out_spec, (size_t)s->grid.n_bins);
    s->phase = 0;
    s->frame_index += 1;
    return ULCNET_NN_STAGE_OK;
}

int ulcnet_nn_stage_process(UlcnetNnStage* s, Complex* out_spec) {
    UlcnetNnJob job;
    /* No runtime attached: the frame stays unfetched and is the identity. */
    if (s && s->model.infer && ulcnet_nn_stage_nn_begin(s, &job) == 1)
        (void)ulcnet_nn_stage_nn_end(
            s, s->model.infer(s->model.user, job.err_re, job.err_im,
                              job.far_re, job.far_im, job.out_re, job.out_im));
    return ulcnet_nn_stage_end_spec(s, out_spec);
}
