/* aiaec.h implementation: a block-level (hop_in samples per call) wrapper over deepvqe_stream's and
 * ulcnet_stream's hop-level (one model_hop samples per call) PCM APIs. The catalog below is the only
 * place that knows which (model, sample_rate, n_fft) combinations exist and which of their
 * frames_per_call variants have an actual shipped graph. The graph's port wiring is the sample's;
 * this file keeps only the per-hop signal and head views AiaecWork carries.
 *
 * The product binds one model (aiaec_product.h). AIAEC_ENABLE_DEEPVQE_S / AIAEC_ENABLE_ALIGN_ULCNET
 * (the Makefile's AIAEC_MODEL knob) let a product link only the model it ships, so its .o never
 * references the other model's symbols (nm -u proves it). aiaec.h is identical across every variant;
 * a build whose bound model is not compiled in refuses every entry with AIAEC_ERR_MODEL_ID. */

#include "aiaec.h"
#include "aiaec_internal.h"
#include "aiaec_product.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "stream_mem.h"

#if !AIAEC_ENABLE_DEEPVQE_S && !AIAEC_ENABLE_ALIGN_ULCNET
#error "aiaec.c: at least one of AIAEC_ENABLE_DEEPVQE_S/AIAEC_ENABLE_ALIGN_ULCNET must be set"
#endif

#if AIAEC_ENABLE_DEEPVQE_S
#include "deepvqe_stream.h"
#include "deepvqe_prepost.h"
#endif
#if AIAEC_ENABLE_ALIGN_ULCNET
#include "ulcnet_stream.h"
#endif

/* The one model this product ships (aiaec_product.h). Never a caller input. */
#define AIAEC_BOUND_MODEL_ID ((int)(AIAEC_PRODUCT_MODEL_ID))

static int aiaec_bound_model_in_build(void)
{
    return 0
#if AIAEC_ENABLE_DEEPVQE_S
        || AIAEC_BOUND_MODEL_ID == AIAEC_MODEL_ID_DEEPVQE_S
#endif
#if AIAEC_ENABLE_ALIGN_ULCNET
        || AIAEC_BOUND_MODEL_ID == AIAEC_MODEL_ID_ALIGN_ULCNET
#endif
        ;
}

/* graph_path is NULL for a frames_per_call that is architecturally meaningful but has no delivered
 * graph yet -- the resolver treats it exactly like "not in this list at all", never as "pick some
 * other graph", since N changes the model's own input window length. */
typedef struct AiaecFramesVariant {
    int frames_per_call;
    const char *graph_path;
} AiaecFramesVariant;

typedef struct AiaecModelEntry {
    int model_id;
    int sample_rate, n_fft;
    float max_delay_seconds;   /* physical delay span, grid-independent; delay_depth (D) is derived at
                                * resolve time (ceilf(max_delay_seconds * sample_rate / hop)) */
    const AiaecFramesVariant *variants;
    int n_variants;
} AiaecModelEntry;

/* AIAEC_TEST_CATALOG: defined only by this directory's test builds. Substitutes placeholder graph
 * paths so a test binary can drive the real gated public API end to end. The production catalog has
 * a delivered graph for only one row. */
#ifdef AIAEC_TEST_CATALOG
static const AiaecFramesVariant AIAEC_DEEPVQE_16K_FFT256_VARIANTS[] = {
    { 1, "test://aiaec/deepvqe_s/16k_fft256_n1" },
    { 8, "test://aiaec/deepvqe_s/16k_fft256_n8" },
};
static const AiaecFramesVariant AIAEC_DEEPVQE_16K_FFT512_VARIANTS[] = {
    { 1, "test://aiaec/deepvqe_s/16k_fft512_n1" },
    { 8, (const char *)0 },
};
static const AiaecFramesVariant AIAEC_ULCNET_16K_FFT256_VARIANTS[] = {
    { 1, "test://aiaec/align_ulcnet/16k_fft256_n1" },
    { 8, "test://aiaec/align_ulcnet/16k_fft256_n8" },   /* the packet graph (layout 19) */
};
static const AiaecFramesVariant AIAEC_ULCNET_16K_FFT512_VARIANTS[] = {
    { 1, "test://aiaec/align_ulcnet/16k_fft512_n1" },
    { 8, (const char *)0 },
};
#else
static const AiaecFramesVariant AIAEC_DEEPVQE_16K_FFT256_VARIANTS[] = {
    { 1, (const char *)0 },   /* exported with random weights only; no trained/quantized graph yet */
    { 8, (const char *)0 },
};
/* Shipped 16k/512 grid (hop 256). N == 1 only; the default 16x16 quantization's path. */
static const AiaecFramesVariant AIAEC_DEEPVQE_16K_FFT512_VARIANTS[] = {
    { 1, "/mnt/sd/audio_model/16x16/deepvqes/nvt_model.bin" },
    { 8, (const char *)0 },
};
static const AiaecFramesVariant AIAEC_ULCNET_16K_FFT256_VARIANTS[] = {
    { 1, (const char *)0 },
    { 8, (const char *)0 },
};
/* Registered grid without a delivered graph: refused at init until one is. */
static const AiaecFramesVariant AIAEC_ULCNET_16K_FFT512_VARIANTS[] = {
    { 1, (const char *)0 },
    { 8, (const char *)0 },
};
#endif

/* 0.5 s target physical delay span for every row: D = 32 at hop 256, 63 at hop 128. */
#define AIAEC_MAX_DELAY_SECONDS 0.5f

static const AiaecModelEntry AIAEC_CATALOG[] = {
    { AIAEC_MODEL_ID_DEEPVQE_S,    16000, 256, AIAEC_MAX_DELAY_SECONDS, AIAEC_DEEPVQE_16K_FFT256_VARIANTS, 2 },
    { AIAEC_MODEL_ID_DEEPVQE_S,    16000, 512, AIAEC_MAX_DELAY_SECONDS, AIAEC_DEEPVQE_16K_FFT512_VARIANTS, 2 },
    { AIAEC_MODEL_ID_ALIGN_ULCNET, 16000, 256, AIAEC_MAX_DELAY_SECONDS, AIAEC_ULCNET_16K_FFT256_VARIANTS,  2 },
    { AIAEC_MODEL_ID_ALIGN_ULCNET, 16000, 512, AIAEC_MAX_DELAY_SECONDS, AIAEC_ULCNET_16K_FFT512_VARIANTS,  2 },
};
#define AIAEC_CATALOG_ROWS ((int)(sizeof(AIAEC_CATALOG) / sizeof(AIAEC_CATALOG[0])))

/* Resolves cfg to a catalog entry, N, that variant's graph path and the derived delay depth. Order:
 * the bound model is in this build; (model, sr, fft) row match; hop_in divisibility; then the
 * shipped-variants list, where "shipped" also requires a non-NULL graph_path. graph_path_out and
 * delay_depth_out may be NULL. */
static int aiaec_resolve(const AiaecConfig *cfg, const AiaecModelEntry **entry_out, int *n_out,
                         const char **graph_path_out, int *delay_depth_out)
{
    int i, j, model_hop, n;
    const AiaecModelEntry *entry = (const AiaecModelEntry *)0;
    const char *graph_path = (const char *)0;

    if (!cfg || !entry_out || !n_out) return AIAEC_ERR_NULL;
    if (!aiaec_bound_model_in_build()) return AIAEC_ERR_MODEL_ID;

    for (i = 0; i < AIAEC_CATALOG_ROWS; ++i) {
        if (AIAEC_CATALOG[i].model_id == AIAEC_BOUND_MODEL_ID &&
            AIAEC_CATALOG[i].sample_rate == cfg->sr && AIAEC_CATALOG[i].n_fft == cfg->fft) {
            entry = &AIAEC_CATALOG[i];
            break;
        }
    }
    if (!entry) return AIAEC_ERR_GRID;

    model_hop = cfg->fft / 2;
    if (model_hop <= 0 || cfg->hop_in <= 0 || cfg->hop_in % model_hop != 0) return AIAEC_ERR_IN_HOP;
    n = cfg->hop_in / model_hop;

    for (j = 0; j < entry->n_variants; ++j) {
        if (entry->variants[j].frames_per_call == n) { graph_path = entry->variants[j].graph_path; break; }
    }
    if (j == entry->n_variants || !graph_path) return AIAEC_ERR_FRAMES_PER_CALL;

    *entry_out = entry;
    *n_out = n;
    if (graph_path_out) *graph_path_out = graph_path;
    if (delay_depth_out)
        *delay_depth_out = (int)ceilf(entry->max_delay_seconds * (float)entry->sample_rate / (float)model_hop);
    return AIAEC_OK;
}

#if AIAEC_ENABLE_DEEPVQE_S
static int aiaec_build_grid(const AiaecModelEntry *entry, int n, int delay_depth, DeepVqeGrid *grid)
{
    DeepVqeGridSpec spec;

    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = entry->sample_rate;
    spec.n_fft = entry->n_fft;
    spec.delay_depth = delay_depth;
    spec.frames_per_call = n;
    return deepvqe_grid_make(grid, &spec) == DEEPVQE_GRID_OK ? AIAEC_OK : AIAEC_ERR_GRID;
}
#endif

#if AIAEC_ENABLE_ALIGN_ULCNET
/* UlcnetGrid has no frames_per_call: N is purely a block concept of this layer for this model. */
static int aiaec_build_grid_ulcnet(const AiaecModelEntry *entry, int delay_depth, UlcnetGrid *grid)
{
    UlcnetGridSpec spec;

    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = entry->sample_rate;
    spec.n_fft = entry->n_fft;
    spec.delay_depth = delay_depth;
    return ulcnet_grid_make(grid, &spec) == ULCNET_GRID_OK ? AIAEC_OK : AIAEC_ERR_GRID;
}
#endif

/* ---------------------------------------------------------------------------------------------
 * AiaecWork views: the per-hop signal and head tensors, in graph order.
 * ------------------------------------------------------------------------------------------- */

static void aiaec_set_in(AiaecTensorIn *t, const float *data, unsigned int b, unsigned int c, unsigned int h,
                         unsigned int w)
{
    t->data = data;
    t->elements = (size_t)b * c * h * w;
    t->b = b; t->c = c; t->h = h; t->w = w;
}

static void aiaec_set_out(AiaecTensorOut *t, float *data, unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w)
{
    t->data = data;
    t->elements = (size_t)b * c * h * w;
    t->b = b; t->c = c; t->h = h; t->w = w;
}

#if AIAEC_ENABLE_DEEPVQE_S
/* mic/far are [1, 2 (real plane, imaginary plane), N, n_bins]; the CCM taps are
 * [1, N, n_bins, TIME_ORDER * FREQ_TAPS * 2], oldest frame first along the second axis. */
static void aiaec_block_fill_deepvqe(const DeepVqeGrid *grid, const DeepVqePrepostInputs *in,
                                     const DeepVqePrepostOutputs *out, unsigned int generation,
                                     AiaecWork *work)
{
    const unsigned int nn = (unsigned int)grid->frames_per_call, bins = (unsigned int)grid->n_bins;

    memset(work, 0, sizeof(*work));
    work->action = AIAEC_SUBMIT;
    aiaec_set_in(&work->inputs[0], in->mic, 1u, 2u, nn, bins);
    aiaec_set_in(&work->inputs[1], in->far, 1u, 2u, nn, bins);
    work->input_count = 2;
    aiaec_set_out(&work->outputs[0], out->taps, 1u, nn, bins, (unsigned int)(DEEPVQE_TIME_ORDER * DEEPVQE_FREQ_TAPS * 2));
    work->output_count = 1;
    work->state_generation = generation;
}
#endif /* AIAEC_ENABLE_DEEPVQE_S */

#if AIAEC_ENABLE_ALIGN_ULCNET
static void aiaec_set_restart(AiaecWork *work, uint64_t mask)
{
    work->synthesis_restart_mask = mask;
    work->synthesis_restarted = mask != 0u;
}

/* Four signals [1, 1, N, n_bins] (error_mag, far_mag, error_cos, error_sin; N frames, oldest first); at N > 1 a
 * fifth, frame_reset [1, 1, 1, N], the last signal; one head, the complex mask planar [1, 2, N, n_bins] (the real
 * plane of all N frames, then the imaginary). */
static void aiaec_block_fill_align(const UlcnetGrid *grid, int n, const UlcnetModelIoInputs *in,
                                   const UlcnetModelIoOutputs *out, uint32_t generation,
                                   AiaecWork *work)
{
    const unsigned int nn = (unsigned int)n, bins = (unsigned int)grid->n_bins;

    memset(work, 0, sizeof(*work));
    work->action = AIAEC_SUBMIT;
    aiaec_set_in(&work->inputs[0], in->error_mag, 1u, 1u, nn, bins);
    aiaec_set_in(&work->inputs[1], in->far_mag, 1u, 1u, nn, bins);
    aiaec_set_in(&work->inputs[2], in->error_cos, 1u, 1u, nn, bins);
    aiaec_set_in(&work->inputs[3], in->error_sin, 1u, 1u, nn, bins);
    work->input_count = 4;
    if (n > 1) aiaec_set_in(&work->inputs[work->input_count++], in->frame_reset, 1u, 1u, 1u, nn);
    aiaec_set_out(&work->outputs[0], out->output, 1u, 2u, nn, bins);
    work->output_count = 1;
    work->state_generation = (unsigned int)generation;
}
#endif /* AIAEC_ENABLE_ALIGN_ULCNET */

/* ---------------------------------------------------------------------------------------------
 * Context
 * ------------------------------------------------------------------------------------------- */

/* One core serves all three families; the public context types are thin wrappers so a pool or ctx of
 * one family cannot be handed to another family's entry at compile time. */
typedef enum AiaecFamily { AIAEC_FAMILY_PCM = 0, AIAEC_FAMILY_SPEC = 1, AIAEC_FAMILY_FREQ = 2 } AiaecFamily;

typedef struct AiaecCore {
    AiaecConfig cfg;
    AiaecFamily family;
    int model_id;
    int frames_per_call;   /* N */
    int hop;
    int n_bins;
    union {
#if AIAEC_ENABLE_DEEPVQE_S
        struct {
            DeepVqeStream *inner;
            DeepVqeGrid    grid;
            int16_t       *stash;     /* PCM family: (N - 1) * hop; NULL when N == 1 */
            float         *stash_f;   /* _spec/_freq: (N - 1) * n_bins real, then as many imaginary */
        } deepvqe;
#endif
#if AIAEC_ENABLE_ALIGN_ULCNET
        struct {
            UlcnetStream *inner;
            UlcnetGrid    grid;
        } align;
#endif
    } u;
    int block_open;
} AiaecCore;

struct AiaecCtx { AiaecCore core; };
struct AiaecSpecCtx  { AiaecCore core; };
struct AiaecFreqCtx  { AiaecCore core; };

/* Carves (base != NULL) or just sizes (base == NULL) the core header, the DeepVQE-S stash (if
 * applicable) and the bound inner stream's own pool region, in that order. dgrid/ugrid are
 * `const void *` (cast back inside each #if branch) so the signature compiles in every build
 * variant, even one where only one of DeepVqeGrid/UlcnetGrid exists. */
static int aiaec_pool_layout(AiaecFamily family, int n, const void *dgrid, const void *ugrid,
                             unsigned char *base, AiaecCore **core_out, void **stash_out,
                             void **inner_pool_out, size_t *inner_bytes_out, size_t *total_out)
{
    size_t cursor = 0;
    void *p;

    if (stream_carve(base, &cursor, sizeof(AiaecCore), &p) != 0) return -1;
    if (core_out) *core_out = (AiaecCore *)p;

    if (AIAEC_BOUND_MODEL_ID == AIAEC_MODEL_ID_DEEPVQE_S) {
#if AIAEC_ENABLE_DEEPVQE_S
        const DeepVqeGrid *grid = (const DeepVqeGrid *)dgrid;
        DeepVqeStreamConfig scfg;
        DeepVqeStreamMemReq sreq;
        size_t stash_bytes = 0;

        (void)ugrid;
        if (n > 1) {
            stash_bytes = family == AIAEC_FAMILY_PCM
                ? (size_t)(n - 1) * (size_t)grid->hop * sizeof(int16_t)
                : (size_t)2u * (size_t)(n - 1) * (size_t)grid->n_bins * sizeof(float);
            if (stream_carve(base, &cursor, stash_bytes, &p) != 0) return -1;
        } else {
            p = (void *)0;
        }
        if (stash_out) *stash_out = p;

        memset(&scfg, 0, sizeof(scfg));
        scfg.grid = *grid;
        if (deepvqe_stream_get_mem_size(&scfg, &sreq) != 0) return -1;
        if (stream_carve(base, &cursor, (size_t)sreq.bytes, &p) != 0) return -1;
        if (inner_pool_out) *inner_pool_out = p;
        if (inner_bytes_out) *inner_bytes_out = (size_t)sreq.bytes;
#else
        (void)family; (void)dgrid; (void)ugrid; (void)n;
        return -1;
#endif
    } else {   /* AIAEC_MODEL_ID_ALIGN_ULCNET: the inner stream batches N hops itself, no stash */
#if AIAEC_ENABLE_ALIGN_ULCNET
        const UlcnetGrid *grid = (const UlcnetGrid *)ugrid;
        UlcnetStreamConfig ucfg;
        UlcnetStreamMemReq ureq;

        (void)family; (void)dgrid;
        if (stash_out) *stash_out = (void *)0;

        memset(&ucfg, 0, sizeof(ucfg));
        ucfg.grid = *grid;   /* no state registration: the accelerator owns its own state copy */
        ucfg.frames_per_call = n;
        if (ulcnet_stream_get_mem_size(&ucfg, &ureq) != 0) return -1;
        if (stream_carve(base, &cursor, (size_t)ureq.bytes, &p) != 0) return -1;
        if (inner_pool_out) *inner_pool_out = p;
        if (inner_bytes_out) *inner_bytes_out = (size_t)ureq.bytes;
#else
        (void)family; (void)dgrid; (void)ugrid; (void)n;
        return -1;
#endif
    }

    if (total_out) *total_out = cursor;
    return 0;
}

static int aiaec_core_mem_size(const AiaecConfig *cfg, AiaecFamily family, StreamMemReq *req)
{
    const AiaecModelEntry *entry;
    int n, delay_depth, status;
    size_t total = 0;

    if (!req) return AIAEC_ERR_NULL;
    status = aiaec_resolve(cfg, &entry, &n, (const char **)0, &delay_depth);
    if (status != AIAEC_OK) return status;

    if (AIAEC_BOUND_MODEL_ID == AIAEC_MODEL_ID_DEEPVQE_S) {
#if AIAEC_ENABLE_DEEPVQE_S
        DeepVqeGrid dgrid;
        status = aiaec_build_grid(entry, n, delay_depth, &dgrid);
        if (status != AIAEC_OK) return status;
        if (aiaec_pool_layout(family, n, &dgrid, (const void *)0, (unsigned char *)0,
                              (AiaecCore **)0, (void **)0, (void **)0, (size_t *)0, &total) != 0)
            return AIAEC_ERR_POOL;
#else
        return AIAEC_ERR_MODEL_ID;
#endif
    } else {
#if AIAEC_ENABLE_ALIGN_ULCNET
        UlcnetGrid ugrid;
        status = aiaec_build_grid_ulcnet(entry, delay_depth, &ugrid);
        if (status != AIAEC_OK) return status;
        if (aiaec_pool_layout(family, n, (const void *)0, &ugrid, (unsigned char *)0,
                              (AiaecCore **)0, (void **)0, (void **)0, (size_t *)0, &total) != 0)
            return AIAEC_ERR_POOL;
#else
        return AIAEC_ERR_MODEL_ID;
#endif
    }

    req->alignment = AIAEC_ALIGNMENT;
    req->bytes = (uint64_t)total;
    return AIAEC_OK;
}

/* Round trip on the final PCM, measured with an identity accelerator; the same for all three families
 * (the _spec/_freq caller's own ISTFT supplies the framing hop the PCM family has inside).
 * DeepVQE-S: N * hop -- one hop of STFT/ISTFT framing latency, present even at N == 1, plus (N - 1)
 * hops of model queue. Align-ULCNet: the packet graph returns the N enhanced frames of the N frames it
 * was given, so there is no queue: one hop of framing latency whatever N. */
static int aiaec_trim_samples(int model_id, int n, int hop)
{
    return model_id == AIAEC_MODEL_ID_ALIGN_ULCNET ? hop : n * hop;
}

static int aiaec_queue_frames(int model_id, int n)
{
    return model_id == AIAEC_MODEL_ID_ALIGN_ULCNET ? 0 : n - 1;
}

static void aiaec_core_info(const AiaecCore *core, const char *graph_path, AiaecInfo *info)
{
    const int model_hop = core->cfg.fft / 2;

    memset(info, 0, sizeof(*info));
    info->graph_path = graph_path;
    info->trim_samples = aiaec_trim_samples(core->model_id, core->frames_per_call, model_hop);
    info->queue_frames = aiaec_queue_frames(core->model_id, core->frames_per_call);
    info->tail_zero_blocks = (info->trim_samples + core->cfg.hop_in - 1) / core->cfg.hop_in;
}

/* The pool is checked against aiaec_core_mem_size() after the configuration resolves; on any error
 * info is untouched. */
static int aiaec_core_init(const AiaecConfig *cfg, AiaecFamily family, void *pool, size_t pool_bytes,
                           AiaecCore **core_out, AiaecInfo *info)
{
    StreamMemReq req;
    int status, n, delay_depth;
    const AiaecModelEntry *entry;
    const char *graph_path = (const char *)0;
    AiaecCore *core = (AiaecCore *)0;
    void *stash = (void *)0;
    void *inner_pool = (void *)0;
    size_t inner_bytes = 0;
    size_t total = 0;

    if (!cfg || !pool || !info) return AIAEC_ERR_NULL;

    status = aiaec_resolve(cfg, &entry, &n, &graph_path, &delay_depth);
    if (status != AIAEC_OK) return status;
    status = aiaec_core_mem_size(cfg, family, &req);
    if (status != AIAEC_OK) return status;
    if (pool_bytes < (size_t)req.bytes) return AIAEC_ERR_POOL;
    if (((uintptr_t)pool % (uintptr_t)req.alignment) != 0u) return AIAEC_ERR_POOL;

    memset(pool, 0, (size_t)req.bytes);

    if (AIAEC_BOUND_MODEL_ID == AIAEC_MODEL_ID_DEEPVQE_S) {
#if AIAEC_ENABLE_DEEPVQE_S
        DeepVqeGrid dgrid;
        DeepVqeStreamConfig scfg;

        status = aiaec_build_grid(entry, n, delay_depth, &dgrid);
        if (status != AIAEC_OK) return status;
        if (aiaec_pool_layout(family, n, &dgrid, (const void *)0, (unsigned char *)pool,
                              &core, &stash, &inner_pool, &inner_bytes, &total) != 0)
            return AIAEC_ERR_POOL;
        if (total != (size_t)req.bytes) return AIAEC_ERR_POOL;

        memset(&scfg, 0, sizeof(scfg));
        scfg.grid = dgrid;
        core->u.deepvqe.inner = deepvqe_stream_init(&scfg, inner_pool, inner_bytes);
        if (!core->u.deepvqe.inner) return AIAEC_ERR_POOL;
        core->u.deepvqe.grid = dgrid;
        if (family == AIAEC_FAMILY_PCM) core->u.deepvqe.stash = (int16_t *)stash;
        else core->u.deepvqe.stash_f = (float *)stash;
        core->hop = dgrid.hop;
        core->n_bins = dgrid.n_bins;
#else
        return AIAEC_ERR_MODEL_ID;
#endif
    } else {
#if AIAEC_ENABLE_ALIGN_ULCNET
        UlcnetGrid ugrid;
        UlcnetStreamConfig ucfg;

        status = aiaec_build_grid_ulcnet(entry, delay_depth, &ugrid);
        if (status != AIAEC_OK) return status;
        if (aiaec_pool_layout(family, n, (const void *)0, &ugrid, (unsigned char *)pool,
                              &core, &stash, &inner_pool, &inner_bytes, &total) != 0)
            return AIAEC_ERR_POOL;
        if (total != (size_t)req.bytes) return AIAEC_ERR_POOL;

        memset(&ucfg, 0, sizeof(ucfg));
        ucfg.grid = ugrid;
        ucfg.frames_per_call = n;
        core->u.align.inner = ulcnet_stream_init(&ucfg, inner_pool, inner_bytes);
        if (!core->u.align.inner) return AIAEC_ERR_POOL;
        core->u.align.grid = ugrid;
        core->hop = ugrid.hop;
        core->n_bins = ugrid.n_bins;
#else
        return AIAEC_ERR_MODEL_ID;
#endif
    }

    core->cfg = *cfg;
    core->family = family;
    core->model_id = AIAEC_BOUND_MODEL_ID;
    core->frames_per_call = n;
    core->block_open = 0;
    aiaec_core_info(core, graph_path, info);
    *core_out = core;
    return AIAEC_OK;
}

static void aiaec_core_reset(AiaecCore *core)
{
    if (!core) return;
    if (core->model_id == AIAEC_MODEL_ID_DEEPVQE_S) {
#if AIAEC_ENABLE_DEEPVQE_S
        deepvqe_stream_reset(core->u.deepvqe.inner);
#endif
    } else {
#if AIAEC_ENABLE_ALIGN_ULCNET
        ulcnet_stream_reset(core->u.align.inner);
#endif
    }
    core->block_open = 0;
}

static unsigned int aiaec_core_generation(const AiaecCore *core)
{
    if (!core) return 0u;
    if (core->model_id == AIAEC_MODEL_ID_DEEPVQE_S) {
#if AIAEC_ENABLE_DEEPVQE_S
        return deepvqe_stream_state_generation(core->u.deepvqe.inner);
#else
        return 0u;
#endif
    }
#if AIAEC_ENABLE_ALIGN_ULCNET
    return (unsigned int)ulcnet_stream_state_generation(core->u.align.inner);
#else
    return 0u;
#endif
}

int aiaec_init(const AiaecConfig *cfg, void *pool, size_t pool_bytes, AiaecCtx **ctx, AiaecInfo *info)
{
    AiaecCore *core = (AiaecCore *)0;
    int rc;

    if (!ctx) return AIAEC_ERR_NULL;
    *ctx = (AiaecCtx *)0;
    rc = aiaec_core_init(cfg, AIAEC_FAMILY_PCM, pool, pool_bytes, &core, info);
    if (rc == AIAEC_OK) *ctx = (AiaecCtx *)core;
    return rc;
}

int aiaec_init_spec(const AiaecConfig *cfg, void *pool, size_t pool_bytes, AiaecSpecCtx **ctx, AiaecInfo *info)
{
    AiaecCore *core = (AiaecCore *)0;
    int rc;

    if (!ctx) return AIAEC_ERR_NULL;
    *ctx = (AiaecSpecCtx *)0;
    rc = aiaec_core_init(cfg, AIAEC_FAMILY_SPEC, pool, pool_bytes, &core, info);
    if (rc == AIAEC_OK) *ctx = (AiaecSpecCtx *)core;
    return rc;
}

int aiaec_init_freq(const AiaecConfig *cfg, void *pool, size_t pool_bytes, AiaecFreqCtx **ctx, AiaecInfo *info)
{
    AiaecCore *core = (AiaecCore *)0;
    int rc;

    if (!ctx) return AIAEC_ERR_NULL;
    *ctx = (AiaecFreqCtx *)0;
    rc = aiaec_core_init(cfg, AIAEC_FAMILY_FREQ, pool, pool_bytes, &core, info);
    if (rc == AIAEC_OK) *ctx = (AiaecFreqCtx *)core;
    return rc;
}

int aiaec_get_mem_size(const AiaecConfig *cfg, StreamMemReq *req)
{
    return aiaec_core_mem_size(cfg, AIAEC_FAMILY_PCM, req);
}

int aiaec_get_mem_size_spec(const AiaecConfig *cfg, StreamMemReq *req)
{
    return aiaec_core_mem_size(cfg, AIAEC_FAMILY_SPEC, req);
}

int aiaec_get_mem_size_freq(const AiaecConfig *cfg, StreamMemReq *req)
{
    return aiaec_core_mem_size(cfg, AIAEC_FAMILY_FREQ, req);
}

void aiaec_reset(AiaecCtx *ctx) { if (ctx) aiaec_core_reset(&ctx->core); }
void aiaec_reset_spec(AiaecSpecCtx *ctx)   { if (ctx) aiaec_core_reset(&ctx->core); }
void aiaec_reset_freq(AiaecFreqCtx *ctx)   { if (ctx) aiaec_core_reset(&ctx->core); }

unsigned int aiaec_state_generation(const AiaecCtx *ctx) { return ctx ? aiaec_core_generation(&ctx->core) : 0u; }
unsigned int aiaec_state_generation_spec(const AiaecSpecCtx *ctx)   { return ctx ? aiaec_core_generation(&ctx->core) : 0u; }
unsigned int aiaec_state_generation_freq(const AiaecFreqCtx *ctx)   { return ctx ? aiaec_core_generation(&ctx->core) : 0u; }

/* ---------------------------------------------------------------------------------------------
 * Block entries
 * ------------------------------------------------------------------------------------------- */

#if AIAEC_ENABLE_DEEPVQE_S
/* DeepVQE-S's N-hop stash/drain loop: the first N - 1 hops of a block never carry a real job (each
 * drains what the previous block's job left owed and stashes it); the last one does. The three
 * families differ only in what is analysed (PCM or supplied spectra) and what is drained (PCM, or a
 * spectrum into the float stash). */
static int aiaec_deepvqe_pre(AiaecCore *c, const int16_t *mic_in, const int16_t *far_in,
                             const AiaecFreqInput *fin, DeepVqeHopIO *hio)
{
    DeepVqeStream *inner = c->u.deepvqe.inner;
    const int n = c->frames_per_call, hop = c->hop, nb = c->n_bins;
    int i, rc;

    for (i = 0; i <= n - 1; ++i) {
        const size_t fo = (size_t)i * (size_t)nb;
        if (fin)
            rc = deepvqe_stream_pre_process_freq(inner, fin->u.deepvqe_s.mic_re + fo, fin->u.deepvqe_s.mic_im + fo,
                                                 fin->u.deepvqe_s.far_re + fo, fin->u.deepvqe_s.far_im + fo, hio);
        else
            rc = deepvqe_stream_pre_process(inner, mic_in + (size_t)i * (size_t)hop,
                                            far_in + (size_t)i * (size_t)hop, hio);
        if (i == n - 1) return rc == 1 ? AIAEC_OK : AIAEC_ERR_SEQUENCE;
        if (rc != 0) return AIAEC_ERR_SEQUENCE;
        if (c->family == AIAEC_FAMILY_PCM) {
            if (deepvqe_stream_post_process(inner, 0, c->u.deepvqe.stash + (size_t)i * (size_t)hop) < 0)
                return AIAEC_ERR_SEQUENCE;
        } else {
            float *stash_re = c->u.deepvqe.stash_f, *stash_im = stash_re + (size_t)(n - 1) * (size_t)nb;
            if (deepvqe_stream_post_process_freq(inner, 0, stash_re + fo, stash_im + fo) < 0)
                return AIAEC_ERR_SEQUENCE;
        }
    }
    return AIAEC_ERR_SEQUENCE;
}

static int aiaec_deepvqe_post_pcm(AiaecCore *c, int infer_status, int16_t *pcm_out)
{
    const int n = c->frames_per_call, hop = c->hop;
    int i;

    if (deepvqe_stream_post_process(c->u.deepvqe.inner, infer_status,
                                    pcm_out + (size_t)(n - 1) * (size_t)hop) < 0)
        return AIAEC_ERR_SEQUENCE;
    for (i = 0; i < n - 1; ++i) {
        memcpy(pcm_out + (size_t)i * (size_t)hop, c->u.deepvqe.stash + (size_t)i * (size_t)hop,
               (size_t)hop * sizeof(int16_t));
    }
    return AIAEC_OK;
}

static int aiaec_deepvqe_post_spec(AiaecCore *c, int infer_status, float *out_re, float *out_im)
{
    const int n = c->frames_per_call, nb = c->n_bins;
    const size_t last = (size_t)(n - 1) * (size_t)nb;
    int i;

    if (deepvqe_stream_post_process_freq(c->u.deepvqe.inner, infer_status, out_re + last, out_im + last) < 0)
        return AIAEC_ERR_SEQUENCE;
    for (i = 0; i < n - 1; ++i) {
        const size_t fo = (size_t)i * (size_t)nb;
        memcpy(out_re + fo, c->u.deepvqe.stash_f + fo, (size_t)nb * sizeof(float));
        memcpy(out_im + fo, c->u.deepvqe.stash_f + last + fo, (size_t)nb * sizeof(float));
    }
    return AIAEC_OK;
}
#endif /* AIAEC_ENABLE_DEEPVQE_S */

/* Shared by all three families. `pin` (PCM in) or `fin` (spectra in) is non-NULL, the other NULL;
 * kind, size and NULL gates run in that order before the first call that advances any state. */
static int aiaec_core_preprocess(AiaecCore *c, const AiaecInput *pin, const AiaecFreqInput *fin,
                                 AiaecWork *work)
{
    const AiaecInputKind kind = pin ? pin->kind : fin->kind;
    const int count = pin ? pin->samples : fin->frames;
    const int want = pin ? c->cfg.hop_in : c->frames_per_call;

    if (c->block_open) return AIAEC_ERR_SEQUENCE;

    if (c->model_id == AIAEC_MODEL_ID_DEEPVQE_S) {
#if AIAEC_ENABLE_DEEPVQE_S
        DeepVqeHopIO hio;
        int rc;

        if (kind != AIAEC_INPUT_DEEPVQE_S) return AIAEC_ERR_INPUT_KIND;
        if (count != want) return AIAEC_ERR_INPUT_SIZE;
        if (pin ? (!pin->u.deepvqe_s.mic_in || !pin->u.deepvqe_s.far_in)
                : (!fin->u.deepvqe_s.mic_re || !fin->u.deepvqe_s.mic_im ||
                   !fin->u.deepvqe_s.far_re || !fin->u.deepvqe_s.far_im))
            return AIAEC_ERR_NULL;

        rc = aiaec_deepvqe_pre(c, pin ? pin->u.deepvqe_s.mic_in : (const int16_t *)0,
                               pin ? pin->u.deepvqe_s.far_in : (const int16_t *)0, fin, &hio);
        if (rc != AIAEC_OK) return rc;

        aiaec_block_fill_deepvqe(&c->u.deepvqe.grid, &hio.inputs, &hio.outputs, hio.state_generation, work);
        c->block_open = 1;
        return AIAEC_OK;
#else
        return AIAEC_ERR_MODEL_ID;
#endif
    }

#if AIAEC_ENABLE_ALIGN_ULCNET
    {
        UlcnetStreamDelay delay;
        UlcnetHopIO hio;
        const AiaecAlignEvent *events = pin ? pin->u.align_ulcnet.events : fin->u.align_ulcnet.events;
        const int event_count = pin ? pin->u.align_ulcnet.event_count : fin->u.align_ulcnet.event_count;
        int rc;

        if (kind != AIAEC_INPUT_ALIGN_ULCNET) return AIAEC_ERR_INPUT_KIND;
        if (count != want) return AIAEC_ERR_INPUT_SIZE;
        if (event_count != c->frames_per_call) return AIAEC_ERR_INPUT_SIZE;
        if (!events) return AIAEC_ERR_NULL;
        if (pin ? (!pin->u.align_ulcnet.residual_in || !pin->u.align_ulcnet.far_in)
                : (!fin->u.align_ulcnet.err_re || !fin->u.align_ulcnet.err_im ||
                   !fin->u.align_ulcnet.far_re || !fin->u.align_ulcnet.far_im))
            return AIAEC_ERR_NULL;

        if (c->frames_per_call > 1) {
            /* A packet: one event per model hop, one job for all of them. The graph resets and bypasses an
             * event hop itself (no generation change); a PCM entry restarts its own framing at those hops. */
            UlcnetStreamDelay delays[ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL];
            int k;

            for (k = 0; k < event_count; ++k) {
                delays[k].changed = events[k].changed;
                delays[k].solid = events[k].solid;
            }
            if (pin)
                rc = ulcnet_stream_pre_process_packet_f32(c->u.align.inner, pin->u.align_ulcnet.residual_in,
                                                          pin->u.align_ulcnet.far_in, delays, &hio);
            else
                rc = ulcnet_stream_pre_process_packet_freq(c->u.align.inner, fin->u.align_ulcnet.err_re,
                                                           fin->u.align_ulcnet.err_im, fin->u.align_ulcnet.far_re,
                                                           fin->u.align_ulcnet.far_im, delays, &hio);
            if (rc != 1) return AIAEC_ERR_SEQUENCE;
            aiaec_block_fill_align(&c->u.align.grid, c->frames_per_call, &hio.inputs, &hio.outputs,
                                   hio.state_generation, work);
            aiaec_set_restart(work, pin ? ulcnet_stream_restart_mask(c->u.align.inner) : 0u);
            c->block_open = 1;
            return AIAEC_OK;
        }

        /* N == 1: exactly one event pairs with the one model hop in this block. */
        delay.changed = events[0].changed;
        delay.solid   = events[0].solid;

        if (pin)
            rc = ulcnet_stream_pre_process_f32(c->u.align.inner, pin->u.align_ulcnet.residual_in,
                                               pin->u.align_ulcnet.far_in, delay, &hio);
        else
            rc = ulcnet_stream_pre_process_freq(c->u.align.inner, fin->u.align_ulcnet.err_re,
                                                fin->u.align_ulcnet.err_im, fin->u.align_ulcnet.far_re,
                                                fin->u.align_ulcnet.far_im, delay, &hio);
        if (rc < 0) return AIAEC_ERR_SEQUENCE;

        if (rc == 0) {
            /* Realignment/event hop: no accelerator job. The zeroed work is also action BYPASS. */
            memset(work, 0, sizeof(*work));
        } else {
            aiaec_block_fill_align(&c->u.align.grid, 1, &hio.inputs, &hio.outputs, hio.state_generation, work);
        }
        aiaec_set_restart(work, pin ? ulcnet_stream_restart_mask(c->u.align.inner) : 0u);
        c->block_open = 1;
        return AIAEC_OK;
    }
#else
    (void)kind; (void)count; (void)want; (void)work;
    return AIAEC_ERR_MODEL_ID;
#endif
}

int aiaec_precheck(const AiaecCtx *ctx, AiaecInputKind kind, int samples, int event_count)
{
    const AiaecCore *c;

    if (!ctx) return AIAEC_ERR_NULL;
    c = &ctx->core;
    if (c->block_open) return AIAEC_ERR_SEQUENCE;
    if (kind != (c->model_id == AIAEC_MODEL_ID_DEEPVQE_S ? AIAEC_INPUT_DEEPVQE_S
                                                          : AIAEC_INPUT_ALIGN_ULCNET))
        return AIAEC_ERR_INPUT_KIND;
    if (samples != c->cfg.hop_in) return AIAEC_ERR_INPUT_SIZE;
    if (kind == AIAEC_INPUT_ALIGN_ULCNET && event_count != c->frames_per_call) return AIAEC_ERR_INPUT_SIZE;
    return AIAEC_OK;
}

int aiaec_preprocess(AiaecCtx *ctx, const AiaecInput *input, AiaecWork *work)
{
    if (!ctx || !input || !work) return AIAEC_ERR_NULL;
    return aiaec_core_preprocess(&ctx->core, input, (const AiaecFreqInput *)0, work);
}

int aiaec_preprocess_spec(AiaecSpecCtx *ctx, const AiaecInput *input, AiaecWork *work)
{
    if (!ctx || !input || !work) return AIAEC_ERR_NULL;
    return aiaec_core_preprocess(&ctx->core, input, (const AiaecFreqInput *)0, work);
}

int aiaec_preprocess_freq(AiaecFreqCtx *ctx, const AiaecFreqInput *input, AiaecWork *work)
{
    if (!ctx || !input || !work) return AIAEC_ERR_NULL;
    return aiaec_core_preprocess(&ctx->core, (const AiaecInput *)0, input, work);
}

/* pcm_out (PCM family) or out_re/out_im (spectrum families); the unused outputs are NULL. */
static int aiaec_core_postprocess(AiaecCore *c, int infer_status, int16_t *pcm_out,
                                  float *out_re, float *out_im)
{
    int rc;

    if (!c->block_open) return AIAEC_ERR_SEQUENCE;
    c->block_open = 0;

    if (c->model_id == AIAEC_MODEL_ID_DEEPVQE_S) {
#if AIAEC_ENABLE_DEEPVQE_S
        return pcm_out ? aiaec_deepvqe_post_pcm(c, infer_status, pcm_out)
                       : aiaec_deepvqe_post_spec(c, infer_status, out_re, out_im);
#else
        return AIAEC_ERR_MODEL_ID;
#endif
    }

    /* Align-ULCNet: the stream itself ignores infer_status when the block was a bypass; it returns 1
     * on success. */
#if AIAEC_ENABLE_ALIGN_ULCNET
    rc = pcm_out ? ulcnet_stream_post_process(c->u.align.inner, infer_status, pcm_out)
                 : ulcnet_stream_post_process_freq(c->u.align.inner, infer_status, out_re, out_im);
    return rc == 1 ? AIAEC_OK : AIAEC_ERR_SEQUENCE;
#else
    (void)rc; (void)out_re; (void)out_im;
    return AIAEC_ERR_MODEL_ID;
#endif
}

int aiaec_postprocess(AiaecCtx *ctx, int infer_status, int16_t *pcm_out)
{
    if (!ctx || !pcm_out) return AIAEC_ERR_NULL;
    return aiaec_core_postprocess(&ctx->core, infer_status, pcm_out, (float *)0, (float *)0);
}

int aiaec_postprocess_spec(AiaecSpecCtx *ctx, int infer_status, float *out_re, float *out_im)
{
    if (!ctx || !out_re || !out_im) return AIAEC_ERR_NULL;
    return aiaec_core_postprocess(&ctx->core, infer_status, (int16_t *)0, out_re, out_im);
}

int aiaec_postprocess_freq(AiaecFreqCtx *ctx, int infer_status, float *out_re, float *out_im)
{
    if (!ctx || !out_re || !out_im) return AIAEC_ERR_NULL;
    return aiaec_core_postprocess(&ctx->core, infer_status, (int16_t *)0, out_re, out_im);
}

const char *aiaec_status_text(int code)
{
    switch (code) {
    case AIAEC_OK: return "ok";
    case AIAEC_ERR_NULL: return "null argument";
    case AIAEC_ERR_MODEL_ID: return "the product's bound model is unknown or not in this build";
    case AIAEC_ERR_GRID: return "no catalog entry for (sr, fft)";
    case AIAEC_ERR_IN_HOP: return "hop_in is not a positive multiple of model_hop";
    case AIAEC_ERR_FRAMES_PER_CALL: return "no shipped graph for the resolved frames_per_call";
    case AIAEC_ERR_POOL: return "pool too small or misaligned";
    case AIAEC_ERR_SEQUENCE: return "preprocess/postprocess called out of sequence";
    case AIAEC_ERR_INPUT_KIND: return "block input kind does not match the product's model";
    case AIAEC_ERR_INPUT_SIZE: return "block input sample/event count does not match this ctx";
    default: return "unknown aiaec status";
    }
}
