/* ainr.h implementation: a block-level (hop_in samples per call) wrapper over dfn2_stream's
 * hop entries (N == 1) and packet entries (N > 1). The catalog below is the only place that knows the DFN2 grid
 * fields and which (sample_rate, n_fft, frames_per_call) combinations have a shipped graph. */

#include "ainr.h"
#include "ainr_test.h"
#include "ainr_product.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "dfn2_grid.h"
#include "dfn2_stream.h"
#include "stream_mem.h"

/* Every model_id this archive variant (full or mask-only) can be bound to, including the
 * quantization alternates folded into model_id (ainr_product.h). AINR_PRODUCT_MODEL_ID must be
 * one of them; a product bound to the other variant's ID gets AINR_ERR_MODEL_ID from every
 * resolve. */
#ifdef DFN2_MASK_ONLY
static const int AINR_BUILD_VARIANT_MODEL_IDS[] = {
    AINR_MODEL_ID_DFN2_LITE, AINR_MODEL_ID_DFN2_LITE_ALT1, AINR_MODEL_ID_DFN2_LITE_ALT2
};
#else
static const int AINR_BUILD_VARIANT_MODEL_IDS[] = {
    AINR_MODEL_ID_DFN2_FULL, AINR_MODEL_ID_DFN2_FULL_ALT1, AINR_MODEL_ID_DFN2_FULL_ALT2, AINR_MODEL_ID_DFN2_FULL_L0
};
#endif
#define AINR_BUILD_VARIANT_MODEL_IDS_ROWS \
    ((int)(sizeof(AINR_BUILD_VARIANT_MODEL_IDS) / sizeof(AINR_BUILD_VARIANT_MODEL_IDS[0])))

static int ainr_model_id_in_build(int model_id)
{
    int i;
    for (i = 0; i < AINR_BUILD_VARIANT_MODEL_IDS_ROWS; ++i)
        if (AINR_BUILD_VARIANT_MODEL_IDS[i] == model_id) return 1;
    return 0;
}

/* The one model this product ships (ainr_product.h). Never a caller input. */
#define AINR_BOUND_MODEL_ID ((int)(AINR_PRODUCT_MODEL_ID))

/* Board convention: ERB matrix file naming is independent of quantization, fixed by
 * (sample_rate, n_erb) alone, both already in every catalog row below. */
#ifndef AINR_ERB_MATRIX_DIR
#define AINR_ERB_MATRIX_DIR "/mnt/sd/dfn2_blobs"
#endif
static void ainr_erb_path(int sample_rate, int n_erb, char *out, size_t out_size)
{
    snprintf(out, out_size, "%s/fb_%d_%d.bin", AINR_ERB_MATRIX_DIR, sample_rate, n_erb);
}

/* The recurrent state is never host-registered: the accelerator owns it, so StreamWork lists only the
 * per-hop tensors (erb and spec windows in; mask, coefs, alpha out). */

/* graph_path NULL means this frames_per_call has no shipped graph yet -- the resolver must treat
 * it exactly like "not in this list", never substitute another N's graph: N fixes the model's
 * input window length, so a wrong-N graph reads the wrong tensor shape, not just runs slower. */
typedef struct AinrDfn2FramesVariant {
    int frames_per_call;
    const char *graph_path;
} AinrDfn2FramesVariant;

typedef struct AinrDfn2ModelEntry {
    int model_id;
    int sample_rate, n_fft;
    int n_erb, df_bins, df_order;       /* dfn2_grid_make() inputs, fixed by the checkpoint */
    int mask_lookahead, df_lookahead;   /* shared by every frames_per_call variant below */
    float atten_lim_db;
    int hpf_enable;
    float hpf_cutoff_hz;
    const AinrDfn2FramesVariant *variants;
    int n_variants;
} AinrDfn2ModelEntry;

/* Board convention: quant alternates (ainr.h) each load a different model.bin subdirectory;
 * grid/N still need their own filename -- 16k/N=1, 16k/N=8 and 48k/N=1 are three different
 * exported shapes. Filenames below are placeholders until the release naming is fixed. */
#define AINR_MODEL_DIR_BASE "/mnt/sd/audio_model"

#ifndef DFN2_MASK_ONLY
/* AINR_MODEL_ID_DFN2_FULL -> 16x16 (default); _ALT1 -> 8x8; _ALT2 -> 16x8. */
static const AinrDfn2FramesVariant AINR_DFN2_FULL_16K_256_N_16X16[] = {
    { 1, AINR_MODEL_DIR_BASE "/16x16/dfn2/nvt_model_16k_n1.bin" },
    { 8, AINR_MODEL_DIR_BASE "/16x16/dfn2/nvt_model_16k_n8.bin" },
};
static const AinrDfn2FramesVariant AINR_DFN2_FULL_48K_1024_N_16X16[] = {
    { 1, AINR_MODEL_DIR_BASE "/16x16/dfn2/nvt_model_48k_n1.bin" },   /* shipped grid */
};
static const AinrDfn2FramesVariant AINR_DFN2_FULL_16K_256_N_16X8[] = {
    { 1, AINR_MODEL_DIR_BASE "/16x8/dfn2/nvt_model_16k_n1.bin" },
    { 8, AINR_MODEL_DIR_BASE "/16x8/dfn2/nvt_model_16k_n8.bin" },
};
static const AinrDfn2FramesVariant AINR_DFN2_FULL_48K_1024_N_16X8[] = {
    { 1, AINR_MODEL_DIR_BASE "/16x8/dfn2/nvt_model_48k_n1.bin" },
};

/* AINR_MODEL_ID_DFN2_FULL_L0: mask_lookahead 0, df_lookahead 0 at every grid; its 16 kHz rows are the graphs above. The 32 and
 * 48 kHz paths are NULL (refused as an unlisted N) until set; AINR_TEST_CATALOG gives them test:// paths. The 32 kHz ERB file
 * is fb_32000_28.bin. */
#ifdef AINR_TEST_CATALOG
static const AinrDfn2FramesVariant AINR_DFN2_FULL_L0_32K_512_N[] = {
    { 1, "test://ainr/dfn2_l0/32k_fft512_n1" },
    { 4, "test://ainr/dfn2_l0/32k_fft512_n4" },
};
static const AinrDfn2FramesVariant AINR_DFN2_FULL_L0_48K_1024_N[] = {
    { 1, "test://ainr/dfn2_l0/48k_fft1024_n1" },
    { 2, "test://ainr/dfn2_l0/48k_fft1024_n2" },
};
#else
static const AinrDfn2FramesVariant AINR_DFN2_FULL_L0_32K_512_N[] = {
    { 1, (const char *)0 },
    { 4, (const char *)0 },
};
static const AinrDfn2FramesVariant AINR_DFN2_FULL_L0_48K_1024_N[] = {
    { 1, (const char *)0 },
    { 2, (const char *)0 },
};
#endif

static const AinrDfn2ModelEntry AINR_DFN2_CATALOG[] = {
    { AINR_MODEL_ID_DFN2_FULL,      16000, 256,  24, 96, 5, 0, 0, 0.0f, 0, 0.0f,
      AINR_DFN2_FULL_16K_256_N_16X16, 2 },
    { AINR_MODEL_ID_DFN2_FULL,      48000, 1024, 32, 96, 5, 1, 1, 0.0f, 0, 0.0f,
      AINR_DFN2_FULL_48K_1024_N_16X16, 1 },
    { AINR_MODEL_ID_DFN2_FULL_ALT2, 16000, 256,  24, 96, 5, 0, 0, 0.0f, 0, 0.0f,
      AINR_DFN2_FULL_16K_256_N_16X8, 2 },
    { AINR_MODEL_ID_DFN2_FULL_ALT2, 48000, 1024, 32, 96, 5, 1, 1, 0.0f, 0, 0.0f,
      AINR_DFN2_FULL_48K_1024_N_16X8, 1 },
    { AINR_MODEL_ID_DFN2_FULL_L0,   16000, 256,  24, 96, 5, 0, 0, 0.0f, 0, 0.0f,
      AINR_DFN2_FULL_16K_256_N_16X16, 2 },
    { AINR_MODEL_ID_DFN2_FULL_L0,   32000, 512,  28, 96, 5, 0, 0, 0.0f, 0, 0.0f,
      AINR_DFN2_FULL_L0_32K_512_N, 2 },
    { AINR_MODEL_ID_DFN2_FULL_L0,   48000, 1024, 32, 96, 5, 0, 0, 0.0f, 0, 0.0f,
      AINR_DFN2_FULL_L0_48K_1024_N, 2 },
};
#define AINR_DFN2_CATALOG_ROWS ((int)(sizeof(AINR_DFN2_CATALOG) / sizeof(AINR_DFN2_CATALOG[0])))
#else
/* Lite (mask-only, no deep filter): df_order/df_lookahead are 0. Only the 48 kHz/1024 row exists
 * (frames_per_call = 1; mask_lookahead = 1 forces N = 1, same rule as full). Board assets
 * (model.bin and ERB matrix present and matching) are not verified yet.
 * AINR_MODEL_ID_DFN2_LITE -> 16x16 (default); _ALT1 -> 8x8; _ALT2 -> 16x8. */
static const AinrDfn2FramesVariant AINR_DFN2_LITE_48K_1024_N_16X16[] = {
    { 1, AINR_MODEL_DIR_BASE "/16x16/dfn2_lite/nvt_model_48k_n1.bin" },
};
static const AinrDfn2FramesVariant AINR_DFN2_LITE_48K_1024_N_16X8[] = {
    { 1, AINR_MODEL_DIR_BASE "/16x8/dfn2_lite/nvt_model_48k_n1.bin" },
};

static const AinrDfn2ModelEntry AINR_DFN2_CATALOG[] = {
    { AINR_MODEL_ID_DFN2_LITE,      48000, 1024, 32, 96, 0, 1, 0, 0.0f, 0, 0.0f,
      AINR_DFN2_LITE_48K_1024_N_16X16, 1 },
    { AINR_MODEL_ID_DFN2_LITE_ALT2, 48000, 1024, 32, 96, 0, 1, 0, 0.0f, 0, 0.0f,
      AINR_DFN2_LITE_48K_1024_N_16X8, 1 },
};
#define AINR_DFN2_CATALOG_ROWS ((int)(sizeof(AINR_DFN2_CATALOG) / sizeof(AINR_DFN2_CATALOG[0])))
#endif

/* One context for both families (PCM, _freq): the public Ainr*Ctx types are one-member wrappers around
 * it. scratch (_freq only) holds the block's MODEL-scale est/app input, 4 * n * n_bins floats. The _freq
 * family's inner stream is freq_only. */
typedef struct AinrCore {
    DFN2Stream  *inner;
    DFN2Grid     grid;
    int          n;            /* frames_per_call */
    float       *scratch;
    int          block_open;
} AinrCore;

struct AinrCtx { AinrCore core; };
struct AinrFreqCtx { AinrCore core; };

typedef enum AinrFamily { AINR_FAMILY_PCM = 0, AINR_FAMILY_FREQ } AinrFamily;

/* Resolves cfg to a catalog entry, frames_per_call, and that variant's graph path for the
 * product's bound model. Order: the bound model belongs to this archive variant (see
 * ainr_model_id_in_build()); (model_id, sr, fft) row match; hop_in divisibility; frames_per_call > 1
 * requires mask_lookahead = 0; then the shipped-variants list, where "shipped" also requires a non-NULL graph_path (a
 * listed N with no graph is treated as unlisted, never resolved to another N's graph).
 * graph_path_out may be NULL. */
static int ainr_resolve(const AinrConfig *cfg, const AinrDfn2ModelEntry **entry_out, int *n_out,
                        const char **graph_path_out)
{
    int i, j, model_hop, n;
    const AinrDfn2ModelEntry *entry = (const AinrDfn2ModelEntry *)0;
    const char *graph_path = (const char *)0;

    if (!cfg || !entry_out || !n_out) return AINR_ERR_NULL;
    if (!ainr_model_id_in_build(AINR_BOUND_MODEL_ID)) return AINR_ERR_MODEL_ID;

    for (i = 0; i < AINR_DFN2_CATALOG_ROWS; ++i) {
        if (AINR_DFN2_CATALOG[i].model_id == AINR_BOUND_MODEL_ID &&
            AINR_DFN2_CATALOG[i].sample_rate == cfg->sr && AINR_DFN2_CATALOG[i].n_fft == cfg->fft) {
            entry = &AINR_DFN2_CATALOG[i];
            break;
        }
    }
    if (!entry) return AINR_ERR_GRID;

    model_hop = cfg->fft / 2;
    if (model_hop <= 0 || cfg->hop_in <= 0 || cfg->hop_in % model_hop != 0) return AINR_ERR_IN_HOP;
    n = cfg->hop_in / model_hop;

    if (entry->mask_lookahead > 0 && n > 1) return AINR_ERR_LOOKAHEAD_BLOCK_UNSUPPORTED;

    for (j = 0; j < entry->n_variants; ++j) {
        if (entry->variants[j].frames_per_call == n) { graph_path = entry->variants[j].graph_path; break; }
    }
    if (j == entry->n_variants || !graph_path) return AINR_ERR_FRAMES_PER_CALL;

    *entry_out = entry;
    *n_out = n;
    if (graph_path_out) *graph_path_out = graph_path;
    return AINR_OK;
}

static int ainr_build_grid(const AinrDfn2ModelEntry *entry, int n, DFN2Grid *grid)
{
    DFN2GridSpec spec;

    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = entry->sample_rate;
    spec.n_fft = entry->n_fft;
    spec.n_erb = entry->n_erb;
    spec.df_bins = entry->df_bins;
    spec.df_order = entry->df_order;
    spec.mask_lookahead = entry->mask_lookahead;
    spec.df_lookahead = entry->df_lookahead;
    spec.frames_per_call = n;
    return dfn2_grid_make(grid, &spec) == DFN2_GRID_OK ? AINR_OK : AINR_ERR_GRID;
}

/* The inner stream's config (file mode: no ERB pointers), the same for sizing and init. _freq: a freq_only stream,
 * and no HPF (it belongs before the caller's analysis). */
static void ainr_stream_config(AinrFamily family, const AinrDfn2ModelEntry *entry, const DFN2Grid *grid,
                               DFN2StreamConfig *scfg)
{
    memset(scfg, 0, sizeof(*scfg));
    scfg->grid = *grid;
    scfg->atten_lim_db = entry->atten_lim_db;
    if (family == AINR_FAMILY_FREQ) {
        scfg->freq_only = 1;
    } else {
        scfg->hpf_enable = entry->hpf_enable;
        scfg->hpf_cutoff_hz = entry->hpf_cutoff_hz;
    }
}

typedef struct AinrLayout {
    AinrCore         *core;
    float            *scratch;
    void             *inner_pool;
    DFN2StreamMemReq  sreq;
    size_t            total;
} AinrLayout;

/* Carves (base != NULL) or just sizes (base == NULL) the core header, the _freq scratch and the inner
 * DFN2Stream's own pool region, in that order, using stream_mem.h's overflow-checked stream_carve(). */
static int ainr_pool_layout(AinrFamily family, const AinrDfn2ModelEntry *entry, int n,
                            const DFN2Grid *grid, unsigned char *base, AinrLayout *out)
{
    size_t cursor = 0;
    void *p;
    DFN2StreamConfig scfg;

    memset(out, 0, sizeof(*out));
    if (stream_carve(base, &cursor, sizeof(AinrCore), &p) != 0) return -1;
    out->core = (AinrCore *)p;

    if (family == AINR_FAMILY_FREQ) {
        if (stream_carve(base, &cursor, (size_t)n * (size_t)grid->n_bins * 4u * sizeof(float), &p) != 0) return -1;
        out->scratch = (float *)p;
    }

    ainr_stream_config(family, entry, grid, &scfg);
    if (dfn2_stream_get_mem_size(&scfg, &out->sreq) != 0) return -1;

    if (stream_carve(base, &cursor, (size_t)out->sreq.bytes, &p) != 0) return -1;
    out->inner_pool = p;

    out->total = cursor;
    return 0;
}

/* The pool size of a resolved model entry and grid. */
static int ainr_mem_size_resolved(AinrFamily family, const AinrDfn2ModelEntry *entry, int n, const DFN2Grid *grid,
                                  StreamMemReq *req)
{
    AinrLayout lay;

    if (ainr_pool_layout(family, entry, n, grid, (unsigned char *)0, &lay) != 0) return AINR_ERR_POOL;
    req->alignment = AINR_ALIGNMENT;
    req->bytes = (uint64_t)lay.total;
    return AINR_OK;
}

static int ainr_get_mem_size_family(AinrFamily family, const AinrConfig *cfg, StreamMemReq *req)
{
    const AinrDfn2ModelEntry *entry;
    int n, status;
    DFN2Grid grid;

    if (!req) return AINR_ERR_NULL;
    status = ainr_resolve(cfg, &entry, &n, (const char **)0);
    if (status != AINR_OK) return status;
    status = ainr_build_grid(entry, n, &grid);
    if (status != AINR_OK) return status;
    return ainr_mem_size_resolved(family, entry, n, &grid, req);
}

int ainr_get_mem_size(const AinrConfig *cfg, StreamMemReq *req)
{
    return ainr_get_mem_size_family(AINR_FAMILY_PCM, cfg, req);
}

int ainr_get_mem_size_freq(const AinrConfig *cfg, StreamMemReq *req)
{
    return ainr_get_mem_size_family(AINR_FAMILY_FREQ, cfg, req);
}

static void ainr_fill_info(const DFN2Grid *grid, const char *graph_path, AinrInfo *info);

/* Shared by every init: erb_path is the explicit file (test entry points) or NULL for the
 * release-asset path of the resolved grid (customer entry points). Writes *core and *info only on
 * success; a refused pool is not written. */
static int ainr_init_common(AinrFamily family, const AinrConfig *cfg, const char *erb_path,
                            void *pool, size_t pool_bytes, AinrCore **core_out, AinrInfo *info)
{
    StreamMemReq req;
    int status;
    const AinrDfn2ModelEntry *entry;
    const char *graph_path = (const char *)0;
    int n;
    DFN2Grid grid;
    DFN2StreamConfig scfg;
    AinrLayout lay;
    char erb_buf[256];

    *core_out = (AinrCore *)0;
    if (!cfg || !info) return AINR_ERR_NULL;
    status = ainr_resolve(cfg, &entry, &n, &graph_path);
    if (status != AINR_OK) return status;
    if (!erb_path) {   /* customer entry: the release-asset path of the resolved grid */
        ainr_erb_path(entry->sample_rate, entry->n_erb, erb_buf, sizeof(erb_buf));
        erb_path = erb_buf;
    }
    if (!pool) return AINR_ERR_NULL;

    status = ainr_build_grid(entry, n, &grid);
    if (status != AINR_OK) return status;
    status = ainr_mem_size_resolved(family, entry, n, &grid, &req);
    if (status != AINR_OK) return status;
    if (pool_bytes < (size_t)req.bytes) return AINR_ERR_POOL;
    if (((uintptr_t)pool % (uintptr_t)req.alignment) != 0u) return AINR_ERR_POOL;

    memset(pool, 0, (size_t)req.bytes);
    if (ainr_pool_layout(family, entry, n, &grid, (unsigned char *)pool, &lay) != 0 ||
        lay.total != (size_t)req.bytes)   /* sizing/carve agreement */
        return AINR_ERR_POOL;

    ainr_stream_config(family, entry, &grid, &scfg);

    lay.core->inner = dfn2_stream_init_from_erb_file(&scfg, erb_path, lay.inner_pool,
                                                    (size_t)lay.sreq.bytes);
    if (!lay.core->inner) return AINR_ERR_ERB_FILE;

    lay.core->grid = grid;
    lay.core->n = n;
    lay.core->scratch = lay.scratch;
    lay.core->block_open = 0;
    ainr_fill_info(&grid, graph_path, info);
    *core_out = lay.core;
    return AINR_OK;
}

/* The test entry points take the ERB path from the caller; NULL is an argument error. */
static int ainr_init_file(AinrFamily family, const AinrConfig *cfg, const char *erb_path,
                          void *pool, size_t pool_bytes, AinrCore **core_out, AinrInfo *info)
{
    if (!erb_path) {
        *core_out = (AinrCore *)0;
        return AINR_ERR_NULL;
    }
    return ainr_init_common(family, cfg, erb_path, pool, pool_bytes, core_out, info);
}

/* The public contexts are one-member wrappers around AinrCore, so a core pointer converts to the
 * context pointer. *ctx is NULL on every refusal. */
int ainr_init(const AinrConfig *cfg, void *pool, size_t pool_bytes, AinrCtx **ctx, AinrInfo *info)
{
    AinrCore *core = (AinrCore *)0;
    int rc;

    if (!ctx) return AINR_ERR_NULL;
    rc = ainr_init_common(AINR_FAMILY_PCM, cfg, (const char *)0, pool, pool_bytes, &core, info);
    *ctx = (AinrCtx *)core;
    return rc;
}

int ainr_init_freq(const AinrConfig *cfg, void *pool, size_t pool_bytes, AinrFreqCtx **ctx, AinrInfo *info)
{
    AinrCore *core = (AinrCore *)0;
    int rc;

    if (!ctx) return AINR_ERR_NULL;
    rc = ainr_init_common(AINR_FAMILY_FREQ, cfg, (const char *)0, pool, pool_bytes, &core, info);
    *ctx = (AinrFreqCtx *)core;
    return rc;
}

int ainr_init_from_erb_file(const AinrConfig *cfg, const char *erb_path, void *pool, size_t pool_bytes,
                            AinrCtx **ctx, AinrInfo *info)
{
    AinrCore *core = (AinrCore *)0;
    int rc;

    if (!ctx) return AINR_ERR_NULL;
    rc = ainr_init_file(AINR_FAMILY_PCM, cfg, erb_path, pool, pool_bytes, &core, info);
    *ctx = (AinrCtx *)core;
    return rc;
}

int ainr_init_freq_from_erb_file(const AinrConfig *cfg, const char *erb_path, void *pool, size_t pool_bytes,
                                 AinrFreqCtx **ctx, AinrInfo *info)
{
    AinrCore *core = (AinrCore *)0;
    int rc;

    if (!ctx) return AINR_ERR_NULL;
    rc = ainr_init_file(AINR_FAMILY_FREQ, cfg, erb_path, pool, pool_bytes, &core, info);
    *ctx = (AinrFreqCtx *)core;
    return rc;
}

static void ainr_reset_core(AinrCore *c)
{
    dfn2_stream_reset(c->inner);
    c->block_open = 0;
}

void ainr_reset(AinrCtx *ctx) { if (ctx) ainr_reset_core(&ctx->core); }
void ainr_reset_freq(AinrFreqCtx *ctx) { if (ctx) ainr_reset_core(&ctx->core); }

static int ainr_set_atten_lim_core(AinrCore *c, float atten_lim_db)
{
    if (!isfinite(atten_lim_db)) return AINR_ERR_NONFINITE_INPUT;
    return dfn2_stream_set_atten_lim(c->inner, atten_lim_db) == 0 ? AINR_OK : AINR_ERR_SEQUENCE;
}

int ainr_set_atten_lim(AinrCtx *ctx, float atten_lim_db)
{
    return ctx ? ainr_set_atten_lim_core(&ctx->core, atten_lim_db) : AINR_ERR_NULL;
}

int ainr_set_atten_lim_freq(AinrFreqCtx *ctx, float atten_lim_db)
{
    return ctx ? ainr_set_atten_lim_core(&ctx->core, atten_lim_db) : AINR_ERR_NULL;
}

unsigned int ainr_state_generation(const AinrCtx *ctx)
{
    return ctx ? dfn2_stream_state_generation(ctx->core.inner) : 0u;
}

unsigned int ainr_state_generation_freq(const AinrFreqCtx *ctx)
{
    return ctx ? dfn2_stream_state_generation(ctx->core.inner) : 0u;
}

/* Fills StreamWork from the inner API's one open hop: shapes (ONNX logical [b,c,h,w], w innermost)
 * from the grid, data pointers and element counts from the authoritative DFN2Prepost views. */
static void ainr_fill_work(const DFN2Grid *grid, const DFN2PrepostInputs *in,
                           const DFN2PrepostOutputs *out, unsigned int generation, StreamWork *work)
{
    StreamTensorIn *ti = work->inputs;
    StreamTensorOut *to = work->outputs;

    ti[0].data = in->erb_window;
    ti[0].elements = in->erb_window_elements;
    ti[0].w = (unsigned int)grid->n_erb;
    ti[0].h = (unsigned int)grid->input_frames;
    ti[0].c = 1u;
    ti[0].b = 1u;

    ti[1].data = in->spec_window;
    ti[1].elements = in->spec_window_elements;
    ti[1].w = (unsigned int)grid->df_bins;
    ti[1].h = (unsigned int)grid->input_frames;
    ti[1].c = 2u;
    ti[1].b = 1u;
    work->input_count = 2;

    to[0].data = out->erb_mask;
    to[0].elements = out->erb_mask_elements;
    to[0].w = (unsigned int)grid->n_erb;
    to[0].h = (unsigned int)grid->frames_per_call;
    to[0].c = 1u;
    to[0].b = 1u;
    work->output_count = 1;

#ifndef DFN2_MASK_ONLY
    /* (B, T, df_bins, df_order * 2) of the exported graph, innermost first. */
    to[1].data = out->coefs;
    to[1].elements = out->coefs_elements;
    to[1].w = (unsigned int)grid->df_order * 2u;
    to[1].h = (unsigned int)grid->df_bins;
    to[1].c = (unsigned int)grid->frames_per_call;
    to[1].b = 1u;

    /* (B, T, 1) of the exported graph. */
    to[2].data = out->alpha;
    to[2].elements = out->alpha_elements;
    to[2].w = 1u;
    to[2].h = (unsigned int)grid->frames_per_call;
    to[2].c = 1u;
    to[2].b = 1u;
    work->output_count = 3;
#endif

    work->state_generation = generation;
}

/* The block's work: the hop's action, with the tensors on SUBMIT, else an all-zero work (generation 0). */
static void ainr_prepare_work(const DFN2Grid *grid, const DFN2HopIO *hio, StreamWork *work)
{
    if (hio->action == STREAM_WORK_SUBMIT)
        ainr_fill_work(grid, &hio->inputs, &hio->outputs, hio->state_generation, work);
    else
        memset(work, 0, sizeof(*work));
    work->action = hio->action;
}

/* ---------------------------------------------------------------------------------------------
 * Block entries. N == 1 is the per-hop stream (BYPASS on the first mask_lookahead hops); N > 1 is one packet
 * of N hops and one job. AINR_OK with *hio filled, AINR_ERR_SEQUENCE on a contract error.
 * ------------------------------------------------------------------------------------------- */

static int ainr_open_pcm(AinrCore *c, const int16_t *pcm_in, DFN2HopIO *hio)
{
    int rc = c->n == 1 ? dfn2_stream_pre_process(c->inner, pcm_in, hio)
                       : dfn2_stream_pre_process_packet(c->inner, pcm_in, hio);
    return rc == 1 ? AINR_OK : AINR_ERR_SEQUENCE;
}

/* Spectrum -> spectrum: converts the caller's n audio-scale est/app frames to MODEL scale in `scratch` before the
 * stream's _freq_dual entry. */
static int ainr_open_freq(AinrCore *c, const float *est_re, const float *est_im,
                          const float *app_re, const float *app_im, DFN2HopIO *hio)
{
    const size_t count = (size_t)c->n * (size_t)c->grid.n_bins;
    const float scale = dfn2_stft_scale(c->grid.n_fft);
    float *s_est_re = c->scratch;
    float *s_est_im = c->scratch + count;
    float *s_app_re = c->scratch + 2 * count;
    float *s_app_im = c->scratch + 3 * count;
    size_t k;
    int rc;

    for (k = 0; k < count; ++k) {
        s_est_re[k] = est_re[k] * scale;
        s_est_im[k] = est_im[k] * scale;
        s_app_re[k] = app_re[k] * scale;
        s_app_im[k] = app_im[k] * scale;
    }
    rc = c->n == 1
        ? dfn2_stream_pre_process_freq_dual(c->inner, s_est_re, s_est_im, s_app_re, s_app_im, hio)
        : dfn2_stream_pre_process_packet_freq_dual(c->inner, s_est_re, s_est_im, s_app_re, s_app_im, hio);
    return rc == 1 ? AINR_OK : AINR_ERR_SEQUENCE;
}

/* Close/deliver half of _freq: closes the block and writes its n frames, the SAME dfn2_stft_unscale() multiply
 * applied uniformly to every frame, never twice and never only to some frames. */
static int ainr_close_spectrum(AinrCore *c, int run_status, float *out_re, float *out_im)
{
    const size_t count = (size_t)c->n * (size_t)c->grid.n_bins;
    const float unscale = dfn2_stft_unscale(c->grid.n_fft);
    int valid = 0;
    size_t k;
    int rc = c->n == 1 ? dfn2_stream_post_process_freq(c->inner, run_status, out_re, out_im, &valid)
                       : dfn2_stream_post_process_packet_freq(c->inner, run_status, out_re, out_im, &valid);

    if (rc < 0) return AINR_ERR_SEQUENCE;
    for (k = 0; k < count; ++k) {
        out_re[k] *= unscale;
        out_im[k] *= unscale;
    }
    return AINR_OK;
}

int ainr_preprocess(AinrCtx *ctx, const int16_t *pcm_in, StreamWork *work)
{
    AinrCore *c;
    DFN2HopIO hio;

    if (!ctx || !pcm_in || !work) return AINR_ERR_NULL;
    c = &ctx->core;
    if (c->block_open) return AINR_ERR_SEQUENCE;
    if (ainr_open_pcm(c, pcm_in, &hio) != AINR_OK) return AINR_ERR_SEQUENCE;

    ainr_prepare_work(&c->grid, &hio, work);
    c->block_open = 1;
    return AINR_OK;
}

int ainr_postprocess(AinrCtx *ctx, int run_status, int16_t *pcm_out)
{
    AinrCore *c;
    int rc;

    if (!ctx || !pcm_out) return AINR_ERR_NULL;
    c = &ctx->core;
    if (!c->block_open) return AINR_ERR_SEQUENCE;
    rc = c->n == 1 ? dfn2_stream_post_process(c->inner, run_status, pcm_out)
                   : dfn2_stream_post_process_packet(c->inner, run_status, pcm_out);
    c->block_open = 0;
    return rc < 0 ? AINR_ERR_SEQUENCE : AINR_OK;
}

/* True when every bin of the block's n frames (frame-major, contiguous) passes the stream's guard at the model scale
 * ainr_open_freq() applies, so a block accepted here is never refused midway. */
static int ainr_freq_block_in_range(const AinrCore *c, const float *est_re, const float *est_im,
                                    const float *app_re, const float *app_im)
{
    size_t count = (size_t)c->n * (size_t)c->grid.n_bins;
    float scale = dfn2_stft_scale(c->grid.n_fft);

    return dfn2_stream_bins_in_range(est_re, count, scale) && dfn2_stream_bins_in_range(est_im, count, scale) &&
           (app_re == est_re || dfn2_stream_bins_in_range(app_re, count, scale)) &&
           (app_im == est_im || dfn2_stream_bins_in_range(app_im, count, scale));
}

int ainr_preprocess_freq(AinrFreqCtx *ctx, const float *est_re, const float *est_im,
                         const float *app_re, const float *app_im, StreamWork *work)
{
    AinrCore *c;
    DFN2HopIO hio;

    if (!ctx || !est_re || !est_im || !app_re || !app_im || !work) return AINR_ERR_NULL;
    c = &ctx->core;
    if (c->block_open) return AINR_ERR_SEQUENCE;
    /* The whole block is checked before any frame enters the stream, so a refusal leaves the
     * normaliser state, the queue and the generation untouched and opens no block. */
    if (!ainr_freq_block_in_range(c, est_re, est_im, app_re, app_im)) return AINR_ERR_NONFINITE_INPUT;
    if (ainr_open_freq(c, est_re, est_im, app_re, app_im, &hio) != AINR_OK) return AINR_ERR_SEQUENCE;

    ainr_prepare_work(&c->grid, &hio, work);
    c->block_open = 1;
    return AINR_OK;
}

int ainr_postprocess_freq(AinrFreqCtx *ctx, int run_status, float *out_re, float *out_im)
{
    AinrCore *c;
    int status;

    if (!ctx || !out_re || !out_im) return AINR_ERR_NULL;
    c = &ctx->core;
    if (!c->block_open) return AINR_ERR_SEQUENCE;
    status = ainr_close_spectrum(c, run_status, out_re, out_im);
    c->block_open = 0;
    return status;
}

/* ---------------------------------------------------------------------------------------------
 * Init info: graph, queue length.
 * ------------------------------------------------------------------------------------------- */

static void ainr_fill_info(const DFN2Grid *grid, const char *graph_path, AinrInfo *info)
{
    memset(info, 0, sizeof(*info));
    info->graph_path = graph_path;
    info->queue_frames = dfn2_prepost_latency_frames(grid);
}

const char *ainr_status_text(int code)
{
    switch (code) {
    case AINR_OK: return "ok";
    case AINR_ERR_NULL: return "null argument";
    case AINR_ERR_MODEL_ID: return "the product's bound model does not belong to this archive variant";
    case AINR_ERR_GRID: return "no catalog entry for (sr, fft)";
    case AINR_ERR_IN_HOP: return "hop_in is not a positive multiple of model_hop";
    case AINR_ERR_LOOKAHEAD_BLOCK_UNSUPPORTED:
        return "frames_per_call > 1 requires mask_lookahead = 0";
    case AINR_ERR_FRAMES_PER_CALL: return "no shipped graph for the resolved frames_per_call";
    case AINR_ERR_POOL: return "pool too small or misaligned";
    case AINR_ERR_ERB_FILE: return "ERB file open/header/short-read/dimension mismatch, bytes after the tables, a non-finite forward value, or the inverse is not a partition of unity";
    case AINR_ERR_SEQUENCE: return "preprocess/postprocess called out of sequence";
    case AINR_ERR_NONFINITE_INPUT: return "non-finite input (spectrum bin or attenuation limit); refused, state untouched";
    default: return "unknown ainr status";
    }
}
