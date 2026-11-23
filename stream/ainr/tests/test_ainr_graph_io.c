/* The DFN2 graph rows (stream/ainr/graph_io_dfn2.h) against this build's library and the StreamWork a real preprocess
 * returns: every row's file names, ERB file and grid; accepted rows pass GRAPH_IO_CHECK_WORK, a grid without a row is
 * GRAPH_IO_ERR_NO_ROW, and every ainr_get_mem_size() refusal leaves *ctx NULL. Rows against an export: test_dfn2_port_order.py. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ainr.h"
#include "ainr_test_fixture.h"   /* ErbTables */
#include "ainr_product.h"
#include "dfn2_grids.h"
#include "dfn2_model_io.h"
#include "graph_io.h"
#include "graph_io_dfn2.h"

#ifdef DFN2_MASK_ONLY
#if AINR_PRODUCT_MODEL_ID != AINR_MODEL_ID_DFN2_LITE
#error "the lite archive is tested with its own model"
#endif
#define EXPECTED_ACCEPTED 8    /* 1 grid x N 1..8 */
#define EXPECTED_ROWS 1
#else
#if AINR_PRODUCT_MODEL_ID == AINR_MODEL_ID_DFN2_FULL
#define EXPECTED_ACCEPTED 16   /* 2 grids x N 1..8 */
#define EXPECTED_ROWS 4        /* 16 kHz N 1, 8; 48 kHz mask_lookahead 1 N 1, 2 */
#elif AINR_PRODUCT_MODEL_ID == AINR_MODEL_ID_DFN2_FULL_L0
#define EXPECTED_ACCEPTED 24   /* 3 grids x N 1..8 */
#define EXPECTED_ROWS 6        /* 16 kHz N 1, 8; 32 kHz N 1, 4; 48 kHz mask_lookahead 0 N 1, 2 */
#else
#error "the full archive is tested with model 1 and 3"
#endif
#endif

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); \
                                       fprintf(stderr, "\n"); ++g_fail; } } while (0)

static void compare_work(const char *tag, const GraphIoConfig *mc, const StreamWork *work)
{
    const StreamWiring *w = &mc->wiring;
    int i, rc;

    CHECK(work->action == STREAM_WORK_SUBMIT, "%s action %d", tag, (int)work->action);
    CHECK(work->input_count == w->signal_in_count, "%s input_count %d vs %d", tag, work->input_count, w->signal_in_count);
    CHECK(work->output_count == w->head_out_count, "%s output_count %d vs %d", tag, work->output_count, w->head_out_count);
    for (i = 0; i < work->input_count && i < w->signal_in_count; ++i)
        CHECK(work->inputs[i].b == w->inputs[i].b && work->inputs[i].c == w->inputs[i].c &&
              work->inputs[i].h == w->inputs[i].h && work->inputs[i].w == w->inputs[i].w &&
              work->inputs[i].elements == w->inputs[i].elements, "%s signal %d shape", tag, i);
    for (i = 0; i < work->output_count && i < w->head_out_count; ++i)
        CHECK(work->outputs[i].b == w->outputs[i].b && work->outputs[i].c == w->outputs[i].c &&
              work->outputs[i].h == w->outputs[i].h && work->outputs[i].w == w->outputs[i].w &&
              work->outputs[i].elements == w->outputs[i].elements, "%s head %d shape", tag, i);
    GRAPH_IO_CHECK_WORK(rc, mc, work);
    CHECK(rc == GRAPH_IO_OK, "%s GRAPH_IO_CHECK_WORK rc %d", tag, rc);
    CHECK(graph_io_fits(mc, STREAM_WORK_MAX_INPUTS, STREAM_WORK_MAX_OUTPUTS) == GRAPH_IO_OK, "%s fits", tag);
}

static void fill_heads(StreamWork *w)
{
    int i;
    size_t k;
    for (i = 0; i < w->output_count; ++i)
        for (k = 0; k < w->outputs[i].elements; ++k) w->outputs[i].data[k] = i == 0 ? 0.5f : 0.0f;
}

/* Every refusal of the sizing call comes back from the two init families, with *ctx NULL and *info untouched. */
static void check_refusals(const ErbTables *erb)
{
    static const struct { int sr, fft, hop_in, want; const char *what; } cases[] = {
        { 44100, 256, 128, AINR_ERR_GRID, "unregistered grid" },
        { 48000, 1024, 100, AINR_ERR_IN_HOP, "hop_in not a multiple of model_hop" },
        { 48000, 1024, 65 * 512, AINR_ERR_FRAMES_PER_CALL, "N = 65" },
        { 22000, 333, 65 * 166, AINR_ERR_GRID, "unregistered grid, N = 65" },
#if defined(DFN2_MASK_ONLY) || AINR_PRODUCT_MODEL_ID == AINR_MODEL_ID_DFN2_FULL
        { 32000, 512, 256, AINR_ERR_GRID, "32 kHz is not a grid of this model" },
#endif
    };
    _Alignas(16) static unsigned char pool[1 << 21];
    size_t i;
    AinrInfo info, sentinel;
    AinrConfig cfg = {0};
    StreamMemReq req;
    memset(&sentinel, 0x5a, sizeof(sentinel));
    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        AinrCtx *c0 = (AinrCtx *)pool;
        AinrFreqCtx *c2 = (AinrFreqCtx *)pool;
        cfg.sr = cases[i].sr; cfg.fft = cases[i].fft; cfg.hop_in = cases[i].hop_in;
        CHECK(ainr_get_mem_size(&cfg, &req) == cases[i].want, "refusal (%s): get_mem_size", cases[i].what);
        CHECK(ainr_get_mem_size_freq(&cfg, &req) == cases[i].want, "refusal (%s): get_mem_size_freq", cases[i].what);
        info = sentinel;
        CHECK(ainr_init_t(&cfg, erb, pool, sizeof(pool), &c0, &info) == cases[i].want, "refusal (%s): init",
              cases[i].what);
        CHECK(c0 == NULL && memcmp(&info, &sentinel, sizeof(info)) == 0, "refusal (%s): ctx/info after init", cases[i].what);
        info = sentinel;
        CHECK(ainr_init_freq_t(&cfg, erb, pool, sizeof(pool), &c2, &info) == cases[i].want,
              "refusal (%s): init_freq", cases[i].what);
        CHECK(c2 == NULL && memcmp(&info, &sentinel, sizeof(info)) == 0, "refusal (%s): ctx/info after init_freq",
              cases[i].what);
    }
}

/* One pre call of `family` (0 PCM, 1 _freq) on a fresh init: the blocks without a job (the first mask_lookahead after
 * init at N = 1) must be all-zero BYPASS works that the row refuses; the first block with a job is compared with the row,
 * then the block is closed and the next one compared. */
static void run_family(const AinrConfig *cfg, int family, int n, const GraphIoConfig *mc, int mask_lookahead,
                       const char *tag)
{
    AinrInfo info;
    StreamWork work;
    StreamMemReq req;
    AinrCtx *pcm = NULL;
    AinrFreqCtx *freq = NULL;
    void *pool = NULL;
    int16_t *pcm_in;
    float *zeros, *out_re, *out_im;
    ErbTables erb;
    int rc, bins = cfg->fft / 2 + 1;

    memset(&info, 0, sizeof(info));
    rc = family == 0 ? ainr_get_mem_size(cfg, &req) : ainr_get_mem_size_freq(cfg, &req);
    CHECK(rc == AINR_OK, "%s mem size %d", tag, rc);
    if (rc != AINR_OK) return;
    erb = erb_tables(bins, (int)mc->wiring.outputs[0].w, ERB_BAND0);
    if (posix_memalign(&pool, req.alignment, (size_t)req.bytes) != 0) exit(2);
    rc = family == 0 ? ainr_init_t(cfg, &erb, pool, (size_t)req.bytes, &pcm, &info)
                     : ainr_init_freq_t(cfg, &erb, pool, (size_t)req.bytes, &freq, &info);
    CHECK(rc == AINR_OK, "%s init %d", tag, rc);
    CHECK(info.mask_lookahead == mask_lookahead, "%s mask_lookahead %d, the row's %d", tag, info.mask_lookahead,
          mask_lookahead);
    pcm_in = (int16_t *)calloc((size_t)cfg->hop_in, sizeof(int16_t));
    zeros = (float *)calloc((size_t)n * (size_t)bins, sizeof(float));
    out_re = (float *)calloc((size_t)n * (size_t)bins, sizeof(float));
    out_im = (float *)calloc((size_t)n * (size_t)bins, sizeof(float));
    if (!pcm_in || !zeros || !out_re || !out_im) exit(2);
    memset(&work, 0, sizeof work);
    if (rc == AINR_OK) {
        int bypassed = 0, want_bypassed = n == 1 ? mask_lookahead : 0;
        for (;;) {
            rc = family == 0 ? ainr_preprocess(pcm, pcm_in, &work)
                             : ainr_preprocess_freq(freq, zeros, zeros, zeros, zeros, &work);
            CHECK(rc == AINR_OK, "%s preprocess %d", tag, rc);
            if (rc != AINR_OK || work.action == STREAM_WORK_SUBMIT || bypassed > want_bypassed) break;
            {
                int rc3;
                CHECK(work.input_count == 0 && work.output_count == 0 && work.state_generation == 0u,
                      "%s BYPASS work is not all zero (%d inputs, %d outputs, generation %u)", tag, work.input_count,
                      work.output_count, work.state_generation);
                GRAPH_IO_CHECK_WORK(rc3, mc, &work);
                CHECK(rc3 != GRAPH_IO_OK, "%s the row accepted a BYPASS work", tag);
            }
            rc = family == 0 ? ainr_postprocess(pcm, STREAM_RUN_NOT_STARTED, pcm_in)
                             : ainr_postprocess_freq(freq, STREAM_RUN_NOT_STARTED, out_re, out_im);
            CHECK(rc == AINR_OK, "%s postprocess of a BYPASS block %d", tag, rc);
            ++bypassed;
        }
        CHECK(bypassed == want_bypassed, "%s %d BYPASS blocks after init, want %d", tag, bypassed, want_bypassed);
        if (rc == AINR_OK) {
            GraphIoConfig other;
            int other_model = GRAPH_IO_DFN2_PRODUCT == GRAPH_IO_DFN2_LITE ? GRAPH_IO_DFN2_FULL
                                                                                  : GRAPH_IO_DFN2_LITE;
            compare_work(tag, mc, &work);
            /* the other variant's row of the same grid must refuse this variant's work */
            if (graph_io_dfn2_find(other_model, cfg->sr, cfg->fft, n, mask_lookahead, &other) == GRAPH_IO_OK) {
                int rc2;
                GRAPH_IO_CHECK_WORK(rc2, &other, &work);
                CHECK(rc2 != GRAPH_IO_OK, "%s the other variant's row accepted this work", tag);
            }
            fill_heads(&work);
            rc = family == 0 ? ainr_postprocess(pcm, 0, pcm_in) : ainr_postprocess_freq(freq, 0, out_re, out_im);
            CHECK(rc == AINR_OK, "%s postprocess %d", tag, rc);
            memset(&work, 0, sizeof work);
            rc = family == 0 ? ainr_preprocess(pcm, pcm_in, &work)
                             : ainr_preprocess_freq(freq, zeros, zeros, zeros, zeros, &work);
            CHECK(rc == AINR_OK, "%s second preprocess %d", tag, rc);
            if (rc == AINR_OK) compare_work(tag, mc, &work);
        }
    }
    free(pool); free(pcm_in); free(zeros); free(out_re); free(out_im);
    erb_tables_free(&erb);
}

/* The model ids whose grid table (dfn2_grids.h) has this row's (sr, fft) at the row's mask_lookahead and whose variant is
 * the row's: a bitmask over AINR_MODEL_ID_*. */
static unsigned owners_of(const GraphIoDfn2Row *r)
{
    static const int ids[] = { AINR_MODEL_ID_DFN2_FULL, AINR_MODEL_ID_DFN2_FULL_L0, AINR_MODEL_ID_DFN2_LITE };
    unsigned mask = 0;
    size_t i;
    for (i = 0; i < sizeof(ids) / sizeof(ids[0]); ++i) {
        const DFN2GridSpec *g = dfn2_grids_find(ids[i], r->row.sample_rate, r->row.fft);
        const int lite = ids[i] == AINR_MODEL_ID_DFN2_LITE;
        if (g && g->mask_lookahead == r->mask_lookahead && lite == (r->row.model == GRAPH_IO_DFN2_LITE) &&
            g->n_erb == (int)r->row.signal[0][3])
            mask |= 1u << ids[i];
    }
    return mask;
}

static void check_row_table(void)
{
    int i, j;
    for (i = 0; i < GRAPH_IO_DFN2_ROW_COUNT; ++i) {
        const GraphIoDfn2Row *r = &GRAPH_IO_DFN2_ROWS[i];
        char want[96];
        const char *dir = r->row.model == GRAPH_IO_DFN2_LITE ? "dfn2_lite"
                          : (r->mask_lookahead == 0 && r->row.sample_rate != 16000) ? "dfn2_l0" : "dfn2";
        int n_erb = 0;
        const char *erb_file;
        GraphIoConfig c;

        snprintf(want, sizeof want, "%s/nvt_model_%dk_n%d.bin", dir, r->row.sample_rate / 1000, r->row.n);
        CHECK(r->row.file && strcmp(r->row.file, want) == 0, "row %d: file %s, want %s", i, r->row.file ? r->row.file : "(null)", want);
        CHECK(graph_io_dfn2_row(i, &c) == GRAPH_IO_OK && c.file && strcmp(c.file, r->row.file) == 0, "row %d expands with its file", i);
        snprintf(want, sizeof want, "fb_%d_%d.bin", r->row.sample_rate, (int)r->row.signal[0][3]);
        CHECK(r->erb_file && strcmp(r->erb_file, want) == 0, "row %d: erb_file %s, want %s", i, r->erb_file ? r->erb_file : "(null)", want);
        erb_file = graph_io_dfn2_erb_file(r->row.model, r->row.sample_rate, r->row.fft, &n_erb);
        CHECK(erb_file && strcmp(erb_file, r->erb_file) == 0 && n_erb == (int)r->row.signal[0][3],
              "row %d: graph_io_dfn2_erb_file gives %s / %d", i, erb_file ? erb_file : "(null)", n_erb);
        CHECK(graph_io_dfn2_lookup(r->row.model, r->row.sample_rate, r->row.fft, r->row.n, r->mask_lookahead) == r,
              "row %d: its key does not find itself (duplicate key)", i);
        CHECK(owners_of(r) != 0, "row %d (%s): no model's grid owns it", i, r->row.file);
        for (j = i + 1; j < GRAPH_IO_DFN2_ROW_COUNT; ++j)
            CHECK(strcmp(r->row.file, GRAPH_IO_DFN2_ROWS[j].row.file) != 0, "rows %d and %d share the file %s", i, j, r->row.file);
    }
    CHECK(graph_io_dfn2_row(GRAPH_IO_DFN2_ROW_COUNT, NULL) == GRAPH_IO_ERR_NULL, "row NULL cfg");
    CHECK(graph_io_dfn2_erb_file(GRAPH_IO_DFN2_FULL, 44100, 256, NULL) == NULL, "ERB file of an unregistered grid");
}

int main(void)
{
    static const int srs[] = { 8000, 16000, 32000, 48000 };
    static const int ffts[] = { 128, 256, 512, 1024, 2048 };
    int si, fi, n, i, accepted = 0, own_rows = 0, compared = 0, no_row = 0;

    /* The wiring table and the library name the same graph layouts. */
    CHECK(GRAPH_IO_DFN2_LAYOUT_FULL == DFN2_MODEL_IO_LAYOUT_VERSION_FULL,
          "full layout version: table %d, dfn2_model_io.h %d", (int)GRAPH_IO_DFN2_LAYOUT_FULL,
          (int)DFN2_MODEL_IO_LAYOUT_VERSION_FULL);
    CHECK(GRAPH_IO_DFN2_LAYOUT_LITE == DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY,
          "lite layout version: table %d, dfn2_model_io.h %d", (int)GRAPH_IO_DFN2_LAYOUT_LITE,
          (int)DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY);

    check_row_table();
    for (i = 0; i < GRAPH_IO_DFN2_ROW_COUNT; ++i)
        own_rows += (owners_of(&GRAPH_IO_DFN2_ROWS[i]) >> AINR_PRODUCT_MODEL_ID) & 1u;
    CHECK(own_rows == EXPECTED_ROWS, "the table has %d rows owned by model %d, expected %d", own_rows,
          (int)AINR_PRODUCT_MODEL_ID, EXPECTED_ROWS);

    {
        ErbTables erb = erb_tables(513, 32, ERB_BAND0);
        check_refusals(&erb);
        erb_tables_free(&erb);
    }

    for (si = 0; si < 4; ++si) for (fi = 0; fi < 5; ++fi) for (n = 1; n <= 8; ++n) {
        AinrConfig cfg = {0};
        GraphIoConfig mc;
        StreamMemReq req;
        const DFN2GridSpec *grid;
        int lib_ok, row_ok, row_rc, family;

        cfg.sr = srs[si]; cfg.fft = ffts[fi]; cfg.hop_in = n * (ffts[fi] / 2);
        CHECK(graph_io_frames(cfg.fft, cfg.hop_in) == n, "frames arithmetic");
        grid = dfn2_grids_find(AINR_PRODUCT_MODEL_ID, cfg.sr, cfg.fft);
        lib_ok = ainr_get_mem_size(&cfg, &req) == AINR_OK;
        accepted += lib_ok;
        CHECK(lib_ok == (grid != NULL), "sr %d fft %d N %d: library %s, grid table %s", cfg.sr, cfg.fft, n,
              lib_ok ? "accepts" : "refuses", grid ? "has the grid" : "has no grid");
        if (!lib_ok || !grid) continue;
        row_rc = graph_io_dfn2_find(GRAPH_IO_DFN2_PRODUCT, cfg.sr, cfg.fft, n, grid->mask_lookahead, &mc);
        row_ok = row_rc == GRAPH_IO_OK;
        CHECK(row_ok || (mc.n == 0 && mc.wiring.input_count == 0 && mc.file == NULL), "a refused find leaves cfg zeroed");
        if (!row_ok) {   /* accepted by the library, no graph delivered: the sample's refusal */
            CHECK(row_rc == GRAPH_IO_ERR_NO_ROW, "sr %d fft %d N %d: expected GRAPH_IO_ERR_NO_ROW", cfg.sr, cfg.fft, n);
            ++no_row;
            continue;
        }
        CHECK(mc.model == GRAPH_IO_DFN2_PRODUCT && mc.n == n && mc.sample_rate == cfg.sr && mc.fft == cfg.fft && mc.file,
              "key echo");
        CHECK(mc.layout_version == (GRAPH_IO_DFN2_PRODUCT == GRAPH_IO_DFN2_LITE ? DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY
                                                                                        : DFN2_MODEL_IO_LAYOUT_VERSION_FULL),
              "layout version");
        CHECK((int)mc.wiring.outputs[0].w == grid->n_erb, "sr %d fft %d N %d: row erb width %u, grid %d", cfg.sr, cfg.fft, n,
              mc.wiring.outputs[0].w, grid->n_erb);
        {   /* the row's state count against the library's: a build constant, so any valid grid of the row's rate and size asks it */
            DFN2GridSpec gs;
            DFN2Grid dgrid;
            DFN2ModelIoDescriptor desc;
            memset(&gs, 0, sizeof gs);
            gs.sample_rate = cfg.sr; gs.n_fft = cfg.fft; gs.n_erb = (int)mc.wiring.outputs[0].w; gs.df_bins = 96;
#ifndef DFN2_MASK_ONLY
            gs.df_order = 5;
#endif
            gs.frames_per_call = n;
            CHECK(dfn2_grid_make(&dgrid, &gs) == DFN2_GRID_OK && dfn2_model_io_descriptor_default(&dgrid, &desc) == 0,
                  "sr %d fft %d N %d: no descriptor to compare the state count with", cfg.sr, cfg.fft, n);
            CHECK(graph_io_state_count(&mc) == desc.state_tensor_count, "sr %d fft %d N %d: row has %d states, the library %d",
                  cfg.sr, cfg.fft, n, graph_io_state_count(&mc), desc.state_tensor_count);
        }
        for (family = 0; family < 2; ++family) {
            char tag[96];
            snprintf(tag, sizeof tag, "sr %d fft %d N %d family %d", cfg.sr, cfg.fft, n, family);
            run_family(&cfg, family, n, &mc, grid->mask_lookahead, tag);
        }
        ++compared;
    }
    /* the sweep's accepted configurations are exactly the model's grids x N, and every row this model owns was compared */
    CHECK(accepted == EXPECTED_ACCEPTED, "library accepted %d configs, expected %d", accepted, EXPECTED_ACCEPTED);
    CHECK(compared == EXPECTED_ROWS, "compared %d rows, expected %d", compared, EXPECTED_ROWS);
    CHECK(no_row == EXPECTED_ACCEPTED - EXPECTED_ROWS, "%d accepted configs without a row, expected %d", no_row,
          EXPECTED_ACCEPTED - EXPECTED_ROWS);
    /* the variant switch: a model with no row at a grid is refused, an unknown model id is refused */
    {
        GraphIoConfig c;
        CHECK(graph_io_dfn2_find(GRAPH_IO_DFN2_LITE, 16000, 256, 1, 0, &c) == GRAPH_IO_ERR_NO_ROW, "lite has no 16 kHz row");
        CHECK(graph_io_dfn2_find(GRAPH_IO_DEEPVQES, 16000, 256, 1, 0, &c) == GRAPH_IO_ERR_NO_ROW, "foreign model");
        CHECK(graph_io_dfn2_find(GRAPH_IO_DFN2_FULL, 16000, 256, 1, 0, NULL) == GRAPH_IO_ERR_NULL, "NULL cfg");
        CHECK(graph_io_dfn2_find(GRAPH_IO_DFN2_FULL, 16000, 256, 2, 0, &c) == GRAPH_IO_ERR_NO_ROW, "16 kHz N = 2 has no row");
        CHECK(graph_io_dfn2_find(GRAPH_IO_DFN2_FULL, 48000, 1024, 4, 1, &c) == GRAPH_IO_ERR_NO_ROW, "48 kHz N = 4 has no row");
        CHECK(graph_io_dfn2_find(GRAPH_IO_DFN2_FULL, 48000, 1024, 1, 2, &c) == GRAPH_IO_ERR_NO_ROW, "48 kHz mask_lookahead 2 has no row");
    }
    if (g_fail) return 1;
    printf("test_ainr_graph_io[%s id %d]: PASS (library-accepted configs %d, rows %d of %d, real pre calls compared %d)\n",
           GRAPH_IO_DFN2_PRODUCT == GRAPH_IO_DFN2_LITE ? "lite" : "full", (int)AINR_PRODUCT_MODEL_ID, accepted,
           own_rows, GRAPH_IO_DFN2_ROW_COUNT, compared * 2);
    return 0;
}
