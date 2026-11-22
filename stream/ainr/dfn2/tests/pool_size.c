/* Pool size of dfn2_stream: the STFT and ISTFT borrow the stream's one shared FFT handle, so the pool
 * carries a single FFT table, not three. For each grid row:
 *   - get_mem_size() equals the literal byte count below and does not depend on the ERB pointers (the caller's tables
 *     are borrowed): sizing with NULL tables gives the same descriptor, for a full and a freq_only context alike;
 *   - that count is the byte count of a stream whose STFT and ISTFT each carry their own FFT table
 *     (before) less those two tables (2 * align16(fft_get_mem_size));
 *   - init_ex() succeeds on exactly that many bytes and refuses one aligned unit less, and a descriptor of the
 *     previous DFN2_STREAM_LAYOUT_VERSION;
 *   - a freq_only context (cfg.freq_only) needs the literal freq byte count: the full count less the FFT table,
 *     the STFT, the ISTFT and the PCM scratch slab, each taken from audiolib's own size queries;
 *   - 16 hops of a fixed input (N == 1) or 4 packets of N hops (N > 1) through a fixed stand-in accelerator hash to
 *     the golden value of that stream, with a stand-in inverse ERB table in which every bin splits 0.75 / 0.25 between
 *     two bands (a partition of unity, which init requires). The hash is the same for the NEON and the scalar kernels
 *     (a Row can still carry a separate NEON golden); the byte counts do not depend on SIMD. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_stream.h"
#include "audio_stft.h"
#include "fft_wrapper.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"   /* SK_HAVE_NEON */
#include "erb_fixture.h"

typedef struct Row {
    int sr, fft, n_erb, mask_lookahead, df_lookahead, n;
    uint64_t bytes;                       /* shared FFT handle */
    uint64_t before;                      /* STFT and ISTFT each with their own FFT table */
    uint64_t freq;                        /* freq_only context */
    uint64_t pcm_hash;                    /* scalar kernels (and NEON when it is not different) */
    uint64_t pcm_hash_neon;               /* 0: same as pcm_hash */
} Row;

#ifdef DFN2_MASK_ONLY
static const Row ROWS[] = {
    { 48000, 1024, 32, 1, 0, 1, 91408, 144112, 23888, 0xa46d74b9f5cac26eull, 0 },
};
#else
static const Row ROWS[] = {
    { 16000, 256, 24, 0, 0, 1, 63152, 77456, 45552, 0x2f7313027241b827ull, 0 },
    { 16000, 256, 24, 0, 0, 8, 103968, 118272, 86368, 0x2e1edf1fefe3f177ull, 0 },
    { 48000, 1024, 32, 1, 1, 1, 151792, 204496, 84272, 0xe0a89c487b7e4ad5ull, 0 },
};
#endif

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { fprintf(stderr, "FAIL (line %d): ", __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); ++g_fail; } } while (0)

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const unsigned char *c = (const unsigned char *)p;
    size_t i;
    for (i = 0; i < n; ++i) { h ^= c[i]; h *= 1099511628211ull; }
    return h;
}

static void fake_heads(DFN2PrepostOutputs *out, int hop_idx) {
    size_t i;
    for (i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] = 0.1f + 0.01f * (float)((i + (size_t)hop_idx) % 7);
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out->coefs_elements; ++i) out->coefs[i] = 0.02f * (float)((i + (size_t)hop_idx) % 5) - 0.05f;
    for (i = 0; i < out->alpha_elements; ++i) out->alpha[i] = 0.3f;
#endif
    for (i = 0; i < out->gru_hidden_elements; ++i) out->gru_hidden_next[i] = 0.0f;
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out->df_convp_window_elements; ++i) out->df_convp_window_next[i] = 0.0f;
#endif
}

static DFN2StreamConfig make_cfg(const Row *r, const float *fwd, const float *inv) {
    DFN2GridSpec spec;
    DFN2Grid grid;
    DFN2StreamConfig cfg;
    memset(&spec, 0, sizeof(spec));
    spec.sample_rate = r->sr;
    spec.n_fft = r->fft;
    spec.n_erb = r->n_erb;
    spec.df_bins = 96;
#ifdef DFN2_MASK_ONLY
    spec.df_order = 0;
#else
    spec.df_order = 5;
#endif
    spec.mask_lookahead = r->mask_lookahead;
    spec.df_lookahead = r->df_lookahead;
    spec.frames_per_call = r->n;
    if (dfn2_grid_make(&grid, &spec) != DFN2_GRID_OK) { fprintf(stderr, "grid_make fail\n"); exit(1); }
    memset(&cfg, 0, sizeof(cfg));
    cfg.grid = grid;
    cfg.erb_fwd = fwd;
    cfg.erb_inv = inv;
    return cfg;
}

static void run_row(const Row *r, int idx) {
    const size_t two_ffts = 2u * ck_align16_size(fft_get_mem_size(r->fft));
    DFN2StreamConfig cfg_mem, cfg_null;
    DFN2StreamMemReq req_mem, req_null;
    float *fwd, *inv;
    size_t i;
    int n_bins = r->fft / 2 + 1;
    DFN2Stream *s;
    void *pool;
    uint64_t h = 14695981039346656037ull, want;
    int hop, k, calls = r->n == 1 ? 16 : 4;
    int16_t *pcm_in, *pcm_out;

    fwd = (float *)malloc((size_t)n_bins * (size_t)r->n_erb * sizeof(float));
    inv = (float *)malloc((size_t)n_bins * (size_t)r->n_erb * sizeof(float));
    for (i = 0; i < (size_t)n_bins * (size_t)r->n_erb; ++i) { fwd[i] = 1.0f / (float)r->n_erb; }
    erb_fixture_split_inv_band_major(inv, n_bins, r->n_erb);   /* two bands per bin, so expansion sums */
    cfg_mem = make_cfg(r, fwd, inv);
    cfg_null = make_cfg(r, NULL, NULL);

    CHECK(dfn2_stream_get_mem_size(&cfg_mem, &req_mem) == 0, "row %d: sizing", idx);
    CHECK(dfn2_stream_get_mem_size(&cfg_null, &req_null) == 0, "row %d: sizing without tables", idx);
    CHECK(req_mem.bytes == r->bytes, "row %d (%d/%d N=%d): pool %llu, want %llu", idx, r->sr, r->fft, r->n,
          (unsigned long long)req_mem.bytes, (unsigned long long)r->bytes);
    CHECK(memcmp(&req_mem, &req_null, sizeof(req_mem)) == 0,
          "row %d: the descriptor depends on the ERB pointers (%llu bytes with tables, %llu without)", idx,
          (unsigned long long)req_mem.bytes, (unsigned long long)req_null.bytes);
    CHECK(r->before - two_ffts == r->bytes,
          "row %d: shrink is not exactly two FFT tables (%zu): %llu -> %llu", idx, two_ffts,
          (unsigned long long)r->before, (unsigned long long)r->bytes);

    {   /* freq_only: what is left out, from audiolib's size queries (STFT/ISTFT with their own FFT table less that table) */
        DFN2StreamConfig fm = cfg_mem, fn = cfg_null;
        DFN2StreamMemReq req_fm, req_fn;
        AcStftConfig scfg;
        size_t one_fft = two_ffts / 2u, pcm_half;
        ac_stft_config_default(&scfg, r->fft);
        pcm_half = one_fft + (ac_stft_get_mem_size(&scfg) - one_fft) + (ac_istft_get_mem_size(&scfg) - one_fft) +
                   ck_align16_size((size_t)(2 * (r->fft / 2) + 2 * n_bins) * sizeof(float));
        fm.freq_only = 1;
        fn.freq_only = 1;
        CHECK(dfn2_stream_get_mem_size(&fm, &req_fm) == 0 && dfn2_stream_get_mem_size(&fn, &req_fn) == 0, "row %d: freq_only sizing", idx);
        CHECK(req_fm.bytes == r->freq, "row %d (%d/%d N=%d): freq_only pool %llu, want %llu", idx, r->sr, r->fft, r->n,
              (unsigned long long)req_fm.bytes, (unsigned long long)r->freq);
        CHECK(memcmp(&req_fm, &req_fn, sizeof(req_fm)) == 0, "row %d: the freq_only descriptor depends on the ERB pointers", idx);
        CHECK(r->bytes - r->freq == pcm_half, "row %d: freq_only leaves out %llu, want %zu", idx,
              (unsigned long long)(r->bytes - r->freq), pcm_half);
    }

    /* exact size initialises; one aligned unit less is refused */
    pool = NULL;
    if (posix_memalign(&pool, 16, (size_t)req_mem.bytes) != 0) exit(1);
    CHECK(dfn2_stream_init_ex(&cfg_mem, pool, (size_t)req_mem.bytes - 16u, &req_mem) == NULL, "row %d: undersized pool accepted", idx);
    {   /* a descriptor of the previous layout is refused */
        DFN2StreamMemReq stale = req_mem;
        stale.layout_version = DFN2_STREAM_LAYOUT_VERSION - 1u;
        CHECK(dfn2_stream_init_ex(&cfg_mem, pool, (size_t)req_mem.bytes, &stale) == NULL,
              "row %d: a layout %u descriptor accepted", idx, (unsigned)stale.layout_version);
    }
    s = dfn2_stream_init_ex(&cfg_mem, pool, (size_t)req_mem.bytes, &req_mem);
    CHECK(s != NULL, "row %d: init on the exact pool size failed", idx);
    if (!s) { free(pool); free(fwd); free(inv); return; }

    hop = dfn2_stream_hop_size(s);
    pcm_in = (int16_t *)malloc((size_t)hop * (size_t)r->n * sizeof(int16_t));
    pcm_out = (int16_t *)malloc((size_t)hop * (size_t)r->n * sizeof(int16_t));
    for (i = 0; i < (size_t)calls; ++i) {
        DFN2HopIO io;
        int ai_rc = 0;
        for (k = 0; k < hop * r->n; ++k) pcm_in[k] = (int16_t)((k * 13 + (int)i * 5) % 211 - 100 + ((k / 70) & 1) * 900);
        if ((r->n == 1 ? dfn2_stream_pre_process(s, pcm_in, &io) : dfn2_stream_pre_process_packet(s, pcm_in, &io)) != 1) {
            CHECK(0, "row %d: pre_process at call %zu", idx, i);
            break;
        }
        if (io.action == STREAM_WORK_SUBMIT) fake_heads(&io.outputs, (int)i);
        if ((r->n == 1 ? dfn2_stream_post_process(s, ai_rc, pcm_out) : dfn2_stream_post_process_packet(s, ai_rc, pcm_out)) < 0) {
            CHECK(0, "row %d: post_process at call %zu", idx, i);
            break;
        }
        h = fnv(h, pcm_out, (size_t)hop * (size_t)r->n * sizeof(int16_t));
    }
    want = (SK_HAVE_NEON && r->pcm_hash_neon) ? r->pcm_hash_neon : r->pcm_hash;
    CHECK(h == want, "row %d (%d/%d N=%d): pcm hash %016llx, want %016llx", idx, r->sr, r->fft, r->n,
          (unsigned long long)h, (unsigned long long)want);
    free(pcm_in); free(pcm_out); free(pool); free(fwd); free(inv);
}

/* The descriptor is a fixed-width 32-byte POD and the carve alignment is audiolib's 16 bytes. */
static void pin_abi(void) {
    size_t size = sizeof(DFN2StreamMemReq), align = (size_t)DFN2_STREAM_ALIGNMENT;
    if (size != 32u) { printf("FAIL: DFN2StreamMemReq is %zu bytes, pinned to 32\n", size); ++g_fail; }
    if (align != 16u || ck_align16_size(1u) != align) {
        printf("FAIL: DFN2_STREAM_ALIGNMENT is %zu (ck_align16_size(1) = %zu), pinned to the shared 16\n", align,
               ck_align16_size(1u));
        ++g_fail;
    }
}

int main(void) {
    size_t i;
    pin_abi();
    for (i = 0; i < sizeof(ROWS) / sizeof(ROWS[0]); ++i) run_row(&ROWS[i], (int)i);
    if (g_fail) { printf("pool_size: %d check(s) FAILED\n", g_fail); return 1; }
    printf("pool_size: %d grid(s) -- PASS\n", (int)(sizeof(ROWS) / sizeof(ROWS[0])));
    return 0;
}
