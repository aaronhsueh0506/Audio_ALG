/* Pool size of dfn2_stream: the STFT and ISTFT borrow the stream's one shared FFT handle, so the pool
 * carries a single FFT table, not three. For each grid row, in memory mode (caller ERB tables) and
 * file mode (pool-owned ERB storage):
 *   - get_mem_size() equals the literal byte count below;
 *   - that count is the byte count of a stream whose STFT and ISTFT each carry their own FFT table
 *     (before_*) less those two tables (2 * align16(fft_get_mem_size));
 *   - init_ex() succeeds on exactly that many bytes and refuses one aligned unit less;
 *   - 16 hops of a fixed input through a fixed stand-in accelerator hash to the golden value of
 *     that own-table stream. The mask-only grid has a separate golden for the NEON kernels (not
 *     bit-identical to the scalar ones); the byte counts do not depend on SIMD. */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_stream.h"
#include "fft_wrapper.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"   /* SK_HAVE_NEON */

typedef struct Row {
    int sr, fft, n_erb, mask_lookahead, df_lookahead, n;
    uint64_t bytes_mem, bytes_file;       /* shared FFT handle */
    uint64_t before_mem, before_file;     /* STFT and ISTFT each with their own FFT table */
    uint64_t pcm_hash;                    /* scalar kernels (and NEON when it is not different) */
    uint64_t pcm_hash_neon;               /* 0: same as pcm_hash */
} Row;

#ifdef DFN2_MASK_ONLY
static const Row ROWS[] = {
    { 48000, 1024, 32, 1, 0, 1, 103952, 300944, 156656, 353648, 0xcb761989f38d9434ull, 0xc50a2aa70f79da9dull },
};
#else
static const Row ROWS[] = {
    { 16000, 256, 24, 0, 0, 1, 66464, 103616, 80768, 117920, 0x3249a26ef177c9b1ull, 0 },
    { 16000, 256, 24, 0, 0, 8, 121840, 158992, 136144, 173296, 0xd426f2bbb757d1e3ull, 0 },
    { 48000, 1024, 32, 1, 1, 1, 164320, 361312, 217024, 414016, 0x5970f6d7938c79a7ull, 0 },
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
    DFN2StreamConfig cfg_mem, cfg_file;
    DFN2StreamMemReq req_mem, req_file;
    float *fwd, *inv;
    size_t i;
    int n_bins = r->fft / 2 + 1;
    DFN2Stream *s;
    void *pool;
    uint64_t h = 14695981039346656037ull, want;
    int hop, k, hops = 16;
    int16_t *pcm_in, *pcm_out;

    fwd = (float *)malloc((size_t)n_bins * (size_t)r->n_erb * sizeof(float));
    inv = (float *)malloc((size_t)n_bins * (size_t)r->n_erb * sizeof(float));
    for (i = 0; i < (size_t)n_bins * (size_t)r->n_erb; ++i) { fwd[i] = 1.0f / (float)r->n_erb; inv[i] = 1.0f; }
    cfg_mem = make_cfg(r, fwd, inv);
    cfg_file = make_cfg(r, NULL, NULL);

    CHECK(dfn2_stream_get_mem_size(&cfg_mem, &req_mem) == 0, "row %d: mem sizing", idx);
    CHECK(dfn2_stream_get_mem_size(&cfg_file, &req_file) == 0, "row %d: file sizing", idx);
    CHECK(req_mem.bytes == r->bytes_mem, "row %d (%d/%d N=%d): memory mode pool %llu, want %llu", idx, r->sr, r->fft, r->n,
          (unsigned long long)req_mem.bytes, (unsigned long long)r->bytes_mem);
    CHECK(req_file.bytes == r->bytes_file, "row %d (%d/%d N=%d): file mode pool %llu, want %llu", idx, r->sr, r->fft, r->n,
          (unsigned long long)req_file.bytes, (unsigned long long)r->bytes_file);
    CHECK(r->before_mem - two_ffts == r->bytes_mem && r->before_file - two_ffts == r->bytes_file,
          "row %d: shrink is not exactly two FFT tables (%zu): %llu -> %llu / %llu -> %llu", idx, two_ffts,
          (unsigned long long)r->before_mem, (unsigned long long)r->bytes_mem,
          (unsigned long long)r->before_file, (unsigned long long)r->bytes_file);

    /* exact size initialises; one aligned unit less is refused */
    pool = NULL;
    if (posix_memalign(&pool, 16, (size_t)req_mem.bytes) != 0) exit(1);
    CHECK(dfn2_stream_init_ex(&cfg_mem, pool, (size_t)req_mem.bytes - 16u, &req_mem) == NULL, "row %d: undersized pool accepted", idx);
    s = dfn2_stream_init_ex(&cfg_mem, pool, (size_t)req_mem.bytes, &req_mem);
    CHECK(s != NULL, "row %d: init on the exact pool size failed", idx);
    if (!s) { free(pool); free(fwd); free(inv); return; }

    hop = dfn2_stream_hop_size(s);
    pcm_in = (int16_t *)malloc((size_t)hop * sizeof(int16_t));
    pcm_out = (int16_t *)malloc((size_t)hop * sizeof(int16_t));
    for (i = 0; i < (size_t)hops; ++i) {
        DFN2HopIO io;
        int ai_rc = 0;
        for (k = 0; k < hop; ++k) pcm_in[k] = (int16_t)((k * 13 + (int)i * 5) % 211 - 100 + ((k / 70) & 1) * 900);
        if (dfn2_stream_pre_process(s, pcm_in, &io) != 1) { CHECK(0, "row %d: pre_process at hop %zu", idx, i); break; }
        if (io.job_ready) fake_heads(&io.outputs, (int)i);
        if (dfn2_stream_post_process(s, ai_rc, pcm_out) < 0) { CHECK(0, "row %d: post_process at hop %zu", idx, i); break; }
        h = fnv(h, pcm_out, (size_t)hop * sizeof(int16_t));
    }
    want = (SK_HAVE_NEON && r->pcm_hash_neon) ? r->pcm_hash_neon : r->pcm_hash;
    CHECK(h == want, "row %d (%d/%d N=%d): pcm hash %016llx, want %016llx", idx, r->sr, r->fft, r->n,
          (unsigned long long)h, (unsigned long long)want);
    free(pcm_in); free(pcm_out); free(pool); free(fwd); free(inv);
}

int main(void) {
    size_t i;
    for (i = 0; i < sizeof(ROWS) / sizeof(ROWS[0]); ++i) run_row(&ROWS[i], (int)i);
    if (g_fail) { printf("pool_size: %d check(s) FAILED\n", g_fail); return 1; }
    printf("pool_size: %d grid(s), memory and file mode -- PASS\n", (int)(sizeof(ROWS) / sizeof(ROWS[0])));
    return 0;
}
