/* DFN2 pre/post chain on a grid given by arguments: the driver of test_dfn2_runtime_grid.py.
 *
 *   dfn2_runtime_grid_chain <sample_rate> <n_fft> <n_erb> <df_bins> <mask_lookahead> <df_lookahead>
 *
 * It builds the grid with dfn2_grid_make(), runs the whole pre/post chain through DFN2_IO_FREQ
 * and DFN2_IO_TIME with a stand-in accelerator, and prints a digest of everything the
 * accelerator is shown (the three feature windows, the recurrent state) and of everything the
 * chain emits. The stand-in writes plausible heads, and also tiny, huge and non-finite heads
 * and recurrent state, and takes a skipped and an unwritten frame, so the refusal paths run
 * as well.
 *
 * Nothing here depends on the capacities the library was built with (-DDFN2_MAX_N_FFT, ...):
 * every array below is sized for the largest grid the driver accepts (48 kHz, 1024-point
 * frames), and only the packed prefix the grid uses is ever hashed. The same binary arguments
 * must therefore print the same digests whatever capacities the library units were compiled
 * with; a library stride that reads the capacity instead of the grid breaks that. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_prepost.h"

#define HOPS 640
#define MAX_BINS 513
#define MAX_ERB 32
#define MAX_HOP 512

static DFN2Grid grid;

#define RNG_SEED 0x6b1d5eedu
static uint32_t rng_state = RNG_SEED;
static uint32_t rng(void)
{
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state >> 8;
}
static float uniform(void) { return (float)(rng() & 0xffff) / 32768.0f - 1.0f; }
static void fill(float *x, size_t n, float scale)
{
    for (size_t i = 0; i < n; ++i) x[i] = uniform() * scale;
}

typedef struct {
    uint64_t in, state, post, status;
} Digest;

static void digest_init(Digest *d)
{
    d->in = d->state = d->post = d->status = UINT64_C(14695981039346656037);
}
static void hash_u32(uint64_t *h, uint32_t v) { *h = (*h ^ v) * UINT64_C(1099511628211); }
static void hash_f32(uint64_t *h, const float *x, size_t n)
{
    for (size_t i = 0; i < n; ++i) {
        uint32_t bits;
        if (isnan(x[i])) bits = 0x7fc00000u;   /* one NaN, whatever payload the hardware gives */
        else memcpy(&bits, &x[i], sizeof(bits));
        hash_u32(h, bits);
    }
}
static uint64_t digest_fold(const Digest *d)
{
    uint64_t h = UINT64_C(14695981039346656037);
    const uint64_t parts[4] = {d->in, d->state, d->post, d->status};
    for (int i = 0; i < 4; ++i) {
        hash_u32(&h, (uint32_t)parts[i]);
        hash_u32(&h, (uint32_t)(parts[i] >> 32));
    }
    return h;
}

/* A noisy spectrum whose level changes from hop to hop, with silent, tiny and loud hops. */
static void make_spectrum(int hop, int bins, float *re, float *im)
{
    float level = 0.3f;
    if (hop % 9 == 0) level = 1e-4f;
    if (hop % 31 == 7) level = 4.0f;
    if (hop % 53 == 11) level = 0.0f;
    if (hop % 71 == 13) level = 1e-20f;
    for (int k = 0; k < bins; ++k) {
        float m = level / (1.0f + 0.05f * (float)k) * (0.5f + 0.5f * fabsf(uniform()));
        re[k] = m * uniform();
        im[k] = m * uniform();
    }
}

enum { K_NORMAL, K_TINY, K_HUGE, K_BAD_HEAD, K_BAD_HEAD_INF, K_BAD_STATE_FIRST, K_BAD_STATE_LAST,
       K_BAD_STATE_MID, K_SKIP, K_UNWRITTEN };
static int hop_kind(int hop)
{
    switch (hop % 40) {
    case 5: return K_TINY;
    case 9: return K_HUGE;
    case 13: return K_BAD_HEAD;
    case 17: return K_BAD_HEAD_INF;
    case 21: return K_BAD_STATE_FIRST;
    case 25: return K_BAD_STATE_LAST;
    case 29: return K_BAD_STATE_MID;
    case 33: return K_SKIP;
    case 37: return K_UNWRITTEN;
    default: return K_NORMAL;
    }
}
static float kind_scale(int kind)
{
    return kind == K_TINY ? 1e-30f : kind == K_HUGE ? 1e38f : 1.0f;
}
static void poison(float *x, size_t n, int where, int hop)
{
    static const uint32_t bad[3] = {0x7fc00000u, 0x7f800000u, 0xff800000u};
    size_t at = where == 0 ? 0 : where == 1 ? n - 1 : n / 2;
    memcpy(&x[at], &bad[(hop / 40) % 3], sizeof(float));
}
static int poison_where(int kind)
{
    return kind == K_BAD_STATE_FIRST ? 0 : kind == K_BAD_STATE_LAST ? 1 : 2;
}
static int is_bad_state(int kind)
{
    return kind == K_BAD_STATE_FIRST || kind == K_BAD_STATE_LAST || kind == K_BAD_STATE_MID;
}

/* ERB tables for any grid: contiguous bands with half-weight shared edges (forward
 * bin-major [n_bins][n_erb] normalised by the band width, inverse band-major [n_erb][n_bins]). */
static void tables_banded(int n_bins, int n_erb, float *fwd, float *inv)
{
    int edge[MAX_ERB + 1];
    memset(fwd, 0, sizeof(float) * (size_t)n_bins * n_erb);
    memset(inv, 0, sizeof(float) * (size_t)n_bins * n_erb);
    for (int b = 0; b <= n_erb; ++b) {
        edge[b] = (int)((float)b * (float)n_bins / (float)n_erb);
        if (b > 0 && edge[b] < edge[b - 1] + 2) edge[b] = edge[b - 1] + 2;
    }
    edge[n_erb] = n_bins;
    for (int b = 0; b < n_erb; ++b) {
        int w = edge[b + 1] - edge[b];
        for (int k = edge[b]; k < edge[b + 1]; ++k) {
            float inner = (k == edge[b] && b > 0) ? 0.5f : 1.0f;
            inv[(size_t)b * n_bins + k] = inner;
            fwd[(size_t)k * n_erb + b] = inner / (float)w;
            if (k == edge[b] && b > 0) {
                inv[(size_t)(b - 1) * n_bins + k] = 0.5f;
                fwd[(size_t)k * n_erb + (b - 1)] = 0.5f / (float)(edge[b] - edge[b - 1]);
            }
        }
    }
}

/* Irregular spans at any offset, some weights exactly zero. */
static void tables_irregular(int n_bins, int n_erb, float *fwd, float *inv)
{
    memset(fwd, 0, sizeof(float) * (size_t)n_bins * n_erb);
    memset(inv, 0, sizeof(float) * (size_t)n_bins * n_erb);
    for (int b = 0; b < n_erb; ++b) {
        int width = 1 + (int)(rng() % 40);
        int lo = (int)(rng() % (unsigned)(n_bins - width));
        for (int k = lo; k < lo + width; ++k)
            inv[(size_t)b * n_bins + k] = (k % 7 == 3) ? 0.0f : 0.5f + 0.5f * uniform();
    }
    for (int k = 0; k < n_bins; ++k) {
        int b = k * (n_erb - 1) / n_bins;
        fwd[(size_t)k * n_erb + b] = 0.5f + 0.5f * uniform();
        fwd[(size_t)k * n_erb + b + 1] = 0.5f + 0.5f * uniform();
    }
}

static float fwd_a[MAX_BINS * MAX_ERB], inv_a[MAX_BINS * MAX_ERB];
static float fwd_b[MAX_BINS * MAX_ERB], inv_b[MAX_BINS * MAX_ERB];

static void write_heads(const DFN2PrepostOutputs *out, int kind)
{
    float s = kind_scale(kind);
    fill(out->erb_mask, out->erb_mask_elements, 0.5f * s);
    for (size_t i = 0; i < out->erb_mask_elements; ++i) out->erb_mask[i] += 0.5f * s;
#ifndef DFN2_MASK_ONLY
    fill(out->coefs, out->coefs_elements, 0.1f * s);
    out->alpha[0] = 0.5f * s * (1.0f + uniform());
#endif
}

static uint64_t run_chain(int io_time)
{
    static float re[MAX_BINS], im[MAX_BINS], app_re[MAX_BINS], app_im[MAX_BINS];
    static float out_re[MAX_BINS], out_im[MAX_BINS];
    static float hop_in[MAX_HOP], hop_out[MAX_HOP];
    DFN2PrepostConfig cfg;
    DFN2Prepost *p;
    Digest d;
    FftHandle *fft = NULL;
    AcIstft *istft = NULL;
    void *istft_mem = NULL;

    rng_state = RNG_SEED;
    digest_init(&d);
    tables_banded(grid.n_bins, grid.n_erb, fwd_a, inv_a);
    tables_irregular(grid.n_bins, grid.n_erb, fwd_b, inv_b);
    if (dfn2_prepost_config_defaults(&cfg, io_time ? DFN2_IO_TIME : DFN2_IO_FREQ, &grid) != 0)
        exit(2);
    cfg.erb_fwd = fwd_a;
    cfg.erb_inv = inv_a;
    if (io_time) {
        AcStftConfig sc;
        size_t bytes;
        fft = fft_create(grid.n_fft);
        dfn2_framing_config(&sc, &grid, fft, NULL);
        bytes = ac_istft_get_mem_size(&sc);
        istft_mem = aligned_alloc(16, (bytes + 15) & ~(size_t)15);
        istft = ac_istft_init(istft_mem, bytes, &sc);
        cfg.fft = fft;
        cfg.istft = istft;
    }
    p = dfn2_prepost_create(&cfg);
    if (!p) {
        printf("create failed\n");
        exit(2);
    }
    for (int h = 0; h < HOPS; ++h) {
        int v, valid = 0, kind = hop_kind(h);
        if (h == HOPS / 3) hash_u32(&d.status, (uint32_t)dfn2_prepost_set_atten_lim(p, 12.0f));
        if (h == HOPS / 2)
            hash_u32(&d.status, (uint32_t)dfn2_prepost_set_erb_matrices(p, fwd_b, inv_b));
        if (io_time) {
            for (int i = 0; i < grid.hop; ++i)
                hop_in[i] = 0.2f * uniform() * (h % 17 == 3 ? 0.0f : 1.0f);
            v = dfn2_prepost_pre_process(p, hop_in);
        } else {
            make_spectrum(h, grid.n_bins, re, im);
            if (h % 4 == 3) {
                make_spectrum(h + 1, grid.n_bins, app_re, app_im);
                v = dfn2_prepost_pre_process_freq_dual(p, re, im, app_re, app_im);
            } else {
                v = dfn2_prepost_pre_process_freq(p, re, im);
            }
        }
        hash_u32(&d.status, (uint32_t)v);
        if (v == 1) {
            DFN2PrepostInputs in;
            DFN2PrepostOutputs out;
            float *gru;
#ifndef DFN2_MASK_ONLY
            float *convp;
#endif
            int rc, skipped = 0;
            if (dfn2_prepost_frame_inputs(p, &in, &out) != 0) exit(2);
            hash_f32(&d.in, in.erb_window, in.erb_window_elements);
            hash_f32(&d.in, in.spec_window, in.spec_window_elements);
            hash_f32(&d.in, &in.gru_hidden[0][0], in.gru_hidden_elements);
#ifndef DFN2_MASK_ONLY
            hash_f32(&d.in, in.df_convp_window, in.df_convp_window_elements);
#endif
            gru = &out.gru_hidden_next[0][0];
#ifndef DFN2_MASK_ONLY
            convp = out.df_convp_window_next;
#endif
            if (kind != K_SKIP && kind != K_UNWRITTEN) {
                write_heads(&out, kind);
                fill(gru, out.gru_hidden_elements, kind_scale(kind));
#ifndef DFN2_MASK_ONLY
                fill(convp, out.df_convp_window_elements, kind_scale(kind));
                if (kind == K_BAD_HEAD) out.coefs[(size_t)h % out.coefs_elements] = NAN;
#else
                if (kind == K_BAD_HEAD) out.erb_mask[(size_t)h % out.erb_mask_elements] = NAN;
#endif
                if (kind == K_BAD_HEAD_INF) out.erb_mask[h % grid.n_erb] = INFINITY;
                if (is_bad_state(kind)) {
#ifndef DFN2_MASK_ONLY
                    if (h % 2) poison(gru, out.gru_hidden_elements, poison_where(kind), h);
                    else poison(convp, out.df_convp_window_elements, poison_where(kind), h);
#else
                    poison(gru, out.gru_hidden_elements, poison_where(kind), h);
#endif
                }
            }
            rc = kind == K_SKIP ? dfn2_prepost_frame_skip(p) : dfn2_prepost_frame_commit(p);
            if (rc != 0 && kind != K_SKIP) skipped = dfn2_prepost_frame_skip(p);
            hash_u32(&d.status, (uint32_t)rc);
            hash_u32(&d.status, (uint32_t)skipped);
            hash_u32(&d.status, dfn2_prepost_state_generation(p));
            hash_f32(&d.state, gru, in.gru_hidden_elements);
#ifndef DFN2_MASK_ONLY
            hash_f32(&d.state, convp, in.df_convp_window_elements);
#endif
        }
        if (io_time) {
            int written = 0;
            if (dfn2_prepost_post_process(p, hop_out, &written) != 0) exit(2);
            hash_u32(&d.status, (uint32_t)written);
            hash_f32(&d.post, hop_out, (size_t)grid.hop);
        } else {
            if (dfn2_prepost_post_process_freq(p, out_re, out_im, &valid) != 0) exit(2);
            hash_u32(&d.status, (uint32_t)valid);
            hash_f32(&d.post, out_re, (size_t)grid.n_bins);
            hash_f32(&d.post, out_im, (size_t)grid.n_bins);
        }
    }
    hash_u32(&d.status, (uint32_t)dfn2_prepost_hop_size(p));
    hash_u32(&d.status, (uint32_t)dfn2_prepost_num_bins(p));
    hash_u32(&d.status, (uint32_t)dfn2_prepost_model_lookahead_frames(p));
    dfn2_prepost_destroy(p);
    free(istft_mem);
    if (fft) fft_destroy(fft);
    printf("  %s in=%016llx state=%016llx post=%016llx status=%016llx\n", io_time ? "time" : "freq",
           (unsigned long long)d.in, (unsigned long long)d.state, (unsigned long long)d.post,
           (unsigned long long)d.status);
    return digest_fold(&d);
}

static int arg_int(const char *text, int *value)
{
    char *end = NULL;
    long v = strtol(text, &end, 10);
    if (end == text || *end != '\0' || v < -2147483647L || v > 2147483647L) return -1;
    *value = (int)v;
    return 0;
}

int main(int argc, char **argv)
{
    int v[6], status;
    uint64_t f, t;
    if (argc != 7) {
        fprintf(stderr, "usage: %s sample_rate n_fft n_erb df_bins mask_lookahead df_lookahead\n",
                argv[0]);
        return 2;
    }
    for (int i = 0; i < 6; ++i)
        if (arg_int(argv[i + 1], &v[i]) != 0) {
            fprintf(stderr, "not an integer: %s\n", argv[i + 1]);
            return 2;
        }
    status = dfn2_grid_make(&grid, v[0], v[1], v[2], v[3], v[4], v[5]);
    if (status != DFN2_GRID_OK) {
        printf("grid refused: %s\n", dfn2_grid_status_text(status));
        return 3;
    }
    if (grid.n_bins > MAX_BINS || grid.n_erb > MAX_ERB || grid.hop > MAX_HOP) {
        fprintf(stderr, "the driver's buffers hold grids up to 48 kHz / 1024-point frames\n");
        return 2;
    }
    printf("grid sr=%d fft=%d erb=%d df=%d mask=%d dfla=%d\n", grid.sample_rate, grid.n_fft,
           grid.n_erb, grid.df_bins, grid.mask_lookahead, grid.df_lookahead);
    f = run_chain(0);
    t = run_chain(1);
    printf("chain digest=%016llx:%016llx backend=%s\n", (unsigned long long)f,
           (unsigned long long)t, dfn2_simd_backend());
    return 0;
}
