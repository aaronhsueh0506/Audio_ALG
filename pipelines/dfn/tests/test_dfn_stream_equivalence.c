/* DfnResStage against dfn2_stream_*_freq (stream/ainr/dfn2, memory-mode init): two thin wrappers over
 * the same dfn2_prepost_* calls, the stream's caller applying dfn2_stft_scale / dfn2_stft_unscale
 * that the stage applies itself. For six grids (the catalog's 16k/256 and 48k/1024, 32k/512, 16k/512
 * and lookahead pairs 2/4, 2/3, 0/1 that no catalog row has) and eleven scenarios of 400 hops each, fed the
 * same spectra and the same deterministic model, the model-visible windows, the output spectra
 * (bit for bit), the state generation, the return codes and the valid flag are compared every hop.
 * Scenarios: plain; attenuation limit from the start; a model that reports failure; a model that ran
 * nothing (STREAM_RUN_NOT_STARTED); heads left unwritten; a NaN bin; E and P the same pointers; a 1e21 bin (beyond 2^63 at model scale);
 * the largest bin below the limit and the next float above it; a reset; an attenuation change.
 * The refusal scenarios also pin that a refused block leaves the generation untouched. Two injected
 * faults (a perturbed bin, a reset applied to one side only) must be reported, so the comparator
 * is shown to fail. The build's SIMD setting applies to both sides (make SIMD=0 test). */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn/dfn_res_stage.h"
#include "dfn2_stream.h"

#define HOPS 400

typedef struct Row { int sr, fft, n_erb, df_bins, mask_la, df_la; } Row;

enum { S_PLAIN, S_ATTEN, S_MODEL_RC, S_NOT_STARTED, S_UNWRITTEN, S_NAN, S_SAME_EP, S_LARGE, S_LIMIT, S_RESET, S_ATTEN_CHANGE,
       S_COUNT };
static const char *const S_NAME[S_COUNT] = {
    "plain", "attenuation limit 12 dB from the start", "model reports failure (3 hops)", "model ran nothing (3 hops)",
    "heads left unwritten (3 hops)",
    "NaN bin at hop 80", "E and P the same pointers", "1e21 bin at hop 80", "limit bin at hop 80, next float at 81",
    "reset at hop 100", "attenuation change at hop 100" };

enum { INJ_NONE, INJ_PERTURB, INJ_ONE_SIDED_RESET };

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float urand(uint32_t *s) { return ((float)(int)(lcg(s) >> 9) / 4194304.0f - 1.0f); }

static void *al(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 16, n ? n : 16) != 0) { fprintf(stderr, "out of memory\n"); exit(2); }
    return p;
}

/* Contiguous bands, each bin owned by exactly one: forward = band mean, inverse = 1. */
static void make_erb(const Row *r, float *fwd, float *inv_band, int n_bins) {
    int b, k;
    memset(fwd, 0, sizeof(float) * (size_t)n_bins * (size_t)r->n_erb);
    memset(inv_band, 0, sizeof(float) * (size_t)n_bins * (size_t)r->n_erb);
    for (b = 0; b < r->n_erb; ++b) {
        int lo = (int)((long)b * n_bins / r->n_erb), hi = (int)((long)(b + 1) * n_bins / r->n_erb);
        for (k = lo; k < hi; ++k) {
            fwd[(size_t)k * (size_t)r->n_erb + (size_t)b] = 1.0f / (float)(hi - lo);
            inv_band[(size_t)b * (size_t)n_bins + (size_t)k] = 1.0f;
        }
    }
}

/* A model whose heads depend on the windows it is shown (hashed by bit pattern, so a non-finite
 * window is as well defined as a finite one) and on the hop. */
static void model(uint32_t hop, const float *erb, size_t ne, const float *spec, size_t ns,
                  float *mask, size_t nm, float *coefs, size_t nc, float *alpha, size_t na, int skip_write) {
    uint32_t h = hop * 2654435761u, bits;
    size_t i;
    for (i = 0; i < ne; ++i) { memcpy(&bits, &erb[i], sizeof bits); h = (h ^ bits) * 16777619u; }
    for (i = 0; i < ns; ++i) { memcpy(&bits, &spec[i], sizeof bits); h = (h ^ bits) * 16777619u; }
    if (skip_write) return;
    for (i = 0; i < nm; ++i) { lcg(&h); mask[i] = 0.2f + 0.7f * (float)((h >> 8) & 0xffff) / 65535.0f; }
    for (i = 0; i < nc; ++i) { lcg(&h); coefs[i] = 0.05f * (float)((int)((h >> 8) & 0xffff) - 32768) / 32768.0f; }
    for (i = 0; i < na; ++i) { lcg(&h); alpha[i] = 0.5f * (float)((h >> 8) & 0xffff) / 65535.0f; }
}

/* The largest audio-scale bin whose product with `scale` is within 2^63 (float arithmetic, as both
 * sides compute it), found by walking the floats around the quotient. */
static float last_accepted_bin(float scale) {
    const float limit = 9.223372036854775808e18f;
    float x = limit / scale;
    while (x * scale > limit) x = nextafterf(x, 0.0f);
    while (nextafterf(x, INFINITY) * scale <= limit) x = nextafterf(x, INFINITY);
    return x;
}

typedef struct Result { int bad, refused, valid, compared; } Result;

static Result run(const Row *r, int scenario, int inject) {
    DFN2GridSpec spec; DFN2Grid grid;
    float *fwd, *inv_band, *er, *ei, *pr, *pi_, *ser, *sei, *spr, *spi, *aor, *aoi;
    Complex *E, *P, *out;
    DfnResStageConfig scfg; DfnResStage *st;
    DFN2StreamConfig c; DFN2StreamMemReq req; DFN2Stream *s; void *pool;
    uint32_t seed = 12345u;
    int n_bins, k, h, mi = 0, mo = 0, mg = 0, mr = 0, mv = 0, first = -1;
    float scale, unscale, edge = 0.0f;
    Result res;

    memset(&res, 0, sizeof res);
    memset(&spec, 0, sizeof spec);
    spec.sample_rate = r->sr; spec.n_fft = r->fft; spec.n_erb = r->n_erb; spec.df_bins = r->df_bins;
    spec.df_order = 5; spec.mask_lookahead = r->mask_la; spec.df_lookahead = r->df_la; spec.frames_per_call = 1;
    if (dfn2_grid_make(&grid, &spec) != DFN2_GRID_OK) { fprintf(stderr, "%d/%d: grid refused\n", r->sr, r->fft); res.bad = 1; return res; }
    n_bins = grid.n_bins;
    scale = dfn2_stft_scale(grid.n_fft); unscale = dfn2_stft_unscale(grid.n_fft);
    edge = last_accepted_bin(scale);
    fwd = (float *)al(sizeof(float) * (size_t)n_bins * (size_t)r->n_erb);
    inv_band = (float *)al(sizeof(float) * (size_t)n_bins * (size_t)r->n_erb);
    make_erb(r, fwd, inv_band, n_bins);

    scfg = dfn_res_stage_default_config(&grid); scfg.erb_fwd = fwd; scfg.erb_inv = inv_band;
    st = dfn_res_stage_create(&scfg);
    memset(&c, 0, sizeof c); c.grid = grid; c.erb_fwd = fwd; c.erb_inv = inv_band;
    if (!st || dfn2_stream_get_mem_size(&c, &req) != 0) { fprintf(stderr, "%d/%d: stage or stream sizing refused\n", r->sr, r->fft); res.bad = 1; return res; }
    pool = al((size_t)req.bytes);
    s = dfn2_stream_init_ex(&c, pool, (size_t)req.bytes, &req);
    if (!s) { fprintf(stderr, "%d/%d: stream init refused\n", r->sr, r->fft); res.bad = 1; return res; }

    er = (float *)al(4 * (size_t)n_bins); ei = (float *)al(4 * (size_t)n_bins);
    pr = (float *)al(4 * (size_t)n_bins); pi_ = (float *)al(4 * (size_t)n_bins);
    ser = (float *)al(4 * (size_t)n_bins); sei = (float *)al(4 * (size_t)n_bins);
    spr = (float *)al(4 * (size_t)n_bins); spi = (float *)al(4 * (size_t)n_bins);
    aor = (float *)al(4 * (size_t)n_bins); aoi = (float *)al(4 * (size_t)n_bins);
    E = (Complex *)al(sizeof(Complex) * (size_t)n_bins); P = (Complex *)al(sizeof(Complex) * (size_t)n_bins);
    out = (Complex *)al(sizeof(Complex) * (size_t)n_bins);
    if (scenario == S_ATTEN) { dfn_res_stage_set_atten_lim(st, 12.0f); dfn2_stream_set_atten_lim(s, 12.0f); }

    for (h = 0; h < HOPS; ++h) {
        DfnResStageJob job; DFN2HopIO io;
        int srb, arb, skip = 0, model_rc = 0, same_ep = (scenario == S_SAME_EP), s_end, a_post, valid = -7;
        unsigned int gen_s0, gen_a0;
        for (k = 0; k < n_bins; ++k) {
            float a = urand(&seed) * 30.0f * (1.0f + (float)(h % 7)), b = urand(&seed) * 30.0f * (1.0f + (float)(h % 7));
            float g = 0.3f + 0.6f * (float)((h * 13 + k) % 17) / 17.0f;
            er[k] = a; ei[k] = b; pr[k] = same_ep ? a : a * g; pi_[k] = same_ep ? b : b * g;
        }
        if (h == 150) {   /* digital silence */
            memset(er, 0, 4 * (size_t)n_bins); memset(ei, 0, 4 * (size_t)n_bins);
            memset(pr, 0, 4 * (size_t)n_bins); memset(pi_, 0, 4 * (size_t)n_bins);
        }
        if (scenario == S_NAN && h == 80) er[7] = NAN;
        if (scenario == S_LARGE && h == 80) er[7] = 1e21f;
        if (scenario == S_LIMIT && h == 80) er[7] = edge;
        if (scenario == S_LIMIT && h == 81) er[7] = nextafterf(edge, INFINITY);
        for (k = 0; k < n_bins; ++k) {
            E[k].r = er[k]; E[k].i = ei[k]; P[k].r = pr[k]; P[k].i = pi_[k];
            ser[k] = er[k] * scale; sei[k] = ei[k] * scale; spr[k] = pr[k] * scale; spi[k] = pi_[k] * scale;
        }
        if (inject == INJ_PERTURB && h >= 200) spr[5] = spr[5] * 1.0000002f;
        if (scenario == S_UNWRITTEN && (h == 60 || h == 61 || h == 200)) skip = 1;
        if (scenario == S_MODEL_RC && (h == 40 || h == 41 || h == 120)) model_rc = STREAM_RUN_FAILED;
        if (scenario == S_NOT_STARTED && (h == 40 || h == 41 || h == 120)) model_rc = STREAM_RUN_NOT_STARTED;
        if (scenario == S_RESET && h == 100) { dfn_res_stage_reset(st); if (inject != INJ_ONE_SIDED_RESET) dfn2_stream_reset(s); }
        if (scenario == S_ATTEN_CHANGE && h == 100) { dfn_res_stage_set_atten_lim(st, 9.0f); dfn2_stream_set_atten_lim(s, 9.0f); }

        gen_s0 = dfn_res_stage_state_generation(st); gen_a0 = dfn2_stream_state_generation(s);   /* after a reset or a change of the limit */
        srb = same_ep ? dfn_res_stage_begin(st, E, E, &job) : dfn_res_stage_begin(st, E, P, &job);
        arb = dfn2_stream_pre_process_freq_dual(s, ser, sei, same_ep ? ser : spr, same_ep ? sei : spi, &io);
        /* stage: 1 job, 0 accepted without a model frame, < 0 refused; stream: 1 accepted (io.job_ready says whether
         * there is a job), else refused */
        if ((srb >= 0) != (arb == 1) || (srb == 1) != (arb == 1 && io.job_ready)) {
            ++mr; if (first < 0) first = h;
            fprintf(stderr, "  hop %d: begin disagrees (stage %d, stream %d, job_ready %d)\n", h, srb, arb, io.job_ready);
            break;
        }
        if (srb < 0) {   /* both refused the block: nothing may have been consumed */
            ++res.refused;
            if (dfn_res_stage_state_generation(st) != gen_s0 || dfn2_stream_state_generation(s) != gen_a0) {
                ++mg; if (first < 0) first = h;
            }
            continue;
        }
        if (srb == 0) {   /* no model frame: the status is ignored and the generation does not move */
            s_end = dfn_res_stage_end(st, STREAM_RUN_FAILED, out);
            a_post = dfn2_stream_post_process_freq(s, STREAM_RUN_FAILED, aor, aoi, &valid);
            if (s_end < 0 || a_post != 0 || s_end != 0 || valid != 0) { ++mr; if (first < 0) first = h; continue; }
            if (dfn_res_stage_state_generation(st) != gen_s0 || dfn2_stream_state_generation(s) != gen_a0) {
                ++mg; if (first < 0) first = h;
            }
            ++res.compared;
            continue;
        }
        if (io.inputs.erb_window_elements != job.inputs.erb_window_elements ||
            io.inputs.spec_window_elements != job.inputs.spec_window_elements ||
            memcmp(io.inputs.erb_window, job.inputs.erb_window, job.inputs.erb_window_elements * 4) != 0 ||
            memcmp(io.inputs.spec_window, job.inputs.spec_window, job.inputs.spec_window_elements * 4) != 0 ||
            io.state_generation != dfn_res_stage_state_generation(st)) { ++mi; if (first < 0) first = h; }
        model((uint32_t)h, job.inputs.erb_window, job.inputs.erb_window_elements,
              job.inputs.spec_window, job.inputs.spec_window_elements,
              job.outputs.erb_mask, job.outputs.erb_mask_elements, job.outputs.coefs, job.outputs.coefs_elements,
              job.outputs.alpha, job.outputs.alpha_elements, skip);
        model((uint32_t)h, io.inputs.erb_window, io.inputs.erb_window_elements,
              io.inputs.spec_window, io.inputs.spec_window_elements,
              io.outputs.erb_mask, io.outputs.erb_mask_elements, io.outputs.coefs, io.outputs.coefs_elements,
              io.outputs.alpha, io.outputs.alpha_elements, skip);
        s_end = dfn_res_stage_end(st, model_rc, out);
        a_post = dfn2_stream_post_process_freq(s, model_rc, aor, aoi, &valid);
        if (s_end < 0 || a_post != 0) { ++mr; if (first < 0) first = h; continue; }
        if (s_end != valid) { ++mv; if (first < 0) first = h; }
        res.valid += valid == 1;
        ++res.compared;
        for (k = 0; k < n_bins; ++k) {
            float re = aor[k] * unscale, im = aoi[k] * unscale;
            if (memcmp(&out[k].r, &re, 4) != 0 || memcmp(&out[k].i, &im, 4) != 0) { ++mo; if (first < 0) first = h; break; }
        }
        if (dfn2_stream_state_generation(s) != dfn_res_stage_state_generation(st)) { ++mg; if (first < 0) first = h; }
    }
    res.bad = mi + mo + mg + mr + mv;
    if (res.bad && !inject)
        fprintf(stderr, "%d/%d L%d/%d, %s: mismatches window %d output %d generation %d return code %d valid flag %d, first bad hop %d\n",
                r->sr, r->fft, r->mask_la, r->df_la, S_NAME[scenario], mi, mo, mg, mr, mv, first);
    dfn_res_stage_destroy(st);
    free(pool); free(fwd); free(inv_band); free(er); free(ei); free(pr); free(pi_);
    free(ser); free(sei); free(spr); free(spi); free(aor); free(aoi); free(E); free(P); free(out);
    return res;
}

/* Hops each scenario refuses on both sides. */
static int expected_refused(int scenario) {
    return scenario == S_NAN || scenario == S_LARGE || scenario == S_LIMIT;
}

int main(void) {
    static const Row rows[] = {
        { 48000, 1024, 32, 96, 1, 1 },   /* catalog */
        { 16000, 256, 24, 96, 0, 0 },    /* catalog */
        { 32000, 512, 28, 96, 1, 1 },    /* no catalog row */
        { 48000, 1024, 32, 96, 2, 4 },   /* other lookahead pairs */
        { 16000, 256, 24, 96, 2, 3 },
        { 16000, 512, 24, 96, 0, 1 },    /* other grid and pair */
    };
    int ri, si, failures = 0, det1, det2;
    DFN2_REQUIRE_VARIANT_LINK();
    for (ri = 0; ri < (int)(sizeof(rows) / sizeof(rows[0])); ++ri) {
        for (si = 0; si < S_COUNT; ++si) {
            Result res = run(&rows[ri], si, INJ_NONE);
            if (res.bad) ++failures;
            if (res.refused != expected_refused(si)) {
                fprintf(stderr, "%d/%d L%d/%d, %s: %d hops refused, expected %d\n", rows[ri].sr, rows[ri].fft,
                        rows[ri].mask_la, rows[ri].df_la, S_NAME[si], res.refused, expected_refused(si));
                ++failures;
            }
            if (res.compared != HOPS - res.refused || res.valid < HOPS / 2) {
                fprintf(stderr, "%d/%d L%d/%d, %s: %d hops compared, %d with a frame (a comparison that sees nothing proves nothing)\n",
                        rows[ri].sr, rows[ri].fft, rows[ri].mask_la, rows[ri].df_la, S_NAME[si], res.compared, res.valid);
                ++failures;
            }
        }
        printf("dfn_stream_equivalence: %d Hz / FFT %d / lookahead %d/%d: %d scenarios x %d hops\n", rows[ri].sr,
               rows[ri].fft, rows[ri].mask_la, rows[ri].df_la, S_COUNT, HOPS);
    }
    det1 = run(&rows[2], S_PLAIN, INJ_PERTURB).bad;
    det2 = run(&rows[2], S_RESET, INJ_ONE_SIDED_RESET).bad;
    printf("dfn_stream_equivalence: injected faults reported: perturbed bin %s, one-sided reset %s\n",
           det1 > 0 ? "yes" : "NO", det2 > 0 ? "yes" : "NO");
    if (det1 <= 0 || det2 <= 0) ++failures;
    if (failures) { printf("dfn_stream_equivalence: %d FAILURE(S)\n", failures); return 1; }
    puts("dfn_stream_equivalence: PASS (windows, outputs, generation, return codes and valid flags bit-identical)");
    return 0;
}
