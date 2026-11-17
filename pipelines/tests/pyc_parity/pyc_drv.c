/* pyc_drv.c -- C side of pyc_parity_pipelines.sh.
 *
 * One source, three executables: the 4-channel one links the production AEC archive, the mono one only
 * the libraries, and the 4-channel replay one the AEC_TESTING archive (the only place the array's gain stage
 * can be run on its own), so each build carries only its own front end:
 *
 *   pyc_drv_4ch        -DPYC_WITH_4CH                       modes: chain
 *   pyc_drv_4ch_replay -DPYC_WITH_4CH -DPYC_REPLAY_RES      modes: replay   (links libaec built with AEC_TESTING=1)
 *   pyc_drv_mono       -DPYC_WITH_MONO                      modes: mono    replay   probe
 *
 * Levels (aec=, nr=) are the enum numbers: AEC 0 mild, 1 balanced, 2 aggressive, 3 no_floor; NR 0 mild, 1 moderate,
 * 2 balanced, 3 aggressive.
 *
 * Both run the chain the example programs write out behind the front end (post_hop() below:
 * RES gain -> NR gain -> min -> apply -> comfort noise -> synthesis) over an AecResidualView: the
 * 4-channel build over the array instance's fused view (aec_process_residual() / aec_get_residual())
 * through 4ch/array_sample_post.h, the mono build over the lane's view as mono/aec_nr_res/main.c reads
 * it. A deviation planted in a stage is seen by the comparison.
 *
 * Command line: <mode> key=value ...
 *
 *   chain  mic=F ref=F sr=N fft=N out=F [aec=0..3] [nr=0..3]
 *          [legacy_amin=0|1] [enable_nr=0|1] [enable_res=0|1] [enable_cng=0|1]
 *     The 4-channel chain as 4ch/4ch_external_static.c composes it (array instance built at the aec= level:
 *     aec_process_linear -> aec_process_residual with equal 0.25 weights -> the array's gain stage -> NR
 *     stage -> min -> apply -> comfort noise -> synthesis), every hop dumped in the seam format below. Stops at the first non-zero
 *     status of the core or the chain, recording it in the header; the Python side treats that as
 *     a result.
 *     aec=3 (no_floor) is accepted like the other levels.
 *   mono   mic=F ref=F sr=N fft=N out=F [aec=..] [nr=..] [legacy_amin=0|1] [enable_cng=0|1] [reset_at=H]
 *     Aec + aec_process_linear / aec_process_residual / aec_get_residual + the chain as mono/aec_nr_res/main.c
 *     composes them, with the AEC seam read back after each hop. The chain applies the lane's own RES gain (frame
 *     gain) through the explicit-array forms of the min / apply / comfort-noise steps (the replay mode has no live
 *     Aec to own the generator). The chain renders the AEC's rendering base (AecResidualView.base_spec) and follows
 *     aec_post_reset_serial() (OLA clear, comfort-noise reseed); reset_at=H makes the driver call aec_reset() before hop H.
 *   replay seam=F out=F [aec=..] [nr=..] [legacy_amin=0|1] [enable_cng=0|1] [usable=F]
 *     Open-loop replay of the chain over a recorded seam: RES stage (4-channel: a standalone gain stage at the aec= level,
 *     built on the seam's grid and fed the recorded frame as the array feeds its own; mono: the recorded lane G_res),
 *     NR stage, min, apply to the recorded B, comfort noise, synthesis. `usable=F` is one
 *     byte per hop that replaces the recorded usable_linear flag (the Python verdict of the same hop).
 *     Output: header (8 x int32) then per hop float32 G_res(nf) G_nr(nf) out(hop).
 *   probe  sr=N fft=N          (mono build) derive the lane for the grid and print "hop n_freqs", or fail
 *   cng    seam=F out=F
 *     One aec_cng_fill call (caller-chosen nr floor, which aec_apply_cng_spec fixes at AEC_CNG_NR_GAIN_FLOOR). Input (seam=): int32 n, uint32 rng, float32 floor, int32 use_g_nr, then float32
 *     spec(2n) cn(n) g_res(n) g_nr(n). Output: uint32 rng after the call, then float32 spec(2n).
 *
 * Seam file (all little endian):
 *   int32 hdr[10] = { 0x53435950 'PYCS', 1, kind (0 mono, 1 4ch), sr, fft, n_freqs, hop, n_hops,
 *                     fail_hop (-1 = none), fail_rc }
 *   per hop float32: E(2nf) Y(2nf) B(2nf) R2(nf) CN(nf) Gres(nf) Gnr(nf) far_hop(hop) out(hop) sc[16]
 *   sc: 0 far_power  1 max_dt  2 max_sat  3 all_converged  4 clock_drift  5 usable_linear
 *       6 delay_samples  7 delay_solid  8 delay_changed  9 realign_warm_total  10 realign_soft_total
 *       11 delay_confidence  12 guard (mono: guard_active; 4ch: AecResidualView.guard_lanes)
 *       13 post_reset_serial (mono: aec_post_reset_serial(); 4ch: AecResidualView.post_reset_serial)  14..15 unused (0)
 * Complex spectra are interleaved (re, im). B is the rendering base the chain applied its gain to
 * (mono and 4ch: AecResidualView.base_spec; on the array a flag cannot rebuild it where only some lanes'
 * guards fired). Gres/Gnr are the gains the chain applied (unity when the source is disabled). The
 * seam is the data the stages behind the AEC read; scales as in AecResidualView (aec.h): E, Y, B in audio
 * scale, R2 and CN in int16^2 power scale, gains amplitudes. */
#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "aec_internal.h"
#if defined(PYC_REPLAY_RES)
#include "aec_test_hooks.h"
#endif
#include "anr.h"
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

#if defined(PYC_WITH_4CH)
#include "array_sample_post.h"
#endif
#define PYC_CNG_RNG_SEED AEC_CNG_SEED
/* The floor the Python twins pin; the chain's aec_apply_cng_spec uses the library's, checked in main(). */
#define PYC_CNG_NR_GAIN_FLOOR 0.31622777f

#define PYC_MAGIC 0x53435950
#define PYC_SC 16

typedef struct Args {
    const char *mic, *ref, *out, *seam, *usable;
    int sr, fft, legacy_amin, enable_nr, enable_res, enable_cng;
    int reset_at;                 /* mono: hop on which the application calls aec_reset() (-1: never) */
    AecLevel aec;
    AnrLevel nr;
} Args;

/* A level is its enum number, given as a whole decimal string 0..3. */
static int parse_level_arg(const char* text, int* level) {
    char* end = NULL;
    long v;
    if (!text || !*text || isspace((unsigned char)*text)) return -1;
    errno = 0;
    v = strtol(text, &end, 10);
    if (errno || *end != '\0' || v < 0 || v > 3) return -1;
    *level = (int)v;
    return 0;
}

static int parse_args(int argc, char** argv, Args* a) {
    int i;
    memset(a, 0, sizeof(*a));
    a->enable_nr = a->enable_res = a->enable_cng = 1;
    a->reset_at = -1;
    a->aec = AEC_LEVEL_BALANCED;
    a->nr = ANR_LEVEL_BALANCED;
    for (i = 2; i < argc; ++i) {
        char* eq = strchr(argv[i], '=');
        const char *k = argv[i], *v;
        if (!eq) { fprintf(stderr, "bad argument %s\n", argv[i]); return -1; }
        *eq = 0; v = eq + 1;
        if (!strcmp(k, "mic")) a->mic = v;
        else if (!strcmp(k, "ref")) a->ref = v;
        else if (!strcmp(k, "out")) a->out = v;
        else if (!strcmp(k, "seam")) a->seam = v;
        else if (!strcmp(k, "usable")) a->usable = v;
        else if (!strcmp(k, "sr")) a->sr = atoi(v);
        else if (!strcmp(k, "fft")) a->fft = atoi(v);
        else if (!strcmp(k, "legacy_amin")) a->legacy_amin = atoi(v);
        else if (!strcmp(k, "enable_nr")) a->enable_nr = atoi(v);
        else if (!strcmp(k, "enable_res")) a->enable_res = atoi(v);
        else if (!strcmp(k, "enable_cng")) a->enable_cng = atoi(v);
        else if (!strcmp(k, "reset_at")) a->reset_at = atoi(v);
        else if (!strcmp(k, "aec")) {
            int m;
            if (parse_level_arg(v, &m) != 0) { fprintf(stderr, "unknown aec level %s (0..3)\n", v); return -1; }
            a->aec = (AecLevel)m;
        } else if (!strcmp(k, "nr")) {
            int m;
            if (parse_level_arg(v, &m) != 0) { fprintf(stderr, "unknown nr level %s (0..3)\n", v); return -1; }
            a->nr = (AnrLevel)m;
        } else { fprintf(stderr, "unknown key %s\n", k); return -1; }
    }
    return 0;
}

static long file_len(FILE* f) __attribute__((unused));
static long file_len(FILE* f) {
    long n;
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    return n;
}

/* floats per seam record: E(2nf) Y(2nf) R2 CN Gres Gnr (4nf) far_hop out (2 hop) sc */
static size_t seam_rec_floats(int nf, int hop) { return (size_t)(6 * nf + 4 * nf + 2 * hop + PYC_SC); }

/* A replay seam header the drivers can trust: a power-of-two fft, nf = fft / 2 + 1, hop = fft / 2 (the chain's own synthesis
 * hop, which the output buffer is sized for), a known kind and at least one hop. */
static int seam_header_ok(const int32_t* h) __attribute__((unused));
static int seam_header_ok(const int32_t* h) {
    int kind = h[2], sr = h[3], fft = h[4], nf = h[5], hop = h[6], nh = h[7];
    return (kind == 0 || kind == 1) && sr >= 8000 && sr <= 192000 && fft >= 64 && fft <= 8192 && (fft & (fft - 1)) == 0 &&
           nf == fft / 2 + 1 && hop == fft / 2 && nh >= 1;
}

static void write_hdr(FILE* f, int kind, int sr, int fft, int nf, int hop, int nh, int fail_hop, int fail_rc) __attribute__((unused));
static void write_hdr(FILE* f, int kind, int sr, int fft, int nf, int hop, int nh, int fail_hop, int fail_rc) {
    int32_t h[10];
    h[0] = PYC_MAGIC; h[1] = 1; h[2] = kind; h[3] = sr; h[4] = fft; h[5] = nf; h[6] = hop;
    h[7] = nh; h[8] = fail_hop; h[9] = fail_rc;
    fseek(f, 0, SEEK_SET);
    fwrite(h, sizeof(h[0]), 10, f);
}

static void write_rec(FILE* f, int nf, int hop, const Complex* e, const Complex* y, const Complex* b,
                      const float* r2, const float* cn, const float* gres, const float* gnr,
                      const float* far_hop, const float* out, const float* sc) __attribute__((unused));
static void write_rec(FILE* f, int nf, int hop, const Complex* e, const Complex* y, const Complex* b,
                      const float* r2, const float* cn, const float* gres, const float* gnr,
                      const float* far_hop, const float* out, const float* sc) {
    static float* zero = NULL;
    static int zero_n = 0;
    int need = 2 * nf > hop ? 2 * nf : hop;
    if (zero_n < need) { free(zero); zero = (float*)calloc((size_t)need, sizeof(float)); zero_n = need; }
    fwrite(e ? (const void*)e : (const void*)zero, sizeof(Complex), (size_t)nf, f);
    fwrite(y ? (const void*)y : (const void*)zero, sizeof(Complex), (size_t)nf, f);
    fwrite(b ? (const void*)b : (const void*)zero, sizeof(Complex), (size_t)nf, f);
    fwrite(r2 ? r2 : zero, 4, (size_t)nf, f);
    fwrite(cn ? cn : zero, 4, (size_t)nf, f);
    fwrite(gres ? gres : zero, 4, (size_t)nf, f);
    fwrite(gnr ? gnr : zero, 4, (size_t)nf, f);
    fwrite(far_hop ? far_hop : zero, 4, (size_t)hop, f);
    fwrite(out, 4, (size_t)hop, f);
    fwrite(sc, 4, PYC_SC, f);
}

/* The chain behind the front end. The RES gain is the view's own (POST_RES_FRAME: the lane's, or the
 * array's gain stage), a standalone gain stage at `level` on the view's grid fed from the view (POST_RES_AECRES, the
 * 4-channel replay, built against the AEC_TESTING archive) or the unity vector. The synthesis borrows the window table of the Aec passed to
 * post_create() (aec_synthesis_window()); NULL: audiolib's default window (the 4-channel chain). */
enum { POST_RES_OFF = 0, POST_RES_FRAME, POST_RES_AECRES };

typedef struct Post {
    int nf, res_mode, use_r2, enable_cng;
#if defined(PYC_REPLAY_RES)
    AecTestingRes* res;
    AecConfig res_cfg;
#endif
    AnrDenoiser* nr;
    AcIstft* istft;
    float *unity, *extra;
    Complex* spec;
    uint32_t rng;
    const float *last_res, *last_nr;     /* the gains of the last hop (unity for an absent one) */
    uint32_t serial;                     /* the post_reset_serial value already handled */
    void* mem;
} Post;

static void post_destroy(Post* q) {
    if (q->nr) anr_destroy(q->nr);
    free(q->mem);
    memset(q, 0, sizeof(*q));
}

static int post_create(Post* q, int sr, int fft, int res_mode, AecLevel level, int enable_nr,
                       AnrLevel nr_level, int use_r2, int enable_cng, const Aec* window_src) {
    AnrConfig nc;
    AcStftConfig sc;
    size_t res_b = 0, nr_b = 0, syn_b, vec, cvec;
    unsigned char* cur;
    int k;
    memset(q, 0, sizeof(*q));
    q->nf = fft / 2 + 1;
    q->res_mode = res_mode;
    q->use_r2 = use_r2;
    q->enable_cng = enable_cng;
    if (res_mode == POST_RES_AECRES) {
#if defined(PYC_REPLAY_RES)
        AecConfig rc;
        aec_config_from_level(&rc, level, sr);
        rc.fft_size = fft;
        if ((res_b = aec_testing_res_mem_size(&rc)) == 0) return -1;
        q->res_cfg = rc;
#else
        (void)level;
        return -1;
#endif
    }
    if (enable_nr) {
        nc = anr_config_for_level_grid(sr, fft, nr_level);
        if ((nr_b = anr_get_mem_size(&nc)) == 0) return -1;
    }
    ac_stft_config_default(&sc, fft);
    if (window_src) {   /* the mono chain synthesises with the AEC's own window table (main.c) */
        int win_n = 0;
        if (aec_synthesis_window(window_src, &sc.window, &win_n) != 0 || win_n != fft) return -1;
    }
    if ((syn_b = ac_istft_get_mem_size(&sc)) == 0) return -1;
    vec = ALIGN16((size_t)q->nf * sizeof(float));
    cvec = ALIGN16((size_t)q->nf * sizeof(Complex));
    if (posix_memalign(&q->mem, 16, ALIGN16(res_b) + ALIGN16(nr_b) + ALIGN16(syn_b) + 2 * vec + cvec) != 0) {
        q->mem = NULL;
        return -1;
    }
    cur = (unsigned char*)q->mem;
#if defined(PYC_REPLAY_RES)
    if (res_b) { q->res = aec_testing_res_init(cur, res_b, &q->res_cfg); cur += ALIGN16(res_b); }
    if (res_b && !q->res) { post_destroy(q); return -1; }
#endif
    if (nr_b) { q->nr = anr_init(cur, nr_b, &nc); cur += ALIGN16(nr_b); }
    q->istft = ac_istft_init(cur, syn_b, &sc); cur += ALIGN16(syn_b);
    q->unity = (float*)cur; cur += vec;
    q->extra = (float*)cur; cur += vec;
    q->spec = (Complex*)cur;
    for (k = 0; k < q->nf; ++k) q->unity[k] = 1.0f;
    q->rng = PYC_CNG_RNG_SEED;
    if ((nr_b && !q->nr) || !q->istft) { post_destroy(q); return -1; }
    return 0;
}

/* One hop over a view into out: the explicit-array forms of aec.h (aec_apply_gain_spec / aec_apply_cng_spec) that the
 * sample programs' aec_apply_gain / aec_apply_cng wrap (a replay has no live Aec to own the generator). c->base_spec is
 * the rendering base the denoiser is estimated on (4-channel: array_sample_anr_gain) and the gain is applied to. `serial`
 * is the post-reset serial after this hop's AEC call: a new value clears the OLA tail and reseeds the comfort noise
 * before the first draw and the push.
 * 0, or -1 when a stage refuses. */
static int post_hop(Post* q, const AecResidualView* c, uint32_t serial, float* out) {
    const Complex* base = c->base_spec;
    const float* res = q->res_mode == POST_RES_FRAME ? c->res_gain : q->unity;
    const float* nr = q->unity;
#if defined(PYC_REPLAY_RES)
    if (q->res_mode == POST_RES_AECRES) res = aec_testing_res_process(q->res, c);
#endif
    if (!res || !c->error_spec || !base || c->n_freqs != q->nf) return -1;
    if (serial != q->serial) {
        q->serial = serial;
        ac_istft_reset(q->istft);
        q->rng = PYC_CNG_RNG_SEED;
    }
    if (q->nr) {
#if defined(PYC_WITH_4CH)
        if (!(nr = array_sample_anr_gain(q->nr, c, q->extra, q->use_r2))) return -1;
#else
        const float* residual_psd_audio = NULL;
        if (q->use_r2 && c->r2 && c->usable_linear) {
            sk_scale_f32(q->extra, c->r2, AEC_PSD_SCALE_INV_F, q->nf);
            residual_psd_audio = q->extra;
        }
        if (!(nr = anr_process_gain_with_residual_psd(q->nr, base, residual_psd_audio))) return -1;
#endif
    }
    aec_apply_gain_spec(q->spec, base, res, nr, q->nf);
    if (q->enable_cng && q->res_mode != POST_RES_OFF && c->comfort_noise)
        aec_apply_cng_spec(&q->rng, q->spec, c->comfort_noise, res, nr, q->nf);
    (void)ac_istft_push(q->istft, q->spec, out);
    q->last_res = res;
    q->last_nr = nr;
    return 0;
}

#if defined(PYC_WITH_4CH) && !defined(PYC_REPLAY_RES)
static int run_chain(const Args* a) {
    AecConfig cfg;
    AecMemReq req;
    Aec* aec = NULL;
    void* core_mem = NULL;
    Post chain;
    FILE *fm, *fr, *fo;
    long nm, nr_, n;
    int hop, nf, nh, h, fail_hop = -1, fail_rc = 0, done = 0;
    float *mics, *ref, *out;
    Complex* w;
    /* The array at the requested strength: the lanes ignore the level, the gain stage takes it. */
    aec_config_from_level(&cfg, a->aec, a->sr);
    cfg.fft_size = a->fft;
    cfg.n_mics = AEC_ARRAY_MICS;
    cfg.residual = a->enable_res ? AEC_RESIDUAL_FULL : AEC_RESIDUAL_ESTIMATE;
    if (aec_get_mem_requirements(&cfg, &req) == 0 && posix_memalign(&core_mem, 16, (size_t)req.bytes) == 0)
        aec = aec_init_ex(core_mem, (size_t)req.bytes, &cfg, &req);
    if (!aec) { fprintf(stderr, "aec_init_ex failed\n"); return 1; }
    hop = aec_hop_size(aec); nf = hop + 1;
    if (post_create(&chain, a->sr, 2 * hop, a->enable_res ? POST_RES_FRAME : POST_RES_OFF, a->aec, a->enable_nr, a->nr,
                    !a->legacy_amin, a->enable_cng, NULL) != 0) {
        fprintf(stderr, "post-stage init failed\n");
        return 1;
    }
    fm = fopen(a->mic, "rb"); fr = fopen(a->ref, "rb"); fo = fopen(a->out, "wb");
    if (!fm || !fr || !fo) { fprintf(stderr, "open failed\n"); return 1; }
    nm = file_len(fm) / (4 * 4); nr_ = file_len(fr) / 4;
    n = nm < nr_ ? nm : nr_; nh = (int)(n / hop);
    mics = (float*)malloc(sizeof(float) * 4 * (size_t)hop);
    ref = (float*)malloc(sizeof(float) * (size_t)hop);
    out = (float*)malloc(sizeof(float) * (size_t)hop);
    w = (Complex*)malloc(sizeof(Complex) * 4 * (size_t)nf);
    for (h = 0; h < 4 * nf; ++h) { w[h].r = 0.25f; w[h].i = 0.0f; }
    write_hdr(fo, 1, a->sr, a->fft, nf, hop, nh, -1, 0);
    for (h = 0; h < nh; ++h) {
        float sc[PYC_SC];
        AecArrayView front;
        AecResidualView f;
        AecLinearView lin;
        AecBeam beam;
        int rc;
        if (fread(mics, 4, 4 * (size_t)hop, fm) != 4 * (size_t)hop ||
            fread(ref, 4, (size_t)hop, fr) != (size_t)hop) { fprintf(stderr, "short read\n"); return 1; }
        memset(&front, 0, sizeof(front)); front.struct_size = sizeof(front);
        memset(&f, 0, sizeof(f)); f.struct_size = sizeof(f);
        memset(&lin, 0, sizeof(lin)); lin.struct_size = sizeof(lin);
        beam.weights = w;
        beam.beamformed_error = NULL;
        rc = aec_process_linear(aec, mics, ref);
        if (rc == AEC_OK) rc = aec_get_array(aec, &front);
        if (rc == AEC_OK) rc = aec_process_residual(aec, &beam);
        if (rc == AEC_OK) rc = aec_get_residual(aec, &f);
        if (rc == AEC_OK) rc = aec_get_linear(aec, &lin);
        if (rc == AEC_OK && post_hop(&chain, &f, f.post_reset_serial, out) != 0) rc = AEC_ERR_DSP;
        if (rc != 0) { fail_hop = h; fail_rc = rc; break; }
        memset(sc, 0, sizeof(sc));
        sc[0] = lin.far_power; sc[1] = f.max_dt; sc[2] = f.max_saturation; sc[3] = (float)f.all_converged;
        sc[4] = (float)f.clock_drift; sc[5] = (float)f.usable_linear;
        sc[6] = (float)front.delay.delay_samples; sc[7] = (float)front.delay.solid;
        sc[8] = (float)front.delay.changed;
        sc[9] = (float)aec_array_realign_warm_lane_count(aec);
        sc[10] = (float)aec_array_realign_soft_lane_count(aec);
        sc[11] = front.delay.confidence;
        sc[12] = (float)f.guard_lanes; sc[13] = (float)f.post_reset_serial;
        write_rec(fo, nf, hop, f.error_spec, f.near_spec, f.base_spec, f.r2, f.comfort_noise,
                  chain.last_res, chain.last_nr, f.far_hop, out, sc);
        ++done;
    }
    write_hdr(fo, 1, a->sr, a->fft, nf, hop, done, fail_hop, fail_rc);
    fclose(fo); fclose(fm); fclose(fr);
    post_destroy(&chain);
    aec_destroy(aec);
    free(core_mem);
    free(mics); free(ref); free(out); free(w);
    return 0;
}
#endif /* PYC_WITH_4CH && !PYC_REPLAY_RES */

#if defined(PYC_WITH_MONO)
/* The mono lane as mono/aec_nr_res/main.c derives it: the rate's grid and the AecConfig (the lane keeps its
 * own gain in the residual view, AEC_RESIDUAL_FULL; the chain applies it). */
static int derive_mono_lane(const Args* a, AecSignalGrid* grid, AecConfig* aec_cfg) {
    if (aec_resolve_signal_grid(a->sr, a->fft ? a->fft : aec_default_fft_size(a->sr), grid) != AEC_OK)
        return -1;
    aec_config_from_level(aec_cfg, a->aec, a->sr);
    aec_cfg->fft_size = grid->fft_size;
    aec_cfg->delay_mode = AEC_DELAY_MATCHED;
    aec_cfg->delay_num_filters = AEC_DELAY_NUM_FILTERS_MAX;
    aec_cfg->fixed_delay_samples = -1;
    aec_cfg->residual = AEC_RESIDUAL_FULL;
    return 0;
}

static int run_mono(const Args* a) {
    AecSignalGrid grid;
    AecConfig aec_cfg;
    Aec* aec;
    Post chain;
    void* aec_mem = NULL;
    size_t aec_bytes;
    FILE *fm, *fr, *fo;
    long n;
    int hop, nf, nh, h;
    float *mic, *ref, *out;
    if (derive_mono_lane(a, &grid, &aec_cfg) != 0) { fprintf(stderr, "lane grid refused\n"); return 1; }
    aec_bytes = aec_get_mem_size(&aec_cfg);
    if (aec_bytes == 0 || posix_memalign(&aec_mem, 16, aec_bytes) != 0) { fprintf(stderr, "alloc failed\n"); return 1; }
    aec = aec_init(aec_mem, aec_bytes, &aec_cfg);
    if (!aec || post_create(&chain, a->sr, grid.fft_size,
                            a->enable_res ? POST_RES_FRAME : POST_RES_OFF, a->aec,
                            a->enable_nr, a->nr, !a->legacy_amin,
                            aec_cfg.enable_cng && a->enable_cng, aec) != 0) {
        fprintf(stderr, "mono rig create failed\n");
        return 1;
    }
    hop = grid.hop_size; nf = grid.n_freqs;
    fm = fopen(a->mic, "rb"); fr = fopen(a->ref, "rb"); fo = fopen(a->out, "wb");
    if (!fm || !fr || !fo) { fprintf(stderr, "open failed\n"); return 1; }
    n = file_len(fm) / 4;
    { long nr_ = file_len(fr) / 4; if (nr_ < n) n = nr_; }
    nh = (int)(n / hop);
    mic = (float*)malloc(sizeof(float) * (size_t)hop);
    ref = (float*)malloc(sizeof(float) * (size_t)hop);
    out = (float*)malloc(sizeof(float) * (size_t)hop);
    write_hdr(fo, 0, a->sr, a->fft, nf, hop, nh, -1, 0);
    for (h = 0; h < nh; ++h) {
        AecResidualView c;
        AecLinearView lin;
        AnrDenoiser* nr = chain.nr;
        const float* g;
        float sc[PYC_SC];
        if (fread(mic, 4, (size_t)hop, fm) != (size_t)hop ||
            fread(ref, 4, (size_t)hop, fr) != (size_t)hop) { fprintf(stderr, "short read\n"); return 1; }
        if (h == a->reset_at) aec_reset(aec);       /* the application restarts the AEC: the serial moves */
        memset(&c, 0, sizeof(c));
        c.struct_size = sizeof(c);
        memset(&lin, 0, sizeof(lin));
        lin.struct_size = sizeof(lin);
        if (aec_process_linear(aec, mic, ref) != AEC_OK || aec_process_residual(aec, NULL) != AEC_OK ||
            aec_get_residual(aec, &c) != AEC_OK || aec_get_linear(aec, &lin) != AEC_OK ||
            post_hop(&chain, &c, aec_post_reset_serial(aec), out) != 0) {
            fprintf(stderr, "process failed at hop %d\n", h);
            return 1;
        }
        g = nr ? chain.last_nr : NULL;      /* the denoiser gain of this hop */
        memset(sc, 0, sizeof(sc));
        sc[0] = lin.far_power; sc[1] = lin.dt_indicator; sc[2] = lin.saturation_level;
        sc[3] = (float)lin.filter_converged; sc[5] = (float)c.usable_linear;
        sc[6] = (float)lin.delay_samples; sc[11] = lin.delay_confidence;
        sc[12] = (float)c.guard_active; sc[13] = (float)aec_post_reset_serial(aec);
        write_rec(fo, nf, hop, c.error_spec, c.near_spec, c.base_spec, c.r2, c.comfort_noise, c.res_gain, g,
                  lin.aligned_far_hop, out, sc);
    }
    write_hdr(fo, 0, a->sr, a->fft, nf, hop, nh, -1, 0);
    fclose(fo); fclose(fm); fclose(fr);
    post_destroy(&chain);
    aec_destroy(aec);
    free(aec_mem);
    free(mic); free(ref); free(out);
    return 0;
}
#endif /* PYC_WITH_MONO */

#if !defined(PYC_WITH_4CH)
/* An Aec kept only for its synthesis window table: a replay has no live AEC to borrow it from. */
typedef struct WindowSrc {
    Aec* aec;
    void* mem;
} WindowSrc;

static int window_src_open(WindowSrc* w, int sr, int fft, AecLevel level) {
    AecConfig cfg;
    size_t bytes;
    memset(w, 0, sizeof(*w));
    aec_config_from_level(&cfg, level, sr);
    cfg.fft_size = fft;
    bytes = aec_get_mem_size(&cfg);
    if (bytes == 0 || posix_memalign(&w->mem, 16, bytes) != 0) { w->mem = NULL; return -1; }
    w->aec = aec_init(w->mem, bytes, &cfg);
    return w->aec ? 0 : -1;
}

static void window_src_close(WindowSrc* w) {
    if (w->aec) aec_destroy(w->aec);
    free(w->mem);
    memset(w, 0, sizeof(*w));
}
#endif

static int run_replay(const Args* a) {
    FILE *fs = fopen(a->seam, "rb"), *fo = NULL, *fu = NULL;
    int32_t h[10];
    int kind, sr, fft, nf, hop, nh, t;
    size_t rec;
    float* buf;
    unsigned char* usable = NULL;
    Post chain;
    const Aec* window_src = NULL;       /* 4-channel: audiolib's default window */
#if !defined(PYC_WITH_4CH)
    WindowSrc ws;
#endif
    if (!fs) { fprintf(stderr, "cannot open seam\n"); return 1; }
    if (fread(h, 4, 10, fs) != 10 || h[0] != PYC_MAGIC || !seam_header_ok(h)) { fprintf(stderr, "bad seam header\n"); return 1; }
    kind = h[2]; sr = h[3]; fft = h[4]; nf = h[5]; hop = h[6]; nh = h[7];
    rec = seam_rec_floats(nf, hop);
    {   /* the records the header announces must all be in the file before anything is allocated for them */
        long here = ftell(fs), end;
        if (fseek(fs, 0, SEEK_END) != 0 || (end = ftell(fs)) < here || fseek(fs, here, SEEK_SET) != 0 ||
            (size_t)(end - here) / (sizeof(float) * rec) < (size_t)nh) { fprintf(stderr, "short seam\n"); return 1; }
    }
    buf = (float*)malloc(rec * sizeof(float) * (size_t)nh);
    if (!buf) { fprintf(stderr, "out of memory\n"); return 1; }
    if (fread(buf, sizeof(float) * rec, (size_t)nh, fs) != (size_t)nh) { fprintf(stderr, "short seam\n"); return 1; }
    fclose(fs);
    if (a->usable) {
        usable = (unsigned char*)malloc((size_t)nh);
        fu = fopen(a->usable, "rb");
        if (!fu || fread(usable, 1, (size_t)nh, fu) != (size_t)nh) { fprintf(stderr, "bad usable file\n"); return 1; }
        fclose(fu);
    }
#if !defined(PYC_WITH_4CH)
    if (window_src_open(&ws, sr, fft, a->aec) != 0) { fprintf(stderr, "window source init failed\n"); return 1; }
    window_src = ws.aec;
#endif
    /* 4-channel: a standalone gain stage at the requested strength; mono: the recorded lane gain. */
    if (post_create(&chain, sr, fft,
                    !a->enable_res ? POST_RES_OFF : kind == 1 ? POST_RES_AECRES : POST_RES_FRAME,
                    a->aec, a->enable_nr, a->nr, !a->legacy_amin, a->enable_cng, window_src) != 0) {
        fprintf(stderr, "stage init failed\n");
        return 1;
    }
    fo = fopen(a->out, "wb");
    if (!fo) { fprintf(stderr, "cannot open out\n"); return 1; }
    { int32_t oh[8] = { PYC_MAGIC, 1, kind, sr, fft, nf, hop, nh }; fwrite(oh, 4, 8, fo); }
    for (t = 0; t < nh; ++t) {
        const float* r = buf + rec * (size_t)t;
        const float* sc = r + 10 * nf + 2 * hop;
        float* out = (float*)malloc(sizeof(float) * (size_t)hop);
        int rc;
        AecResidualView c;
        memset(&c, 0, sizeof(c));
        c.struct_size = sizeof(c);
        c.n_freqs = nf; c.hop_size = hop;
        c.error_spec = (const Complex*)(r);
        c.near_spec = (const Complex*)(r + 2 * nf);
        c.base_spec = (const Complex*)(r + 4 * nf);              /* recorded rendering base */
        c.r2 = r + 6 * nf;
        c.comfort_noise = r + 7 * nf;
        c.res_gain = r + 8 * nf;                                 /* recorded G_res (mono; the 4-channel stage recomputes it) */
        c.far_hop = r + 10 * nf;
        c.all_converged = (int)sc[3]; c.max_dt = sc[1]; c.max_saturation = sc[2]; c.clock_drift = (int)sc[4];
        c.usable_linear = usable ? usable[t] : (int)sc[5];
        c.guard_lanes = (unsigned)sc[12];
        rc = post_hop(&chain, &c, (uint32_t)sc[13], out);
        if (rc != 0) { fprintf(stderr, "stage refused hop %d\n", t); return 1; }
        fwrite(chain.last_res, 4, (size_t)nf, fo);
        fwrite(chain.last_nr, 4, (size_t)nf, fo);
        fwrite(out, 4, (size_t)hop, fo);
        free(out);
    }
    fclose(fo);
    post_destroy(&chain);
#if !defined(PYC_WITH_4CH)
    window_src_close(&ws);
#endif
    free(buf); free(usable);
    return 0;
}

#if defined(PYC_WITH_MONO)
static int run_probe(const Args* a) {
    AecSignalGrid grid;
    AecConfig aec_cfg;
    if (derive_mono_lane(a, &grid, &aec_cfg) != 0 || aec_get_mem_size(&aec_cfg) == 0) {
        printf("create failed\n");
        return 1;
    }
    printf("hop=%d n_freqs=%d\n", grid.hop_size, grid.n_freqs);
    return 0;
}
#endif

static int run_cng(const Args* a) {
    FILE* f = fopen(a->seam, "rb");
    int32_t n, use_g_nr;
    uint32_t rng;
    float floor_;
    Complex* spec;
    float *cn, *gr, *gn;
    if (!f) { fprintf(stderr, "cannot open input\n"); return 1; }
    if (fread(&n, 4, 1, f) != 1 || fread(&rng, 4, 1, f) != 1 || fread(&floor_, 4, 1, f) != 1 ||
        fread(&use_g_nr, 4, 1, f) != 1 || n < 1) { fprintf(stderr, "bad header\n"); return 1; }
    spec = (Complex*)malloc(sizeof(Complex) * (size_t)n);
    cn = (float*)malloc(4 * (size_t)n); gr = (float*)malloc(4 * (size_t)n); gn = (float*)malloc(4 * (size_t)n);
    if (fread(spec, sizeof(Complex), (size_t)n, f) != (size_t)n || fread(cn, 4, (size_t)n, f) != (size_t)n ||
        fread(gr, 4, (size_t)n, f) != (size_t)n || fread(gn, 4, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short input\n"); return 1; }
    fclose(f);
    aec_cng_fill(&rng, spec, cn, gr, use_g_nr ? gn : NULL, floor_, n);
    f = fopen(a->out, "wb");
    if (!f) return 1;
    fwrite(&rng, 4, 1, f);
    fwrite(spec, sizeof(Complex), (size_t)n, f);
    fclose(f);
    free(spec); free(cn); free(gr); free(gn);
    return 0;
}

int main(int argc, char** argv) {
    Args a;
    if (argc < 2) { fprintf(stderr, "usage: pyc_drv <mode> key=value ...\n"); return 2; }
    if (PYC_CNG_NR_GAIN_FLOOR != AEC_CNG_NR_GAIN_FLOOR) { fprintf(stderr, "comfort-noise floor moved\n"); return 2; }
    if (parse_args(argc, argv, &a) != 0) return 2;
#if defined(PYC_WITH_4CH) && !defined(PYC_REPLAY_RES)
    if (!strcmp(argv[1], "chain")) return run_chain(&a);
#endif
#if defined(PYC_WITH_MONO)
    if (!strcmp(argv[1], "mono")) return run_mono(&a);
    if (!strcmp(argv[1], "probe")) return run_probe(&a);
#endif
#if defined(PYC_WITH_4CH) && !defined(PYC_REPLAY_RES)
    if (!strcmp(argv[1], "replay")) { fprintf(stderr, "a 4-channel replay needs pyc_drv_4ch_replay\n"); return 2; }
#else
    if (!strcmp(argv[1], "replay")) return run_replay(&a);
#endif
    if (!strcmp(argv[1], "cng")) return run_cng(&a);
    fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
