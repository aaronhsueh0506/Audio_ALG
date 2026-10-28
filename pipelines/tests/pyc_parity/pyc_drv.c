/* pyc_drv.c -- C side of pyc_parity_pipelines.sh.
 *
 * One source, two executables: the 4-channel one links the 4-channel archive, the mono one only
 * the libraries, so each build carries only its own front end:
 *
 *   pyc_drv_4ch  -DPYC_WITH_4CH    modes: chain   replay
 *   pyc_drv_mono -DPYC_WITH_MONO   modes: mono    replay   probe
 *
 * Both run the chain the example programs write out behind the front end (post_hop() below:
 * RES gain -> NR gain -> min -> apply -> comfort noise -> synthesis): the 4-channel build over
 * mcab_fuse()'s McabFrame through 4ch_aec_bf_nr_res/mcab_sample_post.h, the mono build over the
 * lane's AecResContext as mono_aec_nr_res/main.c writes it out. A deviation planted in a stage is
 * seen by the comparison.
 *
 * Command line: <mode> key=value ...
 *
 *   chain  mic=F ref=F sr=N fft=N out=F [aec=balanced|mild|aggressive|no_floor] [nr=mild|balanced|aggressive]
 *          [legacy_amin=0|1] [enable_nr=0|1] [enable_res=0|1] [enable_cng=0|1]
 *     The 4-channel chain as 4ch_aec_bf_nr_res/mcab_external_static.c composes it (front end -> fuse
 *     with equal 0.25 weights -> RES (an AecRes over the fused frame, as the unit builds it) -> NR
 *     stage -> min -> apply -> comfort noise -> synthesis), every hop dumped in the seam format below. Stops at the first non-zero
 *     status of the core or the chain, recording it in the header; the Python side treats that as
 *     a result.
 *     aec=no_floor is accepted like the other levels.
 *   mono   mic=F ref=F sr=N fft=N out=F [aec=..] [nr=..] [legacy_amin=0|1] [enable_cng=0|1] [reset_at=H]
 *     Aec + aec_get_res_context + the chain as mono_aec_nr_res/main.c composes them, with the AEC
 *     seam read back after each hop. The chain applies the lane's own RES gain (frame gain). The chain
 *     renders the AEC's rendering base (aec_get_output_base) and follows aec_post_reset_serial() (OLA clear,
 *     comfort-noise reseed); reset_at=H makes the driver call aec_reset() before hop H.
 *   replay seam=F out=F [aec=..] [nr=..] [legacy_amin=0|1] [enable_cng=0|1] [usable=F]
 *     Open-loop replay of the chain over a recorded seam: RES stage (4-channel only; mono reuses
 *     the recorded G_res), NR stage, min, apply to the recorded B, comfort noise, synthesis. `usable=F` is one
 *     byte per hop that replaces the recorded usable_linear flag (the Python verdict of the same hop).
 *     Output: header (8 x int32) then per hop float32 G_res(nf) G_nr(nf) out(hop).
 *   probe  sr=N fft=N          (mono build) derive the lane for the grid and print "hop n_freqs", or fail
 *   cng    seam=F out=F
 *     One aec_cng_fill call. Input (seam=): int32 n, uint32 rng, float32 floor, int32 use_g_nr, then float32
 *     spec(2n) cn(n) g_res(n) g_nr(n). Output: uint32 rng after the call, then float32 spec(2n).
 *
 * Seam file (all little endian):
 *   int32 hdr[10] = { 0x53435950 'PYCS', 1, kind (0 mono, 1 4ch), sr, fft, n_freqs, hop, n_hops,
 *                     fail_hop (-1 = none), fail_rc }
 *   per hop float32: E(2nf) Y(2nf) B(2nf) R2(nf) CN(nf) Gres(nf) Gnr(nf) far_hop(hop) out(hop) sc[16]
 *   sc: 0 far_power  1 max_dt  2 max_sat  3 all_converged  4 clock_drift  5 usable_linear
 *       6 delay_samples  7 delay_solid  8 delay_changed  9 realign_warm_total  10 realign_soft_total
 *       11 delay_confidence  12 guard (mono: guard_active; 4ch: McabFrame.guard_lanes)
 *       13 post_reset_serial (mono: aec_post_reset_serial(); 4ch: McabFrame.post_reset_serial)  14..15 unused (0)
 * Complex spectra are interleaved (re, im). B is the rendering base the chain applied its gain to
 * (mono: aec_get_output_base(); 4ch: McabFrame.b, which a flag cannot rebuild where only some lanes'
 * guards fired). Gres/Gnr are the gains the chain applied (unity when the source is disabled). The
 * seam is the data the stages behind the AEC read; scales as in McabFrame (mcab.h): E, Y, B in audio
 * scale, R2 and CN in int16^2 power scale, gains amplitudes. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "aec_res.h"
#include "anr.h"
#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"

#if defined(PYC_WITH_4CH)
#include "mcab.h"
#include "mcab_sample_post.h"
#define PYC_CNG_NR_GAIN_FLOOR MCAB_SAMPLE_CNG_NR_GAIN_FLOOR
#define PYC_CNG_RNG_SEED MCAB_SAMPLE_CNG_RNG_SEED
#else
/* The constants of mono_aec_nr_res/main.c. */
#define PYC_CNG_NR_GAIN_FLOOR 0.31622777f
#define PYC_CNG_RNG_SEED AEC_CNG_SEED
#define PYC_PSD_SCALE_INV (1.0f / (32768.0f * 32768.0f))
#endif

#define PYC_MAGIC 0x53435950
#define PYC_SC 16

typedef struct Args {
    const char *mic, *ref, *out, *seam, *usable;
    int sr, fft, legacy_amin, enable_nr, enable_res, enable_cng;
    int reset_at;                 /* mono: hop on which the application calls aec_reset() (-1: never) */
    AecLevel aec;
    AnrLevel nr;
} Args;

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
            a->aec = !strcmp(v, "mild") ? AEC_LEVEL_MILD
                   : !strcmp(v, "aggressive") ? AEC_LEVEL_AGGRESSIVE
                   : !strcmp(v, "no_floor") ? AEC_LEVEL_NO_FLOOR : AEC_LEVEL_BALANCED;
            if (strcmp(v, "mild") && strcmp(v, "aggressive") && strcmp(v, "balanced")
                && strcmp(v, "no_floor")) {
                fprintf(stderr, "unknown aec level %s\n", v); return -1;
            }
        } else if (!strcmp(k, "nr")) {
            int m = anr_nr_level_from_name(v);
            if (m < 0) { fprintf(stderr, "unknown nr level %s\n", v); return -1; }
            a->nr = (AnrLevel)m;
        } else { fprintf(stderr, "unknown key %s\n", k); return -1; }
    }
    return 0;
}

static long file_len(FILE* f) {
    long n;
    fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
    return n;
}

/* floats per seam record: E(2nf) Y(2nf) R2 CN Gres Gnr (4nf) far_hop out (2 hop) sc */
static size_t seam_rec_floats(int nf, int hop) { return (size_t)(6 * nf + 4 * nf + 2 * hop + PYC_SC); }

static void write_hdr(FILE* f, int kind, int sr, int fft, int nf, int hop, int nh, int fail_hop, int fail_rc) {
    int32_t h[10];
    h[0] = PYC_MAGIC; h[1] = 1; h[2] = kind; h[3] = sr; h[4] = fft; h[5] = nf; h[6] = hop;
    h[7] = nh; h[8] = fail_hop; h[9] = fail_rc;
    fseek(f, 0, SEEK_SET);
    fwrite(h, sizeof(h[0]), 10, f);
}

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

/* The chain behind the front end. The RES gain comes from the frame (POST_RES_FRAME), from a
 * separate AecRes at `level` on the frame's grid (POST_RES_AECRES) or is the unity vector. The synthesis
 * borrows the window table of the Aec passed to post_create() (aec_synthesis_window()); NULL: audiolib's
 * default window (the 4-channel chain). */
enum { POST_RES_OFF = 0, POST_RES_FRAME, POST_RES_AECRES };

typedef struct Post {
    int nf, res_mode, use_r2, enable_cng;
    AecRes* res;
    AnrDenoiser* nr;
    AcIstft* istft;
    float *unity, *total, *extra;
    Complex* spec;
    uint32_t rng;
    const float *last_res, *last_nr;     /* the gains of the last hop (unity for an absent one) */
#if !defined(PYC_WITH_4CH)
    uint32_t mono_serial;                /* the aec_post_reset_serial() value already handled */
#endif
    void* mem;
} Post;

static void post_destroy(Post* q) {
    if (q->nr) anr_destroy(q->nr);
    free(q->mem);
    memset(q, 0, sizeof(*q));
}

static int post_create(Post* q, int sr, int fft, int res_mode, AecLevel level, int enable_nr,
                       AnrLevel nr_level, int use_r2, int enable_cng, const Aec* window_src) {
    AecConfig rc;
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
        aec_config_from_level(&rc, level, sr);
        rc.fft_size = fft;
        if ((res_b = aec_res_get_mem_size(&rc)) == 0) return -1;
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
    if (posix_memalign(&q->mem, 16, ALIGN16(res_b) + ALIGN16(nr_b) + ALIGN16(syn_b) + 3 * vec + cvec) != 0) {
        q->mem = NULL;
        return -1;
    }
    cur = (unsigned char*)q->mem;
    if (res_b) { q->res = aec_res_init(cur, res_b, &rc); cur += ALIGN16(res_b); }
    if (nr_b) { q->nr = anr_init(cur, nr_b, &nc); cur += ALIGN16(nr_b); }
    q->istft = ac_istft_init(cur, syn_b, &sc); cur += ALIGN16(syn_b);
    q->unity = (float*)cur; cur += vec;
    q->total = (float*)cur; cur += vec;
    q->extra = (float*)cur; cur += vec;
    q->spec = (Complex*)cur;
    for (k = 0; k < q->nf; ++k) q->unity[k] = 1.0f;
    q->rng = PYC_CNG_RNG_SEED;
    if ((res_b && !q->res) || (nr_b && !q->nr) || !q->istft) { post_destroy(q); return -1; }
    return 0;
}

/* min, apply, comfort noise, synthesis of one hop into out. */
static int post_finish(Post* q, const Complex* e, const float* cn, const float* res,
                       const float* nr, float* out) {
    sk_min_f32(q->total, nr, res, q->nf);
    sk_capply_gain_f32(q->spec, e, q->total, q->nf);
    if (q->enable_cng && q->res_mode != POST_RES_OFF && cn)
        aec_cng_fill(&q->rng, q->spec, cn, res, nr, PYC_CNG_NR_GAIN_FLOOR, q->nf);
    (void)ac_istft_push(q->istft, q->spec, out);
    q->last_res = res;
    q->last_nr = nr;
    return 0;
}

#if defined(PYC_WITH_4CH)
/* One hop over a fused frame into out; 0, or -1 when a stage refuses. */
static int post_hop(Post* q, const McabFrame* f, float* out) {
    const float* res = q->unity;
    const float* nr = q->unity;
    if (q->res_mode == POST_RES_FRAME) {
        res = f->g_res_ctx;
    } else if (q->res_mode == POST_RES_AECRES) {
        AecResInput in;
        mcab_sample_res_input(f, &in);
        res = aec_res_process(q->res, &in);
    }
    if (!res || !f->e || f->n_freqs != q->nf) return -1;
    if (q->nr && !(nr = mcab_sample_anr_gain(q->nr, f, q->extra, q->use_r2))) return -1;
    /* Estimated on E, rendered on the rendering base B (mcab.h). */
    return post_finish(q, f->b, f->cn, res, nr, out);
}
#else
/* One hop over a lane context into out; `res_in` feeds POST_RES_AECRES. `base` is the rendering base
 * (aec_get_output_base) the gain is estimated on and applied to; `serial` is aec_post_reset_serial()
 * after this hop's AEC call: a new value clears the OLA tail and reseeds the comfort noise before the
 * first draw and the push. 0, or -1 when a stage refuses. */
static int post_hop(Post* q, const AecResContext* c, const AecResInput* res_in, const Complex* base,
                    uint32_t serial, float* out) {
    const float* res = q->res_mode == POST_RES_FRAME ? c->res_gain
                     : q->res_mode == POST_RES_AECRES ? aec_res_process(q->res, res_in) : q->unity;
    const float* nr = q->unity;
    if (!res || !c->error_spec || !base || c->n_freqs != q->nf) return -1;
    if (serial != q->mono_serial) {
        q->mono_serial = serial;
        ac_istft_reset(q->istft);
        q->rng = PYC_CNG_RNG_SEED;
    }
    if (q->nr) {
        const float* extra = NULL;
        if (q->use_r2 && c->r2 && c->usable_linear) {
            sk_scale_f32(q->extra, c->r2, PYC_PSD_SCALE_INV, q->nf);
            extra = q->extra;
        }
        if (anr_process_gain(q->nr, base, extra, NULL) != 0) return -1;
        if (!(nr = anr_get_gain(q->nr, NULL))) return -1;
    }
    return post_finish(q, base, c->comfort_noise, res, nr, out);
}
#endif

#if defined(PYC_WITH_4CH)
static int run_chain(const Args* a) {
    McabConfig cfg = mcab_default_config(a->sr);
    Mcab* core;
    /* The application side of mcab.h: four lanes, the shared estimator, mcab's pool. */
    AecConfig lane_cfg;
    McabDelayEstimatorConfig dc;
    McabMemReq req;
    McabLibs libs;
    DelayAec delay;
    void *lane_mem[4] = {NULL, NULL, NULL, NULL}, *delay_mem = NULL, *core_mem = NULL;
    size_t lane_bytes = 0;
    int ch;
    Post chain;
    FILE *fm, *fr, *fo;
    long nm, nr_, n;
    int hop, nf, nh, h, fail_hop = -1, fail_rc = 0, done = 0;
    float *mics, *ref, *out;
    Complex* w;
    cfg.fft_size = a->fft;
    memset(&libs, 0, sizeof(libs));
    core = NULL;
    if (mcab_lane_config(&cfg, &lane_cfg) == 0 && mcab_delay_estimator_config(&cfg, &dc) == 0 &&
        mcab_get_mem_requirements(&cfg, &req) == 0 &&
        (lane_bytes = aec_get_mem_size(&lane_cfg)) != 0) {
        for (ch = 0; ch < 4; ++ch)
            if (posix_memalign(&lane_mem[ch], 16, lane_bytes) == 0)
                libs.lanes[ch] = aec_init(lane_mem[ch], lane_bytes, &lane_cfg);
        if (dc.required) {
            size_t db = delay_aec_get_mem_size(dc.sample_rate, dc.hop_size, dc.num_filters);
            if (db && posix_memalign(&delay_mem, 16, db) == 0 &&
                delay_aec_init(&delay, delay_mem, db, dc.sample_rate, dc.hop_size,
                               dc.num_filters) == 0)
                libs.delay = &delay;
        }
        if (posix_memalign(&core_mem, 16, (size_t)req.bytes) == 0)
            core = mcab_init_ex(core_mem, (size_t)req.bytes, &cfg, &libs, &req);
    }
    if (!core) { fprintf(stderr, "mcab_init_ex failed\n"); return 1; }
    hop = mcab_hop_size(core); nf = mcab_n_freqs(core);
    if (post_create(&chain, a->sr, mcab_fft_size(core),
                    a->enable_res ? POST_RES_AECRES : POST_RES_OFF, a->aec, a->enable_nr, a->nr,
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
        McabFront front;
        McabFrame f;
        int rc;
        if (fread(mics, 4, 4 * (size_t)hop, fm) != 4 * (size_t)hop ||
            fread(ref, 4, (size_t)hop, fr) != (size_t)hop) { fprintf(stderr, "short read\n"); return 1; }
        rc = mcab_process_front(core, mics, ref, &front);
        if (rc == MCAB_OK) rc = mcab_fuse(core, w, NULL, &f);
        if (rc == MCAB_OK && post_hop(&chain, &f, out) != 0) rc = MCAB_DSP_ERROR;
        if (rc != 0) { fail_hop = h; fail_rc = rc; break; }
        memset(sc, 0, sizeof(sc));
        sc[0] = f.far_power; sc[1] = f.max_dt; sc[2] = f.max_sat; sc[3] = (float)f.all_converged;
        sc[4] = (float)f.clock_drift; sc[5] = (float)f.usable_linear;
        sc[6] = (float)front.delay.delay_samples; sc[7] = (float)front.delay.solid;
        sc[8] = (float)front.delay.changed;
        sc[9] = (float)mcab_realign_warm_lane_count(core);
        sc[10] = (float)mcab_realign_soft_lane_count(core);
        sc[11] = front.delay.confidence;
        sc[12] = (float)f.guard_lanes; sc[13] = (float)f.post_reset_serial;
        write_rec(fo, nf, hop, f.e, f.y, f.b, f.r2, f.cn, chain.last_res, chain.last_nr, f.far_hop,
                  out, sc);
        ++done;
    }
    write_hdr(fo, 1, a->sr, a->fft, nf, hop, done, fail_hop, fail_rc);
    fclose(fo); fclose(fm); fclose(fr);
    post_destroy(&chain);
    mcab_destroy(core);
    for (ch = 0; ch < 4; ++ch) {
        if (libs.lanes[ch]) aec_destroy(libs.lanes[ch]);
        free(lane_mem[ch]);
    }
    free(delay_mem);
    free(core_mem);
    free(mics); free(ref); free(out); free(w);
    return 0;
}
#endif /* PYC_WITH_4CH */

#if defined(PYC_WITH_MONO)
/* The mono lane as mono_aec_nr_res/main.c derives it: the rate's grid and the linear-stage AecConfig
 * (the lane keeps its own gain in the context; the chain applies it). */
static int derive_mono_lane(const Args* a, AecSignalGrid* grid, AecConfig* aec_cfg) {
    if (!aec_resolve_signal_grid(a->sr, a->fft ? a->fft : aec_default_fft_size(a->sr), grid))
        return -1;
    aec_config_from_level(aec_cfg, a->aec, a->sr);
    aec_cfg->fft_size = grid->fft_size;
    aec_cfg->delay_mode = AEC_DELAY_MATCHED;
    aec_cfg->delay_num_filters = DA_NUM_FILTERS;
    aec_cfg->fixed_delay_samples = -1;
    aec_cfg->enable_delay_est = 1;
    aec_cfg->enable_res = 0;
    aec_cfg->return_res_context = 1;
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
                            a->enable_res ? POST_RES_FRAME : POST_RES_OFF, AEC_LEVEL_BALANCED,
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
        AecResContext c;
        AecLinearContext lin;
        AecOutputBase ob;
        AnrDenoiser* nr = chain.nr;
        const float* g;
        float sc[PYC_SC];
        if (fread(mic, 4, (size_t)hop, fm) != (size_t)hop ||
            fread(ref, 4, (size_t)hop, fr) != (size_t)hop) { fprintf(stderr, "short read\n"); return 1; }
        if (h == a->reset_at) aec_reset(aec);       /* the application restarts the AEC: the serial moves */
        aec_process_context(aec, mic, ref);
        aec_get_res_context(aec, &c);
        aec_get_linear_context(aec, &lin);
        memset(&ob, 0, sizeof(ob));
        ob.struct_size = sizeof(ob);
        if (!c.error_spec || aec_get_output_base(aec, &ob) != 0 ||
            post_hop(&chain, &c, NULL, ob.base_spec, aec_post_reset_serial(aec), out) != 0) {
            fprintf(stderr, "process failed at hop %d\n", h);
            return 1;
        }
        g = nr ? anr_get_gain(nr, NULL) : NULL;
        memset(sc, 0, sizeof(sc));
        sc[0] = c.far_power; sc[1] = c.dt_indicator; sc[2] = c.saturation_level;
        sc[3] = (float)c.filter_converged; sc[5] = (float)c.usable_linear;
        sc[6] = (float)lin.delay_samples; sc[11] = lin.delay_confidence;
        sc[12] = (float)ob.guard_active; sc[13] = (float)aec_post_reset_serial(aec);
        write_rec(fo, nf, hop, c.error_spec, c.near_spec, ob.base_spec, c.r2, c.comfort_noise, c.res_gain, g,
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
    if (fread(h, 4, 10, fs) != 10 || h[0] != PYC_MAGIC) { fprintf(stderr, "bad seam header\n"); return 1; }
    kind = h[2]; sr = h[3]; fft = h[4]; nf = h[5]; hop = h[6]; nh = h[7];
    rec = seam_rec_floats(nf, hop);
    buf = (float*)malloc(rec * sizeof(float) * (size_t)nh);
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
    /* 4-channel: a separate AecRes at the requested strength; mono: the recorded lane gain. */
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
#if defined(PYC_WITH_4CH)
        McabFrame f;
        memset(&f, 0, sizeof(f));
        f.n_freqs = nf; f.hop = hop;
        f.e = (const Complex*)(r);
        f.y = (const Complex*)(r + 2 * nf);
        f.b = (const Complex*)(r + 4 * nf);                      /* recorded rendering base */
        f.r2 = r + 6 * nf;
        f.cn = r + 7 * nf;
        f.g_res_ctx = r + 8 * nf;                                /* recorded G_res */
        f.far_hop = r + 10 * nf;
        f.far_power = sc[0]; f.max_dt = sc[1]; f.max_sat = sc[2];
        f.all_converged = (int)sc[3]; f.clock_drift = (int)sc[4];
        f.usable_linear = usable ? usable[t] : (int)sc[5];
        f.guard_lanes = (int)sc[12]; f.post_reset_serial = (uint32_t)sc[13];
        rc = post_hop(&chain, &f, out);
#else
        AecResContext c;
        AecResInput in;
        memset(&c, 0, sizeof(c));
        c.n_freqs = nf; c.hop_size = hop;
        c.error_spec = (const Complex*)(r);
        c.near_spec = (const Complex*)(r + 2 * nf);
        c.r2 = r + 6 * nf;
        c.comfort_noise = r + 7 * nf;
        c.res_gain = r + 8 * nf;                                 /* recorded G_res */
        c.far_power = sc[0]; c.dt_indicator = sc[1]; c.saturation_level = sc[2];
        c.filter_converged = (int)sc[3];
        c.usable_linear = usable ? usable[t] : (int)sc[5];
        in.error_spec = c.error_spec;
        in.near_spec = c.near_spec;
        in.r2 = c.r2;
        in.comfort_noise = c.comfort_noise;
        in.render_hop = r + 10 * nf;
        in.all_converged = c.filter_converged;
        in.max_dt = c.dt_indicator;
        in.max_saturation = c.saturation_level;
        in.clock_drift = (int)sc[4];
        rc = post_hop(&chain, &c, &in, (const Complex*)(r + 4 * nf), (uint32_t)sc[13], out);
#endif
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
    if (parse_args(argc, argv, &a) != 0) return 2;
#if defined(PYC_WITH_4CH)
    if (!strcmp(argv[1], "chain")) return run_chain(&a);
#endif
#if defined(PYC_WITH_MONO)
    if (!strcmp(argv[1], "mono")) return run_mono(&a);
    if (!strcmp(argv[1], "probe")) return run_probe(&a);
#endif
    if (!strcmp(argv[1], "replay")) return run_replay(&a);
    if (!strcmp(argv[1], "cng")) return run_cng(&a);
    fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}
