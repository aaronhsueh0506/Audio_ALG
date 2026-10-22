/* pyc_drv.c -- C side of pyc_parity_pipelines.sh.
 *
 * One source, two executables (the 4-channel and the mono archive both carry the
 * pipelines/common objects, so they are never linked together):
 *
 *   pyc_drv_4ch  -DPYC_WITH_4CH    modes: chain   replay
 *   pyc_drv_mono -DPYC_WITH_MONO   modes: mono    replay
 *
 * Command line: <mode> key=value ...
 *
 *   chain  mic=F ref=F sr=N fft=N out=F [aec=balanced|mild|aggressive|no_floor] [nr=mild|balanced|aggressive]
 *          [legacy_amin=0|1] [enable_nr=0|1] [enable_res=0|1] [enable_cng=0|1]
 *     The conventional 4-channel chain exactly as tests/conv_chain_4ch.h writes it out (front end ->
 *     fuse with equal 0.25 weights -> RES stage -> NR stage -> min -> apply -> comfort noise ->
 *     synthesis), every hop dumped in the seam format below. Stops at the first non-zero status of
 *     the core, recording it in the header; the Python side treats that as a result.
 *     aec=no_floor is a construction-time preset like any other: se_derive()'s preset_valid()
 *     (pipelines/common/se_config.c) whitelists mild/balanced/aggressive/no_floor, matching the
 *     Python 4ch pipeline's construction path (FourChannelAecConfig(aec_preset=NO_FLOOR)) and the
 *     AEC C lib's own AecPreset enum. run_chain() passes it straight through in cfg.aec_preset.
 *   mono   mic=F ref=F sr=N fft=N out=F [aec=..] [nr=..] [legacy_amin=0|1] [enable_cng=0|1]
 *     audio_pipeline_process() as the applications call it, with the AEC seam read back after each hop.
 *   replay seam=F out=F [aec=..] [nr=..] [legacy_amin=0|1] [enable_cng=0|1] [usable=F]
 *     Open-loop replay of the stages behind the AEC over a recorded seam: RES stage (4-channel only;
 *     mono reuses the recorded G_res), NR stage, min, apply, comfort noise, synthesis. `usable=F` is one
 *     byte per hop that replaces the recorded usable_linear flag (the Python verdict of the same hop).
 *     Output: header (8 x int32) then per hop float32 G_res(nf) G_nr(nf) out(hop).
 *   probe  sr=N fft=N          (mono build) create a pipeline for the grid and print "hop n_freqs", or fail
 *   cng    seam=F out=F
 *     One aec_cng_fill call. Input (seam=): int32 n, uint32 rng, float32 floor, int32 use_g_nr, then float32
 *     spec(2n) cn(n) g_res(n) g_nr(n). Output: uint32 rng after the call, then float32 spec(2n).
 *
 * Seam file (all little endian):
 *   int32 hdr[10] = { 0x53435950 'PYCS', 1, kind (0 mono, 1 4ch), sr, fft, n_freqs, hop, n_hops,
 *                     fail_hop (-1 = none), fail_rc }
 *   per hop float32: E(2nf) Y(2nf) R2(nf) CN(nf) Gres(nf) Gnr(nf) far_hop(hop) out(hop) sc[16]
 *   sc: 0 far_power  1 max_dt  2 max_sat  3 all_converged  4 clock_drift  5 usable_linear
 *       6 delay_samples  7 delay_solid  8 delay_changed  9 realign_warm_total  10 realign_soft_total
 *       11 delay_confidence  12..15 unused (0)
 * Complex spectra are interleaved (re, im). Gres/Gnr are the gains the chain applied (unity when the
 * source is disabled). The seam is the data the stages behind the AEC read; scales as in se_stage.h. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_stft.h"
#include "utility/mem_align.h"
#include "utility/simd_kernels.h"
#include "common/se_anr_stage.h"
#include "common/se_stage.h"

#if defined(PYC_WITH_4CH)
#include "tests/conv_chain_4ch.h"
#include "aec_res.h"   /* aec_res_get_mem_size()/aec_res_init(): replay mode's RES stage below */
#endif
#if defined(PYC_WITH_MONO)
#include "audio_pipeline.h"
#endif

#define PYC_MAGIC 0x53435950
#define PYC_SC 16

typedef struct Args {
    const char *mic, *ref, *out, *seam, *usable;
    int sr, fft, legacy_amin, enable_nr, enable_res, enable_cng;
    AecPreset aec;
    AnrMode nr;
} Args;

static int parse_args(int argc, char** argv, Args* a) {
    int i;
    memset(a, 0, sizeof(*a));
    a->enable_nr = a->enable_res = a->enable_cng = 1;
    a->aec = AEC_PRESET_BALANCED;
    a->nr = ANR_MODE_BALANCED;
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
        else if (!strcmp(k, "aec")) {
            a->aec = !strcmp(v, "mild") ? AEC_PRESET_MILD
                   : !strcmp(v, "aggressive") ? AEC_PRESET_AGGRESSIVE
                   : !strcmp(v, "no_floor") ? AEC_PRESET_NO_FLOOR : AEC_PRESET_BALANCED;
            if (strcmp(v, "mild") && strcmp(v, "aggressive") && strcmp(v, "balanced")
                && strcmp(v, "no_floor")) {
                fprintf(stderr, "unknown aec preset %s\n", v); return -1;
            }
        } else if (!strcmp(k, "nr")) {
            int m = anr_nr_mode_from_name(v);
            if (m < 0) { fprintf(stderr, "unknown nr preset %s\n", v); return -1; }
            a->nr = (AnrMode)m;
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
static size_t seam_rec_floats(int nf, int hop) { return (size_t)(4 * nf + 4 * nf + 2 * hop + PYC_SC); }

static void write_hdr(FILE* f, int kind, int sr, int fft, int nf, int hop, int nh, int fail_hop, int fail_rc) {
    int32_t h[10];
    h[0] = PYC_MAGIC; h[1] = 1; h[2] = kind; h[3] = sr; h[4] = fft; h[5] = nf; h[6] = hop;
    h[7] = nh; h[8] = fail_hop; h[9] = fail_rc;
    fseek(f, 0, SEEK_SET);
    fwrite(h, sizeof(h[0]), 10, f);
}

static void write_rec(FILE* f, int nf, int hop, const Complex* e, const Complex* y, const float* r2,
                      const float* cn, const float* gres, const float* gnr, const float* far_hop,
                      const float* out, const float* sc) {
    static float* zero = NULL;
    static int zero_n = 0;
    int need = 2 * nf > hop ? 2 * nf : hop;
    if (zero_n < need) { free(zero); zero = (float*)calloc((size_t)need, sizeof(float)); zero_n = need; }
    fwrite(e ? (const void*)e : (const void*)zero, sizeof(Complex), (size_t)nf, f);
    fwrite(y ? (const void*)y : (const void*)zero, sizeof(Complex), (size_t)nf, f);
    fwrite(r2 ? r2 : zero, 4, (size_t)nf, f);
    fwrite(cn ? cn : zero, 4, (size_t)nf, f);
    fwrite(gres ? gres : zero, 4, (size_t)nf, f);
    fwrite(gnr ? gnr : zero, 4, (size_t)nf, f);
    fwrite(far_hop ? far_hop : zero, 4, (size_t)hop, f);
    fwrite(out, 4, (size_t)hop, f);
    fwrite(sc, 4, PYC_SC, f);
}

#if defined(PYC_WITH_4CH)
static int run_chain(const Args* a) {
    FourAecNrResConfig cfg = four_aec_nr_res_default_config(a->sr);
    ConvChain* c;
    FILE *fm, *fr, *fo;
    long nm, nr_, n;
    int hop, nf, nh, h, fail_hop = -1, fail_rc = 0, done = 0;
    float *mics, *ref, *out;
    Complex* w;
    cfg.fft_size = a->fft;
    cfg.aec_preset = a->aec;   /* se_derive()'s preset_valid() now accepts NO_FLOOR directly */
    cfg.nr_mode = a->nr;
    cfg.legacy_amin = a->legacy_amin;
    cfg.enable_nr = a->enable_nr;
    cfg.enable_res = a->enable_res;
    cfg.enable_cng = a->enable_cng;
    c = conv_create(&cfg);
    if (!c) { fprintf(stderr, "conv_create failed\n"); return 1; }
    hop = c->hop; nf = c->n_freqs;
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
        const SeSpecFrame* f;
        int rc;
        if (fread(mics, 4, 4 * (size_t)hop, fm) != 4 * (size_t)hop ||
            fread(ref, 4, (size_t)hop, fr) != (size_t)hop) { fprintf(stderr, "short read\n"); return 1; }
        rc = conv_hop(c, mics, ref, w, NULL, out);
        if (rc != 0) { fail_hop = h; fail_rc = rc; break; }
        f = &c->frame;
        memset(sc, 0, sizeof(sc));
        sc[0] = f->far_power; sc[1] = f->max_dt; sc[2] = f->max_sat; sc[3] = (float)f->all_converged;
        sc[4] = (float)f->clock_drift; sc[5] = (float)f->usable_linear;
        sc[6] = (float)c->front.delay.delay_samples; sc[7] = (float)c->front.delay.solid;
        sc[8] = (float)c->front.delay.changed;
        sc[9] = (float)four_aec_nr_res_realign_warm_lane_count(c->core);
        sc[10] = (float)four_aec_nr_res_realign_soft_lane_count(c->core);
        sc[11] = c->front.delay.confidence;
        write_rec(fo, nf, hop, f->e, f->y, f->r2, f->cn, c->res_gain, c->nr_gain, f->far_hop, out, sc);
        ++done;
    }
    write_hdr(fo, 1, a->sr, a->fft, nf, hop, done, fail_hop, fail_rc);
    fclose(fo); fclose(fm); fclose(fr);
    conv_destroy(c);
    free(mics); free(ref); free(out); free(w);
    return 0;
}
#endif /* PYC_WITH_4CH */

#if defined(PYC_WITH_MONO)
static int run_mono(const Args* a) {
    AudioPipelineConfig cfg = audio_pipeline_default_config(a->sr);
    AudioPipeline* p;
    FILE *fm, *fr, *fo;
    long n;
    int hop, nf, nh, h;
    float *mic, *ref, *out;
    cfg.fft_size = a->fft;
    cfg.aec_preset = a->aec;
    cfg.nr_mode = a->nr;
    cfg.legacy_amin = a->legacy_amin;
    cfg.enable_cng = a->enable_cng;
    cfg.enable_nr = a->enable_nr;
    cfg.enable_res = a->enable_res;
    p = audio_pipeline_create(&cfg);
    if (!p) { fprintf(stderr, "audio_pipeline_create failed\n"); return 1; }
    hop = audio_pipeline_hop_size(p); nf = audio_pipeline_n_freqs(p);
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
        AnrDenoiser* nr = audio_pipeline_get_nr(p);
        const float* g;
        float sc[PYC_SC];
        if (fread(mic, 4, (size_t)hop, fm) != (size_t)hop ||
            fread(ref, 4, (size_t)hop, fr) != (size_t)hop) { fprintf(stderr, "short read\n"); return 1; }
        if (audio_pipeline_process(p, mic, ref, out) != 0) { fprintf(stderr, "process failed at hop %d\n", h); return 1; }
        aec_get_res_context(audio_pipeline_get_aec(p), &c);
        aec_get_linear_context(audio_pipeline_get_aec(p), &lin);
        g = nr ? anr_get_gain(nr, NULL) : NULL;
        memset(sc, 0, sizeof(sc));
        sc[0] = c.far_power; sc[1] = c.dt_indicator; sc[2] = c.saturation_level;
        sc[3] = (float)c.filter_converged; sc[5] = (float)c.usable_linear;
        sc[6] = (float)lin.delay_samples; sc[11] = lin.delay_confidence;
        write_rec(fo, nf, hop, c.error_spec, c.near_spec, c.r2, c.comfort_noise, c.res_gain, g,
                  lin.aligned_far_hop, out, sc);
    }
    write_hdr(fo, 0, a->sr, a->fft, nf, hop, nh, -1, 0);
    fclose(fo); fclose(fm); fclose(fr);
    audio_pipeline_destroy(p);
    free(mic); free(ref); free(out);
    return 0;
}
#endif /* PYC_WITH_MONO */

static int run_replay(const Args* a) {
    FILE *fs = fopen(a->seam, "rb"), *fo = NULL, *fu = NULL;
    int32_t h[10];
    int kind, sr, fft, nf, hop, nh, t, k;
    size_t rec;
    float* buf;
    unsigned char* usable = NULL;
    AecConfig res_cfg;
    AnrConfig nr_cfg;
    AecRes* res = NULL;
    AnrDenoiser* nr;
    AcIstft* istft;
    AcStftConfig scfg;
    void *mem_res = NULL, *mem_nr = NULL, *mem_is = NULL;
    float *unity, *total, *extra;
    Complex* spec;
    uint32_t rng = SE_CNG_RNG_SEED;
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
#if defined(PYC_WITH_4CH)
    if (kind == 1) {
        FourAecNrResConfig cfg = four_aec_nr_res_default_config(sr);
        cfg.fft_size = fft; cfg.aec_preset = a->aec; cfg.nr_mode = a->nr;
        if (four_aec_nr_res_stage_configs(&cfg, &res_cfg, &nr_cfg) != 0) { fprintf(stderr, "stage configs failed\n"); return 1; }
    } else
#endif
    {
        if (kind == 1) { fprintf(stderr, "4ch replay needs pyc_drv_4ch\n"); return 1; }
        nr_cfg = se_anr_compose_config(sr, fft, hop, a->nr);
        memset(&res_cfg, 0, sizeof(res_cfg));
    }
    {
        size_t res_b = 0, nr_b = ALIGN16(se_anr_stage_mem_size(&nr_cfg));
        size_t is_b;
        ac_stft_config_default(&scfg, fft);
        is_b = ALIGN16(ac_istft_get_mem_size(&scfg));
        if (kind == 1) {
            res_b = ALIGN16(aec_res_get_mem_size(&res_cfg));
            if (posix_memalign(&mem_res, 16, res_b) != 0) return 1;
            res = aec_res_init(mem_res, aec_res_get_mem_size(&res_cfg), &res_cfg);
            if (!res) { fprintf(stderr, "aec_res_init failed\n"); return 1; }
        }
        if (posix_memalign(&mem_nr, 16, nr_b) != 0 || posix_memalign(&mem_is, 16, is_b) != 0) return 1;
        nr = se_anr_stage_init(mem_nr, se_anr_stage_mem_size(&nr_cfg), &nr_cfg);
        istft = ac_istft_init(mem_is, ac_istft_get_mem_size(&scfg), &scfg);
        if (!nr || !istft) { fprintf(stderr, "stage init failed\n"); return 1; }
    }
    unity = (float*)malloc(sizeof(float) * (size_t)nf);
    total = (float*)malloc(sizeof(float) * (size_t)nf);
    extra = (float*)malloc(sizeof(float) * (size_t)nf);
    spec = (Complex*)malloc(sizeof(Complex) * (size_t)nf);
    for (k = 0; k < nf; ++k) unity[k] = 1.0f;
    fo = fopen(a->out, "wb");
    if (!fo) { fprintf(stderr, "cannot open out\n"); return 1; }
    { int32_t oh[8] = { PYC_MAGIC, 1, kind, sr, fft, nf, hop, nh }; fwrite(oh, 4, 8, fo); }
    for (t = 0; t < nh; ++t) {
        const float* r = buf + rec * (size_t)t;
        SeSpecFrame f;
        const float *gres, *gnr;
        float* out = (float*)malloc(sizeof(float) * (size_t)hop);
        memset(&f, 0, sizeof(f));
        f.n_freqs = nf; f.hop = hop;
        f.e = (const Complex*)(r);
        f.y = (const Complex*)(r + 2 * nf);
        f.r2 = r + 4 * nf;
        f.cn = r + 5 * nf;
        f.far_hop = r + 8 * nf;
        {
            const float* sc = r + 8 * nf + 2 * hop;
            f.far_power = sc[0]; f.max_dt = sc[1]; f.max_sat = sc[2];
            f.all_converged = (int)sc[3]; f.clock_drift = (int)sc[4];
            f.usable_linear = usable ? usable[t] : (int)sc[5];
        }
        if (kind == 1) {
            gres = a->enable_res ? se_res_stage_gain(res, &f) : unity;
        } else {
            gres = a->enable_res ? r + 6 * nf : unity;           /* recorded G_res */
        }
        gnr = a->enable_nr ? se_anr_stage_gain(nr, &f, extra, !a->legacy_amin) : unity;
        if (!gres || !gnr) { fprintf(stderr, "stage refused hop %d\n", t); return 1; }
        sk_min_f32(total, gnr, gres, nf);
        sk_capply_gain_f32(spec, f.e, total, nf);
        if (a->enable_cng && a->enable_res) aec_cng_fill(&rng, spec, f.cn, gres, gnr, SE_CNG_NR_GAIN_FLOOR, nf);
        ac_istft_push(istft, spec, out);
        fwrite(gres, 4, (size_t)nf, fo);
        fwrite(gnr, 4, (size_t)nf, fo);
        fwrite(out, 4, (size_t)hop, fo);
        free(out);
    }
    fclose(fo);
    free(buf); free(usable); free(unity); free(total); free(extra); free(spec);
    free(mem_res); free(mem_nr); free(mem_is);
    return 0;
}

#if defined(PYC_WITH_MONO)
static int run_probe(const Args* a) {
    AudioPipelineConfig cfg = audio_pipeline_default_config(a->sr);
    AudioPipeline* p;
    cfg.fft_size = a->fft;
    p = audio_pipeline_create(&cfg);
    if (!p) { printf("create failed\n"); return 1; }
    printf("hop=%d n_freqs=%d\n", audio_pipeline_hop_size(p), audio_pipeline_n_freqs(p));
    audio_pipeline_destroy(p);
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
