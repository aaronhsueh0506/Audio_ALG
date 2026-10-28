/* Rate bridge between a pipeline's 8/16 kHz grid and the DFN2 stage's 48 kHz grid;
 * see dfn_rate_bridge.h for the signal, clock and delay contract. */

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "dfn_rate_bridge.h"
#include "utility/mem_align.h"
#include "audio_resampler.h"
#include "fft_wrapper.h"
#include "audio_stft.h"

#define BRIDGE_ALIGN 16u
#define B48_FFT  DFN2_N_FFT
#define B48_HOP  DFN2_HOP_LEN
#define B48_BINS DFN2_N_BINS
/* FIFO capacities: 511 leftover samples plus one upsampled native hop (<= 768);
 * the output FIFO holds the prefill, one hop of surplus and two downsampled frames. */
#define FIFO48_CAP   1536
#define OUT_FIFO_CAP 1024
/* Hops simulated at init to find the schedule's largest emitted-minus-produced
 * deficit (every supported grid repeats within four hops). */
#define PREFILL_SIM_HOPS 8

struct DfnRateBridge {
    int fft_n;
    int hop_n;
    int prefill;
    int added_delay_sixths;   /* added delay in sixths of a native sample */
    FftHandle *fft_n_handle;
    FftHandle *fft48;
    DfnResStage *stage;
    AudioResampler *up_e;
    AudioResampler *up_p;
    AudioResampler *down;
    float *win_n;           /* shared by the native synthesis objects */
    float *win48;           /* shared by the 48 kHz analysis and synthesis objects */
    AcIstft *synth_e;       /* native grid, estimate lane */
    AcIstft *synth_p;       /* native grid, apply lane */
    AcIstft *synth48;       /* 48 kHz stage output */
    AcStft *stft_e;         /* 48 kHz analysis, estimate lane (owns its history) */
    AcStft *stft_p;         /* 48 kHz analysis, apply lane */
    float *e_hop;
    float *p_hop;
    float *fifo_e;          /* pending 48 kHz samples */
    float *fifo_p;
    float *chunk48;
    float *out_fifo;
    Complex *spec_e48;
    Complex *spec_p48;
    Complex *out48;
    int fill;               /* pending samples in each 48 kHz FIFO */
    int out_fill;
    long long frames48;
    int last_hop_frames;
    int phase;              /* 0 idle, 1 hop begun, 2 stage frame open */
    void *owned_heap;
};

static size_t align_up(size_t v) {
    if (v > SIZE_MAX - (BRIDGE_ALIGN - 1u)) return 0u;
    return (v + BRIDGE_ALIGN - 1u) & ~(size_t)(BRIDGE_ALIGN - 1u);
}

static uint32_t fnv_u32(uint32_t h, uint32_t v) {
    int i;
    for (i = 0; i < 4; ++i) {
        h ^= (v >> (8 * i)) & 0xffu;
        h *= 16777619u;
    }
    return h;
}

/* Capability gate: the three native grids the resamplers, FIFO capacities and
 * prefill calibration are sized for; hop = fft / 2. */
static int grid_dims(int sample_rate, int fft_size, int *fft, int *hop) {
    int f = fft_size == 0 ? 256 : fft_size;
    if (sample_rate == 8000) {
        if (f != 256) return -1;
    } else if (sample_rate == 16000) {
        if (f != 256 && f != 512) return -1;
    } else {
        return -1;
    }
    *fft = f;
    *hop = f / 2;
    return 0;
}

/* Everything init needs, derived from the config once. */
typedef struct Resolved {
    int fft_n;
    int hop_n;
    size_t fft_n_bytes;
    size_t fft48_bytes;
    size_t synth_n_bytes;
    size_t synth48_bytes;
    size_t stft48_bytes;
    size_t up_bytes;
    size_t down_bytes;
    DfnResStageMemReq stage_req;
    size_t bytes;
} Resolved;

/* Bytes of an STFT/ISTFT object that borrows the bridge's FFT handle and window
 * table (the object that carves its own, less those two regions). */
static size_t borrowed_bytes(size_t own_bytes, int fft_size) {
    size_t shared = align_up(fft_get_mem_size(fft_size)) +
                    align_up((size_t)fft_size * sizeof(float));
    return own_bytes > shared ? align_up(own_bytes - shared) : 0u;
}

static int resolve(const DfnRateBridgeConfig *cfg, Resolved *r) {
    size_t value;
    /* added_delay depends on the lookahead pair, which must be dfn2_process.o's. */
    DFN2_REQUIRE_LOOKAHEAD_PAIR_LINK();
    if (!cfg || !r || grid_dims(cfg->sample_rate, cfg->fft_size,
                                &r->fft_n, &r->hop_n) != 0)
        return -1;
    if (cfg->stage.sample_rate != DFN2_SR ||
        dfn_res_stage_get_mem_requirements(&cfg->stage, &r->stage_req) != 0)
        return -1;
    r->fft_n_bytes = fft_get_mem_size(r->fft_n);
    r->fft48_bytes = fft_get_mem_size(B48_FFT);
    {
        AcStftConfig sc;
        ac_stft_config_default(&sc, r->fft_n);
        r->synth_n_bytes = borrowed_bytes(ac_istft_get_mem_size(&sc), r->fft_n);
        ac_stft_config_default(&sc, B48_FFT);
        r->synth48_bytes = borrowed_bytes(ac_istft_get_mem_size(&sc), B48_FFT);
        r->stft48_bytes = borrowed_bytes(ac_stft_get_mem_size(&sc), B48_FFT);
    }
    r->up_bytes = audio_resampler_get_mem_size(cfg->sample_rate, DFN2_SR, 1);
    r->down_bytes = audio_resampler_get_mem_size(DFN2_SR, cfg->sample_rate, 1);
    if (!r->fft_n_bytes || !r->fft48_bytes || !r->synth_n_bytes ||
        !r->synth48_bytes || !r->stft48_bytes || !r->up_bytes ||
        !r->down_bytes)
        return -1;
    value = align_up(sizeof(DfnRateBridge));
#define ADD_REGION(bytes_) do { \
        size_t add_ = align_up((size_t)(bytes_)); \
        if (!add_ || value > SIZE_MAX - add_) return -1; \
        value += add_; \
    } while (0)
    ADD_REGION(r->fft_n_bytes);
    ADD_REGION(r->fft48_bytes);
    ADD_REGION(r->stage_req.bytes);
    ADD_REGION(r->up_bytes);
    ADD_REGION(r->up_bytes);
    ADD_REGION(r->down_bytes);
    ADD_REGION((size_t)r->fft_n * sizeof(float));        /* win_n   */
    ADD_REGION((size_t)B48_FFT * sizeof(float));         /* win48   */
    ADD_REGION(r->synth_n_bytes);                        /* synth_e */
    ADD_REGION(r->synth_n_bytes);                        /* synth_p */
    ADD_REGION(r->synth48_bytes);                        /* synth48 */
    ADD_REGION(r->stft48_bytes);                         /* stft_e  */
    ADD_REGION(r->stft48_bytes);                         /* stft_p  */
    ADD_REGION((size_t)r->hop_n * sizeof(float));        /* e_hop   */
    ADD_REGION((size_t)r->hop_n * sizeof(float));        /* p_hop   */
    ADD_REGION((size_t)FIFO48_CAP * sizeof(float));      /* fifo_e  */
    ADD_REGION((size_t)FIFO48_CAP * sizeof(float));      /* fifo_p  */
    ADD_REGION((size_t)B48_HOP * sizeof(float));         /* chunk48 */
    ADD_REGION((size_t)OUT_FIFO_CAP * sizeof(float));    /* out_fifo*/
    ADD_REGION((size_t)B48_BINS * sizeof(Complex));      /* spec_e48*/
    ADD_REGION((size_t)B48_BINS * sizeof(Complex));      /* spec_p48*/
    ADD_REGION((size_t)B48_BINS * sizeof(Complex));      /* out48   */
#undef ADD_REGION
    r->bytes = value;
    return 0;
}

static void fill_descriptor(const DfnRateBridgeConfig *cfg, const Resolved *r,
                            DfnRateBridgeMemReq *out) {
    uint32_t h = 2166136261u;
    h = fnv_u32(h, DFN_RATE_BRIDGE_LAYOUT_VERSION);
    h = fnv_u32(h, (uint32_t)cfg->sample_rate);
    h = fnv_u32(h, (uint32_t)r->fft_n);
    h = fnv_u32(h, r->stage_req.layout_version);
    h = fnv_u32(h, r->stage_req.build_flags_hash);
    h = fnv_u32(h, (uint32_t)FIFO48_CAP);
    h = fnv_u32(h, (uint32_t)OUT_FIFO_CAP);
    /* The resamplers' pool sizes: a pool recorded against another resampler layout
     * is refused on the hash. */
    h = fnv_u32(h, (uint32_t)r->up_bytes);
    h = fnv_u32(h, (uint32_t)r->down_bytes);
    memset(out, 0, sizeof(*out));
    out->descriptor_version = DFN_RATE_BRIDGE_DESCRIPTOR_VERSION;
    out->layout_version = DFN_RATE_BRIDGE_LAYOUT_VERSION;
    out->backend_id = r->stage_req.backend_id;
    out->build_flags_hash = ac_memreq_abi_hash(h);
    out->alignment = BRIDGE_ALIGN;
    out->bytes = (uint64_t)r->bytes;
}

DfnRateBridgeConfig dfn_rate_bridge_default_config(int sample_rate) {
    DfnRateBridgeConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.sample_rate = sample_rate;
    cfg.fft_size = 0;
    cfg.stage = dfn_res_stage_default_config(DFN2_SR);
    return cfg;
}

int dfn_rate_bridge_get_mem_requirements(const DfnRateBridgeConfig *cfg,
                                         DfnRateBridgeMemReq *out) {
    Resolved r;
    if (!out || resolve(cfg, &r) != 0) return -1;
    fill_descriptor(cfg, &r, out);
    return 0;
}

static int descriptor_matches(const DfnRateBridgeMemReq *a,
                              const DfnRateBridgeMemReq *b) {
    return a && b && a->descriptor_version == b->descriptor_version &&
        a->layout_version == b->layout_version &&
        a->backend_id == b->backend_id &&
        a->build_flags_hash == b->build_flags_hash &&
        a->alignment == b->alignment && a->reserved == 0u &&
        a->bytes >= b->bytes;
}

static void clear_streams(DfnRateBridge *b) {
    audio_resampler_reset(b->up_e);
    audio_resampler_reset(b->up_p);
    audio_resampler_reset(b->down);
    ac_istft_reset(b->synth_e);
    ac_istft_reset(b->synth_p);
    ac_istft_reset(b->synth48);
    ac_stft_reset(b->stft_e);
    ac_stft_reset(b->stft_p);
    memset(b->out_fifo, 0, (size_t)OUT_FIFO_CAP * sizeof(float));
    b->fill = 0;
    b->out_fill = b->prefill;
    b->frames48 = 0;
    b->last_hop_frames = 0;
    b->phase = 0;
}

/* Walks the count schedule with the real resamplers on silence to find how many
 * native samples the output FIFO must hold before the first hop. Only counts are
 * used; buffers are cleared again by clear_streams(). Leaves the resamplers reset. */
static int calibrate_prefill(DfnRateBridge *b) {
    int fill = 0, produced_total = 0, emitted = 0, max_deficit = 0;
    int t, consumed, produced;
    memset(b->e_hop, 0, (size_t)b->hop_n * sizeof(float));
    memset(b->chunk48, 0, (size_t)B48_HOP * sizeof(float));
    audio_resampler_reset(b->up_e);
    audio_resampler_reset(b->down);
    for (t = 0; t < PREFILL_SIM_HOPS; ++t) {
        if (audio_resampler_process(b->up_e, b->e_hop, b->hop_n,
                                    b->fifo_e + fill,
                                    FIFO48_CAP - fill,
                                    &consumed, &produced) != 0 ||
            consumed != b->hop_n)
            return -1;
        fill += produced;
        while (fill >= B48_HOP) {
            fill -= B48_HOP;
            if (audio_resampler_process(b->down, b->chunk48, B48_HOP,
                                        b->out_fifo, OUT_FIFO_CAP,
                                        &consumed, &produced) != 0 ||
                consumed != B48_HOP)
                return -1;
            produced_total += produced;
        }
        emitted += b->hop_n;
        if (emitted - produced_total > max_deficit)
            max_deficit = emitted - produced_total;
    }
    audio_resampler_reset(b->up_e);
    audio_resampler_reset(b->down);
    return max_deficit;
}

DfnRateBridge *dfn_rate_bridge_init_ex(void *mem, size_t bytes,
                                       const DfnRateBridgeConfig *cfg,
                                       const DfnRateBridgeMemReq *expected) {
    Resolved r;
    DfnRateBridgeMemReq req;
    DfnRateBridge *b;
    unsigned char *cursor;
    int lat_up, lat_down;
    long long sixths;
    if (!mem || (uintptr_t)mem % BRIDGE_ALIGN != 0u || resolve(cfg, &r) != 0)
        return NULL;
    fill_descriptor(cfg, &r, &req);
    if (bytes < r.bytes || (expected && !descriptor_matches(expected, &req)))
        return NULL;
    memset(mem, 0, r.bytes);
    b = (DfnRateBridge *)mem;
    b->fft_n = r.fft_n;
    b->hop_n = r.hop_n;
    cursor = (unsigned char *)mem + align_up(sizeof(*b));
#define TAKE(ptr_, type_, bytes_) do { \
        ptr_ = (type_)cursor; \
        cursor += align_up((size_t)(bytes_)); \
    } while (0)
    b->fft_n_handle = fft_init(cursor, r.fft_n_bytes, r.fft_n);
    cursor += align_up(r.fft_n_bytes);
    b->fft48 = fft_init(cursor, r.fft48_bytes, B48_FFT);
    cursor += align_up(r.fft48_bytes);
    b->stage = dfn_res_stage_init_ex(cursor, (size_t)r.stage_req.bytes,
                                     &cfg->stage, &r.stage_req);
    cursor += align_up((size_t)r.stage_req.bytes);
    b->up_e = audio_resampler_init(cursor, r.up_bytes, cfg->sample_rate,
                                   DFN2_SR, 1);
    cursor += align_up(r.up_bytes);
    b->up_p = audio_resampler_init(cursor, r.up_bytes, cfg->sample_rate,
                                   DFN2_SR, 1);
    cursor += align_up(r.up_bytes);
    b->down = audio_resampler_init(cursor, r.down_bytes, DFN2_SR,
                                   cfg->sample_rate, 1);
    cursor += align_up(r.down_bytes);
    if (!b->fft_n_handle || !b->fft48 || !b->stage || !b->up_e || !b->up_p ||
        !b->down)
        return NULL;
    TAKE(b->win_n, float *, (size_t)r.fft_n * sizeof(float));
    TAKE(b->win48, float *, (size_t)B48_FFT * sizeof(float));
    ac_window_build(b->win_n, r.fft_n);
    ac_window_build(b->win48, B48_FFT);
    {
        AcStftConfig nc, wc;
        void *region;
        ac_stft_config_default(&nc, r.fft_n);
        nc.fft = b->fft_n_handle;
        nc.window = b->win_n;
        ac_stft_config_default(&wc, B48_FFT);
        wc.fft = b->fft48;
        wc.window = b->win48;
        TAKE(region, void *, r.synth_n_bytes);
        b->synth_e = ac_istft_init(region, r.synth_n_bytes, &nc);
        TAKE(region, void *, r.synth_n_bytes);
        b->synth_p = ac_istft_init(region, r.synth_n_bytes, &nc);
        TAKE(region, void *, r.synth48_bytes);
        b->synth48 = ac_istft_init(region, r.synth48_bytes, &wc);
        TAKE(region, void *, r.stft48_bytes);
        b->stft_e = ac_stft_init(region, r.stft48_bytes, &wc);
        TAKE(region, void *, r.stft48_bytes);
        b->stft_p = ac_stft_init(region, r.stft48_bytes, &wc);
    }
    if (!b->synth_e || !b->synth_p || !b->synth48 || !b->stft_e || !b->stft_p)
        return NULL;
    TAKE(b->e_hop, float *, (size_t)r.hop_n * sizeof(float));
    TAKE(b->p_hop, float *, (size_t)r.hop_n * sizeof(float));
    TAKE(b->fifo_e, float *, (size_t)FIFO48_CAP * sizeof(float));
    TAKE(b->fifo_p, float *, (size_t)FIFO48_CAP * sizeof(float));
    TAKE(b->chunk48, float *, (size_t)B48_HOP * sizeof(float));
    TAKE(b->out_fifo, float *, (size_t)OUT_FIFO_CAP * sizeof(float));
    TAKE(b->spec_e48, Complex *, (size_t)B48_BINS * sizeof(Complex));
    TAKE(b->spec_p48, Complex *, (size_t)B48_BINS * sizeof(Complex));
    TAKE(b->out48, Complex *, (size_t)B48_BINS * sizeof(Complex));
#undef TAKE

    b->prefill = calibrate_prefill(b);
    lat_up = audio_resampler_latency_input_frames(b->up_e);
    lat_down = audio_resampler_latency_input_frames(b->down);
    if (b->prefill < 0 || b->prefill > OUT_FIFO_CAP / 2 || lat_up < 0 ||
        lat_down < 0)
        return NULL;
    /* Prefill and up latency are whole native samples; the stage and the down latency are
     * 48 kHz frames, and sample_rate is a multiple of 8000, so frames * sample_rate / 8000
     * is their native-sample value in sixths. */
    sixths = ((long long)b->prefill + lat_up) * DFN_RATE_BRIDGE_DELAY_DENOMINATOR +
             ((long long)(DFN_RES_STAGE_LOOKAHEAD_FRAMES + 1) * B48_HOP + lat_down) *
                 cfg->sample_rate / 8000;
    if (sixths > INT_MAX) return NULL;
    b->added_delay_sixths = (int)sixths;
    clear_streams(b);
    return b;
}

DfnRateBridge *dfn_rate_bridge_init(void *mem, size_t bytes,
                                    const DfnRateBridgeConfig *cfg) {
    return dfn_rate_bridge_init_ex(mem, bytes, cfg, NULL);
}

DfnRateBridge *dfn_rate_bridge_create(const DfnRateBridgeConfig *cfg) {
    DfnRateBridgeMemReq req;
    DfnRateBridge *b;
    void *mem = NULL;
    if (dfn_rate_bridge_get_mem_requirements(cfg, &req) != 0) return NULL;
    if (posix_memalign(&mem, req.alignment, (size_t)req.bytes) != 0) return NULL;
    b = dfn_rate_bridge_init_ex(mem, (size_t)req.bytes, cfg, &req);
    if (!b) { free(mem); return NULL; }
    b->owned_heap = mem;
    return b;
}

void dfn_rate_bridge_destroy(DfnRateBridge *b) {
    void *owned;
    if (!b) return;
    owned = b->owned_heap;
    b->owned_heap = NULL;
    dfn_res_stage_destroy(b->stage);
    audio_resampler_destroy(b->down);
    audio_resampler_destroy(b->up_p);
    audio_resampler_destroy(b->up_e);
    fft_destroy(b->fft48);
    fft_destroy(b->fft_n_handle);
    if (owned) free(owned);
}

void dfn_rate_bridge_reset(DfnRateBridge *b) {
    if (!b) return;
    dfn_res_stage_reset(b->stage);
    clear_streams(b);
}

int dfn_rate_bridge_reset_synthesis(DfnRateBridge *b) {
    if (!b || b->phase != 0) return -1;
    ac_istft_reset(b->synth_e);
    ac_istft_reset(b->synth_p);
    return 0;
}

/* Upsamples one native hop into the FIFO's pending region; returns the count
 * produced, or -1. */
static int push_up(DfnRateBridge *b, AudioResampler *up, const float *hop,
                   float *fifo) {
    int consumed, produced;
    if (audio_resampler_process(up, hop, b->hop_n, fifo + b->fill,
                                FIFO48_CAP - b->fill, &consumed,
                                &produced) != 0 ||
        consumed != b->hop_n)
        return -1;
    return produced;
}

int dfn_rate_bridge_begin(DfnRateBridge *b,
                          const Complex *estimate_spec,
                          const Complex *apply_spec) {
    int produced_e, produced_p;
    if (!b || !estimate_spec || !apply_spec || b->phase != 0) return -1;
    ac_istft_push(b->synth_e, estimate_spec, b->e_hop);
    ac_istft_push(b->synth_p, apply_spec, b->p_hop);
    produced_e = push_up(b, b->up_e, b->e_hop, b->fifo_e);
    produced_p = push_up(b, b->up_p, b->p_hop, b->fifo_p);
    if (produced_e < 0 || produced_p != produced_e) return -1;
    b->fill += produced_e;
    b->last_hop_frames = 0;
    b->phase = 1;
    return 0;
}

/* Closes the open (or warm-up) stage frame: stage output -> 48 kHz WOLA -> resample
 * down into the native FIFO. */
static int finish_frame(DfnRateBridge *b, int model_rc) {
    int consumed, produced;
    if (dfn_res_stage_end(b->stage, model_rc, b->out48) < 0) return -1;
    ac_istft_push(b->synth48, b->out48, b->chunk48);
    if (audio_resampler_process(b->down, b->chunk48, B48_HOP,
                                b->out_fifo + b->out_fill,
                                OUT_FIFO_CAP - b->out_fill,
                                &consumed, &produced) != 0 ||
        consumed != B48_HOP)
        return -1;
    b->out_fill += produced;
    ++b->frames48;
    ++b->last_hop_frames;
    return 0;
}

int dfn_rate_bridge_nn_begin(DfnRateBridge *b, DfnResStageJob *job) {
    if (!b || !job || b->phase != 1) return -1;
    while (b->fill >= B48_HOP) {
        int r;
        /* 48 kHz sqrt-Hann 1024/512 analysis; each STFT object keeps its lane's history. */
        ac_stft_push(b->stft_e, b->fifo_e, b->spec_e48);
        ac_stft_push(b->stft_p, b->fifo_p, b->spec_p48);
        b->fill -= B48_HOP;
        memmove(b->fifo_e, b->fifo_e + B48_HOP,
                (size_t)b->fill * sizeof(float));
        memmove(b->fifo_p, b->fifo_p + B48_HOP,
                (size_t)b->fill * sizeof(float));
        r = dfn_res_stage_begin(b->stage, b->spec_e48, b->spec_p48, job);
        if (r < 0) return -1;
        if (r == 1) {
            b->phase = 2;
            return 1;
        }
        if (finish_frame(b, 0) != 0) return -1;   /* warm-up: nothing to run */
    }
    return 0;
}

int dfn_rate_bridge_nn_end(DfnRateBridge *b, int model_rc) {
    if (!b || b->phase != 2) return -1;
    b->phase = 1;
    return finish_frame(b, model_rc);
}

int dfn_rate_bridge_finish(DfnRateBridge *b, float *out_hop) {
    DfnResStageJob job;
    int r;
    if (!b || !out_hop || b->phase == 0) return -1;
    /* Frames left open or never fetched take the identity. */
    if (b->phase == 2 && dfn_rate_bridge_nn_end(b, -1) != 0) return -1;
    while ((r = dfn_rate_bridge_nn_begin(b, &job)) == 1)
        if (dfn_rate_bridge_nn_end(b, -1) != 0) return -1;
    if (r < 0) return -1;
    b->phase = 0;
    if (b->out_fill < b->hop_n) return -1;   /* excluded by the prefill */
    memcpy(out_hop, b->out_fifo, (size_t)b->hop_n * sizeof(float));
    b->out_fill -= b->hop_n;
    memmove(b->out_fifo, b->out_fifo + b->hop_n,
            (size_t)b->out_fill * sizeof(float));
    return 0;
}

int dfn_rate_bridge_process(DfnRateBridge *b,
                            const Complex *estimate_spec,
                            const Complex *apply_spec,
                            float *out_hop) {
    DfnResStageJob job;
    int r;
    if (!b || !out_hop) return -1;
    if (dfn_rate_bridge_begin(b, estimate_spec, apply_spec) != 0) return -1;
    while ((r = dfn_rate_bridge_nn_begin(b, &job)) == 1)
        if (dfn_rate_bridge_nn_end(b, dfn_res_stage_infer(b->stage, &job)) != 0)
            return -1;
    if (r < 0) return -1;
    return dfn_rate_bridge_finish(b, out_hop);
}

int dfn_rate_bridge_hop_size(const DfnRateBridge *b) {
    return b ? b->hop_n : -1;
}

int dfn_rate_bridge_n_freqs(const DfnRateBridge *b) {
    return b ? b->fft_n / 2 + 1 : -1;
}

int dfn_rate_bridge_added_delay_samples(const DfnRateBridge *b) {
    return b ? b->added_delay_sixths / DFN_RATE_BRIDGE_DELAY_DENOMINATOR : -1;
}

int dfn_rate_bridge_added_delay_exact(const DfnRateBridge *b, int *numerator,
                                      int *denominator) {
    if (!b || !numerator || !denominator) return -1;
    *numerator = b->added_delay_sixths;
    *denominator = DFN_RATE_BRIDGE_DELAY_DENOMINATOR;
    return 0;
}

DfnResStage *dfn_rate_bridge_stage(const DfnRateBridge *b) {
    return b ? b->stage : NULL;
}

void dfn_rate_bridge_get_counters(const DfnRateBridge *b,
                                  long long *frames_48k, int *last_hop_frames) {
    if (frames_48k) *frames_48k = b ? b->frames48 : 0;
    if (last_hop_frames) *last_hop_frames = b ? b->last_hop_frames : 0;
}

int dfn_rate_bridge_get_host_hops(const DfnRateBridge *b, const float **estimate_hop,
                                  const float **apply_hop) {
    if (!b) return -1;
    if (estimate_hop) *estimate_hop = b->e_hop;
    if (apply_hop) *apply_hop = b->p_hop;
    return 0;
}
