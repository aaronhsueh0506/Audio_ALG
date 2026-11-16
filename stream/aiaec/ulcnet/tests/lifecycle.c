/* The call lifecycle of ulcnet_stream, through the public entries only: what a run status, a faulty mask, an alignment
 * event or a misordered call does to the hop's frames and to the state generation, one hop and packet (N = 4), on the
 * spectrum entries of a spectrum-only stream and on both entry families of a full stream.
 *
 *   guards    a post with nothing open, a second pre while a hop is open, a reset with a hop open, an entry of the other N,
 *             NULL arguments
 *   reprime   an alignment edge at one hop: the hop is the error spectrum, no job, io zeroed, ai_rc ignored, one restart
 *   outcomes  every status and mask fault x {one hop, packet without event, packet with an event}: which frames are the
 *             error spectrum, how far the generation moves (once, however many frames are bad), and the next hop
 *   range     the four pre entries refuse a NaN / Inf / over-limit input with ULCNET_STREAM_ERR_RANGE before anything
 *             moves: the next calls are bit for bit those of a twin stream that never saw the refused one; the limits
 *             themselves are accepted, one ulp above is not
 * Every expectation is built here; the mask is an input-dependent stand-in written by the test. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ulcnet_stream.h"
#include "stream_run.h"

static int g_fail;
#define CHECK(c, ...) do { if (!(c)) { printf("FAIL (line %d): ", __LINE__); printf(__VA_ARGS__); printf("\n"); ++g_fail; } } while (0)

enum { NP = 4, PLANE_MAX = NP * 129, HOP = 128, BINS = 129 };
enum { FAM_PCM = 0, FAM_FREQ_ONLY = 1, FAM_FREQ_FULL = 2 };
enum { K_GOOD, K_UNWRITTEN, K_NAN, K_HUGE_LAST, K_HUGE_TWO, K_HUGE_EVENT, K_NAN_EVENT, K_NOT_STARTED, K_FAILED, K_ODD };

static uint32_t lcg(uint32_t *s) { *s = *s * 1664525u + 1013904223u; return *s; }
static float urand(uint32_t *s) { return (float)(int)(lcg(s) >> 9) / 4194304.0f - 1.0f; }

typedef struct Rig {
    UlcnetStream *s;
    void *pool;
    int n, fam;
} Rig;

static int rig_make(Rig *r, int n, int fam) {
    UlcnetGridSpec spec = { 16000, 256, 8 };
    UlcnetStreamConfig cfg;
    UlcnetStreamMemReq req;
    memset(r, 0, sizeof *r);
    memset(&cfg, 0, sizeof cfg);
    if (ulcnet_grid_make(&cfg.grid, &spec) != ULCNET_GRID_OK) return -1;
    cfg.frames_per_call = n;
    cfg.freq_only = fam == FAM_FREQ_ONLY;
    if (ulcnet_stream_get_mem_size(&cfg, &req) != 0 || posix_memalign(&r->pool, 16, (size_t)req.bytes) != 0) return -1;
    r->s = ulcnet_stream_init_ex(&cfg, r->pool, (size_t)req.bytes, &req);
    r->n = n;
    r->fam = fam;
    return r->s ? 0 : -1;
}

static void rig_free(Rig *r) { free(r->pool); memset(r, 0, sizeof *r); }

/* One hop (or packet) of input, spectra and PCM, with a distinct level per frame. */
typedef struct Input {
    float er[PLANE_MAX], ei[PLANE_MAX], fr[PLANE_MAX], fi[PLANE_MAX];
    float epcm[NP * HOP], fpcm[NP * HOP];
} Input;

static void make_input(Input *in, uint32_t *seed, int n) {
    int i;
    for (i = 0; i < n * BINS; ++i) {
        const int edge = i % BINS == 0 || i % BINS == BINS - 1;
        in->er[i] = 0.2f * urand(seed) + 0.05f;
        in->ei[i] = edge ? 0.0f : 0.2f * urand(seed);
        in->fr[i] = 0.3f * urand(seed);
        in->fi[i] = edge ? 0.0f : 0.3f * urand(seed);
    }
    for (i = 0; i < n * HOP; ++i) { in->epcm[i] = 0.1f * urand(seed); in->fpcm[i] = 0.1f * urand(seed); }
}

static int pre_call(Rig *r, const Input *in, const UlcnetStreamDelay *d, UlcnetHopIO *io) {
    if (r->n == 1)
        return r->fam == FAM_PCM ? ulcnet_stream_pre_process_f32(r->s, in->epcm, in->fpcm, d[0], io)
                                 : ulcnet_stream_pre_process_freq(r->s, in->er, in->ei, in->fr, in->fi, d[0], io);
    return r->fam == FAM_PCM ? ulcnet_stream_pre_process_packet_f32(r->s, in->epcm, in->fpcm, d, io)
                             : ulcnet_stream_pre_process_packet_freq(r->s, in->er, in->ei, in->fr, in->fi, d, io);
}

/* The spectrum families: out_re / out_im get the frames. The PCM family: pcm gets r->n * HOP samples. */
static int post_call(Rig *r, int ai_rc, float *out_re, float *out_im, int16_t *pcm) {
    return r->fam == FAM_PCM ? ulcnet_stream_post_process(r->s, ai_rc, pcm)
                             : ulcnet_stream_post_process_freq(r->s, ai_rc, out_re, out_im);
}

static UlcnetStreamDelay quiet(void) { UlcnetStreamDelay d = { 0, 1 }; return d; }

/* The stand-in accelerator: an input-dependent mask, then the fault of `kind`. `event` is the packet's event frame. */
static int run_kind(UlcnetHopIO *io, int kind, int n, int event) {
    const size_t bins = BINS, plane = (size_t)n * bins;
    double acc = 0.0;
    size_t i;
    if (kind == K_NOT_STARTED) return STREAM_RUN_NOT_STARTED;
    if (kind == K_FAILED) return STREAM_RUN_FAILED;
    if (kind == K_ODD) return 12345;
    for (i = 0; i < plane; ++i) acc += io->inputs.error_mag[i] + 0.5 * io->inputs.far_mag[i];
    if (kind != K_UNWRITTEN) {
        for (i = 0; i < plane; ++i) {
            io->outputs.output[i] = (float)(0.5 + 0.25 * sin(acc + (double)i));
            io->outputs.output[plane + i] = (float)(0.125 * cos(acc + (double)i));
        }
    }
    if (kind == K_NAN) io->outputs.output[plane + 7] = NAN;
    if (kind == K_HUGE_LAST) io->outputs.output[(size_t)(n - 1) * bins + 3] = 3.0e30f;           /* last frame, real plane */
    if (kind == K_HUGE_TWO) { io->outputs.output[bins + 3] = 3.0e30f; io->outputs.output[plane + (size_t)(n - 1) * bins + 5] = 3.0e30f; }
    if (kind == K_HUGE_EVENT) io->outputs.output[(size_t)event * bins + 4] = 3.0e30f;           /* the event frame: not applied */
    if (kind == K_NAN_EVENT) io->outputs.output[(size_t)event * bins + 4] = NAN;
    return STREAM_RUN_OK;
}

static int same(const float *a, const float *b, size_t n) { return memcmp(a, b, n * sizeof(float)) == 0; }

/* ---- guards ---------------------------------------------------------------------------------------------- */

static void row_guards(void) {
    int n, fam;
    for (n = 1; n <= NP; n += NP - 1)
        for (fam = FAM_PCM; fam <= FAM_FREQ_FULL; ++fam) {
            Rig r;
            Input in;
            UlcnetHopIO io;
            UlcnetStreamDelay d[NP];
            float re[PLANE_MAX], im[PLANE_MAX];
            int16_t pcm[NP * HOP];
            uint32_t seed = 11u, gen;
            int k;
            CHECK(rig_make(&r, n, fam) == 0, "guards: setup n=%d fam=%d", n, fam);
            if (!r.s) continue;
            for (k = 0; k < NP; ++k) d[k] = quiet();
            make_input(&in, &seed, n);
            CHECK(post_call(&r, 0, re, im, pcm) == -1, "guards: a post with nothing open (n=%d fam=%d)", n, fam);
            CHECK(pre_call(&r, &in, d, &io) == 1 && io.state_generation == ulcnet_stream_state_generation(r.s),
                  "guards: pre (n=%d fam=%d)", n, fam);
            gen = ulcnet_stream_state_generation(r.s);
            CHECK(pre_call(&r, &in, d, &io) == -1, "guards: a second pre while a hop is open (n=%d fam=%d)", n, fam);
            CHECK(ulcnet_stream_state_generation(r.s) == gen, "guards: the refused second pre moved the generation");
            CHECK(post_call(&r, STREAM_RUN_NOT_STARTED, re, im, pcm) == 1, "guards: the open hop is still the caller's");
            CHECK(post_call(&r, 0, re, im, pcm) == -1, "guards: a second post");
            /* a reset with a hop open drops it */
            CHECK(pre_call(&r, &in, d, &io) == 1, "guards: pre before reset");
            gen = ulcnet_stream_state_generation(r.s);
            ulcnet_stream_reset(r.s);
            CHECK(ulcnet_stream_state_generation(r.s) == gen + 1u, "guards: reset must advance the generation once");
            CHECK(post_call(&r, 0, re, im, pcm) == -1, "guards: a post after a reset with the hop open");
            CHECK(pre_call(&r, &in, d, &io) == 1, "guards: pre after reset");
            CHECK(post_call(&r, STREAM_RUN_NOT_STARTED, re, im, pcm) == 1, "guards: post after reset");
            rig_free(&r);
        }
    {   /* an entry of the other N is refused and leaves nothing open: the stream still takes its own entry */
        static const struct { int n, fam; } rows[] = { { 1, FAM_PCM }, { NP, FAM_PCM }, { 1, FAM_FREQ_FULL }, { NP, FAM_FREQ_FULL } };
        size_t i;
        for (i = 0; i < sizeof rows / sizeof rows[0]; ++i) {
            Rig r;
            Input in;
            UlcnetHopIO io;
            UlcnetStreamDelay d[NP];
            float re[PLANE_MAX], im[PLANE_MAX];
            int16_t pcm[NP * HOP];
            uint32_t seed = 21u, gen;
            int k, rc;
            CHECK(rig_make(&r, rows[i].n, rows[i].fam) == 0, "guards: wrong-N setup row %zu", i);
            if (!r.s) continue;
            for (k = 0; k < NP; ++k) d[k] = quiet();
            make_input(&in, &seed, NP);
            gen = ulcnet_stream_state_generation(r.s);
            if (rows[i].n == 1) {
                rc = rows[i].fam == FAM_PCM ? ulcnet_stream_pre_process_packet_f32(r.s, in.epcm, in.fpcm, d, &io)
                                            : ulcnet_stream_pre_process_packet_freq(r.s, in.er, in.ei, in.fr, in.fi, d, &io);
            } else {
                rc = rows[i].fam == FAM_PCM ? ulcnet_stream_pre_process_f32(r.s, in.epcm, in.fpcm, d[0], &io)
                                            : ulcnet_stream_pre_process_freq(r.s, in.er, in.ei, in.fr, in.fi, d[0], &io);
            }
            CHECK(rc == -1, "guards: an entry of the other N must be refused (n=%d fam=%d, rc=%d)", rows[i].n, rows[i].fam, rc);
            if (rows[i].n == 1 && rows[i].fam == FAM_FREQ_FULL)
                CHECK(ulcnet_stream_pre_process_packet_f32(r.s, in.epcm, in.fpcm, d, &io) == -1, "guards: packet f32 on N = 1");
            CHECK(ulcnet_stream_state_generation(r.s) == gen, "guards: the refused entry moved the generation");
            CHECK(pre_call(&r, &in, d, &io) == 1, "guards: the stream takes its own entry after the refusal (n=%d fam=%d)", rows[i].n, rows[i].fam);
            CHECK(post_call(&r, STREAM_RUN_NOT_STARTED, re, im, pcm) == 1, "guards: its own hop closes");
            rig_free(&r);
        }
    }
    {   /* NULL arguments change nothing */
        Rig r;
        Input in;
        UlcnetHopIO io;
        UlcnetStreamDelay d = quiet();
        uint32_t seed = 5u;
        float re[BINS], im[BINS];
        CHECK(rig_make(&r, 1, FAM_FREQ_ONLY) == 0, "guards: NULL setup");
        make_input(&in, &seed, 1);
        CHECK(ulcnet_stream_pre_process_freq(NULL, in.er, in.ei, in.fr, in.fi, d, &io) == -1 &&
              ulcnet_stream_pre_process_freq(r.s, NULL, in.ei, in.fr, in.fi, d, &io) == -1 &&
              ulcnet_stream_pre_process_freq(r.s, in.er, in.ei, in.fr, NULL, d, &io) == -1 &&
              ulcnet_stream_pre_process_freq(r.s, in.er, in.ei, in.fr, in.fi, d, NULL) == -1, "guards: NULL pre arguments");
        CHECK(ulcnet_stream_post_process_freq(r.s, 0, NULL, im) == -1 && ulcnet_stream_post_process_freq(r.s, 0, re, NULL) == -1 &&
              ulcnet_stream_post_process_freq(NULL, 0, re, im) == -1, "guards: NULL post arguments");
        CHECK(ulcnet_stream_pre_process_freq(r.s, in.er, in.ei, in.fr, in.fi, d, &io) == 1, "guards: still usable after the refusals");
        rig_free(&r);
    }
}

/* ---- the reprime hop ------------------------------------------------------------------------------------- */

static void row_reprime(void) {
    static const int statuses[] = { STREAM_RUN_OK, STREAM_RUN_FAILED, 777, STREAM_RUN_NOT_STARTED };
    int fam;
    size_t i;
    for (fam = FAM_FREQ_ONLY; fam <= FAM_FREQ_FULL; ++fam) {
        Rig r;
        uint32_t seed = 21u;
        CHECK(rig_make(&r, 1, fam) == 0, "reprime: setup");
        for (i = 0; r.s && i < sizeof statuses / sizeof statuses[0]; ++i) {
            Input in;
            UlcnetHopIO io, zero;
            UlcnetStreamDelay d = { 1, 1 };
            float re[BINS], im[BINS];
            uint32_t gen = ulcnet_stream_state_generation(r.s);
            make_input(&in, &seed, 1);
            memset(&io, 0x5A, sizeof io);
            memset(&zero, 0, sizeof zero);
            CHECK(ulcnet_stream_pre_process_freq(r.s, in.er, in.ei, in.fr, in.fi, d, &io) == 0 && memcmp(&io, &zero, sizeof io) == 0,
                  "reprime: an edge hop returns 0 and zeroes io");
            CHECK(ulcnet_stream_state_generation(r.s) == gen + 1u, "reprime: the edge restarts the model once");
            /* no job ran: whatever status the caller reports, the hop is the error spectrum and nothing moves */
            CHECK(ulcnet_stream_post_process_freq(r.s, statuses[i], re, im) == 1 && same(re, in.er, BINS) && same(im, in.ei, BINS) &&
                  ulcnet_stream_state_generation(r.s) == gen + 1u,
                  "reprime: ai_rc %d is ignored, the hop is the error spectrum", statuses[i]);
            d.changed = 0;
            make_input(&in, &seed, 1);
            CHECK(ulcnet_stream_pre_process_freq(r.s, in.er, in.ei, in.fr, in.fi, d, &io) == 1 &&
                  io.state_generation == gen + 1u, "reprime: the hop after it opens on the new generation");
            CHECK(run_kind(&io, K_GOOD, 1, 0) == STREAM_RUN_OK && ulcnet_stream_post_process_freq(r.s, STREAM_RUN_OK, re, im) == 1 &&
                  !same(re, in.er, BINS) && ulcnet_stream_state_generation(r.s) == gen + 1u,
                  "reprime: the hop after it is enhanced");
        }
        rig_free(&r);
    }
}

/* ---- outcomes of a closed call --------------------------------------------------------------------------- */

typedef struct Outcome {
    const char *name;
    int kind;
    int event;        /* packet event frame, -1: none */
    int identity;     /* 1: every frame is the error spectrum, 0: the frames that are not events are enhanced */
    int steps;        /* generation steps of the call */
} Outcome;

static void row_outcomes(int n, int fam, const Outcome *rows, size_t count) {
    size_t i;
    for (i = 0; i < count; ++i) {
        Rig r;
        Input in;
        UlcnetHopIO io;
        UlcnetStreamDelay d[NP];
        float re[PLANE_MAX], im[PLANE_MAX];
        int16_t pcm[NP * HOP];
        uint32_t seed = 31u + (uint32_t)i, gen;
        int k, ai, hop;
        const Outcome *o = &rows[i];
        CHECK(rig_make(&r, n, fam) == 0, "%s: setup", o->name);
        if (!r.s) continue;
        for (hop = 0; hop < 3; ++hop) {   /* a good hop, then the faulty one, then a good one: the fault does not outlive its call */
            const int bad = hop == 1;
            int frames_ok = 1;
            for (k = 0; k < NP; ++k) { d[k] = quiet(); }
            if (bad && o->event >= 0) d[o->event].changed = 1;
            make_input(&in, &seed, n);
            gen = ulcnet_stream_state_generation(r.s);
            CHECK(pre_call(&r, &in, d, &io) == 1, "%s: hop %d did not open (n=%d)", o->name, hop, n);
            if (bad && o->event >= 0 && n > 1)
                CHECK(io.inputs.frame_reset && io.inputs.frame_reset[o->event] == 1.0f, "%s: frame_reset of the event frame", o->name);
            CHECK(io.state_generation == ulcnet_stream_state_generation(r.s), "%s: io generation", o->name);
            ai = run_kind(&io, bad ? o->kind : K_GOOD, n, o->event);
            CHECK(post_call(&r, ai, re, im, pcm) == 1, "%s: post hop %d", o->name, hop);
            CHECK(ulcnet_stream_state_generation(r.s) == gen + (bad ? (uint32_t)o->steps : 0u),
                  "%s (n=%d fam=%d): generation %u -> %u, expected +%d", o->name, n, fam, (unsigned)gen,
                  (unsigned)ulcnet_stream_state_generation(r.s), bad ? o->steps : 0);
            if (fam == FAM_PCM) continue;
            for (k = 0; k < n; ++k) {
                const int is_event = bad && o->event == k;
                const int ident = same(re + k * BINS, in.er + k * BINS, BINS) && same(im + k * BINS, in.ei + k * BINS, BINS);
                const int want_ident = bad ? (o->identity || is_event) : 0;
                if (ident != want_ident) frames_ok = 0;
            }
            CHECK(frames_ok, "%s (n=%d fam=%d): frames of hop %d are not as expected (identity %d)", o->name, n, fam, hop,
                  bad ? o->identity : 0);
        }
        rig_free(&r);
    }
}

/* ---- range ----------------------------------------------------------------------------------------------- */

typedef struct Twin { uint64_t h; } Twin;
static void fold(Twin *t, const void *p, size_t bytes) {
    const unsigned char *b = (const unsigned char *)p;
    size_t i;
    for (i = 0; i < bytes; ++i) { t->h ^= b[i]; t->h *= 1099511628211ull; }
}

/* A scripted run of 40 calls (events, failures, a reset) hashed; `refuse_at` >= 0 offers a bad input before every call
 * that is a multiple of it, with an event on it, through the same entry. Returns the number of refusals seen. */
static int script(int n, int fam, int refuse_at, int bad_kind, Twin *t) {
    Rig r;
    Input in, bad;
    UlcnetHopIO io;
    UlcnetStreamDelay d[NP];
    float re[PLANE_MAX], im[PLANE_MAX];
    int16_t pcm[NP * HOP];
    uint32_t seed = 77u;
    int c, k, refused = 0;
    t->h = 1469598103934665603ull;
    if (rig_make(&r, n, fam) != 0) return -1;
    for (c = 0; c < 40; ++c) {
        int ai, pre;
        uint32_t gen;
        make_input(&in, &seed, n);
        for (k = 0; k < NP; ++k) { d[k].changed = (c + k) % 13 == 5; d[k].solid = !((c + k) % 17 == 9); }
        if (c == 25) ulcnet_stream_reset(r.s);
        if (refuse_at > 0 && c % refuse_at == 1) {
            UlcnetStreamDelay dd[NP];
            UlcnetHopIO junk;
            int rc;
            bad = in;
            for (k = 0; k < NP; ++k) { dd[k].changed = 1; dd[k].solid = 0; }
            switch (bad_kind) {
            case 0: bad.epcm[n * HOP - 3] = NAN; bad.er[n * BINS - 2] = NAN; break;
            case 1: bad.fpcm[1] = INFINITY; bad.fi[1] = -INFINITY; break;
            case 2: bad.epcm[0] = nextafterf(ldexpf(1.0f, 112), INFINITY); bad.fr[n * BINS - 1] = INFINITY; break;
            default: bad.ei[2] = NAN; bad.epcm[2] = -INFINITY; break;
            }
            gen = ulcnet_stream_state_generation(r.s);
            memset(&junk, 0x5A, sizeof junk);
            rc = pre_call(&r, &bad, dd, &junk);
            if (rc == ULCNET_STREAM_ERR_RANGE) ++refused;
            else printf("FAIL: the bad input of call %d was not refused (rc %d)\n", c, rc), ++g_fail;
            if (ulcnet_stream_state_generation(r.s) != gen || post_call(&r, 0, re, im, pcm) != -1) { printf("FAIL: a refusal moved the stream\n"); ++g_fail; }
        }
        pre = pre_call(&r, &in, d, &io);
        fold(t, &pre, sizeof pre);
        gen = ulcnet_stream_state_generation(r.s);
        fold(t, &gen, sizeof gen);
        if (pre == 1) {
            fold(t, &io.state_generation, sizeof io.state_generation);
            fold(t, io.inputs.error_mag, (size_t)n * BINS * sizeof(float));
            fold(t, io.inputs.error_cos, (size_t)n * BINS * sizeof(float));
            ai = run_kind(&io, c % 11 == 3 ? K_FAILED : c % 7 == 2 ? K_NAN : K_GOOD, n, 0);
        } else {
            ai = STREAM_RUN_OK;
        }
        if (post_call(&r, ai, re, im, pcm) != 1) { printf("FAIL: script post\n"); ++g_fail; }
        if (fam == FAM_PCM) fold(t, pcm, (size_t)n * HOP * sizeof(int16_t));
        else { fold(t, re, (size_t)n * BINS * sizeof(float)); fold(t, im, (size_t)n * BINS * sizeof(float)); }
        gen = ulcnet_stream_state_generation(r.s);
        fold(t, &gen, sizeof gen);
    }
    rig_free(&r);
    return refused;
}

static void row_range_twin(void) {
    static const struct { int n, fam; const char *name; } cfgs[] = {
        { 1, FAM_PCM, "f32 entry" }, { 1, FAM_FREQ_ONLY, "freq entry, freq_only" }, { 1, FAM_FREQ_FULL, "freq entry, full" },
        { NP, FAM_PCM, "packet f32 entry" }, { NP, FAM_FREQ_ONLY, "packet freq entry, freq_only" },
        { NP, FAM_FREQ_FULL, "packet freq entry, full" },
    };
    size_t i;
    int bad_kind;
    for (i = 0; i < sizeof cfgs / sizeof cfgs[0]; ++i)
        for (bad_kind = 0; bad_kind < 4; ++bad_kind) {
            Twin clean, dirty, again;
            const int refused = script(cfgs[i].n, cfgs[i].fam, 3, bad_kind, &dirty);
            script(cfgs[i].n, cfgs[i].fam, 0, bad_kind, &clean);
            script(cfgs[i].n, cfgs[i].fam, 0, bad_kind, &again);
            CHECK(refused >= 10, "range: %s kind %d: only %d refusals offered", cfgs[i].name, bad_kind, refused);
            CHECK(clean.h == again.h, "range: %s: the script is not deterministic", cfgs[i].name);
            CHECK(clean.h == dirty.h, "range: %s kind %d: the stream that saw the refusals differs from the twin that did not",
                  cfgs[i].name, bad_kind);
        }
}

/* The limits are inclusive: 2^112 for PCM, FLT_MAX for a bin; one ulp above, NaN, +-Inf are refused, on every entry. */
static void row_range_limits(void) {
    static const struct { int fam; int pcm; uint32_t bits; int accepted; const char *name; } rows[] = {
        { FAM_PCM, 1, ULCNET_STREAM_PCM_LIMIT_BITS, 1, "pcm at 2^112" },
        { FAM_PCM, 1, ULCNET_STREAM_PCM_LIMIT_BITS + 1u, 0, "pcm one ulp above 2^112" },
        { FAM_PCM, 1, 0xFF800000u, 0, "pcm -Inf" },
        { FAM_PCM, 1, 0xFFC00001u, 0, "pcm NaN" },
        { FAM_PCM, 1, ULCNET_STREAM_PCM_LIMIT_BITS | 0x80000000u, 1, "pcm at -2^112" },
        { FAM_FREQ_ONLY, 0, ULCNET_STREAM_BIN_LIMIT_BITS, 1, "bin at FLT_MAX" },
        { FAM_FREQ_ONLY, 0, 0x7F800000u, 0, "bin +Inf" },
        { FAM_FREQ_ONLY, 0, 0x7FC00000u, 0, "bin NaN" },
        { FAM_FREQ_ONLY, 0, ULCNET_STREAM_BIN_LIMIT_BITS | 0x80000000u, 1, "bin at -FLT_MAX" },
    };
    size_t i;
    int n, plane_of;
    for (n = 1; n <= NP; n += NP - 1)
        for (i = 0; i < sizeof rows / sizeof rows[0]; ++i) {
            for (plane_of = 0; plane_of < 2; ++plane_of) {   /* the value in the error input, then in the far input */
                Rig r;
                Input in;
                UlcnetHopIO io;
                UlcnetStreamDelay d[NP];
                float v, re[PLANE_MAX], im[PLANE_MAX];
                int16_t pcm[NP * HOP];
                uint32_t seed = 3u;
                int k, rc;
                const int last = rows[i].pcm ? n * HOP - 1 : n * BINS - 1;
                memcpy(&v, &rows[i].bits, sizeof v);
                CHECK(rig_make(&r, n, rows[i].fam) == 0, "limits: setup");
                for (k = 0; k < NP; ++k) d[k] = quiet();
                make_input(&in, &seed, n);
                if (rows[i].pcm) { (plane_of ? in.fpcm : in.epcm)[last] = v; }
                else { (plane_of ? in.fi : in.er)[last] = v; }
                rc = pre_call(&r, &in, d, &io);
                if (rows[i].accepted) {
                    CHECK(rc == 1, "limits (n=%d): %s was refused (rc %d)", n, rows[i].name, rc);
                    if (rc == 1) CHECK(post_call(&r, STREAM_RUN_NOT_STARTED, re, im, pcm) == 1, "limits: post");
                } else {
                    CHECK(rc == ULCNET_STREAM_ERR_RANGE, "limits (n=%d): %s was accepted (rc %d)", n, rows[i].name, rc);
                    CHECK(post_call(&r, 0, re, im, pcm) == -1, "limits: a refused entry left a hop open");
                }
                rig_free(&r);
            }
        }
}

int main(void) {
    static const Outcome one[] = {
        { "good mask", K_GOOD, -1, 0, 0 }, { "unwritten mask", K_UNWRITTEN, -1, 1, 1 }, { "NaN mask bin", K_NAN, -1, 1, 1 },
        { "overflowing enhanced bin", K_HUGE_LAST, -1, 1, 1 }, { "run never started", K_NOT_STARTED, -1, 1, 0 },
        { "run failed", K_FAILED, -1, 1, 1 }, { "unknown status", K_ODD, -1, 1, 1 },
    };
    static const Outcome packet[] = {
        { "packet good mask", K_GOOD, -1, 0, 0 }, { "packet unwritten mask", K_UNWRITTEN, -1, 1, 1 },
        { "packet NaN mask bin", K_NAN, -1, 1, 1 },
        { "packet overflow in the last frame: all frames identity, one restart", K_HUGE_LAST, -1, 1, 1 },
        { "packet two overflowing frames: one restart", K_HUGE_TWO, -1, 1, 1 },
        { "packet run never started", K_NOT_STARTED, -1, 1, 0 }, { "packet run failed", K_FAILED, -1, 1, 1 },
        { "packet event, good mask: event frame is the error spectrum", K_GOOD, 2, 0, 0 },
        { "packet event, huge mask in the event frame: not applied, not scanned for overflow", K_HUGE_EVENT, 2, 0, 0 },
        { "packet event, NaN mask in the event frame: refused", K_NAN_EVENT, 2, 1, 1 },
        { "packet event, run never started: the graph restart never ran, +1", K_NOT_STARTED, 2, 1, 1 },
        { "packet event, run failed: once, not twice", K_FAILED, 2, 1, 1 },
        { "packet event, overflow in the last frame: once", K_HUGE_LAST, 2, 1, 1 },
    };
    int fam;
    row_guards();
    row_reprime();
    for (fam = FAM_FREQ_ONLY; fam <= FAM_FREQ_FULL; ++fam) {
        row_outcomes(1, fam, one, sizeof one / sizeof one[0]);
        row_outcomes(NP, fam, packet, sizeof packet / sizeof packet[0]);
    }
    row_outcomes(1, FAM_PCM, one, sizeof one / sizeof one[0]);   /* the PCM family: every call completes */
    row_outcomes(NP, FAM_PCM, packet, sizeof packet / sizeof packet[0]);
    row_range_twin();
    row_range_limits();
    if (g_fail) { printf("lifecycle: %d check(s) FAILED\n", g_fail); return 1; }
    printf("lifecycle: guards, reprime, outcomes (one hop, packet, event), range twin and limits -- PASS\n");
    return 0;
}
