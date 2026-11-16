/* Run-status table of dfn2_stream_post_process() and dfn2_stream_post_process_packet(): each status x the first job
 * after a start and a steady job at N == 1 (the first mask_lookahead hops have no job), and every packet of N > 1
 * (one job each). A stand-in accelerator follows the
 * documented protocol (zero its state when io.state_generation changed); the test compares it with what the rules predict:
 *   STREAM_RUN_OK            the run advanced the state;
 *   STREAM_RUN_NOT_STARTED   nothing ran: the state and the generation stay;
 *   any other value          the run may have corrupted the state: it is zeroed before the next run, generation + 1.
 * A hop without a job ignores the status and moves neither the state nor the generation.
 * The failed hop's PCM is the bypass output, identical to a NOT_STARTED hop's. */

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_stream.h"
#include "erb_fixture.h"

static int g_failures;
#define CHECK(c, ...) do { if (!(c)) { ++g_failures; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
                                       fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); } } while (0)

typedef struct Device {
    int          state;      /* runs since the last zeroing; 1000 after a run that failed half way */
    unsigned int seen;       /* generation at the last zeroing */
} Device;

/* Heads from the device state; the run advances it. `status` != 0: the run fails after moving the state, except
 * STREAM_RUN_NOT_STARTED, which touches nothing. */
static int device_run(Device *dev, const DFN2HopIO *io, int status) {
    DFN2PrepostOutputs out = io->outputs;
    size_t i;
    if (status == STREAM_RUN_NOT_STARTED) return status;
    for (i = 0; i < out.erb_mask_elements; ++i) out.erb_mask[i] = 0.1f + 0.001f * (float)((dev->state + (int)i) % 97);
#ifndef DFN2_MASK_ONLY
    for (i = 0; i < out.coefs_elements; ++i) out.coefs[i] = 0.002f * (float)((dev->state + (int)i) % 13);
    for (i = 0; i < out.alpha_elements; ++i) out.alpha[i] = 0.2f;
#endif
    if (status != 0) { dev->state = 1000; return status; }
    ++dev->state;
    return 0;
}

typedef struct Stream {
    DFN2Stream *s;
    void *pool;
    Device dev;
    int expect;              /* the state the rules predict */
} Stream;

static int stream_open(Stream *st, const DFN2Grid *grid, const float *fwd, const float *inv) {
    DFN2StreamConfig cfg;
    DFN2StreamMemReq req;
    memset(st, 0, sizeof *st);
    memset(&cfg, 0, sizeof cfg);
    cfg.grid = *grid; cfg.erb_fwd = fwd; cfg.erb_inv = inv;
    if (dfn2_stream_get_mem_size(&cfg, &req) != 0) return -1;
    if (posix_memalign(&st->pool, (size_t)req.alignment, (size_t)req.bytes) != 0) return -1;
    st->s = dfn2_stream_init_ex(&cfg, st->pool, (size_t)req.bytes, &req);
    if (!st->s) return -1;
    st->dev.seen = dfn2_stream_state_generation(st->s);
    return 0;
}

/* One call: a hop (N == 1) or a packet of N hops. job_status: the status the accelerator reports if the call
 * carries a job (a non-job hop is given STREAM_RUN_FAILED, which must be ignored). Returns the job's ordinal
 * (0-based) or -1 for a hop without a job. */
static int hop_once(Stream *st, const DFN2Grid *grid, int hop_index, int job_ordinal, int fail_job, int fail_status,
                    int16_t *pcm_out, unsigned int *gen_before, unsigned int *gen_after) {
    int16_t pcm_in[1024];
    DFN2HopIO io;
    const int packet = grid->frames_per_call > 1;
    int k, status = 0, is_job;
    for (k = 0; k < grid->hop * grid->frames_per_call; ++k)
        pcm_in[k] = (int16_t)((((long long)k * 7 + hop_index * 3) % 101 - 50) * 100);
    *gen_before = dfn2_stream_state_generation(st->s);
    CHECK((packet ? dfn2_stream_pre_process_packet(st->s, pcm_in, &io) : dfn2_stream_pre_process(st->s, pcm_in, &io)) == 1,
          "pre_process at hop %d", hop_index);
    is_job = io.action == STREAM_WORK_SUBMIT;
    if (is_job) {
        int fail = (job_ordinal == fail_job);
        if (io.state_generation != st->dev.seen) { st->dev.state = 0; st->dev.seen = io.state_generation; }
        CHECK(st->dev.state == st->expect, "job %d: device state %d, rules predict %d", job_ordinal, st->dev.state, st->expect);
        status = device_run(&st->dev, &io, fail ? fail_status : 0);
    } else {
        status = STREAM_RUN_FAILED;
    }
    CHECK((packet ? dfn2_stream_post_process_packet(st->s, status, pcm_out) : dfn2_stream_post_process(st->s, status, pcm_out)) >= 0,
          "post_process at hop %d", hop_index);
    *gen_after = dfn2_stream_state_generation(st->s);
    return is_job ? job_ordinal : -1;
}

static void run_case(const DFN2GridSpec *spec, const char *label, int fail_status) {
    DFN2Grid grid;
    Stream a, b;   /* a: the status under test; b: STREAM_RUN_NOT_STARTED at the same job */
    float *fwd, *inv;
    int L, N, fail_job, hops, h, jobs_a = 0, jobs_b = 0, span;
    int16_t pcm_a[1024], pcm_b[1024];
    int differs_after = 0;

    if (dfn2_grid_make(&grid, spec) != DFN2_GRID_OK) { CHECK(0, "[%s] grid", label); return; }
    L = grid.mask_lookahead; N = grid.frames_per_call;
    span = grid.hop * N;   /* samples of one call */
    fwd = malloc((size_t)grid.n_bins * (size_t)grid.n_erb * sizeof(float));
    inv = malloc((size_t)grid.n_bins * (size_t)grid.n_erb * sizeof(float));
    {
        size_t i;
        for (i = 0; i < (size_t)grid.n_bins * (size_t)grid.n_erb; ++i) fwd[i] = 1.0f / (float)grid.n_erb;
        erb_fixture_unit_inv(inv, grid.n_bins, grid.n_erb);
    }
    if (stream_open(&a, &grid, fwd, inv) != 0 || stream_open(&b, &grid, fwd, inv) != 0) { CHECK(0, "[%s] open", label); return; }
    hops = N == 1 ? L + 10 : 10;   /* calls */
    /* Fail a steady job after two good runs, then (after a reset) the first job. */
    for (int pass = 0; pass < 2; ++pass) {
        fail_job = pass == 0 ? 2 : 0;
        if (pass == 1) {   /* back to the start: the first L hops have no job again */
            dfn2_stream_reset(a.s); dfn2_stream_reset(b.s);
            a.expect = b.expect = 0; jobs_a = jobs_b = 0;
        }
        for (h = 0; h < hops; ++h) {
            unsigned int ga0, ga1, gb0, gb1;
            int ja, jb, fail_hop;
            ja = hop_once(&a, &grid, h, jobs_a, fail_job, fail_status, pcm_a, &ga0, &ga1);
            jb = hop_once(&b, &grid, h, jobs_b, fail_job, STREAM_RUN_NOT_STARTED, pcm_b, &gb0, &gb1);
            CHECK((ja >= 0) == (jb >= 0), "[%s] job cadence differs", label);
            if (ja < 0) {   /* no job: the status is ignored and nothing moves */
                CHECK(ga1 == ga0 && gb1 == gb0, "[%s] hop %d: generation moved on a hop without a job", label, h);
                continue;
            }
            fail_hop = (ja == fail_job);
            /* the rules */
            if (fail_hop) {
                int dirty = fail_status != 0 && fail_status != STREAM_RUN_NOT_STARTED;
                CHECK(ga1 == ga0 + (dirty ? 1u : 0u), "[%s] status %d: generation %u -> %u, want %s", label, fail_status, ga0, ga1, dirty ? "+1" : "unchanged");
                CHECK(gb1 == gb0, "[%s] NOT_STARTED: generation %u -> %u, want unchanged", label, gb0, gb1);
                a.expect = dirty ? 0 : a.expect + (fail_status == 0 ? 1 : 0);   /* b: nothing ran, unchanged */
                if (fail_status != 0)
                    CHECK(memcmp(pcm_a, pcm_b, (size_t)span * sizeof(int16_t)) == 0,
                          "[%s] status %d: failed hop's PCM is not the bypass output", label, fail_status);
            } else {
                CHECK(ga1 == ga0 && gb1 == gb0, "[%s] job %d ok: generation moved", label, ja);
                ++a.expect; ++b.expect;
            }
            if (ja > fail_job && memcmp(pcm_a, pcm_b, (size_t)span * sizeof(int16_t)) != 0) differs_after = 1;
            jobs_a = ja + 1; jobs_b = jb + 1;
        }
    }
    /* A dirty status zeroes the state and NOT_STARTED keeps it, so the runs after the failed one differ. */
    {
        int dirty = fail_status != 0 && fail_status != STREAM_RUN_NOT_STARTED;
        if (dirty) CHECK(differs_after, "[%s] status %d: later output equals the NOT_STARTED stream's", label, fail_status);
    }
    free(a.pool); free(b.pool); free(fwd); free(inv);
}

int main(void) {
    static const int statuses[] = { 0, STREAM_RUN_NOT_STARTED, STREAM_RUN_FAILED, 1, 7, STREAM_RUN_NOT_STARTED + 1,
                                    STREAM_RUN_NOT_STARTED - 1, INT_MAX, INT_MIN };
    DFN2GridSpec n1, w1, n8;
    size_t i;
    memset(&n1, 0, sizeof n1); memset(&w1, 0, sizeof w1); memset(&n8, 0, sizeof n8);
    n1.sample_rate = 16000; n1.n_fft = 256; n1.n_erb = 24; n1.df_bins = 96; n1.frames_per_call = 1;       /* L0, N1 */
    w1.sample_rate = 48000; w1.n_fft = 1024; w1.n_erb = 32; w1.df_bins = 96; w1.mask_lookahead = 1; w1.frames_per_call = 1;  /* L1 */
    n8 = n1; n8.frames_per_call = 8;                                                                       /* packets */
#ifdef DFN2_MASK_ONLY
    n1.df_order = n8.df_order = w1.df_order = 0; w1.df_lookahead = 0;
#else
    n1.df_order = n8.df_order = w1.df_order = 5; w1.df_lookahead = 1;
#endif
    for (i = 0; i < sizeof statuses / sizeof statuses[0]; ++i) {
        run_case(&n1, "16k N=1 L=0", statuses[i]);
        run_case(&w1, "48k N=1 L=1", statuses[i]);
        run_case(&n8, "16k N=8 L=0", statuses[i]);
    }
    printf("fail_table: %zu statuses x 3 grids (N = 1 L = 0, N = 1 L = 1, N = 8 packets) -- %s\n", sizeof statuses / sizeof statuses[0],
           g_failures ? "FAIL" : "PASS");
    return g_failures ? 1 : 0;
}
