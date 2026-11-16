"""DeepVQE-S: the CCM ring restarts with the state generation.

``DeepVqePrepost`` keeps the last TIME_ORDER raw microphone spectra in a ring
that the CCM taps convolve over. Whenever the state generation advances (reset,
a commit refused for a non-finite value, a failed call with a started run) the
ring is emptied together with the registered state, so the next frame equals the
Python model's cold start: an empty ring and a zero state. A bypassed call and
a failed call whose run never started do not advance the generation and keep
the ring. The per-call order is the stream's, run by ``deepvqe_lifecycle_driver.h``
(the ``dvl_*`` calls) over the per-frame functions.

Every equality below compares the frames after the restart with a freshly
initialised instance fed the same frames (memcmp on fp32), and is paired with a
check that the same frames after NO restart differ, so no case passes on a ring
that does not matter. The accelerator stand-in is state dependent and uses every
CCM tap, so a ring that kept an old spectrum shows in the output.

Cases of the one driver (argv: case, grid=<sample rate>,<n_fft>, n=<frames per call>):

  refused    commit, a commit refused for a non-finite state value, commit: the
             last two batches equal a fresh instance fed them
  fail       the same with frame_fail(p, 1) after a completed run
  notstarted frame_fail(p, 0) for a run that never started keeps the ring and
             the state: equal to the same frames without that batch, unequal
             to a fresh instance
  board      nothing registered (the accelerator owns the state): the ring
             alone makes the difference, restart and keep both shown
  reset      reset is still a cold start
  stream     the _freq entries of deepvqe_stream: a failed, an odd and a
             refused run restart; NOT_STARTED does not
"""

import os
import shutil
import subprocess

import pytest


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AIAEC_DIR = os.path.dirname(_THIS_DIR)
_ROOT = os.path.join(_AIAEC_DIR, '..')
_DEEPVQE_DIR = os.path.join(_ROOT, 'lib', 'aiaec', 'deepvqe_s')
_STREAM_DIR = os.path.join(_ROOT, 'stream', 'aiaec', 'deepvqe_s')
_COMMON_DIR = os.path.join(_ROOT, 'stream', 'common')
_AC_DIR = os.path.abspath(
    os.path.join(_THIS_DIR, '..', '..', '..', 'audiolib', 'common'))
_AC_INCLUDE = os.path.join(_AC_DIR, 'include')

_SOURCES = (
    os.path.join(_DEEPVQE_DIR, 'deepvqe_prepost.c'),
    os.path.join(_DEEPVQE_DIR, 'deepvqe_process.c'),
    os.path.join(_STREAM_DIR, 'deepvqe_stream.c'),
)

_GRID_16K = 'grid=16000,512'
_GRID_48K = 'grid=48000,1024'


_DRIVER = r'''
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "deepvqe_lifecycle_driver.h"
#include "deepvqe_stream.h"
#include "stream_run.h"

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

#define BINS_MAX  513
#define N_MAX     8
#define ELEMS     8          /* registered state elements per tensor (the library knows only the count) */
#define TAPS_MAX  (N_MAX * BINS_MAX * DEEPVQE_TIME_ORDER * DEEPVQE_FREQ_TAPS * 2)

static DeepVqeGrid g_grid;
#define BINS ((size_t)g_grid.n_bins)
#define N    (g_grid.frames_per_call)

typedef struct Inst {
    DeepVqePrepost *p;
    DeepVqePrepostConfig cfg;
    void *pool;
    float *buf[DEEPVQE_STATE_COUNT];
    int has_state;
} Inst;

static int inst_make(Inst *in, int has_state) {
    DeepVqePrepostMemReq req;
    int id, k;
    memset(in, 0, sizeof *in);
    in->has_state = has_state;
    if (deepvqe_prepost_config_defaults(&in->cfg, &g_grid) != 0) return -1;
    for (id = 0; has_state && id < DEEPVQE_STATE_COUNT; ++id) {
        in->buf[id] = (float *)malloc(ELEMS * sizeof(float));
        if (!in->buf[id]) return -1;
        for (k = 0; k < ELEMS; ++k) in->buf[id][k] = NAN;
        in->cfg.state[id] = in->buf[id];
        in->cfg.state_elements[id] = ELEMS;
    }
    if (deepvqe_prepost_get_mem_size(&in->cfg, &req) != 0) return -1;
    if (posix_memalign(&in->pool, req.alignment, (size_t)req.bytes) != 0) return -1;
    in->p = deepvqe_prepost_init_ex(in->pool, (size_t)req.bytes, &in->cfg, &req);
    return in->p ? 0 : -1;
}

static void inst_free(Inst *in) {
    int id;
    dvl_forget(in->p);
    free(in->pool);
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) free(in->buf[id]);
}

/* The stand-in for the accelerator: every tap is nonzero and follows the current frames and, when a state is
 * registered, all sixteen state tensors (read whole before any is written); the state is written in place. */
static void run_device(const DeepVqePrepostInputs *in, DeepVqePrepostOutputs *out) {
    double acc = 0.0;
    size_t i;
    int id;
    for (i = 0; i < in->spectrum_ri_elements; ++i) acc += (double)in->mic[i] + 0.5 * (double)in->far[i];
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        for (i = 0; i < in->state_elements[id]; ++i) acc += 1e-2 * (double)in->state[id][i];
    for (i = 0; i < out->taps_elements; ++i) out->taps[i] = (float)(0.2 * sin(acc + 0.37 * (double)i));
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id)
        for (i = 0; i < out->state_elements[id]; ++i)
            out->state_out[id][i] = (float)(0.5 * sin(acc + (double)id + 1e-2 * (double)i));
}

/* Frame `idx` of the test signal: mic and far spectra, planar. */
static void make_frame(long idx, float *mr, float *mi, float *fr, float *fi) {
    size_t b;
    for (b = 0; b < BINS; ++b) {
        mr[b] = (float)(0.8 * sin(0.37 * (double)b + 1.3 * (double)idx) + 0.1 * (double)(idx % 5));
        mi[b] = (float)(0.6 * cos(0.21 * (double)b - 0.9 * (double)idx));
        fr[b] = (float)(0.5 * sin(0.11 * (double)b + 0.7 * (double)idx + 1.0));
        fi[b] = (float)(0.4 * cos(0.29 * (double)b + 0.4 * (double)idx));
    }
}

typedef enum Mode {
    M_COMMIT,      /* run, finish */
    M_REFUSE,      /* run, a non-finite value, finish refuses, bypass */
    M_FAIL_RUN,    /* run, frame_fail(p, 1) */
    M_FAIL_IDLE,   /* no run, frame_fail(p, 0) */
    M_BYPASS_RUN   /* run, bypass */
} Mode;

typedef struct Step { long batch; Mode mode; } Step;

typedef struct Out { float re[N_MAX][BINS_MAX], im[N_MAX][BINS_MAX]; } Out;

/* One batch of N frames (batch b = frames b*N .. b*N+N-1) through the class. */
static int do_batch(Inst *in, long batch, Mode mode, Out *out) {
    static float mr[BINS_MAX], mi[BINS_MAX], fr[BINS_MAX], fi[BINS_MAX];
    DeepVqePrepostInputs inputs;
    DeepVqePrepostOutputs outputs;
    int k, rc = 0;
    for (k = 0; k < N; ++k) {
        make_frame(batch * N + k, mr, mi, fr, fi);
        rc = dvl_pre_process_freq(in->p, mr, mi, fr, fi);
        CHECK(rc == (k == N - 1 ? 1 : 0));
    }
    CHECK(dvl_frame_open(in->p, &inputs, &outputs) == 0);
    if (mode != M_FAIL_IDLE) run_device(&inputs, &outputs);
    if (mode == M_REFUSE) {
        if (in->has_state) outputs.state_out[DEEPVQE_STATE_H_GRU][1] = NAN;
        else outputs.taps[outputs.taps_elements / 3u] = NAN;
    }
    switch (mode) {
    case M_COMMIT:     CHECK(dvl_frame_finish(in->p) == 0); break;
    case M_REFUSE:     CHECK(dvl_frame_finish(in->p) == -1);
                       CHECK(dvl_frame_bypass(in->p) == 0); break;
    case M_FAIL_RUN:   CHECK(dvl_frame_fail(in->p, 1) == 0); break;
    case M_FAIL_IDLE:  CHECK(dvl_frame_fail(in->p, 0) == 0); break;
    case M_BYPASS_RUN: CHECK(dvl_frame_bypass(in->p) == 0); break;
    }
    for (k = 0; k < N; ++k) CHECK(dvl_post_process_freq(in->p, out->re[k], out->im[k]) == 0);
    return 0;
}

static int same_out(const Out *a, const Out *b) {
    int k;
    for (k = 0; k < N; ++k)
        if (memcmp(a->re[k], b->re[k], BINS * sizeof(float)) || memcmp(a->im[k], b->im[k], BINS * sizeof(float))) return 0;
    return 1;
}

static int silent(const Out *a) {
    int k;
    size_t b;
    for (k = 0; k < N; ++k)
        for (b = 0; b < BINS; ++b) if (a->re[k][b] != 0.0f || a->im[k][b] != 0.0f) return 0;
    return 1;
}

static int has_signal(const Out *a) { return !silent(a); }

#define MAX_STEPS 8
static Out g_out[MAX_STEPS];

static int run_steps(Inst *in, const Step *steps, int n) {
    int i;
    for (i = 0; i < n; ++i) if (do_batch(in, steps[i].batch, steps[i].mode, &g_out[i]) != 0) return 1;
    return 0;
}

/* `first` steps leave the instance in some state; `x` then holds the later steps. Returns the outputs of the last
 * `later` steps of X in xs[], those of the fresh Y in ys[]. */
static Out xs[4], ys[4];

static int copy_tail(Out *dst, int count, int from) {
    int i;
    for (i = 0; i < count; ++i) dst[i] = g_out[from + i];
    return 0;
}

/* X: batch 0 commits, batch 1 ends the way `mode` says, batches 2 and 3 commit; Y: a fresh instance fed batches 2 and 3.
 * Returns 0 when X's batches 2 and 3 equal Y's bit for bit. `want_restart` 0 inverts the claim (the pair must differ). */
static int against_fresh(int has_state, Mode mode, int want_restart) {
    Inst x, y;
    const Step sx[4] = { {0, M_COMMIT}, {1, mode}, {2, M_COMMIT}, {3, M_COMMIT} };
    const Step sy[2] = { {2, M_COMMIT}, {3, M_COMMIT} };
    uint32_t gen_x;
    CHECK(inst_make(&x, has_state) == 0 && inst_make(&y, has_state) == 0);
    CHECK(run_steps(&x, sx, 4) == 0);
    copy_tail(xs, 2, 2);
    gen_x = deepvqe_prepost_state_generation(x.p);
    CHECK(run_steps(&y, sy, 2) == 0);
    copy_tail(ys, 2, 0);
    CHECK(has_signal(&xs[0]) && has_signal(&xs[1]));
    CHECK(deepvqe_prepost_state_generation(y.p) == 0u);
    if (want_restart) {
        CHECK(gen_x == 1u);
        CHECK(same_out(&xs[0], &ys[0]) && same_out(&xs[1], &ys[1]));
    } else {
        CHECK(gen_x == 0u);
        CHECK(!same_out(&xs[0], &ys[0]));
    }
    inst_free(&x);
    inst_free(&y);
    return 0;
}

static int case_refused(void) {
    CHECK(against_fresh(1, M_REFUSE, 1) == 0);    /* registered state, a non-finite state value */
    CHECK(against_fresh(1, M_BYPASS_RUN, 0) == 0);   /* a plain bypass: no restart, the frames after it differ from cold */
    return 0;
}

static int case_fail(void) {
    CHECK(against_fresh(1, M_FAIL_RUN, 1) == 0);
    return 0;
}

static int case_board(void) {
    CHECK(against_fresh(0, M_REFUSE, 1) == 0);       /* no state registered: the ring alone */
    CHECK(against_fresh(0, M_FAIL_RUN, 1) == 0);
    CHECK(against_fresh(0, M_BYPASS_RUN, 0) == 0);   /* the same frames with the ring kept differ */
    CHECK(against_fresh(0, M_FAIL_IDLE, 0) == 0);
    return 0;
}

/* frame_fail(p, 0) for a run that never started: neither state nor ring move. X: batch 0, batch 1 opened and failed
 * with no run, batch 2. W: batch 0, batch 2. Equal, and different from a fresh instance fed batch 2. */
static int case_notstarted(void) {
    Inst x, w, y;
    const Step sx[3] = { {0, M_COMMIT}, {1, M_FAIL_IDLE}, {2, M_COMMIT} };
    const Step sw[2] = { {0, M_COMMIT}, {2, M_COMMIT} };
    const Step sy[1] = { {2, M_COMMIT} };
    CHECK(inst_make(&x, 1) == 0 && inst_make(&w, 1) == 0 && inst_make(&y, 1) == 0);
    CHECK(run_steps(&x, sx, 3) == 0);
    xs[0] = g_out[1];
    xs[1] = g_out[2];
    CHECK(run_steps(&w, sw, 2) == 0);
    ys[0] = g_out[1];
    CHECK(run_steps(&y, sy, 1) == 0);
    ys[1] = g_out[0];
    CHECK(silent(&xs[0]));                                  /* the failed batch is silence */
    CHECK(deepvqe_prepost_state_generation(x.p) == 0u);
    CHECK(same_out(&xs[1], &ys[0]));                        /* ring and state kept: as if the batch never ran */
    CHECK(has_signal(&xs[1]) && !same_out(&xs[1], &ys[1])); /* ...which is not a cold start */
    inst_free(&x);
    inst_free(&w);
    inst_free(&y);
    return 0;
}

static int case_reset(void) {
    Inst x, y;
    const Step sy[2] = { {2, M_COMMIT}, {3, M_COMMIT} };
    CHECK(inst_make(&x, 1) == 0 && inst_make(&y, 1) == 0);
    CHECK(do_batch(&x, 0, M_COMMIT, &g_out[0]) == 0);
    dvl_reset(x.p);
    CHECK(do_batch(&x, 2, M_COMMIT, &xs[0]) == 0 && do_batch(&x, 3, M_COMMIT, &xs[1]) == 0);
    CHECK(run_steps(&y, sy, 2) == 0);
    copy_tail(ys, 2, 0);
    CHECK(same_out(&xs[0], &ys[0]) && same_out(&xs[1], &ys[1]));
    inst_free(&x);
    inst_free(&y);
    return 0;
}

/* ---- the _freq entries of the stream (N == 1, spectrum only) ---- */

static float s_buf[2][DEEPVQE_STATE_COUNT][ELEMS];   /* each stream registers its own sixteen tensors */

static DeepVqeStream *stream_make(int which, void *pool_out[1]) {
    DeepVqeStreamConfig cfg;
    DeepVqeStreamMemReq req;
    DeepVqeStream *s;
    int id;
    memset(&cfg, 0, sizeof cfg);
    cfg.grid = g_grid;
    cfg.freq_only = 1;
    for (id = 0; id < DEEPVQE_STATE_COUNT; ++id) {
        cfg.state[id] = s_buf[which][id];
        cfg.state_elements[id] = ELEMS;
    }
    if (deepvqe_stream_get_mem_size(&cfg, &req) != 0) return NULL;
    if (posix_memalign(&pool_out[0], req.alignment, (size_t)req.bytes) != 0) return NULL;
    s = deepvqe_stream_init_ex(&cfg, pool_out[0], (size_t)req.bytes, &req);
    return s;
}

/* One hop through the stream: the device runs unless `status` is STREAM_RUN_NOT_STARTED; `poison` leaves a state value
 * non-finite. */
static int stream_hop(DeepVqeStream *s, long idx, int status, int poison, float *re, float *im) {
    static float mr[BINS_MAX], mi[BINS_MAX], fr[BINS_MAX], fi[BINS_MAX];
    DeepVqeHopIO io;
    make_frame(idx, mr, mi, fr, fi);
    CHECK(deepvqe_stream_pre_process_freq(s, mr, mi, fr, fi, &io) == 1);
    if (status != STREAM_RUN_NOT_STARTED) run_device(&io.inputs, &io.outputs);
    if (poison) io.outputs.state_out[DEEPVQE_STATE_H_GRU][1] = NAN;
    CHECK(deepvqe_stream_post_process_freq(s, status, re, im) == 1);
    return 0;
}

static int stream_against_fresh(int status, int poison, int want_restart) {
    static float xr[2][BINS_MAX], xi[2][BINS_MAX], yr[2][BINS_MAX], yi[2][BINS_MAX], jr[BINS_MAX], ji[BINS_MAX];
    void *px[1] = { NULL }, *py[1] = { NULL };
    DeepVqeStream *x = stream_make(0, px), *y = stream_make(1, py);
    const size_t bytes = BINS * sizeof(float);
    CHECK(x && y);
    CHECK(stream_hop(x, 0, STREAM_RUN_OK, 0, jr, ji) == 0);
    CHECK(stream_hop(x, 1, status, poison, jr, ji) == 0);
    CHECK(stream_hop(x, 2, STREAM_RUN_OK, 0, xr[0], xi[0]) == 0);
    CHECK(stream_hop(x, 3, STREAM_RUN_OK, 0, xr[1], xi[1]) == 0);
    CHECK(stream_hop(y, 2, STREAM_RUN_OK, 0, yr[0], yi[0]) == 0);
    CHECK(stream_hop(y, 3, STREAM_RUN_OK, 0, yr[1], yi[1]) == 0);
    CHECK(deepvqe_stream_state_generation(x) == (want_restart ? 1u : 0u));
    if (want_restart) {
        CHECK(memcmp(xr[0], yr[0], bytes) == 0 && memcmp(xi[0], yi[0], bytes) == 0);
        CHECK(memcmp(xr[1], yr[1], bytes) == 0 && memcmp(xi[1], yi[1], bytes) == 0);
    } else {
        CHECK(memcmp(xr[0], yr[0], bytes) != 0);
    }
    free(px[0]);
    free(py[0]);
    return 0;
}

static int case_stream(void) {
    CHECK(g_grid.frames_per_call == 1);
    CHECK(stream_against_fresh(STREAM_RUN_FAILED, 0, 1) == 0);
    CHECK(stream_against_fresh(7, 0, 1) == 0);
    CHECK(stream_against_fresh(STREAM_RUN_OK, 1, 1) == 0);           /* a refused commit */
    CHECK(stream_against_fresh(STREAM_RUN_NOT_STARTED, 0, 0) == 0);   /* nothing restarts */
    return 0;
}

int main(int argc, char **argv) {
    int sr = 0, n_fft = 0, n = 0;
    DeepVqeGridSpec spec;
    if (argc < 4 || sscanf(argv[2], "grid=%d,%d", &sr, &n_fft) != 2 || sscanf(argv[3], "n=%d", &n) != 1) {
        fprintf(stderr, "usage: %s <case> grid=<sample rate>,<n_fft> n=<frames per call>\n", argv[0]);
        return 2;
    }
    spec.sample_rate = sr; spec.n_fft = n_fft; spec.delay_depth = 8; spec.frames_per_call = n;
    CHECK(deepvqe_grid_make(&g_grid, &spec) == DEEPVQE_GRID_OK);
    CHECK(g_grid.n_bins <= BINS_MAX && N <= N_MAX);
    dvl_use_grid(N, g_grid.n_bins);
    if (strcmp(argv[1], "refused") == 0) return case_refused();
    if (strcmp(argv[1], "fail") == 0) return case_fail();
    if (strcmp(argv[1], "notstarted") == 0) return case_notstarted();
    if (strcmp(argv[1], "board") == 0) return case_board();
    if (strcmp(argv[1], "reset") == 0) return case_reset();
    if (strcmp(argv[1], "stream") == 0) return case_stream();
    fprintf(stderr, "unknown case: %s\n", argv[1]);
    return 2;
}
'''


@pytest.fixture(scope='module')
def audio_common_lib():
    if shutil.which('make') is None:
        pytest.skip('no make available')
    subprocess.run(
        ['make', '-s', '-C', _AC_DIR, 'BACKEND=ne10', 'lib'],
        check=True, capture_output=True,
    )
    out = subprocess.run(
        ['make', '-s', '-C', _AC_DIR, 'BACKEND=ne10', 'print-lib-path'],
        check=True, capture_output=True, text=True,
    )
    lib = out.stdout.strip().splitlines()[-1]
    assert os.path.isfile(lib), lib
    return lib


def _build(tmp_path_factory, audio_common_lib, name, *flags):
    cc = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if cc is None:
        pytest.skip('no C compiler available')
    work = tmp_path_factory.mktemp(name)
    source = work / 'driver.c'
    source.write_text(_DRIVER, encoding='utf-8')
    executable = work / 'driver'
    subprocess.run(
        [cc, '-O2', '-std=c11', '-D_POSIX_C_SOURCE=200112L',
         '-Wall', '-Wextra', '-Wpedantic', '-Werror', '-ffp-contract=off',
         '-fno-math-errno', *flags,
         '-I', _DEEPVQE_DIR, '-I', _STREAM_DIR, '-I', _COMMON_DIR, '-I', _THIS_DIR,
         '-I', _AC_INCLUDE, '-I', _AC_DIR,
         str(source), *_SOURCES, audio_common_lib, '-lm',
         '-o', str(executable)],
        check=True, capture_output=True,
    )
    return executable


@pytest.fixture(scope='module', params=['native', 'scalar'])
def driver(request, tmp_path_factory, audio_common_lib):
    """The class and the stream behind it, at the house flags with -Werror, on
    the NEON kernels and on their forced-scalar twins."""
    flags = ('-DSIMD_KERNELS_FORCE_SCALAR',) if request.param == 'scalar' else ()
    return _build(tmp_path_factory, audio_common_lib,
                  'deepvqe_ccm_restart_' + request.param, *flags)


def _run(driver, case, grid=_GRID_16K, frames_per_call=1):
    done = subprocess.run(
        [str(driver), case, grid, 'n=%d' % frames_per_call],
        capture_output=True, text=True)
    assert done.returncode == 0, (
        'case %s (%s, N=%d) failed (rc=%d)\n%s' % (
            case, grid, frames_per_call, done.returncode, done.stderr))


@pytest.mark.parametrize('frames_per_call', [1, 4])
@pytest.mark.parametrize('grid', [_GRID_16K, _GRID_48K])
def test_a_refused_commit_restarts_the_ccm_ring(driver, grid, frames_per_call):
    """commit, a commit refused for a non-finite state value, commit, commit:
    the last two batches equal a freshly initialised instance fed them. The
    generation moved by one, the frames are not silence, and a plain bypass in
    the same place (no restart) gives frames that differ from the cold ones."""
    _run(driver, 'refused', grid, frames_per_call)


@pytest.mark.parametrize('frames_per_call', [1, 4])
def test_frame_fail_with_a_started_run_restarts_the_ccm_ring(driver, frames_per_call):
    """The same through frame_fail(p, 1) after a completed run."""
    _run(driver, 'fail', _GRID_16K, frames_per_call)


@pytest.mark.parametrize('frames_per_call', [1, 4])
def test_frame_fail_for_a_run_that_never_started_keeps_the_ring(driver, frames_per_call):
    """frame_fail(p, 0) leaves state, ring and generation as they were: the
    frames after it equal the same frames with that batch never presented, and
    differ from a cold start."""
    _run(driver, 'notstarted', _GRID_16K, frames_per_call)


@pytest.mark.parametrize('grid', [_GRID_16K, _GRID_48K])
def test_the_ring_alone_makes_the_difference_when_no_state_is_registered(driver, grid):
    """A board registers nothing: the restart is the generation and the ring.
    A refused commit and frame_fail(p, 1) give the cold-start frames; a
    bypass and frame_fail(p, 0) keep the ring and the frames differ, so the
    equality above is the ring and nothing else."""
    _run(driver, 'board', grid)


def test_reset_is_a_cold_start(driver):
    _run(driver, 'reset')


@pytest.mark.parametrize('grid', [_GRID_16K, _GRID_48K])
def test_the_stream_freq_entries_restart_after_a_dirty_run_status(driver, grid):
    """deepvqe_stream_post_process_freq: STREAM_RUN_FAILED, an odd status and a
    refused commit restart the state and the ring (the frames after equal a
    fresh stream); STREAM_RUN_NOT_STARTED moves nothing. PCM entries are not
    compared: their STFT and overlap-add history survives a state restart
    by design."""
    _run(driver, 'stream', grid)
