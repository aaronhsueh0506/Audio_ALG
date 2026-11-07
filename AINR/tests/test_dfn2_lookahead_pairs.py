"""Every (mask_lookahead, df_lookahead) pair the DFN2 trainer accepts must
export and agree with Python; the C host is built and run for
``dfn_pairs.c_build_pairs``.

The shipped pair is config.ini's. The model constructor accepts
``0 <= mask_lookahead <= 2`` and ``0 <= df_lookahead < df_order`` independently
(15 pairs), and a checkpoint trained for one of them is tied to it by the
checkpoint contract. What changes between pairs is not the graph but the HOST:
the input window that yields ``heads[t]`` is ``[t-(2-L) .. t+L]``
(L = mask_lookahead), the first heads exist once frame L has been pushed, and
the deep-filter taps shift with df_lookahead. These tests pin that end to end:

  exported ONNX      streamed through onnxruntime with ``feature_windows``
                     must equal the offline model heads, per pair
  C host vs Python   the C model-I/O window and readiness, the compose clock and
                     the deep filter (``dfn2_model_io.c`` + ``dfn2_process.c``,
                     driven with the DFN2Grid of ``tests/dfn2_test_grid.h``
                     for the pair, selected with ``-DDFN2_MASK_LOOKAHEAD`` /
                     ``-DDFN2_DF_LOOKAHEAD``) driven with the heads of the
                     exported-graph wrapper must reproduce the offline
                     ``model(spec, feat_erb, feat_spec)`` frame for frame
  C test suites     tests/test_c_prepost.c and tests/test_nn_chains.c, rebuilt
                     per pair, and for the shipped pair and the deepest ring (2/4)
                     on the 32 kHz and 16 kHz grids too (the non-48 kHz branches of
                     both programs: generated band borders, scaled bin addresses)
  run time          the library holds no pair: ``dfn2_grid_make`` refuses a
                     pair outside the accepted set, checked from C compiled as
                     C11 and gnu99; the other run-time gates (pool hash,
                     model-I/O descriptor, hand-edited grid) are in
                     test_dfn_lookahead_link.py

The pair of the build that takes no override is config.ini's (the default of
``dfn2_test_grid.h``); the others are builds with the two macros overridden
(``dfn_pairs.defines``). The pre/post class is covered per pair in
test_dfn2_prepost_c.py.
"""

import configparser
import importlib.util
import re
import subprocess
import sys

import numpy as np
import pytest
import torch

import dfn_pairs

FAMILY = 'DeepFilterNet2'
DFN2 = dfn_pairs.family_dir(FAMILY)

LOOKAHEAD_PAIRS = dfn_pairs.accepted_pairs(FAMILY)
C_PAIRS = dfn_pairs.c_build_pairs(FAMILY)
SHIPPED_PAIR = dfn_pairs.shipped_pair(FAMILY)
_pair_id = dfn_pairs.pair_id


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def dfn2_modules():
    """model.py, train.py and export_onnx.py of DeepFilterNet2, loaded by path
    so a sibling project's identically named modules cannot be picked up."""
    model = _load('dfn2_pairs_model', DFN2 / 'model.py')
    old_model = sys.modules.get('model')
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(DFN2))
        sys.modules['model'] = model
        train = _load('dfn2_pairs_train', DFN2 / 'train.py')
    finally:
        sys.path[:] = old_path
        if old_model is None:
            sys.modules.pop('model', None)
        else:
            sys.modules['model'] = old_model
    export = _load('dfn2_pairs_export', DFN2 / 'export_onnx.py')
    return model, train, export


@pytest.fixture(scope='module')
def audio_common_lib():
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    return dfn_pairs.audio_common_lib()


# ---------------------------------------------------------------------------
# exported graph, streamed through onnxruntime
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('pair', LOOKAHEAD_PAIRS, ids=_pair_id)
def test_exported_graph_streams_to_the_offline_heads(
        tmp_path, dfn2_modules, pair):
    """Export through the real path (optimiser, folding, validator, ORT parity)
    and feed the written graph the host's windows frame by frame: the heads
    must equal the offline model's, so the exported artefact -- not just the
    wrapper -- is consistent with the alignment its metadata publishes."""
    ort = pytest.importorskip('onnxruntime')
    pytest.importorskip('onnx')
    model_mod, _train, export = dfn2_modules
    mask_lookahead, df_lookahead = pair
    torch.manual_seed(101)
    model = model_mod.DeepFilterNet2(
        n_fft=512, sr=16000, n_erb=32, df_bins=64, enc_ch=8, emb_size=32,
        df_hidden=32, lin_groups=4, enc_lin_groups=4,
        mask_lookahead=mask_lookahead, df_lookahead=df_lookahead,
    ).eval()
    checkpoint = tmp_path / 'ckpt.pth'
    checkpoint.write_bytes(b'only hashed')
    wrapper = export.StatelessDFN2Heads(model).eval()
    path = tmp_path / 'stream.onnx'
    metadata = export.export_graph(
        wrapper, {'SR': 16000, 'N_FFT': 512, 'WIN_LEN': 512, 'HOP_LEN': 256},
        str(checkpoint), str(path), verify=True)
    assert metadata['mask_lookahead'] == mask_lookahead
    assert metadata['df_lookahead'] == df_lookahead
    assert metadata['input_window_alignment'] == (
        export.window_alignment(mask_lookahead))

    session = ort.InferenceSession(
        str(path), providers=['CPUExecutionProvider'])
    frames = 10
    erb = torch.randn(1, 1, frames, model.n_erb)
    spec = torch.randn(1, 2, frames, model.df_bins)
    with torch.no_grad():
        reference = model.heads(erb, spec)
    state = [value.numpy() for value in wrapper.initial_inputs()[2:]]
    mask, coefs, alpha = [], [], []
    for erb_window, spec_window in zip(
            export.feature_windows(erb, mask_lookahead),
            export.feature_windows(spec, mask_lookahead)):
        outputs = session.run(None, dict(
            zip(export.INPUT_NAMES,
                [erb_window.numpy(), spec_window.numpy()] + state)))
        mask.append(outputs[0])
        coefs.append(outputs[1])
        alpha.append(outputs[2])
        state = list(outputs[3:])
    for got, want in zip(
            (np.concatenate(mask, axis=2), np.concatenate(coefs, axis=1),
             np.concatenate(alpha, axis=1)), reference):
        np.testing.assert_allclose(got, want.numpy(), rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# C host (model I/O + compose clock) vs the Python model
# ---------------------------------------------------------------------------

_STREAM_DRIVER = r'''
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tests/dfn2_test_grid.h"
#include "DeepFilterNet2/dfn2_model_io.h"
#include "DeepFilterNet2/dfn2_process.h"

#define COEFS (DFN2_DF_BINS * DFN2_DF_ORDER * 2)

static float *take(const float **cursor, size_t count) {
    float *out = (float *)malloc(count * sizeof(float));
    memcpy(out, *cursor, count * sizeof(float));
    *cursor += count;
    return out;
}

int main(int argc, char **argv) {
    FILE *in, *out;
    long size;
    unsigned char *blob;
    const float *cursor;
    int32_t frames;
    float *erb_fwd, *erb_inv, *feat_erb, *feat_spec, *spec_re, *spec_im;
    float *head_mask, *head_coefs, *head_alpha, *win_erb, *win_spec;
    static DFN2State st;
    static DFN2ModelIOState io;
    void *state_block, *io_block;
    static float zero_erb[DFN2_N_ERB];
    static float zero_spec[2 * DFN2_DF_BINS];
    static float zero_bins[DFN2_N_BINS];
    static float unit_mask[DFN2_N_ERB];
    static float zero_coefs[COEFS];
    static float out_re[DFN2_N_BINS], out_im[DFN2_N_BINS];
    int total, c, ready_bad = 0, window_bad = 0, windows = 0, emitted = 0;
    int first_ready = -1;
    const DFN2Grid *grid = dfn2_test_grid();
    const size_t state_bytes = dfn2_state_mem_size(grid);
    const size_t io_bytes = dfn2_model_io_mem_size(grid);
    const size_t erb_window_bytes = (size_t)DFN2_MODEL_INPUT_FRAMES *
                                    DFN2_N_ERB * sizeof(float);
    const size_t spec_window_bytes = (size_t)2 * DFN2_MODEL_INPUT_FRAMES *
                                     DFN2_DF_BINS * sizeof(float);

    float atten_lim_db = 0.0f;

    if (argc != 3 && argc != 4) return 2;
    if (argc == 4) atten_lim_db = (float)atof(argv[3]);
    in = fopen(argv[1], "rb");
    if (!in) return 2;
    fseek(in, 0, SEEK_END);
    size = ftell(in);
    fseek(in, 0, SEEK_SET);
    blob = (unsigned char *)malloc((size_t)size);
    if (fread(blob, 1, (size_t)size, in) != (size_t)size) return 2;
    fclose(in);
    memcpy(&frames, blob, sizeof frames);
    cursor = (const float *)(blob + sizeof frames);
    erb_fwd = take(&cursor, (size_t)DFN2_N_BINS * DFN2_N_ERB);
    erb_inv = take(&cursor, (size_t)DFN2_N_ERB * DFN2_N_BINS);
    feat_erb = take(&cursor, (size_t)frames * DFN2_N_ERB);
    feat_spec = take(&cursor, (size_t)frames * 2 * DFN2_DF_BINS);
    spec_re = take(&cursor, (size_t)frames * DFN2_N_BINS);
    spec_im = take(&cursor, (size_t)frames * DFN2_N_BINS);
    head_mask = take(&cursor, (size_t)frames * DFN2_N_ERB);
    head_coefs = take(&cursor, (size_t)frames * COEFS);
    head_alpha = take(&cursor, (size_t)frames);
    win_erb = take(&cursor, (size_t)frames * DFN2_MODEL_INPUT_FRAMES *
                                DFN2_N_ERB);
    win_spec = take(&cursor, (size_t)frames * 2 * DFN2_MODEL_INPUT_FRAMES *
                                 DFN2_DF_BINS);
    for (c = 0; c < DFN2_N_ERB; ++c) unit_mask[c] = 1.0f;

    state_block = aligned_alloc(16, state_bytes);
    io_block = aligned_alloc(16, io_bytes);
    if (!state_block || !io_block) return 2;
    if (dfn2_state_init(&st, state_block, state_bytes, grid) != 0) return 2;
    dfn2_set_erb_matrices(&st, erb_fwd, erb_inv);
    if (dfn2_model_io_init(&io, io_block, io_bytes, grid, NULL, 0, NULL, 0) != 0) return 2;
    out = fopen(argv[2], "wb");
    if (!out) return 2;

    /* The caller keeps pushing zero frames after the last real one: that is
     * the right-hand zero padding of the offline model, and it flushes the
     * last MASK + DF source frames. */
    total = frames + DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD;
    for (c = 0; c < total; ++c) {
        const int real = c < frames;
        const float *erb = real ? feat_erb + (size_t)c * DFN2_N_ERB : zero_erb;
        const float *spec = real ? feat_spec + (size_t)c * 2 * DFN2_DF_BINS
                                 : zero_spec;
        const float *re = real ? spec_re + (size_t)c * DFN2_N_BINS : zero_bins;
        const float *im = real ? spec_im + (size_t)c * DFN2_N_BINS : zero_bins;
        int ready = dfn2_model_io_push_features(&io, erb, spec);
        int valid;
        long long frame = -1;
        if (ready != (c >= DFN2_MASK_LOOKAHEAD ? 1 : 0)) ++ready_bad;
        if (ready == 1) {
            const int t = c - DFN2_MASK_LOOKAHEAD;
            const int has = t < frames;
            if (first_ready < 0) first_ready = c;
            if (has) {
                const float *we = win_erb + (size_t)t *
                                  DFN2_MODEL_INPUT_FRAMES * DFN2_N_ERB;
                const float *ws = win_spec + (size_t)t * 2 *
                                  DFN2_MODEL_INPUT_FRAMES * DFN2_DF_BINS;
                ++windows;
                /* The windows are packed at the grid's stride: compare the run. */
                if (memcmp(we, io.erb_window, erb_window_bytes) != 0 ||
                    memcmp(ws, io.spec_window, spec_window_bytes) != 0)
                    ++window_bad;
            }
            valid = dfn2_enhance_spectrum_stream(
                &st, re, im, 1,
                has ? head_mask + (size_t)t * DFN2_N_ERB : unit_mask,
                has ? head_coefs + (size_t)t * COEFS : zero_coefs,
                has ? head_alpha[t] : 0.0f, atten_lim_db, out_re, out_im,
                &frame);
        } else {
            valid = dfn2_enhance_spectrum_stream(&st, re, im, 0, NULL, NULL, 0.0f,
                                        atten_lim_db, out_re, out_im, &frame);
        }
        if (valid < 0) return 3;
        if (valid == 1) {
            int32_t index = (int32_t)frame;
            fwrite(&index, sizeof index, 1, out);
            fwrite(out_re, sizeof(float), DFN2_N_BINS, out);
            fwrite(out_im, sizeof(float), DFN2_N_BINS, out);
            ++emitted;
        }
    }
    fclose(out);
    printf("ready_frames=%u first_ready=%d windows=%d window_bad=%d "
           "ready_bad=%d emitted=%d\n",
           dfn2_model_io_ready_frames(&io), first_ready, windows, window_bad,
           ready_bad, emitted);
    return (ready_bad || window_bad) ? 1 : 0;
}
'''


# The signal grids the stream test runs besides the shipped 48 kHz one: the
# presets of dfn2_test_grid.h (-DDFN2_GRID_KHZ), a custom grid of another
# model (-DDFN2_SR and the three that go with it), and the model that matches
# each. The shipped model keeps its sizes from config.ini; these change the
# frame size, the band count and the deep-filter bins, i.e. the stride of every
# packed window (the presets all keep 96 deep-filter bins, the custom grid 48).
GRIDS = {
    'khz16': (['-DDFN2_GRID_KHZ=16'],
              dict(sr=16000, n_fft=256, n_erb=24, df_bins=96)),
    'khz32': (['-DDFN2_GRID_KHZ=32'],
              dict(sr=32000, n_fft=512, n_erb=28, df_bins=96)),
    'custom': (['-DDFN2_SR=16000', '-DDFN2_N_FFT=512', '-DDFN2_N_ERB=20',
                '-DDFN2_DF_BINS=48'],
               dict(sr=16000, n_fft=512, n_erb=20, df_bins=48)),
}
OTHER_GRID_PAIRS = (SHIPPED_PAIR, (2, 4))


def _build_stream_driver(tmp_path_factory, audio_common_lib, pair,
                         grid_flags=()):
    work = tmp_path_factory.mktemp('dfn2_stream_%d%d' % pair)
    source = work / 'driver.c'
    source.write_text(_STREAM_DRIVER, encoding='utf-8')
    exe = work / 'driver'
    subprocess.run(
        ['cc', *dfn_pairs.house_cflags(), *grid_flags,
         *dfn_pairs.defines(FAMILY, pair),
         str(source), str(dfn_pairs.source_path(FAMILY, 'process')),
         str(dfn_pairs.source_path(FAMILY, 'model_io')), audio_common_lib,
         '-lm', '-o', str(exe)],
        check=True, capture_output=True)
    return exe


@pytest.fixture(scope='module')
def stream_drivers(tmp_path_factory, audio_common_lib):
    return {pair: _build_stream_driver(tmp_path_factory, audio_common_lib,
                                       pair)
            for pair in C_PAIRS}


@pytest.fixture(scope='module')
def other_grid_drivers(tmp_path_factory, audio_common_lib):
    return {(grid, pair): _build_stream_driver(
                tmp_path_factory, audio_common_lib, pair, GRIDS[grid][0])
            for grid in GRIDS for pair in OTHER_GRID_PAIRS}


def _model_for(model_mod, train, pair, grid=None):
    cfg = configparser.ConfigParser()
    cfg.read(DFN2 / 'config.ini')
    kwargs = {**train.read_model_config(cfg),
              'mask_lookahead': pair[0], 'df_lookahead': pair[1],
              **(GRIDS[grid][1] if grid else {})}
    torch.manual_seed(7)
    model = model_mod.DeepFilterNet2(**kwargs).eval()
    # Non-trivial batch-norm statistics: with identity statistics a window that
    # is off by a frame can still land inside the tolerance.
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_(0.0, 0.1)
            module.running_var.uniform_(0.5, 1.5)
    return model.eval()


def _check_c_stream(tmp_path, dfn2_modules, driver, pair, atten_lim,
                    grid=None):
    model_mod, train, export = dfn2_modules
    mask_lookahead, df_lookahead = pair
    model = _model_for(model_mod, train, pair, grid)
    wrapper = export.StatelessDFN2Heads(model).eval()
    frames = 20
    generator = torch.Generator().manual_seed(11)
    erb = torch.randn(1, 1, frames, model.n_erb, generator=generator)
    spec_feat = torch.randn(1, 2, frames, model.df_bins, generator=generator)
    spec = torch.complex(
        torch.randn(1, model.n_bins, frames, generator=generator),
        torch.randn(1, model.n_bins, frames, generator=generator))
    with torch.no_grad():
        reference, _ = model(spec, erb, spec_feat)
        erb_windows = list(export.feature_windows(erb, mask_lookahead))
        spec_windows = list(export.feature_windows(spec_feat, mask_lookahead))
        state = wrapper.initial_inputs()[2:]
        mask, coefs, alpha = [], [], []
        for erb_window, spec_window in zip(erb_windows, spec_windows):
            output = wrapper(erb_window, spec_window, *state)
            mask.append(output[0].numpy())
            coefs.append(output[1].numpy())
            alpha.append(output[2].numpy())
            state = output[3:]

    source = tmp_path / 'stream_in.bin'
    result = tmp_path / 'stream_out.bin'
    dfn_pairs.dfn2_write_stream_input(
        source, model, frames, erb, spec_feat, spec, (mask, coefs, alpha),
        (erb_windows, spec_windows))
    done = subprocess.run(
        [str(driver), str(source), str(result), str(atten_lim)],
        capture_output=True, text=True)
    assert done.returncode == 0, (done.stdout, done.stderr)
    # dfn2_model_io_ready_frames: the pushes the first call needs (L + 1).
    assert 'ready_frames=%d ' % (mask_lookahead + 1) in done.stdout, done.stdout
    assert 'first_ready=%d ' % mask_lookahead in done.stdout, done.stdout
    assert 'window_bad=0 ready_bad=0 emitted=%d' % frames in done.stdout, (
        done.stdout)

    record = 4 + 2 * 4 * model.n_bins
    raw = result.read_bytes()
    assert len(raw) == frames * record
    got = np.zeros((model.n_bins, frames), dtype=np.complex64)
    for index in range(frames):
        chunk = raw[index * record:(index + 1) * record]
        frame = int(np.frombuffer(chunk[:4], '<i4')[0])
        assert frame == index, 'emitted frames must count from 0 without a gap'
        values = np.frombuffer(chunk[4:], '<f4')
        got[:, frame] = values[:model.n_bins] + 1j * values[model.n_bins:]
    want = reference[0].numpy()
    if atten_lim:
        mix = 10 ** (-abs(atten_lim) / 20)
        want = spec[0].numpy() * mix + want * (1 - mix)
    np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)
    # Not a vacuous pass: the enhanced spectrum differs from the noisy input.
    assert float(np.abs(want - spec[0].numpy()).max()) > 1e-2


@pytest.mark.parametrize('atten_lim', [0.0, -20.0])
@pytest.mark.parametrize('pair', C_PAIRS, ids=_pair_id)
def test_c_host_stream_reproduces_the_offline_model(
        tmp_path, dfn2_modules, stream_drivers, pair, atten_lim):
    """The graph's heads, fed back through the C host's window, readiness and
    compose clock (built for this pair), reproduce the offline model's
    enhanced spectrum frame for frame: the Python exporter's window and the C
    ``dfn2_model_io`` window are the same alignment, and the deep filter taps
    follow df_lookahead. With an attenuation limit the noisy frame of the
    output is mixed in, which reads the noisy ring MASK + DF frames back."""
    _check_c_stream(tmp_path, dfn2_modules, stream_drivers[pair], pair,
                    atten_lim)


@pytest.mark.parametrize('atten_lim', [0.0, -20.0])
@pytest.mark.parametrize('pair', OTHER_GRID_PAIRS, ids=_pair_id)
@pytest.mark.parametrize('grid', sorted(GRIDS))
def test_c_host_stream_reproduces_the_offline_model_on_another_grid(
        tmp_path, dfn2_modules, other_grid_drivers, grid, pair, atten_lim):
    """The same comparison on the 16 kHz and 32 kHz presets and a custom grid
    (20 bands, 48 deep-filter bins). The library holds no grid: the frame size,
    band count and the stride of the packed feature windows come from the
    DFN2Grid the program builds, so a window packed at the wrong stride (the
    shipped grid's) fails here. The deepest ring (2/4) is included."""
    _check_c_stream(tmp_path, dfn2_modules, other_grid_drivers[(grid, pair)],
                    pair, atten_lim, grid)


# ---------------------------------------------------------------------------
# the C test suites, rebuilt per pair and per grid
# ---------------------------------------------------------------------------

# The grid axis of the suites: the build with no flag (the shipped 48 kHz grid) and the two
# presets of dfn2_test_grid.h. Every pair of C_PAIRS runs on the default grid; the shipped pair
# and the deepest ring (2/4) also run on the other two, which is what builds the non-48 kHz
# branches of test_c_prepost.c and test_nn_chains.c (neither `make test` nor `make test-simd`
# does).
SUITE_GRID_FLAGS = {'khz48': [], 'khz32': ['-DDFN2_GRID_KHZ=32'],
                    'khz16': ['-DDFN2_GRID_KHZ=16']}
SUITE_CASES = ([('khz48', pair) for pair in C_PAIRS] +
               [(grid, pair) for grid in ('khz32', 'khz16') for pair in OTHER_GRID_PAIRS])
SUITES = ('test_c_prepost.c', 'test_nn_chains.c')


def _suite_case_id(case):
    return '%s-%s' % (case[0], _pair_id(case[1]))


@pytest.fixture(scope='module')
def suite_runs(tmp_path_factory, audio_common_lib):
    """``run(suite, grid, pair)`` -> the finished run of a contract binary built with the flags
    and sources of ``make test`` for that grid and pair; built once, kept."""
    done = {}

    def run(suite, grid, pair):
        key = (suite, grid, tuple(pair))
        if key not in done:
            work = tmp_path_factory.mktemp('dfn2_suite')
            exe = work / 'suite'
            subprocess.run(
                ['cc', *dfn_pairs.house_cflags(), *SUITE_GRID_FLAGS[grid],
                 *dfn_pairs.defines(FAMILY, pair), *dfn_pairs.suite_sources(suite),
                 audio_common_lib, '-lm', '-o', str(exe)],
                check=True, capture_output=True)
            done[key] = subprocess.run([str(exe)], capture_output=True, text=True)
        return done[key]
    return run


def _digest(run):
    match = re.search(r'digest=([0-9a-f:]+)\s*$', run.stdout)
    assert match, run.stdout
    return match.group(1)


@pytest.mark.parametrize('suite', SUITES)
@pytest.mark.parametrize('case', SUITE_CASES, ids=_suite_case_id)
def test_c_suite_passes_for_every_lookahead_pair(suite_runs, case, suite):
    """The Makefile's contract binaries (stream alignment, model I/O, ring,
    chains) compiled with the flags and sources of ``make test``, with the two
    macros overridden for the non-shipped pairs, on the default grid and, for
    the shipped pair and 2/4, on the 32 kHz and 16 kHz presets."""
    grid, pair = case
    done = suite_runs(suite, grid, pair)
    assert done.returncode == 0, (done.stdout, done.stderr)


@pytest.mark.parametrize('suite', SUITES)
@pytest.mark.parametrize('pair', OTHER_GRID_PAIRS, ids=_pair_id)
def test_the_grid_flag_reaches_the_suite(suite_runs, pair, suite):
    """Not a vacuous axis: a program built for the 32 kHz or the 16 kHz preset prints another
    digest than the one built for 48 kHz, for the same pair, so the flag is what the program
    ran with (a dropped ``-DDFN2_GRID_KHZ`` leaves the 48 kHz digest and fails here)."""
    digests = {grid: _digest(suite_runs(suite, grid, pair)) for grid in SUITE_GRID_FLAGS}
    assert len(set(digests.values())) == len(digests), digests


@pytest.mark.parametrize('std', ['c11', 'gnu99'])
@pytest.mark.parametrize('pair', [(3, 0), (3, 3), (-1, 0), (0, -1), (1, 5),
                                  (2, 5)])
def test_dfn2_grid_refuses_pairs_the_trainer_would_not_accept(
        tmp_path, audio_common_lib, pair, std):
    """The library carries no pair, so the range check is
    ``dfn2_grid_make``'s, at run time, and it holds in every language mode (the
    integration build is gnu99): a mask lookahead outside 0..2 is
    ERR_MASK_LOOKAHEAD, a deep-filter lookahead outside 0..4 is
    ERR_DF_LOOKAHEAD, and the grid is left untouched. The shipped pair builds
    and validates."""
    codes = dfn_pairs.dfn2_status_codes()
    exe = dfn_pairs.build_dfn2_grid_probe(tmp_path, std)
    result = dfn_pairs.run_dfn2_grid_probe(exe)
    status, _, _, untouched, text = result[pair]
    expected = ('ERR_MASK_LOOKAHEAD' if not 0 <= pair[0] <= 2
                else 'ERR_DF_LOOKAHEAD')
    assert status == codes[expected], (pair, status, text)
    assert untouched
    assert result[SHIPPED_PAIR][0] == codes['OK']
    assert result[SHIPPED_PAIR][2] == codes['OK']   # dfn2_grid_validate
