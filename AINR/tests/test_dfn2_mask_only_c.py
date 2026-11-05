"""The C host built with -DDFN2_MASK_ONLY: stage 1 alone (ERB mask, no deep filter).

The build is the model of ``export_onnx.py --mask-only``. Gates:

* the contract the build publishes: layout 9, one state tensor (h_gru, three rows),
  no deep-filter GRU layers or pathway window, no coefficient/alpha head in the
  accelerator boundary structs (they do not compile in), descriptor of the full
  build refused;
* the output of the class is ``noisy * (mask @ erb_inv)`` of the head frame, the
  arithmetic of ``DeepFilterNet2.compose_mask_only`` (the Python reference), with
  and without an attenuation limit, one hop per frame and MASK_LOOKAHEAD hops of
  latency;
* a deep-filter lookahead other than 0 is refused by the grid constructor of the
  mask-only build (``dfn2_grid_make``), and a mask lookahead of 2 needs the ring
  capacity raised to 3.

The library holds no grid, so the host is run on several: the shipped one
(config.ini, no override), the 16 kHz and 32 kHz grids of ``dfn2_test_grid.h``
and mask lookaheads 0, 1 and 2. The comparison is against numpy on the same ERB
matrices the C class loads; the check is shown to fail on a mask taken one frame
off. The grid rules are probed from a program built both ways, so the refusal is
shown to belong to the mask-only build.
"""

import configparser
import subprocess
import sys

import numpy as np
import pytest

import dfn_pairs

FAMILY = 'DeepFilterNet2'
FRAMES = 40

# The grids of dfn2_test_grid.h (-DDFN2_GRID_KHZ=<khz>; 48 when nothing is
# chosen): sample rate, n_fft, n_erb, df_bins. The 48 kHz row is config.ini's.
GRIDS = {48: (48000, 1024, 32, 96), 32: (32000, 512, 28, 96),
         16: (16000, 256, 24, 96)}
# (kHz, mask_lookahead): the first is the build with no override at all, the
# others change the grid or the lookahead. A mask lookahead of 2 is a ring of 3
# frames, one above the mask-only capacity (2), so that build raises it.
CASES = [pytest.param(khz, mask, id='%dk-mask%d' % (khz, mask))
         for khz, mask in ((48, 1), (48, 0), (48, 2), (16, 1), (32, 2))]

_DRIVER = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dfn2_test_grid.h"

#ifndef DFN2_MASK_ONLY
#error "this driver is the mask-only build"
#endif
_Static_assert(DFN2_MODEL_GRU_ROWS == 3, "encoder row + two ERB rows");
_Static_assert(DFN2_MODEL_DF_GRU_LAYERS == 0, "no deep-filter decoder");
_Static_assert(DFN2_MODEL_DF_PATHWAY_WINDOW == 0, "no pathway window");
_Static_assert(DFN2_DF_LOOKAHEAD == 0, "no deep filter to look ahead for");
_Static_assert(DFN2_MODEL_IO_LAYOUT_VERSION == DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY,
               "mask-only layout number");

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #x); \
    return 1; \
} } while (0)

static float *load(const char *path, size_t count) {
    FILE *f = fopen(path, "rb");
    float *data = (float *)malloc(count * sizeof(float));
    if (!f || !data || fread(data, sizeof(float), count, f) != count) exit(3);
    fclose(f);
    return data;
}

int main(int argc, char **argv) {
    const DFN2Grid *grid = dfn2_test_grid();
    DFN2Grid other;
    DFN2ModelIoDescriptor descriptor;
    DFN2PrepostConfig cfg;
    DFN2PrepostInputs inputs;
    DFN2PrepostOutputs outputs;
    DFN2Prepost *p;
    float re[DFN2_N_BINS], im[DFN2_N_BINS];
    float *erb_fwd, *erb_inv, *spec, *mask;
    FILE *out;
    int frames, t, valid;
    long long frame;

    if (argc != 7) return 2;
    frames = atoi(argv[5]);
    {
        char path[1024];
        snprintf(path, sizeof(path), "%s/erb_fwd.bin", argv[1]);
        erb_fwd = load(path, (size_t)DFN2_N_BINS * DFN2_N_ERB);
        snprintf(path, sizeof(path), "%s/erb_inv.bin", argv[1]);
        erb_inv = load(path, (size_t)DFN2_N_ERB * DFN2_N_BINS);
    }
    spec = load(argv[2], (size_t)frames * 2 * DFN2_N_BINS);
    mask = load(argv[3], (size_t)frames * DFN2_N_ERB);

    CHECK(dfn2_model_io_descriptor_default(grid, &descriptor) == 0);
    CHECK(descriptor.layout_version ==
          (unsigned int)DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY);
    CHECK(descriptor.state_tensor_count == 1);
    CHECK(descriptor.mask_lookahead == DFN2_MASK_LOOKAHEAD &&
          descriptor.df_lookahead == 0);
    CHECK(dfn2_model_io_descriptor_validate(grid, &descriptor) == 0);
    /* The grid is part of the contract: a graph described for this grid is
     * refused by an instance that runs another one. */
    CHECK(dfn2_grid_make(&other, DFN2_SR, DFN2_N_FFT, DFN2_N_ERB - 4,
                         DFN2_DF_BINS, DFN2_MASK_LOOKAHEAD, 0) == DFN2_GRID_OK);
    CHECK(dfn2_model_io_descriptor_validate(&other, &descriptor) != 0);
    descriptor.layout_version = 8u;          /* the full build's graph */
    CHECK(dfn2_model_io_descriptor_validate(grid, &descriptor) != 0);

    CHECK(dfn2_prepost_config_defaults(&cfg, DFN2_IO_FREQ, grid) == 0);
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;
    cfg.atten_lim_db = (float)atof(argv[6]);
    p = dfn2_prepost_create(&cfg);
    CHECK(p != NULL);
    CHECK(dfn2_prepost_layout_version(p) ==
          DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY);
    CHECK(dfn2_prepost_model_lookahead_frames(p) == DFN2_MASK_LOOKAHEAD);
    CHECK(dfn2_prepost_grid(p) != NULL &&
          dfn2_prepost_grid(p)->n_erb == DFN2_N_ERB);

    out = fopen(argv[4], "wb");
    CHECK(out != NULL);
    for (t = 0; t < frames; ++t) {
        const float *frame_re = spec + (size_t)t * 2 * DFN2_N_BINS;
        const float *frame_im = frame_re + DFN2_N_BINS;
        int ready = dfn2_prepost_pre_process_freq(p, frame_re, frame_im);
        CHECK(ready == (t >= DFN2_MASK_LOOKAHEAD ? 1 : 0));
        if (ready) {
            CHECK(dfn2_prepost_frame_inputs(p, &inputs, &outputs) == 0);
            CHECK(inputs.gru_hidden_elements ==
                  (size_t)DFN2_MODEL_GRU_ROWS * DFN2_MODEL_GRU_HIDDEN);
            CHECK(outputs.erb_mask_elements == DFN2_N_ERB);
            memcpy(outputs.erb_mask,
                   mask + (size_t)(t - DFN2_MASK_LOOKAHEAD) * DFN2_N_ERB,
                   DFN2_N_ERB * sizeof(float));
            CHECK(dfn2_prepost_frame_commit(p) == 0);
        }
        valid = -1;
        CHECK(dfn2_prepost_post_process_freq(p, re, im, &valid) == 0);
        CHECK(valid == (t >= dfn2_prepost_latency_frames(grid) ? 1 : 0));
        if (valid) {
            int32_t index;
            CHECK(dfn2_prepost_output_frame_index(p, &frame) == 0);
            index = (int32_t)frame;
            fwrite(&index, sizeof(index), 1, out);
            fwrite(re, sizeof(float), DFN2_N_BINS, out);
            fwrite(im, sizeof(float), DFN2_N_BINS, out);
        }
    }
    fclose(out);
    dfn2_prepost_destroy(p);
    return 0;
}
'''


def _grid_flags(khz, mask):
    """-D overrides of one case; none for the build with no override (the
    shipped 48 kHz grid, mask lookahead 1)."""
    flags = []
    if khz != 48:
        flags.append('-DDFN2_GRID_KHZ=%d' % khz)
    if mask != 1:
        flags.append('-DDFN2_MASK_LOOKAHEAD=%d' % mask)
    if mask + 1 > 2:                       # ring = mask + 1; capacity is 2
        flags.append('-DDFN2_MAX_DF_RING=%d' % (mask + 1))
    return flags


@pytest.fixture(scope='module')
def driver(tmp_path_factory):
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    lib = dfn_pairs.audio_common_lib()
    built = {}

    def get(khz, mask):
        if (khz, mask) not in built:
            work = tmp_path_factory.mktemp('dfn2_mask_only_c')
            source = work / 'driver.c'
            source.write_text(_DRIVER, encoding='utf-8')
            executable = work / 'driver'
            subprocess.run(
                ['cc', *dfn_pairs.house_cflags(), '-DDFN2_MASK_ONLY',
                 *_grid_flags(khz, mask), '-I', str(dfn_pairs.AINR / 'tests'),
                 str(source),
                 *[str(dfn_pairs.source_path(FAMILY, kind))
                   for kind in ('prepost', 'process', 'model_io')],
                 lib, '-lm', '-o', str(executable)],
                check=True, capture_output=True)
            built[(khz, mask)] = executable
        return built[(khz, mask)]
    return get


@pytest.fixture(scope='module')
def erb_dir(tmp_path_factory):
    """The runtime ERB matrices of a grid, written the way a deployment does:
    config.ini with the grid's signal and band numbers."""
    # The families share module names (model, train, export_erb_matrix, ...): whichever
    # another test imported first would answer for this one, and this one must not leave
    # its own behind for the next.
    family_modules = ('train', 'inference', 'model', 'checkpoint_utils',
                      'export_onnx', 'stream_model', 'export_erb_matrix')
    for name in family_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(dfn_pairs.family_dir(FAMILY)))
    try:
        from export_erb_matrix import write_runtime_bins
    finally:
        for name in family_modules:
            sys.modules.pop(name, None)
    built = {}

    def get(khz):
        if khz not in built:
            sr, n_fft, n_erb, df_bins = GRIDS[khz]
            cfg = configparser.ConfigParser()
            cfg.read(dfn_pairs.family_dir(FAMILY) / 'config.ini')
            for section, key, value in (
                    ('signal', 'sr', sr), ('signal', 'n_fft', n_fft),
                    ('signal', 'win_len', n_fft),
                    ('signal', 'hop_len', n_fft // 2),
                    ('model', 'n_erb', n_erb), ('model', 'df_bins', df_bins)):
                cfg.set(section, key, str(value))
            work = tmp_path_factory.mktemp('erb')
            config = work / 'config.ini'
            with config.open('w') as stream:
                cfg.write(stream)
            out = work / 'bins'
            write_runtime_bins(str(config), str(out))
            built[khz] = out
        return built[khz]
    return get


def _run(driver, erb_dir, tmp_path, atten_db, n_bins, n_erb):
    rng = np.random.default_rng(11)
    spec = (rng.standard_normal((FRAMES, 2, n_bins)) * 0.05).astype('<f4')
    mask = rng.uniform(0.0, 1.0, (FRAMES, n_erb)).astype('<f4')
    spec.tofile(tmp_path / 'spec.bin')
    mask.tofile(tmp_path / 'mask.bin')
    subprocess.run([str(driver), str(erb_dir), str(tmp_path / 'spec.bin'),
                    str(tmp_path / 'mask.bin'), str(tmp_path / 'out.bin'),
                    str(FRAMES), str(atten_db)],
                   check=True, capture_output=True)
    raw = (tmp_path / 'out.bin').read_bytes()
    record = 4 + 2 * n_bins * 4
    assert len(raw) % record == 0
    frames, outputs = [], []
    for at in range(0, len(raw), record):
        frames.append(int(np.frombuffer(raw[at:at + 4], '<i4')[0]))
        body = np.frombuffer(raw[at + 4:at + record], '<f4')
        outputs.append(body.reshape(2, n_bins))
    return spec, mask, frames, np.stack(outputs)


def _expected(spec, mask, inverse, frames, atten_db, shift=0):
    gain = mask[np.array(frames) + shift] @ inverse            # [n, bins]
    noisy = spec[np.array(frames)]
    enhanced = noisy * gain[:, None, :]
    if atten_db:
        lim = np.float32(10 ** (-abs(atten_db) / 20))
        enhanced = noisy * lim + enhanced * (np.float32(1) - lim)
    return enhanced


@pytest.mark.parametrize('atten_db', [0, 20])
@pytest.mark.parametrize('khz,lookahead', CASES)
def test_mask_only_output_is_the_masked_noisy_spectrum(
        driver, erb_dir, tmp_path, khz, lookahead, atten_db):
    _, n_fft, n_erb, _ = GRIDS[khz]
    n_bins = n_fft // 2 + 1
    bins = erb_dir(khz)
    spec, mask, frames, got = _run(driver(khz, lookahead), bins, tmp_path,
                                   atten_db, n_bins, n_erb)
    inverse = np.fromfile(bins / 'erb_inv.bin', '<f4').reshape(n_erb, n_bins)
    # one output per hop after MASK_LOOKAHEAD hops, in frame order, no deep
    # filter delay on top
    assert frames == list(range(0, FRAMES - lookahead))
    want = _expected(spec, mask, inverse, frames, atten_db)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-7)
    # the check can fail: a mask taken one frame off is not within tolerance
    off = _expected(spec, mask, inverse, frames[:-1], atten_db, shift=1)
    assert not np.allclose(got[:-1], off, rtol=1e-5, atol=1e-7)


# The grid constructor of the build, probed with its arguments on the command
# line: prints the status of dfn2_grid_make for (sr, n_fft, n_erb, df_bins,
# mask_lookahead, df_lookahead) and the names of the statuses.
_GRID_PROBE = r'''
#include <stdio.h>
#include <stdlib.h>

#include "DeepFilterNet2/dfn2_grid.h"

int main(int argc, char **argv) {
    DFN2Grid g;
    int status;
    if (argc != 7) return 2;
    status = dfn2_grid_make(&g, atoi(argv[1]), atoi(argv[2]), atoi(argv[3]),
                            atoi(argv[4]), atoi(argv[5]), atoi(argv[6]));
    printf("%d %d %d %d %s\n", status, DFN2_GRID_OK, DFN2_GRID_ERR_DF_LOOKAHEAD,
           DFN2_GRID_ERR_CAPACITY, dfn2_grid_status_text(status));
    return 0;
}
'''


@pytest.fixture(scope='module')
def grid_probe(tmp_path_factory):
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    lib = dfn_pairs.audio_common_lib()
    work = tmp_path_factory.mktemp('dfn2_grid_probe')
    source = work / 'probe.c'
    source.write_text(_GRID_PROBE, encoding='utf-8')
    built = {}

    def get(*flags):
        if flags not in built:
            exe = work / ('probe%d' % len(built))
            subprocess.run(
                ['cc', *dfn_pairs.house_cflags(), *flags, str(source),
                 str(dfn_pairs.source_path(FAMILY, 'process')), lib, '-lm',
                 '-o', str(exe)], check=True, capture_output=True)

            def ask(sr, n_fft, n_erb, df_bins, mask, df):
                done = subprocess.run(
                    [str(exe)] + [str(v) for v in (sr, n_fft, n_erb, df_bins,
                                                   mask, df)],
                    check=True, capture_output=True, text=True)
                status, ok, df_la, capacity = map(
                    int, done.stdout.split()[:4])
                return status, {'ok': ok, 'df_lookahead': df_la,
                                'capacity': capacity}
            built[flags] = ask
        return built[flags]
    return get


def test_a_deep_filter_lookahead_is_refused_by_the_mask_only_grid(grid_probe):
    """In the mask-only build there is no deep filter to look ahead for: the
    grid constructor refuses a nonzero df_lookahead and accepts 0. The same
    arguments are accepted by the full build, so the refusal belongs to
    -DDFN2_MASK_ONLY and the probe can fail."""
    mask_only = grid_probe('-DDFN2_MASK_ONLY')
    full = grid_probe()
    status, names = mask_only(48000, 1024, 32, 96, 1, 0)
    assert status == names['ok']
    status, names = mask_only(48000, 1024, 32, 96, 1, 1)
    assert status == names['df_lookahead'] != names['ok']
    status, names = full(48000, 1024, 32, 96, 1, 1)
    assert status == names['ok']


def test_a_mask_lookahead_of_two_needs_a_ring_of_three(grid_probe):
    """The mask-only state keeps only the noisy ring, mask_lookahead + 1
    frames, and its default capacity is 2: mask 2 is refused until the
    capacity is raised on the build line, mask 0 and 1 fit."""
    default = grid_probe('-DDFN2_MASK_ONLY')
    raised = grid_probe('-DDFN2_MASK_ONLY', '-DDFN2_MAX_DF_RING=3')
    for mask in (0, 1):
        assert default(48000, 1024, 32, 96, mask, 0)[0] == 0
    status, names = default(48000, 1024, 32, 96, 2, 0)
    assert status == names['capacity'] != names['ok']
    assert raised(48000, 1024, 32, 96, 2, 0)[0] == 0


@pytest.mark.parametrize('process_flags,model_io_flags', [
    (['-DDFN2_MASK_ONLY'], []),
    ([], ['-DDFN2_MASK_ONLY']),
])
def test_units_built_for_different_variants_do_not_link(
        tmp_path, process_flags, model_io_flags):
    """-DDFN2_MASK_ONLY changes the state and boundary structs; a unit built the
    other way fails to link instead of running a different layout."""
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    main = tmp_path / 'main.c'
    main.write_text('#include "DeepFilterNet2/dfn2_model_io.h"\n'
                    'int main(void) {\n'
                    '    return dfn2_model_io_push_features(0, 0, 0);\n}\n')
    objects = []
    for name, flags in (('dfn2_process', process_flags),
                        ('dfn2_model_io', model_io_flags)):
        obj = tmp_path / (name + '.o')
        subprocess.run(
            ['cc', *dfn_pairs.house_cflags(), *flags, '-c',
             str(dfn_pairs.source_path(FAMILY, name.split('_', 1)[1])),
             '-o', str(obj)], check=True, capture_output=True)
        objects.append(str(obj))
    # DFN2_MULTI_FRAME_HOST is defined by dfn2_model_io.c itself only
    flags = ['cc', *dfn_pairs.house_cflags(), str(main), *objects,
             dfn_pairs.audio_common_lib(), '-lm', '-o', str(tmp_path / 'exe')]
    bad = subprocess.run(flags, capture_output=True, text=True)
    assert bad.returncode != 0 and 'dfn2_variant' in bad.stderr, bad.stderr
