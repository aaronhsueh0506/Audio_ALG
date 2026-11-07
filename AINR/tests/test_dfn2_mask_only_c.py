"""The C host built with -DDFN2_MASK_ONLY: stage 1 alone (ERB mask, no deep filter).

The build is the model of ``export_onnx.py --mask-only``. Gates:

* the contract the build publishes: layout 9, one state tensor (h_gru, three rows,
  registered by the host), no coefficient/alpha head and no pathway window in the
  accelerator boundary structs (they do not compile in), descriptor of the full
  build refused;
* the output of the class is ``noisy * (mask @ erb_inv)`` of the head frame, the
  arithmetic of ``DeepFilterNet2.compose_mask_only`` (the Python reference), with
  and without an attenuation limit, one hop per frame and MASK_LOOKAHEAD hops of
  latency;
* a deep-filter lookahead other than 0 is refused by the grid constructor of the
  mask-only build (``dfn2_grid_make``); a mask lookahead of 2 gives a ring of 3.

The library holds no grid, so the host is run on several: the shipped one
(config.ini, no override), the 16 kHz and 32 kHz grids of ``dfn2_test_grid.h``
and mask lookaheads 0, 1 and 2. The ERB matrices reach the C host the way a
deployment hands them over: two headerless little-endian float32 files, forward
bin-major [n_bins][n_erb] and inverse band-major [n_erb][n_bins], written here
from the model the config builds; the host reads each with an exact-size check
against its grid. The comparison is against numpy on the same matrices; the
check is shown to fail on a mask taken one frame off. The grid rules are probed from a program built both
ways, so the refusal is shown to belong to the mask-only build.
"""

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
# others change the grid or the lookahead.
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

/* Headerless float32 file of exactly count values; any other size is fatal. */
static float *load_matrix(const char *path, size_t count) {
    FILE *f = fopen(path, "rb");
    float *data = (float *)malloc(count * sizeof(float));
    unsigned char extra;
    if (!f || !data || fread(data, sizeof(float), count, f) != count ||
        fread(&extra, 1, 1, f) != 0) exit(3);
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
    static float gru_hidden[DFN2_TEST_GRU_ELEMENTS];
    FILE *out;
    int frames, t, valid;
    long long frame;

    if (argc != 8) return 2;
    frames = atoi(argv[6]);
    erb_fwd = load_matrix(argv[1], (size_t)grid->n_bins * grid->n_erb);
    erb_inv = load_matrix(argv[2], (size_t)grid->n_bins * grid->n_erb);
    spec = load(argv[3], (size_t)frames * 2 * DFN2_N_BINS);
    mask = load(argv[4], (size_t)frames * DFN2_N_ERB);

    CHECK(dfn2_model_io_descriptor_default(grid, &descriptor) == 0);
    CHECK(descriptor.layout_version ==
          (unsigned int)DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY);
    CHECK(descriptor.state_tensor_count == 1);
    CHECK(descriptor.mask_lookahead == DFN2_MASK_LOOKAHEAD &&
          descriptor.df_lookahead == 0);
    CHECK(dfn2_model_io_descriptor_validate(grid, &descriptor) == 0);
    /* The grid is part of the contract: a graph described for this grid is
     * refused by an instance that runs another one. */
    {
        DFN2GridSpec other_spec = dfn2_test_grid_spec();
        other_spec.n_erb = DFN2_N_ERB - 4;
        CHECK(dfn2_grid_make(&other, &other_spec) == DFN2_GRID_OK);
    }
    CHECK(dfn2_model_io_descriptor_validate(&other, &descriptor) != 0);
    descriptor.layout_version = 8u;          /* the full build's graph */
    CHECK(dfn2_model_io_descriptor_validate(grid, &descriptor) != 0);

    CHECK(dfn2_prepost_config_defaults(&cfg, grid) == 0);
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;
    cfg.gru_hidden = gru_hidden;
    cfg.gru_hidden_elements = DFN2_TEST_GRU_ELEMENTS;
    cfg.atten_lim_db = (float)atof(argv[7]);
    p = dfn2_prepost_create(&cfg);
    CHECK(p != NULL);
    CHECK(dfn2_prepost_layout_version(p) ==
          DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY);
    CHECK(dfn2_prepost_model_lookahead_frames(p) == DFN2_MASK_LOOKAHEAD);
    CHECK(dfn2_prepost_grid(p) != NULL &&
          dfn2_prepost_grid(p)->n_erb == DFN2_N_ERB);

    out = fopen(argv[5], "wb");
    CHECK(out != NULL);
    for (t = 0; t < frames; ++t) {
        const float *frame_re = spec + (size_t)t * 2 * DFN2_N_BINS;
        const float *frame_im = frame_re + DFN2_N_BINS;
        const unsigned int generation = dfn2_prepost_state_generation(p);
        int warm = t < DFN2_MASK_LOOKAHEAD;
        size_t i;
        /* every hop opens a frame; the first DFN2_MASK_LOOKAHEAD are warm-up frames */
        CHECK(dfn2_prepost_pre_process_freq(p, frame_re, frame_im) == 1);
        CHECK(dfn2_prepost_frame_open(p, &inputs, &outputs) == 0);
        CHECK(inputs.gru_hidden == gru_hidden &&
              inputs.gru_hidden_elements == DFN2_TEST_GRU_ELEMENTS);
        CHECK(outputs.gru_hidden_next == gru_hidden);
        CHECK(outputs.erb_mask_elements == DFN2_N_ERB);
        if (warm) {
            /* the head stays NaN and the accelerator scribbles on the state: finish reads
             * no head and leaves the state zero, the generation advances once */
            for (i = 0; i < DFN2_TEST_GRU_ELEMENTS; ++i) gru_hidden[i] = 99.0f;
        } else {
            memcpy(outputs.erb_mask,
                   mask + (size_t)(t - DFN2_MASK_LOOKAHEAD) * DFN2_N_ERB,
                   DFN2_N_ERB * sizeof(float));
        }
        CHECK(dfn2_prepost_frame_finish(p) == 0);
        if (warm) {
            for (i = 0; i < DFN2_TEST_GRU_ELEMENTS; ++i) CHECK(gru_hidden[i] == 0.0f);
            CHECK(dfn2_prepost_state_generation(p) == generation + 1u);
        } else {
            CHECK(dfn2_prepost_state_generation(p) == generation);
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
    free(erb_fwd);
    free(erb_inv);
    free(spec);
    free(mask);
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
    """The ERB matrices of a grid as the two raw files an application provisions, from the model
    the grid's config builds (the matrices the exporter writes): returns (forward file, inverse
    file, forward [n_bins][n_erb], inverse [n_erb][n_bins])."""
    # The families share module names (model, train, ...): whichever another test imported
    # first would answer for this one, and this one must not leave its own behind for the next.
    family_modules = ('train', 'inference', 'model', 'checkpoint_utils',
                      'export_onnx', 'stream_model')
    for name in family_modules:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(dfn_pairs.family_dir(FAMILY)))
    try:
        import torch
        from model import DeepFilterNet2
        from train import read_model_config
    finally:
        sys.path.remove(str(dfn_pairs.family_dir(FAMILY)))
        for name in family_modules:
            sys.modules.pop(name, None)
    built = {}

    def get(khz):
        if khz not in built:
            sr, n_fft, n_erb, df_bins = GRIDS[khz]
            cfg = dfn_pairs.dfn2_grid_config(sr, n_fft, n_erb, df_bins)
            with torch.no_grad():
                model = DeepFilterNet2(**read_model_config(cfg))
                forward = model.erb_fb.detach().numpy().astype('<f4')
                inverse = model.erb_inv.detach().numpy().astype('<f4')
            if forward.shape[0] < forward.shape[1]:
                forward = np.ascontiguousarray(forward.T)
            if inverse.shape[0] > inverse.shape[1]:
                inverse = np.ascontiguousarray(inverse.T)
            n_bins, n_erb_rows = forward.shape
            assert inverse.shape == (n_erb_rows, n_bins)
            folder = tmp_path_factory.mktemp('erb')
            fwd_path, inv_path = folder / 'erb_fwd.bin', folder / 'erb_inv.bin'
            fwd_path.write_bytes(np.ascontiguousarray(forward, '<f4').tobytes())
            inv_path.write_bytes(np.ascontiguousarray(inverse, '<f4').tobytes())
            built[khz] = (fwd_path, inv_path, forward, inverse)
        return built[khz]
    return get


def _run(driver, erb_fwd, erb_inv, tmp_path, atten_db, n_bins, n_erb):
    rng = np.random.default_rng(11)
    spec = (rng.standard_normal((FRAMES, 2, n_bins)) * 0.05).astype('<f4')
    mask = rng.uniform(0.0, 1.0, (FRAMES, n_erb)).astype('<f4')
    spec.tofile(tmp_path / 'spec.bin')
    mask.tofile(tmp_path / 'mask.bin')
    subprocess.run([str(driver), str(erb_fwd), str(erb_inv), str(tmp_path / 'spec.bin'),
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
    erb_fwd, erb_inv, _forward, inverse = erb_dir(khz)
    spec, mask, frames, got = _run(driver(khz, lookahead), erb_fwd, erb_inv,
                                   tmp_path, atten_db, n_bins, n_erb)
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
# mask_lookahead, df_lookahead), the names of the statuses and the ring.
_GRID_PROBE = r'''
#include <stdio.h>
#include <stdlib.h>

#include "DeepFilterNet2/dfn2_grid.h"

int main(int argc, char **argv) {
    DFN2GridSpec spec = {
        .df_order = @ORDER@, .frames_per_call = 1,
        .erb_norm_tau_sec = @ERB_TAU@, .erb_norm_init_lo_db = @ERB_LO@,
        .erb_norm_init_hi_db = @ERB_HI@, .erb_norm_scale_db = @ERB_SCALE@,
        .spec_norm_tau_sec = @SPEC_TAU@, .spec_norm_init_lo = @SPEC_LO@,
        .spec_norm_init_hi = @SPEC_HI@,
    };
    DFN2Grid g;
    int status;
    if (argc != 7) return 2;
    spec.sample_rate = atoi(argv[1]);
    spec.n_fft = atoi(argv[2]);
    spec.n_erb = atoi(argv[3]);
    spec.df_bins = atoi(argv[4]);
    spec.mask_lookahead = atoi(argv[5]);
    spec.df_lookahead = atoi(argv[6]);
    status = dfn2_grid_make(&g, &spec);
    printf("%d %d %d %d %s\n", status, DFN2_GRID_OK, DFN2_GRID_ERR_DF_LOOKAHEAD,
           status == DFN2_GRID_OK ? g.ring : -1, dfn2_grid_status_text(status));
    return 0;
}
'''


def _grid_probe_source():
    values = dfn_pairs.dfn2_spec_values()
    text = _GRID_PROBE
    for token, key in (('ERB_TAU', 'erb_norm_tau_sec'), ('ERB_LO', 'erb_norm_init_lo_db'),
                       ('ERB_HI', 'erb_norm_init_hi_db'), ('ERB_SCALE', 'erb_norm_scale_db'),
                       ('SPEC_TAU', 'spec_norm_tau_sec'), ('SPEC_LO', 'spec_norm_init_lo'),
                       ('SPEC_HI', 'spec_norm_init_hi')):
        text = text.replace('@%s@' % token, repr(values[key]) + 'f')
    return text.replace('@ORDER@', str(values['df_order']))


@pytest.fixture(scope='module')
def grid_probe(tmp_path_factory):
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    lib = dfn_pairs.audio_common_lib()
    work = tmp_path_factory.mktemp('dfn2_grid_probe')
    source = work / 'probe.c'
    source.write_text(_grid_probe_source(), encoding='utf-8')
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
                status, ok, df_la, ring = map(int, done.stdout.split()[:4])
                return status, {'ok': ok, 'df_lookahead': df_la, 'ring': ring}
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


def test_the_mask_only_ring_is_the_mask_lookahead_plus_one(grid_probe):
    """The mask-only state keeps only the noisy ring, mask_lookahead + 1 frames; no size
    bounds it, so a mask lookahead of 2 is accepted with a ring of 3."""
    mask_only = grid_probe('-DDFN2_MASK_ONLY')
    for mask in (0, 1, 2):
        status, names = mask_only(48000, 1024, 32, 96, mask, 0)
        assert status == names['ok'] and names['ring'] == mask + 1, (mask, status, names)


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
