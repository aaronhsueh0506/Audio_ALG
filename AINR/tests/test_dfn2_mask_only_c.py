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
* a deep-filter lookahead other than 0 is a compile error.

The comparison is against numpy on the same ERB matrices the C class loads; the
check is shown to fail on a mask taken one frame off.
"""

import subprocess
import sys

import numpy as np
import pytest

import dfn_pairs

FAMILY = 'DeepFilterNet2'
FRAMES = 40
N_BINS = 513
N_ERB = 32

_DRIVER = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "DeepFilterNet2/dfn2_prepost.h"

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

    CHECK(dfn2_model_io_descriptor_default(&descriptor) == 0);
    CHECK(descriptor.layout_version ==
          (unsigned int)DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY);
    CHECK(descriptor.state_tensor_count == 1);
    CHECK(dfn2_model_io_descriptor_validate(&descriptor) == 0);
    descriptor.layout_version = 8u;          /* the full build's graph */
    CHECK(dfn2_model_io_descriptor_validate(&descriptor) != 0);

    CHECK(dfn2_prepost_config_defaults(&cfg, DFN2_IO_FREQ) == 0);
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;
    cfg.atten_lim_db = (float)atof(argv[6]);
    p = dfn2_prepost_create(&cfg);
    CHECK(p != NULL);
    CHECK(dfn2_prepost_layout_version(p) ==
          DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY);
    CHECK(dfn2_prepost_model_lookahead_frames(p) == DFN2_MASK_LOOKAHEAD);

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
        CHECK(valid == (t >= DFN2_PREPOST_LATENCY_FRAMES ? 1 : 0));
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


@pytest.fixture(scope='module')
def driver(tmp_path_factory):
    reason = dfn_pairs.require_toolchain()
    if reason:
        pytest.skip(reason)
    lib = dfn_pairs.audio_common_lib()
    work = tmp_path_factory.mktemp('dfn2_mask_only_c')
    source = work / 'driver.c'
    source.write_text(_DRIVER, encoding='utf-8')
    executable = work / 'driver'
    subprocess.run(
        ['cc', *dfn_pairs.house_cflags(), '-DDFN2_MASK_ONLY', str(source),
         *[str(dfn_pairs.source_path(FAMILY, kind))
           for kind in ('prepost', 'process', 'model_io')],
         lib, '-lm', '-o', str(executable)],
        check=True, capture_output=True)
    return executable


@pytest.fixture(scope='module')
def erb_dir(tmp_path_factory):
    sys.path.insert(0, str(dfn_pairs.family_dir(FAMILY)))
    from export_erb_matrix import write_runtime_bins
    out = tmp_path_factory.mktemp('erb')
    write_runtime_bins(str(dfn_pairs.family_dir(FAMILY) / 'config.ini'),
                       str(out))
    return out


def _run(driver, erb_dir, tmp_path, atten_db):
    rng = np.random.default_rng(11)
    spec = (rng.standard_normal((FRAMES, 2, N_BINS)) * 0.05).astype('<f4')
    mask = rng.uniform(0.0, 1.0, (FRAMES, N_ERB)).astype('<f4')
    spec.tofile(tmp_path / 'spec.bin')
    mask.tofile(tmp_path / 'mask.bin')
    subprocess.run([str(driver), str(erb_dir), str(tmp_path / 'spec.bin'),
                    str(tmp_path / 'mask.bin'), str(tmp_path / 'out.bin'),
                    str(FRAMES), str(atten_db)],
                   check=True, capture_output=True)
    raw = (tmp_path / 'out.bin').read_bytes()
    record = 4 + 2 * N_BINS * 4
    assert len(raw) % record == 0
    frames, outputs = [], []
    for at in range(0, len(raw), record):
        frames.append(int(np.frombuffer(raw[at:at + 4], '<i4')[0]))
        body = np.frombuffer(raw[at + 4:at + record], '<f4')
        outputs.append(body.reshape(2, N_BINS))
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
def test_mask_only_output_is_the_masked_noisy_spectrum(
        driver, erb_dir, tmp_path, atten_db):
    spec, mask, frames, got = _run(driver, erb_dir, tmp_path, atten_db)
    inverse = np.fromfile(erb_dir / 'erb_inv.bin', '<f4').reshape(N_ERB, N_BINS)
    lookahead = 1                                  # DFN2_MASK_LOOKAHEAD default
    # one output per hop after MASK_LOOKAHEAD hops, in frame order, no deep
    # filter delay on top
    assert frames == list(range(0, FRAMES - lookahead))
    want = _expected(spec, mask, inverse, frames, atten_db)
    assert np.allclose(got, want, rtol=1e-5, atol=1e-7)
    # the check can fail: a mask taken one frame off is not within tolerance
    off = _expected(spec, mask, inverse, frames[:-1], atten_db, shift=1)
    assert not np.allclose(got[:-1], off, rtol=1e-5, atol=1e-7)


def test_a_deep_filter_lookahead_is_a_compile_error(tmp_path):
    only = ['-DDFN2_MASK_ONLY']
    ok = dfn_pairs.header_compiles(FAMILY, (1, 0), 'c11', tmp_path, extra=only)
    assert ok.returncode == 0, ok.stderr
    bad = dfn_pairs.header_compiles(FAMILY, (1, 1), 'c11', tmp_path,
                                    extra=only)
    assert bad.returncode != 0
    assert 'DFN2_DF_LOOKAHEAD must be 0' in bad.stderr


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
