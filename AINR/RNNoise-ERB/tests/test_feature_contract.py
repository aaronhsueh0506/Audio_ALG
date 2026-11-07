"""Drift guard for the feature constants shared by config.ini, train.py, the C library and the C
tests' shipped grid. Needs cc and audiolib (no torch)."""

import configparser
import math
import os
import pathlib
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from rnnoise_c_build import compile_c  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

PROBE = r"""
#include "process.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    RNNoiseGrid g;
    RNNoiseGridSpec s;
    int status;
    if (argc != 15) return 2;
    s = (RNNoiseGridSpec){
        .sample_rate = atoi(argv[1]), .n_fft = atoi(argv[2]), .n_bands = atoi(argv[3]),
        .min_bins_per_band = atoi(argv[4]), .lookahead_frames = atoi(argv[5]),
        .spec_max_hz = atoi(argv[6]),
        .erb_norm_tau_sec = atof(argv[7]), .erb_norm_init_lo_db = (float)atof(argv[8]),
        .erb_norm_init_hi_db = (float)atof(argv[9]), .erb_norm_scale_db = (float)atof(argv[10]),
        .spec_norm_tau_sec = atof(argv[11]), .spec_norm_init_lo = (float)atof(argv[12]),
        .spec_norm_init_hi = (float)atof(argv[13]), .spec_norm_eps = (float)atof(argv[14]),
    };
    status = rnnoise_grid_make(&g, &s);
    printf("status %d\n", status);
    if (status != 0) return 0;
    printf("hop %d\nn_bins %d\nconv_delay %d\nspec_bins %d\n", g.hop, g.n_bins, g.conv_delay, g.spec_bins);
    printf("erb_norm_alpha %a\nspec_norm_alpha %a\n", (double)g.erb_norm_alpha, (double)g.spec_norm_alpha);
    printf("stft_scale %a\nstft_unscale %a\n", (double)g.stft_scale, (double)g.stft_unscale);
    return 0;
}
"""

_C_FLOAT = re.compile(r"""^[+-]?(
      \d+\.\d*([eE][+-]?\d+)?    # 1.  1.5  1.5e3
    | \.\d+([eE][+-]?\d+)?        # .5
    | \d+[eE][+-]?\d+              # 1e5
)[fF]?$""", re.VERBOSE)


def c_float(text):
    """Parse a C float literal, rejecting ones a C compiler would not accept.

    The `f` suffix requires a FLOATING constant, so `-20f` is a syntax error while `-20.0f` is
    fine; stripping the suffix and calling float() would accept both.
    """
    body = text.strip().strip('()').strip()
    if not _C_FLOAT.match(body):
        raise AssertionError(
            f"{text!r} is not a valid C floating constant "
            f"(an `f` suffix needs a decimal point or exponent)")
    return float(body.rstrip('fF'))


def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]


def norm_alpha(sr, hop_len, tau):
    """train.py make_norm_alpha: three decimals, one more while the result reaches 1.0."""
    exact = math.exp(-(hop_len / sr) / tau)
    precision = 3
    alpha = 1.0
    while alpha >= 1.0:
        alpha = round(exact, precision)
        precision += 1
    return alpha


def shipped_spec(text):
    """The `.field = value` initialisers of RNN_TEST_SHIPPED in the C tests' support header."""
    body = re.search(r'RNN_TEST_SHIPPED\s*=\s*\{(.*?)\};', text, flags=re.DOTALL)
    assert body, 'RNN_TEST_SHIPPED not found'
    fields = dict(re.findall(r'\.(\w+)\s*=\s*([^,]+),', body.group(1)))
    assert fields, 'no designated initialisers in RNN_TEST_SHIPPED'
    return fields


def spec_field_types(header):
    """C type of each field of RNNoiseGridSpec in process.h."""
    body = re.search(r'typedef struct RNNoiseGridSpec\s*\{(.*?)\}\s*RNNoiseGridSpec\s*;',
                     header, flags=re.DOTALL)
    assert body, 'RNNoiseGridSpec not found in process.h'
    types = dict((name, ctype) for ctype, name in re.findall(
        r'^\s*(int|float|double)\s+(\w+)\s*;', body.group(1), flags=re.MULTILINE))
    assert types, 'no fields parsed from RNNoiseGridSpec'
    return types


def stored_value(literal, ctype):
    """The value a field of `ctype` holds after initialisation from the C literal."""
    value = c_float(literal)
    if literal.strip().strip('()').strip().endswith(('f', 'F')):
        value = f32(value)
    return f32(value) if ctype == 'float' else value


def enum_value(text, name):
    match = re.search(rf'\b{re.escape(name)}\s*=\s*(\d+)', text)
    assert match, f'missing enum constant {name}'
    return int(match.group(1))


def main():
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / 'config.ini')
    header = (ROOT / 'process.h').read_text()
    support = (ROOT / 'test' / 'rnnoise_test_support.h').read_text()
    train = (ROOT / 'train.py').read_text()

    # The feature tag: config.ini, train.py and process.h name the same contract.
    version = cfg.get('feature', 'version')
    tag = re.search(r'^#define\s+RNNOISE_FEATURE_VERSION\s+"([^"]*)"', header, flags=re.MULTILINE)
    assert tag and tag.group(1) == version, (tag and tag.group(1), version)
    py_version = re.search(r"^FEATURE_VERSION\s*=\s*['\"]([^'\"]+)['\"]",
                           train, flags=re.MULTILINE)
    assert py_version and py_version.group(1) == version

    sr = cfg.getint('signal', 'sr')
    n_fft = cfg.getint('signal', 'n_fft')
    n_bands = cfg.getint('signal', 'n_bands')
    min_bins = cfg.getint('signal', 'min_bins_per_band', fallback=2)
    lookahead = cfg.getint('signal', 'lookahead_frames')
    win_len = cfg.getint('signal', 'win_len', fallback=n_fft)
    hop_len = cfg.getint('signal', 'hop_len', fallback=win_len // 2)
    spec_max_hz = cfg.getint('feature', 'spec_max_hz')
    # The library has one framing: a full n_fft frame per hop of n_fft / 2.
    assert win_len == n_fft and hop_len == n_fft // 2, (win_len, hop_len, n_fft)

    ints = dict(sample_rate=sr, n_fft=n_fft, n_bands=n_bands, min_bins_per_band=min_bins,
                lookahead_frames=lookahead, spec_max_hz=spec_max_hz)
    floats = dict(
        erb_norm_tau_sec=cfg.getfloat('feature', 'erb_norm_tau_sec'),
        erb_norm_init_lo_db=cfg.getfloat('feature', 'erb_norm_init_lo_db'),
        erb_norm_init_hi_db=cfg.getfloat('feature', 'erb_norm_init_hi_db'),
        erb_norm_scale_db=cfg.getfloat('feature', 'erb_norm_scale_db'),
        spec_norm_tau_sec=cfg.getfloat('feature', 'spec_norm_tau_sec'),
        spec_norm_init_lo=cfg.getfloat('feature', 'spec_norm_init_lo'),
        spec_norm_init_hi=cfg.getfloat('feature', 'spec_norm_init_hi'),
        spec_norm_eps=cfg.getfloat('feature', 'spec_norm_eps'),
    )

    # The C tests' shipped grid is config.ini's.
    shipped = shipped_spec(support)
    assert set(shipped) == set(ints) | set(floats), sorted(set(shipped) ^ (set(ints) | set(floats)))
    for key, want in ints.items():
        assert int(shipped[key]) == want, (key, shipped[key], want)
    types = spec_field_types(header)
    for key, want in floats.items():
        # exact at the precision the struct stores (the C library reads the same float)
        want_stored = f32(want) if types[key] == 'float' else want
        assert stored_value(shipped[key], types[key]) == want_stored, (key, shipped[key], want)
    n_bins = n_fft // 2 + 1
    spec_bins = spec_max_hz * n_fft // sr + 1
    for name, want in (('RNN_TEST_N_FFT', n_fft), ('RNN_TEST_HOP', hop_len),
                       ('RNN_TEST_N_BINS', n_bins), ('RNN_TEST_N_BANDS', n_bands),
                       ('RNN_TEST_SPEC_BINS', spec_bins)):
        assert enum_value(support, name) == want, (name, want)

    # The library derives from those numbers what train.py does.
    with tempfile.TemporaryDirectory(prefix='rnnoise-contract-') as tmp:
        source = pathlib.Path(tmp) / 'probe.c'
        source.write_text(PROBE)
        exe = pathlib.Path(tmp) / 'probe'
        compile_c([source], exe)
        args = [str(ints[k]) for k in ('sample_rate', 'n_fft', 'n_bands', 'min_bins_per_band',
                                       'lookahead_frames', 'spec_max_hz')]
        args += [repr(floats[k]) for k in (
            'erb_norm_tau_sec', 'erb_norm_init_lo_db', 'erb_norm_init_hi_db',
            'erb_norm_scale_db', 'spec_norm_tau_sec', 'spec_norm_init_lo',
            'spec_norm_init_hi', 'spec_norm_eps')]
        out = subprocess.run([str(exe), *args], check=True, capture_output=True, text=True).stdout
    got = dict(line.split() for line in out.splitlines())
    assert got['status'] == '0', got
    assert (int(got['hop']), int(got['n_bins']), int(got['conv_delay']), int(got['spec_bins'])) == (
        hop_len, n_bins, 2 - lookahead, spec_bins), got
    erb_alpha = norm_alpha(sr, hop_len, floats['erb_norm_tau_sec'])
    spec_alpha = norm_alpha(sr, hop_len, floats['spec_norm_tau_sec'])
    # alpha may also be pinned directly in config.ini (frame-invariant); the C library derives
    # it from tau only, so a pinned value must be absent or zero.
    for key in ('erb_norm_alpha', 'spec_norm_alpha'):
        assert cfg.getfloat('feature', key, fallback=0.0) == 0.0, f'{key} pinned in config.ini'
    assert float.fromhex(got['erb_norm_alpha']) == f32(erb_alpha), (got['erb_norm_alpha'], erb_alpha)
    assert float.fromhex(got['spec_norm_alpha']) == f32(spec_alpha), (got['spec_norm_alpha'], spec_alpha)
    assert math.isclose(float.fromhex(got['stft_scale']), 1.0 / math.sqrt(n_fft), rel_tol=1e-6)
    assert math.isclose(float.fromhex(got['stft_unscale']), math.sqrt(n_fft), rel_tol=1e-6)

    print(f'PASS: config.ini, train.py, process.h and the C tests agree on {version}')


def test_main():
    """Expose main() to pytest.

    This file is script-shaped (the Makefile runs it directly); without a ``test_``-prefixed
    entry point pytest would collect zero items from it and report the run green.
    """
    main()


if __name__ == '__main__':
    main()
