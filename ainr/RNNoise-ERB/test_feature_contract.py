"""Dependency-free drift guard for config/Python/C feature constants."""

import configparser
import math
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parent


def macro(text, name):
    match = re.search(rf'^#define\s+{re.escape(name)}\s+(.+?)(?:\s*/\*.*)?$',
                      text, flags=re.MULTILINE)
    if not match:
        raise AssertionError(f'missing C macro: {name}')
    return match.group(1).strip()


def c_float(text):
    return float(text.strip('()').rstrip('fF'))


def main():
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / 'config.ini')
    header = (ROOT / 'process.h').read_text()
    train = (ROOT / 'train.py').read_text()

    version = cfg.get('feature', 'version')
    assert macro(header, 'RNNOISE_FEATURE_VERSION').strip('"') == version
    py_version = re.search(r"^FEATURE_VERSION\s*=\s*['\"]([^'\"]+)['\"]",
                           train, flags=re.MULTILINE)
    assert py_version and py_version.group(1) == version

    pairs = [
        ('erb_center_db', 'RNNOISE_ERB_CENTER_DB'),
        ('erb_scale_db', 'RNNOISE_ERB_SCALE_DB'),
        ('erb_clip', 'RNNOISE_ERB_CLIP'),
        ('spec_norm_tau_sec', 'RNNOISE_SPEC_NORM_TAU_SEC'),
        ('spec_norm_init_lo', 'RNNOISE_SPEC_NORM_INIT_LO'),
        ('spec_norm_init_hi', 'RNNOISE_SPEC_NORM_INIT_HI'),
        ('spec_norm_eps', 'RNNOISE_SPEC_NORM_EPS'),
        ('spec_clip', 'RNNOISE_SPEC_CLIP'),
    ]
    for key, name in pairs:
        got = c_float(macro(header, name))
        want = cfg.getfloat('feature', key)
        assert math.isclose(got, want, rel_tol=0.0, abs_tol=1e-7), (name, got, want)

    spec_max_hz = int(macro(header, 'RNNOISE_SPEC_MAX_HZ').split()[0])
    spec_bins = int(macro(header, 'RNNOISE_SPEC_BINS').split()[0])
    sr = cfg.getint('signal', 'sr')
    n_fft = cfg.getint('signal', 'n_fft')
    hop_len = cfg.getint(
        'signal', 'hop_len', fallback=cfg.getint('signal', 'win_len', fallback=n_fft) // 2)
    assert int(macro(header, 'RNNOISE_SR').split()[0]) == sr
    assert int(macro(header, 'RNNOISE_N_FFT').split()[0]) == n_fft
    assert int(macro(header, 'RNNOISE_HOP_LEN').split()[0]) == hop_len
    assert spec_max_hz == cfg.getint('feature', 'spec_max_hz')
    assert spec_bins == spec_max_hz * n_fft // sr + 1
    assert int(macro(header, 'RNNOISE_N_BANDS').split()[0]) == cfg.getint('signal', 'n_bands')
    assert int(macro(header, 'RNNOISE_LOOKAHEAD').split()[0]) == cfg.getint(
        'signal', 'lookahead_frames')

    print(f'PASS: config.ini, train.py and process.h agree on {version}')


if __name__ == '__main__':
    main()
