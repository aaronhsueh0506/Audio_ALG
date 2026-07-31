"""Dependency-free drift guard for config/Python/C feature constants."""

import configparser
import math
import pathlib
import re


ROOT = pathlib.Path(__file__).resolve().parents[1]


def macro(text, name):
    match = re.search(rf'^#define\s+{re.escape(name)}\s+(.+?)(?:\s*/\*.*)?$',
                      text, flags=re.MULTILINE)
    if not match:
        raise AssertionError(f'missing C macro: {name}')
    return match.group(1).strip()


_C_FLOAT = re.compile(r"""^[+-]?(
      \d+\.\d*([eE][+-]?\d+)?    # 1.  1.5  1.5e3
    | \.\d+([eE][+-]?\d+)?        # .5
    | \d+[eE][+-]?\d+              # 1e5
)[fF]?$""", re.VERBOSE)


def c_float(text):
    """Parse a C float macro, rejecting literals a C compiler would not accept.

    The `f` suffix requires a FLOATING constant, so `-20f` is a syntax error
    while `-20.0f` is fine.  This check used to just strip the suffix and call
    float(), which accepts both -- so a header that could not compile still
    passed the contract test.  Only the builds that #include process.h caught
    it, and those are not what this file is guarding.
    """
    body = text.strip().strip('()').strip()
    if not _C_FLOAT.match(body):
        raise AssertionError(
            f"{text!r} is not a valid C floating constant "
            f"(an `f` suffix needs a decimal point or exponent)")
    return float(body.rstrip('fF'))


def norm_alpha(sr, hop_len, tau):
    exact = math.exp(-(hop_len / sr) / tau)
    precision = 3
    alpha = 1.0
    while alpha >= 1.0:
        alpha = round(exact, precision)
        precision += 1
    return alpha


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
        ('erb_norm_tau_sec', 'RNNOISE_ERB_NORM_TAU_SEC'),
        ('erb_norm_init_lo_db', 'RNNOISE_ERB_NORM_INIT_LO_DB'),
        ('erb_norm_init_hi_db', 'RNNOISE_ERB_NORM_INIT_HI_DB'),
        ('erb_norm_scale_db', 'RNNOISE_ERB_NORM_SCALE_DB'),
        ('spec_norm_tau_sec', 'RNNOISE_SPEC_NORM_TAU_SEC'),
        ('spec_norm_init_lo', 'RNNOISE_SPEC_NORM_INIT_LO'),
        ('spec_norm_init_hi', 'RNNOISE_SPEC_NORM_INIT_HI'),
        ('spec_norm_eps', 'RNNOISE_SPEC_NORM_EPS'),
    ]
    for key, name in pairs:
        got = c_float(macro(header, name))
        want = cfg.getfloat('feature', key)
        assert math.isclose(got, want, rel_tol=0.0, abs_tol=1e-7), (name, got, want)

    spec_max_hz = int(macro(header, 'RNNOISE_SPEC_MAX_HZ').split()[0])
    spec_bins = int(macro(header, 'RNNOISE_SPEC_BINS').split()[0])
    sr = cfg.getint('signal', 'sr')
    n_fft = cfg.getint('signal', 'n_fft')
    win_len = cfg.getint('signal', 'win_len', fallback=n_fft)
    hop_len = cfg.getint(
        'signal', 'hop_len', fallback=win_len // 2)
    assert int(macro(header, 'RNNOISE_SR').split()[0]) == sr
    assert int(macro(header, 'RNNOISE_N_FFT').split()[0]) == n_fft
    assert int(macro(header, 'RNNOISE_WIN_LEN').split()[0]) == win_len
    assert int(macro(header, 'RNNOISE_HOP_LEN').split()[0]) == hop_len
    erb_alpha = c_float(macro(header, 'RNNOISE_ERB_NORM_ALPHA'))
    spec_alpha = c_float(macro(header, 'RNNOISE_SPEC_NORM_ALPHA'))
    # alpha may be pinned directly (frame-invariant: memory stays 1/(1-alpha)
    # frames across any sr/hop) or derived from tau (seconds-invariant).  An
    # explicit alpha wins; 0 or absent means "derive".
    def expected_alpha(alpha_key, tau_key):
        explicit = cfg.getfloat('feature', alpha_key, fallback=0.0)
        if explicit > 0.0:
            return explicit
        return norm_alpha(sr, hop_len, cfg.getfloat('feature', tau_key))

    assert math.isclose(
        erb_alpha, expected_alpha('erb_norm_alpha', 'erb_norm_tau_sec'),
        rel_tol=0.0, abs_tol=1e-7)
    assert math.isclose(
        spec_alpha, expected_alpha('spec_norm_alpha', 'spec_norm_tau_sec'),
        rel_tol=0.0, abs_tol=1e-7)
    assert spec_max_hz == cfg.getint('feature', 'spec_max_hz')
    assert spec_bins == spec_max_hz * n_fft // sr + 1
    assert int(macro(header, 'RNNOISE_N_BANDS').split()[0]) == cfg.getint('signal', 'n_bands')
    assert int(macro(header, 'RNNOISE_LOOKAHEAD').split()[0]) == cfg.getint(
        'signal', 'lookahead_frames')
    # Changing min_bins_per_band moves the ERB band borders, which changes the
    # filterbank matrices and invalidates every checkpoint and generated table.
    # It used to be a bare literal in train.py and gen_rnnoise_tables.c with no
    # guard at all.
    assert int(macro(header, 'RNNOISE_MIN_BINS_PER_BAND').split()[0]) == cfg.getint(
        'signal', 'min_bins_per_band', fallback=2)

    print(f'PASS: config.ini, train.py and process.h agree on {version}')


def test_main():
    """Expose main() to pytest.

    This file is script-shaped (Makefile runs it directly), so without a
    ``test_``-prefixed entry point pytest collected ZERO items from it and
    still reported the run green -- the C/Python parity and feature-contract
    guards were simply absent from any `pytest` invocation.
    """
    main()


if __name__ == '__main__':
    main()
