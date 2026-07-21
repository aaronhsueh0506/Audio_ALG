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
        ('norm_tau_sec', 'RNNOISE_NORM_TAU_SEC'),
        ('mean_init_db', 'RNNOISE_NORM_MEAN_INIT_DB'),
        ('clip', 'RNNOISE_NORM_CLIP'),
    ]
    for key, name in pairs:
        got = c_float(macro(header, name))
        want = cfg.getfloat('feature', key)
        assert math.isclose(got, want, rel_tol=0.0, abs_tol=1e-7), (name, got, want)

    var_init = c_float(macro(header, 'RNNOISE_NORM_VAR_INIT_DB2'))
    var_floor = c_float(macro(header, 'RNNOISE_NORM_VAR_FLOOR_DB2'))
    assert math.isclose(var_init, cfg.getfloat('feature', 'std_init_db') ** 2)
    assert math.isclose(var_floor, cfg.getfloat('feature', 'std_floor_db') ** 2)
    assert int(macro(header, 'RNNOISE_N_BANDS').split()[0]) == cfg.getint('signal', 'n_bands')
    assert int(macro(header, 'RNNOISE_LOOKAHEAD').split()[0]) == cfg.getint(
        'signal', 'lookahead_frames')

    print(f'PASS: config.ini, train.py and process.h agree on {version}')


if __name__ == '__main__':
    main()
