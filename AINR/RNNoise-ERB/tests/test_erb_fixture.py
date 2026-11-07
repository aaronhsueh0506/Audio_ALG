"""The ERB matrix writer the C tests use (standard library) against train.py's filterbank and
against export_erb_matrix.py's matrices, bit for bit. Needs the training Python environment."""

import configparser
import pathlib
import subprocess
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

# Each model project has its own top-level train.py; drop any cached sibling.
for _stale in ('train', 'inference', 'model', 'checkpoint_utils', 'export_erb_matrix'):
    sys.modules.pop(_stale, None)

from train import compute_erb_matrix, erb_bandborder  # noqa: E402
from erb_fixture import erb_files, erb_files_from_config, erb_files_from_params  # noqa: E402

# sample rate, n_fft, n_bands, min_bins_per_band
GRIDS = [
    (16000, 512, 22, 2),
    (16000, 256, 18, 2),
    (48000, 1024, 32, 2),
    (8000, 128, 16, 1),
    (32000, 768, 26, 3),
]


def train_files(sr, n_fft, n_bands, min_bins):
    borders = erb_bandborder(n_bands, sr, n_fft, min_bins)
    return erb_files(compute_erb_matrix(borders, n_fft, mode=0),
                     compute_erb_matrix(borders, n_fft, mode=1))


def test_fixture_matches_train_py_on_every_grid():
    for grid in GRIDS:
        assert erb_files_from_params(*grid) == train_files(*grid), grid


def _config_path(tmp, sr, n_fft, n_bands, min_bins):
    path = pathlib.Path(tmp) / 'config.ini'
    cfg = configparser.ConfigParser()
    assert cfg.read(ROOT / 'config.ini')
    cfg['signal']['sr'] = str(sr)
    cfg['signal']['n_fft'] = str(n_fft)
    cfg['signal']['win_len'] = str(n_fft)
    cfg['signal']['hop_len'] = str(n_fft // 2)
    cfg['signal']['n_bands'] = str(n_bands)
    cfg['signal']['min_bins_per_band'] = str(min_bins)
    with open(path, 'w') as f:
        cfg.write(f)
    return path, cfg


def test_fixture_matches_the_exporter_matrices():
    """The exporter's own matrices, written as the two raw files, are the fixture."""
    for grid in GRIDS[:3]:
        with tempfile.TemporaryDirectory(prefix='rnnoise-erb-') as tmp:
            config, cfg = _config_path(tmp, *grid)
            out = pathlib.Path(tmp) / 'out'
            subprocess.run([sys.executable, '-B', str(ROOT / 'export_erb_matrix.py'),
                            '--config', str(config), '--format', 'npy',
                            '--output-dir', str(out)],
                           check=True, capture_output=True, cwd=ROOT)
            fwd = np.load(out / 'erb_fwd.npy')
            inv = np.load(out / 'erb_inv.npy')
            assert erb_files(fwd, inv) == erb_files_from_config(cfg), grid


def test_config_ini_is_the_shipped_grid():
    cfg = configparser.ConfigParser()
    assert cfg.read(ROOT / 'config.ini')
    sr, n_fft = cfg.getint('signal', 'sr'), cfg.getint('signal', 'n_fft')
    n_bands = cfg.getint('signal', 'n_bands')
    min_bins = cfg.getint('signal', 'min_bins_per_band', fallback=2)
    assert erb_files_from_config(cfg) == train_files(sr, n_fft, n_bands, min_bins)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
    print('PASS: ERB matrix writer matches train.py and export_erb_matrix.py on every grid')


if __name__ == '__main__':
    main()
