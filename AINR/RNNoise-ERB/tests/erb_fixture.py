"""Writer of the two ERB matrix files for the C tests (standard library only).

erb_fwd.bin: float32 [n_bins][n_erb] (bin-major); erb_inv.bin: float32 [n_erb][n_bins]
(band-major). Little-endian, no header.

The triangular ERB filterbank is computed here from the formulas of train.py, so the C tests need
no numpy or torch. tests/test_erb_fixture.py pins this file to train.py and to the exporter.

    python3 erb_fixture.py [--config config.ini] [--fwd FILE] [--inv FILE]
"""

import argparse
import configparser
import math
import pathlib
import struct

ROOT = pathlib.Path(__file__).resolve().parents[1]


def f32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def freq2erb(freq_hz):
    return 9.265 * math.log(1.0 + freq_hz / (24.7 * 9.265))


def erb2freq(n_erb):
    return 24.7 * 9.265 * (math.exp(n_erb / 9.265) - 1.0)


def erb_bandborder(n_bands, sr, n_fft, min_bins_per_band):
    """FFT-bin band borders (n_bands entries): ideal ERB-scale borders, each at least
    min_bins_per_band after the previous one, the last pinned to n_fft // 2 + 1."""
    assert n_bands >= 2, n_bands
    n_bins = n_fft // 2 + 1
    high_lim = sr / 2.0
    bw = high_lim / (n_fft / 2.0)
    e_lo = freq2erb(0.0)
    step = (freq2erb(high_lim) - e_lo) / (n_bands - 1)
    borders = [0]
    for i in range(1, n_bands):
        ideal = int(round((erb2freq(e_lo + i * step) + bw / 2.0) / bw))
        borders.append(min(max(ideal, borders[-1] + min_bins_per_band), n_bins))
    borders[-1] = n_bins
    return borders


def erb_matrix(borders, n_fft, mode):
    """float32 (as Python floats) rows [n_bins][n_bands]. mode 0: forward, the two one-sided edge
    bands doubled; mode 1: inverse, a partition of unity."""
    n_bins = n_fft // 2 + 1
    n = len(borders)
    rows = [[0.0] * n for _ in range(n_bins)]
    for i in range(n - 1):
        lo, hi = borders[i], borders[i + 1]
        width = hi - lo
        for j in range(width):
            rows[lo + j][i] = f32(1.0 - j / width)
            rows[lo + j][i + 1] = f32(j / width)
    if mode == 0:
        for row in rows:
            row[0] *= 2.0
            row[n - 1] *= 2.0
    return rows


def erb_files(fwd_bin_major, inv_bin_major):
    """(forward, inverse) file bytes from the two [n_bins][n_erb] matrices (rows of floats or an
    ndarray); the inverse is written band-major."""
    fwd = [[float(v) for v in row] for row in fwd_bin_major]
    inv = [[float(v) for v in row] for row in inv_bin_major]
    n_bins, n_erb = len(fwd), len(fwd[0])
    assert len(inv) == n_bins and all(len(r) == n_erb for r in fwd + inv), (len(inv), n_bins)
    fwd_bytes = b''.join(struct.pack(f'<{n_erb}f', *row) for row in fwd)
    inv_bytes = b''.join(struct.pack(f'<{n_bins}f', *col) for col in zip(*inv))
    return fwd_bytes, inv_bytes


def erb_files_from_params(sr, n_fft, n_bands, min_bins):
    borders = erb_bandborder(n_bands, sr, n_fft, min_bins)
    return erb_files(erb_matrix(borders, n_fft, 0), erb_matrix(borders, n_fft, 1))


def erb_files_from_config(cfg):
    """(forward, inverse) file bytes for the [signal] section of a parsed config.ini."""
    return erb_files_from_params(cfg.getint('signal', 'sr'), cfg.getint('signal', 'n_fft'),
                                 cfg.getint('signal', 'n_bands'),
                                 cfg.getint('signal', 'min_bins_per_band', fallback=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default=str(ROOT / 'config.ini'))
    parser.add_argument('--fwd', help='forward matrix file to write')
    parser.add_argument('--inv', help='inverse matrix file to write')
    args = parser.parse_args()
    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(args.config)
    if not (args.fwd or args.inv):
        parser.error('give --fwd FILE, --inv FILE or both')
    fwd, inv = erb_files_from_config(cfg)
    for path, data in ((args.fwd, fwd), (args.inv, inv)):
        if path:
            pathlib.Path(path).write_bytes(data)
            print(f'wrote {path} ({len(data)} bytes)')


if __name__ == '__main__':
    main()
