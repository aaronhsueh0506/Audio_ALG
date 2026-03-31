"""
匯出 ERB 轉換矩陣

用法:
    python export_erb_matrix.py --config config.ini --format all
    python export_erb_matrix.py --config config.ini --format npy
    python export_erb_matrix.py --config config.ini --format c
"""

import argparse
import configparser
import numpy as np


def erb_rate(f):
    """頻率 (Hz) → ERB-rate (Glasberg & Moore 1990)"""
    return 21.4 * np.log10(0.00437 * f + 1)


def erb_inv(e):
    """ERB-rate → 頻率 (Hz)"""
    return (10 ** (e / 21.4) - 1) / 0.00437


def compute_erb_bands(n_fft, sr, n_bands):
    """計算 ERB band 的 FFT bin 邊界"""
    n_bins = n_fft // 2 + 1
    e_low = erb_rate(0)
    e_high = erb_rate(sr / 2)
    erb_edges = np.linspace(e_low, e_high, n_bands + 1)
    freq_edges = erb_inv(erb_edges)
    bin_edges = np.round(freq_edges / (sr / n_fft)).astype(int)
    bin_edges = np.clip(bin_edges, 0, n_bins - 1)
    for i in range(1, len(bin_edges)):
        if bin_edges[i] <= bin_edges[i - 1]:
            bin_edges[i] = bin_edges[i - 1] + 1
    bin_edges[-1] = min(bin_edges[-1], n_bins)
    return bin_edges


def compute_erb_matrix(n_fft, n_bands, bin_edges=None):
    """建構 ERB 轉換矩陣 W, shape = (n_bins, n_bands)"""
    if bin_edges is None:
        raise ValueError("bin_edges required")
    n_bins = n_fft // 2 + 1
    W = np.zeros((n_bins, n_bands), dtype=np.float32)
    for b in range(n_bands):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        W[lo:hi, b] = 1.0
    return W


def export_npy(W, output_path):
    np.save(output_path, W)
    print(f"已儲存: {output_path}")


def export_c_header(W, bin_edges, output_path, sr, n_fft):
    n_bins, n_bands = W.shape
    with open(output_path, 'w') as f:
        f.write("/* ERB 轉換矩陣 - 自動產生 */\n")
        f.write(f"/* N_BINS={n_bins}, N_BANDS={n_bands}, "
                f"N_FFT={n_fft}, SR={sr} */\n\n")
        f.write("#ifndef ERB_MATRIX_H\n")
        f.write("#define ERB_MATRIX_H\n\n")

        f.write(f"#define ERB_N_BINS {n_bins}\n")
        f.write(f"#define ERB_N_BANDS {n_bands}\n\n")

        # bin edges
        f.write(f"static const int ERB_BIN_EDGES[{n_bands + 1}] = {{\n    ")
        f.write(", ".join(str(int(e)) for e in bin_edges))
        f.write("\n};\n\n")

        # 完整矩陣 (row-major: W[bin][band])
        f.write(f"/* W[n_bins][n_bands] - row major */\n")
        f.write(f"static const float ERB_MATRIX[{n_bins}][{n_bands}] = {{\n")
        for i in range(n_bins):
            row = ", ".join(f"{v:.1f}f" for v in W[i])
            f.write(f"    {{{row}}}")
            if i < n_bins - 1:
                f.write(",")
            f.write(f"  /* bin {i} */\n")
        f.write("};\n\n")

        # 轉置矩陣
        f.write(f"/* W_T[n_bands][n_bins] - for backward expansion */\n")
        f.write(f"static const float ERB_MATRIX_T[{n_bands}][{n_bins}] = {{\n")
        W_T = W.T
        for b in range(n_bands):
            row = ", ".join(f"{v:.1f}f" for v in W_T[b])
            f.write(f"    {{{row}}}")
            if b < n_bands - 1:
                f.write(",")
            f.write(f"  /* band {b} */\n")
        f.write("};\n\n")

        f.write("#endif /* ERB_MATRIX_H */\n")

    print(f"已儲存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='匯出 ERB 轉換矩陣')
    parser.add_argument('--config', default='config.ini', help='Config 檔案路徑')
    parser.add_argument('--format', choices=['npy', 'c', 'all'], default='all')
    parser.add_argument('--output-dir', default=None,
                        help='輸出目錄 (預設從 config 讀取)')
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    SR = cfg.getint('signal', 'sr')
    N_FFT = cfg.getint('signal', 'n_fft')
    HYBRID_CUTOFF = cfg.getint('signal', 'hybrid_cutoff_hz', fallback=0)
    N_ERB_HIGH = cfg.getint('signal', 'n_erb_high_bands', fallback=0)

    output_dir = args.output_dir or cfg.get('paths', 'output_dir', fallback='output')

    import os
    os.makedirs(output_dir, exist_ok=True)

    if HYBRID_CUTOFF > 0 and N_ERB_HIGH > 0:
        from train import compute_hybrid_bands as _compute_hybrid
        bin_edges, N_BANDS = _compute_hybrid(N_FFT, SR, N_ERB_HIGH, HYBRID_CUTOFF)
        print(f"Hybrid mode: {HYBRID_CUTOFF}Hz cutoff, {N_BANDS} total bands")
    else:
        N_BANDS = cfg.getint('signal', 'n_bands')
        bin_edges = compute_erb_bands(N_FFT, SR, N_BANDS)
    W = compute_erb_matrix(N_FFT, N_BANDS, bin_edges)
    n_bins, n_bands = W.shape

    print(f"Config: SR={SR}, N_FFT={N_FFT}, N_BANDS={N_BANDS}")
    print(f"ERB Matrix: ({n_bins}, {n_bands})")
    print(f"Bin edges: {bin_edges.tolist()}")
    print(f"Non-zero entries: {int(W.sum())}")
    print()

    if args.format in ('npy', 'all'):
        export_npy(W, os.path.join(output_dir, 'erb_matrix.npy'))

    if args.format in ('c', 'all'):
        export_c_header(W, bin_edges, os.path.join(output_dir, 'erb_matrix.h'),
                        SR, N_FFT)

    # 驗證
    print("\n=== 驗證 ===")
    power = np.random.rand(n_bins).astype(np.float32)
    band_energy = power @ W
    print(f"Forward: power ({n_bins},) @ W ({n_bins},{n_bands}) "
          f"→ band_energy ({n_bands},)")

    band_gains = np.random.rand(n_bands).astype(np.float32)
    bin_gains = W @ band_gains
    print(f"Backward: W ({n_bins},{n_bands}) @ band_gains ({n_bands},) "
          f"→ bin_gains ({n_bins},)")

    # 驗證與 for-loop 版本一致
    band_energy_loop = np.zeros(n_bands)
    for b in range(n_bands):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        band_energy_loop[b] = power[lo:hi].sum()

    bin_gains_loop = np.zeros(n_bins)
    for b in range(n_bands):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        bin_gains_loop[lo:hi] = band_gains[b]

    print(f"Forward  max diff: {np.abs(band_energy - band_energy_loop).max():.2e}")
    print(f"Backward max diff: {np.abs(bin_gains - bin_gains_loop).max():.2e}")


if __name__ == '__main__':
    main()
