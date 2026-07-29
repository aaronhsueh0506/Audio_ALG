"""
匯出 ERB 轉換矩陣 (三角版, 對齊 train.py / denoise.py)

產出給目標平台 C runtime 的 ERB 前/後處理矩陣:
    feature:  power @ W_fwd            (mode=0, edge x2)
    mask→bin: W_inv @ band_gains       (mode=1, partition of unity)
兩者都在 NN 圖外的 C 端跑。矩陣用 train.compute_erb_matrix 建 (三角 DFN-Keras 版),
與模型訓練/推論所見完全一致。

用法:
    python export_erb_matrix.py --config config.ini --format bin
    python export_erb_matrix.py --config config.ini --model output/rnnoise_best.pth --format bin
    python export_erb_matrix.py --config config.ini --format all
"""

import argparse
import configparser
import os
import struct

import numpy as np

# 唯一權威來源: 用 train.py 的三角 ERB filterbank (與 denoise.load_model 相同)。
from train import erb_bandborder, compute_erb_matrix


def resolve_nfftborder(cfg, model_path):
    """對齊 denoise.load_model: 優先用 ckpt['nfftborder'], 否則從 config 重算。"""
    SR = cfg.getint('signal', 'sr')
    N_FFT = cfg.getint('signal', 'n_fft')
    HYBRID_CUTOFF = cfg.getint('signal', 'hybrid_cutoff_hz', fallback=0)
    N_ERB_HIGH = cfg.getint('signal', 'n_erb_high_bands', fallback=0)
    if HYBRID_CUTOFF > 0 and N_ERB_HIGH > 0:
        raise NotImplementedError(
            "hybrid bands 不支援 faithful DFN/Keras 三角 ERB filterbank; "
            "設 hybrid_cutoff_hz=0 使用純 ERB (對齊 denoise.load_model)")
    N_BANDS = cfg.getint('signal', 'n_bands')

    if model_path:
        import torch
        ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
        if 'nfftborder' in ckpt:
            nfftborder = np.asarray(ckpt['nfftborder'], dtype=int)
            print(f"nfftborder: 取自 ckpt (len={len(nfftborder)})")
            return nfftborder, SR, N_FFT
        # 舊 ckpt 可能只有 legacy 'bin_edges' (矩形版, len=n_bands+1) — 語意不同, 不可當 nfftborder。
        if 'bin_edges' in ckpt:
            print("  ⚠ ckpt 只有 legacy 'bin_edges' (舊矩形版, 與三角 nfftborder 不相容) → 忽略, 改用 config 重算")
        else:
            print("  ⚠ ckpt 無 nfftborder → 改用 config 重算")

    MIN_BINS = cfg.getint('signal', 'min_bins_per_band', fallback=2)
    nfftborder = erb_bandborder(N_BANDS, SR, N_FFT, MIN_BINS)
    print(f"nfftborder: erb_bandborder({N_BANDS}, {SR}, {N_FFT}) (len={len(nfftborder)})")
    return nfftborder, SR, N_FFT


def export_bin(W_fwd, W_inv, nfftborder, output_path):
    """raw binary (little-endian), 給 C fread:
        int32   n_bins
        int32   n_bands
        float32 W_fwd[n_bins*n_bands]   row-major  (feature: power @ W_fwd)
        float32 W_inv[n_bins*n_bands]   row-major  (mask→bin: W_inv @ band_gains)
        int32   nfftborder[n_bands]
    """
    n_bins, n_bands = W_fwd.shape
    with open(output_path, 'wb') as f:
        f.write(struct.pack('<ii', n_bins, n_bands))
        f.write(np.ascontiguousarray(W_fwd, dtype='<f4').tobytes())
        f.write(np.ascontiguousarray(W_inv, dtype='<f4').tobytes())
        f.write(np.ascontiguousarray(np.asarray(nfftborder), dtype='<i4').tobytes())
    print(f"已儲存: {output_path}  "
          f"(header 8B + W_fwd {n_bins*n_bands*4}B + W_inv {n_bins*n_bands*4}B + "
          f"nfftborder {n_bands*4}B)")


def export_npy(W_fwd, W_inv, output_dir):
    np.save(os.path.join(output_dir, 'erb_fwd.npy'), W_fwd)
    np.save(os.path.join(output_dir, 'erb_inv.npy'), W_inv)
    print(f"已儲存: {output_dir}/erb_fwd.npy, {output_dir}/erb_inv.npy")


def _c_matrix(f, name, M):
    n_rows, n_cols = M.shape
    f.write(f"static const float {name}[{n_rows}][{n_cols}] = {{\n")
    for i in range(n_rows):
        row = ", ".join(f"{v:.8f}f" for v in M[i])
        f.write(f"    {{{row}}}")
        f.write("," if i < n_rows - 1 else "")
        f.write(f"  /* {i} */\n")
    f.write("};\n\n")


def export_c_header(W_fwd, W_inv, nfftborder, output_path, sr, n_fft):
    n_bins, n_bands = W_fwd.shape
    with open(output_path, 'w') as f:
        f.write("/* ERB 轉換矩陣 (三角版) - 自動產生, 對齊 train.compute_erb_matrix */\n")
        f.write(f"/* N_BINS={n_bins}, N_BANDS={n_bands}, N_FFT={n_fft}, SR={sr} */\n\n")
        f.write("#ifndef ERB_MATRIX_H\n#define ERB_MATRIX_H\n\n")
        f.write(f"#define ERB_N_BINS {n_bins}\n")
        f.write(f"#define ERB_N_BANDS {n_bands}\n\n")
        f.write(f"static const int ERB_NFFTBORDER[{n_bands}] = {{\n    ")
        f.write(", ".join(str(int(e)) for e in nfftborder))
        f.write("\n};\n\n")
        f.write("/* feature: power @ ERB_FWD  (mode=0, edge x2) */\n")
        _c_matrix(f, "ERB_FWD", W_fwd)
        f.write("/* mask->bin: ERB_INV @ band_gains  (mode=1, partition of unity) */\n")
        _c_matrix(f, "ERB_INV", W_inv)
        f.write("#endif /* ERB_MATRIX_H */\n")
    print(f"已儲存: {output_path}")


def sanity_check(W_fwd, W_inv, nfftborder):
    """三角 ERB 特性檢查 (取代舊矩形版的 for-loop 對照)。"""
    lo, hi = int(nfftborder[0]), int(nfftborder[-1])
    # mode=1: 每個被覆蓋的 bin 列和 ≈ 1 (partition of unity, mask=1 → bin_gain=1)
    row_sum = W_inv[lo:hi].sum(axis=1)
    pou_err = float(np.abs(row_sum - 1.0).max()) if hi > lo else float('nan')
    # mode=0 = mode=1 但兩側 edge column x2
    W_inv_edge2 = W_inv.copy()
    W_inv_edge2[:, 0] *= 2.0
    W_inv_edge2[:, -1] *= 2.0
    edge_ok = np.allclose(W_fwd, W_inv_edge2)
    print(f"=== 三角 ERB sanity ===")
    print(f"  mode=1 partition-of-unity max|rowsum-1| (bins {lo}..{hi-1}): {pou_err:.2e}")
    print(f"  mode=0 == mode=1 with edge x2: {edge_ok}")
    return pou_err, edge_ok


def main():
    parser = argparse.ArgumentParser(description='匯出 ERB 轉換矩陣 (三角版)')
    parser.add_argument('--config', default='config.ini', help='Config 檔案路徑')
    parser.add_argument('--model', default=None,
                        help='(選填) ckpt; 有 nfftborder 就用它, 否則從 config 重算')
    parser.add_argument('--format', choices=['bin', 'npy', 'c', 'all'], default='bin')
    parser.add_argument('--output-dir', default=None,
                        help='輸出目錄 (預設從 config paths.output_dir)')
    args = parser.parse_args()

    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    nfftborder, SR, N_FFT = resolve_nfftborder(cfg, args.model)
    W_fwd = compute_erb_matrix(nfftborder, N_FFT, mode=0)   # feature (edge x2)
    W_inv = compute_erb_matrix(nfftborder, N_FFT, mode=1)   # mask→bin (partition of unity)
    n_bins, n_bands = W_fwd.shape

    output_dir = args.output_dir or cfg.get('paths', 'output_dir', fallback='output')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Config: SR={SR}, N_FFT={N_FFT}, N_BANDS={n_bands}")
    print(f"ERB Matrix: ({n_bins}, {n_bands})   nfftborder={np.asarray(nfftborder).tolist()}")
    print()

    if args.format in ('bin', 'all'):
        export_bin(W_fwd, W_inv, nfftborder, os.path.join(output_dir, 'erb_matrix.bin'))
    if args.format in ('npy', 'all'):
        export_npy(W_fwd, W_inv, output_dir)
    if args.format in ('c', 'all'):
        export_c_header(W_fwd, W_inv, nfftborder,
                        os.path.join(output_dir, 'erb_matrix.h'), SR, N_FFT)

    print()
    sanity_check(W_fwd, W_inv, nfftborder)


if __name__ == '__main__':
    main()
