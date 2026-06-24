"""
畫學習曲線 (train/val loss vs epoch)。

讀 train.py 訓練時產生的 output_dir/train_log.csv (欄位 epoch,train_loss,val_loss,lr),
畫出 train/val loss + lr (右軸) 並存成 png。獨立工具，訓練本身不依賴 matplotlib。

用法:
    python plot_curve.py --log output/train_log.csv --output output/curve.png
    python plot_curve.py --log output/train_log.csv          # 預設存到同目錄 curve.png

備註: 即使沒這個 CSV，舊 checkpoint 的 ckpt['loss']/['epoch'] 也存了 per-epoch val loss，
可另行重建 val 曲線。
"""

import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser(description='Plot RNNoise-ERB training curve')
    ap.add_argument('--log', required=True, help='train_log.csv 路徑')
    ap.add_argument('--output', default=None, help='輸出 png (預設: csv 同目錄 curve.png)')
    args = ap.parse_args()

    if not os.path.isfile(args.log):
        raise FileNotFoundError(f'找不到 log: {args.log}')

    epochs, train, val, lr = [], [], [], []
    with open(args.log) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row['epoch']))
            train.append(float(row['train_loss']))
            val.append(float(row['val_loss']))
            lr.append(float(row['lr']))

    if not epochs:
        raise ValueError(f'{args.log} 沒有資料列')

    import matplotlib
    matplotlib.use('Agg')          # 無顯示環境也能存檔
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, train, '-o', ms=3, label='train_loss', color='tab:blue')
    ax1.plot(epochs, val,   '-o', ms=3, label='val_loss',   color='tab:orange')
    best_i = min(range(len(val)), key=lambda i: val[i])
    ax1.axvline(epochs[best_i], color='tab:green', ls='--', lw=1,
                label=f'best val={val[best_i]:.5f} @ep{epochs[best_i]}')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, lr, ':', lw=1, color='tab:gray', label='lr')
    ax2.set_ylabel('lr')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax1.set_title('RNNoise-ERB training curve')

    out = args.output or os.path.join(os.path.dirname(args.log) or '.', 'curve.png')
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f'已存: {out}  (epochs {epochs[0]}..{epochs[-1]}, '
          f'best val {val[best_i]:.5f} @ epoch {epochs[best_i]})')


if __name__ == '__main__':
    main()
