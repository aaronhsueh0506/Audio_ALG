"""
RNNoise ONNX 匯出 — 逐幀串流推論

流程: torch.onnx.export → onnxoptimizer (圖清理) → shape inference

用法:
    python export_onnx.py --config config.ini --model output/rnnoise_best.pth \
                          --output rnnoise.onnx
"""

import argparse
import configparser
import numpy as np
import torch
import torch.nn as nn

from train import RNNoiseModel, compute_hybrid_bands


class RNNoiseStreaming(nn.Module):
    """單幀串流推論 wrapper，輸入 3 frame 特徵，輸出 1 frame gains"""

    def __init__(self, model: RNNoiseModel):
        super().__init__()
        self.conv1 = model.conv1
        self.conv2 = model.conv2
        self.gru1 = model.gru1
        self.gru2 = model.gru2
        self.gru3 = model.gru3
        self.dense_out = model.dense_out
        self.gru_size = model.gru_size

    def forward(self, x, h1, h2, h3):
        """
        x:  (1, 3, n_bands) — 3 frame 特徵
        h1, h2, h3: (1, 1, gru_size) — GRU hidden states
        回傳: gains (1, 1, n_bands), h1_out, h2_out, h3_out
        """
        tmp = x.permute(0, 2, 1)
        tmp = torch.tanh(self.conv1(tmp))
        tmp = torch.tanh(self.conv2(tmp))
        conv_out = tmp.permute(0, 2, 1)  # (1, 1, 128)

        g1, h1_out = self.gru1(conv_out, h1)
        g2, h2_out = self.gru2(g1, h2)
        g3, h3_out = self.gru3(g2, h3)

        cat = torch.cat([conv_out, g1, g2, g3], dim=-1)
        gains = torch.sigmoid(self.dense_out(cat))

        return gains, h1_out, h2_out, h3_out


# ============================================================
# 圖優化
# ============================================================

def count_nodes(model, op_type):
    return sum(1 for n in model.graph.node if n.op_type == op_type)


def optimize_with_onnxoptimizer(inp, outp):
    """onnxoptimizer: 消除冗餘 op、融合 MatMul+Add→Gemm 等"""
    try:
        import onnxoptimizer
        from onnxoptimizer import get_fuse_and_elimination_passes, get_available_passes
    except ImportError:
        print("[skip] onnxoptimizer 未安裝，略過此步")
        import onnx
        onnx.save(onnx.load(inp), outp)
        return outp

    import onnx
    m = onnx.load(inp)
    before = len(m.graph.node)

    base = set(get_fuse_and_elimination_passes())
    avail = set(get_available_passes())

    extra_wanted = {
        "eliminate_nop_pad",
        "eliminate_nop_transpose",
        "eliminate_identity",
        "eliminate_deadend",
        "eliminate_unused_initializer",
        "fuse_consecutive_transposes",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_unsqueezes",
        "fuse_matmul_add_bias_into_gemm",
        "fuse_add_bias_into_conv",
    }
    passes = list((base | (extra_wanted & avail)))

    m2 = onnxoptimizer.optimize(m, passes, fixed_point=True)
    onnx.save(m2, outp)

    after = len(m2.graph.node)
    print(f"[onnxoptimizer] {before} → {after} nodes")
    return outp


# ============================================================
# 匯出
# ============================================================

def export(args):
    import onnx
    from collections import Counter

    # Load config
    cfg = configparser.ConfigParser()
    cfg.read(args.config)

    HYBRID_CUTOFF = cfg.getint('signal', 'hybrid_cutoff_hz', fallback=0)
    N_ERB_HIGH = cfg.getint('signal', 'n_erb_high_bands', fallback=0)
    if HYBRID_CUTOFF > 0 and N_ERB_HIGH > 0:
        N_FFT = cfg.getint('signal', 'n_fft')
        SR = cfg.getint('signal', 'sr')
        _, N_BANDS = compute_hybrid_bands(N_FFT, SR, N_ERB_HIGH, HYBRID_CUTOFF)
    else:
        N_BANDS = cfg.getint('signal', 'n_bands')

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    model = RNNoiseModel(n_bands=N_BANDS, cond_size=64, gru_size=128)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    streaming = RNNoiseStreaming(model)
    streaming.eval()

    gru_size = model.gru_size
    x = torch.randn(1, 3, N_BANDS)
    h = torch.zeros(1, 1, gru_size)

    raw_path = args.output.replace('.onnx', '_raw.onnx')

    # 1) torch.onnx.export
    torch.onnx.export(
        streaming,
        (x, h, h, h),
        raw_path,
        input_names=['input', 'h1_in', 'h2_in', 'h3_in'],
        output_names=['gains', 'h1_out', 'h2_out', 'h3_out'],
        opset_version=17,
        do_constant_folding=True,
    )
    print_stats("torch.onnx.export", raw_path)

    # 2) onnxoptimizer
    optimize_with_onnxoptimizer(raw_path, args.output)
    print_stats("onnxoptimizer", args.output)

    # 3) shape inference
    m = onnx.load(args.output)
    m = onnx.shape_inference.infer_shapes(m)
    onnx.save(m, args.output)
    print_stats("shape inference (final)", args.output)

    # 清理中間檔
    import os
    if os.path.exists(raw_path) and raw_path != args.output:
        os.remove(raw_path)

    if args.verify:
        verify_output(model, N_BANDS, args.output)


def print_stats(stage, path):
    import onnx
    from collections import Counter
    m = onnx.load(path)
    ops = Counter(n.op_type for n in m.graph.node)
    print(f"[{stage}] 節點數: {len(m.graph.node)}, Op: {dict(ops)}")


def verify_output(model, n_bands, onnx_path):
    """用 PyTorch streaming forward 比較 ONNX 輸出"""
    try:
        import onnxruntime as ort

        x = torch.randn(1, 3, n_bands)
        h = torch.zeros(1, 1, model.gru_size)

        with torch.no_grad():
            tmp = x.permute(0, 2, 1)
            tmp = torch.tanh(model.conv1(tmp))
            tmp = torch.tanh(model.conv2(tmp))
            conv_out = tmp.permute(0, 2, 1)
            g1, h1 = model.gru1(conv_out, h)
            g2, h2 = model.gru2(g1, h)
            g3, h3 = model.gru3(g2, h)
            cat = torch.cat([conv_out, g1, g2, g3], dim=-1)
            pt_gains = torch.sigmoid(model.dense_out(cat))

        sess = ort.InferenceSession(onnx_path)
        ort_out = sess.run(None, {
            'input': x.numpy(),
            'h1_in': h.numpy(),
            'h2_in': h.numpy(),
            'h3_in': h.numpy(),
        })

        diff = np.abs(pt_gains.numpy() - ort_out[0]).max()
        print(f"  PyTorch vs ONNX 最大誤差: {diff:.8f}")
        if diff < 1e-5:
            print("  ✓ 驗證通過")
        else:
            print("  ⚠ 誤差偏大，請檢查")
    except ImportError:
        print("需要安裝 onnxruntime 來驗證: pip install onnxruntime")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RNNoise ONNX 匯出')
    parser.add_argument('--config', default='config.ini', help='Config 檔案路徑')
    parser.add_argument('--model', required=True, help='訓練好的 .pth 檔')
    parser.add_argument('--output', default='rnnoise.onnx', help='輸出 .onnx 路徑')
    parser.add_argument('--verify', action='store_true', help='驗證 ONNX 輸出一致性')
    export(parser.parse_args())
