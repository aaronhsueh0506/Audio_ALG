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

from train import (
    RNNoiseModel, compute_hybrid_bands, read_feature_config,
    require_checkpoint_feature_config,
)


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
    sr = cfg.getint('signal', 'sr')
    win_len = cfg.getint('signal', 'win_len', fallback=cfg.getint('signal', 'n_fft'))
    hop_len = cfg.getint('signal', 'hop_len', fallback=win_len // 2)
    feature_cfg = read_feature_config(cfg, sr, hop_len)

    HYBRID_CUTOFF = cfg.getint('signal', 'hybrid_cutoff_hz', fallback=0)
    N_ERB_HIGH = cfg.getint('signal', 'n_erb_high_bands', fallback=0)
    if HYBRID_CUTOFF > 0 and N_ERB_HIGH > 0:
        N_FFT = cfg.getint('signal', 'n_fft')
        SR = cfg.getint('signal', 'sr')
        _, N_BANDS = compute_hybrid_bands(N_FFT, SR, N_ERB_HIGH, HYBRID_CUTOFF)
    else:
        N_BANDS = cfg.getint('signal', 'n_bands')

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    require_checkpoint_feature_config(ckpt, feature_cfg, context=args.model)
    # 架構直接從 state_dict 張量形狀推導 (唯一權威來源) — 避免硬寫 64/128 或
    # config/ckpt-config 漂移導致 load_state_dict 失敗. 舊 ckpt 可能無 'config' key.
    #   conv1.weight: [cond_size, n_bands, 3]   conv2.weight: [gru_size, cond_size, 1]
    sd = ckpt['state_dict']
    cond_size = sd['conv1.weight'].shape[0]
    n_bands   = sd['conv1.weight'].shape[1]
    gru_size  = sd['conv2.weight'].shape[0]
    if n_bands != N_BANDS:
        print(f"  ⚠ ckpt n_bands={n_bands} 與 config n_bands={N_BANDS} 不符 → 以 ckpt 為準")
    N_BANDS = n_bands
    model = RNNoiseModel(n_bands=N_BANDS, cond_size=cond_size, gru_size=gru_size)
    model.load_state_dict(sd)
    model.eval()
    print(f"Model: n_bands={N_BANDS}, cond_size={cond_size}, gru_size={gru_size}")

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
    onnx.helper.set_model_props(m, {
        'feature_version': feature_cfg['version'],
        'feature_norm_tau_sec': str(feature_cfg['tau_sec']),
        'feature_mean_init_db': str(feature_cfg['mean_init_db']),
        'feature_std_init_db': str(feature_cfg['var_init_db2'] ** 0.5),
        'feature_std_floor_db': str(feature_cfg['var_floor_db2'] ** 0.5),
        'feature_clip': str(feature_cfg['clip']),
    })
    onnx.save(m, args.output)
    print_stats("shape inference + feature metadata (final)", args.output)

    # 清理中間檔
    import os
    if os.path.exists(raw_path) and raw_path != args.output:
        os.remove(raw_path)

    # 4) 乾淨性驗證: 無 custom op / 無 unknown dim (目標 NPU 佈署要求)
    validate_clean_onnx(args.output)

    if args.verify:
        verify_output(model, N_BANDS, args.output)


def validate_clean_onnx(path):
    """強制驗證匯出的 ONNX 適合 NPU 佈署: (1) 無 custom op, (2) 無 unknown/dynamic dim.
    任一不過就 raise — 讓匯出直接失敗, 而非產出不可佈署的模型."""
    import onnx
    m = onnx.load(path)
    # shape inference 補齊中間 tensor 的 value_info 再檢查
    m = onnx.shape_inference.infer_shapes(m)

    # (1) 無 custom op: 所有 node domain 必須是標準 ONNX domain
    STD_DOMAINS = {"", "ai.onnx", "ai.onnx.ml"}
    custom = sorted({f"{n.op_type}[{n.domain}]" for n in m.graph.node
                     if n.domain not in STD_DOMAINS})
    op_types = sorted({n.op_type for n in m.graph.node})
    print(f"[validate] op_types: {op_types}")
    if custom:
        raise RuntimeError(f"[validate] 發現 custom op (非標準 domain): {custom}")

    # (2) 無 unknown dim: graph input/output/value_info 每個 tensor 每一維都要有具體 dim_value
    def bad_dims(vi):
        t = vi.type.tensor_type
        out = []
        for i, d in enumerate(t.shape.dim):
            # dim_param 非空 = 符號維度; 兩者皆未設 = 未知
            if d.dim_param or not d.HasField("dim_value"):
                out.append((i, d.dim_param or "?"))
        return out

    symbolic = []
    for group in (m.graph.input, m.graph.output, m.graph.value_info):
        for vi in group:
            bad = bad_dims(vi)
            if bad:
                symbolic.append((vi.name, bad))
    if symbolic:
        msg = "; ".join(f"{name}: dims {bad}" for name, bad in symbolic)
        raise RuntimeError(
            f"[validate] 發現 unknown/dynamic dim: {msg}\n"
            f"  修法: 用 onnxsim 固化, 或 onnx.tools.update_model_dims 釘死後重跑 shape_inference")

    print(f"[validate] ✓ 無 custom op / 無 unknown dim ({len(m.graph.node)} nodes)")


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
