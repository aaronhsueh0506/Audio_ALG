"""
RNNoise ONNX 匯出 — stateless 逐幀串流推論

流程: torch.onnx.export → onnxoptimizer (圖清理) → shape inference

用法:
    python export_onnx.py --config config.ini --model output/rnnoise_best.pth \
                          --output output/rnnoise.onnx --verify

BIN calibration（每個 frame、每個 tensor 一個檔）:
    python inference.py --config config.ini --model output/rnnoise_best.pth \
                      --input noisy.wav --output enhanced.wav \
                      --dump-calib calib/rnnoise_erb --format bin \
                      --max-frames 8192

NPZ calibration:
    python inference.py --config config.ini --model output/rnnoise_best.pth \
                      --input noisy.wav --output enhanced.wav \
                      --dump-calib calib/rnnoise_erb.npz --format npz \
                      --max-frames 8192
"""

import argparse
import configparser
import os
import numpy as np
import torch
import torch.nn as nn

try:
    from .train import (
        RNNoiseModel, read_feature_config,
        require_checkpoint_feature_config, model_capacity_from_checkpoint,
    )
except ImportError:  # direct ``python export_onnx.py`` execution
    import os as _os
    import sys as _sys
    _SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
    if _SCRIPT_DIR not in _sys.path:
        _sys.path.insert(0, _SCRIPT_DIR)
    from train import (
        RNNoiseModel, read_feature_config,
        require_checkpoint_feature_config, model_capacity_from_checkpoint,
    )


# 與 process.h 的 RNNOISE_MODEL_IO_LAYOUT_VERSION 保持數值相同 — 該巨集宣告
# 這張圖消費/回傳的 caller-owned GRU state struct。
# tests/test_export_metadata.py 把兩邊釘在一起。
STATE_LAYOUT_VERSION = 1


class RNNoiseStreaming(nn.Module):
    """單幀串流推論 wrapper，輸入 3 frame 雙路特徵。"""

    def __init__(self, model: RNNoiseModel):
        super().__init__()
        self.model = model

    def forward(self, erb_input, spec_input, h1, h2, h3):
        """
        erb_input:  (1, 3, n_bands)
        spec_input: (1, 3, 2, spec_bins)
        h1, h2, h3: (1, 1, gru_size) — GRU hidden states
        回傳: gains, h1_out, h2_out, h3_out
        """
        gains, states = self.model(
            erb_input, spec_input, [h1, h2, h3])
        return gains, states[0], states[1], states[2]


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

def build_metadata(feature_cfg, n_bands, gru_size, use_complex_input=False):
    """匯出圖的 metadata。

    從 export() 抽出來, 讓 contract test 不必真的做一次 ONNX 匯出就能把
    state_layout_version 與 process.h 的巨集對起來。
    """
    return {
        'boundary': 'stateless_streaming_explicit_state',
        'state_layout_version': str(STATE_LAYOUT_VERSION),
        'input_feature_frames': '3',
        'output_frames_per_invocation': '1',
        'accelerator_persistent_state': 'false',
        'recurrent_state': 'h1_h2_h3_explicit_input_output',
        'host_updates_feature_window': 'true',
        'feature_version': feature_cfg['version'],
        'sr': str(feature_cfg['sr']),
        'n_fft': str(feature_cfg['n_fft']),
        'win_len': str(feature_cfg['win_len']),
        'hop_len': str(feature_cfg['hop_len']),
        'lookahead_frames': str(feature_cfg['lookahead_frames']),
        'feature_erb_norm_tau_sec': str(feature_cfg['erb_tau_sec']),
        'feature_erb_norm_alpha': str(feature_cfg['erb_alpha']),
        'feature_erb_norm_init_lo_db': str(feature_cfg['erb_norm_init_lo_db']),
        'feature_erb_norm_init_hi_db': str(feature_cfg['erb_norm_init_hi_db']),
        'feature_erb_norm_scale_db': str(feature_cfg['erb_norm_scale_db']),
        'feature_spec_max_hz': str(feature_cfg['spec_max_hz']),
        'feature_spec_bins': str(feature_cfg['spec_bins']),
        'feature_spec_norm_tau_sec': str(feature_cfg['spec_tau_sec']),
        'feature_spec_norm_alpha': str(feature_cfg['spec_alpha']),
        'feature_spec_norm_init_lo': str(feature_cfg['spec_norm_init_lo']),
        'feature_spec_norm_init_hi': str(feature_cfg['spec_norm_init_hi']),
        'feature_spec_norm_eps': str(feature_cfg['spec_norm_eps']),
        'input_schema': (
            f'erb_input[1,3,{n_bands}];' +
            (f'spec_input[1,3,2,{feature_cfg["spec_bins"]}];'
             if use_complex_input else '') +
            f'h1_in/h2_in/h3_in[1,1,{gru_size}]'
        ),
        'use_complex_input': str(bool(use_complex_input)).lower(),
        'output_schema': (f'gains[1,1,{n_bands}];'
                          f'h1_out/h2_out/h3_out[1,1,{gru_size}]'),
        'c_prepost': 'process.c/process.h',
    }


def export_graph(config_path, checkpoint_path, output_path, verify=False):
    """Programmatic entry over ``export(args)``.

    Lives here, next to every ``args.*`` field ``export`` reads, so a new
    CLI knob gets added to this synthesis in the same edit instead of
    silently breaking a caller that hand-built the namespace elsewhere.
    """
    from types import SimpleNamespace
    export(SimpleNamespace(config=config_path, model=checkpoint_path,
                           output=output_path, verify=verify))


def export(args):
    import onnx
    from collections import Counter

    # Load config
    cfg = configparser.ConfigParser()
    cfg.read(args.config)
    sr = cfg.getint('signal', 'sr')
    n_fft = cfg.getint('signal', 'n_fft')
    win_len = cfg.getint('signal', 'win_len', fallback=cfg.getint('signal', 'n_fft'))
    hop_len = cfg.getint('signal', 'hop_len', fallback=win_len // 2)
    feature_cfg = read_feature_config(cfg, sr, hop_len, n_fft, win_len)

    HYBRID_CUTOFF = cfg.getint('signal', 'hybrid_cutoff_hz', fallback=0)
    N_ERB_HIGH = cfg.getint('signal', 'n_erb_high_bands', fallback=0)
    if HYBRID_CUTOFF > 0 and N_ERB_HIGH > 0:
        raise NotImplementedError(
            "hybrid bands are not supported with the faithful DFN/Keras ERB "
            "filterbank; set hybrid_cutoff_hz=0 to use pure ERB")
    N_BANDS = cfg.getint('signal', 'n_bands')

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    require_checkpoint_feature_config(ckpt, feature_cfg, context=args.model)
    # 架構容量從 checkpoint 取；feature contract 則由
    # require_checkpoint_feature_config 嚴格限定。
    sd = ckpt['state_dict']
    capacity = model_capacity_from_checkpoint(ckpt)
    if capacity['n_bands'] != N_BANDS:
        print(f"  ⚠ ckpt n_bands={capacity['n_bands']} 與 config n_bands={N_BANDS}"
              f" 不符 → 以 ckpt 為準")
    N_BANDS = capacity['n_bands']
    model = RNNoiseModel(spec_bins=feature_cfg['spec_bins'], **capacity)
    model.load_state_dict(sd)
    model.eval()
    gru_size = model.gru_size
    print(f"Model: n_bands={N_BANDS}, spec_bins={feature_cfg['spec_bins']}, "
          f"cond_size={capacity['cond_size']}, gru_size={gru_size}, "
          f"use_complex_input={capacity['use_complex_input']}")

    streaming = RNNoiseStreaming(model)
    streaming.eval()
    erb_input = torch.randn(1, 3, N_BANDS)
    spec_input = torch.randn(1, 3, 2, feature_cfg['spec_bins'])
    h = torch.zeros(1, 1, gru_size)

    output_root, output_ext = os.path.splitext(args.output)
    raw_path = output_root + '_raw' + (output_ext or '.onnx')

    # 1) torch.onnx.export
    torch.onnx.export(
        streaming,
        (erb_input, spec_input, h, h, h),
        raw_path,
        input_names=['erb_input', 'spec_input', 'h1_in', 'h2_in', 'h3_in'],
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
    onnx.helper.set_model_props(m, build_metadata(
        feature_cfg, N_BANDS, gru_size, model.use_complex_input
    ))
    onnx.save(m, args.output)
    print_stats("shape inference + feature metadata (final)", args.output)

    # 清理中間檔
    if os.path.exists(raw_path) and raw_path != args.output:
        os.remove(raw_path)

    # 4) 乾淨性驗證: 無 custom op / 無 unknown dim (目標 NPU 佈署要求)
    validate_clean_onnx(args.output)

    if args.verify:
        verify_output(streaming, N_BANDS, feature_cfg['spec_bins'],
                      model.gru_size, args.output)


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


def verify_output(streaming, n_bands, spec_bins, gru_size, onnx_path):
    """用 PyTorch streaming forward 比較 ONNX 輸出"""
    try:
        import onnxruntime as ort

        erb_input = torch.randn(1, 3, n_bands)
        spec_input = torch.randn(1, 3, 2, spec_bins)
        h = torch.zeros(1, 1, gru_size)

        with torch.no_grad():
            pt_out = streaming(erb_input, spec_input, h, h, h)

        sess = ort.InferenceSession(onnx_path)
        # Feed whatever the graph actually declares, not a fixed name list.
        # In a pure-ERB export `spec_input` is dead in the graph, and exporters
        # are free to prune a dead input -- passing a name the session does not
        # have is a hard error, so the verification would fail on a perfectly
        # good model.  (Not reproducible here: `onnx` is not installed, so the
        # export path itself cannot run in this environment.)
        available = {i.name for i in sess.get_inputs()}
        feed = {'erb_input': erb_input.numpy(), 'spec_input': spec_input.numpy(),
                'h1_in': h.numpy(), 'h2_in': h.numpy(), 'h3_in': h.numpy()}
        missing = available - set(feed)
        if missing:
            raise RuntimeError(f"ONNX graph expects unknown inputs: {sorted(missing)}")
        dropped = set(feed) - available
        if dropped:
            print(f"  note: exporter pruned unused graph input(s): {sorted(dropped)}")
        ort_out = sess.run(None, {k: v for k, v in feed.items() if k in available})

        diff = max(np.abs(pt.detach().numpy() - ort).max()
                   for pt, ort in zip(pt_out, ort_out))
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
