#!/usr/bin/env python3
"""DeepVQE-S streaming inference on one mic/far-end pair.

用法:
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav --device cpu
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav --verify
    # 每次呼叫 N 個 frame (與板端 N > 1 的 graph 同排程)
    python3 inference.py checkpoint.pth mic.wav far.wav out.wav \\
        --frames-per-call 8

目錄批次 (--mic-dir 取代三個位置參數):
    # 沒帶 --ref-dir: reference 為數位靜音, 等同純降噪
    python3 inference.py checkpoint.pth --mic-dir /path/to/mic \\
        --out-dir /path/to/out
    # 帶了 --ref-dir: 逐檔以檔名配對, stem 的 mic 換成 lpb
    python3 inference.py checkpoint.pth --mic-dir /path/to/mic \\
        --ref-dir /path/to/lpb --out-dir /path/to/out

Calibration (also exports the ONNX graph the tensors bind to; the default
graph path is <output>.onnx, override with --onnx):
    python3 inference.py calib --checkpoint checkpoint.pth \\
        --primary-dir /path/to/mic --far-dir /path/to/far \\
        --frames 8192 --format bin --output calib/deepvqe_s
    # N frames per call (板端 1024 點錄音: 16k/256 用 N=8, 48k/1024 用 N=2).
    # The graph and every tensor carry N frames; --frames counts calls,
    # so 1024 calls x 8 = 8192 frames, as many as the example above.
    python3 inference.py calib --checkpoint checkpoint.pth \\
        --primary-dir /path/to/mic --far-dir /path/to/far \\
        --frames 1024 --frames-per-call 8 --format bin \\
        --output calib/deepvqe_s_n8

mic.wav / far.wav must be mono. Both are resampled to the checkpoint's sample
rate before inference when needed; output stays at that model rate.

Output is the common denoised, echo-free, EARLY/DEREVERBERATED near-speech
estimate (DeepVQE's published task already includes dereverberation).

config.ini is not read here: every shape-relevant setting (grid, model
kwargs) is recovered from the checkpoint's own contract (see train.py /
AIAEC/training_common.py's make_checkpoint_contract), so inference cannot
silently drift from what the weights were trained with.
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_AUDIO_ALG_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _AUDIO_ALG_ROOT not in sys.path:
    sys.path.insert(0, _AUDIO_ALG_ROOT)

from AIAEC.DeepVQE_S import DeepVQES
from AIAEC._cli_common import make_inference_cli


build_parser, load_model, main, cli = make_inference_cli(
    'DeepVQE_S', DeepVQES, __doc__
)


if __name__ == '__main__':
    cli()
