"""匯出 metadata 與 config.ini、process.h 的圖契約必須是同一份。"""

import configparser
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# process.h moved to lib/ainr/rnnoise_erb/ in the single-model-release restructuring; ROOT/
# 'process.h' is now only a forwarding stub (an #include line), not the real header text.
RNNOISE_C_DIR = ROOT.parents[1] / 'lib' / 'ainr' / 'rnnoise_erb'

# 三個模型專案各有自己的 top-level ``train.py``/``model.py``/``export_onnx.py``。
# 同一個 pytest session 裡先被 import 的那個會佔住 ``sys.modules``, 所以先清掉
# 快取, 這個檔案才會真的跑到 RNNoise 的程式碼。
for _stale in ('train', 'inference', 'model', 'checkpoint_utils', 'export_onnx'):
    sys.modules.pop(_stale, None)


import torch  # noqa: E402

from export_onnx import STATE_LAYOUT_VERSION, RNNoiseStreaming, build_metadata  # noqa: E402
from train import RNNoiseModel, read_feature_config  # noqa: E402


def c_macro(name):
    header = (RNNOISE_C_DIR / 'process.h').read_text(encoding='utf-8')
    match = re.search(
        r'^#define\s+%s\s+(\d+)u?\s*(?:/\*.*)?$' % re.escape(name),
        header,
        flags=re.MULTILINE,
    )
    assert match is not None, name
    return int(match.group(1))


def _config():
    config = configparser.ConfigParser()
    assert config.read(ROOT / 'config.ini')
    return config


def _metadata():
    config = _config()
    sr = config.getint('signal', 'sr')
    n_fft = config.getint('signal', 'n_fft')
    win_len = config.getint('signal', 'win_len', fallback=n_fft)
    hop_len = config.getint('signal', 'hop_len', fallback=win_len // 2)
    feature_cfg = read_feature_config(config, sr, hop_len, n_fft, win_len)
    return feature_cfg, build_metadata(
        feature_cfg,
        config.getint('signal', 'n_bands'),
        config.getint('model', 'gru_size'),
        config.getboolean('model', 'use_complex_input'),
    )


def test_state_layout_version_is_pinned_to_the_c_header():
    """板端從圖裡讀這個值來判斷自己登錄的 h1/h2/h3 state 配置還對不對。

    只斷言 Python 常數抓不到 metadata key 被刪掉的情況, 所以這裡走的是
    ``export()`` 用的同一個 builder。
    """
    _feature_cfg, metadata = _metadata()
    assert int(metadata['state_layout_version']) == c_macro(
        'RNNOISE_MODEL_IO_LAYOUT_VERSION'
    )
    assert STATE_LAYOUT_VERSION == int(metadata['state_layout_version'])


def test_metadata_state_shapes_match_the_model():
    """The schema's h1/h2/h3 and gains shapes are the ones the config.ini model produces.

    The C library holds no state geometry: the application registers its own buffers, sized
    from this schema.
    """
    feature_cfg, metadata = _metadata()
    config = _config()
    n_bands = config.getint('signal', 'n_bands')
    model = RNNoiseModel(
        n_bands=n_bands, spec_bins=feature_cfg['spec_bins'],
        cond_size=config.getint('model', 'cond_size', fallback=64),
        gru_size=config.getint('model', 'gru_size'),
        spec_conv_channels=config.getint('model', 'spec_conv_channels', fallback=8),
        spec_embed_size=config.getint('model', 'spec_embed_size', fallback=64),
        use_complex_input=config.getboolean('model', 'use_complex_input'),
    ).eval()
    width = model.gru1.hidden_size
    zeros = torch.zeros(1, 1, width)
    with torch.no_grad():
        gains, h1, h2, h3 = RNNoiseStreaming(model)(
            torch.zeros(1, 3, n_bands), torch.zeros(1, 3, 2, feature_cfg['spec_bins']),
            zeros, zeros, zeros)
    assert tuple(gains.shape) == (1, 1, n_bands)
    assert all(tuple(h.shape) == (1, 1, width) for h in (h1, h2, h3))
    assert f'h1_in/h2_in/h3_in[1,1,{width}]' in metadata['input_schema']
    assert f'h1_out/h2_out/h3_out[1,1,{width}]' in metadata['output_schema']
    assert f'erb_input[1,{c_macro("RNNOISE_MODEL_INPUT_FRAMES")},{n_bands}]' in metadata['input_schema']
    assert f'gains[1,1,{n_bands}]' in metadata['output_schema']
    # The default pure-ERB graph prunes the unused complex-spectrum input;
    # calibration folders must describe the actual graph, not its Python
    # wrapper's dead formal argument.
    assert metadata['use_complex_input'] == 'false'
    assert 'spec_input' not in metadata['input_schema']


def test_feature_window_matches_the_c_header():
    """The graph's input window is RNNOISE_MODEL_INPUT_FRAMES feature frames."""
    _feature_cfg, metadata = _metadata()
    assert int(metadata['input_feature_frames']) == c_macro('RNNOISE_MODEL_INPUT_FRAMES')
    assert f'erb_input[1,{c_macro("RNNOISE_MODEL_INPUT_FRAMES")},' in metadata['input_schema']
