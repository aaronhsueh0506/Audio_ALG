"""匯出 metadata 與 process.h 的 model-state 契約必須是同一份。"""

import configparser
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 三個模型專案各有自己的 top-level ``train.py``/``model.py``/``export_onnx.py``。
# 同一個 pytest session 裡先被 import 的那個會佔住 ``sys.modules``, 所以先清掉
# 快取, 這個檔案才會真的跑到 RNNoise 的程式碼。
for _stale in ('train', 'denoise', 'model', 'checkpoint_utils', 'export_onnx'):
    sys.modules.pop(_stale, None)


from export_onnx import STATE_LAYOUT_VERSION, build_metadata  # noqa: E402
from train import read_feature_config  # noqa: E402


def c_macro(name):
    header = (ROOT / 'process.h').read_text(encoding='utf-8')
    match = re.search(
        r'^#define\s+%s\s+(\d+)u?\s*(?:/\*.*)?$' % re.escape(name),
        header,
        flags=re.MULTILINE,
    )
    assert match is not None, name
    return int(match.group(1))


def _metadata():
    config = configparser.ConfigParser()
    assert config.read(ROOT / 'config.ini')
    sr = config.getint('signal', 'sr')
    n_fft = config.getint('signal', 'n_fft')
    win_len = config.getint('signal', 'win_len', fallback=n_fft)
    hop_len = config.getint('signal', 'hop_len', fallback=win_len // 2)
    feature_cfg = read_feature_config(config, sr, hop_len, n_fft, win_len)
    return feature_cfg, build_metadata(
        feature_cfg,
        config.getint('signal', 'n_bands'),
        c_macro('RNNOISE_MODEL_GRU_SIZE'),
    )


def test_state_layout_version_is_pinned_to_the_c_header():
    """板端從圖裡讀這個值來判斷自己的 ``RNNoiseModelState`` 還對不對。

    只斷言 Python 常數抓不到 metadata key 被刪掉的情況, 所以這裡走的是
    ``export()`` 用的同一個 builder。
    """
    _feature_cfg, metadata = _metadata()
    assert int(metadata['state_layout_version']) == c_macro(
        'RNNOISE_MODEL_IO_LAYOUT_VERSION'
    )
    assert STATE_LAYOUT_VERSION == int(metadata['state_layout_version'])


def test_metadata_state_shapes_match_the_c_state_struct():
    """h1/h2/h3 的寬度就是 ``RNNoiseModelState.hidden`` 的第二維。"""
    _feature_cfg, metadata = _metadata()
    gru_size = c_macro('RNNOISE_MODEL_GRU_SIZE')
    assert f'h1_in/h2_in/h3_in[1,1,{gru_size}]' in metadata['input_schema']
    assert f'h1_out/h2_out/h3_out[1,1,{gru_size}]' in metadata['output_schema']
    # 圖只有三個 GRU state, C 端也只配置三層。
    assert c_macro('RNNOISE_MODEL_GRU_COUNT') == 3
