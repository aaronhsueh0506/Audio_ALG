"""模型目錄不得含 transform 程式碼（inverse FFT／overlap-add）。

inverse FFT 與 overlap-add 是 audiolib 的 ``AcIstft``（audio_stft.h），由應用端持有；
``AINR/``、``AIAEC/`` 的非測試 .c/.h 只能設定（``*_framing_config``）、借用
（``ac_istft_push*``、``_reset``、``_get_config``）、自有 ``AcStft`` 分析，不得出現
inverse transform、overlap-add 核心或建構／計算 ``AcIstft`` 大小的呼叫。

閘門本身是 ``pipelines/scripts/audit_no_transform_in_models.sh``。這裡除了跑真實樹，
還把每個禁用識別字注入複本的模型檔，確認閘門真的會失敗；只在註解裡出現則不算。
"""

import pathlib
import shutil
import subprocess

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / 'pipelines' / 'scripts' / 'audit_no_transform_in_models.sh'

_FORBIDDEN = ('fft_inverse(h, a, b)', 'fft_inverse_scratch(h, a, b)',
              'sk_wola_accumulate_f32(a, b, c, n)', 'skn_div_floor_f32(a, b, c, n)',
              'ac_istft_init(m, n, &cfg)', 'ac_istft_get_mem_size(&cfg)',
              'ifft(a)', 'irfft(a)', 'overlap_add(a)', 'ola_buf[0] = 0',
              'ola_state = 0')


def _run(root):
    return subprocess.run(['sh', str(_SCRIPT), str(root)],
                          capture_output=True, text=True)


def test_the_real_tree_has_no_transform_code_in_model_dirs():
    done = _run(_REPO)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.startswith('PASS')


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / 'Audio_ALG'
    for part in ('AINR', 'AIAEC'):
        shutil.copytree(
            _REPO / part, root / part,
            ignore=shutil.ignore_patterns('*.pyc', '__pycache__', 'build',
                                          'obj', 'bin', 'output',
                                          'gtcrn_github', 'wav_testset'))
    return root


@pytest.mark.parametrize('call', _FORBIDDEN)
@pytest.mark.parametrize('where', ('AINR/DeepFilterNet3/dfn3_prepost.c',
                                   'AIAEC/DeepVQE_S/deepvqe_process.h'))
def test_an_injected_transform_call_fails_the_gate(tree, call, where):
    path = tree / where
    path.write_text(path.read_text(encoding='utf-8') +
                    '\nstatic void injected(void) { %s; }\n' % call,
                    encoding='utf-8')
    done = _run(tree)
    assert done.returncode == 1, done.stdout + done.stderr
    assert where in done.stdout


def test_a_mention_in_a_comment_or_in_a_test_dir_is_not_a_hit(tree):
    path = tree / 'AINR/DeepFilterNet3/dfn3_prepost.c'
    path.write_text(path.read_text(encoding='utf-8') +
                    '\n/* ac_istft_init(mem, n, &cfg);\n   fft_inverse(h) */\n'
                    '// ifft here\n', encoding='utf-8')
    (tree / 'AINR' / 'tests').mkdir(exist_ok=True)
    (tree / 'AINR' / 'tests' / 'probe.c').write_text(
        'void t(void) { ac_istft_init(0, 0, 0); }\n', encoding='utf-8')
    assert _run(tree).returncode == 0
