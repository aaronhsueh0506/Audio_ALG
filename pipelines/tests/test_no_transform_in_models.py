"""模型目錄不得含 framing 程式碼（分析與合成都不行）。

分析（``AcStft``）、inverse FFT 與 overlap-add（``AcIstft``）都是 audiolib 的
（audio_stft.h），由應用端建立與執行；``AINR/``、``AIAEC/`` 的非測試 .c/.h 不得出現
forward／inverse transform、overlap-add 核心，也不得建構、計算大小或執行
``AcStft``／``AcIstft``。唯一允許的 audiolib framing 用法是 n_fft 檢查：
``ac_stft_config_default`` 填一個 ``AcStftConfig``，再以 ``ac_stft_get_mem_size`` 判斷。

閘門本身是 ``pipelines/scripts/audit_no_transform_in_models.sh``。這裡除了跑真實樹，
還把每個禁用識別字注入複本的模型檔，確認閘門真的會失敗；只在註解、測試目錄裡出現
或是允許的 n_fft 檢查則不算。
"""

import pathlib
import shutil
import subprocess

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SCRIPT = _REPO / 'pipelines' / 'scripts' / 'audit_no_transform_in_models.sh'

_SYNTHESIS = ('fft_inverse(h, a, b)', 'fft_inverse_scratch(h, a, b)',
              'sk_wola_accumulate_f32(a, b, c, n)', 'skn_div_floor_f32(a, b, c, n)',
              'ac_istft_init(m, n, &cfg)', 'ac_istft_get_mem_size(&cfg)',
              'ifft(a)', 'irfft(a)', 'overlap_add(a)', 'ola_buf[0] = 0',
              'ola_state = 0')
_ANALYSIS = ('fft_forward(h, a, b)', 'fft_forward_scratch(h, a, b)',
             'ac_stft_init(m, n, &cfg)', 'ac_stft_push(s, a, b)',
             'ac_stft_push_planar(s, a, b, c)', 'ac_stft_frame_planar(s, a, b, c)',
             'ac_stft_flush_planar(s, a, b)', 'ac_stft_reset(s)', 'ac_stft_window(s)',
             'ac_stft_get_config(s, &cfg)', 'cfg.spec_scale = 1.0f',
             'cfg.ispec_scale = 1.0f')
_FRAMING = ('ac_istft_push(s, a, b)', 'ac_istft_push_planar(s, a, b, c)',
            'ac_istft_flush(s, a)', 'ac_istft_reset(s)', 'ac_istft_get_config(s, &cfg)',
            'AcStft* s = 0', 'AcIstft* s = 0', 'AcOlaMode m = AC_OLA_ENV_NORM',
            'int m = AC_OLA_ACCUMULATE')
_FORBIDDEN = _SYNTHESIS + _ANALYSIS + _FRAMING

# The n_fft check: the only audiolib framing use a model source may have.
_N_FFT_CHECK = ('AcStftConfig sc; ac_stft_config_default(&sc, 512); '
                'return ac_stft_get_mem_size(&sc) != 0u')


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
                    '// ifft here\n/* ac_stft_push_planar(s, a, b, c) */\n',
                    encoding='utf-8')
    (tree / 'AINR' / 'tests').mkdir(exist_ok=True)
    (tree / 'AINR' / 'tests' / 'probe.c').write_text(
        'void t(void) { ac_istft_init(0, 0, 0); ac_stft_push(0, 0, 0); }\n',
        encoding='utf-8')
    assert _run(tree).returncode == 0


def test_the_n_fft_check_pair_is_not_a_hit(tree):
    path = tree / 'AIAEC/DeepVQE_S/deepvqe_process.h'
    path.write_text(path.read_text(encoding='utf-8') +
                    '\nstatic int injected(void) { %s; }\n' % _N_FFT_CHECK,
                    encoding='utf-8')
    assert _run(tree).returncode == 0


def test_a_comment_marker_inside_a_string_does_not_hide_a_call(tree):
    path = tree / 'AINR/DeepFilterNet3/dfn3_prepost.c'
    path.write_text(path.read_text(encoding='utf-8') +
                    '\nstatic const char* url = "http://x"; '
                    'static void injected(void) { ac_stft_push(s, a, b); }\n',
                    encoding='utf-8')
    done = _run(tree)
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'ac_stft_push' in done.stdout
