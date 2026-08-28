"""每個 pipeline 的 ``layout_version`` 在文件站只能有一個值：header 裡的那個。

這五個計數器是整合者判斷「持久化的 descriptor 還能不能用」的**唯一**訊號——
descriptor 的 ``bytes`` 只是下界，擋得住成長、擋不住等量的語意變更，而
``build_flags_hash`` 是 carve token 字串的 FNV-1a，對結構單純長大是盲的。
所以文件把版號寫錯，整合者會沿用一份 ``init_ex()`` 其實會拒絕的 descriptor。

這件事已經漂移過：一次改版同時留下六個檔、十三處過期版號，其中一處
（``pipeline_mono.html`` 的來源檔對照表）甚至和同一頁另外兩處自相矛盾。
C 測試早就把字面值釘死在 header 上（``test_4aec_nr_res.c`` 的
``FOUR_AEC_NR_RES_LAYOUT_VERSION == 16u`` 那一列），文件端在此補上對稱的閘。

只檢查**具名巨集**的宣稱（``FOUR_AEC_NR_RES_LAYOUT_VERSION = 16``）。文件裡另有
「layout v6 起新增四個 delay 欄位」這類**沿革**敘述，以及 AINR/AIAEC 各模型自有的
layout 版號；那些不是在陳述目前值，刻意不納入。
"""

import pathlib
import re

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]

# 巨集名 -> 定義它的檔案（相對 repo 根）。#define 只有一處，所以不必去重。
_SOURCES = {
    'AUDIO_PIPELINE_LAYOUT_VERSION':
        'pipelines/mono_aec_nr_res/audio_pipeline.c',
    'AUDIO_PIPELINE_ULCNET_LAYOUT_VERSION':
        'pipelines/mono_alignulcnet/audio_pipeline_ulcnet.c',
    'AUDIO_PIPELINE_4CH_LAYOUT_VERSION':
        'pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.h',
    'FOUR_AEC_NR_RES_LAYOUT_VERSION':
        'pipelines/4ch_aec_bf_nr_res/4aec_nr_res.h',
    'AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION':
        'pipelines/4ch_alignulcnet/audio_pipeline_4ch_ulcnet.h',
}

_DOC_ROOT = _REPO / 'docs'
_DOC_SUFFIXES = ('.html', '.md')


def _declared_version(macro: str) -> int:
    """讀 header 裡的 ``#define <macro> Nu``。找不到就是測試自己過時了。"""
    source = _REPO / _SOURCES[macro]
    text = source.read_text(encoding='utf-8')
    match = re.search(r'^#define\s+%s\s+(\d+)u?\s*$' % re.escape(macro),
                      text, re.MULTILINE)
    assert match is not None, (
        '%s 沒有在 %s 找到 #define；巨集被改名或搬家了，'
        '這張表要跟著更新' % (macro, _SOURCES[macro]))
    return int(match.group(1))


def _doc_files():
    if not _DOC_ROOT.is_dir():
        return []
    return sorted(p for p in _DOC_ROOT.rglob('*')
                  if p.suffix in _DOC_SUFFIXES and p.is_file())


# 巨集名之後允許夾一個 </code> 與空白，再吃 `=` 與數字。像
# ``<code>FOUR_AEC_NR_RES_LAYOUT_VERSION</code> = 16`` 與
# ``FOUR_AEC_NR_RES_LAYOUT_VERSION=15`` 兩種寫法都要抓到。
def _claim_pattern(macro: str) -> re.Pattern:
    return re.compile(
        r'%s(?:</code>)?\s*=\s*(\d+)' % re.escape(macro))


@pytest.mark.parametrize('macro', sorted(_SOURCES))
def test_docs_publish_the_current_layout_version(macro):
    expected = _declared_version(macro)
    pattern = _claim_pattern(macro)
    stale = []
    for path in _doc_files():
        for lineno, line in enumerate(
                path.read_text(encoding='utf-8').splitlines(), 1):
            for match in pattern.finditer(line):
                if int(match.group(1)) != expected:
                    stale.append('%s:%d 寫 %s，實際為 %d'
                                 % (path.relative_to(_REPO), lineno,
                                    match.group(0), expected))
    assert not stale, (
        '%s 目前是 %d，但文件仍寫著舊值：\n  %s'
        % (macro, expected, '\n  '.join(stale)))


def test_the_scan_actually_reaches_the_pages_that_publish_versions():
    """沒有這一列，上面五個測試在「掃不到任何檔案」時也會綠。

    正好是這種空掃無聲通過，讓版號可以漂移一整個 release。
    """
    files = _doc_files()
    assert files, '沒有掃到任何文件檔，%s 的路徑假設壞了' % _DOC_ROOT
    hits = sum(
        1 for path in files
        for macro in _SOURCES
        if _claim_pattern(macro).search(path.read_text(encoding='utf-8')))
    assert hits >= 2, (
        '只在 %d 個檔案裡找到具名版號宣稱；至少 integration_example 與 '
        'pipeline_4ch 兩頁應該有，比對式可能失效了' % hits)
