"""每個 pipeline 的 ``layout_version`` 在文件站只能有一個值：header 裡的那個。

這五個計數器是整合者判斷「持久化的 descriptor 還能不能用」的**唯一**訊號——
descriptor 的 ``bytes`` 只是下界，擋得住成長、擋不住等量的語意變更，而
``build_flags_hash`` 是 carve token 字串的 FNV-1a，對結構單純長大是盲的。
所以文件把版號寫錯，整合者會沿用一份 ``init_ex()`` 其實會拒絕的 descriptor。

C 測試把字面值釘死在 header 上（``test_4aec_nr_res.c`` 的
``FOUR_AEC_NR_RES_LAYOUT_VERSION == 16u`` 那一列），這裡是文件端對稱的閘。

只檢查**具名巨集**的宣稱（``FOUR_AEC_NR_RES_LAYOUT_VERSION = 16``）。文件裡另有
「layout v6 起新增四個 delay 欄位」這類**沿革**敘述不是在陳述目前值，刻意不納入；
AIAEC 兩個 C 類別的版號（``ULCNET_MODEL_IO_LAYOUT_VERSION``、
``DEEPVQE_PREPOST_LAYOUT_VERSION``）以具名巨集形式納入，AINR 各模型自有的版號則不在此表。
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
    'DFN_RES_STAGE_LAYOUT_VERSION':
        'pipelines/dfn_res_stage.h',
    'DFN_RATE_BRIDGE_LAYOUT_VERSION':
        'pipelines/dfn_rate_bridge.h',
    'MONO_AEC_DFN_RES_LAYOUT_VERSION':
        'pipelines/mono_aec_dfn_res/audio_pipeline_dfn.h',
    'FOUR_AEC_DFN_RES_LAYOUT_VERSION':
        'pipelines/4ch_aec_bf_dfn_res/4aec_dfn_res.h',
    'ULCNET_MODEL_IO_LAYOUT_VERSION':
        'AIAEC/Align_ULCNet/ulcnet_model_io.h',
    'DEEPVQE_PREPOST_LAYOUT_VERSION':
        'AIAEC/DeepVQE_S/deepvqe_prepost.h',
}

_DOC_ROOT = _REPO / 'docs'
_DOC_SUFFIXES = ('.html', '.md')

# A few integration pages publish the current value without naming the C
# macro.  The macro-only scan below cannot distinguish those claims from
# historical prose, so pin only the narrowly-scoped forms that explicitly say
# "current checkout/value" (or identify the measured build in the HTML table).
_UNQUALIFIED_CURRENT_CLAIMS = (
    ('docs/integration_mono_zh_TW.md', 'AUDIO_PIPELINE_LAYOUT_VERSION',
     re.compile(r'本次 checkout（layout_version=(\d+)')),
    ('docs/integration_mono_zh_TW.md', 'AUDIO_PIPELINE_LAYOUT_VERSION',
     re.compile(r'\| 4 \| `layout_version` \| `uint32_t` \| `(\d+)` \|')),
    ('docs/integration_4ch_core_zh_TW.md', 'FOUR_AEC_NR_RES_LAYOUT_VERSION',
     re.compile(r'本次 checkout（`layout_version=(\d+)`')),
    ('docs/integration_4ch_core_zh_TW.md', 'FOUR_AEC_NR_RES_LAYOUT_VERSION',
     re.compile(r'\| 4 \| `layout_version` \| `uint32_t` \| `(\d+)` \|')),
    ('docs/integration_4ch_spatial_zh_TW.md',
     'AUDIO_PIPELINE_4CH_LAYOUT_VERSION',
     re.compile(r'本次 checkout，`layout_version=(\d+)`')),
    ('docs/html/pipeline_4ch.html', 'AUDIO_PIPELINE_4CH_LAYOUT_VERSION',
     re.compile(r'audio_pipeline_4ch_get_mem_requirements\(\)</code>'
                r'\(ne10,layout_version=(\d+)\)')),
    ('docs/html/pipeline_4ch.html', 'AUDIO_PIPELINE_4CH_LAYOUT_VERSION',
     re.compile(r'run_post_stages\(\)</code>;layout v(\d+)\)')),
    ('docs/html/pipeline_mono.html', 'AUDIO_PIPELINE_LAYOUT_VERSION',
     re.compile(r'本頁 API 與實作\(layout v(\d+)\)')),
    ('docs/html/pipeline_ulcnet_mono.html', 'AUDIO_PIPELINE_ULCNET_LAYOUT_VERSION',
     re.compile(r'descriptor v1 / layout v(\d+)\)')),
    ('docs/html/pipeline_ulcnet_4ch.html',
     'AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION',
     re.compile(r'descriptor v1 / <b>layout v(\d+)</b>')),
    ('pipelines/4ch_alignulcnet/README.md',
     'AUDIO_PIPELINE_4CH_ULCNET_LAYOUT_VERSION',
     re.compile(r'LAYOUT_VERSION`\) is \*\*(\d+)\*\*')),
)


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


@pytest.mark.parametrize('relative_path,macro,pattern',
                         _UNQUALIFIED_CURRENT_CLAIMS)
def test_unqualified_current_layout_claims(relative_path, macro, pattern):
    """Current-value prose must move with the corresponding C counter."""
    path = _REPO / relative_path
    matches = pattern.findall(path.read_text(encoding='utf-8'))
    assert matches, '%s 的目前版號宣稱格式已改；請同步更新測試' % relative_path
    expected = _declared_version(macro)
    assert all(int(value) == expected for value in matches), (
        '%s 寫 layout_version=%s，%s 實際為 %d'
        % (relative_path, '/'.join(matches), macro, expected))


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


# Carve versions live with the pre/post classes they version.  A pool carved
# by another walk is refused on the build-flags hash, so a doc that quotes a
# stale carve version sends integrators looking for a mismatch that is not
# there.
_CARVE_SOURCES = {
    'DFN2_PREPOST_CARVE_VERSION': 'AINR/DeepFilterNet2/dfn2_prepost.h',
    'ULCNET_PREPOST_CARVE_VERSION': 'AIAEC/Align_ULCNet/ulcnet_prepost.h',
    'DEEPVQE_PREPOST_CARVE_VERSION': 'AIAEC/DeepVQE_S/deepvqe_prepost.h',
}

# Every spelling that mentions a carve version: the macro name, optionally
# closed by </code> or a backtick, then an optional "=" and a number.  The
# name part is deliberately loose (\w*) so a shortened or misspelled macro
# is found and rejected instead of skipped.
_CARVE_MENTION = re.compile(
    r'\b(\w*CARVE_VERSION)\b(?:</code>|`)*\s*(?:=\s*)?(\d+)?')


def _declared_carve_version(macro: str) -> int:
    text = (_REPO / _CARVE_SOURCES[macro]).read_text(encoding='utf-8')
    match = re.search(r'^#define\s+%s\s+(\d+)u?\b' % re.escape(macro),
                      text, re.MULTILINE)
    assert match is not None, (
        '%s has no #define in %s; the table above is out of date'
        % (macro, _CARVE_SOURCES[macro]))
    return int(match.group(1))


def _carve_mentions(files=None):
    """Yield (path, line number, macro name, claimed number or None)."""
    for path in (_doc_files() if files is None else files):
        text = path.read_text(encoding='utf-8')
        for match in _CARVE_MENTION.finditer(text):
            lineno = text.count('\n', 0, match.start()) + 1
            claimed = match.group(2)
            shown = path.relative_to(_REPO) if _REPO in path.parents else path
            yield (shown, lineno, match.group(1),
                   None if claimed is None else int(claimed))


def _carve_problems(mentions, declared):
    problems = []
    for path, lineno, macro, claimed in mentions:
        if macro not in declared:
            problems.append('%s:%d names %s, which is not a carve version '
                            'macro (write the full macro name)'
                            % (path, lineno, macro))
        elif claimed is not None and claimed != declared[macro]:
            problems.append('%s:%d says %s = %d, the header has %d'
                            % (path, lineno, macro, claimed, declared[macro]))
    return problems


def test_docs_publish_the_current_carve_versions():
    declared = {m: _declared_carve_version(m) for m in _CARVE_SOURCES}
    problems = _carve_problems(_carve_mentions(), declared)
    assert not problems, '\n  '.join(['stale carve version claims:'] + problems)


def test_carve_scan_sees_the_dfn_pipeline_pages_and_rejects_stale_forms(
        tmp_path):
    """The scan has to reach the pipeline pages that quote the DFN2 carve
    version, and the checker has to fail on stale forms."""
    pages = [_REPO / 'docs/html/pipeline_dfn_mono.html',
             _REPO / 'docs/html/pipeline_dfn_4ch.html']
    declared = {m: _declared_carve_version(m) for m in _CARVE_SOURCES}
    for page in pages:
        mentions = list(_carve_mentions([page]))
        assert [(m, c) for _p, _l, m, c in mentions] == [
            ('DFN2_PREPOST_CARVE_VERSION', declared[
                'DFN2_PREPOST_CARVE_VERSION'])], page
    stale = tmp_path / 'stale.html'
    for text in ('<code>CARVE_VERSION 5</code>',
                 '<code>DFN2_PREPOST_CARVE_VERSION 5</code>',
                 '<code>DFN2_PREPOST_CARVE_VERSION = 5</code>',
                 '`ULCNET_PREPOST_CARVE_VERSION`\n  = 9'):
        stale.write_text(text, encoding='utf-8')
        assert _carve_problems(_carve_mentions([stale]), declared), text
    stale.write_text('<code>DFN2_PREPOST_CARVE_VERSION = %d</code>'
                     % declared['DFN2_PREPOST_CARVE_VERSION'],
                     encoding='utf-8')
    assert not _carve_problems(_carve_mentions([stale]), declared)
