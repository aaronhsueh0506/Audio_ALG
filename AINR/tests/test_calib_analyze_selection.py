"""``PTQ/calib_analyze.py`` selection, grids, sources and coverage."""

import json
import os
import sys

import numpy as np
import pytest

from calibration_io import write_calibration_artifact

_PTQ = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'PTQ')
if _PTQ not in sys.path:
    sys.path.insert(0, _PTQ)

import calib_analyze  # noqa: E402


def _write(tmp_path, arrays, report=None, name='a.npz'):
    out = tmp_path / name
    n = next(iter(arrays.values())).shape[0]
    write_calibration_artifact(out, arrays, dict(report or {}, frames=n), 'npz')
    return str(out)


def _heavy(rng, n, width):
    """Heavy-tailed float32 values (student-t), so frames differ in extremeness."""
    return rng.standard_t(2, size=(n, 1, width)).astype(np.float32)


def _with_sources(rng, lengths, width=16, state_width=8):
    """Feature + state arrays whose state resets mark each source start."""
    n = sum(lengths)
    state = (0.1 * rng.standard_normal((n, state_width))).astype(np.float32)
    starts = np.cumsum([0] + lengths[:-1])
    state[starts] = 0.0
    return {'feat': _heavy(rng, n, width), 'h': state}, list(starts)


def _pool(arrays, report=None):
    return calib_analyze.scan_artifact(
        calib_analyze.Artifact('x', 'npz', arrays, report or {}))


def _pow2_up(top):
    """Smallest power of two >= top, by doubling/halving (0 for 0)."""
    if top <= 0:
        return 0.0
    bound = 1.0
    while bound < top:
        bound *= 2.0
    while bound / 2.0 >= top:
        bound /= 2.0
    return bound


def _grid(lo, hi, scheme):
    if scheme == 'asym':
        return min(lo, 0.0), max(hi, 0.0)
    top = max(abs(lo), abs(hi))
    if scheme == 'pow2':
        top = _pow2_up(top)
    return -top, top


@pytest.mark.parametrize('scheme', ['asym', 'sym', 'pow2'])
def test_subset_table_frame_shares_match_brute_force(scheme):
    rng = np.random.default_rng(2)
    feat = _heavy(rng, 150, 10)
    pool = _pool({'feat': feat})
    schemes = {'feat': scheme}
    models = {'feat': calib_analyze.GridError(pool.stats['feat']['_sample'],
                                              scheme)}
    flat = feat.reshape(150, -1).astype(np.float64)
    for draw in range(5):
        keep = rng.choice(150, 30, replace=False)
        row = calib_analyze.subset_table(pool.stats, schemes, models, keep, '',
                                         (8, 16))['rows'][0]
        lo, hi = _grid(flat[keep].min(), flat[keep].max(), scheme)
        outside = ((flat < lo) | (flat > hi)).any(axis=1)
        assert row['frames_clip_pct'] == pytest.approx(100.0 * outside.mean(),
                                                       abs=1e-12)
        assert row['sqnr16_typ'] > row['sqnr8_typ']
        assert 'clip_pct' not in row


def test_quantize_puts_zero_on_the_grid_and_saturates_outside():
    values = np.array([-3.0, -1.0, 0.0, 0.3, 2.0, 9.0])
    q = calib_analyze.quantize(values, -1.0, 2.0, 8, 'asym')
    assert q[2] == 0.0
    assert q[0] == pytest.approx(-1.0, abs=2e-2)
    assert q[-1] == pytest.approx(2.0, abs=2e-2)
    s = calib_analyze.quantize(values, -2.0, 2.0, 8, 'sym')
    assert s[2] == 0.0 and s[-1] == pytest.approx(2.0)
    error8 = calib_analyze.GridError(values[1:5]).evaluate(-1.0, 2.0, 8)
    error16 = calib_analyze.GridError(values[1:5]).evaluate(-1.0, 2.0, 16)
    assert error16['sqnr_all'] > error8['sqnr_all'] + 40


def test_power_of_two_grid_rounds_the_bound_up_and_uses_every_level():
    assert calib_analyze.grid_range(-3.0, 2.5, 'pow2') == (-4.0, 4.0)
    assert calib_analyze.grid_range(-0.3, 0.2, 'pow2') == (-0.5, 0.5)
    assert calib_analyze.grid_range(-1.0, 1.0, 'pow2') == (-1.0, 1.0)
    assert calib_analyze.grid_range(0.0, 0.0, 'pow2') == (0.0, 0.0)
    lo, hi = calib_analyze.grid_range(-3.0, 2.5, 'pow2')
    step = 4.0 / 128            # 8 bits: -128 .. 127 steps of 2^2 / 2^7
    q = calib_analyze.quantize(np.array([-9.0, -4.0, 0.0, step, 3.99, 4.0]),
                               lo, hi, 8, 'pow2')
    assert list(q) == [-4.0, -4.0, 0.0, step, 4.0 - step, 4.0 - step]
    assert not calib_analyze.quantize(np.ones(3), 0.0, 0.0, 8, 'pow2').any()
    for bad in (lambda: calib_analyze.grid_range(0.0, 1.0, 'bogus'),
                lambda: calib_analyze.quantize(np.ones(2), 0.0, 1.0, 8, 'x')):
        with pytest.raises(ValueError, match='unknown scheme'):
            bad()


def test_power_of_two_report_counts_frames_per_bound_and_flags_the_edge():
    rng = np.random.default_rng(10)
    a = rng.uniform(-1.0, 1.0, (100, 1, 6)).astype(np.float32)
    a[[3, 50, 77], 0, 2] = [2.5, -2.2, 3.0]     # bound 4, three frames past 2
    b = rng.uniform(-1.0, 1.0, (100, 1, 6)).astype(np.float32)
    b[10, 0, 0] = 3.99                          # within one 8-bit step of 4
    result = calib_analyze.analyze(_pool({'a': a, 'b': b}), 'pow2', count=40)
    texts = [c['text'] for c in result['conclusions']]
    bound = next(t for t in texts if t.startswith('2 的冪次 grid'))
    assert 'a ±4（降到 ±2 要丟 3 個 frame' in bound
    assert 'b ±4（降到 ±2 要丟 1 個 frame' in bound
    edge = [c['text'] for c in result['conclusions']
            if c['level'] == 'warn' and '6 dB' in c['text']]
    assert len(edge) == 1 and edge[0].startswith('b ')


def test_asymmetric_grid_holds_zero_for_an_all_positive_tensor():
    # A magnitude-like input never crosses 0; the grid must still hold it.
    assert calib_analyze.grid_range(5.0, 10.0, 'asym') == (0.0, 10.0)
    values = np.random.default_rng(9).uniform(5.0, 10.0, 4000)
    lo, hi = calib_analyze.grid_range(5.0, 10.0, 'asym')
    q = calib_analyze.quantize(values, lo, hi, 8, 'asym')
    assert np.abs(q - values).max() <= (hi - lo) / 255 / 2 + 1e-9
    assert calib_analyze.GridError(values).evaluate(lo, hi, 8)['sqnr_all'] > 45


def test_count_is_a_cap_and_keeps_every_extreme_holder():
    rng = np.random.default_rng(3)
    arrays = {'a': _heavy(rng, 300, 8), 'b': _heavy(rng, 300, 4)}
    pool = _pool(arrays)
    sel = calib_analyze.select_patterns(pool, 40)
    keep = sel['keep']
    assert keep.size == 40
    for name, x in arrays.items():
        flat = x.reshape(300, -1)
        assert flat[keep].min() == flat.min()
        assert flat[keep].max() == flat.max()
    whole = calib_analyze.select_patterns(pool, 1000)
    assert whole['keep'].size == 300 and whole['notes']


def test_state_inputs_are_scored_on_a_symmetric_grid():
    rng = np.random.default_rng(4)
    arrays, _ = _with_sources(rng, [50, 60])
    art = calib_analyze.Artifact('x', 'npz', arrays, {})
    assert art.state == ['h']
    assert art.schemes('asym') == {'feat': 'asym', 'h': 'sym'}
    # sym and pow2 already put 0 at zero point 0; state keeps the input grid.
    assert art.schemes('pow2') == {'feat': 'pow2', 'h': 'pow2'}
    # A selection need not start on a fresh state: the report names it.
    arrays['h'][0] = 1.0
    named = calib_analyze.Artifact('x', 'npz', arrays, {'state_inputs': ['h']})
    assert named.state == ['h']
    assert calib_analyze.Artifact('x', 'npz', arrays, {}).state == []


def test_sources_unavailable_without_recorded_source_files():
    rng = np.random.default_rng(5)
    arrays, _ = _with_sources(rng, [40, 40, 40])
    analysis = calib_analyze.analyze(_pool(arrays), count=50)
    assert analysis['sources']['available'] is False
    assert analysis['folds'] == {'available': False}
    assert any('來源資訊不可用' in c['text'] for c in analysis['conclusions'])


def test_source_report_finds_the_source_that_holds_the_range():
    rng = np.random.default_rng(6)
    arrays, starts = _with_sources(rng, [40, 50, 60])
    arrays['feat'] = np.clip(arrays['feat'], -3, 3)
    arrays['feat'][starts[1] + 7, 0, 2] = 40.0
    pool = _pool(arrays, {'source_files': ['s0.wav', 's1.wav', 's2.wav']})
    score = calib_analyze.frame_score(pool.stats)
    keep = np.arange(0, 150, 3)
    src = calib_analyze.source_report(pool, keep, score, pool.schemes('asym'))
    assert src['available']
    rows = src['rows']
    assert [r['frames'] for r in rows] == [40, 50, 60]
    assert [r['kept'] for r in rows] == [
        int(((keep >= a) & (keep < a + r['frames'])).sum())
        for a, r in zip(starts, rows)]
    top = max(rows, key=lambda r: r['loso_shrink'])
    assert top['source'] == 's1.wav' and top['loso_tensor'] == 'feat'
    # The asymmetric grid [-3, 40] becomes [-3, 3].
    assert top['loso_shrink'] == pytest.approx(1 - 6 / 43, rel=1e-6)
    assert top['loso']['frames'] == [7]
    assert top['loso']['step'] == pytest.approx(43 / 65535)
    assert top['loso']['step_without'] == pytest.approx(6 / 65535)
    assert src['growth']['median'][-1] == pytest.approx(1.0)


def test_a_source_that_moves_the_minimum_but_not_the_symmetric_bound_buys_nothing():
    """[-10, 6] -> [-10, 4] narrows the span; on a symmetric grid the bound is
    10 either way, so the step does not change."""
    rng = np.random.default_rng(13)
    arrays, starts = _with_sources(rng, [30, 30, 30], width=4)
    arrays['h'][:] = 0.01             # the same range in every source
    arrays['h'][starts] = 0.0
    feat = (0.1 * rng.standard_normal((90, 1, 4))).astype(np.float32)
    feat[5, 0, 0] = -10.0
    feat[starts[1] + 3, 0, 1] = 6.0
    feat[starts[2] + 3, 0, 1] = 4.0
    arrays['feat'] = feat
    pool = _pool(arrays, {'source_files': ['a.wav', 'b.wav', 'c.wav']})
    score = calib_analyze.frame_score(pool.stats)
    sym = calib_analyze.source_report(pool, None, score, {'feat': 'sym',
                                                          'h': 'sym'})
    b = sym['rows'][1]
    assert b['loso_tensor'] != 'feat' or b['loso_shrink'] == 0.0
    asym = calib_analyze.source_report(pool, None, score, pool.schemes('asym'))
    assert asym['rows'][1]['loso_tensor'] == 'feat'
    assert asym['rows'][1]['loso_shrink'] == pytest.approx(1 - 14 / 16)


def test_groups_left_out_in_turn_find_the_source_the_others_do_not_cover():
    rng = np.random.default_rng(14)
    lengths = [30] * 6
    arrays, starts = _with_sources(rng, lengths, width=4)
    arrays['feat'] = np.clip(arrays['feat'], -2, 2)
    arrays['feat'][starts[4] + 2, 0, 1] = 12.0       # only e.wav goes this far
    # An all-positive input: b.wav goes below the others' minimum but stays
    # inside their grid, which holds 0, so nothing of it is clipped.
    arrays['mag'] = rng.uniform(1.0, 2.0, (180, 1, 3)).astype(np.float32)
    arrays['mag'][np.asarray(starts) + 1, 0, 1] = 2.0    # one top in every source
    arrays['mag'][starts[1] + 4, 0, 0] = 0.5
    names = ['%s.wav' % c for c in 'abcdef']
    pool = _pool(arrays, {'source_files': names})
    folds = calib_analyze.source_folds(pool, pool.schemes('asym'))
    assert {row['tensor']: row for row in folds['rows']}['mag']['overshoot'] == 0
    assert folds['available'] and folds['sources'] == 6
    assert folds['groups'] == calib_analyze.FOLDS
    row = folds['rows'][0]
    assert row['tensor'] == 'feat' and row['source'] == 'e.wav'
    others = np.delete(arrays['feat'].reshape(180, -1),
                       np.arange(starts[4], starts[4] + 30), axis=0)
    assert row['overshoot'] >= (12.0 - others.max()) / (others.max()
                                                         - min(others.min(), 0))
    assert row['frames_past_pct'] > 0
    texts = [c['text'] for c in calib_analyze.analyze(
        pool, count=40)['conclusions']]
    assert any('輪流拿掉一組' in t and 'e.wav' in t for t in texts)


def _cal_and_held(tmp_path, a, b, report_a, report_b):
    cal = calib_analyze.load_artifact(_write(tmp_path, a, report_a, 'cal.npz'))
    held = calib_analyze.load_artifact(_write(tmp_path, b, report_b,
                                              'held.npz'))
    return (cal, calib_analyze.scan_artifact(cal),
            calib_analyze.scan_artifact(held))


def test_coverage_refuses_another_graph_and_warns_on_shared_sources(tmp_path):
    rng = np.random.default_rng(7)
    a, _ = _with_sources(rng, [30, 30])
    b, _ = _with_sources(rng, [25, 35])
    _, cal, other = _cal_and_held(
        tmp_path, a, b,
        {'checkpoint_sha256': 'x', 'source_files': ['p.wav', 'q.wav']},
        {'checkpoint_sha256': 'y', 'source_files': ['r.wav', 's.wav']})
    with pytest.raises(ValueError, match='checkpoint_sha256'):
        calib_analyze.coverage(cal, other)
    _, cal, held = _cal_and_held(
        tmp_path, a, b,
        {'checkpoint_sha256': 'x', 'source_files': ['p.wav', 'q.wav']},
        {'checkpoint_sha256': 'x', 'source_files': ['q.wav', 's.wav']})
    result = calib_analyze.coverage(cal, held)
    assert any('appear in both' in w for w in result['warnings'])
    for row in result['rows']:
        x_cal = a[row['tensor']].reshape(60, -1).astype(np.float64)
        lo, hi = _grid(x_cal.min(), x_cal.max(),
                       'sym' if row['tensor'] == 'h' else 'asym')
        x = b[row['tensor']].reshape(60, -1)
        outside = (x < lo) | (x > hi)
        assert row['frames_clip_pct'] == pytest.approx(
            100.0 * outside.any(axis=1).mean())
        assert set(row['worst_source']) == {'frames_clip_pct'}


def test_coverage_counts_frames_on_the_power_of_two_grid(tmp_path):
    rng = np.random.default_rng(11)
    a, _ = _with_sources(rng, [40, 40])
    b, _ = _with_sources(rng, [30, 50])
    _, cal, held = _cal_and_held(tmp_path, a, b, {}, {})
    result = calib_analyze.coverage(cal, held, 'pow2')
    for row in result['rows']:
        x_cal = a[row['tensor']].reshape(80, -1).astype(np.float64)
        lo, hi = _grid(x_cal.min(), x_cal.max(), 'pow2')
        assert (row['lo'], row['hi']) == (lo, hi)
        x = b[row['tensor']].reshape(80, -1)
        outside = (x < lo) | (x > hi)
        assert row['frames_clip_pct'] == pytest.approx(
            100.0 * outside.any(axis=1).mean())


def test_cli_analyze_writes_the_selection_and_coverage_scores_it(tmp_path,
                                                                 capsys):
    rng = np.random.default_rng(8)
    arrays, _ = _with_sources(rng, [70, 80, 90])
    report = {'source_files': ['a.wav', 'b.wav', 'c.wav']}
    path = _write(tmp_path, arrays, report, 'pool.npz')
    held = _write(tmp_path, _with_sources(rng, [60, 60])[0],
                  {'source_files': ['d.wav', 'e.wav']}, 'held.npz')
    html = tmp_path / 'r.html'
    out = tmp_path / 'sel.npz'
    assert calib_analyze.main(['analyze', path, '--out', str(html),
                               '--count', '100', '--output', str(out)]) == 0
    page = html.read_text(encoding='utf-8')
    assert '/*__DATA__*/null' not in page and '"conclusions":' in page
    assert 'no graph beside the artifact' in capsys.readouterr().out
    sel = calib_analyze.load_artifact(str(out))
    assert sel.frames == 100
    assert sel.report['state_inputs'] == ['h']
    assert sel.report['selection']['parent_frames'] == 240
    for name, value in arrays.items():
        assert sel.arrays[name].min() == value.min()
        assert sel.arrays[name].max() == value.max()
    assert calib_analyze.main(['coverage', str(out), '--against', held,
                               '--json', str(tmp_path / 'c.json')]) == 0
    assert json.loads((tmp_path / 'c.json').read_text())['frames'] == 100


def test_bits_and_jobs_options_reject_out_of_range_values(tmp_path):
    with pytest.raises(SystemExit):
        calib_analyze.main(['analyze', 'x.npz', '--bits', '1'])
    with pytest.raises(SystemExit):
        calib_analyze.main(['analyze', 'x.npz', '--jobs', '0'])


def test_one_source_and_too_few_frames_are_both_flagged():
    rng = np.random.default_rng(10)
    arrays, _ = _with_sources(rng, [120])
    texts = [c['text'] for c in calib_analyze.analyze(
        _pool(arrays, {'source_files': ['only.wav']}), count=1000)['conclusions']]
    assert any('同一個來源' in t for t in texts)
    assert any('等於沒有挑選' in t and '--select' in t for t in texts)


def test_conclusion_matches_whether_the_extremes_fit_the_count():
    rng = np.random.default_rng(12)
    arrays = {name: _heavy(rng, 60, 8) for name in ('a', 'b', 'c')}
    for index, name in enumerate(arrays):
        arrays[name][10 * index] = 50.0       # max in frame 0, 10, 20
        arrays[name][10 * index + 5] = -50.0  # min in frame 5, 15, 25
    pool = _pool(arrays)

    def texts(count):
        return [c['text'] for c in
                calib_analyze.analyze(pool, count=count)['conclusions']]

    short = texts(4)
    assert any('超過 --count 4' in t and '比整池窄' in t for t in short)
    assert not any('和整池相同' in t for t in short)
    assert any('保有整池每個輸入 tensor' in t for t in texts(20))
    assert any('保有整池每個輸入 tensor' in t for t in texts(6))  # count == must
    whole = texts(1000)
    assert any('全部 60 個 frame 都保留' in t for t in whole)
    assert not any('至少需要 0 個' in t for t in whole)


def test_one_frame_artifact_is_analyzed():
    feat = np.arange(8, dtype=np.float32).reshape(1, 1, 8)
    analysis = calib_analyze.analyze(_pool({'feat': feat}))
    assert list(analysis['recommend']['keep_frames']) == [0]
    assert analysis['tensors']['feat']['chart']['start'].tolist() == [0]


def test_bad_output_name_is_refused_before_the_scan(tmp_path, monkeypatch):
    rng = np.random.default_rng(13)
    path = _write(tmp_path, {'feat': _heavy(rng, 50, 8)})
    monkeypatch.setattr(calib_analyze, 'scan_artifact', lambda *args: 1 / 0)
    with pytest.raises(SystemExit):
        calib_analyze.main(['analyze', path, '--count', '10',
                            '--out', str(tmp_path / 'r.html'),
                            '--output', str(tmp_path / 'pool_10')])
    assert not (tmp_path / 'r.html').exists()


def test_a_bin_output_the_write_would_refuse_is_refused_before_the_scan(
        tmp_path, monkeypatch, capsys):
    rng = np.random.default_rng(13)
    path = _write(tmp_path, {'feat': _heavy(rng, 50, 8)})
    (tmp_path / 'taken').mkdir()
    monkeypatch.setattr(calib_analyze, 'scan_artifact', lambda *args: 1 / 0)
    with pytest.raises(SystemExit):
        calib_analyze.main(['analyze', path, '--count', '10', '--format',
                            'bin', '--out', str(tmp_path / 'r.html'),
                            '--output', str(tmp_path / 'taken') + '/'])
    assert 'already exists' in capsys.readouterr().err
    assert not (tmp_path / 'r.html').exists()
    with pytest.raises(SystemExit):
        calib_analyze.main(['analyze', path, '--count', '10', '--format',
                            'bin', '--out', str(tmp_path / 'sel'),
                            '--output', str(tmp_path / 'sel')])
    assert 'at or inside' in capsys.readouterr().err


def test_contiguous_artifact_gets_no_recommendation(tmp_path):
    rng = np.random.default_rng(11)
    arrays, _ = _with_sources(rng, [80])
    report = {'model_family': 'Align_CRUSE', 'contiguous_state_frames': 80}
    analysis = calib_analyze.analyze(_pool(arrays, report), count=20)
    assert analysis['recommend'] is None
    assert any('不中斷' in c['text'] for c in analysis['conclusions'])
    path = _write(tmp_path, arrays, report)
    with pytest.raises(SystemExit, match='uninterrupted'):
        calib_analyze.main(['analyze', path, '--output',
                            str(tmp_path / 'o.npz')])


def test_a_tensor_whose_values_fit_is_sampled_whole_past_the_frame_sample(
        monkeypatch):
    """Every value enters the SQNR estimate while the total fits SAMPLE_CAP,
    however many frames hold them; a larger tensor is drawn from the frames
    of smallest key."""
    monkeypatch.setattr(calib_analyze, 'SAMPLE_FRAMES', 10)
    rng = np.random.default_rng(15)
    small = _heavy(rng, 40, 4)
    monkeypatch.setattr(calib_analyze, 'SAMPLE_CAP', 200)
    pool = _pool({'small': small, 'big': _heavy(rng, 40, 8)})
    assert np.array_equal(pool.stats['small']['_sample'], small.reshape(-1))
    assert not pool.stats['small']['global']['sampled']
    big = pool.stats['big']
    assert big['global']['sampled'] and big['_sample'].size == 10 * 8


def _captured(art, numbers, scans, reserve):
    """A Pool of ``art`` scanned as a capture does: recording k numbered
    ``numbers[k]`` and dealt to scan ``k % scans``."""
    starts, reliable, _ = calib_analyze.detect_segments(art)
    assert reliable
    bounds = list(starts) + [art.frames]
    parts = [calib_analyze.Scan(art.specs, None, art.report, reserve=reserve)
             for _ in range(scans)]
    for k, number in enumerate(numbers):
        for i in range(bounds[k], bounds[k + 1]):
            parts[k % scans].add({name: value[i] for name, value
                                  in art.arrays.items()}, number)
    names = dict(zip(numbers, art.report['source_files']))
    return calib_analyze.assemble([scan.part() for scan in parts], art.report,
                                  source_names=names, reserve=reserve)


def test_the_fill_candidates_do_not_depend_on_how_recordings_are_numbered():
    """Frames are keyed by content: a capture numbering its recordings its own
    way, in one process or several, keeps the candidates a scan of the stored
    frames finds."""
    rng = np.random.default_rng(21)
    arrays, _ = _with_sources(rng, [70, 50, 90, 40])
    art = calib_analyze.Artifact('x', 'npz', arrays, {
        'source_files': ['r%d.wav' % k for k in range(4)]})
    whole = calib_analyze.scan_artifact(art, reserve=60)
    want = calib_analyze.select_patterns(whole, 20, 0)['keep']
    for numbers, scans in (([3, 8, 11, 40], 1), ([5, 6, 7, 9], 3)):
        pool = _captured(art, numbers, scans, 60)
        assert pool.count == whole.count > 60
        np.testing.assert_array_equal(pool.candidates, whole.candidates)
        np.testing.assert_array_equal(pool.features, whole.features)
        keep = calib_analyze.select_patterns(pool, 20, 0)['keep']
        np.testing.assert_array_equal(keep, want)
        for name, value in arrays.items():
            np.testing.assert_array_equal(pool.frames(keep, name), value[keep])


def test_equal_frames_take_one_place_among_the_fill_candidates():
    rng = np.random.default_rng(22)
    feat = _heavy(rng, 300, 8)
    feat[100:250] = 0.0                     # a stretch of digital silence
    art = calib_analyze.Artifact('x', 'npz', {'feat': feat}, {})
    flat = feat.reshape(300, -1).astype(np.float64)
    keys = calib_analyze._frame_keys(0, [flat.min(1), flat.max(1),
                                         flat.mean(1)])
    # A reserve the silence's key falls inside, so its repeats would crowd it.
    distinct = np.unique(keys)
    reserve = int(np.searchsorted(distinct, keys[100])) + 5
    silent = np.arange(100, 250)
    pool = calib_analyze.scan_artifact(art, reserve=reserve)
    assert np.isin(pool.candidates, silent).sum() == 1
    assert pool.candidates.size >= reserve
    # A capture holds the frames it needs; all of them while they fit.
    scan = calib_analyze.Scan(art.specs, reserve=reserve)
    for i in range(300):
        scan.add({'feat': feat[i]})
    held = calib_analyze.assemble([scan.part()], reserve=reserve)
    np.testing.assert_array_equal(held.candidates, pool.candidates)
    np.testing.assert_array_equal(held.frames(held.candidates, 'feat'),
                                  feat[held.candidates])
    scan = calib_analyze.Scan(art.specs, reserve=300)
    for i in range(300):
        scan.add({'feat': feat[i]})
    held = calib_analyze.assemble([scan.part()], reserve=300)
    keep = calib_analyze.select_patterns(held, 300, 0)['keep']
    np.testing.assert_array_equal(held.frames(keep, 'feat'), feat)


def test_a_selection_from_one_recording_still_names_it(tmp_path):
    rng = np.random.default_rng(23)
    arrays, _ = _with_sources(rng, [80])
    path = _write(tmp_path, arrays, {'source_files': ['only.wav']})
    calib_analyze.main(['analyze', path, '--count', '10',
                        '--out', str(tmp_path / 'page.html'),
                        '--output', str(tmp_path / 'o.npz')])
    report = calib_analyze.load_artifact(str(tmp_path / 'o.npz')).report
    assert report['source_files'] == ['only.wav']
    assert report['selection']['parent'] == 'a.npz'
