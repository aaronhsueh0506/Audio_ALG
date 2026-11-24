"""``PTQ/calib_analyze.py`` selection, exact clip counts, sources and coverage."""

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


def _brute(x, keep, scheme):
    flat = x.reshape(len(x), -1).astype(np.float64)
    lo, hi = _grid(flat[keep].min(), flat[keep].max(), scheme)
    out = (flat < lo) | (flat > hi)
    return out.sum(), 100.0 * out.mean(), 100.0 * out.any(axis=1).mean()


def test_zero_budget_keeps_a_unique_extreme_even_when_it_is_not_sampled(
        tmp_path, monkeypatch):
    # The old budget test ran on a value sample; with the extreme element
    # outside the sample it saw no clipping and dropped the frame.
    rng = np.random.default_rng(0)
    feat = rng.standard_normal((200, 1, 20)).astype(np.float32)
    feat[37, 0, 5] = 50.0
    path = _write(tmp_path, {'feat': feat})
    monkeypatch.setattr(calib_analyze, 'SAMPLE_CAP', 50)
    keep_out = tmp_path / 'keep.json'
    calib_analyze.main(['select', path, '--max-clip-pct', '0',
                        '--count', '1000', '--keep-out', str(keep_out)])
    keep = json.loads(keep_out.read_text())['keep_frames']
    assert len(keep) == 200 and 37 in keep


@pytest.mark.parametrize('scheme', ['asym', 'sym', 'pow2'])
@pytest.mark.parametrize('pct', [0.05, 0.5, 2.0, 10.0])
def test_drop_budget_matches_brute_force(scheme, pct, monkeypatch):
    rng = np.random.default_rng(1)
    feat = _heavy(rng, 120, 12)
    art = calib_analyze.Artifact('x', 'npz', {'feat': feat}, {})
    monkeypatch.setattr(calib_analyze, 'SAMPLE_CAP', 64)
    stats = calib_analyze.all_stats(art)
    schemes = {'feat': scheme}
    order, _ = calib_analyze.frame_order(stats)
    got = calib_analyze.drop_budget(stats, schemes, order, pct)
    limit = pct / 100.0 * feat.size
    expected = 0
    for k in range(1, 120):
        if _brute(feat, order[k:], scheme)[0] > limit:
            break
        expected = k
    assert got == expected
    if limit >= 1:              # 0.05 % of 1440 values allows none
        assert got > 0


def test_budget_takes_back_frames_that_clip_nothing_on_a_power_of_two_grid():
    rng = np.random.default_rng(12)
    feat = _heavy(rng, 200, 8)
    art = calib_analyze.Artifact('x', 'npz', {'feat': feat}, {})
    stats = calib_analyze.all_stats(art)
    schemes = {'feat': 'pow2'}
    order, _ = calib_analyze.frame_order(stats)
    k = calib_analyze.drop_budget(stats, schemes, order, 5.0)
    pool, dropped = calib_analyze.budget_pool(stats, schemes, order, 5.0)
    flat = feat.reshape(200, -1).astype(np.float64)
    lo, hi = _grid(flat[order[k:]].min(), flat[order[k:]].max(), 'pow2')
    assert _grid(flat[pool].min(), flat[pool].max(), 'pow2') == (lo, hi)
    out = np.setdiff1d(np.arange(200), pool)
    assert out.size == dropped < k
    assert all(((flat[f] < lo) | (flat[f] > hi)).any() for f in out)


@pytest.mark.parametrize('scheme', ['asym', 'sym', 'pow2'])
def test_subset_table_clip_shares_match_brute_force(scheme):
    rng = np.random.default_rng(2)
    feat = _heavy(rng, 150, 10)
    art = calib_analyze.Artifact('x', 'npz', {'feat': feat}, {})
    stats = calib_analyze.all_stats(art)
    schemes = {'feat': scheme}
    models = {'feat': calib_analyze.GridError(stats['feat']['_sample'], scheme)}
    for draw in range(5):
        keep = rng.choice(150, 30, replace=False)
        row = calib_analyze.subset_table(stats, schemes, models, keep, 150, '',
                                         (8, 16))['rows'][0]
        _, clip_pct, frames_pct = _brute(feat, keep, scheme)
        assert row['clip_pct'] == pytest.approx(clip_pct, abs=1e-12)
        assert row['frames_clip_pct'] == pytest.approx(frames_pct, abs=1e-12)
        assert row['sqnr16_typ'] > row['sqnr8_typ']


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
    art = calib_analyze.Artifact('x', 'npz', {'a': a, 'b': b}, {})
    result = calib_analyze.analyze(art, 'pow2', count=40)
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
    art = calib_analyze.Artifact('x', 'npz', arrays, {})
    stats = calib_analyze.all_stats(art)
    schemes = art.schemes('asym')
    sel = calib_analyze.select_patterns(art, stats, schemes, 40)
    keep = sel['keep']
    assert keep.size == 40 and sel['k_budget'] == 0
    for name, x in arrays.items():
        flat = x.reshape(300, -1)
        assert flat[keep].min() == flat.min()
        assert flat[keep].max() == flat.max()
    whole = calib_analyze.select_patterns(art, stats, schemes, 1000)
    assert whole['keep'].size == 300 and whole['notes']


def test_state_inputs_are_scored_on_a_symmetric_grid():
    rng = np.random.default_rng(4)
    arrays, _ = _with_sources(rng, [50, 60])
    art = calib_analyze.Artifact('x', 'npz', arrays, {})
    assert art.state == ['h']
    assert art.schemes('asym') == {'feat': 'asym', 'h': 'sym'}
    # sym and pow2 already put 0 at zero point 0; state keeps the input grid.
    assert art.schemes('pow2') == {'feat': 'pow2', 'h': 'pow2'}


def test_sources_unavailable_without_recorded_source_files():
    rng = np.random.default_rng(5)
    arrays, _ = _with_sources(rng, [40, 40, 40])
    art = calib_analyze.Artifact('x', 'npz', arrays, {})
    analysis = calib_analyze.analyze(art, count=50)
    assert analysis['sources']['available'] is False
    assert any('來源資訊不可用' in c['text'] for c in analysis['conclusions'])


def test_source_report_finds_the_source_that_holds_the_range():
    rng = np.random.default_rng(6)
    arrays, starts = _with_sources(rng, [40, 50, 60])
    arrays['feat'] = np.clip(arrays['feat'], -3, 3)
    arrays['feat'][starts[1] + 7, 0, 2] = 40.0
    report = {'source_files': ['s0.wav', 's1.wav', 's2.wav']}
    art = calib_analyze.Artifact('x', 'npz', arrays, report)
    stats = calib_analyze.all_stats(art)
    _, score = calib_analyze.frame_order(stats)
    keep = np.arange(0, 150, 3)
    src = calib_analyze.source_report(art, stats, keep, score)
    assert src['available']
    rows = src['rows']
    assert [r['frames'] for r in rows] == [40, 50, 60]
    assert [r['kept'] for r in rows] == [
        int(((keep >= a) & (keep < a + r['frames'])).sum())
        for a, r in zip(starts, rows)]
    assert max(rows, key=lambda r: r['loso_shrink'])['source'] == 's1.wav'
    assert rows[1]['loso_shrink'] > 0.8 and rows[1]['loso_tensor'] == 'feat'
    assert src['growth']['median'][-1] == pytest.approx(1.0)


def test_coverage_refuses_another_graph_and_warns_on_shared_sources(tmp_path):
    rng = np.random.default_rng(7)
    a, _ = _with_sources(rng, [30, 30])
    b, _ = _with_sources(rng, [25, 35])
    cal = calib_analyze.load_artifact(_write(
        tmp_path, a, {'checkpoint_sha256': 'x', 'source_files': ['p.wav', 'q.wav']},
        'cal.npz'))
    other = calib_analyze.load_artifact(_write(
        tmp_path, b, {'checkpoint_sha256': 'y', 'source_files': ['r.wav', 's.wav']},
        'other.npz'))
    with pytest.raises(ValueError, match='checkpoint_sha256'):
        calib_analyze.coverage(cal, other)
    held = calib_analyze.load_artifact(_write(
        tmp_path, b, {'checkpoint_sha256': 'x', 'source_files': ['q.wav', 's.wav']},
        'held.npz'))
    keep = np.arange(0, 60, 4)
    result = calib_analyze.coverage(cal, held, keep)
    assert any('appear in both' in w for w in result['warnings'])
    for row in result['rows']:
        x_cal = a[row['tensor']].reshape(60, -1)[keep].astype(np.float64)
        lo, hi = x_cal.min(), x_cal.max()
        if row['tensor'] == 'h':
            top = max(abs(lo), abs(hi))
            lo, hi = -top, top
        x = b[row['tensor']].reshape(60, -1)
        outside = (x < lo) | (x > hi)
        assert row['clip_pct'] == pytest.approx(100.0 * outside.mean())
        assert row['frames_clip_pct'] == pytest.approx(
            100.0 * outside.any(axis=1).mean())
        assert set(row['worst_source']) == {'frames_clip_pct', 'sqnr8_all',
                                            'sqnr16_all'}


def test_coverage_counts_exactly_on_the_power_of_two_grid(tmp_path):
    rng = np.random.default_rng(11)
    a, _ = _with_sources(rng, [40, 40])
    b, _ = _with_sources(rng, [30, 50])
    cal = calib_analyze.load_artifact(_write(tmp_path, a, {}, 'cal.npz'))
    held = calib_analyze.load_artifact(_write(tmp_path, b, {}, 'held.npz'))
    keep = np.arange(1, 80, 5)
    result = calib_analyze.coverage(cal, held, keep, 'pow2')
    for row in result['rows']:
        x_cal = a[row['tensor']].reshape(80, -1)[keep].astype(np.float64)
        lo, hi = _grid(x_cal.min(), x_cal.max(), 'pow2')
        assert (row['lo'], row['hi']) == (lo, hi)
        x = b[row['tensor']].reshape(80, -1)
        outside = (x < lo) | (x > hi)
        assert row['clip_pct'] == pytest.approx(100.0 * outside.mean())
        assert row['frames_clip_pct'] == pytest.approx(
            100.0 * outside.any(axis=1).mean())


def test_cli_analyze_select_coverage_apply(tmp_path):
    rng = np.random.default_rng(8)
    arrays, _ = _with_sources(rng, [70, 80, 90])
    report = {'source_files': ['a.wav', 'b.wav', 'c.wav']}
    path = _write(tmp_path, arrays, report, 'pool.npz')
    held = _write(tmp_path, _with_sources(rng, [60, 60])[0],
                  {'source_files': ['d.wav', 'e.wav']}, 'held.npz')
    html = tmp_path / 'r.html'
    assert calib_analyze.main(['analyze', path, '--out', str(html),
                               '--count', '100']) == 0
    page = html.read_text(encoding='utf-8')
    assert '/*__DATA__*/null' not in page and '"conclusions":' in page
    keep = tmp_path / 'keep.json'
    out = tmp_path / 'sel.npz'
    assert calib_analyze.main(['select', path, '--count', '100',
                               '--keep-out', str(keep),
                               '--output', str(out)]) == 0
    assert calib_analyze.load_artifact(str(out)).frames == 100
    assert calib_analyze.main(['coverage', path, '--keep', str(keep),
                               '--against', held,
                               '--json', str(tmp_path / 'c.json')]) == 0
    assert json.loads((tmp_path / 'c.json').read_text())['frames'] == 100
    assert calib_analyze.main(['apply', path, '--keep', str(keep),
                               '--output', str(tmp_path / 'ap.npz')]) == 0


def test_bits_option_rejects_out_of_range_widths():
    with pytest.raises(SystemExit):
        calib_analyze.main(['analyze', 'x.npz', '--bits', '1'])


def test_one_source_and_too_few_frames_are_both_flagged():
    rng = np.random.default_rng(10)
    arrays, _ = _with_sources(rng, [120])
    art = calib_analyze.Artifact('x', 'npz', arrays,
                                 {'source_files': ['only.wav']})
    texts = [c['text'] for c in
             calib_analyze.analyze(art, count=1000)['conclusions']]
    assert any('同一個來源' in t for t in texts)
    assert any('等於沒有挑選' in t for t in texts)


def test_conclusion_matches_whether_the_extremes_fit_the_count():
    rng = np.random.default_rng(12)
    arrays = {name: _heavy(rng, 60, 8) for name in ('a', 'b', 'c')}
    for index, name in enumerate(arrays):
        arrays[name][10 * index] = 50.0       # max in frame 0, 10, 20
        arrays[name][10 * index + 5] = -50.0  # min in frame 5, 15, 25
    art = calib_analyze.Artifact('x', 'npz', arrays, {})

    def texts(count):
        return [c['text'] for c in
                calib_analyze.analyze(art, count=count)['conclusions']]

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
    art = calib_analyze.Artifact('x', 'npz', {'feat': feat}, {})
    assert calib_analyze.analyze(art)['curves']['k'] == [0]


def test_drop_list_outside_the_artifact_is_refused():
    art = calib_analyze.Artifact(
        'x', 'npz', {'feat': np.ones((5, 1, 4), dtype=np.float32)}, {})
    assert list(calib_analyze.keep_indices(art, {'drop_frames': [4]})) == [
        0, 1, 2, 3]
    for bad in ([5], [-1]):
        with pytest.raises(ValueError):
            calib_analyze.keep_indices(art, {'drop_frames': bad})


def test_bad_output_name_is_refused_before_the_keep_file(tmp_path):
    rng = np.random.default_rng(13)
    path = _write(tmp_path, {'feat': _heavy(rng, 50, 8)})
    keep = tmp_path / 'keep.json'
    with pytest.raises(SystemExit):
        calib_analyze.main(['select', path, '--count', '10',
                            '--keep-out', str(keep),
                            '--output', str(tmp_path / 'pool_10')])
    assert not keep.exists()


def test_contiguous_artifact_gets_no_recommendation():
    rng = np.random.default_rng(11)
    arrays, _ = _with_sources(rng, [80])
    art = calib_analyze.Artifact('x', 'npz', arrays, {
        'model_family': 'Align_CRUSE', 'contiguous_state_frames': 80})
    analysis = calib_analyze.analyze(art, count=20)
    assert analysis['recommend'] is None
    assert any('不中斷' in c['text'] for c in analysis['conclusions'])
