"""Align-ULCNet per-hop alignment events replayed through the official CLIs.

The C stream restarts the model on an alignment event (stream/aiaec/ulcnet/ulcnet_stream.c: ``changed``, or
``solid`` false -> true after an earlier hop, ``prev_solid`` carried across packets). Python has no delay-state
accessor, so ``inference.py --events`` and ``inference.py calib --events`` replay an event record file, through ONE
edge-rule helper (``alignment_events``). The tests here drive the CLIs themselves:

  * the helper against an independent transcription of the C rule, on random streams and named cases;
  * ``inference.py`` at N = 1, 2, 8 on the same WAV and record: the packet's frame_reset control (event on hop 3 ->
    ``00100000``), the padded last packet, the real hop count, end-of-packet state against the N = 1 stream, the next
    packet, the output length;
  * ``inference.py calib``: the captured ``frame_reset`` is the binary control, carries no PTQ range, N = 1 skips
    event hops and restarts its state;

Random weights (no trained checkpoint exists at 16 kHz / FFT 256): the tests prove the boundary, not audio quality.
"""

import contextlib
import io
import json
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from AIAEC.Align_ULCNet import AlignULCNet
from AIAEC.Align_ULCNet import alignment_events as ae
from AIAEC.Align_ULCNet import export_onnx as ex
from AIAEC.Align_ULCNet import inference as al_inference
from AIAEC.aiaec_common import SignalGrid
from AIAEC.aiaec_streaming import StreamISTFT
from AIAEC.dataset_gen import make_linear_aec_contract

SR, N_FFT, HOP = 16000, 256, 128
GRID = SignalGrid(SR, N_FFT, N_FFT, HOP)
F = GRID.n_freqs
CLIP = SR + 111 + 5                        # 16116 samples: 126 real hops, the last one partial
HOPS = -(-CLIP // HOP)                     # 126
CLIP_ALIGNED = HOP * 120                   # 15360 samples: 120 hops, the analyser still emits a 121st frame


def c_edge_rule(changed, solid):
    """Transcription of ulcnet_stream.c apply_alignment_edge / packet_events, written independently of the helper."""
    had_hop, prev_solid, out = 0, 0, []
    for c, s in zip(changed, solid):
        out.append(1 if (c or (had_hop and not prev_solid and s)) else 0)
        had_hop, prev_solid = 1, s
    return out


def record(hops, changed=(), solid_from=None, solid_until=None):
    """(changed, solid) lists of ``hops`` records."""
    flags = [1 if h in set(changed) else 0 for h in range(hops)]
    solid = [0] * hops
    if solid_from is not None:
        solid = [1 if (h >= solid_from and (solid_until is None or h < solid_until)) else 0 for h in range(hops)]
    return flags, solid


def write_record(path, changed, solid):
    path.write_text(json.dumps({'schema': ae.EVENTS_SCHEMA, 'events': [[c, s] for c, s in zip(changed, solid)]}))
    return str(path)


# Named streams over HOPS = 126 hops (16 packets of 8, the last one holding 6 real hops).
STREAMS = {
    'none': record(HOPS),
    'hop3': record(HOPS, changed=[2]),                                  # 1-based hop 3 -> packet slot 2
    'first_hop': record(HOPS, changed=[0]),
    'last_real_hop': record(HOPS, changed=[HOPS - 1]),                  # the partial hop, inside the short last packet
    'adjacent': record(HOPS, changed=[40, 41]),
    'cross_packet': record(HOPS, changed=[7, 8]),                       # last slot of packet 0, first of packet 1
    'solid_edge_after_earlier_hops': record(HOPS, solid_from=20),       # solid 0 for hops 0..19, then 1: edge at 20
    'solid_edge_across_packets': record(HOPS, solid_from=16),           # hop 15 (last of packet 1) low, hop 16 high
    'solid_from_first_hop_is_no_event': record(HOPS, solid_from=0),     # no earlier hop: no edge
    'late_after_attention_depth': record(HOPS, changed=[100, 119]),     # state warmed past D = 63 frames
    'mixed': record(HOPS, changed=[3, 4, 60], solid_from=30, solid_until=90),
}


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(1)
    return AlignULCNet(GRID, max_delay_seconds=0.5).eval()


@pytest.fixture(scope='module')
def checkpoint(model, tmp_path_factory):
    work = tmp_path_factory.mktemp('al256_events')
    linear = make_linear_aec_contract(SR, frame_size=N_FFT)
    contract = {
        'model_name': 'Align_ULCNet', 'task': AlignULCNet.task, 'sr': SR,
        'n_fft': N_FFT, 'win_len': N_FFT, 'hop_len': HOP,
        'loss_version': 'test', 'linear_aec': linear.as_dict(),
        'linear_aec_contract_hash': linear.fingerprint(),
        'ctor_max_delay_seconds': 0.5,
    }
    path = work / 'al256.pth'
    torch.save({'contract': contract, 'state_dict': model.state_dict()}, path)
    rng = np.random.RandomState(7)
    for name, length in (('', CLIP), ('_aligned', CLIP_ALIGNED)):
        error = 0.1 * rng.randn(length).astype('f')
        far = 0.1 * rng.randn(length).astype('f')
        sf.write(work / ('err%s.wav' % name), error, SR)
        sf.write(work / ('far%s.wav' % name), far, SR)
    return work, str(path)


# ---- the helper ----------------------------------------------------------------------------------------------

def test_edge_rule_equals_the_c_transcription_on_random_streams():
    rng = np.random.RandomState(3)
    for _ in range(300):
        n = int(rng.randint(1, 40))
        changed = (rng.rand(n) < 0.1).astype(int).tolist()
        solid = (rng.rand(n) < 0.6).astype(int).tolist()
        assert ae.edge_flags(changed, solid) == c_edge_rule(changed, solid)
        tracker = ae.EdgeTracker()
        assert [tracker.step(c, s) for c, s in zip(changed, solid)] == c_edge_rule(changed, solid)


def test_edge_rule_named_cases():
    # solid already true on the first hop: no earlier hop, no event; changed on hop 0 is an event
    assert ae.edge_flags([0, 0, 0], [1, 1, 1]) == [0, 0, 0]
    assert ae.edge_flags([1, 0, 0], [0, 0, 0]) == [1, 0, 0]
    # false -> true only counts after an earlier hop; true -> false and a held true are not events
    assert ae.edge_flags([0] * 6, [0, 0, 1, 1, 0, 1]) == [0, 0, 1, 0, 0, 1]
    # changed with solid held high is still an event
    assert ae.edge_flags([0, 1, 0], [1, 1, 1]) == [0, 1, 0]


def test_packet_control_pads_with_zero_and_tail_frames_are_never_events():
    events = ae.AlignmentEvents(*record(10, changed=[2, 9]))
    assert events.packet_control(0, 8, 8) == [0, 0, 1, 0, 0, 0, 0, 0]
    assert events.packet_control(8, 2, 8) == [0, 1, 0, 0, 0, 0, 0, 0]
    # a short packet: slots past ``real`` are 0 even if the record kept going
    assert events.packet_control(0, 3, 8) == [0, 0, 1, 0, 0, 0, 0, 0]
    assert events.packet_control(0, 2, 8) == [0] * 8
    assert events.flag(10) == events.flag(500) == events.flag(-1) == 0      # tail frames past the record


def test_record_formats_agree_and_bad_records_are_refused(tmp_path):
    changed, solid = record(12, changed=[3], solid_from=5)
    paths = [write_record(tmp_path / 'a.json', changed, solid)]
    (tmp_path / 'b.json').write_text(json.dumps([{'changed': c, 'solid': s} for c, s in zip(changed, solid)]))
    (tmp_path / 'c.csv').write_text('hop,changed,solid\n' + ''.join(
        '%d,%d,%d\n' % (i, c, s) for i, (c, s) in enumerate(zip(changed, solid))))
    np.savez(tmp_path / 'd.npz', changed=np.array(changed), solid=np.array(solid))
    paths += [str(tmp_path / name) for name in ('b.json', 'c.csv', 'd.npz')]
    flags = [ae.AlignmentEvents.load(p).flags for p in paths]
    assert all(f == c_edge_rule(changed, solid) for f in flags)

    (tmp_path / 'bad.json').write_text(json.dumps([[0, 2]]))
    with pytest.raises(ValueError, match='0/1'):
        ae.AlignmentEvents.load(str(tmp_path / 'bad.json'))
    (tmp_path / 'bad.csv').write_text('hop,changed,solid\n0,0,0\n2,0,0\n')
    with pytest.raises(ValueError, match='hop column'):
        ae.AlignmentEvents.load(str(tmp_path / 'bad.csv'))
    (tmp_path / 'bad.txt').write_text('x')
    with pytest.raises(ValueError, match='.json, .csv or .npz'):
        ae.AlignmentEvents.load(str(tmp_path / 'bad.txt'))
    events = ae.AlignmentEvents.load(paths[0])
    events.require_hops(12 * HOP, HOP)                 # exactly full hops
    events.require_hops(11 * HOP + 1, HOP)             # a partial last hop is a real hop
    for samples in (11 * HOP, 12 * HOP + 1):
        with pytest.raises(ValueError, match='hop records'):
            events.require_hops(samples, HOP)


def test_csv_with_a_bom_loads_and_a_plain_npy_is_not_an_archive(tmp_path):
    """A spreadsheet export carries a UTF-8 BOM; a bare .npy renamed to .npz is refused as a ValueError."""
    (tmp_path / 'bom.csv').write_text('changed,solid\n1,0\n0,1\n', encoding='utf-8-sig')
    loaded = ae.AlignmentEvents.load(str(tmp_path / 'bom.csv'))
    assert loaded.changed == [1, 0] and loaded.solid == [0, 1]
    np.save(tmp_path / 'plain.npy', np.zeros(3))
    (tmp_path / 'plain.npy').rename(tmp_path / 'plain.npz')
    with pytest.raises(ValueError, match='not an NPZ archive'):
        ae.AlignmentEvents.load(str(tmp_path / 'plain.npz'))


def test_records_are_validated_before_any_cast(tmp_path):
    """A value that is not exactly 0/1, or an array that is not 1-D, is refused with the file name -- never cast."""
    def refused(name, match, **arrays):
        path = tmp_path / name
        np.savez(path, **arrays)
        with pytest.raises(ValueError, match=match) as info:
            ae.AlignmentEvents.load(str(path))
        assert name in str(info.value)

    solid = np.zeros(3)
    refused('frac2d.npz', '1-D', changed=np.array([[0, .9, 1.9]]), solid=solid.reshape(1, 3))
    refused('column.npz', '1-D', changed=np.array([[0], [1], [0]]), solid=solid)
    refused('scalar.npz', '1-D', changed=np.array(1), solid=solid)
    refused('frac1d.npz', r'0/1, got \[0.9\]', changed=np.array([0, .9, 1.0]), solid=solid)
    refused('two.npz', r'got \[2\]', changed=np.array([0, 2, 1]), solid=solid)
    refused('neg.npz', r'got \[-1\]', changed=solid, solid=np.array([0, -1, 0]))
    refused('nan.npz', 'nan', changed=np.array([0, np.nan, 1.0]), solid=solid)
    refused('inf.npz', 'inf', changed=np.array([0, np.inf, 1.0]), solid=solid)
    refused('str.npz', 'dtype', changed=np.array(['0', '1', '0']), solid=solid)
    refused('cplx.npz', 'dtype', changed=np.array([0, 1j, 0]), solid=solid)
    refused('obj.npz', 'cannot be read', changed=np.array([0, None, 1], dtype=object), solid=solid)
    # exact binary in any numeric dtype is accepted
    for dtype in (bool, np.int8, np.uint8, np.int64, np.float32, np.float64):
        path = tmp_path / ('ok_%s.npz' % np.dtype(dtype).name)
        np.savez(path, changed=np.array([0, 1, 0], dtype), solid=np.array([0, 0, 1], dtype))
        loaded = ae.AlignmentEvents.load(str(path))
        assert loaded.changed == [0, 1, 0] and loaded.solid == [0, 0, 1]

    for name, text, match in (
            ('frac.json', '[[0, 0.9], [1.9, 0]]', r'got 1.9 at record 1'),
            ('str.json', '[["1", 0]]', 'got \'1\''),
            ('nan.json', '[[NaN, 0]]', 'nan'),
            ('frac.csv', 'changed,solid\n0,0\n0.9,0\n', r'got 0.9 at record 1'),
            ('inf.csv', 'changed,solid\n0,inf\n', 'inf'),
            ('text.csv', 'changed,solid\nTrue,0\n', "got 'True' in column changed"),
            ('blank.csv', 'changed,solid\n,0\n', 'column changed at record 0'),
            ('short.csv', 'changed,solid\n1\n', 'column solid at record 0')):
        (tmp_path / name).write_text(text)
        with pytest.raises(ValueError, match=match) as info:
            ae.AlignmentEvents.load(str(tmp_path / name))
        assert name in str(info.value)
    # exact 1.0 / 0.0 in JSON and CSV is still binary
    (tmp_path / 'f.json').write_text('[[1.0, 0.0], [true, false]]')
    (tmp_path / 'f.csv').write_text('changed,solid\n1.0,0\n0,1\n')
    assert ae.AlignmentEvents.load(str(tmp_path / 'f.json')).changed == [1, 1]
    assert ae.AlignmentEvents.load(str(tmp_path / 'f.csv')).solid == [0, 1]


# ---- inference.py --events -----------------------------------------------------------------------------------

def run_inference(work, checkpoint_path, out_name, n=None, events=None, wav='', input_is_linear_error=True):
    """The official CLI entry (parser + ``inference.main``), with the packet graph calls recorded."""
    argv = [checkpoint_path, str(work / ('err%s.wav' % wav)), str(work / ('far%s.wav' % wav)), str(work / out_name)]
    if input_is_linear_error:
        argv.append('--input-is-linear-error')
    if n is not None:
        argv += ['--frames-per-call', str(n)]
    if events is not None:
        argv += ['--events', events]
    args = al_inference.build_parser().parse_args(argv)
    calls = []
    real_run_packet = ex.run_packet

    def spy(wrapper, model, error_c, far_c, event_list, state):
        enhanced, next_state = real_run_packet(wrapper, model, error_c, far_c, event_list, state)
        calls.append({'error': error_c.detach().cpu(), 'far': far_c.detach().cpu(), 'events': list(event_list),
                      'state_in': tuple(v.detach().cpu().clone() for v in state),
                      'state_out': tuple(v.detach().cpu().clone() for v in next_state),
                      'enhanced': enhanced.detach().cpu()})
        return enhanced, next_state

    pushed = []
    real_push = StreamISTFT.push

    def push_spy(self, frame):
        pushed.append(frame.detach().cpu().clone())
        return real_push(self, frame)

    out = io.StringIO()
    ex.run_packet = spy
    StreamISTFT.push = push_spy
    try:
        with contextlib.redirect_stdout(out):
            al_inference.main(args)
    finally:
        ex.run_packet = real_run_packet
        StreamISTFT.push = real_push
    audio, rate = sf.read(str(work / out_name), dtype='float32')
    assert rate == SR
    run_inference.pushed = pushed                                       # the spectra handed to the synthesis
    return audio, out.getvalue(), calls


def sequential_reference(model, error_c, far_c, flags, state):
    """The N = 1 stream: reset + bypass at the flagged frames, graph step otherwise."""
    wrapper = ex.AlignUlcnetStreamingExport(model).eval()
    frames = []
    for index in range(error_c.shape[1]):
        *signals, compressed = ex.stream_features(
            model, torch.view_as_real(error_c[:, index:index + 1]), torch.view_as_real(far_c[:, index:index + 1]))
        if flags[index]:
            state = tuple(torch.zeros_like(value) for value in state)
            frames.append(error_c[:, index:index + 1])
        else:
            outputs = wrapper(*signals, *state)
            state = ex.next_state(outputs)
            frames.append(torch.view_as_complex(ex.host_back_end(model, compressed, outputs[0]).contiguous()))
    return torch.cat(frames, dim=1), state


def expected_controls(flags, n, frames):
    """Per-packet frame_reset lists for ``frames`` real frames: the packet's own flags, padding slots 0."""
    padded = list(flags) + [0] * (frames + n)
    controls = []
    for first in range(0, frames, n):
        real = min(n, frames - first)
        controls.append([padded[first + k] if k < real else 0 for k in range(n)])
    return controls


@pytest.fixture(scope='module')
def baseline(checkpoint):
    work, path = checkpoint
    audio, _text, _calls = run_inference(work, path, 'base_n1.wav')
    return audio


@pytest.mark.parametrize('name', sorted(STREAMS))
def test_cli_n1_and_n8_replay_the_same_events(checkpoint, baseline, model, tmp_path, name):
    work, path = checkpoint
    changed, solid = STREAMS[name]
    flags = c_edge_rule(changed, solid)
    events = write_record(tmp_path / (name + '.json'), changed, solid)
    frames = CLIP // HOP + 1                                           # 126 analyser frames

    n1, text1, calls1 = run_inference(work, path, 'n1.wav', events=events)
    pushed1 = run_inference.pushed
    n8, text8, calls8 = run_inference(work, path, 'n8.wav', n=8, events=events)
    pushed8 = run_inference.pushed
    n2, _text2, calls2 = run_inference(work, path, 'n2.wav', n=2, events=events)

    # tail length: the output is cut back to the input
    assert n1.shape == n8.shape == n2.shape == (CLIP,)
    scale = np.abs(n1).max()
    assert scale > 1e-4
    assert np.abs(n8 - n1).max() < 1e-5 * scale, 'N = 8 and N = 1 differ hop-for-hop'
    assert np.abs(n2 - n1).max() < 1e-5 * scale
    assert np.abs(n8[-400:] - n1[-400:]).max() < 1e-5 * scale

    # real hop / model call counts
    assert not calls1
    assert len(calls8) == -(-frames // 8) == 16 and len(calls2) == 63
    assert '%d frames, 16 model calls of 8, %d event frames bypassed' % (frames, sum(flags)) in text8
    assert '%d frames, %d event frames bypassed' % (frames, sum(flags)) in text1

    # the control the graph received: the packet's own events, padding slots 0 (hop count 126 -> last packet 6 real)
    assert [call['events'] for call in calls8] == expected_controls(flags, 8, frames)
    assert calls8[-1]['events'][6:] == [0, 0]
    assert [call['events'] for call in calls2] == expected_controls(flags, 2, frames)

    # end-of-packet state and the next packet against the N = 1 stream fed the very same spectra
    sequential_state = calls8[0]['state_in']
    for index, call in enumerate(calls8):
        real = min(8, frames - index * 8)
        want, sequential_state = sequential_reference(
            model, call['error'], call['far'], call['events'], sequential_state)
        assert (call['enhanced'][:, :real] - want[:, :real]).abs().max().item() < 1e-5
        if real == 8:
            assert max((a - b).abs().max().item() for a, b in zip(call['state_out'], sequential_state)) < 1e-5
        # an event hop is the error spectrum (graph identity: to rounding); a normal hop is enhanced
        for slot in range(real):
            close = torch.allclose(call['enhanced'][:, slot], call['error'][:, slot], atol=1e-6, rtol=1e-5)
            assert close == bool(call['events'][slot]), 'packet %d slot %d' % (index, slot)
            # what the synthesis receives at an event frame is the error spectrum itself, at N = 8 and N = 1 alike
            if call['events'][slot]:
                frame = index * 8 + slot
                assert torch.equal(pushed8[frame], call['error'][:, slot])
                assert torch.equal(pushed1[frame], pushed8[frame])

    # events change the audio (the replay is not a no-op) unless there are none
    if sum(flags):
        assert np.abs(n1 - baseline).max() > 1e-3 * scale
    else:
        assert np.array_equal(n1, baseline)


def test_hop3_prints_the_packet_control_00100000(checkpoint, tmp_path):
    work, path = checkpoint
    changed, solid = STREAMS['hop3']
    events = write_record(tmp_path / 'hop3.json', changed, solid)
    audio1, text1, _ = run_inference(work, path, 'h1.wav', events=events)
    audio8, text8, calls = run_inference(work, path, 'h8.wav', n=8, events=events)
    assert 'packet 0 frame_reset 00100000' in text8
    assert calls[0]['events'] == [0, 0, 1, 0, 0, 0, 0, 0]
    assert ''.join(str(int(v)) for v in calls[0]['events']) == '00100000'
    assert 'alignment events: 126 hop records' in text8 and '1 event frame(s) [2]' in text8
    assert all(not any(call['events']) for call in calls[1:])
    assert np.abs(audio8 - audio1).max() < 1e-5 * np.abs(audio1).max()


def test_no_event_source_keeps_todays_behaviour(checkpoint, baseline):
    work, path = checkpoint
    again, text, _ = run_inference(work, path, 'again.wav')
    assert np.array_equal(again, baseline)
    assert 'alignment events' not in text and 'event frames' not in text
    packets, text8, calls = run_inference(work, path, 'zeros8.wav', n=8)
    assert all(call['events'] == [0] * 8 for call in calls)
    assert np.abs(packets - baseline).max() < 1e-5 * np.abs(baseline).max()
    assert 'frame_reset' not in text8
    # an all-zero record is the same as no record, byte for byte
    zeros = write_record(work / 'zero.json', [0] * HOPS, [0] * HOPS)
    replay, _text, _ = run_inference(work, path, 'zero_events.wav', events=zeros)
    assert np.array_equal(replay, baseline)


def test_real_hop_count_and_tail_when_the_length_is_a_multiple_of_the_hop(checkpoint, tmp_path):
    work, path = checkpoint
    hops = CLIP_ALIGNED // HOP                                         # 120; the analyser still emits 121 frames
    changed, solid = record(hops, changed=[hops - 1, 5])
    flags = c_edge_rule(changed, solid)
    events = write_record(tmp_path / 'aligned.json', changed, solid)
    n1, _t, _c = run_inference(work, path, 'a1.wav', events=events, wav='_aligned')
    n8, text8, calls = run_inference(work, path, 'a8.wav', n=8, events=events, wav='_aligned')
    assert n1.shape == n8.shape == (CLIP_ALIGNED,)
    assert np.abs(n8 - n1).max() < 1e-5 * np.abs(n1).max()
    # 121 frames: 15 full packets and a last packet of one real frame; the tail frame has no record -> no event
    assert len(calls) == 16 and '121 frames, 16 model calls of 8' in text8
    assert calls[-1]['events'] == [0] * 8
    assert calls[-2]['events'][7] == 1 and flags[hops - 1] == 1       # last real hop is an event in packet 14
    # one record too many or too few is refused (padding hops do not belong in a record)
    for wrong in (hops + 1, hops - 1):
        bad = write_record(tmp_path / ('bad%d.json' % wrong), *record(wrong))
        with pytest.raises(ValueError, match='hop records'):
            run_inference(work, path, 'bad.wav', n=8, events=bad, wav='_aligned')


def test_events_are_refused_in_directory_mode(checkpoint, tmp_path):
    work, path = checkpoint
    args = al_inference.build_parser().parse_args(
        [path, '--mic-dir', str(work), '--out-dir', str(tmp_path / 'o'), '--events', str(tmp_path / 'x.json')])
    with pytest.raises(ValueError, match='--mic-dir'):
        al_inference.main(args)


def test_cli_help_documents_events():
    text = al_inference.build_parser().format_help()
    assert '--events PATH' in text and 'frame_reset' in text


# ---- calib --events ---------------------------------------------------------------------------------------

def run_calib(tmp_path, checkpoint_path, n, events, name, fmt='npz', frames=1000):
    from AIAEC import _streaming_calibration as calibration
    out = tmp_path / (name + ('.npz' if fmt == 'npz' else ''))
    argv = ['calib', '--checkpoint', checkpoint_path, '--primary-dir', str(tmp_path / 'cal_err'),
            '--far-dir', str(tmp_path / 'cal_far'), '--allow-unstamped-linear-error', '--frames', str(frames),
            '--format', fmt, '--frames-per-call', str(n), '--output', str(out)]
    if events is not None:
        argv += ['--events', events]
    old, sys.argv = sys.argv, argv
    try:
        with contextlib.redirect_stdout(io.StringIO()), pytest.warns(RuntimeWarning):
            calibration.main('Align_ULCNet')
    finally:
        sys.argv = old
    return out


@pytest.fixture(scope='module')
def calib_data(checkpoint, tmp_path_factory):
    work, _path = checkpoint
    root = tmp_path_factory.mktemp('calib_events')
    (root / 'cal_err').mkdir(), (root / 'cal_far').mkdir()
    for kind in ('err', 'far'):
        audio, _ = sf.read(str(work / ('%s.wav' % kind)), dtype='float32')
        sf.write(root / ('cal_%s' % kind) / 'a.wav', audio, SR)
    return root


def test_calib_n8_writes_the_binary_control_and_n1_skips_event_hops(checkpoint, calib_data):
    _work, path = checkpoint
    root = calib_data
    changed, solid = STREAMS['mixed']                                    # events at 3, 4, 30 (solid edge), 60
    flags = c_edge_rule(changed, solid)
    events = write_record(root / 'a.events.json', changed, solid)
    frames = CLIP // HOP + 1
    packets = -(-frames // 8)                                            # 16 packets: --events keeps the 6-frame tail

    out8 = run_calib(root, path, 8, events, 'c8')
    data8 = np.load(out8)
    reset = data8['frame_reset']
    assert reset.shape == (packets, 1, 8) and reset.dtype == np.float32
    assert set(np.unique(reset)) <= {0.0, 1.0}                           # exact binary, no scaling
    padded = flags + [0] * (packets * 8 - frames)
    assert reset[:, 0].tolist() == [[float(v) for v in padded[p * 8:p * 8 + 8]] for p in range(packets)]
    report = json.load(open(str(out8).replace('.npz', '.calib.json')))
    assert report['state_precision_policy']['frame_reset'] == 'binary_control_no_ptq'
    assert 'min' not in report['inputs']['frame_reset']
    assert report['alignment_events']['event_frames_replayed'] == sum(flags)
    assert report['frames'] == packets

    out1 = run_calib(root, path, 1, events, 'c1')
    data1 = np.load(out1)
    assert 'frame_reset' not in data1.files
    skipped = [h for h in range(frames) if flags[h]]
    kept = [h for h in range(frames) if not flags[h]]
    assert data1['error_mag'].shape[0] == len(kept)                      # event hops are not invocations
    report1 = json.load(open(str(out1).replace('.npz', '.calib.json')))
    assert report1['alignment_events']['event_frames_replayed'] == len(skipped)

    # the same signals reach both graphs: every N = 8 frame equals the N = 1 capture of that hop
    for index, hop in enumerate(kept):
        packet, slot = divmod(hop, 8)
        assert np.allclose(data1['error_mag'][index, 0, 0], data8['error_mag'][packet, 0, slot], atol=1e-6)
    # the zero-completed slots of the last packet hold the front end's floor (zero spectra), real slots carry signal
    tail = data8['error_mag'][packets - 1, 0]
    assert tail[frames - (packets - 1) * 8:].max() <= 1e-5 < 1e-3 < tail[:frames - (packets - 1) * 8].min(axis=-1).max()
    # end-of-packet state: the N = 1 state entering a packet's first (non-event) hop is the N = 8 packet's input state
    for packet in range(packets):
        first = packet * 8
        if flags[first]:
            continue
        index = kept.index(first)
        for name in ('key_history', 'value_history', 'logit_history', 'h_gru'):
            assert np.allclose(data1[name][index], data8[name][packet], atol=1e-5), (name, packet)
    # an event hop restarts the N = 1 state: the capture of the hop after it sees a zero state
    after = kept.index(5)                                                # hop 4 was the event; hop 5 enters from zero
    assert not np.any(data1['key_history'][after]) and not np.any(data1['h_gru'][after])


def test_calib_without_events_still_writes_a_zero_control_and_events_need_a_matching_record(checkpoint, calib_data):
    _work, path = checkpoint
    root = calib_data
    out = run_calib(root, path, 8, None, 'czero')
    data = np.load(out)
    assert not data['frame_reset'].any()
    report = json.load(open(str(out).replace('.npz', '.calib.json')))
    assert 'alignment_events' not in report
    wrong = write_record(root / 'wrong.json', *record(HOPS - 3))
    with pytest.raises(ValueError, match='hop records'):
        run_calib(root, path, 8, wrong, 'cwrong')
    (root / 'events_dir').mkdir()
    with pytest.raises(ValueError, match='expected exactly one'):
        run_calib(root, path, 8, str(root / 'events_dir'), 'cmissing')
    write_record(root / 'events_dir' / 'a.events.json', *STREAMS['hop3'])
    out = run_calib(root, path, 8, str(root / 'events_dir'), 'cdir')
    assert np.load(out)['frame_reset'][0, 0].tolist() == [0, 0, 1, 0, 0, 0, 0, 0]


def test_calib_first_packet_control_is_00100000_in_the_bin_format(checkpoint, calib_data):
    _work, path = checkpoint
    root = calib_data
    events = write_record(root / 'bin.events.json', *STREAMS['hop3'])
    out = run_calib(root, path, 8, events, 'cbin', fmt='bin')
    manifest = json.load(open(str(out / 'manifest.json')))
    first = sorted((out / 'frame_reset').glob('*.bin'))[0]
    control = np.fromfile(first, dtype='<f4')
    assert ''.join('%d' % v for v in control) == '00100000'
    assert 'frame_reset' in json.dumps(manifest)


def test_calib_events_keep_and_pad_the_tail_packet(checkpoint, calib_data):
    """With --events the last real hops are kept at every N; padding slots never carry an event."""
    _work, path = checkpoint
    root = calib_data
    frames = CLIP // HOP + 1                                            # 126 STFT frames, the last event hop is 125
    for name, hops_set in (('lastev', [HOPS - 1]), ('padev', [HOPS - 1, 2]), ('none', [])):
        write_record(root / (name + '.events.json'), *record(HOPS, changed=hops_set))
    for n in (2, 4, 8):                                                  # 2 divides 126 (no padding), 4 and 8 pad
        packets = -(-frames // n)
        for name, hops_set in (('lastev', [HOPS - 1]), ('padev', [HOPS - 1, 2]), ('none', [])):
            data = np.load(run_calib(root, path, n, str(root / (name + '.events.json')), 'tail%s%d' % (name, n)))
            reset = data['frame_reset'][:, 0]
            assert reset.shape == (packets, n)
            expect = np.zeros(packets * n, dtype=np.float32)
            expect[hops_set] = 1.0                                       # event on the last real hop sits at its slot
            assert reset.reshape(-1).tolist() == expect.tolist()
            assert not reset.reshape(-1)[frames:].any()                  # padding slots are never events
            last = data['error_mag'][packets - 1, 0]
            real = frames - (packets - 1) * n
            assert last[:real].max(axis=-1).min() > 1e-3 and (real == n or last[real:].max() <= 1e-5)   # real present, pad = floor
            assert data['error_mag'].shape[0] == packets
    # a clip holding fewer frames than one packet still yields its (padded) packet
    # (and the event on the last hop is replayed once)
    data = np.load(run_calib(root, path, 8, str(root / 'lastev.events.json'), 'tailrep'))
    report = json.load(open(str(root / 'tailrep.calib.json')))
    assert report['alignment_events']['event_frames_replayed'] == 1 and report['frames'] == data['frame_reset'].shape[0]


def test_calib_without_events_still_drops_the_short_tail(checkpoint, calib_data):
    """No --events: whole packets only (the documented calib contract), bit for bit as before."""
    _work, path = checkpoint
    root = calib_data
    frames = CLIP // HOP + 1
    for n in (2, 8):
        plain = np.load(run_calib(root, path, n, None, 'whole%d' % n))
        assert plain['error_mag'].shape[0] == frames // n
        # an all-zero record keeps the tail but leaves every whole packet identical to the no-events capture
        zero = write_record(root / 'zero_tail.events.json', *STREAMS['none'])
        kept = np.load(run_calib(root, path, n, zero, 'zerotail%d' % n))
        assert kept['error_mag'].shape[0] == -(-frames // n)
        for name in plain.files:
            assert np.array_equal(plain[name], kept[name][:plain[name].shape[0]]), name
