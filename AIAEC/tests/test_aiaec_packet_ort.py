"""Align-ULCNet N=8 packet graph through the C stream, ONNX Runtime as the accelerator.

The C side (stream/aiaec/aiaec.c, the _freq family, built into a small binary driver by
``make -C stream/aiaec packet-driver`` with the test catalog) frames a packet of 8 hops: features, the
frame_reset control, one job, the host back end, the error spectrum for event hops. Between its preprocess and
postprocess requests this test runs the exported layout-19 graph (random weights, tiny) in ONNX Runtime and hands
the output mask back, exactly as a board sample would with its AI3 run. The packet is compared, frame by frame, with

  (a) the Python sequential N=1 stream that resets and bypasses at the event hops (the proof in
      test_align_ulcnet_frames_per_call.py), and
  (b) the C N=1 stream (ONNX Runtime running the layout-18 graph) applying the same events -- the stream
      whose event handling this packet path must equal.

Event streams: none, frame 1, frame 3, frame 8, two in one packet, adjacent, one in every packet, every frame, a
solid false -> true edge on frame 2, and a solid edge whose previous hop is the last hop of the previous packet.
It also counts accelerator calls (exactly one per packet at N=8) and checks that the state generation, which makes
a board sample reload its state, never changes for an event at N=8.
"""

import contextlib
import io
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from AIAEC.Align_ULCNet import AlignULCNet
from AIAEC.Align_ULCNet import export_onnx as ex
from AIAEC.aiaec_common import SignalGrid
from AIAEC.dataset_gen import make_linear_aec_contract

REPO = Path(__file__).resolve().parents[2]
STREAM_AIAEC = REPO / 'stream' / 'aiaec'

SR, N_FFT, HOP = 16000, 256, 128
GRID = SignalGrid(SR, N_FFT, N_FFT, HOP)
F = GRID.n_freqs
PACKETS = 3
NF = 8

# Measured on random weights (print lines of -s): C against the Python reference differs by about 1e-6 of the
# largest value (float32 rounding of two independent front/back ends), C N=8 against C N=1 by 0 (the two graphs run
# the same arithmetic in ONNX Runtime). The bounds leave a margin for another runtime and still catch a wrong frame,
# a wrong event or a swapped plane, which move values by O(1).
TOL_PYTHON = 1e-4
TOL_N1_GRAPH = 1e-4
TOL_STATE = 1e-4


class Driver:
    """The packet_driver binary: OPEN / PRE / POST over stdin / stdout."""

    def __init__(self, path):
        self.proc = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def _send(self, *chunks):
        for chunk in chunks:
            self.proc.stdin.write(chunk)
        self.proc.stdin.flush()

    def _read(self, nbytes):
        data = self.proc.stdout.read(nbytes)
        assert len(data) == nbytes, 'driver closed its output (crashed?)'
        return data

    def _int(self):
        return struct.unpack('<i', self._read(4))[0]

    def open(self, n):
        self._send(struct.pack('<ii', 1, n))
        rc, generation, slot = self._int(), self._int(), self._int()
        assert rc == 0 and slot >= 0, 'open N=%d: rc=%d' % (n, rc)
        return slot, generation

    def pre(self, slot, n, err, far, changed, solid):
        """err / far complex [n, F]. Returns (rc, action, work generation, live generation, [input tensors])."""
        planes = [np.ascontiguousarray(p, dtype='<f4').tobytes()
                  for p in (err.real, err.imag, far.real, far.imag)]
        self._send(struct.pack('<ii', 2, slot), *planes,
                   np.asarray(changed, dtype='<i4').tobytes(), np.asarray(solid, dtype='<i4').tobytes())
        rc, action, work_gen, live_gen, count = (self._int() for _ in range(5))
        tensors = []
        for _ in range(count):
            elements = self._int()
            tensors.append(np.frombuffer(self._read(4 * elements), dtype='<f4').copy())
        out_elements = self._int()
        return rc, action, work_gen, live_gen, tensors, out_elements

    def post(self, slot, n, status, mask):
        """mask: the accelerator's output [1, 2, n, F] as flat float32, or None for a bypass block."""
        flat = b'' if mask is None else np.ascontiguousarray(mask, dtype='<f4').tobytes()
        self._send(struct.pack('<iiii', 3, slot, status, 0 if mask is None else mask.size), flat)
        rc, live_gen = self._int(), self._int()
        re = np.frombuffer(self._read(4 * n * F), dtype='<f4').copy().reshape(n, F)
        im = np.frombuffer(self._read(4 * n * F), dtype='<f4').copy().reshape(n, F)
        return rc, live_gen, re + 1j * im

    def close(self):
        try:
            self._send(struct.pack('<i', 9))
        except BrokenPipeError:
            pass
        self.proc.stdin.close()
        self.proc.stdout.close()
        self.proc.wait(timeout=10)


@pytest.fixture(scope='module')
def driver_path():
    override = os.environ.get('AIAEC_PACKET_DRIVER')
    if override:
        return override
    if shutil.which('make') is None or shutil.which('cc') is None:
        pytest.skip('no make / C compiler available')
    out = subprocess.run(['make', '-s', '-C', str(STREAM_AIAEC), 'packet-driver'],
                         check=True, capture_output=True, text=True)
    path = out.stdout.strip().splitlines()[-1]
    assert os.path.isfile(path), path
    return path


@pytest.fixture
def driver(driver_path):
    drv = Driver(driver_path)
    yield drv
    drv.close()


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(1)
    return AlignULCNet(GRID, max_delay_seconds=0.5).eval()


@pytest.fixture(scope='module')
def sessions(model, tmp_path_factory):
    ort = pytest.importorskip('onnxruntime')
    work = tmp_path_factory.mktemp('al256_ort')
    linear = make_linear_aec_contract(SR, frame_size=N_FFT)
    contract = {
        'model_name': 'Align_ULCNet', 'task': AlignULCNet.task, 'sr': SR,
        'n_fft': N_FFT, 'win_len': N_FFT, 'hop_len': HOP,
        'loss_version': 'test', 'linear_aec': linear.as_dict(),
        'linear_aec_contract_hash': linear.fingerprint(),
        'ctor_max_delay_seconds': 0.5,
    }
    checkpoint = work / 'al256.pth'
    torch.save({'contract': contract, 'state_dict': model.state_dict()}, checkpoint)
    result = {}
    for n in (1, NF):
        path = work / ('al_n%d.onnx' % n)
        with contextlib.redirect_stdout(io.StringIO()):
            ex.export_graph(model, str(checkpoint), str(path), verify=False, frames_per_call=n)
        result[n] = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    return result


# ---- event streams ---------------------------------------------------------------------------------------------

def _changed(*hops):
    flags = [0] * (PACKETS * NF)
    for hop in hops:
        flags[hop] = 1
    return flags, [0] * (PACKETS * NF)


def _solid_from(first, changed_hops=()):
    flags, _ = _changed(*changed_hops)
    return flags, [1 if h >= first else 0 for h in range(PACKETS * NF)]


EVENT_STREAMS = {
    'none': _changed(),
    'frame1': _changed(0),
    'frame3': _changed(2),
    'frame8': _changed(7),
    'two_in_one_packet': _changed(2, 5),
    'adjacent': _changed(1, 2),
    'every_packet': _changed(2, 8, 15, 19),
    'all_frames': _changed(*range(NF)),
    'solid_edge_frame2': _solid_from(9),
    'solid_edge_across_packets': _solid_from(8),   # hop 7 (last of packet 0) not solid, hop 8 (first of packet 1) is
}


def edge_rule(changed, solid):
    """The one-hop stream's rule (ulcnet_stream.c apply_alignment_edge) over the whole stream."""
    edges, had, prev = [], False, 0
    for c, s in zip(changed, solid):
        edges.append(bool(c or (had and not prev and s)))
        had, prev = True, s
    return edges


def _complex(generator, frames):
    shape = (frames, F)
    return torch.complex(0.5 * torch.randn(shape, generator=generator),
                         0.5 * torch.randn(shape, generator=generator))


def _zero_state(model):
    wrapper = ex.AlignUlcnetStreamingExport(model)
    return tuple(t.numpy().copy() for t in
                 ex.dummy_inputs(wrapper.delay_depth, wrapper.n_freqs, wrapper.ta_bins)[ex.SIGNAL_INPUTS:])


def python_sequential(model, error_c, far_c, edges, state):
    """The Python N=1 stream applying reset + bypass at the edge hops (state: numpy tuple)."""
    wrapper = ex.AlignUlcnetStreamingExport(model).eval()
    state = tuple(torch.from_numpy(s) for s in state)
    enhanced = []
    for index in range(error_c.shape[0]):
        *signals, compressed = ex.stream_features(
            model, torch.view_as_real(error_c[index].reshape(1, 1, F)),
            torch.view_as_real(far_c[index].reshape(1, 1, F)))
        if edges[index]:
            state = tuple(torch.zeros_like(v) for v in state)
            enhanced.append(error_c[index].reshape(1, F))
        else:
            with torch.no_grad():
                outputs = wrapper(*signals, *state)
            mask, state = outputs[0], ex.next_state(outputs)
            enhanced.append(torch.view_as_complex(
                ex.host_back_end(model, compressed, mask).contiguous()).reshape(1, F))
    return torch.cat(enhanced, dim=0).numpy(), tuple(v.numpy() for v in state)


# ---- the C streams around ONNX Runtime --------------------------------------------------------------------------

class CStream:
    """One C _freq context (N = 1 or 8) with its ONNX Runtime accelerator and the accelerator's own state."""

    def __init__(self, driver, session, model, n):
        self.driver, self.session, self.n = driver, session, n
        self.slot, self.live_gen = driver.open(n)
        self.state = _zero_state(model)
        self.model = model
        self.calls = 0
        self.generations = [self.live_gen]
        self.names = list(ex.PACKET_INPUT_NAMES if n > 1 else ex.INPUT_NAMES)

    def run(self, error_c, far_c, changed, solid):
        """One block of n hops. Returns the enhanced spectra [n, F]."""
        rc, action, work_gen, live_gen, tensors, out_elements = self.driver.pre(
            self.slot, self.n, error_c.numpy(), far_c.numpy(), changed, solid)
        assert rc == 0, 'preprocess rc=%d' % rc
        if live_gen != self.live_gen:                      # the accelerator reloads its state on a change
            self.state = tuple(np.zeros_like(s) for s in self.state)
            self.live_gen = live_gen
            self.generations.append(live_gen)
        mask = None
        if action == 1:
            n_signal = len(tensors)
            assert n_signal == (5 if self.n > 1 else 4) and out_elements == 2 * self.n * F
            feed = {}
            for name, tensor in zip(self.names[:n_signal], tensors):
                feed[name] = tensor.reshape(1, self.n, F) if name != 'frame_reset' else tensor.reshape(1, self.n)
            if self.n == 1:
                feed = {k: v.reshape(1, 1, F) for k, v in feed.items()}
            for name, value in zip(self.names[n_signal:], self.state):
                feed[name] = value
            outputs = self.session.run(None, feed)
            self.calls += 1
            mask = outputs[0].reshape(-1)
            self.state = tuple(outputs[1:])
        rc, live_gen, enhanced = self.driver.post(self.slot, self.n, 0, mask)
        assert rc == 0, 'postprocess rc=%d' % rc
        assert live_gen == self.live_gen or self.n == 1, 'generation moved in postprocess'
        return enhanced


def _run_scenario(driver, model, sessions, name):
    changed, solid = EVENT_STREAMS[name]
    edges = edge_rule(changed, solid)
    generator = torch.Generator().manual_seed(21)
    packets = [(_complex(generator, NF), _complex(generator, NF)) for _ in range(PACKETS)]

    c8 = CStream(driver, sessions[NF], model, NF)
    c1 = CStream(driver, sessions[1], model, 1)
    py_state = _zero_state(model)
    worst = {'py': 0.0, 'n1': 0.0, 'state_py': 0.0, 'state_n1': 0.0, 'event': 0.0}
    for p, (err, far) in enumerate(packets):
        sl = slice(p * NF, (p + 1) * NF)
        got8 = c8.run(err, far, changed[sl], solid[sl])
        got1 = np.concatenate([c1.run(err[k:k + 1], far[k:k + 1], changed[p * NF + k:p * NF + k + 1],
                                      solid[p * NF + k:p * NF + k + 1]) for k in range(NF)], axis=0)
        want, py_state = python_sequential(model, err, far, edges[sl], py_state)
        scale = max(np.abs(want).max(), 1e-6)
        worst['py'] = max(worst['py'], np.abs(got8 - want).max() / scale)
        worst['n1'] = max(worst['n1'], np.abs(got8 - got1).max() / scale)
        worst['state_py'] = max(worst['state_py'], max(
            np.abs(a - b).max() for a, b in zip(c8.state, py_state)))
        worst['state_n1'] = max(worst['state_n1'], max(
            np.abs(a - b).max() for a, b in zip(c8.state, c1.state)))
        for k in range(NF):                                 # event hops are the error spectrum, bit for bit
            if edges[p * NF + k]:
                assert np.array_equal(got8[k], err[k].numpy()), 'N=8 event hop %d.%d is not the error spectrum' % (p, k)
                assert np.array_equal(got1[k], err[k].numpy()), 'N=1 event hop %d.%d is not the error spectrum' % (p, k)
            else:
                assert not np.array_equal(got8[k], err[k].numpy()), 'N=8 hop %d.%d was not enhanced' % (p, k)
    return c8, c1, edges, worst


@pytest.mark.parametrize('name', sorted(EVENT_STREAMS))
def test_packet_stream_equals_python_sequential_and_the_c_n1_stream(driver, model, sessions, name):
    c8, c1, edges, worst = _run_scenario(driver, model, sessions, name)
    print('%s: events %d, N=8 calls %d (N=1 calls %d); max rel |C N=8 - python| %.3g, |C N=8 - C N=1| %.3g, '
          'state vs python %.3g, vs N=1 %.3g'
          % (name, sum(edges), c8.calls, c1.calls, worst['py'], worst['n1'], worst['state_py'], worst['state_n1']))
    assert c8.calls == PACKETS, 'exactly one accelerator call per packet'
    assert c1.calls == PACKETS * NF - sum(edges)
    assert len(c8.generations) == 1, \
        'an event at N=8 must not change the state generation (a board sample would reload its state): %s' % c8.generations
    assert worst['py'] < TOL_PYTHON and worst['n1'] < TOL_N1_GRAPH
    assert worst['state_py'] < TOL_STATE and worst['state_n1'] < TOL_STATE


def test_the_n1_stream_does_change_the_generation_on_an_event(driver, model, sessions):
    """The contrast that makes the previous assertion meaningful: N=1 restarts the state on the host."""
    _c8, c1, edges, _worst = _run_scenario(driver, model, sessions, 'two_in_one_packet')
    assert len(c1.generations) == 1 + sum(edges)


def test_the_packet_layout_constants_equal_the_exporter():
    header = (REPO / 'lib' / 'aiaec' / 'ulcnet' / 'ulcnet_model_io.h').read_text(encoding='utf-8')
    import re
    for name, value in (('ULCNET_MODEL_IO_LAYOUT_VERSION', ex.LAYOUT_VERSION),
                        ('ULCNET_MODEL_IO_PACKET_LAYOUT_VERSION', ex.PACKET_LAYOUT_VERSION),
                        ('ULCNET_MODEL_IO_MAX_FRAMES_PER_CALL', ex.MAX_PACKET_FRAMES)):
        match = re.search(r'^#define\s+%s\s+(\d+)u?\b' % name, header, flags=re.MULTILINE)
        assert match is not None and int(match.group(1)) == value, name
