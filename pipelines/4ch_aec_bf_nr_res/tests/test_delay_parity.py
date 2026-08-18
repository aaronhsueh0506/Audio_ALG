"""Per-hop C/Python parity for the shared four-channel reference alignment.

The C core (4aec_nr_res.c) and this directory's Python reference
(pipeline.py) publish the same three per-hop fields -- applied delay,
``solid``, ``changed`` -- and hand the four linear lanes the same aligned far
hop.  Both sides are driven here from ONE synthetic scene (deterministic
xorshift32 far noise plus a half-amplitude echo at a known bulk delay) and
compared hop by hop, including BYTE equality of the served far, so an
acquisition or ring-fill rule that drifts between the two implementations
fails here instead of surfacing as an unexplained audio difference.

The C side is ``tests/dump_delay_parity.c``, built through this directory's
Makefile; it prints one CSV row per hop.

Mutation checks for the backward-jump quarantine rows (each breaks one line
and must go red here; run 2026-08-17, one failing test each, control green
before and after):

  C core (4aec_nr_res.c)      | pipeline.py mirror
  ----------------------------|-------------------------------------------
  drop the direction test     | drop the direction test
    -> forward row red        |   -> forward row red
  drop the expiry             | drop the expiry
    -> quarantine row red     |   -> quarantine row red
  judge ANY lane              | judge ANY lane
    -> proxy row red          |   -> proxy row red
  ignore the enable           |
    -> quarantine row red     |
  hardcode the window         |
    -> quarantine row red     |

The ANY-lane pair is why ``--proxy-noise`` exists: on every other scene all
four microphones carry identical audio, so "the estimator's own lane" and
"any lane" cannot be told apart and both mutations would read green.
"""

import csv
import os
import shutil
import subprocess
from importlib import import_module

import numpy as np
import pytest

_pipeline = import_module("pipelines.4ch_aec_bf_nr_res.pipeline")
FourChannelAecConfig = _pipeline.FourChannelAecConfig
FourChannelAecPipeline = _pipeline.FourChannelAecPipeline
_SharedReferenceDelayLine = _pipeline._SharedReferenceDelayLine

_CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SAMPLE_RATE = 16000
_SEED = 0x1234567

# Non-hop-multiple FIXED delay: 320 is 2.5 hops at hop 128, so the whole-hop
# ring-fill rule (3 raw hops, then aligned) differs from any per-sample rule.
_FIXED_SCENE = dict(mode="fixed", delay=320, fft=256, hops=40)
_MATCHED_SCENE = dict(mode="matched", delay=2000, fft=512, hops=120)
# Acquires at applied delay 0 (the estimator reports early and clamps), then
# relocks onto a far larger path: the first event is exactly the acquisition a
# value-only ``changed`` comparison misses, the second a value change.
_RELOCK_SCENE = dict(mode="matched", delay=64, fft=512, hops=200,
                     shift_at=60, shift_delay=3000)
# The pre-echo mis-lock scene behind delay_backward_quarantine_enabled: the
# echo never moves, but the estimator acquires 6336 and then re-locks 1600
# samples (100 ms) EARLY and stays there. 260 hops covers the acquisition
# (hop 34), the unquarantined re-lock (hop 50), the quarantined adoption at
# the widest window swept below (hop 175) and enough afterwards to show it is
# sustained.
_QUARANTINE_SCENE = dict(mode="matched", delay=6400, fft=512, hops=260)
_QUARANTINE_CORRECT_DELAY = 6336  # the true path, on the 64-sample grid
_QUARANTINE_WRONG_DELAY = 4800    # 6400 - 1600: the pre-echo answer
# Measured, both sides: unquarantined adoption on hop 51 -- the estimator
# offers the wrong delay on hop 50 and it is applied on the next eligible hop,
# once the two-step admission has seen it twice -- and, with the quarantine on,
# adoption on that hop + window_hops for every window swept. hop 256 at 16 kHz
# makes 1.0 s exactly 62 hops.
_QUARANTINE_UNHELD_HOP = 51
_QUARANTINE_WINDOWS_S = (0.5, 1.0, 2.0)


def _window_hops(seconds, hop):
    """The library's seconds -> hops conversion, restated (not imported) so a
    silent change to it fails here instead of being absorbed."""
    return max(1, round(seconds / (hop / _SAMPLE_RATE)))


# ---------------------------------------------------------------------------
# Scene + hashing mirrors of dump_delay_parity.c
# ---------------------------------------------------------------------------

def _far_noise(count: int, seed: int = _SEED) -> np.ndarray:
    """Bit-exact mirror of the dumper's xorshift32 noise.

    Every step is exactly representable in float32 (a 24-bit integer feed and
    two power-of-two scale factors), so this is the same sequence rather than
    a close one.
    """
    state = seed
    mask = 0xFFFFFFFF
    raw = np.empty(count, dtype=np.uint32)
    for index in range(count):
        state ^= (state << 13) & mask
        state ^= state >> 17
        state ^= (state << 5) & mask
        raw[index] = state
    scaled = (raw >> 8).astype(np.float32) * np.float32(1.0 / 16777216.0)
    return (np.float32(0.25) * (scaled - np.float32(0.5))).astype(np.float32)


_PROXY_SEED = 0x89ABCDEF
# Enough uncorrelated noise on the capture-proxy microphone that its lane
# cannot reach either cancellation threshold, while the other three stay on the
# clean echo -- and small enough that the estimator (which is fed from that
# same proxy channel) still reproduces the mis-lock. Measured at this level:
# acquisition hop 59, unquarantined adoption hop 186.
_PROXY_NOISE_GAIN = 0.5
_PROXY_SCENE = dict(mode="matched", delay=6400, fft=512, hops=300)
_PROXY_ACQUIRE_HOP = 59
_PROXY_ADOPT_HOP = 186


def _fnv1a(samples: np.ndarray) -> int:
    """FNV-1a over the raw sample bytes; the dumper's ``hash_hop``."""
    digest = 1469598103934665603
    for byte in np.asarray(samples, dtype=np.float32).tobytes():
        digest = ((digest ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return digest


def _proxy_noise(count: int) -> np.ndarray:
    """The dumper's second, independent stream (``proxy_noise_next``)."""
    return _far_noise(count, _PROXY_SEED)


def _scene(delay, fft, hops, shift_at=None, shift_delay=None):
    """Return ``(hop, far_stream, mic_stream)`` for one scene.

    ``far_stream`` is what the caller hands the pipeline; ``mic_stream`` is the
    half-amplitude echo of the far ``delay`` samples earlier.  Both drop the
    dumper's pre-history pad, which exists only so the echo is valid from the
    first streamed sample.
    """
    hop = fft // 2
    pad = delay if shift_delay is None else max(delay, shift_delay)
    history = _far_noise(pad + hops * hop)
    far = history[pad:]
    mic = np.empty(hops * hop, dtype=np.float32)
    for index in range(hops):
        start = pad + index * hop
        echo_delay = (shift_delay if shift_at is not None and index >= shift_at
                      else delay)
        mic[index * hop:(index + 1) * hop] = (
            np.float32(0.5) * history[start - echo_delay:
                                      start - echo_delay + hop]
        )
    return hop, far, mic


# ---------------------------------------------------------------------------
# C side
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dumper():
    if shutil.which("make") is None or shutil.which("cc") is None:
        pytest.skip("C build tools are unavailable")
    subprocess.run(["make", "-s", "-C", _CORE_DIR, "BACKEND=ne10",
                    "dump_delay_parity"], check=True)
    located = subprocess.run(
        ["make", "-s", "-C", _CORE_DIR, "BACKEND=ne10", "print-bin-dir"],
        check=True, capture_output=True, text=True)
    return os.path.join(located.stdout.strip().splitlines()[-1],
                        "dump_delay_parity")


def _c_rows(dumper, mode, delay, fft, hops, shift_at=None, shift_delay=None,
            quarantine=False, quarantine_s=None, proxy_noise=None):
    command = [dumper, "--mode", mode, "--delay", str(delay),
               "--fft-size", str(fft), "--hops", str(hops),
               "--sample-rate", str(_SAMPLE_RATE), "--seed", hex(_SEED)]
    if shift_at is not None:
        command += ["--shift-at", str(shift_at),
                    "--shift-delay", str(shift_delay)]
    if quarantine:
        command.append("--backward-quarantine")
    if quarantine_s is not None:
        command += ["--quarantine-s", repr(float(quarantine_s))]
    if proxy_noise is not None:
        command += ["--proxy-noise", repr(float(proxy_noise))]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    header = dict(field.split("=") for field in lines[0].lstrip("# ").split())
    rows = []
    for row in csv.DictReader(lines[1:]):
        rows.append((
            int(row["hop"]), int(row["delay_samples"]), int(row["solid"]),
            int(row["changed"]), int(row["far_hash"], 16),
            int(row["aligned_hash"], 16), float.fromhex(row["aligned_first"]),
        ))
    assert len(rows) == hops
    return int(header["hop_size"]), rows


def _row(index, state_delay, solid, changed, far_hop, aligned_hop):
    return (index, int(state_delay), int(bool(solid)), int(bool(changed)),
            _fnv1a(far_hop), _fnv1a(aligned_hop), float(aligned_hop[0]))


def _assert_rows_equal(c_rows, python_rows):
    assert len(c_rows) == len(python_rows)
    for expected, actual in zip(c_rows, python_rows):
        assert actual == expected, (
            f"hop {expected[0]}: C {expected[1:]} != Python {actual[1:]}"
        )


# ---------------------------------------------------------------------------
# Python side
# ---------------------------------------------------------------------------

def _python_fixed_rows(delay, fft, hops):
    """Drive the shared delay line the way the C core's FIXED mode does.

    pipeline.py has no FIXED policy of its own -- the C core supplies the
    constant delay and publishes ``solid`` as the ring-fill state -- so the
    scene is applied to ``_SharedReferenceDelayLine`` directly and the two
    scalar fields are spelled from the hop index, which is what makes the
    served-far comparison below an independent check rather than a restatement
    of the ring's own bookkeeping.
    """
    hop, far, _mic = _scene(delay, fft, hops)
    line = _SharedReferenceDelayLine(
        int(1024.0 * _SAMPLE_RATE / 1000.0), hop)
    rows = []
    for index in range(hops):
        far_hop = far[index * hop:(index + 1) * hop]
        # FIXED publishes "the ring can serve the offset"; `changed` is
        # unreachable without an estimator.
        solid = index * hop >= delay
        aligned = line.process(far_hop, delay)
        rows.append(_row(index, delay, solid, False, far_hop, aligned))
    return hop, far, rows


def _python_matched_rows(delay, fft, hops, shift_at=None, shift_delay=None,
                         quarantine=False, quarantine_s=1.0, proxy_noise=0.0):
    hop, far, mic = _scene(delay, fft, hops, shift_at, shift_delay)
    config = FourChannelAecConfig(
        sample_rate=_SAMPLE_RATE, frame_size=fft, hop_size=hop,
        delay_backward_quarantine_enabled=quarantine,
        delay_backward_quarantine_s=quarantine_s)
    pipeline = FourChannelAecPipeline(config)
    # Mirror of the dumper's --proxy-noise: an independent stream added to the
    # capture-proxy microphone only, so "judge the estimator's own lane" and
    # "judge any lane" become distinguishable.
    if proxy_noise:
        pad = delay if shift_delay is None else max(delay, shift_delay)
        proxy_extra = (_proxy_noise(pad + hops * hop)[pad:]
                       * np.float32(proxy_noise)).astype(np.float32)
    else:
        proxy_extra = None
    rows = []
    for index in range(hops):
        far_hop = far[index * hop:(index + 1) * hop]
        mic_hop = mic[index * hop:(index + 1) * hop]
        mics = np.stack([mic_hop] * 4, axis=1)
        if proxy_extra is not None:
            mics = mics.copy()
            mics[:, config.capture_proxy_channel] += (
                proxy_extra[index * hop:(index + 1) * hop])
        pre = pipeline.process_pre_beamformer(mics, far_hop)
        rows.append(_row(index, pre.delay.delay_samples, pre.delay.solid,
                         pre.delay.changed, far_hop, pre.aligned_render))
    return hop, far, rows


def _assert_served_far_is_the_delayed_far(rows, far, hop):
    """Ground truth, independent of both implementations: every hop the ring
    can serve carries far[t - delay], every earlier hop carries the raw far."""
    for index, delay, _solid, _changed, _far_hash, aligned_hash, _first in rows:
        start = index * hop
        if start >= delay:
            want = far[start - delay:start + hop - delay]
        else:
            want = far[start:start + hop]
        assert _fnv1a(want) == aligned_hash, (
            f"hop {index}: served far is not far[t-{delay}]"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fixed_ring_fill_is_whole_hop_and_matches_c(dumper):
    scene = _FIXED_SCENE
    c_hop, c_rows = _c_rows(dumper, **scene)
    hop, far, python_rows = _python_fixed_rows(
        scene["delay"], scene["fft"], scene["hops"])
    assert c_hop == hop
    _assert_rows_equal(c_rows, python_rows)
    _assert_served_far_is_the_delayed_far(python_rows, far, hop)

    # The scene must actually contain the whole-hop switch it is here to pin:
    # the first ceil(delay/hop) hops serve the RAW far (aligned hash == far
    # hash), and the switch happens on exactly one hop boundary.
    raw_hops = -(-scene["delay"] // hop)
    assert raw_hops == 3
    assert all(row[4] == row[5] for row in c_rows[:raw_hops])
    assert all(row[4] != row[5] for row in c_rows[raw_hops:])
    assert [row[2] for row in c_rows] == [0] * raw_hops + [1] * (
        scene["hops"] - raw_hops)
    assert not any(row[3] for row in c_rows), "FIXED must never set `changed`"


def test_matched_acquisition_matches_c_per_hop(dumper):
    scene = _MATCHED_SCENE
    c_hop, c_rows = _c_rows(dumper, **scene)
    hop, far, python_rows = _python_matched_rows(
        scene["delay"], scene["fft"], scene["hops"])
    assert c_hop == hop
    _assert_rows_equal(c_rows, python_rows)
    _assert_served_far_is_the_delayed_far(python_rows, far, hop)

    changed = [row[0] for row in c_rows if row[3]]
    assert len(changed) == 1, "scene must contain exactly one acquisition"
    assert c_rows[changed[0]][1] > 0
    assert all(row[2] == 0 for row in c_rows[:changed[0]])
    assert all(row[2] == 1 for row in c_rows[changed[0]:])


class _ConfidenceDip:
    """Shim proxy that reports a not-solid estimator without disturbing the
    accepted estimate, i.e. the confidence dip the wrapper must absorb."""

    def __init__(self, inner):
        self._inner = inner

    def accumulate(self, capture, render):
        return self._inner.accumulate(capture, render)

    @property
    def estimated_delay(self):
        return self._inner.estimated_delay

    @property
    def confidence(self):
        return 0.5

    @property
    def is_solid(self):
        return False

    @property
    def _n_updates(self):
        return self._inner._n_updates


def test_published_solid_survives_a_confidence_dip():
    """``solid`` is "a usable accepted alignment generation exists", not the
    estimator's live confidence: once accepted, a dip must not retract an
    alignment the audio path is still applying (the C core's
    ``now_usable = was_usable || eligible``).

    The dip is injected rather than provoked: the shipped aggregator latches
    ``significant_candidate_found`` (lag_aggregator.py) so a live stream never
    withdraws REFINED on its own, which is precisely why publishing the raw
    confidence looks harmless until an estimator change makes it not.
    """
    scene = _RELOCK_SCENE
    hop, far, mic = _scene(scene["delay"], scene["fft"], 40)
    estimator = _pipeline.SharedMatchedDelayEstimator(
        sample_rate=_SAMPLE_RATE, hop_size=hop)
    accepted = None
    for index in range(40):
        state = estimator.accumulate(mic[index * hop:(index + 1) * hop],
                                     far[index * hop:(index + 1) * hop])
        if state.solid and accepted is None:
            accepted = state.delay_samples
            estimator._estimator = _ConfidenceDip(estimator._estimator)
        elif accepted is not None:
            assert state.solid, f"hop {index}: usability retracted on a dip"
            assert not state.changed
            assert state.delay_samples == accepted
            assert state.confidence == 0.5
    assert accepted is not None, "scene never acquired"


class _ScriptedEstimator:
    """Shim proxy that publishes a scripted ``(delay, solid)`` per hop.

    Scripted rather than provoked, for the same reason the C side drives its
    admission state machine directly (tests/test_4aec_nr_res.c): a live
    DelayAec3 re-offers a movement on every hop once it has one, so a held
    candidate is always resolved on the very next eligible hop and its TTL
    never runs out.  Every scene in this file is therefore blind to the expiry
    rule -- removing the countdown from either implementation leaves them
    agreeing hop for hop -- so it is pinned here instead.
    """

    def __init__(self, script):
        self._script = list(script)
        self._index = -1

    def accumulate(self, capture, render):
        self._index += 1

    def reset(self):
        self._index = -1

    @property
    def _current(self):
        index = min(self._index, len(self._script) - 1)
        return self._script[max(index, 0)]

    @property
    def estimated_delay(self):
        return self._current[0]

    @property
    def confidence(self):
        return 1.0 if self._current[1] else 0.5

    @property
    def is_solid(self):
        return bool(self._current[1])

    @property
    def _n_updates(self):
        return 3


def _scripted_changes(script):
    """Return the per-hop ``(changed, delay_samples)`` for one script."""
    estimator = _pipeline.SharedMatchedDelayEstimator(
        sample_rate=_SAMPLE_RATE, hop_size=128)
    estimator._estimator = _ScriptedEstimator(script)
    hop = np.zeros(128, dtype=np.float32)
    rows = []
    for _ in script:
        state = estimator.accumulate(hop, hop)
        rows.append((state.changed, state.delay_samples))
    return estimator, rows


def test_change_candidate_lives_for_three_hops_like_lib_aec():
    """The held candidate's bounded life, mirroring lib/aec Path B's
    ``pending_delay_ttl = 3``: a movement is admitted when it is offered again
    INSIDE that life, and a single reappearance after it has expired only
    starts a new candidate.  Same four cases as the C rows, and the same
    values on both sides, which is what the C/Python mirror is worth here.
    """
    ttl = _pipeline._DELAY_CHANGE_CANDIDATE_TTL
    lapse = (-1, False)          # a hop with no usable estimate
    hold = (512, True)           # the alignment in force

    # Offered, then repeated on the very next hop: admitted.
    _est, rows = _scripted_changes([hold, hold, (1024, True), (1024, True)])
    assert [row[0] for row in rows] == [True, False, False, True]
    assert rows[-1][1] == 1024

    # One hop without a usable estimate does not end it.
    _est, rows = _scripted_changes([hold, (1024, True), lapse, (1024, True)])
    assert [row[0] for row in rows] == [True, False, False, True]

    # Aged out: the reappearance is only a new first sighting, and it takes a
    # repeat inside the new life to admit it.
    _est, rows = _scripted_changes(
        [hold, (1024, True)] + [lapse] * ttl + [(1024, True), (1024, True)])
    assert [row[0] for row in rows] == (
        [True, False] + [False] * ttl + [False, True])
    assert rows[-3][1] == 512, "the alignment holds while the candidate is dead"

    # reset() clears the candidate and the life left on it.
    estimator, rows = _scripted_changes([hold, (1024, True)])
    assert estimator._pending_ttl > 0 and estimator._pending_delay == 1024
    estimator.reset()
    assert estimator._pending_ttl == 0 and estimator._pending_delay == 0


def test_matched_relock_and_zero_delay_acquisition_match_c_per_hop(dumper):
    scene = _RELOCK_SCENE
    c_hop, c_rows = _c_rows(dumper, **scene)
    hop, far, python_rows = _python_matched_rows(
        scene["delay"], scene["fft"], scene["hops"],
        scene["shift_at"], scene["shift_delay"])
    assert c_hop == hop
    _assert_rows_equal(c_rows, python_rows)
    _assert_served_far_is_the_delayed_far(python_rows, far, hop)

    changed = [row[0] for row in c_rows if row[3]]
    assert len(changed) == 2, "scene must acquire once and relock once"
    # The acquisition lands on applied delay 0 -- the case a value-only
    # `changed` comparison (estimated != accepted_delay, accepted starting at
    # 0) silently drops -- and the relock is a plain value change.
    assert c_rows[changed[0]][1] == 0
    assert c_rows[changed[1]][1] > 0
    assert all(row[2] == 1 for row in c_rows[changed[0]:]), (
        "usability must be sticky once a generation is accepted"
    )


def test_backward_quarantine_delays_the_pre_echo_relock_and_matches_c_per_hop(dumper):
    """``delay_backward_quarantine_enabled`` (default OFF) on both sides.

    The scene is the pre-echo mis-lock this mechanism exists for: a true bulk
    delay of 6400 samples, where the shared estimator acquires CORRECTLY at
    6336 and then re-locks to 4800 -- exactly 1600 samples (100 ms) EARLY --
    and sustains that wrong answer, so no consecutive-confirmation rule can
    reject it. The quarantine's evidence is the capture-proxy lane's own
    cancellation, and its bound is the configured window.

    What is asserted, and deliberately NOT asserted: adoption of the wrong
    delay is DELAYED by exactly the window, not prevented. A mis-lock is not
    cured here and never was -- that is estimator work. Sweeping the window
    is what makes "the expiry is what releases" falsifiable: the adoption hop
    tracks it linearly, on BOTH implementations.

    Parity is worth more here than on the unquarantined scenes: the C core
    reads ``p->lanes[capture_proxy_channel]`` inside ``update_shared_delay()``
    while the Python reference reads ``self._lanes[...]`` one call earlier,
    and each side carries its own countdown -- so this is the one rule whose
    two implementations do not share a shape. A per-hop match across the whole
    run is what pins them together, including the ordering: a mirror reading
    the lane AFTER it processes this hop would judge on a different ERLE and
    diverge.
    """
    scene = _QUARANTINE_SCENE
    off_hop, off_rows = _c_rows(dumper, **scene)

    # The scene must actually mis-lock with the quarantine OFF, or the rows
    # below prove nothing.
    off_changed = [row[0] for row in off_rows if row[3]]
    assert len(off_changed) == 2, "scene must acquire once and re-lock once"
    assert off_rows[off_changed[0]][1] == _QUARANTINE_CORRECT_DELAY
    assert off_changed[1] == _QUARANTINE_UNHELD_HOP
    assert off_rows[off_changed[1]][1] == _QUARANTINE_WRONG_DELAY
    assert off_rows[-1][1] == _QUARANTINE_WRONG_DELAY, "the wrong delay is sustained"

    for window_s in _QUARANTINE_WINDOWS_S:
        c_hop, c_rows = _c_rows(dumper, quarantine=True, quarantine_s=window_s,
                                **scene)
        hop, far, python_rows = _python_matched_rows(
            scene["delay"], scene["fft"], scene["hops"],
            quarantine=True, quarantine_s=window_s)
        assert c_hop == hop == off_hop
        _assert_rows_equal(c_rows, python_rows)
        _assert_served_far_is_the_delayed_far(python_rows, far, hop)

        held = _window_hops(window_s, hop)
        on_changed = [row[0] for row in c_rows if row[3]]
        assert on_changed[0] == off_changed[0], (
            "first acquisition is never quarantined"
        )
        assert len(on_changed) == 2, "the wrong delay IS adopted, at expiry"
        assert on_changed[1] == _QUARANTINE_UNHELD_HOP + held, (
            f"window {window_s}s = {held} hops: adoption must land on "
            f"{_QUARANTINE_UNHELD_HOP} + {held}"
        )
        assert c_rows[on_changed[1]][1] == _QUARANTINE_WRONG_DELAY
        # The correct delay is what the array is actually ALIGNED to for the
        # whole window -- the property a consumer sees.
        assert {row[1] for row in c_rows[on_changed[0]:on_changed[1]]} == {
            _QUARANTINE_CORRECT_DELAY
        }
        assert all(row[2] == 1 for row in c_rows[on_changed[0]:])


def test_a_forward_shared_delay_change_is_never_quarantined(dumper):
    """The direction test, on the shared estimate. When the echo moves to a
    LARGER delay -- which pre-echo mis-attribution cannot produce -- the
    quarantine is not this mechanism's business, so every published field on
    every hop is identical with it on and off. Measured: adoption on hop 442
    either way.

    This is the row that fails if the v1 predicate ("any differing estimate
    while cancelling") is reintroduced: under it the forward move was held
    too, and on the 4ch path it was held by ANY lane rather than the
    estimator's own."""
    scene = _RELOCK_SCENE | dict(hops=560, shift_at=375, shift_delay=3000,
                                 delay=64)
    _hop_off, off_rows = _c_rows(dumper, **scene)
    _hop_on, on_rows = _c_rows(dumper, quarantine=True, **scene)
    _hop, _far, python_rows = _python_matched_rows(
        scene["delay"], scene["fft"], scene["hops"],
        scene["shift_at"], scene["shift_delay"], quarantine=True)
    _assert_rows_equal(on_rows, python_rows)
    assert on_rows == off_rows, (
        "a forward move is unquarantined: every published field, every hop"
    )

    accepted = [row[0] for row in on_rows
                if row[3] and row[0] > scene["shift_at"]]
    assert accepted, "control: the core accepts the move at all"
    assert on_rows[-1][1] == 2944, "and ends on the moved delay"


def test_only_the_capture_proxy_lane_gates_the_quarantine(dumper):
    """The 4ch-specific rule, pinned where it is falsifiable.

    The shared estimator is fed from ``capture_proxy_channel``, so that lane
    is the only one whose cancellation is evidence about the estimate being
    judged. Reading ANY lane instead is what the previous version did, and it
    let one microphone's surviving old reflection veto the shared update for
    the whole array.

    Every scene above hands all four microphones identical audio, which makes
    the two rules indistinguishable -- so this one does not: the capture proxy
    carries enough uncorrelated noise that its lane cannot reach either
    cancellation threshold, while the other three stay on the clean echo. The
    correct rule therefore never engages here and the accepted delay is the
    unquarantined trajectory; an ANY-lane rule would engage on the three clean
    lanes and hold adoption for a window.
    """
    scene = _PROXY_SCENE
    kwargs = dict(proxy_noise=_PROXY_NOISE_GAIN, **scene)
    off_hop, off_rows = _c_rows(dumper, **kwargs)
    c_hop, c_rows = _c_rows(dumper, quarantine=True, **kwargs)
    hop, far, python_rows = _python_matched_rows(
        scene["delay"], scene["fft"], scene["hops"], quarantine=True,
        proxy_noise=_PROXY_NOISE_GAIN)
    assert c_hop == hop == off_hop
    _assert_rows_equal(c_rows, python_rows)
    _assert_served_far_is_the_delayed_far(python_rows, far, hop)

    # The scene must still mis-lock, or "the quarantine did not engage" would
    # be vacuous.
    off_changed = [row[0] for row in off_rows if row[3]]
    assert off_changed == [_PROXY_ACQUIRE_HOP, _PROXY_ADOPT_HOP], (
        "control: the noisy-proxy scene still acquires and then mis-locks"
    )
    assert off_rows[off_changed[0]][1] == _QUARANTINE_CORRECT_DELAY
    assert off_rows[off_changed[1]][1] == _QUARANTINE_WRONG_DELAY

    assert c_rows == off_rows, (
        "the proxy lane is not cancelling, so the quarantine does not engage "
        "-- an ANY-lane rule would hold adoption by one window here"
    )
