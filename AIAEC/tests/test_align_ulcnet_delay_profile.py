"""Joint delay-profile validation: small n + aligned far + small D.

Two knobs decide how much delay an Align-ULCNet deployment can absorb, and
they are sized at different times by different owners:

* ``n`` (``delay_num_filters``) sizes the AEC's matched-filter bank at AEC
  init. It decides how far the bulk far-to-mic delay search reaches
  (125/221/317/413/509 ms for n=1..5) and how much AEC pool it costs.
* ``D`` (``max_delay_frames``) sizes the model's temporal-alignment history at
  ONNX export. It decides how many past 16 ms frames the attention can reach
  and how much model state it costs.

They are NOT one delay budget in two halves. Each layer only has to satisfy
the input condition the previous one delivers, so this file drives the two
together and checks each layer's own contract:

  n in {1, 2, 5}  x  far in {raw_far, aligned_far}  x  D in {4, 8, 64}

``n`` is a RUNTIME override and deliberately not part of the ``linear_aec``
data contract: dataset generation is frozen at
``DATASET_DELAY_NUM_FILTERS`` and the recorded contract carries no delay
profile at all. That decision is itself guarded here (assertions 7-8): the
override must change the engine and leave the contract, its dict and its
fingerprint untouched, and the generation entry points must have no way to
reach another bank size.

What is asserted, and why each assertion can fail:

1. REACH IS REAL. A delay inside the bank's reliable reach locks at every n;
   a delay outside it must NOT be locked at all. The 200 ms case is inside
   n=2's 221 ms reach and outside n=1's 125 ms reach, so the pair of
   assertions goes red the moment n stops reaching lib/aec -- a dropped
   argument anywhere along the way degrades to the default of 5, which would
   let n=1 lock. ``test_mutation_pinned_bank_size_*`` proves exactly that by
   re-running the n=1 case with the harness knob removed.
2. NOT LOCKING OUT OF REACH IS AN HONEST OUTCOME for the single-path fixture.
   That out-of-range input may instead mis-lock confidently when an in-range
   early reflection exists is pinned by the C known-delay fixtures
   (test_known_delay_mislock_is_detectable in both pipeline test families);
   the Python QA side has no such fixture yet. In-range failure to acquire
   is invalid (``not_acquired_in_range``).
3. THE ALIGNED-FAR SEAM IS EXACT. The far the PBFDKF consumed is the raw far
   delayed by exactly the applied delay -- checked byte-for-byte, so a
   one-sample slip fails -- and it LEADS the echo by a bounded positive
   margin over settled locked windows.
4. FAR MODE CHANGES ONLY THE FAR. raw_far and aligned_far runs share one
   frontend and therefore one bit-identical ``linear_error``.
5. MEMORY SCALES. Model stream state grows linearly and strictly with D; the
   AEC's search ring and aggregator histogram grow strictly with n.
6. THE QA GATE CAN FAIL. ``alignment_qa`` is proved against a real mis-lock
   this repository currently has (see the pre-echo note in
   ``test_qa_gate_catches_the_known_pre_echo_mislock``), not only against
   healthy clips.
7. THE OVERRIDE DOES NOT TOUCH THE CONTRACT. Same contract object, same dict,
   same fingerprint at every n -- so no shard or checkpoint stamp can shift
   because a diagnostic ran at a different bank size.
8. GENERATION STAYS AT THE FROZEN BANK. The materialization entry point takes
   no bank size, resolves to the frozen one, and the config-driven builder
   refuses a delay-profile key rather than honouring it.

No checkpoint is needed: the frontend is the real lib/aec PBFDKF and the
model runs at random weights, because every property here is about the
timeline, the state and the pool -- never about audio quality. Quality is the
training machine's gate.

Run:
    python3 -m pytest AIAEC/tests/test_align_ulcnet_delay_profile.py
"""

import configparser
import functools

import numpy as np
import pytest
import torch

from AIAEC.Align_ULCNet import AlignULCNet
from AIAEC.Align_ULCNet.sweep_delay_depth import (
    MATCHED_REACH_MS,
    alignment_qa,
    measure_offline_bulk_delay,
    resolve_delay_num_filters,
    run_streaming_frames,
)
from AIAEC.aiaec_common import SignalGrid
from AIAEC.dataset_gen import AecGrid, stft
from AIAEC.dataset_gen.linear_aec import (
    DATASET_DELAY_NUM_FILTERS,
    LinearAecProcessor,
    linear_aec_contract_from_config,
    make_linear_aec_config,
    make_linear_aec_contract,
)
from AIAEC.dataset_gen.measure_align_residual import (
    first_lock_hop,
    measure_residual_windows,
    run_linear_aec_with_taps,
    synth_echo_scene,
)

SAMPLE_RATE = 16000
GRID = SignalGrid(SAMPLE_RATE, 512, 512, 256)
# Same fixed grid, in the analysis package's own type: 16 kHz, FFT/frame 512,
# hop 256. One hop is 16 ms and one D step of the model's attention.
AEC_GRID = AecGrid(SAMPLE_RATE, 512, 512, 256)
HOP = GRID.hop_len
SECONDS = 6.0

BANK_SIZES = (1, 2, 5)
DEPTHS = (4, 8, 64)
FAR_MODES = ("raw_far", "aligned_far")

# 80 ms sits well inside even n=1's 125 ms reach, so every bank size under
# test must lock it. 200 ms sits inside n=2's 221 ms reach and outside n=1's,
# which is what makes the n=1 no-lock assertion measure the geometry rather
# than a broken estimator.
IN_REACH_MS = 80.0
OUT_OF_REACH_MS = 200.0

# The applied delay is expected to sit AT or slightly BEFORE the true bulk
# delay: the reference ring deliberately leaves alignment headroom so the far
# leads the echo. One estimator quantum is 64 samples; 12 ms bounds the
# headroom the frontend's own self-test already enforces.
MAX_HEADROOM_MS = 12.0
QA_MAX_LAG = 16384          # +-1024 ms at 16 kHz: past n=5's 509 ms reach.
MODEL_FRAMES = 24           # enough to fill a D=8 ring and start a D=64 one.


def _delay_samples(delay_ms: float) -> int:
    return int(round(delay_ms * SAMPLE_RATE / 1000.0))


@functools.lru_cache(maxsize=None)
def _scene(delay_ms: float):
    """White-noise far plus a short echo tail delayed by ``delay_ms``."""
    return synth_echo_scene(
        int(SECONDS * SAMPLE_RATE), _delay_samples(delay_ms), seed=0
    )


@functools.lru_cache(maxsize=None)
def _contract():
    """The one frozen contract every run in this file uses, at every n.

    Built once and shared on purpose: it makes "the bank size is a runtime
    override, not a contract field" a property of the fixture rather than
    something each test has to remember.
    """
    return make_linear_aec_contract(SAMPLE_RATE)


@functools.lru_cache(maxsize=None)
def _run(bank_size: int, delay_ms: float):
    """One tapped frontend pass. Cached: the matrix reuses each (n, delay)."""
    microphone, far_end = _scene(delay_ms)
    run = run_linear_aec_with_taps(
        microphone, far_end, _contract(), delay_num_filters=bank_size
    )
    # Read back off the engine, not echoed from the argument: everything below
    # attributes its result to this bank size.
    assert run.delay_num_filters == bank_size
    return run


@functools.lru_cache(maxsize=None)
def _offline(delay_ms: float):
    """Offline bulk-delay measurement of one cached scene.

    Depends only on the raw signals, never on a run, so every bank size QA'd
    against the same scene shares one measurement.
    """
    microphone, far_end = _scene(delay_ms)
    return measure_offline_bulk_delay(
        microphone, far_end, SAMPLE_RATE, max_lag=QA_MAX_LAG
    )


def _locked_residual_ms(run):
    windows, _ = measure_residual_windows(
        run.aligned_far, run.echo_estimate, run.sample_rate,
        hop_size=run.hop_size,
        delay_samples=run.delay_samples,
        confidence=run.confidence,
    )
    return [w.lag_ms for w in windows if w.locked]


# ---------------------------------------------------------------------------
# 1-2. Matched-filter reach: locking what it can, refusing what it cannot
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bank_size", BANK_SIZES)
def test_in_reach_delay_locks_at_every_bank_size(bank_size):
    run = _run(bank_size, IN_REACH_MS)
    applied = int(run.delay_samples[-1])
    true = _delay_samples(IN_REACH_MS)
    assert first_lock_hop(run.delay_samples) >= 0, (
        f"n={bank_size} reaches {MATCHED_REACH_MS[bank_size]:.0f} ms and must "
        f"lock a {IN_REACH_MS:.0f} ms delay"
    )
    headroom_ms = (true - applied) * 1000.0 / SAMPLE_RATE
    assert 0.0 <= headroom_ms <= MAX_HEADROOM_MS, (
        f"n={bank_size}: applied {applied} vs true {true} samples "
        f"({headroom_ms:+.2f} ms); the reference must lead the echo, never "
        "trail it"
    )


@pytest.mark.parametrize("bank_size", BANK_SIZES)
def test_out_of_reach_delay_is_refused_rather_than_guessed(bank_size):
    run = _run(bank_size, OUT_OF_REACH_MS)
    applied = int(run.delay_samples[-1])
    true = _delay_samples(OUT_OF_REACH_MS)
    in_reach = OUT_OF_REACH_MS <= MATCHED_REACH_MS[bank_size]
    if in_reach:
        assert first_lock_hop(run.delay_samples) >= 0
        headroom_ms = (true - applied) * 1000.0 / SAMPLE_RATE
        assert 0.0 <= headroom_ms <= MAX_HEADROOM_MS, (
            f"n={bank_size}: applied {applied} vs true {true} "
            f"({headroom_ms:+.2f} ms)"
        )
    else:
        # The can-fail core of the whole file. A bank that no longer honours
        # n silently becomes the 509 ms default one and locks here.
        assert applied < 0, (
            f"n={bank_size} reaches only "
            f"{MATCHED_REACH_MS[bank_size]:.0f} ms, so a "
            f"{OUT_OF_REACH_MS:.0f} ms delay must stay unacquired; got "
            f"applied={applied} samples"
        )


@pytest.mark.parametrize("bank_size", BANK_SIZES)
def test_qa_reports_out_of_reach_as_not_acquired_not_as_a_mislock(bank_size):
    """Fail-open must be legible as fail-open in the sweep's own QA output."""
    microphone, far_end = _scene(OUT_OF_REACH_MS)
    qa = alignment_qa(
        _run(bank_size, OUT_OF_REACH_MS), microphone, far_end,
        max_lag=QA_MAX_LAG, offline=_offline(OUT_OF_REACH_MS),
    )
    expected = (
        "ok" if OUT_OF_REACH_MS <= MATCHED_REACH_MS[bank_size]
        else "not_acquired"
    )
    assert qa["qa_status"] == expected, qa
    # Either way the clip stays usable: an out-of-reach observation is a valid
    # observation OF the profile, not a corrupt measurement of it.
    assert qa["qa_valid"] is True
    assert qa["qa_offline_bulk_delay_ms"] == pytest.approx(
        OUT_OF_REACH_MS, abs=4.0
    )


# ---------------------------------------------------------------------------
# 3-4. The aligned-far seam
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bank_size", BANK_SIZES)
def test_aligned_far_is_the_raw_far_delayed_by_exactly_the_applied_delay(
        bank_size):
    run = _run(bank_size, IN_REACH_MS)
    applied = int(run.delay_samples[-1])
    if applied < 0:
        pytest.skip(f"n={bank_size} did not acquire; covered elsewhere")
    _microphone, far_end = _scene(IN_REACH_MS)
    # Sample a settled stretch, well past acquisition.
    start = 3 * SAMPLE_RATE
    consumed = run.aligned_far[start:start + SAMPLE_RATE]
    expected = far_end[start - applied:start - applied + SAMPLE_RATE]
    # Byte-equal, not correlated: the tap IS the ring's output, so anything
    # other than an exact match is a timeline slip.
    assert np.array_equal(consumed, expected), (
        f"n={bank_size}: the far the PBFDKF consumed is not the raw far "
        f"delayed by the applied {applied} samples "
        f"(max |diff| {float(np.max(np.abs(consumed - expected))):.3e})"
    )


@pytest.mark.parametrize("bank_size", BANK_SIZES)
def test_aligned_far_leads_the_echo_by_a_bounded_positive_margin(bank_size):
    run = _run(bank_size, IN_REACH_MS)
    if first_lock_hop(run.delay_samples) < 0:
        pytest.skip(f"n={bank_size} did not acquire; covered elsewhere")
    locked = _locked_residual_ms(run)
    assert len(locked) >= 2, (
        f"n={bank_size}: too few settled locked windows ({len(locked)}) to "
        "bound the residual"
    )
    assert min(locked) > 0.0, (
        f"n={bank_size}: residual {min(locked):.2f} ms is negative -- "
        "explaining the echo would need FUTURE far-end samples"
    )
    assert max(locked) < 10.0, (
        f"n={bank_size}: residual {max(locked):.2f} ms exceeds the 10 ms "
        "structural envelope (alignment headroom + estimator quantisation), "
        "which must stay inside the NN's one 16 ms attention hop"
    )


@pytest.mark.parametrize("bank_size", BANK_SIZES)
def test_far_mode_changes_only_the_far_stream(bank_size):
    """raw_far and aligned_far share one frontend, hence one linear_error.

    One tapped run carries BOTH far semantics -- the raw far is its input and
    the aligned far is its tap -- and the error stream exists before the
    choice between them does, so the raw/aligned A/B is attributable to the
    far semantic alone. (The one-pass wiring itself lives in
    sweep_delay_depth.main, which reads both streams off a single EngineRun.)
    """
    run = _run(bank_size, IN_REACH_MS)
    _microphone, far_end = _scene(IN_REACH_MS)
    applied = int(run.delay_samples[-1])
    if applied > 0:
        assert not np.array_equal(run.aligned_far, far_end), (
            f"n={bank_size}: a {applied}-sample delay was applied, so the "
            "consumed far cannot be identical to the raw far"
        )


# ---------------------------------------------------------------------------
# 5. Memory: model state by D, AEC search geometry by n
# ---------------------------------------------------------------------------

def _model_state_bytes(depth: int) -> int:
    torch.manual_seed(0)
    model = AlignULCNet(GRID, max_delay_frames=depth).eval()
    generator = torch.Generator().manual_seed(11)

    def spec():
        return torch.complex(
            torch.randn(1, MODEL_FRAMES, GRID.n_freqs, generator=generator),
            torch.randn(1, MODEL_FRAMES, GRID.n_freqs, generator=generator),
        )

    # State cells allocate lazily, so measure a state that has actually run
    # frames -- create_stream_state() alone reports zero at every D.
    return run_streaming_frames(model, spec(), spec()).state_bytes


def test_model_state_bytes_scale_linearly_and_strictly_with_depth():
    bytes_by_depth = {depth: _model_state_bytes(depth) for depth in DEPTHS}
    ordered = [bytes_by_depth[d] for d in sorted(DEPTHS)]
    assert ordered[0] < ordered[1] < ordered[2], bytes_by_depth
    # Derived from the measurements themselves rather than compared to a
    # transcribed constant: a per-frame cost that is not constant means some
    # D-dependent buffer is not actually shrinking with D.
    depths = sorted(DEPTHS)
    steps = [
        (bytes_by_depth[b] - bytes_by_depth[a]) / (b - a)
        for a, b in zip(depths, depths[1:])
    ]
    assert steps[0] == pytest.approx(steps[1]), (
        f"state bytes per delay frame is not constant across D: {steps} "
        f"from {bytes_by_depth}"
    )


def test_aec_search_geometry_shrinks_strictly_with_bank_size():
    """The n knob must move the AEC's own sized arrays, not just a field.

    Sized at the joint product grid (16 kHz / 512 / 256), not at lib/aec's own
    default grid, so this is the geometry the ULCNet applications actually
    carve a pool for.
    """
    # lib/aec's python package is put on sys.path by the linear_aec import
    # above; importing here keeps that dependency local to the one test that
    # needs the engine's internals.
    from aec import AEC

    sizes = {}
    for bank_size in (1, 2, 3, 4, 5):
        config = make_linear_aec_config(
            SAMPLE_RATE, delay_num_filters=bank_size
        )
        assert (config.frame_size, config.hop_size) == (512, 256)
        engine = AEC(config)
        estimator = engine.delay_est._estimator
        sizes[bank_size] = (
            int(estimator._render_ring.size),
            int(estimator._aggregator._highest_peak._histogram.size),
        )
    for smaller, larger in zip(sorted(sizes), sorted(sizes)[1:]):
        assert sizes[smaller][0] < sizes[larger][0], sizes
        assert sizes[smaller][1] < sizes[larger][1], sizes


# ---------------------------------------------------------------------------
# The joint matrix itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bank_size", BANK_SIZES)
@pytest.mark.parametrize("far_mode", FAR_MODES)
@pytest.mark.parametrize("depth", DEPTHS)
def test_joint_profile_runs_and_stays_finite(bank_size, far_mode, depth):
    """Every (n, far mode, D) cell drives the real frontend into the real model.

    Random weights: this proves the profile plumbs together and stays
    numerically well-formed at every combination, which is the part that can
    be settled without a checkpoint. Nothing here claims anything about
    audio quality.
    """
    run = _run(bank_size, IN_REACH_MS)
    error = torch.from_numpy(run.error).unsqueeze(0)
    far = torch.from_numpy(
        run.aligned_far if far_mode == "aligned_far" else _scene(IN_REACH_MS)[1]
    ).unsqueeze(0)
    error_spec = stft(error, AEC_GRID).transpose(-2, -1)[:, :MODEL_FRAMES]
    far_spec = stft(far, AEC_GRID).transpose(-2, -1)[:, :MODEL_FRAMES]

    torch.manual_seed(0)
    model = AlignULCNet(GRID, max_delay_frames=depth).eval()
    result = run_streaming_frames(model, error_spec, far_spec)

    assert result.enhanced.shape == error_spec.shape
    assert result.delay_distribution.shape == (1, MODEL_FRAMES, depth)
    assert result.state_bytes > 0
    # The attention output is a distribution over the D reachable frames.
    weights = result.delay_distribution.sum(dim=-1)
    assert torch.allclose(weights, torch.ones_like(weights), atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Instrument proofs: the gates must be able to go red
# ---------------------------------------------------------------------------

def test_qa_gate_catches_the_known_pre_echo_mislock():
    """The QA gate is proved on a real mis-lock, not only on healthy clips.

    lib/aec's matched-filter aggregator currently reports one specific delay
    per bank size exactly 1600 samples (100 ms) early -- a pre-echo
    attribution whose fix is still branch-pending. At n=2 that lands on a
    ~100 ms bulk delay, i.e. squarely inside the range a short-route product
    would run. The point here is NOT to bless that behaviour: it is that the
    sweep's per-clip QA refuses such a clip instead of averaging it into a
    delay-profile statistic.
    """
    delay_ms = 100.0
    microphone, far_end = _scene(delay_ms)
    run = _run(2, delay_ms)
    applied = int(run.delay_samples[-1])
    assert applied >= 0, (
        "this case is about a CONFIDENT wrong lock; if the estimator stopped "
        "locking here the fixture, not the gate, needs updating"
    )
    qa = alignment_qa(run, microphone, far_end, max_lag=QA_MAX_LAG,
                      offline=_offline(delay_ms))
    assert qa["qa_status"] == "mislock", qa
    assert qa["qa_valid"] is False
    # The same clip at n=5 is healthy, so the gate is discriminating between
    # profiles rather than rejecting the fixture.
    healthy = alignment_qa(
        _run(5, delay_ms), microphone, far_end, max_lag=QA_MAX_LAG,
        offline=_offline(delay_ms),
    )
    assert healthy["qa_status"] == "ok", healthy


def test_mutation_pinned_bank_size_makes_the_out_of_reach_case_lock():
    """Mutation control for the harness's own n knob.

    ``run_linear_aec_with_taps(..., delay_num_filters=n)`` is the one place
    the requested bank size turns into the engine the frontend runs. Drop it
    -- the behaviour of a harness that accepts the flag and forwards nothing,
    which is what the tool did before this arc -- and the bank silently
    reverts to the frozen 509 ms one. The 200 ms delay then comes back into
    reach, so ``test_out_of_reach_delay_is_refused_rather_than_guessed``
    would go green for the wrong reason. If THIS test fails, that assertion is
    passing on something other than the wiring.
    """
    microphone, far_end = _scene(OUT_OF_REACH_MS)

    def ignore_the_flag(mic, far, contract, delay_num_filters=None):
        return run_linear_aec_with_taps(mic, far, contract)

    mutated = ignore_the_flag(microphone, far_end, _contract(), 1)
    assert mutated.delay_num_filters == DATASET_DELAY_NUM_FILTERS, (
        "the mutation must revert the bank to the frozen size"
    )

    applied = int(mutated.delay_samples[-1])
    assert applied >= 0, (
        "an un-wired bank size must let the 200 ms delay lock again; it did "
        "not, so the n=1 no-lock assertion is not measuring the wiring"
    )
    true = _delay_samples(OUT_OF_REACH_MS)
    assert abs(true - applied) <= _delay_samples(MAX_HEADROOM_MS), (
        f"mutated run reported {applied} for a true {true}"
    )


def test_mutation_dropping_the_config_forwarding_cannot_pass_silently():
    """The second half of the wiring, and it cannot lean on the contract.

    ``make_linear_aec_config`` is where the requested bank size reaches
    ``AecConfig``. Drop it there and the engine would run a 5-filter bank
    while the caller believes it deployed n=1 -- and, because the bank size is
    deliberately absent from the contract, no contract comparison anywhere can
    notice. The guard therefore has to be a direct read of the ESTIMATOR the
    engine built, which is what ``LinearAecProcessor`` asserts at construction.
    """
    from AIAEC.dataset_gen import linear_aec as module

    original = module.make_linear_aec_config

    def swallow_bank_size(sample_rate, **kwargs):
        kwargs.pop("delay_num_filters", None)
        return original(sample_rate, **kwargs)

    contract = _contract()
    module.make_linear_aec_config = swallow_bank_size
    try:
        # The contract is byte-identical either way, so this must be refused
        # on the engine read-back alone.
        with pytest.raises(ValueError, match="matched bank"):
            module.LinearAecProcessor(contract, delay_num_filters=1)
        # The un-overridden path builds the frozen bank and stays legal, which
        # is what makes the failure above about the override and not about the
        # mutation breaking construction outright.
        assert module.LinearAecProcessor(contract).delay_num_filters == (
            DATASET_DELAY_NUM_FILTERS
        )
    finally:
        module.make_linear_aec_config = original

    # ... and the same override succeeds once the forwarding is back.
    assert LinearAecProcessor(
        contract, delay_num_filters=1
    ).delay_num_filters == 1


# ---------------------------------------------------------------------------
# 7-8. The knob is a runtime override, not a data-contract change
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bank_size", (1, 2, 3, 4, 5))
def test_bank_size_override_leaves_the_contract_and_its_stamp_untouched(
        bank_size):
    """The guard for the decision that n stays OUT of the data contract.

    A shard or checkpoint is verified by its contract fingerprint. If a
    diagnostic bank size could reach any recorded field, the same corpus would
    stamp differently depending on which profile last ran a tool over it. The
    processor must therefore carry the recorded contract through unmodified --
    same object, same dict, same hash -- at every n.
    """
    contract = _contract()
    before = contract.as_dict()
    processor = LinearAecProcessor(contract, delay_num_filters=bank_size)

    assert processor.delay_num_filters == bank_size
    assert processor.contract is contract
    assert processor.contract.as_dict() == before
    assert processor.contract.fingerprint() == contract.fingerprint()
    # Nothing named the profile got smuggled in as a field either.
    assert not [
        name for name in before
        if "delay_num_filters" in name or "delay_mode" in name
    ], before


def test_the_contract_factory_has_no_bank_size_knob():
    """The complement, one level up: the profile cannot reach a contract.

    ``make_linear_aec_contract`` is what stamps a shard and a checkpoint. It
    must reject a bank size outright rather than accept and ignore one --
    accepting it is how a caller comes to believe a differently-searched
    corpus is distinguishable by its stamp. (That an ACCEPTED override leaves
    the stamp untouched is the previous test's job, asserted there against a
    fresh dict rather than the cached object.)
    """
    with pytest.raises(TypeError):
        make_linear_aec_contract(SAMPLE_RATE, delay_num_filters=1)


def test_dataset_generation_stays_at_the_frozen_bank_size():
    """The corpus path must have no way to reach another bank size.

    Two independent halves: the materialization entry point takes no bank-size
    argument at all (so a caller cannot pass one), and the config-driven
    contract builder REFUSES a delay-profile key instead of honouring it (so a
    config file cannot smuggle one in). Both matter, because a corpus
    materialized at a smaller bank has a different linear_error and an
    identical recorded contract -- nothing downstream could tell the shards
    apart afterwards.
    """
    import inspect

    from AIAEC.dataset_gen.linear_aec import materialize_linear_error

    signature = inspect.signature(materialize_linear_error)
    assert "delay_num_filters" not in signature.parameters, signature
    assert LinearAecProcessor(_contract()).delay_num_filters == (
        DATASET_DELAY_NUM_FILTERS
    )
    assert resolve_delay_num_filters(None) == DATASET_DELAY_NUM_FILTERS

    def config_with(**options) -> configparser.ConfigParser:
        cfg = configparser.ConfigParser()
        cfg.read_dict({
            "signal": {"sr": "16000", "n_fft": "512", "win_len": "512",
                       "hop_len": "256"},
            "linear_aec": dict({"preset": "balanced",
                                "filter_length": "-1"}, **options),
        })
        return cfg

    # The same config is accepted without the key, so the rejection below is
    # about the key and not about a malformed section.
    assert linear_aec_contract_from_config(config_with()).fingerprint() == (
        _contract().fingerprint()
    )
    for key, value in (
        ("delay_num_filters", "1"),
        ("delay_mode", "fixed"),
        ("fixed_delay_samples", "800"),
    ):
        with pytest.raises(ValueError, match="not supported"):
            linear_aec_contract_from_config(config_with(**{key: value}))


def test_mutation_residual_meter_detects_an_injected_misalignment():
    """The residual assertion's instrument must be able to report a failure.

    Shift the meter's OWN view of the consumed far by a known amount (the
    engine is untouched) and require the measured residual to move by that
    amount, and to go negative once the shift exceeds the real headroom. A
    meter that always answers "small and positive" would pass the residual
    assertion above no matter what the frontend did.
    """
    run = _run(5, IN_REACH_MS)
    baseline = np.median(_locked_residual_ms(run))

    def measure(shift):
        windows, _ = measure_residual_windows(
            run.aligned_far, run.echo_estimate, run.sample_rate,
            hop_size=run.hop_size,
            delay_samples=run.delay_samples,
            confidence=run.confidence,
            far_tap_shift=shift,
        )
        return float(np.median([w.lag_ms for w in windows if w.locked]))

    injected = HOP  # exactly one hop of the product grid
    moved = measure(injected)
    assert moved - baseline == pytest.approx(
        injected * 1000.0 / SAMPLE_RATE, abs=1.0
    ), f"baseline {baseline:.2f} ms -> {moved:.2f} ms under a +{injected} shift"
    late = measure(-(int(round(baseline * SAMPLE_RATE / 1000.0)) + HOP))
    assert late < 0.0, (
        f"a reference pushed past the echo must report a negative residual, "
        f"got {late:.2f} ms"
    )
