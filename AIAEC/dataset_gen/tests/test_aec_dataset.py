"""End-to-end tests for the AEC generator.

These render a tiny synthetic corpus through the REAL pipeline -- manifest,
generator CLI, packer, packed dataset -- rather than unit-testing the pieces in
isolation.  Every invariant checked here is one that a consumer cannot detect
being broken: a swapped stem channel trains a model that cancels the talker and
converges beautifully, a leaked speaker produces a validation curve that looks
excellent, and a sequence packed out of order looks like slow convergence.
"""

import argparse
import collections
import contextlib
import copy
import configparser
import dataclasses
import itertools
import math
import pathlib
import random
import re

import numpy as np
import pytest
import torch
import torchaudio

from AINR.dataset_gen.dataset import active_rms

from AIAEC.dataset_gen import (
    BASE_STEM_ORDER,
    PACKED_STEM_ORDER,
    STEM_ORDER,
    AecGrid,
    AecStems,
    SequenceChunkSampler,
    alpha_from_tau,
    assert_source_disjoint,
    istft,
    lane_reset_mask,
    LinearAecContract,
    LinearAecProcessor,
    make_linear_aec_config,
    make_linear_aec_contract,
    stft,
)
from AIAEC.dataset_gen.aec_dataset import (
    ACOUSTIC_TAILS,
    ACTIVITY_LABEL_DBFS,
    DELAY_STEP_EDGE_MARGIN_SEC,
    FAR_THEN_NEAR_MODES,
    IMPAIRMENTS,
    PATH_MOTION_MODES,
    STATIC_PATH,
    AecSequenceRenderer,
    SequencePlan,
    _chunk_scenario,
    _apply_far_then_near_curriculum,
    _force_overlap,
    _normalised,
    check_rate_dependent_values,
    chunk_samples_from_config,
    moving_chunks,
    path_motion_mode,
    plan_sequences,
    resolve_acoustic_tails,
    resolve_sequence_plan,
    resample_by_ratio,
    stable_seed,
)
from AIAEC.dataset_gen import aec_dataset as aec_dataset_module
from AIAEC.dataset_gen.path_drift_metrics import (
    ANALYSIS_FRAME_SEC,
    CALIBRATION_TARGETS,
    CALIBRATION_TARGET_OF_MODE,
    DEFAULT_LAGS_SEC,
    estimate_path_track,
    mixture_correlation,
    path_drift_metrics,
    position_gram,
)
from AIAEC.dataset_gen import gen_aec_dataset as gen_aec_dataset_module
from AIAEC.dataset_gen.gen_aec_dataset import build_parser, gen_aec_dataset
from AIAEC.dataset_gen.linear_aec import linear_aec_contract_from_config
from AIAEC.dataset_gen import manifest as manifest_module
from AIAEC.dataset_gen.manifest import (
    MANIFEST_VERSION,
    UNIFIED_SPLIT,
    build_manifest,
    build_unified_manifest,
    load_manifest,
    pools_for_split,
)
from AIAEC.dataset_gen.pack_aec_dataset import pack
from AIAEC.dataset_gen.packed_aec_dataset import PackedAecDataset
from AIAEC.dataset_gen.rematerialize_linear_aec import rematerialize
from aec import AEC  # noqa: E402 -- sys.path wired by linear_aec's own import above


SR = 16000
SEED = 42


# ============================================================
# A synthetic corpus
# ============================================================

def _write(path, audio, sr=SR):
    import torchaudio
    path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(path), audio.unsqueeze(0), sr,
                    encoding='PCM_F', bits_per_sample=32)


def _speechlike(n_samples, generator):
    """Bursty band-limited noise: enough structure for active_rms to be real."""
    base = torch.randn(n_samples, generator=generator)
    # Crude formant-ish shaping plus an amplitude envelope with pauses.
    smooth = torch.nn.functional.avg_pool1d(
        base.view(1, 1, -1), kernel_size=5, stride=1, padding=2).view(-1)
    t = torch.arange(n_samples, dtype=torch.float32) / SR
    envelope = (0.5 + 0.5 * torch.sin(2 * math.pi * 1.7 * t)).clamp_min(0.05)
    return (smooth * envelope * 0.2)[:n_samples]


def _rir(n_samples, rt60, generator, gain=1.0, tail=1.0, sr=SR):
    """One synthetic room position: a direct path plus a decaying tail.

    ⚠ ``tail`` is the reverberant level relative to the direct path, and it is
    load-bearing wherever a path is measured through
    ``path_drift_metrics``. Two positions in one room share their direct
    impulse and differ in their tails, so the tail is BOTH what lets a
    trajectory decorrelate the path at all and what the estimator's 64 ms
    analysis frame cannot see: a response whose energy is almost all
    reverberant has a frequency response the estimator reads as noise, and a
    STILL path then measures less correlated than a moving one. ``tail = 0.23``
    puts two positions' shared fraction near 0.25 with the direct path still
    dominant, which is what the motion fixtures below are built from; the
    default leaves the plain sources unchanged.
    """
    t = torch.arange(n_samples, dtype=torch.float32) / sr
    decay = torch.exp(-6.9078 * t / rt60)
    out = torch.randn(n_samples, generator=generator) * decay * tail
    out[0] = 1.0                      # a clear direct path
    return out * 0.5 * gain


def _example_config():
    """The shipped example config, unmodified. One place knows how to find
    and parse it; every other helper layers its own overrides on top."""
    cfg = configparser.ConfigParser()
    cfg.read(pathlib.Path(__file__).parents[1] / 'config.example.ini')
    return cfg


def _base_cfg(root):
    """config.example.ini pointed at ``root``'s sources, on a tiny but
    PBFDKF-hop-exact grid: 1.024 s = 64 hops @16 kHz/256.

    Shared by every fixture/helper in this file that builds a corpus under
    its own ``root`` -- each caller layers its own extra ``cfg.set(...)`` on
    top (e.g. ``corpus`` sets ``val_fraction``/``p_ref_dropout``).
    """
    cfg = _example_config()
    cfg.set('signal', 'sr', str(SR))
    cfg.set('paths', 'speech_dir', str(root / 'speech'))
    cfg.set('paths', 'noise_dir', str(root / 'noise'))
    cfg.set('paths', 'rir_dir', str(root / 'rir'))
    cfg.set('sequence', 'seq_sec_min', '2.048')
    cfg.set('sequence', 'seq_sec_max', '3.072')
    cfg.set('sequence', 'chunk_sec', '1.024')
    # The fast fixture's chunk is much shorter than the shipped 8/10 s grids.
    # Keep the new curriculum disabled for unrelated tests and give its bounds
    # a valid miniature geometry; focused tests below exercise it explicitly.
    cfg.set('activity', 'p_far_then_near', '0')
    cfg.set('activity', 'far_pre_roll_sec_min', '0.2')
    cfg.set('activity', 'far_pre_roll_sec_max', '0.6')
    cfg.set('activity', 'far_stop_lead_sec_min', '0.05')
    cfg.set('activity', 'far_stop_lead_sec_max', '0.1')
    cfg.set('activity', 'far_restart_gap_sec_min', '0.1')
    cfg.set('activity', 'far_restart_gap_sec_max', '0.2')
    cfg.set('rir', 'rt60_min', '0.05')
    cfg.set('rir', 'rt60_max', '2.0')
    return cfg


@pytest.fixture(scope='module')
def corpus(tmp_path_factory):
    """Sources + config + manifest, built once for the whole module."""
    root = tmp_path_factory.mktemp('aec_corpus')
    generator = torch.Generator().manual_seed(7)

    for speaker in range(6):
        for take in range(2):
            _write(root / 'speech' / f'reader_{speaker:03d}' / f'take_{take}.wav',
                   _speechlike(4 * SR, generator))
    for index in range(6):
        _write(root / 'noise' / f'noise_{index:02d}.wav',
               torch.randn(3 * SR, generator=generator) * 0.05)
    # Four RIRs per room: the shipped config draws a 3--4 position trajectory
    # for most far-capable sequences and every position has to come from the
    # same room, so a two-RIR room could not render the default corpus at all.
    for room in range(4):
        for index in range(4):
            _write(root / 'rir' / f'room_{room:02d}' / f'rir_{index}.wav',
                   _rir(int(0.35 * SR), 0.3 + 0.1 * room, generator))

    cfg = _base_cfg(root)
    cfg.set('split', 'val_fraction', '0.25')
    # Boosted so the small corpus reliably contains the load-bearing scenario.
    cfg.set('echo_modes', 'p_ref_dropout', '0.30')

    config_path = root / 'config.ini'
    with open(config_path, 'w') as handle:
        cfg.write(handle)

    manifest = build_manifest(cfg, seed=SEED, progress=False)
    return {'root': root, 'cfg': cfg, 'config_path': config_path,
            'manifest': manifest}


def _pack(corpus, input_dir, output_dir, **overrides):
    """pack() with this module's standard arguments.

    ``--config`` is the packer's one non-audio input (the frozen linear-AEC
    contract cannot be recovered from a WAV), so every call site needs it.
    """
    args = dict(config=str(corpus['config_path']), input=str(input_dir),
                output=str(output_dir), shard_clips=8, dtype='float32',
                overwrite=False)
    args.update(overrides)
    return pack(argparse.Namespace(**args))


def _render_plans(cfg, manifest, split, hours=0.012):
    """Render this split's plans in-process, yielding (plan, RenderedSequence).

    The renderer's per-chunk metadata is no longer persisted anywhere (see
    gen_aec_dataset.py), so tests about WHAT was rendered read it here, at the
    only place it still exists.
    """
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(manifest, split), corpus_seed=manifest['seed'])
    for plan in plan_sequences(cfg, hours, SEED, split):
        yield plan, renderer.render(plan)


@pytest.fixture(scope='module')
def packed(corpus, tmp_path_factory):
    """Render and pack a small corpus through the real CLI entry points."""
    output = tmp_path_factory.mktemp('aec_data')
    gen_aec_dataset(argparse.Namespace(
        config=str(corpus['config_path']), output=str(output), hours=0.012,
        workers=0, resume=False, seed=SEED, split='train',
        manifest=None, rebuild_manifest=False, wav_encoding='float32',
    ))
    gen_aec_dataset(argparse.Namespace(
        config=str(corpus['config_path']), output=str(output), hours=0.004,
        workers=0, resume=False, seed=SEED, split='val',
        manifest=None, rebuild_manifest=False, wav_encoding='float32',
    ))
    for split in ('train', 'val'):
        _pack(corpus, output / split, output / 'packed' / split)
    return {
        'output': output,
        'train': PackedAecDataset(str(output / 'packed' / 'train'), verbose=False),
        'val': PackedAecDataset(str(output / 'packed' / 'val'), verbose=False),
    }


# ============================================================
# Stem layout
# ============================================================

def test_stem_channel_order_matches_declared_list(packed):
    """WAVs stay five-channel; packed training shards project to four."""
    assert list(STEM_ORDER) == [
        'far_render', 'near_speech', 'near_target',
        'mic_postclip', 'linear_error',
    ]
    assert list(PACKED_STEM_ORDER) == [
        'far_render', 'mic_postclip', 'linear_error', 'near_target',
    ]
    for split in ('train', 'val'):
        dataset = packed[split]
        assert tuple(dataset.stems) == PACKED_STEM_ORDER
        stems, _meta = dataset[0]
        assert stems.shape[0] == len(PACKED_STEM_ORDER)
        # And the named view must reach the same channels by name.
        view = AecStems(stems, dataset.stems)
        for position, name in enumerate(PACKED_STEM_ORDER):
            assert torch.equal(view.stem(name), stems[position])


def test_named_view_rejects_a_wrong_order():
    """A shard claiming a corrupt order (dup + missing name) must fail loudly.

    A genuine PERMUTATION of STEM_ORDER is not actually invalid input --
    ``AecStems`` looks channels up by name via ``order``, so it resolves a
    reordered declaration correctly by construction. What must be rejected is
    an order that duplicates one name and drops another, which is what a
    truly corrupt shard header looks like.
    """
    with pytest.raises(ValueError):
        AecStems(torch.zeros(len(STEM_ORDER), 16),
                 (STEM_ORDER[1],) + STEM_ORDER[1:])
    with pytest.raises(ValueError):
        AecStems(torch.zeros(len(STEM_ORDER) - 1, 16))


def test_packer_preserves_the_early_target_and_drops_reverberant_near(packed):
    """The four-channel shard is an exact projection of the generated WAV."""
    observed_difference = False
    for split in ('train', 'val'):
        dataset = packed[split]
        for index in range(len(dataset)):
            view = dataset.stems_of(index)
            assert torch.isfinite(view.near_target).all()
            meta = dataset.meta(index)
            wav_path = (packed['output'] / split / 'seqs' /
                        f"{meta['sequence_id']:06d}_{meta['chunk_index']:03d}.wav")
            wav, _sr = torchaudio.load(str(wav_path))
            for name in PACKED_STEM_ORDER:
                torch.testing.assert_close(
                    view.stem(name), wav[STEM_ORDER.index(name)],
                    rtol=0.0, atol=0.0,
                )
            near_speech = wav[STEM_ORDER.index('near_speech')]
            if (float(near_speech.abs().max()) > 1e-6
                    and not torch.allclose(view.near_target, near_speech)):
                observed_difference = True
    assert observed_difference, "early/full near RIR targets were accidentally identical"


def test_linear_error_is_finite_and_dhat_is_exactly_derivable(packed):
    for split in ('train', 'val'):
        dataset = packed[split]
        for index in range(len(dataset)):
            view = dataset.stems_of(index)
            assert torch.isfinite(view.linear_error).all()
            torch.testing.assert_close(
                view.mic_postclip - view.linear_error,
                view.D_hat,
                rtol=0.0, atol=0.0,
            )


def test_linear_aec_state_is_continuous_across_future_chunk_boundaries():
    contract = make_linear_aec_contract(16000, frame_size=512)
    chunk_samples = 32768
    generator = torch.Generator().manual_seed(123)
    far = torch.randn(2 * chunk_samples, generator=generator) * 0.05
    echo = torch.zeros_like(far)
    echo[96:] = 0.7 * far[:-96]
    mic = echo.clone()

    full_error, _ = LinearAecProcessor(contract).process(mic, far)

    continuous = LinearAecProcessor(contract)
    first, _ = continuous.process(mic[:chunk_samples], far[:chunk_samples])
    second, _ = continuous.process(mic[chunk_samples:], far[chunk_samples:])
    torch.testing.assert_close(
        torch.cat([first, second]), full_error, rtol=0.0, atol=0.0,
    )

    reset_second, _ = LinearAecProcessor(contract).process(
        mic[chunk_samples:], far[chunk_samples:]
    )
    assert not torch.equal(second, reset_second)


def test_linear_aec_ch5_uses_formed_output_seam(monkeypatch):
    """ch5 must use the selected/crossfaded WOLA seam on every hop."""
    sample_rate = 16000
    contract = make_linear_aec_contract(sample_rate)
    hop = contract.hop_size

    rng = np.random.RandomState(0x11317E5)
    n_hops = 80
    mic = np.empty(n_hops * hop, dtype=np.float32)
    far = np.empty(n_hops * hop, dtype=np.float32)
    for i in range(n_hops):
        amp = 0.9 if 20 <= i < 30 else 0.02
        far[i * hop:(i + 1) * hop] = (
            amp * 0.3 * rng.uniform(-1.0, 1.0, hop)
        ).astype(np.float32)
        mic[i * hop:(i + 1) * hop] = (
            amp * rng.uniform(-1.0, 1.0, hop)
        ).astype(np.float32)

    raw_engine = AEC(make_linear_aec_config(sample_rate))
    formed_oracle = np.empty_like(mic)
    for i in range(n_hops):
        start, stop = i * hop, (i + 1) * hop
        raw_engine.process(
            mic[start:stop].copy(), far[start:stop].copy()
        )
        formed_oracle[start:stop] = raw_engine.get_formed_output()
    assert not hasattr(raw_engine, "_limiter_gain")

    original_get = AEC.get_formed_output
    calls = 0

    def counted_get_formed_output(engine):
        nonlocal calls
        calls += 1
        return original_get(engine)

    monkeypatch.setattr(AEC, "get_formed_output", counted_get_formed_output)

    processor = LinearAecProcessor(contract)
    ch5, _ = processor.process(
        torch.from_numpy(mic), torch.from_numpy(far)
    )
    ch5_np = ch5.numpy()

    np.testing.assert_array_equal(ch5_np, formed_oracle)
    assert calls == n_hops


@pytest.mark.parametrize(
    ('field', 'value'),
    (('sample_rate', 44100), ('frame_size', 1024), ('hop_size', 128)),
)
def test_linear_aec_contract_rejects_wrong_sr_frame_or_hop(field, value):
    contract = make_linear_aec_contract(16000, frame_size=512).as_dict()
    contract[field] = value
    with pytest.raises(ValueError, match='linear AEC'):
        LinearAecContract.from_dict(contract)


def test_dataset_config_rejects_mismatched_model_and_pbfdkf_grid(corpus):
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('linear_aec', 'frame_size', '1024')
    with pytest.raises(ValueError, match='frame/hop'):
        AecSequenceRenderer(
            cfg, pools_for_split(corpus['manifest'], 'train'), corpus_seed=SEED
        )


def test_packed_dataset_rejects_legacy_four_channel_shard(tmp_path):
    contract = make_linear_aec_contract(16000, frame_size=512)
    path = tmp_path / 'legacy.pt'
    torch.save({
        'stems': list(BASE_STEM_ORDER),
        'data': torch.zeros(1, len(BASE_STEM_ORDER), 256),
        'sr': 16000,
        'meta': [{
            'sequence_id': 0, 'chunk_index': 0,
            'linear_aec_contract_hash': contract.fingerprint(),
        }],
        'linear_aec': contract.as_dict(),
        'linear_aec_contract_hash': contract.fingerprint(),
        'manifest_version': MANIFEST_VERSION,
    }, path)
    with pytest.raises(ValueError, match='stem order'):
        PackedAecDataset(str(path), verbose=False)


def test_four_channel_shard_is_loadable_with_mmap(tmp_path):
    contract = make_linear_aec_contract(16000, frame_size=512)
    data = torch.randn(2, len(PACKED_STEM_ORDER), 256)
    path = tmp_path / 'shard_00000.pt'
    torch.save({
        'stems': list(PACKED_STEM_ORDER),
        'data': data,
        'sr': 16000,
        'meta': [
            {'sequence_id': 0, 'chunk_index': 0},
            {'sequence_id': 0, 'chunk_index': 1},
        ],
        'linear_aec': contract.as_dict(),
        'linear_aec_contract_hash': contract.fingerprint(),
    }, path)
    dataset = PackedAecDataset(str(path), mmap=True, verbose=False)
    torch.testing.assert_close(dataset[1][0], data[1], rtol=0.0, atol=0.0)


def test_rematerialize_upgrades_legacy_and_resumes_mixed_channel_sequence(
        packed, corpus, tmp_path):
    """A four-channel corpus is upgraded from the audio alone.

    Nothing declares the channel count, the rate or the chunk count any more,
    so this also covers the re-materializer discovering all three from the
    files themselves -- including a half-finished directory where one chunk is
    already five-channel and the rest are still legacy four. That mixture is
    recovered from the first four stems regardless of --resume: a
    half-rewritten sequence is never a resumable one, because the ledger only
    ever records sequences whose chunks all landed.
    """
    source_seqs = packed['output'] / 'train' / 'seqs'
    destination = tmp_path / 'legacy_train'
    seqs = destination / 'seqs'
    seqs.mkdir(parents=True)

    sequence_id = 0
    source_chunks = sorted(source_seqs.glob(f'{sequence_id:06d}_[0-9]*.wav'))
    assert len(source_chunks) >= 2, "need a multi-chunk sequence for this test"
    expected_errors = []

    for chunk_index, source_wav in enumerate(source_chunks):
        audio, sr = torchaudio.load(str(source_wav))
        expected_errors.append(audio[STEM_ORDER.index('linear_error')].clone())
        # Simulate interruption: the first file was already rewritten to five
        # channels, while the remaining files are still legacy four-channel.
        write_audio = audio if chunk_index == 0 else audio[:len(BASE_STEM_ORDER)]
        torchaudio.save(
            str(seqs / source_wav.name), write_audio, sr,
            encoding='PCM_F', bits_per_sample=32,
        )

    args = argparse.Namespace(
        input=str(destination), config=str(corpus['config_path']),
        resume=True, wav_encoding='auto', jobs=1,
    )
    rematerialize(args)

    actual_errors = []
    for chunk_index in range(len(source_chunks)):
        wav_path = seqs / f'{sequence_id:06d}_{chunk_index:03d}.wav'
        audio, sr = torchaudio.load(str(wav_path))
        assert sr == SR and audio.shape[0] == len(STEM_ORDER)
        actual_errors.append(audio[STEM_ORDER.index('linear_error')])
    torch.testing.assert_close(
        torch.cat(actual_errors), torch.cat(expected_errors),
        rtol=0.0, atol=0.0,
    )

    # A second --resume must rewrite nothing. The first pass recorded this
    # sequence in the contract-keyed ledger, which is what --resume reads --
    # note the FIRST pass still had to do the work, because a corpus with no
    # ledger is a corpus this contract cannot claim, whatever shape its files
    # happen to be in.
    before = [torch.load if False else torchaudio.load(
        str(seqs / f'{sequence_id:06d}_{i:03d}.wav'))[0]
        for i in range(len(source_chunks))]
    rematerialize(args)
    after = [torchaudio.load(str(seqs / f'{sequence_id:06d}_{i:03d}.wav'))[0]
             for i in range(len(source_chunks))]
    for one, two in zip(before, after):
        torch.testing.assert_close(one, two, rtol=0.0, atol=0.0)

    packed_dir = tmp_path / 'repacked'
    _pack(corpus, destination, packed_dir)
    upgraded = PackedAecDataset(str(packed_dir), verbose=False)
    assert tuple(upgraded.stems) == PACKED_STEM_ORDER
    assert upgraded.linear_aec_contract_hash == \
        linear_aec_contract_from_config(corpus['cfg']).fingerprint()


def test_stems_recombine(corpus):
    """mic_preclip == near_speech + local_noise + echo, for every clip.

    This is the corpus's central invariant.  If it fails, the stems have been
    scaled independently somewhere and no consumer can trust that the echo
    generation is what actually reached the microphone.

    ``mic_preclip``, ``echo`` and ``local_noise`` are NOT persisted (see
    STEM_ORDER's docstring in aec_features.py) -- they are computed on every
    render regardless, so this checks the invariant against the renderer's
    ``RenderedSequence.audit`` output directly rather than a packed shard.
    """
    renderer = AecSequenceRenderer(
        corpus['cfg'], pools_for_split(corpus['manifest'], 'train'),
        corpus_seed=SEED)
    checked = 0
    for sequence_id, scenario in enumerate(
            ('double_talk', 'far_only', 'near_only', 'clipping_agc'), start=2001):
        rendered = renderer.render(SequencePlan(
            sequence_id=sequence_id, n_chunks=2, scenario=scenario,
            seed=stable_seed(SEED, 'test', f'recombine-{scenario}')))
        view = AecStems(rendered.stems)
        recombined = (view.near_speech + rendered.audit['noise']
                      + rendered.audit['echo'])
        error = (rendered.audit['mic_preclip'] - recombined).abs().max().item()
        assert error < 1e-5, f"{scenario}: stems do not sum: max error {error:.2e}"
        checked += 1
    assert checked > 0


def test_postclip_differs_exactly_where_the_metadata_says(corpus):
    """`clipped` / `agc` must describe the data, not sit alongside it.

    Both directions matter.  A flag that is set when nothing happened would
    poison an ablation; a flag that is clear when the mic path WAS altered
    would make mic_postclip silently untrustworthy.

    ``mic_preclip`` is audit-only (not persisted, see STEM_ORDER's
    docstring), so this renders directly rather than reading a packed shard.
    """
    renderer = AecSequenceRenderer(
        corpus['cfg'], pools_for_split(corpus['manifest'], 'train'),
        corpus_seed=SEED)
    checked = 0
    for sequence_id, scenario in enumerate(
            ('double_talk', 'far_only', 'clipping_agc', 'near_only'), start=3001):
        rendered = renderer.render(SequencePlan(
            sequence_id=sequence_id, n_chunks=2, scenario=scenario,
            seed=stable_seed(SEED, 'test', f'postclip-{scenario}')))
        view = AecStems(rendered.stems)
        for chunk_index, meta in enumerate(rendered.chunk_meta):
            window = slice(chunk_index * rendered.chunk_samples,
                           (chunk_index + 1) * rendered.chunk_samples)
            altered = not torch.allclose(
                view.mic_postclip[window], rendered.audit['mic_preclip'][window],
                atol=1e-6)
            flagged = meta['clipped'] or meta['agc']
            assert altered == flagged, (
                f"{scenario}[{chunk_index}]: mic altered={altered} but "
                f"clipped={meta['clipped']} agc={meta['agc']}")
            if meta['sequence_scenario'] == 'clipping_agc':
                assert meta['clipped'] and meta['agc']
            checked += 1
    assert checked > 0


# ============================================================
# Reference dropout
# ============================================================

def test_ref_dropout_clips_have_a_silent_reference(corpus):
    """Every chunk LABELLED ref_dropout really has X == 0.

    ⚠ This is what the idle-loss term and the "ref == 0 implies output ~= mic"
    gate are trained on.  If a dropout-labelled chunk still carried far-end
    audio, the gate would be supervised by contradictory examples.

    ``echo`` is audit-only (not persisted, see STEM_ORDER's docstring), so
    this renders directly rather than reading a packed shard.
    """
    renderer = AecSequenceRenderer(
        corpus['cfg'], pools_for_split(corpus['manifest'], 'train'),
        corpus_seed=SEED)
    found = 0
    for sequence_id in range(4001, 4006):
        rendered = renderer.render(SequencePlan(
            sequence_id=sequence_id, n_chunks=3, scenario='ref_dropout',
            seed=stable_seed(SEED, 'test', f'dropout-silent-{sequence_id}')))
        view = AecStems(rendered.stems)
        for chunk_index, meta in enumerate(rendered.chunk_meta):
            if meta['scenario'] != 'ref_dropout':
                continue
            window = slice(chunk_index * rendered.chunk_samples,
                           (chunk_index + 1) * rendered.chunk_samples)
            assert float(view.far_render[window].abs().max()) == 0.0
            # With the default ref_dropout_echo_continues_p = 0 the far end is
            # silent end to end, so the mic is exactly S + N.
            assert float(rendered.audit['echo'][window].abs().max()) == 0.0
            found += 1
    assert found > 0, "corpus contains no ref_dropout chunks to check"


def test_far_active_no_echo_has_a_loud_reference_and_no_echo(corpus):
    """The converse of ref_dropout: X is loud end to end and D is exactly 0.

    ⚠ This is the only scenario that can express it. Everywhere else the echo
    is tied to the reference through erl_db, whose range stops at 30 dB, so
    the quietest echo the corpus could otherwise produce still sits only
    ser_db_max below the near speech. A model trained without this case has
    never seen a reference that is loud and irrelevant.

    ``echo`` is audit-only (not persisted, see STEM_ORDER's docstring), so
    this renders directly rather than reading a packed shard.
    """
    renderer = AecSequenceRenderer(
        corpus['cfg'], pools_for_split(corpus['manifest'], 'train'),
        corpus_seed=SEED)
    found = 0
    for sequence_id in range(4101, 4106):
        rendered = renderer.render(SequencePlan(
            sequence_id=sequence_id, n_chunks=3,
            scenario='far_active_no_echo',
            seed=stable_seed(SEED, 'test', f'no-echo-{sequence_id}')))
        view = AecStems(rendered.stems)
        assert float(rendered.audit['echo'].abs().max()) == 0.0
        assert float(view.far_render.abs().max()) > 0.0
        # Whole-sequence scenario: no chunk is exempt, so the label has to
        # hold everywhere rather than marking a localised event.
        for meta in rendered.chunk_meta:
            assert meta['scenario'] == 'far_active_no_echo'
        found += 1
    assert found > 0


def test_ref_dropout_sequences_keep_active_chunks_too(corpus):
    """A dropout sequence must not be labelled dropout end to end.

    Rendered directly so the scenario is forced: a ref_dropout parent is
    mostly NOT a dropout, and labelling all of it 'ref_dropout' would both fail
    the test above and train the idle term on active chunks.
    """
    renderer = AecSequenceRenderer(
        corpus['cfg'], pools_for_split(corpus['manifest'], 'train'),
        corpus_seed=SEED)
    rendered = renderer.render(SequencePlan(
        sequence_id=999, n_chunks=3, scenario='ref_dropout',
        seed=stable_seed(SEED, 'test', 'dropout')))

    labels = [meta['scenario'] for meta in rendered.chunk_meta]
    assert 'ref_dropout' in labels
    assert any(label != 'ref_dropout' for label in labels)
    assert all(meta['sequence_scenario'] == 'ref_dropout'
               for meta in rendered.chunk_meta)

    far = AecStems(rendered.stems).far_render
    for chunk_index, meta in enumerate(rendered.chunk_meta):
        window = far[chunk_index * rendered.chunk_samples:
                     (chunk_index + 1) * rendered.chunk_samples]
        if meta['scenario'] == 'ref_dropout':
            assert float(window.abs().max()) == 0.0


# ============================================================
# The split
# ============================================================

def test_manifest_split_is_source_disjoint(corpus):
    """Speaker, speech file, noise, room, RIR and device all disjoint."""
    manifest = corpus['manifest']
    assert_source_disjoint(manifest)          # raises if any axis leaks

    train = manifest['splits']['train']
    val = manifest['splits']['val']
    for axis in ('speakers', 'speech_files', 'noise_ids', 'noise_files',
                 'rooms', 'rir_files', 'devices'):
        assert set(train[axis]) & set(val[axis]) == set(), f"leak on {axis}"
        assert train[axis] and val[axis], f"{axis} empty on one side"

    # File identities are resolved once when the manifest is built.  Keeping
    # these exact maps prevents SourcePools construction from degenerating to
    # an all-files x all-ids substring search in every worker.
    for entry in (train, val):
        assert set(entry['speaker_of']) == set(entry['speech_files'])
        assert set(entry['noise_of']) == set(entry['noise_files'])
        assert set(entry['speaker_of'].values()) <= set(entry['speakers'])
        assert set(entry['noise_of'].values()) <= set(entry['noise_ids'])


def test_unified_manifest_has_every_source_in_one_pool(corpus):
    """build_unified_manifest -- the ESCAPE HATCH -- has no train/val axis."""
    manifest = build_unified_manifest(corpus['cfg'], seed=SEED, progress=False)
    assert manifest['split_mode'] == 'unified'
    assert set(manifest['splits']) == {UNIFIED_SPLIT}

    disjoint = corpus['manifest']
    pool = manifest['splits'][UNIFIED_SPLIT]
    all_disjoint_speakers = (
        set(disjoint['splits']['train']['speakers'])
        | set(disjoint['splits']['val']['speakers'])
    )
    # Same source directories as the disjoint manifest -> same total speaker
    # set, just not partitioned.
    assert set(pool['speakers']) == all_disjoint_speakers
    # load_manifest skipping assert_source_disjoint for this shape (the actual
    # contract) is exercised end to end by
    # test_gen_aec_dataset_split_all_draws_from_one_unified_pool's own
    # load_manifest() call -- not re-asserted here via a KeyError that would
    # only be testing an accident of this dict's shape.


def test_gen_aec_dataset_split_all_draws_from_one_unified_pool(corpus, tmp_path):
    """--split all: one CLI run, WAV-only output, no train/val directories."""
    output = tmp_path / 'aec_data_unified'
    gen_aec_dataset(argparse.Namespace(
        config=str(corpus['config_path']), output=str(output), hours=0.012,
        workers=0, resume=False, seed=SEED, split=UNIFIED_SPLIT,
        manifest=None, rebuild_manifest=False, wav_encoding='float32',
    ))
    assert not (output / 'manifest.json').exists()
    assert list((output / UNIFIED_SPLIT / 'seqs').glob('[0-9]*_[0-9]*.wav'))
    assert not (output / 'train').exists()
    assert not (output / 'val').exists()

    _pack(corpus, output / UNIFIED_SPLIT, output / 'packed' / UNIFIED_SPLIT)
    dataset = PackedAecDataset(str(output / 'packed' / UNIFIED_SPLIT), verbose=False)
    assert len(dataset) > 0
    assert dataset.n_sequences() > 0

def test_a_leaked_source_is_detected(corpus):
    """The disjointness check must actually be able to fail."""
    import copy
    leaky = copy.deepcopy(corpus['manifest'])
    leaky['splits']['val']['speakers'].append(
        leaky['splits']['train']['speakers'][0])
    with pytest.raises(ValueError, match='source leak'):
        assert_source_disjoint(leaky)


def test_generated_clips_only_use_their_split_sources(corpus):
    """The claim must hold in what the RENDERER draws, not only in the manifest.

    A manifest can be perfectly disjoint while the renderer reaches past it.
    Read at the renderer, since the packed corpus no longer carries source ids.
    """
    manifest = corpus['manifest']
    for split, hours in (('train', 0.012), ('val', 0.004)):
        allowed_rooms = set(manifest['splits'][split]['rooms'])
        allowed_devices = set(manifest['splits'][split]['devices'])
        allowed_speakers = set(manifest['splits'][split]['speakers'])
        rendered_any = False
        for _plan, rendered in _render_plans(corpus['cfg'], manifest, split, hours):
            for meta in rendered.chunk_meta:
                rendered_any = True
                assert meta['room_id'] in allowed_rooms
                assert meta['device_id'] in allowed_devices
                if meta['speaker_id']:
                    assert meta['speaker_id'] in allowed_speakers
        assert rendered_any, f"{split} produced no sequences to check"


def test_train_and_val_clips_share_no_room_or_device(corpus):
    def observed(split, hours, key):
        return {
            meta[key]
            for _plan, rendered in _render_plans(
                corpus['cfg'], corpus['manifest'], split, hours)
            for meta in rendered.chunk_meta
        }

    for key in ('room_id', 'device_id'):
        train = observed('train', 0.012, key)
        val = observed('val', 0.004, key)
        assert train and val
        assert train & val == set()


# ============================================================
# Sequence discipline
# ============================================================

def test_sequence_chunks_are_contiguous_and_ordered(packed):
    """Chunks of one sequence must be adjacent in the packed corpus, in order.

    ⚠ The sampler carries recurrent state across consecutive batches on the
    strength of this.  Out-of-order packing would feed a sequence backwards,
    which reads as a convergence failure rather than a data bug.
    """
    for split in ('train', 'val'):
        dataset = packed[split]
        sequence_ids = dataset.sequence_ids()
        chunk_indices = dataset.chunk_indices()

        seen = []
        position = 0
        while position < len(sequence_ids):
            sequence_id = sequence_ids[position]
            assert sequence_id not in seen, (
                f"sequence {sequence_id} appears in two separate runs")
            seen.append(sequence_id)
            expected = 0
            while (position < len(sequence_ids)
                   and sequence_ids[position] == sequence_id):
                assert chunk_indices[position] == expected
                expected += 1
                position += 1
            assert expected >= 1
        assert len(seen) == dataset.n_sequences()


def test_sampler_lanes_walk_one_sequence_in_order(packed):
    dataset = packed['train']
    n_lanes = 2
    sampler = SequenceChunkSampler.from_dataset(dataset, n_lanes, seed=SEED)
    assert len(sampler) > 0

    sequence_ids = dataset.sequence_ids()
    chunk_indices = dataset.chunk_indices()

    previous = None
    for batch in sampler:
        assert len(batch) == n_lanes
        if previous is not None:
            for lane in range(n_lanes):
                before, now = previous[lane], batch[lane]
                same_sequence = sequence_ids[before] == sequence_ids[now]
                if same_sequence:
                    # Continuing: the next chunk, never a jump.
                    assert chunk_indices[now] == chunk_indices[before] + 1
                else:
                    # Switching: the new sequence starts at chunk 0, which is
                    # exactly the reset signal lane_reset_mask reports.
                    assert chunk_indices[now] == 0
        previous = batch

    resets = lane_reset_mask([chunk_indices[i] for i in previous])
    assert resets.dtype == torch.bool and resets.numel() == n_lanes


def test_sampler_rejects_a_sequence_with_holes():
    with pytest.raises(ValueError, match='missing chunks'):
        SequenceChunkSampler([0, 0, 0], [0, 1, 3], n_lanes=1)


def test_sampler_reshuffles_lanes_per_epoch(packed):
    dataset = packed['train']
    sampler = SequenceChunkSampler.from_dataset(dataset, 2, seed=SEED)
    first = [list(batch) for batch in sampler]
    sampler.set_epoch(1)
    second = [list(batch) for batch in sampler]
    sampler.set_epoch(0)
    assert [list(batch) for batch in sampler] == first, "epoch 0 must replay"
    assert second != first, "set_epoch did not change the lane layout"


def test_plan_is_stable_across_hours(corpus):
    """Extending a corpus must not move the sequences it already had."""
    short = plan_sequences(corpus['cfg'], 0.01, SEED, 'train')
    long = plan_sequences(corpus['cfg'], 0.03, SEED, 'train')
    assert len(long) > len(short)
    assert long[:len(short)] == short


def test_layered_plan_rejects_invalid_probabilities(corpus):
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('echo_modes', 'p_ref_dropout', '0.7')
    cfg.set('echo_modes', 'p_far_active_no_echo', '0.4')
    with pytest.raises(ValueError, match=r'p_ref_dropout .* must be <= 1'):
        plan_sequences(cfg, 0.001, SEED, 'train')

    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('impairments', 'p_sro', '1.01')
    with pytest.raises(ValueError, match=r'p_sro must be in \[0, 1\]'):
        plan_sequences(cfg, 0.001, SEED, 'train')

    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('acoustic_tails', 'p_long_delay', '-0.01')
    with pytest.raises(ValueError, match=r'p_long_delay must be in \[0, 1\]'):
        plan_sequences(cfg, 0.001, SEED, 'train')

    cfg = copy.deepcopy(corpus['cfg'])
    for name in ('far_only', 'near_only', 'double_talk', 'duplex_random'):
        cfg.set('talk_modes', f'p_{name}', '0')
    with pytest.raises(ValueError, match='every weight is zero'):
        plan_sequences(cfg, 0.001, SEED, 'train')


def test_layered_plan_composes_dt_nonlinearity_and_capture(corpus):
    """The difficult intersection must be constructible, not three labels.

    The old categorical planner could draw exactly one of double_talk,
    nonlinear_spk and clipping_agc.  Forcing only the conditional combo here
    pins the new contract and its metadata together.

    ⚠ Path motion is deliberately NOT part of the bundle: it is its own axis
    on 67% of compatible echo paths at the shipped defaults, so bundling it
    here would make
    "DT while the path moves" inseparable from "DT while the loudspeaker
    distorts".
    """
    cfg = copy.deepcopy(corpus['cfg'])
    for name in ('far_only', 'near_only', 'duplex_random'):
        cfg.set('talk_modes', f'p_{name}', '0')
    cfg.set('talk_modes', 'p_double_talk', '1')
    cfg.set('echo_modes', 'p_ref_dropout', '0')
    cfg.set('echo_modes', 'p_far_active_no_echo', '0')
    _only_impairments(cfg)
    cfg.set('complex_cases', 'p_dt_stress_combo', '1')

    plan = plan_sequences(cfg, 0.001, SEED, 'train')[0]
    talk_mode, echo_mode, impairments = resolve_sequence_plan(plan)
    assert talk_mode == 'double_talk'
    assert echo_mode == 'normal'
    assert set(impairments) == {'nonlinear_spk', 'clipping_agc'}
    assert path_motion_mode(impairments) == STATIC_PATH

    rendered = AecSequenceRenderer(
        cfg, pools_for_split(corpus['manifest'], 'train'),
        corpus_seed=SEED).render(plan)
    for meta in rendered.chunk_meta:
        assert meta['talk_mode'] == 'double_talk'
        assert meta['echo_mode'] == 'normal'
        assert {'nonlinear_spk', 'clipping_agc'} <= set(meta['impairments'])
        assert not meta['echo_path_change']
        assert meta['path_motion'] == STATIC_PATH
        assert not meta['echo_path_moving']
        assert meta['clipped'] and meta['agc']
        assert meta['nonlinear'] != 'linear'


def test_layered_plan_composes_dt_acoustic_tails(corpus):
    """The difficult DT operating points must exist in one real render."""
    cfg = copy.deepcopy(corpus['cfg'])
    for name in ('far_only', 'near_only', 'duplex_random'):
        cfg.set('talk_modes', f'p_{name}', '0')
    cfg.set('talk_modes', 'p_double_talk', '1')
    cfg.set('echo_modes', 'p_ref_dropout', '0')
    cfg.set('echo_modes', 'p_far_active_no_echo', '0')
    for name in ACOUSTIC_TAILS:
        cfg.set('acoustic_tails', f'p_{name}', '0')
    cfg.set('complex_cases', 'p_dt_acoustic_combo', '1')

    plan = plan_sequences(cfg, 0.001, SEED, 'train')[0]
    assert set(resolve_acoustic_tails(plan)) == set(ACOUSTIC_TAILS)

    rendered = AecSequenceRenderer(
        cfg, pools_for_split(corpus['manifest'], 'train'),
        corpus_seed=SEED).render(plan)
    view = AecStems(rendered.stems)
    meta = rendered.chunk_meta[0]
    assert set(meta['acoustic_tails']) == set(ACOUSTIC_TAILS)
    assert int(0.120 * SR) <= meta['bulk_delay_samples'] <= int(0.300 * SR)
    assert -10.0 <= meta['erl_db'] <= 0.0
    far_dbfs = 20.0 * math.log10(active_rms(view.far_render, SR))
    # A common anti-clipping scale may move all stems down together, but never
    # makes the quiet reference louder than the sampled tail ceiling.
    assert far_dbfs <= -30.0


def test_no_echo_plan_strips_echo_path_impairments_but_keeps_capture(corpus):
    """A no-echo sequence cannot claim an RIR/SRO/nonlinearity event."""
    cfg = copy.deepcopy(corpus['cfg'])
    for name in ('far_only', 'near_only', 'duplex_random'):
        cfg.set('talk_modes', f'p_{name}', '0')
    cfg.set('talk_modes', 'p_double_talk', '1')
    cfg.set('echo_modes', 'p_ref_dropout', '0')
    cfg.set('echo_modes', 'p_far_active_no_echo', '1')
    for name in IMPAIRMENTS:
        # The three motion modes share one draw, so 1 each would exceed the
        # single unit of probability they partition.
        cfg.set('impairments', f'p_{name}',
                '0.33' if name in PATH_MOTION_MODES else '1')
    cfg.set('complex_cases', 'p_dt_stress_combo', '1')

    plan = plan_sequences(cfg, 0.001, SEED, 'train')[0]
    talk_mode, echo_mode, impairments = resolve_sequence_plan(plan)
    assert talk_mode == 'double_talk'
    assert echo_mode == 'far_active_no_echo'
    assert impairments == ('clipping_agc',)
    assert resolve_acoustic_tails(plan) == ()

    rendered = AecSequenceRenderer(
        cfg, pools_for_split(corpus['manifest'], 'train'),
        corpus_seed=SEED).render(plan)
    view = AecStems(rendered.stems)
    assert float(view.far_render.abs().max()) > 0.0
    assert float(rendered.audit['echo'].abs().max()) == 0.0
    for chunk_index, meta in enumerate(rendered.chunk_meta):
        window = slice(chunk_index * rendered.chunk_samples,
                       (chunk_index + 1) * rendered.chunk_samples)
        assert float(view.far_render[window].abs().max()) > 0.0
        assert meta['echo_mode'] == 'far_active_no_echo'
        assert meta['scenario'] == 'far_active_no_echo'


@pytest.mark.parametrize(
    ('talk_mode', 'echo_mode'),
    [('near_only', 'normal'), ('double_talk', 'far_active_no_echo')],
)
def test_acoustic_tails_reject_a_plan_without_a_real_echo_path(
        talk_mode, echo_mode):
    plan = SequencePlan(
        sequence_id=0, n_chunks=1, scenario=talk_mode, seed=7,
        talk_mode=talk_mode, echo_mode=echo_mode,
        acoustic_tails=('quiet_far',),
    )
    with pytest.raises(ValueError, match='require a real far/echo path'):
        resolve_acoustic_tails(plan)


def test_no_echo_far_reference_never_runs_out_on_a_long_sequence(corpus):
    """The hard negative must stay far-ACTIVE in every chunk it labels.

    ⚠ The module fixture writes speech files LONGER than its tiny sequences,
    so no other test here can see this: `_render_talker` draws ONE pool file
    per run and `_load_audio(loop=False)` zero-pads a file shorter than its
    run, so a single whole-sequence far run goes silent after one utterance.
    Those chunks are X == 0 and D == 0 -- signal-identical to `ref_dropout` --
    and would still carry the `far_active_no_echo` label. This test makes the
    sequence LONGER than any source file, which is the production shape
    (20-30 s sequences, single-digit-second utterances).
    """
    cfg = copy.deepcopy(corpus['cfg'])
    for name in ('far_only', 'near_only', 'duplex_random'):
        cfg.set('talk_modes', f'p_{name}', '0')
    cfg.set('talk_modes', 'p_double_talk', '1')
    cfg.set('echo_modes', 'p_ref_dropout', '0')
    cfg.set('echo_modes', 'p_far_active_no_echo', '1')
    # 8.192 s of sequence against the fixture's 4 s speech files.
    cfg.set('sequence', 'seq_sec_min', '8.192')
    cfg.set('sequence', 'seq_sec_max', '8.192')

    renderer = AecSequenceRenderer(
        cfg, pools_for_split(corpus['manifest'], 'train'), corpus_seed=SEED)
    checked = 0
    for plan in plan_sequences(cfg, 0.01, SEED, 'train')[:2]:
        rendered = renderer.render(plan)
        assert rendered.chunk_samples * len(rendered.chunk_meta) > 4 * SR, (
            "this test is only meaningful while the sequence outlasts the "
            "fixture's source files")
        view = AecStems(rendered.stems)
        assert float(rendered.audit['echo'].abs().max()) == 0.0
        for index, meta in enumerate(rendered.chunk_meta):
            window = slice(index * rendered.chunk_samples,
                           (index + 1) * rendered.chunk_samples)
            rms = float(view.far_render[window].pow(2).mean().sqrt())
            assert rms > 10.0 ** (ACTIVITY_LABEL_DBFS / 20.0), (
                f"chunk {index} reference fell silent (rms {rms:.2e}); the "
                f"class degenerated into a silent-reference interval")
            assert meta['scenario'] == 'far_active_no_echo'
            checked += 1
    assert checked >= 8


def test_layered_plan_requires_every_layer_section_and_option(corpus):
    """A missing section or option must not become a silent corpus change.

    configparser returns the fallback for an absent SECTION exactly as for an
    absent option, so a layered config without [impairments] would render a
    full, plausible, impairment-free corpus and say nothing about it.
    """
    for section in ('talk_modes', 'echo_modes', 'impairments', 'acoustic_tails',
                    'complex_cases'):
        cfg = copy.deepcopy(corpus['cfg'])
        cfg.remove_section(section)
        with pytest.raises(ValueError, match=re.escape(f'[{section}]')):
            plan_sequences(cfg, 0.001, SEED, 'train')

    required = {
        'talk_modes': tuple(f'p_{name}' for name in
                            ('far_only', 'near_only', 'double_talk',
                             'duplex_random')),
        'echo_modes': ('p_ref_dropout', 'p_far_active_no_echo'),
        'impairments': tuple(f'p_{name}' for name in
                             ('echo_path_change', 'slow_drift', 'movement',
                              'nonlinear_spk', 'clipping_agc', 'delay_jitter',
                              'delay_step', 'sro', 'codec_mismatch')),
        'acoustic_tails': (
            'p_long_delay', 'long_delay_ms_min', 'long_delay_ms_max',
            'p_quiet_far', 'quiet_far_dbfs_min', 'quiet_far_dbfs_max',
            'p_strong_echo', 'strong_echo_erl_db_min',
            'strong_echo_erl_db_max', 'quiet_far_erl_db_max',
        ),
        'complex_cases': ('p_dt_stress_combo', 'p_dt_acoustic_combo'),
    }
    for section, options in required.items():
        for option in options:
            cfg = copy.deepcopy(corpus['cfg'])
            assert cfg.remove_option(section, option)
            with pytest.raises(ValueError, match=re.escape(
                    f'[{section}] {option}')):
                plan_sequences(cfg, 0.001, SEED, 'train')

    # ⚠ A config that predates the trajectory model has no [path_motion]
    # section AND no planner probability for the modes that section drives, and
    # the first message has to name the section. Sent to add three
    # probabilities to [impairments] instead, the author fixes those, re-runs,
    # and only then learns what the config really lacks.
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.remove_section('path_motion')
    for mode in ('slow_drift', 'movement', 'delay_step'):
        assert cfg.remove_option('impairments', f'p_{mode}')
    with pytest.raises(ValueError,
                       match=re.escape('section [path_motion] is required')):
        plan_sequences(cfg, 0.001, SEED, 'train')


def test_far_active_no_echo_chunk_label_is_measured_for_legacy_plans():
    """The compatibility path must not restore a label the signal disproves."""
    common = dict(
        echo_mode='far_active_no_echo',
        legacy_scenario='far_active_no_echo', chunk_index=0,
        dropout_chunks=set(), switch_chunk=-1,
    )
    assert _chunk_scenario(
        **common, far_active=False, near_active=True) == 'near_only'
    assert _chunk_scenario(
        **common, far_active=True, near_active=True) == 'far_active_no_echo'


def test_impairments_are_drawn_independently_of_one_another(corpus):
    """Every impairment owns its own seed.

    One shared draw across the loop would make them perfectly NESTED -- the
    rarer impairment could never occur without every commoner one -- so the
    co-occurrence this whole layered planner exists to create would collapse
    to a chain. Two fair coins must produce all four combinations.
    """
    cfg = copy.deepcopy(corpus['cfg'])
    for name in ('far_only', 'near_only', 'duplex_random'):
        cfg.set('talk_modes', f'p_{name}', '0')
    cfg.set('talk_modes', 'p_double_talk', '1')
    cfg.set('echo_modes', 'p_ref_dropout', '0')
    cfg.set('echo_modes', 'p_far_active_no_echo', '0')
    cfg.set('complex_cases', 'p_dt_stress_combo', '0')
    _only_impairments(cfg, sro='0.5', nonlinear_spk='0.5')

    seen = {resolve_sequence_plan(plan)[2]
            for plan in plan_sequences(cfg, 0.5, SEED, 'train')}
    assert seen == {(), ('sro',), ('nonlinear_spk',),
                    ('nonlinear_spk', 'sro')}, sorted(seen)


def test_acoustic_tails_are_drawn_independently(corpus):
    """Tail mixtures must not collapse into one all-or-nothing profile."""
    cfg = copy.deepcopy(corpus['cfg'])
    for name in ('far_only', 'near_only', 'duplex_random'):
        cfg.set('talk_modes', f'p_{name}', '0')
    cfg.set('talk_modes', 'p_double_talk', '1')
    cfg.set('echo_modes', 'p_ref_dropout', '0')
    cfg.set('echo_modes', 'p_far_active_no_echo', '0')
    cfg.set('complex_cases', 'p_dt_acoustic_combo', '0')
    for name in ACOUSTIC_TAILS:
        cfg.set('acoustic_tails', f'p_{name}', '0.5')

    seen = {resolve_acoustic_tails(plan)
            for plan in plan_sequences(cfg, 0.5, SEED, 'train')}
    expected = {
        tuple(name for bit, name in enumerate(ACOUSTIC_TAILS)
              if mask & (1 << bit))
        for mask in range(1 << len(ACOUSTIC_TAILS))
    }
    assert seen == expected, sorted(seen)


def test_quiet_far_alone_caps_the_erl_draw(corpus):
    """A quiet reference must not also draw a 30 dB ERL (inaudible echo)."""
    cfg = copy.deepcopy(corpus['cfg'])
    for name in ('far_only', 'near_only', 'duplex_random'):
        cfg.set('talk_modes', f'p_{name}', '0')
    cfg.set('talk_modes', 'p_double_talk', '1')
    cfg.set('echo_modes', 'p_ref_dropout', '0')
    cfg.set('echo_modes', 'p_far_active_no_echo', '0')
    cfg.set('complex_cases', 'p_dt_acoustic_combo', '0')
    for name in ACOUSTIC_TAILS:
        cfg.set('acoustic_tails', f'p_{name}', '1' if name == 'quiet_far' else '0')
    cfg.set('acoustic_tails', 'quiet_far_erl_db_max', '5')
    cfg.set('levels', 'erl_db_min', '4')

    plans = plan_sequences(cfg, 0.004, SEED, 'train')
    assert all(resolve_acoustic_tails(p) == ('quiet_far',) for p in plans)
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(corpus['manifest'], 'train'), corpus_seed=SEED)
    for plan in plans[:3]:
        meta = renderer.render(plan).chunk_meta[0]
        assert 4.0 <= meta['erl_db'] <= 5.0, meta['erl_db']


@pytest.mark.parametrize(
    ('key', 'value', 'message'),
    [
        ('long_delay_ms_min', '119', 'long_delay_ms_min'),
        ('long_delay_ms_max', '470', 'worst-case long delay'),
        ('quiet_far_dbfs_max', '-29', 'quiet_far_dbfs_max'),
        ('strong_echo_erl_db_max', '1', 'strong_echo_erl_db_max'),
        ('quiet_far_erl_db_max', '31', 'quiet_far_erl_db_max'),
    ],
)
def test_acoustic_tail_ranges_are_guarded(corpus, key, value, message):
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('acoustic_tails', key, value)
    with pytest.raises(ValueError, match=message):
        plan_sequences(cfg, 0.001, SEED, 'train')


def test_forced_edge_overlap_survives_the_minimum_window(corpus):
    """A forced edge is a guarantee, so a short draw widens instead of skipping.

    The 0.2 s floor is applied before the edge branch and used to `continue`,
    which silently cost ~21% of stress-combo sequences at least one of their
    two edges while the config and README promised both unconditionally.
    Here every draw lands under the floor, so only the clamp can save them.
    """
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('activity', 'dt_overlap_p', '1')
    cfg.set('activity', 'dt_overlap_frac_min', '0.01')
    cfg.set('activity', 'dt_overlap_frac_max', '0.01')
    far_runs = [(0, SR), (2 * SR, 3 * SR), (4 * SR, 5 * SR)]

    near_runs = _force_overlap(
        far_runs, [], random.Random(7), SR, cfg, force_edges=True)

    def overlaps(run):
        return any(max(run[0], start) < min(run[1], end)
                   for start, end in near_runs)

    assert overlaps(far_runs[0]), "leading edge was dropped by the floor"
    assert overlaps(far_runs[-1]), "trailing edge was dropped by the floor"
    # A middle burst has no guarantee, so the floor still drops it.
    assert not overlaps(far_runs[1])
    assert near_runs[0][0] == 0
    assert near_runs[-1][1] == far_runs[-1][1]

    # A burst too SHORT to reach the floor at all is covered whole rather than
    # skipped -- activity_runs' own minimum run is 0.15 s, below the 0.2 s
    # floor, so this is a shape the shipped config really produces.
    short = int(SR * 0.16)
    edge_runs = [(0, short), (2 * SR, 2 * SR + short)]
    near_short = _force_overlap(
        edge_runs, [], random.Random(11), SR, cfg, force_edges=True)
    assert any(max(0, s) < min(short, e) for s, e in near_short), \
        "a sub-floor leading burst lost its guaranteed overlap"
    assert any(max(edge_runs[1][0], s) < min(edge_runs[1][1], e)
               for s, e in near_short), \
        "a sub-floor trailing burst lost its guaranteed overlap"


def test_dt_edge_overlap_covers_cold_start_and_mature_state(corpus):
    """First and last far bursts receive near speech even when p=0."""
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('activity', 'dt_overlap_p', '0')
    cfg.set('activity', 'dt_overlap_frac_min', '0.5')
    cfg.set('activity', 'dt_overlap_frac_max', '0.5')
    far_runs = [(0, SR), (2 * SR, 3 * SR), (4 * SR, 5 * SR)]
    near_runs = _force_overlap(
        far_runs, [], random.Random(7), SR, cfg, force_edges=True)

    def overlaps(run):
        return any(max(run[0], start) < min(run[1], end)
                   for start, end in near_runs)

    assert overlaps(far_runs[0])
    assert not overlaps(far_runs[1])
    assert overlaps(far_runs[-1])
    assert near_runs[0][0] == 0
    assert near_runs[-1][1] == far_runs[-1][1]


def test_render_is_deterministic_and_order_independent(corpus):
    """Sequence 5 renders identically regardless of what was rendered first."""
    pools = pools_for_split(corpus['manifest'], 'train')
    renderer = AecSequenceRenderer(corpus['cfg'], pools, corpus_seed=SEED)
    plans = plan_sequences(corpus['cfg'], 0.01, SEED, 'train')

    direct = renderer.render(plans[2])
    _other = renderer.render(plans[0])          # perturbs any leaked global RNG
    again = AecSequenceRenderer(
        corpus['cfg'], pools, corpus_seed=SEED).render(plans[2])

    assert torch.equal(direct.stems, again.stems)
    assert direct.chunk_meta == again.chunk_meta


# ============================================================
# The shared signal grid
# ============================================================

def test_stft_istft_round_trips():
    grid = AecGrid(sr=SR, n_fft=512, win_len=512, hop_len=256)
    assert grid.n_freqs == 257
    assert grid.frame_rate == pytest.approx(62.5)

    torch.manual_seed(0)
    wave = torch.randn(2, 3, SR)
    spec = stft(wave, grid)
    assert spec.shape == (2, 3, grid.n_freqs, grid.n_frames(SR))

    back = istft(spec, grid, length=SR)
    assert back.shape == wave.shape
    assert (back - wave).abs().max() < 1e-4


def test_grid_scales_to_48k_by_config_alone():
    cfg = configparser.ConfigParser()
    cfg.read_dict({'signal': {'sr': '48000', 'n_fft': '1024',
                              'win_len': '1024', 'hop_len': '512'}})
    grid = AecGrid.from_config(cfg)
    assert (grid.n_freqs, grid.frame_rate) == (513, 93.75)
    assert grid.n_frames(48000) == 94

    torch.manual_seed(0)
    wave = torch.randn(4800)
    assert (istft(stft(wave, grid), grid, length=4800) - wave).abs().max() < 1e-4


def test_renderer_produces_finite_recombinable_eight_second_48k_stems(corpus):
    """Exercise the real renderer, not only STFT helpers, on the DFN grid."""
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('signal', 'sr', '48000')
    cfg.set('signal', 'n_fft', '1024')
    cfg.set('signal', 'win_len', '1024')
    cfg.set('signal', 'hop_len', '512')
    cfg.set('sequence', 'seq_sec_min', '8.0')
    cfg.set('sequence', 'seq_sec_max', '8.0')
    cfg.set('sequence', 'chunk_sec', '8.0')
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(corpus['manifest'], 'train'), corpus_seed=SEED,
    )
    rendered = renderer.render(SequencePlan(
        sequence_id=48000, n_chunks=1, scenario='double_talk',
        seed=stable_seed(SEED, 'test', '48k-render'),
    ))
    view = AecStems(rendered.stems)
    assert rendered.stems.shape == (len(STEM_ORDER), 8 * 48000)
    assert torch.isfinite(rendered.stems).all()
    torch.testing.assert_close(
        rendered.audit['mic_preclip'],
        view.near_speech + rendered.audit['noise'] + rendered.audit['echo'],
        rtol=0.0, atol=1e-5,
    )


# ============================================================
# The documented 48 kHz recipe
# ============================================================

_RECIPE_HEADING = 'COMPLETE 48 kHz recipe'
_RECIPE_LINE = re.compile(r'^;\s+\[(?P<section>\w+)\]\s+(?P<settings>\S.*?)\s*$')


def _documented_48k_recipe():
    """The 48 kHz recipe exactly as config.example.ini's header states it.

    Read out of the file rather than restated here, so an incomplete recipe
    fails the tests below instead of only failing whoever follows it.
    Returns ``[(section, key, value), ...]`` in the order the header lists.
    """
    lines = (pathlib.Path(__file__).parents[1] / 'config.example.ini').read_text(
        encoding='utf-8').splitlines()
    start = next(i for i, line in enumerate(lines) if _RECIPE_HEADING in line)
    recipe = []
    for line in lines[start + 1:]:
        match = _RECIPE_LINE.match(line)
        if match is None:
            break
        # Commas separate keys, but a value may carry commas of its own
        # ([codec] source_sr_values); only a fragment holding '=' starts a key.
        fragments = []
        for fragment in match.group('settings').split(','):
            if '=' in fragment or not fragments:
                fragments.append(fragment)
            else:
                fragments[-1] += ',' + fragment
        for fragment in fragments:
            key, _, value = fragment.partition('=')
            recipe.append((match.group('section'), key.strip(), value.strip()))
    return recipe


def test_rate_dependent_values_left_at_the_other_rates_defaults_are_refused():
    """The two keys that degrade a corpus SILENTLY.

    A 48 kHz run that kept the 16 kHz loudspeaker fractions or codec ladder
    generates a complete, finite, plausible corpus -- with no band-limit and
    a 4-6x codec resample. Nothing downstream can tell, so the only place it
    can be caught is before generation starts.
    """
    cfg = _example_config()
    for section, key, value in _documented_48k_recipe():
        if section == 'signal':
            cfg.set(section, key, value)          # rate moved, editorials not
    with pytest.raises(ValueError, match="still carries another rate"):
        check_rate_dependent_values(cfg)

    # The full recipe is accepted.
    cfg = _example_config()
    for section, key, value in _documented_48k_recipe():
        cfg.set(section, key, value)
    check_rate_dependent_values(cfg)

    # So is the shipped 16 kHz config, untouched.
    check_rate_dependent_values(_example_config())

    # A deliberately DIFFERENT device population is not blocked -- only an
    # exact match with the other rate's shipped default is.
    cfg.set('devices', 'speaker_lp_nyquist_frac_min', '0.21')
    check_rate_dependent_values(cfg)


def test_the_documented_48k_recipe_is_complete_and_hop_exact():
    """Every claim the header makes about the recipe, checked against the file.

    ⚠ Nothing here restates a key or a duration: they all come from
    config.example.ini, which is the artefact that was wrong.
    """
    recipe = _documented_48k_recipe()
    assert [(section, key) for section, key, _ in recipe] == [
        ('signal', 'sr'), ('signal', 'n_fft'), ('signal', 'win_len'),
        ('signal', 'hop_len'), ('sequence', 'chunk_sec'),
        ('codec', 'source_sr_values'),
        ('devices', 'speaker_lp_nyquist_frac_min'),
        ('devices', 'speaker_lp_nyquist_frac_max'),
    ]
    values = {key: value for _, key, value in recipe}

    shipped = _example_config()
    shipped_chunk_sec = shipped.getfloat('sequence', 'chunk_sec')
    shipped_codec = [int(v) for v in
                     shipped.get('codec', 'source_sr_values').split(',')]

    cfg = _example_config()
    for section, key, value in recipe:
        cfg.set(section, key, value)

    # [signal] + [sequence]: the grid is one the frozen PBFDKF supports, and
    # the chunk is a whole number of its hops.
    contract = linear_aec_contract_from_config(cfg)
    assert (contract.sample_rate, contract.frame_size, contract.hop_size) == (
        48000, 1024, 512)
    hop = contract.hop_size
    assert chunk_samples_from_config(cfg, hop) % hop == 0

    # [devices]: the loudspeaker low-pass is a fraction of Nyquist, so the
    # recipe has to rescale it or the band-limit stops band-limiting. The 48
    # kHz fractions must land on the same ABSOLUTE band the 16 kHz ones do,
    # because a real driver's rolloff does not move with the sample rate.
    for bound in ('min', 'max'):
        shipped_hz = 16000 / 2 * shipped.getfloat(
            'devices', 'speaker_lp_nyquist_frac_' + bound)
        recipe_hz = 48000 / 2 * cfg.getfloat(
            'devices', 'speaker_lp_nyquist_frac_' + bound)
        assert abs(recipe_hz - shipped_hz) < 5.0, (bound, recipe_hz, shipped_hz)

    # The same chunk_sec stays exact on the 16 kHz grid, so one duration
    # serves both rates -- the header's reason for choosing it.
    both = _example_config()
    both.set('sequence', 'chunk_sec', values['chunk_sec'])
    hop_16k = linear_aec_contract_from_config(both).hop_size
    assert chunk_samples_from_config(both, hop_16k) % hop_16k == 0

    # ... and it leaves the sequence shape alone: the same whole-chunk count
    # per sequence that the 16 kHz duration gives over seq_sec_min/max.
    chunk_sec = float(values['chunk_sec'])
    seq_min = cfg.getfloat('sequence', 'seq_sec_min')
    seq_max = cfg.getfloat('sequence', 'seq_sec_max')
    assert (int(seq_min / chunk_sec), int(seq_max / chunk_sec)) == (
        int(seq_min / shipped_chunk_sec), int(seq_max / shipped_chunk_sec))

    # [codec]: source_sr_values is filtered by `< sr`, so at 48 kHz the 16 kHz
    # list would leave only ratios far harsher than any it produces at
    # 16 kHz. The recipe's list restores a mild end.
    recipe_codec = [int(v) for v in values['source_sr_values'].split(',')]
    assert min(48000 / v for v in shipped_codec if v < 48000) == 4.0
    assert max(16000 / v for v in shipped_codec if v < 16000) == 2.0
    assert min(48000 / v for v in recipe_codec if v < 48000) == 1.5


def test_the_shipped_config_is_a_copy_of_the_example():
    """`config.ini` is what the README tells an operator to copy.

    Nothing reads `config.example.ini` at run time, so an edit made to one of
    the two and not the other is invisible until a corpus is generated from the
    stale half -- and the two files are meant to be the same file.
    """
    here = pathlib.Path(__file__).parents[1]
    shipped = (here / 'config.ini').read_bytes()
    example = (here / 'config.example.ini').read_bytes()
    assert shipped == example, (
        "config.ini and config.example.ini have drifted apart; the README "
        "documents the former as a copy of the latter")


def test_the_48k_example_config_is_the_recipe_already_applied():
    """`config.example.48k.ini` must BE `config.example.ini` + the recipe.

    Shipping two configs means two places to forget, and every key the recipe
    covers degrades SILENTLY when it is missed. So the 48 kHz file is not
    trusted to have been edited correctly by hand: it is compared against the
    16 kHz file with that file's OWN documented recipe applied, key by key.
    A value changed in one file and not the other fails here, and so does a
    recipe key left at the 16 kHz default.

    ⚠ Comments are deliberately not compared. They are prose, and the
    rate-specific ones (worked hop arithmetic, the loudspeaker band table)
    legitimately differ; the VALUES are the contract.
    """
    directory = pathlib.Path(__file__).parents[1]
    rate_48k = configparser.ConfigParser()
    assert rate_48k.read(directory / 'config.example.48k.ini',
                         encoding='utf-8'), "config.example.48k.ini is missing"

    shipped = _example_config()
    assert rate_48k.sections() == shipped.sections()
    for section in shipped.sections():
        assert set(rate_48k[section]) == set(shipped[section]), section

    expected = _example_config()
    for section, key, value in _documented_48k_recipe():
        expected.set(section, key, value)
    for section in expected.sections():
        for key in expected[section]:
            assert _normalised(rate_48k[section][key]) == _normalised(
                expected[section][key]), f"[{section}] {key}"

    # Not merely consistent with the 16 kHz file -- actually a config the
    # generator accepts AT 48 kHz, on a grid the frozen PBFDKF supports and
    # with a chunk that is a whole number of its hops.
    check_rate_dependent_values(rate_48k)
    contract = linear_aec_contract_from_config(rate_48k)
    assert (contract.sample_rate, contract.frame_size, contract.hop_size) == (
        48000, 1024, 512)
    assert chunk_samples_from_config(
        rate_48k, contract.hop_size) % contract.hop_size == 0


def test_the_documented_48k_recipe_generates_and_packs_end_to_end(corpus, tmp_path):
    """Follow the recipe verbatim, from sources to a packed shard.

    ⚠ chunk_sec is NOT overridden here -- it is whatever the recipe says. The
    other 48 kHz tests pick a duration that happens to be hop-exact at both
    rates, which is exactly what let an incomplete recipe survive: the shipped
    16 kHz chunk_sec is not hop-exact at 48 kHz, so a recipe that omits it
    cannot render a single chunk.
    """
    cfg = copy.deepcopy(corpus['cfg'])
    # Stand on the shipped sequence geometry, not the tiny one the 16 kHz
    # fixtures shrink to; only [paths] stays local.
    shipped = _example_config()
    for key in ('seq_sec_min', 'seq_sec_max', 'chunk_sec'):
        cfg.set('sequence', key, shipped.get('sequence', key))
    for section, key, value in _documented_48k_recipe():
        cfg.set(section, key, value)

    config_path = tmp_path / 'config_48k.ini'
    with open(config_path, 'w') as handle:
        cfg.write(handle)

    output = tmp_path / 'data_48k'
    gen_aec_dataset(argparse.Namespace(
        config=str(config_path), output=str(output), hours=0.004, workers=0,
        resume=False, seed=SEED, split=UNIFIED_SPLIT, manifest=None,
        rebuild_manifest=False, wav_encoding='float32',
    ))
    pack(argparse.Namespace(
        config=str(config_path), input=str(output / UNIFIED_SPLIT),
        output=str(output / 'packed'), shard_clips=8, dtype='float32',
        overwrite=False,
    ))

    chunk_sec = cfg.getfloat('sequence', 'chunk_sec')
    contract = linear_aec_contract_from_config(cfg)
    chunk_samples = chunk_samples_from_config(cfg, contract.hop_size)

    packed_48k = PackedAecDataset(str(output / 'packed'), verbose=False)
    assert packed_48k.sr == 48000
    for index in range(len(packed_48k)):
        clip, _meta = packed_48k[index]
        assert clip.shape == (len(PACKED_STEM_ORDER), chunk_samples)
        assert torch.isfinite(clip).all()

    # --hours is sized for exactly one sequence, so the chunk count is the
    # recipe's own whole-chunks-per-sequence range and nothing else.
    metas = [packed_48k[index][1] for index in range(len(packed_48k))]
    assert {meta['sequence_id'] for meta in metas} == {0}
    assert len(metas) in range(
        int(cfg.getfloat('sequence', 'seq_sec_min') / chunk_sec),
        int(cfg.getfloat('sequence', 'seq_sec_max') / chunk_sec) + 1)


@pytest.mark.parametrize('sample_rate, hop, chunk_sec', [
    (48000, 512, 10.0),        # the shipped 16 kHz duration, 480000/512 = 937.5
    (16000, 256, 1.0),         # a whole second is not hop-exact at 16 kHz either
])
def test_chunk_geometry_refusal_names_the_key_the_rate_and_a_working_value(
        sample_rate, hop, chunk_sec):
    cfg = configparser.ConfigParser()
    cfg.read_dict({'signal': {'sr': str(sample_rate)},
                   'sequence': {'chunk_sec': str(chunk_sec)}})
    with pytest.raises(ValueError) as excinfo:
        chunk_samples_from_config(cfg, hop)
    message = str(excinfo.value)
    assert '[sequence] chunk_sec' in message
    assert str(sample_rate) in message
    assert f'hop={hop}' in message

    # The value it offers has to be one that actually renders. Anchored on the
    # 'e.g.' lead-in rather than the end of the message, which is free to grow
    # further advice after the suggestion.
    suggested = re.search(
        r'e\.g\. chunk_sec = ([0-9]+(?:\.[0-9]+)?)\.', message).group(1)
    cfg.set('sequence', 'chunk_sec', suggested)
    assert chunk_samples_from_config(cfg, hop) % hop == 0


def test_gen_refuses_an_inexact_chunk_sec_before_touching_the_sources(
        corpus, tmp_path, monkeypatch):
    """The [signal]-only 48 kHz change -- the recipe this config used to give.

    Two things: that it is refused at all, and that the refusal is free. This
    check used to live in the renderer, so it arrived as a worker traceback,
    after the sequence plan, the manifest and the full RIR RT60 scan.
    """
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('signal', 'sr', '48000')
    cfg.set('signal', 'n_fft', '1024')
    cfg.set('signal', 'win_len', '1024')
    cfg.set('signal', 'hop_len', '512')
    shipped = _example_config()
    for key in ('seq_sec_min', 'seq_sec_max', 'chunk_sec'):
        cfg.set('sequence', key, shipped.get('sequence', key))

    config_path = tmp_path / 'config_48k_signal_only.ini'
    with open(config_path, 'w') as handle:
        cfg.write(handle)

    def _too_late(*args, **kwargs):
        raise AssertionError(
            'the source inventory was reached before the geometry check')

    for name in ('plan_sequences', 'build_manifest', 'build_unified_manifest'):
        monkeypatch.setattr(gen_aec_dataset_module, name, _too_late)

    output = tmp_path / 'data_48k_signal_only'
    with pytest.raises(ValueError, match=r'\[sequence\] chunk_sec'):
        gen_aec_dataset(argparse.Namespace(
            config=str(config_path), output=str(output), hours=0.004,
            workers=0, resume=False, seed=SEED, split=UNIFIED_SPLIT,
            manifest=None, rebuild_manifest=False, wav_encoding='float32',
        ))
    assert not output.exists()


def test_gen_has_no_sample_rate_override():
    """The rate belongs to the config file the packer is later handed.

    --sample-rate moved [signal] sr alone, which could never produce a valid
    run at the other rate (the grid, chunk_sec and the codec rates stay put)
    and left pack_aec_dataset.py rebuilding the contract at whatever rate the
    file still claimed.
    """
    flags = {action.dest for action in build_parser()._actions}
    assert 'sample_rate' not in flags
    with pytest.raises(SystemExit):
        build_parser().parse_args(['--sample-rate', '48000'])


def test_grid_rejects_a_non_cola_hop():
    with pytest.raises(ValueError, match='COLA'):
        AecGrid(sr=SR, n_fft=512, win_len=512, hop_len=128)


def test_alpha_from_tau_is_frame_rate_independent():
    """One tau must mean the same PHYSICAL time at both rates.

    The coefficients differ -- the two grids have different frame periods -- and
    that is precisely the point: a literal 0.92 written into a config would be
    the same coefficient and therefore a different time constant.
    """
    def recovered_tau(alpha, hop_len, sr):
        return -hop_len / (sr * math.log(alpha))

    a16 = alpha_from_tau(0.2, 256, 16000)
    a48 = alpha_from_tau(0.2, 512, 48000)
    assert recovered_tau(a16, 256, 16000) == pytest.approx(0.2, rel=1e-9)
    assert recovered_tau(a48, 512, 48000) == pytest.approx(0.2, rel=1e-9)

    # What a hardcoded coefficient would have cost: the same 0.92 is 191 ms on
    # one grid and 128 ms on the other.
    assert recovered_tau(a16, 256, 16000) == pytest.approx(
        recovered_tau(a48, 512, 48000), rel=1e-9)
    assert recovered_tau(0.92, 256, 16000) != pytest.approx(
        recovered_tau(0.92, 512, 48000), rel=1e-3)

    assert alpha_from_tau(0.0, 256, 16000) == 0.0
    with pytest.raises(ValueError):
        alpha_from_tau(-1.0, 256, 16000)


# ============================================================
# Scenario mechanics
# ============================================================

def test_sro_produces_sub_sample_drift():
    """A few ppm must be expressible; an integer-rate resampler cannot do it."""
    torch.manual_seed(0)
    signal = torch.randn(SR)
    drifted = resample_by_ratio(signal, 1.0 + 5e-6, SR)
    assert drifted.shape == signal.shape
    # Identical at the start, measurably apart by the end.
    assert (drifted[:100] - signal[:100]).abs().max() < 1e-3
    assert (drifted[-100:] - signal[-100:]).abs().max() > 1e-3
    assert torch.equal(resample_by_ratio(signal, 1.0, SR), signal)


@pytest.mark.parametrize('mode', FAR_THEN_NEAR_MODES)
def test_far_then_near_curriculum_covers_each_first_chunk_phase(corpus, mode):
    """The complete adaptation/onset event must fit one shuffled chunk."""
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('activity', 'p_far_then_near', '1')
    cfg.set('activity', 'far_pre_roll_sec_min', '4')
    cfg.set('activity', 'far_pre_roll_sec_max', '4')
    cfg.set('activity', 'far_stop_lead_sec_min', '0.25')
    cfg.set('activity', 'far_stop_lead_sec_max', '0.25')
    cfg.set('activity', 'far_restart_gap_sec_min', '1')
    cfg.set('activity', 'far_restart_gap_sec_max', '1')
    chunk = 10 * SR
    onset = 4 * SR

    class FixedModeRandom(random.Random):
        def choice(self, values):
            assert tuple(values) == FAR_THEN_NEAR_MODES
            return mode

        def expovariate(self, lambd):
            # Stable utterance boundaries at integer multiples of the mean;
            # in particular, none is accidentally placed at the 4 s onset.
            return 1.0 / lambd

    far, near, event = _apply_far_then_near_curriculum(
        [(0, 2 * chunk)], [(0, 2 * chunk)],
        n_samples=2 * chunk, chunk_samples=chunk, sr=SR,
        rng=FixedModeRandom(7), cfg=cfg,
    )

    def coverage(runs, start, end):
        return sum(max(0, min(hi, end) - max(lo, start))
                   for lo, hi in runs)

    assert event.mode == mode
    expected_onset = onset if mode == 'far_continues' else onset + SR // 4
    assert event.near_onset_sample == expected_onset
    assert coverage(near, 0, expected_onset) == 0
    assert coverage(near, expected_onset, chunk) == chunk - expected_onset
    assert coverage(far, 0, onset) == onset
    if mode == 'far_continues':
        assert coverage(far, onset, chunk) == chunk - onset
        assert event.far_stop_sample == -1
        assert event.far_restart_sample == -1
        assert all(end != onset for _start, end in far[:-1])
    elif mode == 'far_stops':
        assert event.far_stop_sample == onset
        assert coverage(far, onset, chunk) == 0
        assert event.far_restart_sample == -1
    else:
        restart = expected_onset + SR
        assert event.far_stop_sample == onset
        assert event.far_restart_sample == restart
        assert coverage(far, onset, restart) == 0
        assert coverage(far, restart, chunk) == chunk - restart

    # Only chunk zero is scripted; the original schedule survives afterward.
    assert coverage(far, chunk, 2 * chunk) == chunk
    assert coverage(near, chunk, 2 * chunk) == chunk


def test_far_then_near_curriculum_is_not_a_stress_bundle(corpus):
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('activity', 'p_far_then_near', '1')
    far_runs = [(0, SR)]
    near_runs = [(2 * SR, 3 * SR)]

    far, near, event = _apply_far_then_near_curriculum(
        far_runs, near_runs,
        n_samples=10 * SR, chunk_samples=10 * SR, sr=SR,
        rng=random.Random(9), cfg=cfg, eligible=False,
    )
    assert far == far_runs
    assert near == near_runs
    assert event.mode == 'none'


def test_renderer_applies_far_then_near_curriculum_to_eligible_plan(corpus):
    """The renderer gate, not only the helper, must emit the curriculum."""
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('activity', 'p_far_then_near', '1')
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(corpus['manifest'], 'train'), corpus_seed=SEED)
    rendered = renderer.render(SequencePlan(
        sequence_id=1000, n_chunks=2, scenario='double_talk',
        seed=stable_seed(SEED, 'test', 'far-then-near-renderer'),
    ))

    meta = rendered.chunk_meta[0]
    onset = meta['near_onset_sample']
    stop = meta['far_stop_sample']
    restart = meta['far_restart_sample']
    view = AecStems(rendered.stems)
    assert meta['far_then_near_mode'] in FAR_THEN_NEAR_MODES
    assert 0 < onset < rendered.chunk_samples
    near_active_peak = float(
        view.near_speech[onset:rendered.chunk_samples].abs().max())
    assert near_active_peak > 0.0
    assert float(view.near_speech[:onset].abs().max()) < 1e-6 * near_active_peak
    if meta['far_then_near_mode'] == 'far_continues':
        assert stop == restart == -1
    else:
        assert 0 < stop < onset
        silence_end = (restart if restart >= 0 else rendered.chunk_samples)
        far_peak = float(view.X[:rendered.chunk_samples].abs().max())
        assert float(view.X[stop:silence_end].abs().max()) < 1e-6 * far_peak


def test_far_then_near_ranges_must_leave_room_for_restart(corpus):
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('activity', 'p_far_then_near', '1')
    cfg.set('activity', 'far_pre_roll_sec_max', '0.9')
    cfg.set('activity', 'far_restart_gap_sec_max', '0.2')
    with pytest.raises(ValueError, match='restart.*observable'):
        plan_sequences(cfg, 0.001, SEED, 'train')


def test_disabled_far_then_near_ignores_unused_geometry(corpus):
    """A zero-probability curriculum cannot constrain unrelated chunk sizes."""
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('activity', 'p_far_then_near', '0')
    for key in (
        'far_pre_roll_sec_min', 'far_pre_roll_sec_max',
        'far_stop_lead_sec_min', 'far_stop_lead_sec_max',
        'far_restart_gap_sec_min', 'far_restart_gap_sec_max',
    ):
        cfg.remove_option('activity', key)
    aec_dataset_module._validate_far_then_near_curriculum(cfg, 1.024)


def test_echo_is_really_an_echo_of_the_reference(corpus):
    """D must be a delayed, filtered copy of X -- not noise that looks busy.

    Cross-correlating the two puts the peak at the bulk delay plus the RIR's
    own direct-path offset.  Without this, an echo path that silently produced
    an unrelated signal would still pass every other test here, and the model
    would simply fail to converge for no visible reason.
    """
    renderer = AecSequenceRenderer(
        corpus['cfg'], pools_for_split(corpus['manifest'], 'train'),
        corpus_seed=SEED)
    rendered = renderer.render(SequencePlan(
        sequence_id=1001, n_chunks=3, scenario='far_only',
        seed=stable_seed(SEED, 'test', 'echo')))
    view = AecStems(rendered.stems)
    reference, echo = view.X, rendered.audit['echo']

    n = reference.shape[-1]
    spectrum = (torch.fft.rfft(echo, n=2 * n)
                * torch.fft.rfft(reference, n=2 * n).conj())
    correlation = torch.fft.irfft(spectrum, n=2 * n)[:n]
    lag = int(correlation.argmax())
    normalised = float(correlation.max() / (reference.norm() * echo.norm()))

    delay = rendered.chunk_meta[0]['bulk_delay_samples']
    # The RIR keeps 1 ms before its peak (prepare_rir's pre_delay_keep_ms) and
    # the loudspeaker biquads add a little group delay, so a small positive
    # offset from the recorded bulk delay is expected -- a large one is not.
    assert delay <= lag <= delay + int(0.02 * SR), (
        f"echo peaks at lag {lag}, recorded bulk delay {delay}")
    assert normalised > 0.02, f"echo barely correlates with the reference: {normalised}"

    # far_only means no near talker, so the mic is echo + noise alone.
    assert float(view.near_speech.abs().max()) == 0.0
    assert torch.allclose(rendered.audit['mic_preclip'],
                          echo + rendered.audit['noise'], atol=1e-5)


def test_renderer_metadata_covers_the_declared_contract(corpus):
    """The renderer still describes every chunk it makes.

    Nothing persists this any more (the corpus is WAVs only), so it is checked
    where it exists: on the RenderedSequence a worker hands back.
    """
    from AIAEC.dataset_gen.aec_dataset import SCENARIOS
    required = {
        'sequence_id', 'chunk_index', 'speaker_id', 'noise_id', 'rir_id',
        'ser_db', 'snr_db', 'erl_db', 'bulk_delay_samples', 'delay_jitter',
        'sro_ppm', 'nonlinear', 'clipped', 'scenario', 'talk_mode',
        'echo_mode', 'impairments', 'acoustic_tails', 'echo_path_change',
        'codec_mismatch',
        'far_then_near_mode', 'near_onset_sample', 'near_onset_sec',
        'far_stop_sample', 'far_restart_sample',
    }
    checked = 0
    for _plan, rendered in _render_plans(
            corpus['cfg'], corpus['manifest'], 'train'):
        for meta in rendered.chunk_meta:
            checked += 1
            assert required <= set(meta), f"missing {required - set(meta)}"
            assert meta['scenario'] in SCENARIOS
            assert meta['sequence_scenario'] in SCENARIOS
            assert meta['talk_mode'] in ('far_only', 'near_only',
                                         'double_talk', 'duplex_random')
            assert meta['echo_mode'] in ('normal', 'ref_dropout',
                                         'far_active_no_echo')
            assert isinstance(meta['impairments'], list)
            assert isinstance(meta['acoustic_tails'], list)
            assert isinstance(meta['sequence_seed'], int)
            assert isinstance(meta['delay_jitter'], bool)
            assert isinstance(meta['clipped'], bool)
            # ⚠ +-inf is deliberate: it marks a ratio that is undefined because
            # one of its two signals is absent, not a fabricated number.
            assert not math.isnan(meta['ser_db'])
            assert not math.isnan(meta['snr_db'])
    assert checked


def test_packed_metadata_is_provenance_only(packed):
    """A packed entry says WHERE a clip came from and nothing else.

    ⚠ This is the deliberate consequence of dropping the sidecars: a curriculum
    that wants an acoustic property has to measure the stems, which is possible
    precisely because they are stored separately.
    """
    dataset = packed['train']
    for index in range(len(dataset)):
        assert set(dataset.meta(index)) == {'sequence_id', 'chunk_index'}


def test_shard_records_what_a_wav_cannot(packed):
    """The shard carries the one thing the audio cannot: the PBFDKF contract."""
    dataset = packed['train']
    shard = torch.load(dataset.paths[0], map_location='cpu', weights_only=False)
    assert set(shard) == {
        'stems', 'data', 'sr', 'meta', 'linear_aec', 'linear_aec_contract_hash',
    }
    assert shard['sr'] == SR
    assert shard['data'].dtype == torch.float32
    assert shard['linear_aec_contract_hash'] == \
        LinearAecContract.from_dict(shard['linear_aec']).fingerprint()


def test_packed_fingerprint_tracks_geometry_and_inventory(packed, tmp_path):
    """What the fingerprint can and cannot distinguish, stated as a test.

    It covers the corpus's shape and its (sequence, chunk) inventory. It does
    NOT cover which config or seed rendered the audio: nothing on disk records
    that any more, so two same-shaped corpora from different runs fingerprint
    identically. A checkpoint resumed against the wrong one is not caught here.
    """
    original = PackedAecDataset(packed['train'].paths[0], verbose=False)
    shard = torch.load(original.paths[0], map_location='cpu', weights_only=False)

    renumbered = copy.deepcopy(shard)
    renumbered['meta'][0]['sequence_id'] += 1000
    path = tmp_path / 'renumbered.pt'
    torch.save(renumbered, path)
    assert PackedAecDataset(str(path), verbose=False).fingerprint() != \
        original.fingerprint()

    # Same inventory, same geometry, different audio: indistinguishable.
    other_audio = copy.deepcopy(shard)
    other_audio['data'] = torch.randn_like(other_audio['data'])
    path = tmp_path / 'other_audio.pt'
    torch.save(other_audio, path)
    assert PackedAecDataset(str(path), verbose=False).fingerprint() == \
        original.fingerprint()


def test_packed_dataset_rejects_mixed_generation_identity(packed, tmp_path):
    """Two individually valid shards from different runs are not one corpus."""
    first = torch.load(
        packed['train'].paths[0], map_location='cpu', weights_only=False,
    )
    second = copy.deepcopy(first)
    second['sr'] = first['sr'] * 2
    first_path = tmp_path / 'shard_00000.pt'
    second_path = tmp_path / 'shard_00001.pt'
    torch.save(first, first_path)
    torch.save(second, second_path)

    with pytest.raises(ValueError, match='packed-corpus identity'):
        PackedAecDataset(str(tmp_path), verbose=False)


def test_manifest_round_trips(corpus, tmp_path):
    from AIAEC.dataset_gen.manifest import save_manifest
    path = tmp_path / 'manifest.json'
    save_manifest(corpus['manifest'], str(path))
    assert load_manifest(str(path))['splits'] == corpus['manifest']['splits']


def test_a_manifest_written_under_another_split_rule_is_refused(corpus,
                                                                tmp_path):
    """Which sources a split holds is what the manifest IS.

    A manifest file carries a split, and the rules that produce one -- which
    device is held out, for instance -- decide what a corpus rendered from it
    contains. Reusing a file written under a different rule silently renders
    the old corpus, and nothing downstream can tell, so the version is part of
    the file and a mismatch is refused rather than reinterpreted.
    """
    from AIAEC.dataset_gen.manifest import save_manifest
    stale = copy.deepcopy(corpus['manifest'])
    stale['version'] = 'aec_manifest_under_an_earlier_split_rule'
    path = tmp_path / 'manifest.json'
    save_manifest(stale, str(path))
    with pytest.raises(ValueError, match='Rebuild it'):
        load_manifest(str(path))


# ============================================================
# Independent far-end reference pool (far_speech_dir)
# ============================================================

def test_far_speech_pool_defaults_to_the_near_pool_when_unconfigured(corpus):
    """Unset far_speech_dir must be a byte-for-byte no-op."""
    pools = pools_for_split(corpus['manifest'], 'train')
    assert pools.far_speech_files is pools.speech_files
    assert pools.far_speaker_of is pools.speaker_of


def test_far_speech_pool_never_overlaps_the_near_pool_when_configured(corpus, tmp_path):
    generator = torch.Generator().manual_seed(23)
    far_root = tmp_path / 'far_speech'
    for index in range(4):
        # One subdirectory per far speaker, matching the near pool's own
        # directory-per-speaker convention, so _grouping_key's default
        # 'parent_dir' fallback yields a distinct id per far speaker instead
        # of collapsing every flat file into a single '.' group.
        _write(far_root / f'far_{index:02d}' / 'clip.wav',
              _speechlike(4 * SR, generator))

    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('paths', 'far_speech_dir', str(far_root))
    manifest = build_unified_manifest(cfg, seed=SEED, progress=False)
    assert set(manifest['far_speech_files']) == {
        f'far_{index:02d}/clip.wav' for index in range(4)
    }

    renderer = AecSequenceRenderer(
        cfg, pools_for_split(manifest, UNIFIED_SPLIT), corpus_seed=SEED)
    far_used, near_used = set(), set()
    for sequence_id in range(10):
        rendered = renderer.render(SequencePlan(
            sequence_id=sequence_id, n_chunks=2, scenario='double_talk',
            seed=stable_seed(SEED, 'test', f'far-pool-{sequence_id}')))
        for meta in rendered.chunk_meta:
            if meta['far_speaker_id']:
                far_used.add(meta['far_speaker_id'])
            if meta['speaker_id']:
                near_used.add(meta['speaker_id'])
    assert far_used, "no far-end speech was rendered to check"
    assert near_used, "no near-end speech was rendered to check"
    assert far_used <= {f'far_{index:02d}' for index in range(4)}
    assert near_used.isdisjoint(far_used)


# ============================================================
# Resume identity, repack integrity, room invariants, load failures
# ============================================================

def test_resume_rerenders_a_sequence_whose_chunks_are_missing_or_reshaped(
        corpus, tmp_path):
    """What --resume can still see, now that nothing records how audio was made.

    A sequence counts as done only if chunks 0..n-1 are all present with the
    expected rate, length and channel count -- and a chunk BEYOND that range
    (a leftover from a longer earlier render) makes it not-done too, because
    the packer would otherwise pack that surplus chunk as real.
    """
    from AIAEC.dataset_gen.gen_aec_dataset import _pending, _sequence_is_complete

    output = tmp_path / 'aec_data_resume'
    gen_aec_dataset(argparse.Namespace(
        config=str(corpus['config_path']), output=str(output), hours=0.012,
        workers=0, resume=False, seed=SEED, split='train',
        manifest=None, rebuild_manifest=False, wav_encoding='float32',
    ))
    cfg = corpus['cfg']
    seqs_dir = output / 'train' / 'seqs'
    chunk_samples = int(round(cfg.getfloat('sequence', 'chunk_sec') * SR))
    common = dict(sample_rate=SR, chunk_samples=chunk_samples)
    plans = plan_sequences(cfg, 0.012, SEED, 'train')
    assert plans, "fixture produced no sequences to check"

    assert _pending(plans, str(seqs_dir), True, **common) == [], \
        "a freshly generated corpus must resume as fully complete"

    # A missing chunk.
    (seqs_dir / f'{plans[0].sequence_id:06d}_000.wav').unlink()
    assert _pending(plans, str(seqs_dir), True, **common) == [plans[0]]

    # A surplus chunk one past the plan's own count.
    surplus = seqs_dir / f'{plans[-1].sequence_id:06d}_{plans[-1].n_chunks:03d}.wav'
    donor = seqs_dir / f'{plans[-1].sequence_id:06d}_000.wav'
    surplus.write_bytes(donor.read_bytes())
    assert not _sequence_is_complete(plans[-1], str(seqs_dir), **common)


def test_resume_cannot_see_a_config_or_seed_change(corpus, tmp_path):
    """The documented cost of dropping the sidecars, pinned as a test.

    Chunks rendered by a DIFFERENT config or --seed are indistinguishable from
    the right ones as long as their shape matches, so --resume accepts them.
    This is why gen_aec_dataset.py's --resume help says to resume only into a
    directory the same run started.
    """
    from AIAEC.dataset_gen.aec_dataset import SCENARIOS
    from AIAEC.dataset_gen.gen_aec_dataset import _sequence_is_complete

    output = tmp_path / 'aec_data_seed_drift'
    gen_aec_dataset(argparse.Namespace(
        config=str(corpus['config_path']), output=str(output), hours=0.012,
        workers=0, resume=False, seed=SEED, split='train',
        manifest=None, rebuild_manifest=False, wav_encoding='float32',
    ))
    cfg = corpus['cfg']
    seqs_dir = output / 'train' / 'seqs'
    chunk_samples = int(round(cfg.getfloat('sequence', 'chunk_sec') * SR))
    plan = plan_sequences(cfg, 0.012, SEED, 'train')[0]

    drifted = dataclasses.replace(
        plan,
        scenario=next(name for name in SCENARIOS if name != plan.scenario),
        seed=plan.seed + 1,
    )
    assert _sequence_is_complete(
        drifted, str(seqs_dir), sample_rate=SR, chunk_samples=chunk_samples,
    ), "shape-only resume accepts this; if it ever stops, the docs are stale"


def test_resume_forces_a_rerender_when_wav_encoding_changes(corpus, tmp_path):
    """A float32 corpus must not be accepted as an int16 resume (or vice versa)."""
    from AIAEC.dataset_gen.gen_aec_dataset import _pending

    output = tmp_path / 'aec_data_encoding_drift'
    gen_aec_dataset(argparse.Namespace(
        config=str(corpus['config_path']), output=str(output), hours=0.012,
        workers=0, resume=False, seed=SEED, split='train',
        manifest=None, rebuild_manifest=False, wav_encoding='float32',
    ))
    cfg = corpus['cfg']
    plans = plan_sequences(cfg, 0.012, SEED, 'train')
    common = dict(
        sample_rate=SR,
        chunk_samples=int(round(cfg.getfloat('sequence', 'chunk_sec') * SR)),
    )
    assert _pending(plans, str(output / 'train' / 'seqs'), True,
                    wav_encoding='float32', **common) == []
    assert _pending(plans, str(output / 'train' / 'seqs'), True,
                    wav_encoding='int16', **common) == plans


def test_reusing_manifest_with_a_different_seed_is_rejected(corpus, tmp_path):
    """The manifest seed owns the source split and the renderer corpus seed."""
    output = tmp_path / 'aec_data_manifest_seed_drift'
    kwargs = dict(
        config=str(corpus['config_path']), output=str(output), hours=0.012,
        workers=0, resume=False, split='train',
        manifest=str(output / 'source_split.json'), rebuild_manifest=False,
        wav_encoding='float32',
    )
    gen_aec_dataset(argparse.Namespace(seed=SEED, **kwargs))
    second = {**kwargs, 'output': str(tmp_path / 'aec_data_other_output')}
    with pytest.raises(ValueError, match='manifest seed'):
        gen_aec_dataset(argparse.Namespace(seed=SEED + 1, **second))


def test_pack_takes_the_whole_directory_and_only_chunk_files(corpus, tmp_path):
    """Whatever chunk WAVs are there get packed -- and nothing else does.

    There is no declared inventory any more, so the directory IS the corpus.
    A `tmp.` file from a killed write, and anything that is not named
    SSSSSS_CCC.wav, must stay out of the shard.
    """
    output = tmp_path / 'aec_data_whole_dir'
    gen_aec_dataset(argparse.Namespace(
        config=str(corpus['config_path']), output=str(output), hours=0.012,
        workers=0, resume=False, seed=SEED, split='train',
        manifest=None, rebuild_manifest=False, wav_encoding='float32',
    ))
    seqs_dir = output / 'train' / 'seqs'
    real_chunks = sorted(seqs_dir.glob('[0-9]*_[0-9]*.wav'))
    donor = real_chunks[0].read_bytes()
    (seqs_dir / f'tmp.{real_chunks[0].name}').write_bytes(donor)
    (seqs_dir / 'notes.wav').write_bytes(donor)
    (seqs_dir / 'meta.json').write_text('{"old": true}')
    (seqs_dir / '000000.json').write_text('[{"old": true}]')
    (output / 'train' / 'index.json').write_text('{"old": true}')

    _pack(corpus, output / 'train', output / 'packed' / 'train')
    dataset = PackedAecDataset(str(output / 'packed' / 'train'), verbose=False)
    assert len(dataset) == len(real_chunks)


def test_pack_fails_loudly_when_a_sequence_is_missing_some_chunk_wavs(
    corpus, tmp_path,
):
    """A hole in a sequence's chunk numbering (interrupted run, partial copy)
    must fail pack rather than silently ship a sequence shorter than it looks.
    """
    from AIAEC.dataset_gen.pack_aec_dataset import _collect

    output = tmp_path / 'aec_data_partial'
    gen_aec_dataset(argparse.Namespace(
        config=str(corpus['config_path']), output=str(output), hours=0.012,
        workers=0, resume=False, seed=SEED, split='train',
        manifest=None, rebuild_manifest=False, wav_encoding='float32',
    ))
    seqs_dir = output / 'train' / 'seqs'
    (seqs_dir / '000000_000.wav').unlink()

    with pytest.raises(FileNotFoundError, match='000000'):
        _collect(str(seqs_dir))


def test_pack_rejects_a_chunk_of_the_wrong_length(corpus, tmp_path):
    """Every chunk must match the geometry the first one sets.

    Nothing declares T any more, so a short chunk (a truncated copy, a file
    from another run) would otherwise be stacked into a shard of the wrong
    shape -- or crash torch.stack with no idea which file caused it.
    """
    output = tmp_path / 'aec_data_short_chunk'
    gen_aec_dataset(argparse.Namespace(
        config=str(corpus['config_path']), output=str(output), hours=0.012,
        workers=0, resume=False, seed=SEED, split='train',
        manifest=None, rebuild_manifest=False, wav_encoding='float32',
    ))
    victim = sorted((output / 'train' / 'seqs').glob('[0-9]*_[0-9]*.wav'))[-1]
    audio, sr = torchaudio.load(str(victim))
    torchaudio.save(str(victim), audio[:, :-16], sr,
                    encoding='PCM_F', bits_per_sample=32)

    with pytest.raises(ValueError, match='T='):
        _pack(corpus, output / 'train', output / 'packed' / 'train')


def test_pack_rejects_non_finite_wav_samples(corpus, tmp_path):
    """NaN/Inf audio must not be serialized into a training shard."""
    output = tmp_path / 'aec_data_non_finite'
    gen_aec_dataset(argparse.Namespace(
        config=str(corpus['config_path']), output=str(output), hours=0.012,
        workers=0, resume=False, seed=SEED, split='train',
        manifest=None, rebuild_manifest=False, wav_encoding='float32',
    ))
    wav_path = output / 'train' / 'seqs' / '000000_000.wav'
    audio, sr = torchaudio.load(str(wav_path))
    audio[0, 0] = float('nan')
    torchaudio.save(str(wav_path), audio, sr, encoding='PCM_F', bits_per_sample=32)

    with pytest.raises(ValueError, match='NaN or Inf'):
        _pack(corpus, output / 'train', output / 'packed' / 'train')


def test_pack_refuses_to_add_shards_to_a_directory_that_already_has_some(
        corpus, tmp_path):
    """Loading a packed directory takes every shard_*.pt in it.

    With no index file naming this pack's own shards, a leftover from an
    earlier/differently-configured pack would silently join the corpus -- so
    the packer refuses to write into a non-empty shard directory, and
    --overwrite is the explicit way through.
    """
    output = tmp_path / 'aec_data_repack'
    gen_aec_dataset(argparse.Namespace(
        config=str(corpus['config_path']), output=str(output), hours=0.012,
        workers=0, resume=False, seed=SEED, split='train',
        manifest=None, rebuild_manifest=False, wav_encoding='float32',
    ))
    packed_dir = output / 'packed' / 'train'
    _pack(corpus, output / 'train', packed_dir)
    first = sorted(packed_dir.glob('shard_*.pt'))
    assert first
    original = {path.name: path.read_bytes() for path in first}

    # A non-shard .pt is not dataset input. This lets checkpoints/notes live
    # beside a pack without being deserialized as audio.
    (packed_dir / 'notes.pt').write_bytes(b'not a torch payload')
    assert len(PackedAecDataset(str(packed_dir), verbose=False)) > 0

    with pytest.raises(FileExistsError, match='already contains'):
        _pack(corpus, output / 'train', packed_dir)

    # --overwrite is transactional through validation/serialization: corrupt
    # input must leave the previous complete shard set byte-for-byte intact.
    victim = sorted((output / 'train' / 'seqs').glob('[0-9]*_[0-9]*.wav'))[-1]
    audio, sr = torchaudio.load(str(victim))
    audio[0, 0] = float('nan')
    torchaudio.save(str(victim), audio, sr, encoding='PCM_F', bits_per_sample=32)
    with pytest.raises(ValueError, match='NaN or Inf'):
        _pack(corpus, output / 'train', packed_dir,
              overwrite=True, shard_clips=4)
    assert {path.name: path.read_bytes() for path in first} == original
    assert not list(packed_dir.glob('shard_*.pt.tmp'))

    audio[0, 0] = 0.0
    torchaudio.save(str(victim), audio, sr, encoding='PCM_F', bits_per_sample=32)

    _pack(corpus, output / 'train', packed_dir, overwrite=True, shard_clips=4)
    assert sorted(packed_dir.glob('shard_*.pt')) != first or len(first) == 1
    dataset = PackedAecDataset(str(packed_dir), verbose=False)
    assert len(dataset) > 0


def test_generator_refuses_existing_or_out_of_plan_wav_inventory(tmp_path):
    from AIAEC.dataset_gen.gen_aec_dataset import _validate_existing_output

    seqs = tmp_path / 'seqs'
    seqs.mkdir()
    (seqs / '000000_000.wav').touch()
    plans = [SequencePlan(sequence_id=0, n_chunks=1,
                          scenario='far_only', seed=1)]

    with pytest.raises(FileExistsError, match='--resume'):
        _validate_existing_output(plans, str(seqs), resume=False)
    _validate_existing_output(plans, str(seqs), resume=True)

    (seqs / '000001_000.wav').touch()
    with pytest.raises(ValueError, match='outside the current --hours plan'):
        _validate_existing_output(plans, str(seqs), resume=True)


# The reverberant level of a room whose positions are near-duplicates of one
# another: almost all of each response is the direct impulse every position in
# the room shares, so no mixture of them can decorrelate. Measured on the pool
# below: pairwise band-limited |correlation| 0.965-0.973, and the deepest a
# mixture of them reaches is 0.990 at 3 waypoints and 0.988 at 4 -- above
# either mode's configured target, which is what makes such a room ineligible.
ALIKE_TAIL = 0.01


def _run_generator(cfg, tmp_path, name, **overrides):
    """Write ``cfg`` out and drive the CLI over it.

    The generator reads a config FILE, so every test that drives it has to
    write one first; what they vary is the config, not the two-sequence
    unified run around it, which ``overrides`` can still name differently.
    """
    config_path = tmp_path / f'{name}.ini'
    with open(config_path, 'w') as handle:
        cfg.write(handle)
    arguments = dict(
        config=str(config_path), output=str(tmp_path / 'out'), hours=0.002,
        workers=0, resume=False, seed=SEED, split='all',
        manifest=None, rebuild_manifest=False, wav_encoding='float32')
    arguments.update(overrides)
    gen_aec_dataset(argparse.Namespace(**arguments))


def _only_impairments(cfg, **probabilities):
    """Zero every impairment probability, then set the ones named.

    A test that asks for one impairment means ONLY that one: every shipped
    probability is non-zero, so leaving the rest in place would put something
    else in the plan and the assertion would be about a different sequence.
    """
    for name in IMPAIRMENTS:
        cfg.set('impairments', f'p_{name}', '0')
    for name, probability in probabilities.items():
        cfg.set('impairments', f'p_{name}', probability)


def _write_plain_sources(root, generator):
    """The speech and noise every RIR-shaped corpus below needs, and no more.

    What those corpora vary is the RIR pool; the talkers and the noise are
    there only so a sequence can be rendered at all.
    """
    for speaker in range(3):
        _write(root / 'speech' / f'reader_{speaker:03d}' / 'take_0.wav',
               _speechlike(4 * SR, generator))
    for index in range(3):
        _write(root / 'noise' / f'noise_{index:02d}.wav',
               torch.randn(3 * SR, generator=generator) * 0.05)


def _sparse_rir_manifest(tmp_path, rooms, alike=()):
    """A minimal corpus whose room -> RIR-file-count is fully controlled.

    ``rooms`` maps a room name to how many RIR files it gets, e.g.
    ``{'room_00': 1, 'room_01': 2}``. A room named in ``alike`` gets positions
    that barely differ from one another -- enough files for a trajectory and
    nothing for one to move along.
    """
    generator = torch.Generator().manual_seed(29)
    root = tmp_path / 'sparse_rir_corpus'
    _write_plain_sources(root, generator)
    for room, count in rooms.items():
        for index in range(count):
            _write(root / 'rir' / room / f'rir_{index}.wav',
                  _rir(int(0.35 * SR), 0.3, generator,
                       tail=ALIKE_TAIL if room in alike else 1.0))

    cfg = _base_cfg(root)
    return cfg, build_unified_manifest(cfg, seed=SEED, progress=False)


# A room that is mostly near-duplicates of one position with a few distinctive
# ones among them: the shape room eligibility has to decide by EXISTENCE rather
# than over the worst set it offers. Sized past REACH_SUBSET_LIMIT at the
# shipped waypoint counts, so its certificate is the constructive one, and the
# reaching sets are rare enough that most renders exhaust their draws.
MIXED_ROOM_DUPLICATES = 52
MIXED_ROOM_DISTINCT = 1


def _mixed_rir_manifest(tmp_path, duplicates=MIXED_ROOM_DUPLICATES,
                        distinct=MIXED_ROOM_DISTINCT):
    """A corpus with one near-duplicate-heavy room and one ordinary one."""
    generator = torch.Generator().manual_seed(29)
    root = tmp_path / 'mixed_rir_corpus'
    _write_plain_sources(root, generator)
    for index in range(duplicates + distinct):
        _write(root / 'rir' / 'room_mixed' / f'rir_{index:03d}.wav',
               _rir(int(0.35 * SR), 0.3, generator,
                    tail=ALIKE_TAIL if index < duplicates else 1.0))
    for index in range(4):
        _write(root / 'rir' / 'room_plain' / f'rir_{index}.wav',
               _rir(int(0.35 * SR), 0.3, generator, tail=1.0))
    cfg = _base_cfg(root)
    return cfg, build_unified_manifest(cfg, seed=SEED, progress=False)


def test_a_room_of_identical_positions_is_still_certified_over_a_real_set():
    """The certificate is a SET, including where no set beats another.

    A room whose responses are identical reaches exactly 1.0 over every set of
    them -- the case a search that keeps the best "so far" has to be
    initialised for, because 1.0 is the top of the statistic's range rather
    than a value anything can come in under. An empty answer there would report
    a reach with no positions behind it, which nothing downstream can render or
    fall back to.
    """
    generator = torch.Generator().manual_seed(11)
    position = _rir(SR // 4, 0.2, generator, tail=1.0)
    gram = position_gram([position.clone() for _ in range(4)], SR)
    # Few enough sets that this is the enumerating branch, which is the one
    # that ranks sets against each other.
    assert math.comb(4, 3) <= aec_dataset_module.REACH_SUBSET_LIMIT
    chosen, reach = aec_dataset_module.certified_positions(gram, 3)
    assert len(set(chosen)) == 3, chosen
    assert reach == pytest.approx(1.0, abs=1e-6), reach
    assert aec_dataset_module.positions_reach(gram, chosen) == pytest.approx(
        reach, abs=1e-6)


def test_a_room_of_near_duplicates_hosts_the_sets_of_it_that_reach(tmp_path):
    """Eligibility is an EXISTENCE statement, and the DRAW is what enforces it.

    A room can hold near-duplicate positions next to distinctive ones. Refusing
    it because SOME set of its positions saturates would throw away every set
    that does not; certifying it and then rendering whatever the draw returned
    would put a shallower trajectory in the corpus than the metadata claims. So
    the room is eligible when a reaching set exists in it, and the set a
    sequence renders is checked against the same target and drawn again when it
    fails.

    ⚠ The certificate has to be CONSTRUCTIVE for that to be sound. This room
    offers more sets than REACH_SUBSET_LIMIT enumerates, so the room is
    certified over the most distinct positions it holds -- a set the renderer
    can always fall back to, which is why "this set reaches" proves "the room
    can host" without saying anything about the sets nobody enumerated. Ranking
    sets by their most alike pair instead would certify a room over a set it
    might never draw, and rooms this shape are where that comes apart.
    """
    cfg, manifest = _mixed_rir_manifest(tmp_path)
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(manifest, UNIFIED_SPLIT), corpus_seed=SEED)
    target = cfg.getfloat('path_motion', 'movement_position_correlation')
    tolerance = aec_dataset_module.POSITION_REACH_TOLERANCE

    census = renderer.motion_room_census('movement')
    assert census['eligible'] == ['room_mixed', 'room_plain'], census

    gram = renderer._room_gram('room_mixed')
    counts = renderer._waypoint_counts('movement', 'room_mixed')
    for count in counts:
        assert math.comb(gram.shape[0], count) > (
            aec_dataset_module.REACH_SUBSET_LIMIT), (
            "this room no longer exercises the constructive certificate")
        # The constructive certificate reaches, and so does the exhaustive
        # answer it stands in for: the shortcut is not what makes the room
        # eligible.
        assert aec_dataset_module.certified_positions(
            gram, count)[1] <= target + tolerance
    exhaustive = aec_dataset_module.certified_positions(
        gram, min(counts), subset_limit=math.comb(gram.shape[0], min(counts)))
    assert exhaustive[1] <= target + tolerance, exhaustive
    # And the room would be refused by a rule that certified over its WORST
    # set: its near-duplicate positions saturate.
    assert aec_dataset_module.positions_reach(gram, (0, 1, 2)) > target

    fell_back = 0
    redraws = 0
    rendered_here = 0
    with _without_linear_aec():
        for sequence_id in range(12):
            rendered = renderer.render(SequencePlan(
                sequence_id=sequence_id, n_chunks=2, scenario='far_only',
                seed=stable_seed(SEED, 'test', f'mixed-{sequence_id}'),
                talk_mode='double_talk', impairments=('movement',)))
            meta = rendered.chunk_meta[0]
            assert meta['path_motion'] == 'movement', meta
            # The invariant, on every render: what the audio carries is what
            # the corpus configured, in a room where most sets cannot.
            assert meta['echo_path_position_correlation'] <= target + tolerance
            assert meta['near_path_position_correlation'] <= target + tolerance
            if meta['room_id'] != 'room_mixed':
                continue
            rendered_here += 1
            redraws += meta['path_position_redraws']
            certified = renderer._certificate(
                'room_mixed', len(meta['echo_path_waypoints']))[0]
            fell_back += meta['echo_path_waypoints'] == [
                renderer.pools.rir_id_of[
                    renderer.pools.rirs_by_room['room_mixed'][index]]
                for index in certified]
    assert rendered_here, "no sequence drew the room this test is about"
    assert redraws, (
        "no drawn position set saturated, so nothing here exercises the "
        "re-draw the eligibility rule depends on")
    assert fell_back, (
        "no render fell back to the certified set, so the certificate is "
        "untested as the thing that makes the room renderable")


def test_the_certified_fallback_keeps_the_near_talker_off_the_loudspeaker(
        tmp_path):
    """The fallback is a DRAW's replacement, so it keeps the draw's preference.

    A near talker is drawn away from the loudspeaker's positions because a room
    that can keep them apart should: sharing them hands the model a near path
    and an echo path with the same room response. In a room where most sets
    saturate, the draws run out -- and a fallback that answered "which set of
    this room reaches furthest" without the restriction would put the near
    talker on the loudspeaker's own positions in exactly the rooms where the
    preference is hardest to satisfy, which is where it matters most.

    So the certificate is asked on the room MINUS those positions, and only a
    restriction that leaves nothing reaching gives the shared set back. Both
    halves are checked: that such a disjoint reaching set exists here, and that
    every render finds it -- with the draws forced to zero, so the fallback is
    the only thing under test.
    """
    cfg, manifest = _mixed_rir_manifest(tmp_path, distinct=4)
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(manifest, UNIFIED_SPLIT), corpus_seed=SEED)
    target = cfg.getfloat('path_motion', 'movement_position_correlation')
    tolerance = aec_dataset_module.POSITION_REACH_TOLERANCE

    # A disjoint reaching set EXISTS, at every trajectory length the mode can
    # draw here -- otherwise the assertion below would be about a room that
    # leaves the renderer no choice.
    for count in renderer._waypoint_counts('movement', 'room_mixed'):
        echo_set = renderer._certificate('room_mixed', count)[0]
        near_set, near_reach = renderer._certificate(
            'room_mixed', count, tuple(sorted(echo_set)))
        assert not set(near_set) & set(echo_set), (echo_set, near_set)
        assert near_reach <= target + tolerance, (count, near_reach)

    rendered_here = 0
    with pytest.MonkeyPatch.context() as patch:
        # Every trajectory takes the fallback, whatever its draws would have
        # found: what is asserted is a property of the fallback.
        patch.setattr(aec_dataset_module, 'POSITION_DRAW_ATTEMPTS', 0)
        with _without_linear_aec():
            for sequence_id in range(12):
                rendered = renderer.render(SequencePlan(
                    sequence_id=sequence_id, n_chunks=2, scenario='far_only',
                    seed=stable_seed(SEED, 'test', f'avoid-{sequence_id}'),
                    talk_mode='double_talk', impairments=('movement',)))
                meta = rendered.chunk_meta[0]
                if meta['room_id'] != 'room_mixed':
                    continue
                rendered_here += 1
                assert meta['path_position_fallbacks'] == 2, meta
                assert meta['near_path_shared_positions'] == 0, (
                    "the near talker was placed on the loudspeaker's own "
                    "positions although a disjoint reaching set exists")
                # And the restriction did not cost the invariant: both paths
                # still reach what the corpus configured.
                assert meta['echo_path_position_correlation'] <= (
                    target + tolerance)
                assert meta['near_path_position_correlation'] <= (
                    target + tolerance)
    assert rendered_here, "no sequence drew the room this test is about"


def test_a_room_whose_positions_are_too_alike_cannot_host_a_trajectory(
        tmp_path):
    """Reaching the configured path correlation is an ELIGIBILITY rule.

    A trajectory can only decorrelate as far as the positions it mixes differ.
    Where they barely differ the solve saturates at depth 1.0, the render is
    quietly shallower than the corpus asked for, and nothing in the audio says
    so -- the failure the solved depth exists to expose. It is therefore
    decided BEFORE the room is drawn: such a room does not host the mode, the
    moving sequence draws from the rooms that can, and the renderer asserts
    what the draw guarantees.

    ⚠ Certified per waypoint count, because the count is drawn per sequence,
    and over the least distinctive set of positions the room offers, because
    which of them a sequence draws is a draw too.
    """
    cfg, manifest = _sparse_rir_manifest(
        tmp_path, {'room_00': 4, 'room_01': 4, 'room_02': 4},
        alike=('room_02',))
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(manifest, UNIFIED_SPLIT), corpus_seed=SEED)

    census = renderer.motion_room_census('movement')
    assert census['positions_too_alike'] == ['room_02'], census
    assert census['eligible'] == ['room_00', 'room_01'], census
    assert not census['too_few_positions'], census
    # The one-shot switch renders its two positions PURE -- there is no mixture
    # to solve, so the same room hosts it.
    assert 'room_02' in renderer.motion_room_census('echo_path_change')[
        'eligible']

    target = cfg.getfloat('path_motion', 'movement_position_correlation')
    with _without_linear_aec():
        for sequence_id in range(16):
            rendered = renderer.render(SequencePlan(
                sequence_id=sequence_id, n_chunks=2, scenario='far_only',
                seed=stable_seed(SEED, 'test', f'alike-{sequence_id}'),
                talk_mode='double_talk', impairments=('movement',)))
            meta = rendered.chunk_meta[0]
            assert meta['room_id'] != 'room_02'
            assert meta['path_motion'] == 'movement'
            # The invariant the eligibility rule buys: a render can no longer
            # reach less far than it was configured to.
            assert meta['echo_path_position_correlation'] <= target + 1e-3, meta
            assert meta['near_path_position_correlation'] <= target + 1e-3, meta


def test_the_preflight_says_which_rooms_cannot_reach_the_path_correlation(
        tmp_path, capsys):
    """The operator has to be able to tell the two refusals apart.

    A room too sparse for a trajectory and a room whose positions are too alike
    to move along one need different fixes -- more files per room against
    different positions or a different target -- and both leave the same
    symptom, so the census prints them separately and the refusal names the
    one that bites.
    """
    cfg, _manifest = _sparse_rir_manifest(
        tmp_path, {'room_00': 1, 'room_01': 4, 'room_02': 4},
        alike=('room_02',))
    _only_impairments(cfg, movement='1')
    cfg.set('talk_modes', 'p_near_only', '0')
    # room_00 is too sparse and room_02 cannot reach: one room is left, which
    # is the case the two-room rule refuses -- and the message has to say WHY
    # the third room does not count.
    with pytest.raises(ValueError, match='share too much of their energy'):
        _run_generator(cfg, tmp_path, 'alike')


def test_a_pool_with_few_eligible_rooms_still_renders_every_moving_sequence(
        tmp_path, capsys):
    """Eligibility must not cost the corpus its planned motion share.

    The re-draw is what keeps them equal: a sequence that draws a room which
    cannot host its trajectory draws again among the rooms that can, so a pool
    where most rooms are unusable renders the same share of moving sequences as
    a rich one -- at the price of the room cue this file states elsewhere.
    """
    cfg, manifest = _sparse_rir_manifest(
        tmp_path, {'room_00': 1, 'room_01': 2, 'room_02': 4, 'room_03': 4,
                   'room_04': 4},
        alike=('room_03',))
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(manifest, UNIFIED_SPLIT), corpus_seed=SEED)
    census = renderer.motion_room_census('slow_drift')
    assert census['eligible'] == ['room_02', 'room_04'], census
    assert census['too_few_positions'] == ['room_00', 'room_01'], census
    assert census['positions_too_alike'] == ['room_03'], census

    rooms = collections.Counter()
    with _without_linear_aec():
        for sequence_id in range(20):
            rendered = renderer.render(SequencePlan(
                sequence_id=sequence_id, n_chunks=2, scenario='far_only',
                seed=stable_seed(SEED, 'test', f'eligible-{sequence_id}'),
                talk_mode='far_only', impairments=('slow_drift',)))
            meta = rendered.chunk_meta[0]
            assert meta['path_motion'] == 'slow_drift', (
                "a planned trajectory was downgraded, so the corpus's motion "
                "share is not the planned one")
            rooms[meta['room_id']] += 1
    assert set(rooms) == {'room_02', 'room_04'}, rooms


def test_the_near_talkers_trajectory_solves_from_its_own_positions(tmp_path):
    """The near talker stands somewhere else, so its mixture is its own.

    Its depth is solved from the near positions' Gram, not the loudspeaker's:
    borrowing the echo path's depth would move the near path by however much
    the two sets of responses happen to differ, which is a different number in
    every room. Both solves reach the same configured correlation from
    different depths, and both are recorded.
    """
    cfg, manifest = _sparse_rir_manifest(tmp_path, {'room_00': 8,
                                                    'room_01': 8})
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(manifest, UNIFIED_SPLIT), corpus_seed=SEED)
    target = cfg.getfloat('path_motion', 'movement_position_correlation')

    differed = 0
    with _without_linear_aec():
        for sequence_id in range(8):
            rendered = renderer.render(SequencePlan(
                sequence_id=sequence_id, n_chunks=3, scenario='double_talk',
                seed=stable_seed(SEED, 'test', f'near-solve-{sequence_id}'),
                talk_mode='double_talk', impairments=('movement',)))
            meta = rendered.chunk_meta[0]
            assert meta['near_path_motion'] == 'movement'
            assert meta['echo_path_position_correlation'] == pytest.approx(
                target, abs=1e-3)
            assert meta['near_path_position_correlation'] == pytest.approx(
                target, abs=1e-3)
            # The rooms here hold twice waypoints_max, so the near talker never
            # has to share a position with the loudspeaker and the two Grams
            # are over disjoint sets of responses.
            differed += (meta['near_path_mixture_depth']
                         != meta['echo_path_mixture_depth'])
    assert differed == 8, (
        f"{differed} of 8 near paths solved a different depth from the echo "
        "path's; the near solve is reading the loudspeaker's positions")


def test_echo_path_change_never_leaves_a_room_with_only_one_rir(tmp_path):
    """The post-change RIR must stay in the SAME room as the near talker's.

    room_00 has only 1 RIR file, so a path change is not renderable there --
    the post-change waypoint would have to cross into a different room,
    reintroducing the acoustic "this is echo" leak the same-room invariant
    exists to prevent. A sequence that draws it therefore draws again among the
    rooms that can hold the switch, and every rendered switch comes from one of
    those. (A pool with only one such room is refused by the CLI preflight, for
    the reason tested there; the renderer still has to place the sequences it
    is given.)
    """
    cfg, manifest = _sparse_rir_manifest(tmp_path, {'room_00': 1, 'room_01': 2})
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(manifest, UNIFIED_SPLIT), corpus_seed=SEED)

    with _without_linear_aec():
        for sequence_id in range(20):
            rendered = renderer.render(SequencePlan(
                sequence_id=sequence_id, n_chunks=3,
                scenario='echo_path_change',
                seed=stable_seed(SEED, 'test', f'epc-{sequence_id}')))
            for meta in rendered.chunk_meta:
                assert meta['room_id'] == 'room_01'
                assert meta['room_rir_count'] == 2
                assert meta['path_motion'] == 'echo_path_change'
            assert sum(meta['echo_path_change']
                       for meta in rendered.chunk_meta)


def test_echo_path_change_fails_loudly_when_no_room_qualifies(tmp_path):
    cfg, manifest = _sparse_rir_manifest(tmp_path, {'room_00': 1, 'room_01': 1})
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(manifest, UNIFIED_SPLIT), corpus_seed=SEED)

    with pytest.raises(RuntimeError, match='echo_path_change'):
        renderer.render(SequencePlan(
            sequence_id=0, n_chunks=3, scenario='echo_path_change',
            seed=stable_seed(SEED, 'test', 'epc-none-eligible')))


def test_render_fails_loudly_when_every_speech_file_is_unreadable(corpus):
    """A silently-empty, still-labelled chunk is worse than a loud crash."""
    renderer = AecSequenceRenderer(
        corpus['cfg'], pools_for_split(corpus['manifest'], 'train'),
        corpus_seed=SEED)
    renderer.pools.speech_files = ['/nonexistent/reader/take.wav']
    renderer.pools.far_speech_files = ['/nonexistent/reader/take.wav']
    with pytest.raises(RuntimeError, match='talker run'):
        renderer.render(SequencePlan(
            sequence_id=0, n_chunks=3, scenario='double_talk',
            seed=stable_seed(SEED, 'test', 'unreadable-speech')))


def test_render_fails_loudly_when_every_noise_file_is_unreadable(corpus):
    renderer = AecSequenceRenderer(
        corpus['cfg'], pools_for_split(corpus['manifest'], 'train'),
        corpus_seed=SEED)
    renderer.pools.noise_files = ['/nonexistent/noise/file.wav']
    with pytest.raises(RuntimeError, match='noise file'):
        renderer.render(SequencePlan(
            sequence_id=0, n_chunks=3, scenario='double_talk',
            seed=stable_seed(SEED, 'test', 'unreadable-noise')))


# ============================================================
# Echo-path motion
# ============================================================

# The RIR pool the movement axis is fitted and gated on.
#
# ⚠ Its RT60s come from the SHIPPED [rir] range -- the same reason the fixture
# renders the shipped loudspeaker population -- so the fitted
# `*_position_correlation` describes rooms the corpus can actually draw. They
# are the midpoints of CALIBRATION_ROOMS equal parts of the span, so both ends
# are present in every build rather than in the lucky ones.
#
# ⚠ Most of slow_drift's remaining distance from its own curve is the
# LOUDSPEAKER POPULATION rather than the rooms or the trajectory. Through the
# shipped drives a path that never moves already measures far under the
# estimator floor on THIS pool, inside the span, and at the short lags down to
# the static-device curve itself -- while with linear devices the same pool
# puts it on the floor. No *_position_correlation can give that back at any
# RT60: a trajectory moves the path further from itself, never back toward the
# curve. Both still-path readings are measured and reported by
# ``test_a_still_path_stays_correlated_with_itself``; re-fitting the drive
# population against the untouched-device curve is a separate calibration, and
# this span is where the trajectory can be fitted without one.
# RT60 adds to that floor rather than causing it: a tail several times the
# estimator's own 64 ms analysis frame is not one frequency response, so the
# floor falls further out. Fitting further out would calibrate the trajectory
# against the device floor and the room draw instead of against the path. What
# happens beyond the span is measured separately, in RT60_POOLS, whose
# tolerance is that measurement and which reports the shipped population on
# one long pool.
CALIBRATION_RT60_SPAN = (0.1, 0.35)
CALIBRATION_ROOMS = 4
CALIBRATION_RIRS_PER_ROOM = 4
# The reverberant level every motion fixture's positions are built at: see
# _rir. It has to be the same in the calibration pool and in the per-RT60
# pools, so that RT60 is the only thing those pools vary.
CALIBRATION_TAIL = 0.23


# The [sequence] and [echo_path] settings every long-sequence motion
# measurement is rendered at. 30 s so the 8 s correlation lag still has window
# pairs to measure, 2 s chunks so the moving label can separate a travelling
# chunk from a dwelling one, and a short CONSTANT bulk delay so the estimator's
# own alignment step stays out of the measurement.
_MOTION_SETTINGS = (
    ('sequence', 'seq_sec_min', '30.0'),
    ('sequence', 'seq_sec_max', '30.0'),
    ('sequence', 'chunk_sec', '2.0'),
    ('echo_path', 'bulk_delay_ms_min', '10'),
    ('echo_path', 'bulk_delay_ms_max', '10'),
)


def _motion_cfg(root, *overrides):
    """A config over ``root``'s sources at the motion measurement geometry."""
    cfg = _base_cfg(root)
    for section, key, value in _MOTION_SETTINGS + overrides:
        cfg.set(section, key, value)
    return cfg


def _linear_devices(corpus):
    """``corpus`` with the loudspeaker axis taken out: every device linear.

    The manifest is shared -- it selects sources and RIRs, neither of which the
    device population touches -- so this is the same corpus with one axis
    removed rather than a different one.
    """
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('devices', 'nonlinear_models', 'linear')
    return {'cfg': cfg, 'manifest': corpus['manifest']}


def _calibration_rt60s(rooms=CALIBRATION_ROOMS, span=CALIBRATION_RT60_SPAN):
    return [span[0] + (span[1] - span[0]) * (index + 0.5) / rooms
            for index in range(rooms)]


def _write_motion_sources(root, rt60s, generator, sr=SR,
                          per_room=CALIBRATION_RIRS_PER_ROOM):
    """Speech, noise and one room per RT60, for every motion fixture here."""
    for speaker in range(4):
        _write(root / 'speech' / f'reader_{speaker:03d}' / 'take_0.wav',
               _speechlike(12 * sr, generator), sr=sr)
    for index in range(3):
        _write(root / 'noise' / f'noise_{index:02d}.wav',
               torch.randn(6 * sr, generator=generator) * 0.05, sr=sr)
    for room, rt60 in enumerate(rt60s):
        for index in range(per_room):
            # The positions of one room differ in level as well as in tail: a
            # loudspeaker that moves changes both, and a pool whose positions
            # are equally loud would leave the level axis to the walk alone.
            _write(root / 'rir' / f'room_{room:02d}' / f'rir_{index}.wav',
                   _rir(int(rt60 * sr), rt60, generator, sr=sr,
                        gain=1.0 + 0.3 * index, tail=CALIBRATION_TAIL), sr=sr)


@contextlib.contextmanager
def _without_linear_aec():
    """Replace the frozen linear AEC with zeros for the duration of a block.

    It is by far the slowest part of a render and channel 5 is irrelevant to
    every path-motion assertion; a 30 s sequence is what the 8 s correlation
    lag needs, and paying for PBFDKF on each one costs minutes.

    A context manager rather than a save/restore pair at five call sites: a
    module-scoped fixture that raises between the two would leave the renderer
    patched for the whole session, and every consumer of it here is a block.
    ``pytest.MonkeyPatch.context`` does the undoing, since the function-scoped
    ``monkeypatch`` fixture cannot be injected into a module-scoped one.
    """
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            aec_dataset_module, 'materialize_linear_error',
            lambda mic, far, contract: (torch.zeros_like(mic),
                                        torch.zeros_like(mic)))
        yield


@contextlib.contextmanager
def _frozen_trajectory():
    """Render with the weight schedule pinned to its first corner and no walk.

    ⚠ Both replacements CALL the original first, so every rng draw the real
    trajectory would have made is still made and the render differs from an
    ordinary one in the schedule alone -- not in its speech, levels or noise.
    """
    original_weights = aec_dataset_module.drift_weights
    original_gain = aec_dataset_module.gain_walk_db
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            aec_dataset_module, 'drift_weights',
            lambda keyframes, n_waypoints, depth, n_samples:
            original_weights(keyframes, n_waypoints, depth,
                             n_samples)[:, :1].expand(-1, n_samples))
        patch.setattr(
            aec_dataset_module, 'gain_walk_db',
            lambda *args, **kwargs: torch.zeros_like(
                original_gain(*args, **kwargs)))
        yield


@pytest.fixture(scope='module')
def motion_corpus(tmp_path_factory):
    """A corpus that renders the path-motion axis the way the corpus ships.

    The LOUDSPEAKER POPULATION is the shipped one, because it is what the
    calibrated trajectory has to reproduce the measured curve THROUGH:
    memoryless distortion decorrelates the Wiener path estimate by itself, and
    a path correlation calibrated without it is not the one the corpus needs.
    Its cost is the per-sequence spread -- with a linear loudspeaker a sequence
    reads ~0.998 at every lag and with a distorting one 0.4-0.98, so the
    population median needs tens of sequences (see CALIBRATION_BAND).
    ``motion_corpus_isolated`` is the same corpus with that axis removed, for
    the statements that are about the trajectory alone.

    The RIR POOL is the shipped [rir] RT60 range for the same reason -- see
    CALIBRATION_RT60_SPAN, which also says why it stops where it does. One room
    per RT60, so a sequence's room draw is a draw over the range.

    ⚠ The capture axis is NOT part of what the gate measures, whatever
    ``[mic]`` says here: the estimator reads the pre-clip echo, so clipping and
    the AGC cannot move any number below.  They are left at their shipped
    probabilities because this fixture also serves the tests that DO read the
    microphone.
    """
    root = tmp_path_factory.mktemp('aec_motion')
    generator = torch.Generator().manual_seed(17)
    rt60s = _calibration_rt60s()
    _write_motion_sources(root, rt60s, generator)

    cfg = _motion_cfg(root)
    manifest = build_unified_manifest(cfg, seed=SEED, progress=False)
    # The manifest filters RIRs by MEASURED RT60, so a pool built at nominal
    # values can silently lose rooms and take the span with them.
    rooms = manifest['splits'][UNIFIED_SPLIT]['rooms_to_rirs']
    assert sorted(len(rirs) for rirs in rooms.values()) == (
        [CALIBRATION_RIRS_PER_ROOM] * len(rt60s)), (
        f"the RT60 filter dropped part of the calibration pool: {rooms}")
    return {'cfg': cfg, 'manifest': manifest}


@pytest.fixture(scope='module')
def motion_corpus_isolated(motion_corpus):
    """``motion_corpus`` with the loudspeaker axis taken out.

    Every device linear, so the trajectory is the only thing that can move the
    path estimate.  Two things are asserted here that the shipped mixture
    cannot carry: what the band MEANS -- that a path found once fails it -- and
    that the long-lag decorrelation comes from the TRAJECTORY rather than from
    the loudspeaker (``test_the_trajectory_alone_carries_the_long_lag_...``).
    On the mixture a distorting loudspeaker supplies much of the same
    decorrelation a real device's own curve carries, so a path that never moves
    is not separated from one that does and neither statement can be made
    there. How close the still and slow_drift medians sit on the shipped
    population is reported by
    ``test_the_two_drift_modes_are_ordered_on_the_shipped_population``.
    The manifest is shared: it selects sources and RIRs, neither of which the
    device population touches.

    ⚠ Only the loudspeaker is reset. Clipping and the AGC are left where the
    shipped fixture has them because they are NOT what isolates this corpus:
    the estimator reads the pre-clip echo, so resetting them would change no
    number here.
    """
    return _linear_devices(motion_corpus)


@pytest.fixture(scope='module')
def motion_corpus_half_reach(motion_corpus_isolated):
    """``motion_corpus_isolated`` with both drift targets halfway to 1.0.

    A corpus whose trajectory still runs and whose mixture only decorrelates
    half as far -- what a pool of near-alike positions would render if nothing
    checked the reach. It is one of the two mutations ISOLATED_TRAJECTORY_MAX
    exists to reject, and rendering it here is what holds those ceilings from
    ABOVE: a ceiling nothing sits above is not a control.

    Isolated for the same reason the green side is: through the shipped
    loudspeaker population the device's own decorrelation would supply most of
    the distance between the two, and the statement is about the trajectory.
    """
    cfg = copy.deepcopy(motion_corpus_isolated['cfg'])
    for mode in ('slow_drift', 'movement'):
        key = f'{mode}_position_correlation'
        cfg.set('path_motion', key,
                str(0.5 * (cfg.getfloat('path_motion', key) + 1.0)))
    return {'cfg': cfg, 'manifest': motion_corpus_isolated['manifest']}


@pytest.fixture(scope='module')
def delay_corpus(tmp_path_factory):
    """A corpus with a SHORT echo path, for the delay measurements.

    What those tests measure is where the echo's alignment against its
    reference changes, and a reverberant path smears exactly that: the echo at
    any instant is a mixture of the reference's recent past over the room's
    whole tail, so a delay step shorter than the tail moves the correlation
    very little. Measured on the calibration pool, the two smallest steps of
    that fixture leave the pre-step alignment winning by 0.03 anywhere and put
    the located instant 0.9-1.4 s from the recorded one. That is a fact about
    measuring a reverberant room rather than about the impairment -- the same
    step is rendered either way -- so the pool here is 60 ms, short enough that
    the alignment is a delay and nothing else. Nothing is calibrated on it.

    Every device linear for the same reason: the measurement is a
    cross-correlation, and loudspeaker distortion only lowers it.
    """
    root = tmp_path_factory.mktemp('aec_delay')
    generator = torch.Generator().manual_seed(17)
    _write_motion_sources(root, [0.06, 0.06], generator)
    cfg = _motion_cfg(root, ('rir', 'rt60_min', '0.01'))
    manifest = build_unified_manifest(cfg, seed=SEED, progress=False)
    return _linear_devices({'cfg': cfg, 'manifest': manifest})


# One renderer per (config object, corpus seed), kept for the module's life.
# Constructing one hashes the config -- every key of every section -- and shells
# out for the frozen frontend's revision, while its own state is caches keyed by
# room, so a reused renderer makes the SAME render as a fresh one.  The config
# is held in the key so an object cannot be collected and another take its id.
_MOTION_RENDERERS = {}


def _motion_renderer(motion_corpus, cfg, corpus_seed):
    key = (id(cfg), corpus_seed)
    if key not in _MOTION_RENDERERS:
        _MOTION_RENDERERS[key] = (cfg, AecSequenceRenderer(
            cfg, pools_for_split(motion_corpus['manifest'], UNIFIED_SPLIT),
            corpus_seed=corpus_seed))
    return _MOTION_RENDERERS[key][1]


def _render_motion(motion_corpus, mode, *, talk_mode='far_only', index=0,
                   cfg=None, echo_mode='normal', corpus_seed=SEED,
                   renderer=None):
    """One motion render, from the renderer that config and seed share."""
    cfg = motion_corpus['cfg'] if cfg is None else cfg
    if renderer is None:
        renderer = _motion_renderer(motion_corpus, cfg, corpus_seed)
    chunk_samples = chunk_samples_from_config(
        cfg, linear_aec_contract_from_config(cfg).hop_size)
    # The config's own rate, not this file's: the same helper renders the
    # 48 kHz config, where a sequence built on 16 kHz would be a third as long
    # and the 8 s lag would have no window pairs left to measure.
    n_chunks = int(cfg.getfloat('sequence', 'seq_sec_max')
                   * cfg.getint('signal', 'sr')) // chunk_samples
    return renderer.render(SequencePlan(
        sequence_id=index, n_chunks=n_chunks, scenario=talk_mode,
        seed=stable_seed(SEED, 'motion', mode, index),
        talk_mode=talk_mode, echo_mode=echo_mode,
        impairments=() if mode == STATIC_PATH else (mode,)))


def _far_echo(rendered):
    """The stored reference and the echo it caused.

    The pair every path measurement here is made on -- the estimator's own
    inputs, and the two channels a real capture would supply.
    """
    return (rendered.stems[STEM_ORDER.index('far_render')],
            rendered.audit['echo'])


def _chunk_window(rendered, chunk_index):
    return slice(chunk_index * rendered.chunk_samples,
                 (chunk_index + 1) * rendered.chunk_samples)


def _active_seconds(far, window, floor_rms, frame_sec=0.01):
    """Seconds of far-end activity inside ``window``.

    Framed and RELATIVE. Speech crosses zero constantly, so counting loud
    samples would call a fully active second partly active; and the far level
    itself spans 16 dB, so an absolute threshold would measure a different
    thing on a quiet sequence. Outside a scheduled run the reference is
    EXACTLY zero, so any small relative floor separates the two cleanly.
    """
    frame = int(frame_sec * SR)
    span = far[window]
    span = span[:span.shape[-1] // frame * frame].reshape(-1, frame)
    return float((span.pow(2).mean(dim=1).sqrt() > floor_rms).sum()) * frame_sec


# The bootstrap every count, band and ceiling in this section is sized by.
# Fixed draws and a fixed seed: these are measurements of the fixture's own
# renders, and a gate sized by a re-seeded bootstrap could pass and fail
# alternately on the same corpus.
BOOTSTRAP_DRAWS = 20000
BOOTSTRAP_SEED = 0


def _bootstrap_resamples(values):
    """``BOOTSTRAP_DRAWS`` same-size resamples of ``values``."""
    sample = np.asarray(values, dtype=float)
    index = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(sample), size=(BOOTSTRAP_DRAWS, len(sample)))
    return sample[index]


def _bootstrap_medians(values):
    """Medians of ``BOOTSTRAP_DRAWS`` same-size resamples of ``values``."""
    return np.median(_bootstrap_resamples(values), axis=1)


def _bootstrap_means(values):
    """Means of the same resamples.

    Reported beside the median wherever the quantity is bimodal: a median is a
    step function of how many draws fall in the near-zero mode, so it moves in
    jumps where the mean moves smoothly.
    """
    return _bootstrap_resamples(values).mean(axis=1)


def _bootstrap_percentiles(values, percentile):
    """That ``percentile`` of the same resamples."""
    return np.percentile(_bootstrap_resamples(values), percentile, axis=1)


def _median_spread(values, target):
    """99th percentile of |median - target| over the bootstrap."""
    return float(np.percentile(np.abs(_bootstrap_medians(values) - target), 99))


def _assert_band_clears_spread(values, reference, band, margin, label,
                               unit='', note=''):
    """A band is a gate only while it is wider than the spread it gates.

    ``band >= p99(|median(values) - reference|) + margin``, bootstrapped from
    the values themselves rather than read off a comment -- a band sized once
    and then read at a different count is a number nobody re-derived. Reported
    on every run as well as asserted, so a band that closes says so before it
    turns red.
    """
    spread = _median_spread(values, reference)
    _report(f"{label}: the median's own 99th percentile is {spread:.4f}{unit} "
            f"from the target, band {band:.2f}{unit}, head "
            f"{band - spread:+.4f}{note}")
    assert band >= spread + margin, (
        f"{label}: the median's own 99th percentile is {spread:.4f}{unit} "
        f"from the target, so the {band:.2f}{unit} band keeps only "
        f"{band - spread:.4f} above the sampling spread it has to survive, "
        f"against {margin:.2f}")
    return spread


def _report(line):
    """A measured row nothing here gates, printed under ``pytest -s``."""
    print(line)


# How many sequences each mode's median is taken over.
#
# ⚠ Set by the SPREAD OF THE MEDIAN, not by taste. One 30 s clip yields ~20
# window pairs at an 8 s lag and a single clip's median swings between 0.38 and
# 0.98; on top of that the shipped device population spreads a whole sequence's
# reading -- one that draws a linear loudspeaker reads ~0.998 at every lag and
# one that draws a hard-clipping one 0.4-0.9 -- and the pool's rooms differ in
# RT60, which moves it again. Each count is a step of MOTION_SEQUENCE_STEP at
# which the bands below clear that mode's own sampling spread; the rule is
# measured on these very renders by
# `test_the_calibration_bands_clear_their_own_sampling_spread`, so no number
# here can go stale without a test failing -- and that test reports the head
# one step down as well, so a count that stops being the smallest says so on
# the run instead of here. They differ per mode because the spreads do --
# slow_drift is calibrated against a curve its own device floor already sits
# near, which is what the 2 s and 8 s lags cost it.
MOTION_SEQUENCES = {STATIC_PATH: 40, 'slow_drift': 160, 'movement': 100}
MOTION_SEQUENCE_STEP = 20
# The same statement for the isolated corpus, where the per-sequence spread is
# an order of magnitude smaller and the ceilings, not the bands, are what has
# to clear it (`test_the_isolated_ceilings_clear_their_own_sampling_spread`).
# ⚠ The drift counts here are NOT the smallest that clears the lower side.
# The ceilings are held from both sides -- above the green median's own spread
# and below the mutations they exist to reject -- and slow_drift's 4 s corridor
# between those two is a few thousandths wide, so what buys the ceiling room to
# sit inside it with the margin on both sides is renders.
ISOLATED_SEQUENCES = {STATIC_PATH: 40, 'echo_path_change': 40,
                      'slow_drift': 160, 'movement': 160}

# How far each lag's median may sit from its calibration target, and how much
# room every band keeps above the sampling spread it has to survive. The rule
# is `band >= p99(|median - target|) + CALIBRATION_MARGIN` at that mode's
# count, asserted rather than described.
# ⚠ 2 s and 8 s are wider than the rest, and that is a statement about what the
# corpus can be held to rather than a concession. The short lags are the
# LOUDSPEAKER population's: the trajectory cannot buy them (anything faster
# than the estimator's own 2 s window is averaged, which RAISES the
# correlation), so their median inherits the whole spread of the device draw
# and their p99 stops shrinking with the count. The 4 s band is the tightest of
# the four and is what the trajectory is held to.
CALIBRATION_BAND = {1.0: 0.10, 2.0: 0.17, 4.0: 0.12, 8.0: 0.13}
CALIBRATION_MARGIN = 0.01
# The same rule for the level step, per mode, against the curve each mode is
# calibrated against. The spread is the device population's again, because the
# statistic counts shape change at a constant level, and it is wider for
# movement -- which is why that band is.
# ⚠ Neither band is what rejects a corpus shipped without a level walk. At
# sigma = 0 the movement median falls far enough for its own band to catch it,
# but slow_drift's does not leave its band at all: most of what the statistic
# counts is path SHAPE change at a constant level, which the trajectory
# supplies either way. What rejects such a corpus for BOTH modes is the walk's
# own spread, asserted below.
LEVEL_STEP_BAND_DB = {'movement': 0.45, 'slow_drift': 0.35}
# The level WALK's own spread, p5-p95 of the recorded track, and how far the
# median of it may sit from that. This is the quantity
# ``*_gain_sigma_db_per_sec`` and ``*_gain_clamp_db`` are set by (a moving
# device measures ~6 dB), and it is asserted separately because the per-second
# step above cannot stand in for it -- for slow_drift the step barely notices
# the walk at all, so a band that does not fail a green generator cannot
# reject a corpus shipped with no level walk. The walk's spread separates them
# completely (6.0 and 3.9 dB against 0.0).
LEVEL_SPREAD_DB = {'movement': 6.0, 'slow_drift': 3.9}
LEVEL_SPREAD_BAND_DB = 1.0


def _render_set(corpus, counts):
    """``counts[mode]`` sequences of each mode, without the frozen linear AEC.

    Index-seeded, so a mode's Nth render is the same render whichever fixture
    asked for it and two fixtures over the same corpus agree by construction.
    """
    with _without_linear_aec():
        return {mode: [_render_motion(corpus, mode, index=index)
                       for index in range(count)]
                for mode, count in counts.items()}


@pytest.fixture(scope='module')
def motion_renders(motion_corpus):
    return _render_set(motion_corpus, MOTION_SEQUENCES)


@pytest.fixture(scope='module')
def isolated_renders(motion_corpus_isolated):
    return _render_set(motion_corpus_isolated, ISOLATED_SEQUENCES)


@pytest.fixture(scope='module')
def half_reach_renders(motion_corpus_half_reach):
    """The half-reach mutation at the counts the ceilings are read at.

    Only the two drift modes: the mutation is a drift target, and a still path
    or a one-shot switch does not read one.
    """
    return _render_set(
        motion_corpus_half_reach,
        {mode: ISOLATED_SEQUENCES[mode] for mode in ('slow_drift',
                                                     'movement')})


def _drift_of(rendered):
    """The drift metrics of one render, computed once and kept on it.

    Every lag comes out of a single estimator pass, so asking per lag would
    redo the clip's GCC-PHAT and its whole STFT four times per sequence -- on
    every module-scoped render.
    """
    measured = getattr(rendered, 'drift_metrics', None)
    if measured is None:
        measured = path_drift_metrics(*_far_echo(rendered), SR)
        rendered.drift_metrics = measured
    return measured


def _drift_values(renders, field, lag=None):
    """Each render's ``field``, with none of them missing.

    ``None`` is a clip that had no window pair to measure the lag over -- a
    fixture that cannot answer the question rather than a low reading, so it is
    refused here instead of dropping out of a median.
    """
    values = [row[field][lag] if lag is not None else row[field]
              for row in (_drift_of(rendered) for rendered in renders)]
    assert all(value is not None for value in values), values
    return values


def _correlations(renders, lag):
    """Each render's path correlation at ``lag``."""
    return _drift_values(renders, 'correlation', lag)


def _bootstrap_correlation(renders, lag):
    """The bootstrap of the correlation MEDIAN at ``lag``."""
    return _bootstrap_medians(_correlations(renders, lag))


def _drift_median(renders, field, lag=None):
    return float(np.median(_drift_values(renders, field, lag)))


def _calibration_deviation(renders, mode):
    """Per-lag distance from the target curve ``mode`` is calibrated against."""
    target = CALIBRATION_TARGETS[CALIBRATION_TARGET_OF_MODE[mode]]['correlation']
    return {lag: abs(_drift_median(renders, 'correlation', lag) - target[lag])
            for lag in DEFAULT_LAGS_SEC}


@pytest.mark.parametrize('mode', ['slow_drift', 'movement'])
def test_drift_reproduces_the_measured_path_correlation(motion_renders, mode):
    """The corpus's movement axis is calibrated, not merely present.

    The targets come from paired static/movement far-end captures on the
    AEC-challenge blind set -- path_drift_metrics.CALIBRATION_TARGETS, the one
    place they live -- measured with the SAME estimator used here, and
    reproduced from the rendered audio alone rather than from the weight
    schedule the renderer wrote down. ``slow_drift`` is calibrated against the
    UNTOUCHED real device, which is what CALIBRATION_TARGET_OF_MODE says.

    ⚠ This renders the SHIPPED device population, because that is the corpus
    the targets have to be reproduced by. The trajectory only has to supply
    what the loudspeaker does not: the same ``*_position_correlation`` lands
    well above this on the isolated corpus, so a value calibrated without the
    device axis is not the one this corpus needs. Both readings are reported --
    these ones here, the isolated long-lag ones by the ceiling test below.

    ⚠ Anything faster than the estimator's own 2 s window is AVERAGED, not
    resolved: adding a fast weight component RAISES this correlation instead of
    lowering it, so the short lags cannot be bought with a faster trajectory.

    ⚠ What is calibrated is a target PATH CORRELATION, not a mixture depth.
    The depth that reaches it is solved per room from the room's own responses
    (``solve_mixture_depth``), which is what makes this number a property of
    the corpus rather than of the fixture's RIR pool -- see
    ``test_the_axis_survives_the_whole_shipped_rt60_range``.
    """
    deviation = _calibration_deviation(motion_renders[mode], mode)
    target = CALIBRATION_TARGETS[CALIBRATION_TARGET_OF_MODE[mode]]['correlation']
    for lag in DEFAULT_LAGS_SEC:
        _report(f"mixture {mode} at {lag:g} s over "
                f"{len(motion_renders[mode])} renders: "
                f"{_drift_median(motion_renders[mode], 'correlation', lag):.3f} "
                f"against a target of {target[lag]:.2f}, band "
                f"{CALIBRATION_BAND[lag]:.2f}")
        assert deviation[lag] <= CALIBRATION_BAND[lag], (
            f"{mode} lag {lag:g} s is {deviation[lag]:.3f} from its target, "
            f"band {CALIBRATION_BAND[lag]:.2f}")


@pytest.mark.parametrize('mode', ['slow_drift', 'movement'])
def test_the_level_walk_reproduces_the_measured_one(motion_renders, mode):
    """The level scalar of the same calibration, on the same renders.

    Both drift modes, and both halves of each: the per-second step against its
    calibration target, and the walk's own SPREAD against what
    ``*_gain_sigma_db_per_sec`` / ``*_gain_clamp_db`` were set by. A walk that
    is never gated is a walk that can be shipped at sigma = 0, and the step
    alone cannot notice for either mode -- see LEVEL_SPREAD_DB.

    ⚠ ``gain_step_db`` is NOT a pure level measurement -- it is the
    least-squares complex scalar between consecutive windows, so a shape change
    contributes to it at a constant level. That is exactly why it is banded
    here rather than tuned away: the level walk supplies a minority of it, and
    a generator that closed the gap on this number with sigma alone would be
    over-driving the level axis by everything the trajectory contributes.

    ⚠ ``relative_change`` is reported by the estimator and deliberately NOT
    banded. At this count its own bootstrap interval spans most of the range
    the statistic takes over the whole level-walk axis, so any ceiling it could
    carry would be wider than the effect it would have to catch; the interval
    is printed with the median so the instability is visible instead of being
    asserted away. What rejects a frozen path is the correlation band and the
    ordering, on more evidence than one scalar.
    """
    target = CALIBRATION_TARGETS[CALIBRATION_TARGET_OF_MODE[mode]]
    measured = _drift_median(motion_renders[mode], 'gain_step_db')
    band = LEVEL_STEP_BAND_DB[mode]
    assert abs(measured - target['gain_step_db']) <= band, (
        f"{mode} gain_step_db is {measured:.3f} against a target of "
        f"{target['gain_step_db']:.3f}, band {band:.2f}")
    change = _drift_values(motion_renders[mode], 'relative_change')
    assert all(math.isfinite(value) for value in change), change
    interval = _bootstrap_medians(change)
    _report(f"{mode} relative_change median {float(np.median(change)):.3f}, "
            f"bootstrap 95% [{float(np.percentile(interval, 2.5)):.3f}, "
            f"{float(np.percentile(interval, 97.5)):.3f}] over "
            f"{len(change)} renders, target "
            f"{target.get('relative_change', float('nan')):.3f}")

    # ⚠ Read off the walk the renderer recorded, not out of the audio, and
    # deliberately so: this is the knob's own quantity, and the audio statistic
    # above cannot resolve it for slow_drift.
    spread = float(np.median([
        rendered.chunk_meta[0]['echo_path_gain_walk_db']['p95_db']
        - rendered.chunk_meta[0]['echo_path_gain_walk_db']['p5_db']
        for rendered in motion_renders[mode]]))
    assert abs(spread - LEVEL_SPREAD_DB[mode]) <= LEVEL_SPREAD_BAND_DB, (
        f"{mode}'s level walk spreads {spread:.2f} dB p5-p95 against the "
        f"{LEVEL_SPREAD_DB[mode]:.1f} dB its sigma and clamp were set by")


@pytest.mark.parametrize('mode', ['slow_drift', 'movement'])
def test_the_calibration_bands_clear_their_own_sampling_spread(motion_renders,
                                                               mode):
    """A band and a count are one decision, and it is measured here.

    A band is a gate only if it is wider than the spread of the statistic it
    gates: a green generator has to pass it at the count the fixture renders,
    at every lag and for the level step too, or the suite is red on the corpus
    it was calibrated from. The rule is therefore
    ``band >= p99(|median - target|) + margin``, bootstrapped from THESE
    renders rather than read off a comment, and MOTION_SEQUENCES is a count
    that satisfies it. What the count buys is reported as well as asserted: a
    prefix of these renders IS the smaller fixture, since each one is seeded
    from its own index, so the head one MOTION_SEQUENCE_STEP down is printed
    beside the head at the shipped count.

    ⚠ The other direction is not free, which is why the counts are not simply
    raised: the 2 s and 8 s spreads belong to the LOUDSPEAKER population -- the
    trajectory cannot buy them, since anything faster than the estimator's own
    2 s window is averaged -- so they stop shrinking with the count and set the
    two widest bands however many sequences are rendered.
    """
    renders = motion_renders[mode]
    assert len(renders) == MOTION_SEQUENCES[mode]
    shorter = renders[:len(renders) - MOTION_SEQUENCE_STEP]
    target = CALIBRATION_TARGETS[CALIBRATION_TARGET_OF_MODE[mode]]
    def check(name, values_of, reference, band, unit=''):
        """One band held to the rule, with what a shorter fixture reads."""
        step_down = _median_spread(values_of(shorter), reference)
        _assert_band_clears_spread(
            values_of(renders), reference, band, CALIBRATION_MARGIN,
            f"{mode} {name} over {len(renders)} renders", unit=unit,
            note=f" against {band - step_down:+.4f} at {len(shorter)} renders")

    for lag in DEFAULT_LAGS_SEC:
        check(f"at {lag:g} s", lambda group: _correlations(group, lag),
              target['correlation'][lag], CALIBRATION_BAND[lag])
    check('gain_step_db',
          lambda group: _drift_values(group, 'gain_step_db'),
          target['gain_step_db'], LEVEL_STEP_BAND_DB[mode], unit=' dB')


@pytest.mark.parametrize('mode', ['slow_drift', 'movement'])
def test_the_isolated_ceilings_clear_their_own_sampling_spread(
        isolated_renders, half_reach_renders, mode):
    """The same rule for the ceilings, and from BOTH sides.

    A ceiling below the sampling spread of a green isolated median fails the
    corpus it was fitted on; one above what it has to reject is not a control
    at all. So the corridor is asserted at both ends and at the same margin:

        p99(green median) + ISOLATED_MARGIN <= ceiling
        ceiling + ISOLATED_MARGIN <= each mutation's median

    The mutations are the two the ceilings exist for -- a path that never moves
    (the isolated still renders this fixture already holds) and a target path
    correlation halfway to 1.0 (``motion_corpus_half_reach``).

    ⚠ slow_drift's 4 s corridor is the narrow one, and that is a property of
    the mode rather than of the count: it is calibrated against a device nobody
    touched, so its green median and a trajectory reaching half as far sit a
    couple of hundredths apart and two margins are most of that. The only end
    that renders can move is the lower one, which is why
    ISOLATED_SEQUENCES holds more sequences than the lower side alone needs.
    Both ends are printed on every run, so a corridor that closes says so.
    """
    assert len(isolated_renders[mode]) == ISOLATED_SEQUENCES[mode]
    assert len(half_reach_renders[mode]) == ISOLATED_SEQUENCES[mode]
    for lag, ceiling in ISOLATED_TRAJECTORY_MAX[mode].items():
        spread = float(np.percentile(
            _bootstrap_correlation(isolated_renders[mode], lag), 99))
        mutations = (
            ('a path that never moves',
             _drift_median(isolated_renders[STATIC_PATH], 'correlation', lag)),
            ('a target correlation halfway to 1.0',
             _drift_median(half_reach_renders[mode], 'correlation', lag)),
        )
        _report(f"isolated {mode} at {lag:g} s over "
                f"{len(isolated_renders[mode])} renders: the median's own "
                f"99th percentile is {spread:.4f}, ceiling {ceiling:.4f}, "
                + ", ".join(f"{name} {value:.4f}"
                            for name, value in mutations)
                + f"; corridor [{spread + ISOLATED_MARGIN:.4f}, "
                f"{min(value for _n, value in mutations) - ISOLATED_MARGIN:.4f}]")
        assert ceiling >= spread + ISOLATED_MARGIN, (
            f"isolated {mode} at {lag:g} s: the median's own 99th percentile "
            f"is {spread:.4f} over {len(isolated_renders[mode])} renders, so "
            f"the {ceiling:.4f} ceiling keeps only {ceiling - spread:.4f} "
            f"above it, against {ISOLATED_MARGIN:.3f}")
        for name, value in mutations:
            assert value >= ceiling + ISOLATED_MARGIN, (
                f"isolated {mode} at {lag:g} s: {name} reads {value:.4f}, so "
                f"the {ceiling:.4f} ceiling keeps only {value - ceiling:.4f} "
                f"below a mutation it has to reject, against "
                f"{ISOLATED_MARGIN:.3f}")


# How many of the four lags an unmoving path has to sit OUTSIDE the band, per
# target curve, on the isolated corpus. The counts are the statement and the
# assertion below is what holds them; the per-lag deviations they come from are
# in that assertion's own message when it fails.
# ⚠ The one-shot switch against the static-device curve is the odd one, and
# not by accident: it DOES move, once, so it lands where a slowly drifting
# device does at every lag but the longest. That is the whole reason the
# trajectory witness below exists -- the band cannot separate a single
# crossfade from a continuous drift, and nothing about widening it would.
UNMOVING_LAGS_OUTSIDE = {
    (STATIC_PATH, 'movement'): 3,
    (STATIC_PATH, 'slow_drift'): 2,
    ('echo_path_change', 'movement'): 3,
    ('echo_path_change', 'slow_drift'): 1,
}


@pytest.mark.parametrize('unmoving', [STATIC_PATH, 'echo_path_change'])
def test_a_path_that_holds_still_fails_the_drift_calibration(isolated_renders,
                                                             unmoving):
    """The band has to REJECT what the drift modes replaced.

    A calibration test that a still path also passes measures nothing. Both
    the frozen path and the one-shot crossfade -- the only motion the generator
    had before continuous trajectories -- must land outside the band, at the
    lags UNMOVING_LAGS_OUTSIDE says they do.

    ⚠ Measured on the ISOLATED corpus, and that is a finding rather than a
    convenience. On the shipped device population the one-shot crossfade sits
    inside the band at every lag against the untouched-device curve, because a
    distorting loudspeaker decorrelates the path estimate by itself, and a
    frozen path's own deviation there moves with the device draw. So the band
    cannot separate "the trajectory moved" from "the loudspeaker distorts" on
    the shipped mixture; what separates the modes there is their ORDERING,
    asserted below.
    """
    for mode in ('slow_drift', 'movement'):
        deviation = _calibration_deviation(isolated_renders[unmoving], mode)
        outside = [lag for lag in DEFAULT_LAGS_SEC
                   if deviation[lag] > CALIBRATION_BAND[lag]]
        assert len(outside) >= UNMOVING_LAGS_OUTSIDE[(unmoving, mode)], (
            f"{unmoving} sits inside the {mode} band at "
            f"{[lag for lag in DEFAULT_LAGS_SEC if lag not in outside]}: "
            f"{ {lag: round(value, 3) for lag, value in deviation.items()} }")


# The long-lag correlation each drift mode's TRAJECTORY has to produce with the
# loudspeaker axis removed. Every ceiling is held from both sides by
# `test_the_isolated_ceilings_clear_their_own_sampling_spread`: ISOLATED_MARGIN
# or more above the bootstrap p99 of its own median at ISOLATED_SEQUENCES, and
# ISOLATED_MARGIN or more below each mutation it has to reject -- a path that
# never moves and a target path correlation halfway to 1.0 -- at BOTH lags for
# BOTH modes. The values are therefore a consequence of that corridor rather
# than a choice, which is why slow_drift's carry a fourth decimal.
# ⚠ The margin is a fifth of the calibration band's, and slow_drift is why:
# that mode is calibrated against a device nobody touched, so the whole
# distance between "drifts as a still device does" and "moves half as far" is
# a few hundredths, and a margin sized for the mixture would not fit inside it.
ISOLATED_TRAJECTORY_MAX = {'movement': {4.0: 0.90, 8.0: 0.87},
                           'slow_drift': {4.0: 0.9590, 8.0: 0.952}}
ISOLATED_MARGIN = 0.005
# The lag the two drift modes are ORDERED at on the shipped device population.
# The longest, because it is the one where the ordering survives its own
# bootstrap: both lags' gaps and their reordering probabilities are printed by
# `test_the_two_drift_modes_are_ordered_on_the_shipped_population`.
ORDERED_MIXTURE_LAG = 8.0


@pytest.mark.parametrize('mode', ['slow_drift', 'movement'])
def test_the_trajectory_alone_carries_the_long_lag_decorrelation(
        isolated_renders, mode):
    """The control the shipped mixture cannot supply: is the path MOVING?

    On the device mixture a drift mode whose trajectory has stopped still
    renders most of the calibration curve -- the loudspeaker's own distortion
    and the level walk through it supply the decorrelation -- so the
    correlation band accepts it and the mode ordering, which puts a frozen
    drift mode between movement and still, never rejects it either. Neither
    control is evidence that the trajectory does anything.

    With the loudspeaker linear there is nothing else left: the long-lag
    correlation here IS the weight schedule. That the ceilings sit below both
    mutations they exist for -- a path that never moves, and a path
    correlation target halfway to 1.0, i.e. a corpus whose positions barely
    differ -- is asserted next to them, at both lags for both modes, rather
    than left to this docstring.
    """
    for lag, ceiling in ISOLATED_TRAJECTORY_MAX[mode].items():
        measured = _drift_median(isolated_renders[mode], 'correlation', lag)
        assert measured <= ceiling, (
            f"isolated {mode} still correlates {measured:.4f} with itself "
            f"{lag:g} s later, ceiling {ceiling:.4f}: the trajectory is not "
            f"what produces this mode's long-lag decorrelation")


def test_the_two_drift_modes_are_ordered_on_the_shipped_population(
        motion_renders, isolated_renders):
    """A band alone cannot tell slow_drift from movement; this can.

    This is the discrimination that holds on the SHIPPED device population: the
    loudspeaker's own distortion moves both modes together, so it cannot
    reorder them, while the trajectory separates them. The gap and the
    bootstrap probability that a green run reads it the other way round are
    reported per lag.

    ⚠ Asserted at 8 s only. At 1 s and 2 s the two modes' medians overlap on
    this population, because the sequences that drew a distorting loudspeaker
    decorrelate inside the estimator's own window whatever the path does; at
    4 s they are ordered but the bootstrap of that ordering is a few per cent
    from a coin toss at MOTION_SEQUENCES, which is not a separation an
    assertion can be built on. The 4 s gap is reported instead, with its own
    probability, so the evidence stays visible without gating the suite on a
    statistic that can turn over.

    ⚠ A STILL path is not part of the ordering ON THE MIXTURE, and that is a
    limit of the gate rather than a property of the corpus. Through the shipped
    loudspeaker population the device floor has already spent most of the
    static-device budget before the trajectory moves, so the still and
    slow_drift medians land inside each other's sampling spread at the still
    count this fixture renders -- the bootstrap probability that still reads
    the higher of the two is reported below, and it is nowhere near a
    separation either way. Rendering enough still sequences to separate them
    costs more than the statement is worth. Where the three-way ordering IS a
    statement is with the loudspeaker axis removed, which is asserted here.
    """
    for lag in (4.0, 8.0):
        drift = _drift_median(motion_renders['slow_drift'], 'correlation', lag)
        moving = _drift_median(motion_renders['movement'], 'correlation', lag)
        reordered = float(
            (_bootstrap_correlation(motion_renders['movement'], lag)
             >= _bootstrap_correlation(motion_renders['slow_drift'],
                                       lag)).mean())
        _report(f"mixture lag {lag:g} s: movement {moving:.3f} vs slow_drift "
                f"{drift:.3f}, gap {drift - moving:+.3f}, P(movement above "
                f"slow_drift) {reordered:.3f}")
        if lag == ORDERED_MIXTURE_LAG:
            assert moving < drift, (
                f"lag {lag:g} s: movement {moving:.3f}, slow_drift "
                f"{drift:.3f}")
        mixture_still = _bootstrap_correlation(
            motion_renders[STATIC_PATH], lag)
        mixture_drift = _bootstrap_correlation(
            motion_renders['slow_drift'], lag)
        _report(f"mixture lag {lag:g} s: still "
                f"{float(np.median(mixture_still)):.3f} vs slow_drift "
                f"{drift:.3f}, P(still above slow_drift) "
                f"{float((mixture_still > mixture_drift).mean()):.2f} at "
                f"{len(motion_renders[STATIC_PATH])} still renders")
        still = _drift_median(isolated_renders[STATIC_PATH], 'correlation',
                              lag)
        isolated_drift = _drift_median(isolated_renders['slow_drift'],
                                       'correlation', lag)
        isolated_moving = _drift_median(isolated_renders['movement'],
                                        'correlation', lag)
        assert isolated_moving < isolated_drift < still, (
            f"isolated lag {lag:g} s: movement {isolated_moving:.3f}, "
            f"slow_drift {isolated_drift:.3f}, still {still:.3f}")


def test_the_position_gram_reads_a_path_the_way_the_estimator_does():
    """The solve's statistic has to BE the measured one, not a relative of it.

    The estimator reads a path through a 64 ms frame: what it reports is the
    response of the first frame of the path, plus the tail beyond it acting as
    state-dependent noise on every window's estimate. A Gram taken over the
    whole response therefore predicts a correlation nobody measures on a
    reverberant pool, and a depth solved from it under-drives the movement axis
    in exactly the rooms the corpus is aimed at.

    Two constant mixture states rendered through the real convolution path and
    read back with the real estimator, against both predictions.

    ⚠ What WITNESSES the frame is the identity below -- nothing past the frame
    may move the Gram, so a whole-response Gram is a different matrix and the
    mutation that computes one fails there.

    ⚠ Which of the two predictions is CLOSER is reported, not asserted. On a
    single pool the two swap places often, and over eight of them the median
    swaps with the seed block as well -- so an assertion on it would be a
    statement about the pools drawn rather than about the frame. What is
    asserted is the bound that does survive a fresh seed block: the framed Gram
    is never worse than the whole-response one by more than the spread of the
    statistic itself, on any pool.
    """
    frame = int(round(ANALYSIS_FRAME_SEC * SR))

    def predictions(seed):
        generator = torch.Generator().manual_seed(seed)
        positions = [_rir(SR, 1.0, generator, tail=1.0) for _ in range(4)]
        # Nothing beyond the frame may move it: that is what "as the estimator
        # sees it" means, and it is the difference from the whole-response Gram.
        framed = position_gram(positions, SR)
        assert np.allclose(framed, position_gram(
            [position[:frame] for position in positions], SR))
        whole = position_gram(positions, SR,
                              frame_sec=positions[0].shape[-1] / SR)
        assert not np.allclose(framed, whole)

        anchor = np.full(len(positions), 1.0 / len(positions))
        corner = aec_dataset_module.drift_corners(len(positions), 1.0).numpy()[0]
        reference = torch.randn(48 * SR, generator=generator) * 0.1
        convolved = [aec_dataset_module.fftconvolve(reference, position)
                     for position in positions]
        weights = torch.zeros(len(positions), convolved[0].shape[-1])
        weights[:, :24 * SR] = torch.tensor(
            anchor, dtype=torch.float32).unsqueeze(1)
        weights[:, 24 * SR:] = torch.tensor(
            corner, dtype=torch.float32).unsqueeze(1)
        echo = aec_dataset_module.apply_weight_schedule(convolved, weights)
        track, _valid, _step = estimate_path_track(
            reference, echo[:reference.shape[-1]], SR)
        first = track[len(track) // 4]
        second = track[len(track) - len(track) // 4]
        measured = float(abs(np.vdot(first, second))
                         / (np.linalg.norm(first) * np.linalg.norm(second)))
        return (abs(mixture_correlation(framed, anchor, corner) - measured),
                abs(mixture_correlation(whole, anchor, corner) - measured))

    framed_errors, whole_errors = [], []
    for seed in (104, 200, 201, 202, 203, 204, 205, 206):
        framed_error, whole_error = predictions(seed)
        framed_errors.append(framed_error)
        whole_errors.append(whole_error)
        _report(f"pool seed {seed}: framed Gram is {framed_error:.4f} from "
                f"what the estimator reads, whole-response Gram "
                f"{whole_error:.4f}")
    _report(f"over {len(framed_errors)} pools the framed Gram predicts what "
            f"the estimator reads to a median of "
            f"{np.median(framed_errors):.4f} and the whole-response one to "
            f"{np.median(whole_errors):.4f}")
    assert max(framed - whole for framed, whole
               in zip(framed_errors, whole_errors)) <= 0.025, (
        f"the framed Gram is worse than the whole-response one by "
        f"{max(f - w for f, w in zip(framed_errors, whole_errors)):.4f} on "
        f"some pool, i.e. by more than the spread of the statistic itself")


def test_the_jitter_floor_rises_when_the_step_reaches_the_output(corpus):
    """``delay_step_at`` indexes the input; the walk's boundaries index the
    output.

    A negative step raises the walk's lower clamp by its own size, but only
    where the step is in the signal -- and the step reaches the OUTPUT one bulk
    delay after the input index it is recorded at. Comparing the recorded index
    against the output boundaries raises the floor that much early, which is
    one-sided in the safe direction and still wrong: the walk is then clamped
    over a stretch that carries no step, by up to one bulk delay's worth of the
    sequence.

    Driven directly, with each stretch's own delay and boundary recorded,
    because the quantity is a decision inside the walk rather than anything the
    audio shows once the walk has been clipped.
    """
    cfg = copy.deepcopy(corpus['cfg'])
    # Many short stretches and a bulk delay close to the raised floor: the two
    # rules differ only over the stretch that ENDS between the step's input
    # index and its arrival, and only where the walk would otherwise go below
    # the raised floor there. The shipped 2-5 stretches over a 0.5 s delay
    # never put a boundary in that window, which is why this drives the walk
    # directly instead of rendering.
    cfg.set('echo_path', 'jitter_steps_min', '40')
    cfg.set('echo_path', 'jitter_steps_max', '60')
    signal = torch.randn(10 * SR, generator=torch.Generator().manual_seed(5))
    base = 3000
    step = -int(0.15 * SR)
    step_at = 4 * SR
    floor = int(SR * cfg.getfloat('echo_path', 'bulk_delay_ms_min') / 1000)
    raised = floor - step

    late = carrying = 0
    for trial in range(200):
        delays, boundaries = [], []
        original_delay = aec_dataset_module.delay_signal
        original_fade = aec_dataset_module._crossfade

        def record_delay(x, samples, _original=original_delay):
            delays.append(int(samples))
            return _original(x, samples)

        def record_fade(a, b, at, fade, _original=original_fade):
            boundaries.append(int(at))
            return _original(a, b, at, fade)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(aec_dataset_module, 'delay_signal', record_delay)
            patch.setattr(aec_dataset_module, '_crossfade', record_fade)
            aec_dataset_module._apply_jittered_delay(
                signal, base, SR,
                random.Random(stable_seed(SEED, 'jitter-floor', trial)),
                cfg, applied_step=step, step_at=step_at)
        # delays[0] is the base delay before any boundary; delays[1:] govern
        # the stretch that starts at the matching boundary and runs to the
        # next one. The step is in the output of that stretch from
        # `step_at + current` onward.
        held = boundaries[1:] + [signal.shape[-1]]
        for current, end in zip(delays[1:], held):
            if step_at < end <= step_at + current:
                late += current < raised
            elif step_at + current < end:
                carrying += 1
                assert current >= raised, (
                    f"a stretch carrying the step ran at {current}, below the "
                    f"{raised}-sample floor its own negative step requires")
    assert carrying, "no stretch carried the step at all"
    assert late, (
        "no stretch ran below the raised floor while the step had not reached "
        "the output yet, over 200 draws: the floor is being raised at the "
        "step's INPUT index instead of where the step arrives")


# The RT60s the movement axis has to survive, as one room-length per pool.
# They cover the whole shipped [rir] range and one pool below its floor, which
# is where the estimator can resolve a path completely: the calibration pool
# itself only spans the range's lower part, for the reason
# CALIBRATION_RT60_SPAN gives, so this is where the rest of the range is
# checked.
RT60_POOLS = {'rt60_60ms': 0.06, 'rt60_200ms': 0.2, 'rt60_350ms': 0.35,
              'rt60_600ms': 0.6, 'rt60_1000ms': 1.0}
POOL_SEQUENCES = 40
# How far a pool's median may sit from the CALIBRATION POOL's own median, per
# lag. Measured over 80 isolated renders per pool and mode, as
# (pool - calibration) at 1/2/4/8 s:
#     60 ms    movement +0.010/+0.026/+0.032/+0.024  slow_drift +0.009/+0.021/+0.023/+0.021
#     200 ms            +0.002/+0.007/+0.002/-0.004             +0.002/+0.004/+0.006/-0.000
#     350 ms            -0.006/-0.009/+0.002/-0.008             -0.004/-0.011/-0.006/-0.010
#     600 ms            -0.020/-0.041/-0.036/-0.032             -0.016/-0.035/-0.034/-0.033
#     1000 ms           -0.032/-0.073/-0.054/-0.044             -0.030/-0.064/-0.061/-0.063
# i.e. a residual that grows with RT60 and reaches 0.073, on top of which the
# bootstrap (B=20000) of this test's own counts -- POOL_SEQUENCES against the
# 80-render reference -- puts the 99th percentile of |pool - calibration| at
# 0.090 for the worst pool. What is left after the Gram is the estimator's own
# view of a path it cannot resolve: it reads a response through a 64 ms frame,
# and the tail beyond that frame acts as state-dependent noise on every
# window's estimate, which no weight schedule can compensate because it is not
# in the weights. Real rooms carry it too.
RT60_POOL_TOLERANCE = 0.12
# How much deeper a mixture the shortest pool needs than the longest, for the
# same configured correlation. Sized from the bootstrap of the two medians at
# POOL_SEQUENCES, like the rest of this section: slow_drift is the narrower of
# the two modes and this is the largest step of 0.005 below the 1st percentile
# of its gap; movement's gap is twice as wide. The measured gap is printed by
# the assertion's own message when it fails.
POOL_DEPTH_SPREAD = 0.03
# The pool the SHIPPED loudspeaker population is reported on, one room length
# above the span the axis is fitted over. Reported and never gated: what that
# population costs is the device calibration's number, not the trajectory's
# (see CALIBRATION_RT60_SPAN), and a gate here would be a gate on a calibration
# this file does not perform. Rendered at POOL_SEQUENCES and read PAIRED
# against the linear renders of the same sequences, so the room and the speech
# drop out of the difference and the DEVICE DRAW is what is left in it.
# ⚠ That difference is bimodal: a near-linear draw costs the correlation
# almost nothing and a hard-clipping one costs it a lot, so its median is a
# step function of how many of the pairs fall in the near-zero mode and moves
# in jumps between index blocks. Both the mean and the median are reported,
# each with its own bootstrap interval, and so is the count in that mode --
# the mean is the summary that behaves at this count.
MIXTURE_POOL = 'rt60_600ms'
# What counts as the near-zero mode of that paired difference, for the count
# printed beside the two summaries.
MIXTURE_COST_NEAR_ZERO = 0.01


@pytest.fixture(scope='module')
def rt60_pool_renders(tmp_path_factory):
    """``motion_corpus_isolated`` rebuilt on one single-RT60 pool per entry.

    MIXTURE_POOL is rendered a second time with the shipped loudspeakers, for
    the reported row that shows what that population does above the span the
    axis is fitted on.
    """
    out = {}
    for name, rt60 in sorted(RT60_POOLS.items()):
        root = tmp_path_factory.mktemp(f'aec_motion_{name}')
        generator = torch.Generator().manual_seed(17)
        _write_motion_sources(root, [rt60, rt60], generator)
        cfg = _motion_cfg(root)
        manifest = build_unified_manifest(cfg, seed=SEED, progress=False)
        mixture = {'cfg': cfg, 'manifest': manifest}
        corpus = _linear_devices(mixture)
        with _without_linear_aec():
            out[name] = {
                mode: [_render_motion(corpus, mode, index=index)
                       for index in range(POOL_SEQUENCES)]
                for mode in ('slow_drift', 'movement')}
            if name == MIXTURE_POOL:
                out[name].update({
                    f'mixture/{mode}': [
                        _render_motion(mixture, mode, index=index)
                        for index in range(POOL_SEQUENCES)]
                    for mode in ('slow_drift', 'movement')})
    return out


@pytest.mark.parametrize('mode', ['slow_drift', 'movement'])
def test_the_axis_survives_the_whole_shipped_rt60_range(
        rt60_pool_renders, isolated_renders, mode):
    """The calibrated number has to describe the CORPUS, not one pool.

    A mixture depth cannot: what the estimator sees is the correlation between
    mixture STATES, and that depends on how much the room's positions share.
    Rendering one fixed depth on these pools moves the same mode 0.15-0.23
    from the pool it was fitted on, because a longer tail decorrelates further
    per unit of weight change -- so a corpus rendered in real rooms would carry
    a different movement axis from the one calibrated here, silently.
    Configuring a target path CORRELATION and solving the depth per room from
    the room's own responses removes that, and every render says which depth it
    needed and what it reached. Median solved depth per pool, movement /
    slow_drift, with the spread the 3-4 waypoint draw adds in brackets:
    60 ms 0.50 (0.47-0.60) / 0.24 (0.22-0.29), 200 ms 0.38 (0.35-0.47) / 0.18
    (0.17-0.23), 350 ms 0.40 (0.38-0.49) / 0.19 (0.19-0.24), 600 ms 0.39
    (0.35-0.47) / 0.19 (0.17-0.23), 1 s 0.36 (0.36-0.45) / 0.18 (0.17-0.22),
    against 0.42 (0.35-0.51) / 0.20 (0.17-0.25) on the calibration pool -- all
    of them rendering the same axis.

    ⚠ Measured on the ISOLATED corpus and against the CALIBRATION POOL's own
    median rather than against the target curve. Both follow from what is being
    asked: this is a statement about the RIR pool, and on the shipped device
    population a per-pool median carries a sampling spread of 0.12-0.20 (see
    MOTION_SEQUENCES) -- ten times the residual it would have to resolve -- so
    a per-pool gate there would either need hundreds of renders per pool or
    measure nothing. The device population is gated on the calibration pool,
    where the count is sized for it.

    ⚠ The rendered axis alone cannot show that the depth is SOLVED here, and
    the depths are therefore asserted too. On a pool whose positions share a
    dominant direct path the Gram correction is a second-order effect: pinning
    every room to the depth the calibration pool solves (0.42 / 0.20) renders
    within 0.079 of the same axis where solving renders within 0.073, which no
    tolerance can separate. What the solve does show is in the depths it picks
    -- the shortest pool needs a markedly deeper mixture than the longest to
    reach the same correlation, by POOL_DEPTH_SPREAD at least -- and that is
    the quantity a configured depth cannot have.
    """
    reference = {lag: _drift_median(isolated_renders[mode], 'correlation', lag)
                 for lag in DEFAULT_LAGS_SEC}
    target = _example_config().getfloat(
        'path_motion', f'{mode}_position_correlation')
    for name in sorted(RT60_POOLS):
        renders = rt60_pool_renders[name][mode]
        for rendered in renders:
            # The solve is what the tolerance below is a residual OF: every
            # render on every pool reaches the configured correlation, at
            # whatever depth that pool needs.
            assert rendered.chunk_meta[0][
                'echo_path_position_correlation'] == pytest.approx(
                    target, abs=1e-3)
        for lag in DEFAULT_LAGS_SEC:
            values = _correlations(renders, lag)
            measured = float(np.median(values))
            assert abs(measured - reference[lag]) <= RT60_POOL_TOLERANCE, (
                f"{mode} on the {name} pool reads {measured:.3f} at "
                f"{lag:g} s against {reference[lag]:.3f} on the calibration "
                f"pool, tolerance {RT60_POOL_TOLERANCE:.2f}")
            # And the tolerance is held to the same rule the bands are: it is
            # a gate only while it is wider than this count's own spread.
            _assert_band_clears_spread(
                values, reference[lag], RT60_POOL_TOLERANCE,
                CALIBRATION_MARGIN,
                f"{mode} on the {name} pool at {lag:g} s over "
                f"{len(values)} renders")

    def depth(name):
        return float(np.median([rendered.chunk_meta[0]['echo_path_mixture_depth']
                                for rendered in rt60_pool_renders[name][mode]]))

    shortest, longest = min(RT60_POOLS, key=RT60_POOLS.get), max(
        RT60_POOLS, key=RT60_POOLS.get)
    assert depth(shortest) - depth(longest) >= POOL_DEPTH_SPREAD, (
        f"{mode} solves {depth(shortest):.3f} on the {shortest} pool and "
        f"{depth(longest):.3f} on the {longest} one: a mixture that reaches "
        f"the same correlation through a 60 ms path and through a 1 s one is "
        f"not being solved from the room's own responses")

    # The shipped loudspeaker population on the same long pool, reported.
    # Everything above ran linear because the device spread is ten times the
    # residual a per-pool gate would have to resolve; this row is what that
    # decision costs in visibility, put back as evidence rather than as a gate.
    # ⚠ PAIRED: the two sets of renders share their sequence seeds and differ
    # only in [devices] nonlinear_models, so the room and the speech drop out
    # of the difference and what is left in it is the device draw -- which is
    # the quantity being reported, and a bimodal one (see MIXTURE_POOL). The
    # mean, the median and the size of the near-zero mode are all printed,
    # because the median of a bimodal sample moves in steps with that size.
    curve = CALIBRATION_TARGETS[CALIBRATION_TARGET_OF_MODE[mode]]['correlation']
    linear = rt60_pool_renders[MIXTURE_POOL][mode]
    shipped = rt60_pool_renders[MIXTURE_POOL][f'mixture/{mode}']
    for lag in DEFAULT_LAGS_SEC:
        cost = [_drift_of(with_devices)['correlation'][lag]
                - _drift_of(without)['correlation'][lag]
                for with_devices, without in zip(shipped, linear)]
        median_interval = _bootstrap_medians(cost)
        mean_interval = _bootstrap_means(cost)
        near_zero = sum(abs(value) < MIXTURE_COST_NEAR_ZERO for value in cost)
        _report(f"{mode} on the {MIXTURE_POOL} pool at {lag:g} s: the shipped "
                f"loudspeaker population reads "
                f"{_drift_median(shipped, 'correlation', lag):.3f} against "
                f"{_drift_median(linear, 'correlation', lag):.3f} linear on "
                f"the same {len(cost)} sequences, a paired cost of "
                f"{float(np.mean(cost)):+.3f} "
                f"[{float(np.percentile(mean_interval, 2.5)):+.3f}, "
                f"{float(np.percentile(mean_interval, 97.5)):+.3f}] mean and "
                f"{float(np.median(cost)):+.3f} "
                f"[{float(np.percentile(median_interval, 2.5)):+.3f}, "
                f"{float(np.percentile(median_interval, 97.5)):+.3f}] median, "
                f"with {near_zero}/{len(cost)} of the pairs costing less than "
                f"{MIXTURE_COST_NEAR_ZERO:g}; the calibration pool reads "
                f"{reference[lag]:.3f} isolated against a target of "
                f"{curve[lag]:.2f}")


# How far the 48 kHz axis may sit from the 16 kHz one, per lag. Measured
# isolated, 60 renders per mode: movement +0.000/-0.008/-0.010/-0.018 and
# slow_drift -0.001/-0.003/-0.001/-0.010 at 1/2/4/8 s, with the bootstrap
# (B=20000) of this test's counts putting the 99th percentile of the
# difference at 0.048.
RATE_TOLERANCE = 0.08
RATE_SEQUENCES = 40


@pytest.mark.parametrize('mode', ['slow_drift', 'movement'])
def test_the_axis_is_the_same_at_the_other_sample_rate(isolated_renders, mode,
                                                       tmp_path):
    """The same statement across the rate grid, on the shipped 48 kHz config.

    A mixture depth does not carry across rates -- the same value renders 0.14
    from the axis it was fitted on here -- while a band-limited path
    correlation does: the Gram is taken over 300-4000 Hz through the
    estimator's own analysis frame, so it means the same thing at every rate
    the corpus is generated at. What is being checked is therefore the CONFIG:
    everything in config.example.48k.ini that the rate touches -- the grid, the
    chunk geometry and the rescaled loudspeaker band limits -- has to leave the
    axis where the 16 kHz one is.

    ⚠ Isolated, and against the 16 kHz ISOLATED axis rather than against the
    calibration curve. The loudspeaker population is where the per-sequence
    spread comes from, and at 48 kHz a render costs three times as much, so a
    gate on the mixture here would need hundreds of renders to resolve a
    difference this small (measured p99 of the difference between the two
    rates' medians: 0.048 against a band of 0.10-0.17). The nonlinearity is
    what the isolated corpus drops; the rescaled band limits, which are the
    rate-dependent part of [devices], are still in force.
    """
    cfg = configparser.ConfigParser()
    assert cfg.read(pathlib.Path(__file__).parents[1] / 'config.example.48k.ini',
                    encoding='utf-8')
    root = tmp_path / 'sources_48k'
    generator = torch.Generator().manual_seed(17)
    _write_motion_sources(root, _calibration_rt60s(), generator, sr=SR)
    for key, value in (('speech_dir', 'speech'), ('noise_dir', 'noise'),
                       ('rir_dir', 'rir')):
        cfg.set('paths', key, str(root / value))
    cfg.set('sequence', 'seq_sec_min', '32.0')
    cfg.set('sequence', 'seq_sec_max', '32.0')
    cfg.set('echo_path', 'bulk_delay_ms_min', '10')
    cfg.set('echo_path', 'bulk_delay_ms_max', '10')
    check_rate_dependent_values(cfg)

    sr = cfg.getint('signal', 'sr')
    corpus = _linear_devices(
        {'cfg': cfg, 'manifest': build_unified_manifest(cfg, seed=SEED,
                                                        progress=False)})
    cfg, manifest = corpus['cfg'], corpus['manifest']
    target = cfg.getfloat('path_motion', f'{mode}_position_correlation')

    with _without_linear_aec():
        renderer = AecSequenceRenderer(
            cfg, pools_for_split(manifest, UNIFIED_SPLIT), corpus_seed=SEED)
        measured = []
        for index in range(RATE_SEQUENCES):
            rendered = _render_motion(corpus, mode, index=index, cfg=cfg,
                                      renderer=renderer)
            # The solve is rate-independent by construction; this is where
            # that is checked rather than assumed.
            assert rendered.chunk_meta[0][
                'echo_path_position_correlation'] == pytest.approx(target,
                                                                   abs=1e-3)
            measured.append(path_drift_metrics(*_far_echo(rendered), sr))
    for lag in DEFAULT_LAGS_SEC:
        values = [row['correlation'][lag] for row in measured]
        median = float(np.median(values))
        reference = _drift_median(isolated_renders[mode], 'correlation', lag)
        assert abs(median - reference) <= RATE_TOLERANCE, (
            f"{mode} at {sr} Hz reads {median:.3f} at {lag:g} s against "
            f"{reference:.3f} at {SR} Hz, tolerance {RATE_TOLERANCE:.2f}")
        _assert_band_clears_spread(
            values, reference, RATE_TOLERANCE, CALIBRATION_MARGIN,
            f"{mode} at {sr} Hz at {lag:g} s over {len(values)} renders")


# Corpus seeds the calibration is checked at. 42 is the gate's own; the other
# two are arbitrary and fixed, because a seed chosen for its result would make
# this test a statement about that seed.
CALIBRATION_SEEDS = (SEED, 7, 2024)
CALIBRATION_SEED_SEQUENCES = 40
# How much wider than CALIBRATION_BAND the per-seed deviation may be. Two
# things live in it: the bootstrap allowance of the smaller count (movement
# n=40 p99 0.083/0.148/0.111/0.102 against n=80's 0.079/0.137/0.090/0.075), and
# the part of the loudspeaker population that is STILL a draw -- the models are
# stratified but each id's drive is not, so a seed whose severe models drew
# high drives renders a lower curve (measured worst: movement at seed 2024 is
# 0.139 from its 4 s target where the gate's own seed is 0.024).
CALIBRATION_SEED_ALLOWANCE = 0.04


def test_every_configured_loudspeaker_model_reaches_every_corpus_seed():
    """The realised loudspeaker population is not a lottery.

    Each device id's nonlinearity is derived from the corpus seed, so an
    independent draw per id leaves whole models out of most corpora: over 200
    seeds of the shipped config, a third have no linear device at all and the
    asymmetric one is missing from more than a quarter. The movement axis is
    calibrated THROUGH this population, so which models it contains is part of
    what the corpus IS, not a detail of --seed.

    Stratifying makes the containment a guarantee whenever there are at least
    as many ids as models, which is also asserted here: with fewer ids the
    permutation still runs but the guarantee it provides is vacuous, and a
    config that quietly slipped under that line would leave this test green
    while saying nothing.
    """
    cfg = _example_config()
    models = [name.strip()
              for name in cfg.get('devices', 'nonlinear_models').split(',')
              if name.strip()]
    ids = [name.strip()
           for name in cfg.get('devices', 'device_ids').split(',')
           if name.strip()]
    assert len(ids) >= len(models), (
        f"{len(ids)} device id(s) cannot realise {len(models)} models")
    for seed in range(200):
        realised = aec_dataset_module.nonlinearity_by_id(cfg, seed)
        assert set(realised) == set(ids)
        assert set(realised.values()) == set(models), (
            f"corpus seed {seed} realises "
            f"{sorted(set(models) - set(realised.values()))} nowhere")
    # Deterministic per (seed, id list), which is what makes a device id mean
    # the same thing across splits, runs and machines.
    assert (aec_dataset_module.nonlinearity_by_id(cfg, SEED)
            == aec_dataset_module.nonlinearity_by_id(cfg, SEED))
    assert (aec_dataset_module.nonlinearity_by_id(cfg, SEED)
            != aec_dataset_module.nonlinearity_by_id(cfg, SEED + 1))


def test_the_train_split_realises_every_loudspeaker_model(corpus):
    """Stratifying the ids is not enough: the corpus renders ONE split.

    ``device_split = disjoint`` holds ids out for validation, so the population
    a --split train corpus renders is a subset of the configured one. Drawn
    without regard to the model, the val id takes its model out of training
    whenever it is the only id carrying it. The movement axis is calibrated
    through the population the corpus actually renders, so the split is where
    the guarantee has to hold; how often the plain permutation misses is
    measured below rather than quoted.

    ⚠ What that means for val FOLLOWS from the assertion rather than needing
    one of its own: val is a subset of the configured ids and train realises
    every configured model, so a val id's model is one train carries. That is
    the whole content of "held-out identity, not held-out model", and a second
    assertion of it could not fail while the first one holds.
    """
    cfg = _example_config()
    models = {name.strip()
              for name in cfg.get('devices', 'nonlinear_models').split(',')
              if name.strip()}
    ids = [name.strip()
           for name in cfg.get('devices', 'device_ids').split(',')
           if name.strip()]
    val_fraction = cfg.getfloat('split', 'val_fraction')
    seeds = range(200)
    plain_misses = 0
    for seed in seeds:
        model_of = aec_dataset_module.nonlinearity_by_id(cfg, seed)
        split = manifest_module._split_devices(
            cfg, ids, val_fraction, random.Random(seed), seed)
        assert set(split['train']) | set(split['val']) == set(ids)
        assert not set(split['train']) & set(split['val'])
        assert {model_of[device] for device in split['train']} == models, (
            f"seed {seed}: the train split realises "
            f"{sorted(models - {model_of[d] for d in split['train']})} nowhere")
        plain = manifest_module._split_groups(
            {device: [device] for device in ids}, val_fraction,
            random.Random(seed), 'device')
        plain_misses += {model_of[d] for d in plain['train']} != models
    # What the model-aware fill is FOR, measured on the shipped id list rather
    # than quoted: a permutation drawn without regard to the model leaves train
    # short of a model on most seeds.
    assert plain_misses > 0.6 * len(seeds), (
        f"a plain permutation misses a model at only {plain_misses} of "
        f"{len(seeds)} seeds, so the model-aware fill is not what keeps the "
        f"train population whole here")

    # And that is the split the manifest carries: the same draw, through
    # build_manifest, on this fixture's own sources.
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('split', 'val_fraction', str(val_fraction))
    built = build_manifest(cfg, seed=SEED, progress=False)
    devices = {name: built['splits'][name]['devices'] for name in ('train',
                                                                   'val')}
    assert not set(devices['train']) & set(devices['val'])
    model_of = aec_dataset_module.nonlinearity_by_id(cfg, SEED)
    assert {model_of[device] for device in devices['train']} == models

    # ⚠ Where the guarantee is not available it is not faked: a val split
    # bigger than the ids whose model another id also carries cannot leave
    # every model in train, and the draw falls back to the plain permutation
    # (this fixture's own val_fraction is such a case). What covers that is the
    # CLI, which prints what each split realises.
    wide = copy.deepcopy(corpus['cfg'])
    split = manifest_module._split_devices(
        wide, ids, wide.getfloat('split', 'val_fraction'),
        random.Random(SEED), SEED)
    assert len(split['val']) == 2 and len(split['train']) == 6
    assert len({model_of[device] for device in split['train']}) < len(models)


@pytest.mark.parametrize('mode', ['slow_drift', 'movement'])
def test_the_calibration_holds_at_other_corpus_seeds(motion_corpus,
                                                     motion_renders, mode):
    """And it is not a property of ONE seed's device realisation either.

    The gate above renders one corpus seed. The loudspeaker population is
    derived from that seed, so a calibration that only holds there is a
    calibration of the seed: this renders the same config at two more and
    checks that every configured model actually reaches the renders and that
    the curve stays inside the band the smaller count admits.

    ⚠ The residual spread is the DRIVE, not the model list: with the models
    stratified, what still moves between seeds is how hard each id happens to
    be driven, and that is what CALIBRATION_SEED_ALLOWANCE covers.
    """
    cfg = motion_corpus['cfg']
    models = {name.strip()
              for name in cfg.get('devices', 'nonlinear_models').split(',')
              if name.strip()}
    target = CALIBRATION_TARGETS[CALIBRATION_TARGET_OF_MODE[mode]]['correlation']
    with _without_linear_aec():
        for seed in CALIBRATION_SEEDS:
            realised = aec_dataset_module.nonlinearity_by_id(cfg, seed)
            measured, drawn = [], set()
            for index in range(CALIBRATION_SEED_SEQUENCES):
                # The gate's own seed is already rendered; re-rendering it here
                # would cost the suite a minute to reproduce metrics it holds.
                rendered = (motion_renders[mode][index] if seed == SEED else
                            _render_motion(motion_corpus, mode, index=index,
                                           corpus_seed=seed))
                measured.append(_drift_of(rendered))
                drawn.add(realised[rendered.chunk_meta[0]['device_id']])
            assert drawn == models, (
                f"corpus seed {seed} renders {mode} through "
                f"{sorted(drawn)}; {sorted(models - drawn)} never appear")
            for lag in DEFAULT_LAGS_SEC:
                values = [row['correlation'][lag] for row in measured]
                deviation = abs(float(np.median(values)) - target[lag])
                allowed = CALIBRATION_BAND[lag] + CALIBRATION_SEED_ALLOWANCE
                assert deviation <= allowed, (
                    f"{mode} at corpus seed {seed} is {deviation:.3f} from its "
                    f"target at {lag:g} s, allowed {allowed:.2f}")
                # ⚠ NOT held to _assert_band_clears_spread. That rule compares
                # a band with p99(|median - target|), which is the sampling
                # spread only where the median sits ON the target. Here the
                # deviation the allowance exists to cover IS a systematic one
                # -- a seed whose severe models drew high drives renders a
                # lower curve -- so the same statistic would be comparing the
                # allowance against the bias it was sized for (measured:
                # movement at 8 s reads 0.1673 against a 0.17 band).
                _report(f"{mode} at corpus seed {seed} at {lag:g} s over "
                        f"{len(values)} renders: {deviation:.4f} from the "
                        f"target, allowed {allowed:.2f}")


# How far a still path may fall on the isolated corpus, and which part of the
# render distribution is held to it. A quantile rather than the worst render:
# see the assertion, which also holds the quantile against its own bootstrap.
STILL_FLOOR = 0.93
STILL_FLOOR_PERCENTILE = 10


def test_a_still_path_stays_correlated_with_itself(isolated_renders,
                                                   motion_renders):
    """The property the whole axis exists to break: a path found once.

    Rendered BOTH ways on the same pool, because the two answers say different
    things and only one of them is a gate.

    ⚠ With the device axis removed what is left is the ESTIMATOR's own floor,
    which is what the ceiling below holds. It is the estimator meeting paths
    several times longer than its own 64 ms analysis frame -- such a path is
    not one frequency response, so its Wiener estimate keeps a
    signal-dependent bias that varies from window to window, and the floor
    falls further the more reverberant the pool is. It is not the Wiener
    regularisation: measured on the sequence that reads lowest, the number does
    not move between an absolute ridge of 1e-9 and one of 1e-5, nor between a
    25 dB and a 15 dB activity gate, nor across +-128 samples of alignment.

    ⚠ Through the SHIPPED loudspeaker population the same motionless path reads
    far below that floor -- on the calibration pool itself, inside
    CALIBRATION_RT60_SPAN, and at the short lags down to the static-device
    curve slow_drift is calibrated against, which is where that mode's own
    remaining deviation is. That is the drive population, not the rooms and not
    the trajectory: no `*_position_correlation` can recover it at any RT60,
    because a trajectory moves the path further from itself and never back
    toward the curve.

    ⚠ The mixture reading is REPORTED with its own bootstrap interval, and the
    interval is why. The drive population is bimodal -- a near-linear device
    reads on the estimator floor and a hard-clipping one far under it -- so the
    median of it swings with the count and cannot carry a threshold at any
    count this fixture can afford. What the test pins is the half that is
    stable: the estimator contributes none of the shortfall. Re-fitting the
    drive population against the untouched-device curve is a separate
    calibration.
    """
    for lag in DEFAULT_LAGS_SEC:
        assert _drift_median(isolated_renders[STATIC_PATH],
                             'correlation', lag) >= 0.97
    curve = CALIBRATION_TARGETS['static_device']['correlation']
    for lag in DEFAULT_LAGS_SEC:
        linear = _drift_median(isolated_renders[STATIC_PATH],
                               'correlation', lag)
        mixture = _correlations(motion_renders[STATIC_PATH], lag)
        interval = _bootstrap_medians(mixture)
        _report(f"still path at {lag:g} s: linear devices {linear:.3f}, "
                f"shipped mixture {float(np.median(mixture)):.3f} "
                f"[{float(np.percentile(interval, 2.5)):.3f}, "
                f"{float(np.percentile(interval, 97.5)):.3f}] over "
                f"{len(mixture)} renders, static-device curve {curve[lag]:.2f}")
    assert _drift_median(isolated_renders[STATIC_PATH], 'gain_step_db') < 0.35
    assert _drift_median(isolated_renders[STATIC_PATH],
                         'relative_change') < 0.05
    for rendered in isolated_renders[STATIC_PATH]:
        measured = _drift_of(rendered)
        assert measured['step_sec'] == 1.0, (
            "the analysis hop must divide the sample rate, or a lag reported "
            "as 8 s is not 8 s")
        assert measured['gain_step_db'] < 0.6
        assert measured['relative_change'] < 0.15
    # The floor is held at a LOW QUANTILE of the renders rather than at every
    # one of them. The still path's per-render reading has a tail -- the
    # estimator's own floor moves with the speech and the room a sequence drew
    # -- so "every one of N" tightens with N and would turn red on a raised
    # count without anything about the corpus having changed. The quantile is
    # sized the way the ceilings above are: its own bootstrap has to clear the
    # floor by ISOLATED_MARGIN, which is asserted here rather than described.
    for lag in DEFAULT_LAGS_SEC:
        values = _correlations(isolated_renders[STATIC_PATH], lag)
        low = float(np.percentile(values, STILL_FLOOR_PERCENTILE))
        spread = float(np.percentile(
            _bootstrap_percentiles(values, STILL_FLOOR_PERCENTILE), 1))
        _report(f"isolated still at {lag:g} s over {len(values)} renders: the "
                f"{STILL_FLOOR_PERCENTILE}th percentile is {low:.4f} (worst "
                f"{min(values):.4f}), floor {STILL_FLOOR:.2f}, its own 1st "
                f"percentile {spread:.4f}")
        assert low >= STILL_FLOOR, (
            f"the bottom {STILL_FLOOR_PERCENTILE}% of still renders reach "
            f"{low:.4f} at {lag:g} s, floor {STILL_FLOOR:.2f}")
        assert spread >= STILL_FLOOR + ISOLATED_MARGIN, (
            f"at {lag:g} s the {STILL_FLOOR_PERCENTILE}th percentile's own "
            f"1st percentile is {spread:.4f} over {len(values)} renders, so "
            f"the {STILL_FLOOR:.2f} floor keeps only {spread - STILL_FLOOR:.4f} "
            f"below it, against {ISOLATED_MARGIN:.3f}")


def test_drift_moves_the_path_level_and_a_still_path_does_not(motion_renders):
    """Level and shape are separate axes, and both have to move."""
    steps = {mode: _drift_median(renders, 'gain_step_db')
             for mode, renders in motion_renders.items()}
    assert steps[STATIC_PATH] < steps['slow_drift'] < steps['movement'], steps
    assert 0.3 <= steps['slow_drift'] <= 1.2, steps
    assert 0.7 <= steps['movement'] <= 1.9, steps


def test_the_weight_schedule_is_a_partition_of_unity(motion_renders):
    """A schedule whose weights did not sum to 1 would modulate the echo's
    level through the back door, and the level axis would stop being separate.
    """
    for mode in ('slow_drift', 'movement'):
        for rendered in motion_renders[mode]:
            weights = rendered.audit['echo_path_weights']
            assert weights.shape[0] >= 3
            assert torch.allclose(weights.sum(dim=0),
                                  torch.ones(weights.shape[1]), atol=1e-5)
            assert float(weights.min()) >= 0.0


def _keyframe_spans(meta):
    """(dwell spans, transition spans) in seconds, from the chunk metadata.

    Read off the KEYFRAMES the renderer recorded, not off moving_chunks(): a
    dwell is a keyframe that repeats its predecessor's waypoint, so this is an
    independent statement of where the path travels and where it holds.
    """
    dwell, transition = [], []
    keyframes = meta['echo_path_keyframes']
    for first, second in zip(keyframes, keyframes[1:]):
        span = (first['t_sec'], second['t_sec'])
        if first['waypoint'] == second['waypoint']:
            dwell.append(span)
        else:
            transition.append(span)
    return dwell, transition


@pytest.mark.parametrize('chunk_sec', [2.0, 10.0])
def test_the_moving_label_separates_travelling_chunks_from_dwelling_ones(
        motion_corpus, chunk_sec):
    """What the label carries, at both geometries the corpus is cut into.

    ⚠ It is the CHUNK LENGTH that decides how much the label says, so both are
    asserted here rather than only the one a fixture happens to pin. A dwell
    lasts at most `*_dwell_sec_max` (4 s for slow_drift), so at the shipped
    `chunk_sec` no chunk can lie inside one: every far-active chunk of a
    drifting sequence contains part of a transition and is labelled, and the
    label says exactly what `path_motion` and `far_active` already say. At a
    chunk shorter than a dwell it separates travelling chunks from dwelling
    ones, which is the only regime where the label is worth reading -- and the
    regime that pins the definition, because a label that fired on the level
    walk as well would be emitted on the dwelling chunks too.

    ⚠ The shipped geometry is READ from config.ini rather than restated, and a
    reference dropout is rendered at it. Every chunk of an ordinary drifting
    sequence is far-active there, so without a dropout the "no far-silent chunk
    is labelled" half of the definition is vacuous at exactly the length the
    corpus is cut into.
    """
    shipped = configparser.ConfigParser()
    assert shipped.read(pathlib.Path(__file__).parents[1] / 'config.ini',
                        encoding='utf-8')
    if chunk_sec > 4.0:
        assert chunk_sec == shipped.getfloat('sequence', 'chunk_sec'), (
            "the degenerate regime is asserted at the length the corpus is "
            "actually cut into, so this parametrisation follows config.ini")
    cfg = copy.deepcopy(motion_corpus['cfg'])
    cfg.set('sequence', 'chunk_sec', str(chunk_sec))
    with _without_linear_aec():
        renders = [_render_motion(motion_corpus, 'slow_drift', index=index,
                                  cfg=cfg)
                   for index in range(10)]
        if chunk_sec > 4.0:
            renders += [_render_motion(motion_corpus, 'slow_drift', index=index,
                                       cfg=cfg, echo_mode='ref_dropout')
                        for index in range(10, 13)]

    dwelling = travelling = travelling_labelled = 0
    chunks = labelled_chunks = far_active_chunks = 0
    for rendered in renders:
        dwell, transition = _keyframe_spans(rendered.chunk_meta[0])
        labelled = {meta['chunk_index'] for meta in rendered.chunk_meta
                    if meta['echo_path_moving']}
        assert labelled, "a drifting sequence with no moving chunk at all"
        chunks += len(rendered.chunk_meta)
        labelled_chunks += len(labelled)
        for meta in rendered.chunk_meta:
            assert not (meta['echo_path_moving'] and not meta['far_active']), (
                f"chunk {meta['chunk_index']} is labelled moving over a "
                "silent reference")
            far_active_chunks += bool(meta['far_active'])
            start = meta['chunk_index'] * chunk_sec
            stop = start + chunk_sec
            inside = lambda spans: any(low <= start and stop <= high
                                       for low, high in spans)
            if inside(dwell):
                dwelling += 1
                assert not meta['echo_path_moving'], (
                    f"chunk {meta['chunk_index']} sits inside a dwell "
                    f"({dwell}) and is still labelled moving")
            elif inside(transition):
                travelling += 1
                travelling_labelled += bool(meta['echo_path_moving'])

    if chunk_sec < 4.0:
        assert labelled_chunks < chunks, (
            f"all {chunks} chunks of every drifting sequence are labelled "
            "moving, so the label says nothing path_motion does not")
        assert dwelling >= 3, f"only {dwelling} chunks landed inside a dwell"
        assert travelling >= 20, travelling
        # Not every one: a transition chunk whose reference happens to be
        # silent carries no echo whose path could be heard moving, and is
        # labelled by its activity instead.
        assert travelling_labelled >= 0.85 * travelling, (
            f"{travelling_labelled}/{travelling} travelling chunks labelled")
    else:
        assert dwelling == 0, (
            f"{dwelling} chunks lie inside a dwell at chunk_sec {chunk_sec:g}; "
            "the label is documented as degenerate above the longest dwell")
        assert labelled_chunks == far_active_chunks, (
            f"{labelled_chunks} of {far_active_chunks} far-active chunks are "
            "labelled; above the longest dwell every one of them must be")
        assert far_active_chunks < chunks, (
            "no far-silent chunk was rendered at this geometry, so the far "
            "gate on the label is not witnessed here")


def test_the_moving_label_agrees_with_the_trajectory_and_the_reference(
        motion_renders, motion_corpus):
    """The label is measured from the trajectory, not asserted from the plan.

    ⚠ It is gated on the chunk's own measured far activity, like every other
    far-side label: with a silent reference the microphone carries no echo, so
    a 'the echo path is moving' chunk would be signal-identical to a near-only
    one and would contradict a ref_dropout chunk beside it.
    """
    cfg = motion_corpus['cfg']
    for mode in ('slow_drift', 'movement'):
        rendered = motion_renders[mode][0]
        schedule = moving_chunks(
            rendered.audit['echo_path_weights'],
            rendered.chunk_samples, len(rendered.chunk_meta),
            cfg.getfloat('path_motion', 'moving_label_weight_delta'))
        assert schedule, f"{mode} rendered a trajectory that never moves"
        for meta in rendered.chunk_meta:
            expected = (meta['chunk_index'] in schedule and meta['far_active'])
            assert meta['echo_path_moving'] == expected, meta['chunk_index']
            if meta['echo_path_moving']:
                assert meta['scenario'] == 'echo_path_moving'
            else:
                assert meta['scenario'] != 'echo_path_moving'
            assert meta['path_motion'] == mode

    for meta in motion_renders[STATIC_PATH][0].chunk_meta:
        assert not meta['echo_path_moving']
        assert meta['scenario'] != 'echo_path_moving'
        assert meta['path_motion'] == STATIC_PATH


def test_the_keyframes_describe_the_trajectory_to_the_end_of_the_render(
        motion_renders, motion_corpus):
    """The metadata has to cover the tail the renderer actually ramps.

    The keyframe walk stops one corner PAST the end of the sequence, and the
    renderer interpolates the final partial span toward that corner. Dropping
    it because its time lies outside the audio leaves the last seconds of every
    sequence with no recorded span at all -- a consumer reading the corners
    back sees the path stop where the audio does not, and `echo_path_event_at`
    can point at a transition whose destination is missing.

    ⚠ What the tail is checked for is its KIND, not a size. How much of the
    final ramp is rendered depends on where the last corner falls: a transition
    starting a fraction of a second before the end renders a sliver of its
    raised cosine (measured excursions down to 0.00003 over these renders), so
    a threshold on the swing would describe the draw rather than the metadata.
    A hold is EXACTLY constant and a ramp moves toward its recorded
    destination -- monotonically, because it is one raised cosine -- and that
    is true of every render.
    """
    holds = ramps = 0
    for mode in ('slow_drift', 'movement'):
        for rendered in motion_renders[mode]:
            meta = rendered.chunk_meta[0]
            keyframes = meta['echo_path_keyframes']
            weights = rendered.audit['echo_path_weights']
            duration = rendered.stems.shape[-1] / SR
            assert len(keyframes) >= 2, keyframes
            assert keyframes[0]['t_sec'] == 0.0
            assert keyframes[-1]['t_sec'] >= duration, (
                f"the corners stop at {keyframes[-1]['t_sec']:.3f} s of a "
                f"{duration:.3f} s render, so the tail has no recorded span")

            # The last recorded span is the one that runs past the end: what
            # it says about the tail has to be what the tail does.
            last, destination = keyframes[-2], keyframes[-1]
            assert last['t_sec'] < duration
            tail = weights[:, int(last['t_sec'] * SR):]
            excursion = float((tail.amax(dim=1) - tail.amin(dim=1)).max())
            if last['waypoint'] == destination['waypoint']:
                # A dwell interpolates between two identical mixtures, so the
                # only movement left is float32 rounding on the blend.
                holds += 1
                # The threshold is float32 rounding on the blend (measured max
                # 1.2e-07 over these renders), not a tolerance: the smallest
                # genuine ramp rise here is 3.3e-05, so anything above that
                # would accept a mislabelled sliver of a transition as a hold.
                assert excursion < 1e-5, (
                    f"the tail is recorded as a hold and moves {excursion:.5f}")
            else:
                ramps += 1
                towards = tail[destination['waypoint']]
                step = (towards[1:] - towards[:-1]).min()
                assert float(step) >= -1e-6, (
                    f"the tail is recorded as a transition to waypoint "
                    f"{destination['waypoint']} and that weight falls by "
                    f"{-float(step):.6f} on the way")
                assert float(towards[-1] - towards[0]) > 0.0, (
                    f"the tail is recorded as a transition to waypoint "
                    f"{destination['waypoint']} and that weight ends where it "
                    f"started")

            transitions = [first['t_sec'] for first, second
                           in zip(keyframes, keyframes[1:])
                           if first['waypoint'] != second['waypoint']]
            assert meta['echo_path_event_at'] / SR in transitions, (
                f"event at {meta['echo_path_event_at'] / SR:.3f} s is not the "
                f"start of any recorded transition {transitions}")
    assert holds and ramps, (
        f"{holds} hold tails and {ramps} ramp tails: both kinds have to occur "
        "for this to be a statement about the kind at all")


def test_a_near_only_sequence_cannot_carry_an_echo_path_impairment():
    """There is no echo path to move, re-time or distort without a far end.

    plan_sequences() strips these; a hand-built plan (a test, a re-render
    script) must be refused rather than rendered as a sequence whose metadata
    claims a moving echo path over a microphone that never carried one.
    """
    for impairment in ('movement', 'slow_drift', 'echo_path_change',
                       'delay_step', 'sro'):
        with pytest.raises(ValueError, match='near_only cannot carry'):
            resolve_sequence_plan(SequencePlan(
                sequence_id=0, n_chunks=2, scenario='near_only', seed=1,
                talk_mode='near_only', impairments=(impairment,)))


def test_freezing_the_trajectory_removes_the_moving_label(motion_corpus):
    """The mutation the label test needs to be worth anything.

    With the weight schedule pinned to its first column and the level walk
    zeroed, a movement sequence is signal-identical to a still one -- so a
    label emitted from the PLAN rather than from the trajectory would survive
    this and is caught here.
    """
    with _without_linear_aec(), _frozen_trajectory():
        rendered = _render_motion(motion_corpus, 'movement')

    assert not any(meta['echo_path_moving'] for meta in rendered.chunk_meta)
    assert not any(meta['scenario'] == 'echo_path_moving'
                   for meta in rendered.chunk_meta)
    # Still a movement PLAN: only the trajectory was frozen.
    assert all(meta['path_motion'] == 'movement'
               for meta in rendered.chunk_meta)


def _event_window_activity(rendered, need_sec):
    """Seconds of far activity inside ``[event, event + need)``, and its cap."""
    far, _echo = _far_echo(rendered)
    event = rendered.chunk_meta[0]['echo_path_event_at']
    assert event >= 0
    stop = min(far.shape[-1], event + int(need_sec * SR))
    floor_rms = 1e-3 * active_rms(far, SR)
    return (_active_seconds(far, slice(event, stop), floor_rms),
            (stop - event) / SR)


def test_a_moving_sequence_carries_far_activity_after_its_last_transition(
        motion_corpus):
    """An echo-path event with no echo behind it teaches nothing.

    ⚠ Measured INSIDE the window, not summed over the whole remainder of the
    sequence. A schedule that falls silent at the event and resumes ten
    seconds later has far activity "after the event" by the loose reading and
    leaves the event itself invisible; that is precisely the case the
    guarantee exists to remove, so the loose reading cannot be what is
    asserted here.
    """
    with _without_linear_aec():
        renders = [_render_motion(motion_corpus, mode, index=index)
                   for mode, index in (('movement', 0), ('movement', 1),
                                       ('slow_drift', 0),
                                       ('echo_path_change', 0))]

    need = motion_corpus['cfg'].getfloat('path_motion',
                                         'far_active_after_event_sec')
    fade = motion_corpus['cfg'].getfloat('activity', 'talk_fade_sec')
    for rendered in renders:
        active, window_sec = _event_window_activity(rendered, need)
        # Capped by what is left of the sequence: a trajectory's last
        # transition can start with less than need to go, and buying the full
        # amount there would mean starting the run BEFORE the event. Every
        # utterance is also faded in and out (activity talk_fade_sec).
        assert active >= window_sec - 4 * fade, (
            f"only {active:.2f} s of far end inside the {window_sec:.2f} s "
            f"window after the event")


def test_a_reference_dropout_never_zeroes_the_stretch_behind_an_event(
        motion_corpus):
    """The dropout yields to the guarantee, not the other way round.

    ⚠ Whether or not the far schedule had to be repaired. A dropout zeroes a
    naturally scheduled run exactly as thoroughly as a repaired one, so
    steering it only around repairs leaves the majority case -- the schedule
    already covered the window -- unprotected.

    ⚠ The sequence indices are CHOSEN, not the first few. Most draws miss the
    protected window on their own, so a test that renders whichever sequences
    come first passes with the keep-out deleted and with the caller passing no
    keep-out at all. Rendering the same fixture with the keep-out ignored, 5
    puts a four-chunk dropout on [11, 12, 13, 14] over a protected [11, 12] and
    14 puts a one-chunk dropout on [8] over [8, 9]; 0 is an ordinary draw that
    misses either way.
    """
    need = motion_corpus['cfg'].getfloat('path_motion',
                                         'far_active_after_event_sec')
    fade = motion_corpus['cfg'].getfloat('activity', 'talk_fade_sec')
    with _without_linear_aec():
        renders = [_render_motion(motion_corpus, 'slow_drift', index=index,
                                  echo_mode='ref_dropout')
                   for index in (0, 5, 14)]

    for rendered in renders:
        far, _echo = _far_echo(rendered)
        dropped = {meta['chunk_index'] for meta in rendered.chunk_meta
                   if meta['scenario'] == 'ref_dropout'}
        assert dropped, "the sequence carries no dropout to steer"
        for index in dropped:
            window = _chunk_window(rendered, index)
            assert float(far[window].abs().max()) == 0.0, (
                "a chunk labelled ref_dropout must have a silent reference")
        event = rendered.chunk_meta[0]['echo_path_event_at']
        protected = set(range(
            event // rendered.chunk_samples,
            (event + int(need * SR) - 1) // rendered.chunk_samples + 1))
        assert not (dropped & protected), (
            f"dropout {sorted(dropped)} covers the event window "
            f"{sorted(protected)}")
        active, window_sec = _event_window_activity(rendered, need)
        assert active >= window_sec - 4 * fade


def test_far_activity_is_guaranteed_inside_the_window_not_merely_after_it():
    """The unit statement of the same rule.

    A schedule active over the first 5 s and again from 14 s carries plenty of
    far end "after" an event at 10 s and none at all where the event is. The
    repair has to see that, and the window it returns is what the dropout is
    steered around -- so it is returned even when nothing had to be added.
    """
    ensure = aec_dataset_module._ensure_far_activity_after
    runs = [(0, 5 * SR), (14 * SR, 30 * SR)]
    repaired, guaranteed = ensure(runs, 10 * SR, 30 * SR, SR, 641)
    assert guaranteed == (10 * SR, 11 * SR)
    covered = sum(max(0, min(end, 11 * SR) - max(start, 10 * SR))
                  for start, end in repaired)
    assert covered == SR

    # Already covered: nothing is added, and the window is STILL returned.
    covered_runs = [(0, 30 * SR)]
    repaired, guaranteed = ensure(covered_runs, 10 * SR, 30 * SR, SR, 641)
    assert repaired == covered_runs
    assert guaranteed == (10 * SR, 11 * SR)

    # Too little sequence left to place an utterance in: no promise is made.
    repaired, guaranteed = ensure(runs, 30 * SR - 80, 30 * SR, SR, 641)
    assert guaranteed is None and repaired == runs


def test_the_one_shot_switch_leaves_room_for_the_window_it_promises(
        motion_corpus):
    """The switch is an event with a guarantee behind it, so it is placed
    where the guarantee fits.

    ⚠ At the SHIPPED chunk geometry, which is where this bites: a 20-30 s
    sequence is 2 or 3 chunks of 10 s, and a switch drawn uniformly inside the
    last of them lands in the final second often enough to matter. The drift
    modes already prefer the last transition that still has the window behind
    it; drawing the one-shot switch without the same limit leaves an event
    whose window is a few milliseconds long -- and the field comment, the
    config and the README all state the guarantee unconditionally.
    """
    cfg = copy.deepcopy(motion_corpus['cfg'])
    cfg.set('sequence', 'chunk_sec', '10.0')
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(motion_corpus['manifest'], UNIFIED_SPLIT),
        corpus_seed=SEED)
    need = int(SR * cfg.getfloat('path_motion', 'far_active_after_event_sec'))

    for n_chunks in (2, 3):
        n_samples = n_chunks * renderer.chunk_samples
        drawn = collections.Counter()
        for trial in range(1000):
            rng = random.Random(stable_seed(SEED, 'switch', n_chunks, trial))
            at = renderer._switch_point(n_chunks, rng)
            assert at >= renderer.chunk_samples, (
                f"switch at {at} is inside the first chunk, which then holds "
                "no audio from the path before it")
            assert at + need <= n_samples, (
                f"switch at {at} of {n_samples} leaves "
                f"{(n_samples - at) / SR:.3f} s for a {need / SR:.1f} s "
                "guaranteed window")
            drawn[at] += 1
        # ⚠ The multiplicity, not the number of distinct values. Clamping the
        # draw to the limit instead of drawing inside it piles ~9% of the
        # switches on the single latest admissible sample and still leaves 91%
        # distinct, which a count of distinct values cannot see; every instant
        # here is drawn from an interval of ~10^5 samples, so the shipped draw
        # repeats one at most twice in a thousand.
        instant, times = drawn.most_common(1)[0]
        assert times <= 5, (
            f"{times} of 1000 switch points landed on sample {instant} "
            f"({instant / SR:.3f} s): the limit is being hit rather than "
            "drawn inside")


def test_a_switch_that_cannot_carry_its_window_renders_still(motion_corpus):
    """A guarantee that cannot be met is a downgrade, not a label.

    The sequence keeps its planned impairment -- comparing `path_motion` with
    `impairments` is how the corpus counts these -- but nothing in the audio is
    claimed: no switch chunk, no event sample, and one waypoint, because the
    second position is never rendered.
    """
    cfg = copy.deepcopy(motion_corpus['cfg'])
    # Longer than the sequence minus one chunk: no switch can be far enough
    # from the end and still leave a chunk of the earlier path in front of it.
    cfg.set('path_motion', 'far_active_after_event_sec', '29.0')
    with _without_linear_aec():
        rendered = _render_motion(motion_corpus, 'echo_path_change', cfg=cfg)

    meta = rendered.chunk_meta[0]
    assert meta['path_motion'] == STATIC_PATH
    assert not meta['echo_path_change']
    assert meta['echo_path_event_at'] == -1
    assert len(meta['echo_path_waypoints']) == 1
    assert 'echo_path_change' in meta['impairments']
    assert all(row['scenario'] != 'echo_path_change'
               for row in rendered.chunk_meta)


def test_merging_runs_keeps_one_that_starts_before_the_others():
    """The repaired run is appended to a list it does not sort into."""
    merge = aec_dataset_module._merge_runs
    assert merge([(5, 10), (0, 3)]) == [(0, 3), (5, 10)]
    assert merge([(5, 10), (0, 6)]) == [(0, 10)]
    assert merge([(0, 3), (3, 5), (9, 10)]) == [(0, 5), (9, 10)]


def test_a_dropout_shrinks_rather_than_covering_the_protected_stretch():
    """And it never silently gives up on the keep-out."""
    place = aec_dataset_module._dropout_placement
    chunk = 1000
    starts, count = place(10, 4, chunk, (2 * chunk, 4 * chunk))
    assert count == 4 and all(start > 3 for start in starts)
    # Only a shorter dropout fits beside the protected chunks -- before them
    # or after them, never over them.
    starts, count = place(6, 4, chunk, (2 * chunk, 4 * chunk))
    assert count == 2 and starts == [0, 4]
    # Nothing to protect: the drawn length is kept.
    starts, count = place(6, 4, chunk, None)
    assert count == 4 and starts == [0, 1, 2]
    # A window straddling the only interior boundary protects both chunks, so
    # nothing fits beside it. The drawn length is kept -- and the placement is
    # the one that zeroes the least of the window, which here is the chunk the
    # window barely reaches into rather than the one holding almost all of it.
    starts, count = place(2, 1, chunk, (chunk - 10, chunk + 390))
    assert count == 1 and starts == [0]
    starts, count = place(2, 1, chunk, (chunk - 390, chunk + 10))
    assert count == 1 and starts == [1]


def test_double_talk_movement_also_moves_the_near_talkers_path(motion_corpus,
                                                               motion_renders,
                                                               monkeypatch):
    """DT movement is a person shifting AND a loudspeaker moving.

    ⚠ Measured on the near-end AUDIO, not on the metadata flag. The regression
    this exists to catch is the renderer taking its static `near_rirs[0]`
    branch while the plan still says the near path moves, and every flag in
    the chunk metadata survives that. The comparison render forces exactly
    that branch -- the near trajectory is replaced by a still path AFTER it has
    been drawn, so both renders consume the same rng and differ in the near
    path alone.
    """
    def _static_near_path():
        original = AecSequenceRenderer._plan_path_motion
        drawn = []

        def _plan(self, *args, **kwargs):
            motion = original(self, *args, **kwargs)
            drawn.append(motion)
            # The near talker's own path is the SECOND draw of a DT sequence.
            return aec_dataset_module.PathMotion() if len(drawn) > 1 else motion
        return _plan

    with _without_linear_aec():
        moving = _render_motion(motion_corpus, 'movement',
                                talk_mode='double_talk')
        monkeypatch.setattr(AecSequenceRenderer, '_plan_path_motion',
                            _static_near_path())
        still_near = _render_motion(motion_corpus, 'movement',
                                    talk_mode='double_talk')

    assert moving.chunk_meta[0]['near_path_motion'] == 'movement'
    # No near talker means no near path to move, and no draws spent on one --
    # read off the far-only movement renders the fixture already made.
    assert (motion_renders['movement'][0].chunk_meta[0]['near_path_motion']
            == STATIC_PATH)

    # The still-near render's near_speech IS this same near-end speech through
    # ONE fixed position, so the path between the two renders is the moving
    # near path itself -- measurable with the same estimator the echo path's
    # calibration uses, and blind to the silence of a conversational schedule
    # and to any constant gain between the two renders.
    near = STEM_ORDER.index('near_speech')
    measured = path_drift_metrics(still_near.stems[near],
                                  moving.stems[near], SR)
    assert measured['correlation'][1.0] >= 0.9, (
        f"the near path is not a smooth trajectory: {measured['correlation']}")
    assert measured['correlation'][8.0] <= 0.9, (
        f"the near talker's path never moved: {measured['correlation']}")
    # ⚠ Not the near path's own correlation curve, and not comparable with the
    # echo path's: what is measured here is the RATIO of the two renders'
    # paths, so the fixed position's own spectral shape is divided out of it.
    # It answers "did this path move", which is the claim under test.


def test_shared_rir_normalisation_keeps_the_relative_position_gains(
        motion_corpus):
    """prepare_rir's per-file L2 normalisation erases exactly the level
    difference between two loudspeaker positions that movement is about."""
    pools = pools_for_split(motion_corpus['manifest'], UNIFIED_SPLIT)
    renderer = AecSequenceRenderer(motion_corpus['cfg'], pools,
                                   corpus_seed=SEED)
    room = pools.rooms[0]
    paths = pools.rirs_by_room[room][:3]
    assert len(paths) == 3

    prepared = renderer._load_rir_set(paths)
    raw = [torchaudio.load(path)[0][0] for path in paths]
    raw_ratio = [float(r.pow(2).mean().sqrt() / raw[0].pow(2).mean().sqrt())
                 for r in raw]
    kept_ratio = [float(full.pow(2).mean().sqrt()
                        / prepared[0][1].pow(2).mean().sqrt())
                  for _target, full in prepared]
    assert kept_ratio == pytest.approx(raw_ratio, rel=1e-4)
    assert max(raw_ratio) > 1.2, "the fixture must vary the position gains"

    # A one-position set is the anchor by definition, so a still sequence is
    # exactly what plain prepare_rir produces.
    alone = renderer._load_rir_set(paths[1:2])
    assert float(alone[0][1].pow(2).sum().sqrt()) == pytest.approx(1.0,
                                                                  rel=1e-5)


def _echo_lag(far, echo, window):
    """Cross-correlation lag of ``echo`` behind ``far`` over ``window``."""
    a = far[window]
    b = echo[window]
    size = 1 << (2 * a.shape[-1] - 1).bit_length()
    correlation = torch.fft.irfft(
        torch.fft.rfft(b, size) * torch.fft.rfft(a, size).conj(), size)
    return int(correlation[:a.shape[-1]].argmax())


def _alignment_score(far, echo, window, lag):
    """How well ``echo[window]`` matches the reference delayed by ``lag``."""
    reference = far[window.start - lag:window.stop - lag]
    observed = echo[window]
    scale = float(reference.pow(2).sum().sqrt() * observed.pow(2).sum().sqrt())
    return float((reference * observed).sum()) / scale if scale > 0 else 0.0


# The sliding measurement the recorded delay-step instant is checked against.
# ⚠ The window is what sets the measurement's error, not the grid step: the
# crossing is biased by how the reference's own energy is distributed inside
# the window, so a shorter window measures the instant more tightly. Measured
# over the analysable renders of the fixture below, |located - recorded| peaks
# at 1200 samples with a 0.2 s window and at 2080 with a 0.4 s one, while the
# grid step moves it by less than one step.
SWITCH_WINDOW = int(0.2 * SR)
SWITCH_STEP = int(0.01 * SR)
SWITCH_REACH = int(2.0 * SR)
# How far the located instant may sit from the recorded one. The error this
# bound has to reject is an instant reported on the reference timeline instead
# of the microphone's, which is exactly one bulk delay -- 6400-8000 samples at
# the range the test below pins -- so it sits at a third of that.
INSTANT_BOUND_SEC = 0.16
# The coarse pass that finds the step WITHOUT being told where it is: the
# echo's alignment over the whole render, 2 s at a time. Long enough that a
# conversational reference carries one lag estimate per window and short enough
# that the pair of windows straddling the step brackets it to the hop.
LAG_TRACK_WINDOW = int(2.0 * SR)
LAG_TRACK_HOP = int(0.5 * SR)


def _lag_track(far, echo):
    """``(centres, lags)`` of the echo's alignment across the whole render.

    Only windows whose reference actually carries energy: a lag measured
    through a silent stretch is the cross-correlation of noise and would put a
    step wherever the far end happens to pause.
    """
    active = 0.2 * float(far.pow(2).mean().sqrt())
    centres, lags = [], []
    for start in range(0, far.shape[-1] - LAG_TRACK_WINDOW, LAG_TRACK_HOP):
        window = slice(start, start + LAG_TRACK_WINDOW)
        if float(far[window].pow(2).mean().sqrt()) < active:
            continue
        centres.append(start + LAG_TRACK_WINDOW // 2)
        lags.append(_echo_lag(far, echo, window))
    return centres, lags


def _two_segment_split(lags):
    """Where a piecewise-constant track changes value.

    The absolute-deviation fit rather than a threshold on consecutive
    differences: one window straddling the step reads somewhere between the two
    lags, and differencing would place the change on whichever side that window
    happened to lean.
    """
    best, index = None, None
    for split in range(1, len(lags)):
        left = np.asarray(lags[:split], dtype=np.float64)
        right = np.asarray(lags[split:], dtype=np.float64)
        cost = (float(np.abs(left - np.median(left)).sum())
                + float(np.abs(right - np.median(right)).sum()))
        if best is None or cost < best:
            best, index = cost, split
    return index


def _reference_is_continuous(far, centres):
    """Is the reference loud enough AROUND these window centres?

    Asked at lag 0, i.e. over the window itself, while the alignment score
    reads the reference one bulk delay earlier -- so this is the reference's
    own continuity at the located instant, not a check of every sample the
    score touched.

    Separated from the locator itself so that "the reference has a gap here"
    and "the two alignments never cross" are different outcomes: the first is
    a render this measurement cannot be made on, the second is a failure.

    ⚠ Asked about the LOCATED instant, never about a recorded one. A reference
    gap at the step drags the crossing to the edge of the gap (measured: up to
    0.37 s), which is a measurement this render cannot carry; asking the
    question at the recorded instant instead would let a regression that moved
    that instant into a gap remove itself from the check.
    """
    quiet = 1e-3 * float(far.pow(2).mean().sqrt())
    return bool(centres) and all(
        float(far[centre - SWITCH_WINDOW // 2:
                  centre + SWITCH_WINDOW // 2].pow(2).mean().sqrt()) >= quiet
        for centre in centres)


def _alignment_switch(far, echo, before_lag, after_lag, centres):
    """Where the echo stops matching ``before_lag`` and starts matching
    ``after_lag``.

    A sliding window's two scores cross when the window straddles the step
    equally, i.e. at the step itself, whatever the window length -- so this
    locates the instant more tightly than the window it uses.

    ⚠ The sign is decided with hysteresis, from a fraction of the difference's
    own range rather than from zero: one window sitting on the step can wobble
    across zero and turn a single boundary into three crossings.

    ⚠ Returns ``None`` when the span holds no boundary to locate, which
    happens two ways: the difference never changes sign at all -- one alignment
    matching everywhere -- or its strong excursions run the wrong way round, no
    window past the last strongly negative one being strongly positive. Either
    means no instant explains the two lags better than its neighbours, which is
    what an instant reported far from where it happened looks like.
    """
    difference = []
    for centre in centres:
        span = slice(centre - SWITCH_WINDOW // 2, centre + SWITCH_WINDOW // 2)
        difference.append(_alignment_score(far, echo, span, after_lag)
                          - _alignment_score(far, echo, span, before_lag))
    if max(difference) <= 0 or min(difference) >= 0:
        return None
    strong = 0.25 * min(max(difference), -min(difference))
    negative = [index for index, value in enumerate(difference)
                if value <= -strong]
    positive = [index for index, value in enumerate(difference)
                if value >= strong and index > (max(negative) if negative
                                                else -1)]
    if not negative or not positive:
        return None
    return (centres[max(negative)] + centres[min(positive)]) / 2


_StepReading = collections.namedtuple('_StepReading',
                                      'before after instant why')


def _locate_delay_step(far, echo):
    """What the AUDIO says the step was and when it happened.

    Returns a ``_StepReading``: ``before``/``after`` are the lags on either
    side (``None`` when the render carries no measurable step at all),
    ``instant`` is the located crossing (``None`` when it cannot be located)
    and ``why`` names the path taken, so a test can pin WHICH renders are
    analysable and for what reason rather than counting them.

    ⚠ Nothing here is told where the step was recorded. The coarse pass finds
    it in the alignment track over the WHOLE render and the fine pass refines
    it inside that bracket, so a render whose recorded instant is wrong is
    still measured and still compared -- rather than dropping out of the check
    because a filter evaluated at the wrong instant found no reference there.
    """
    centres, lags = _lag_track(far, echo)
    if len(lags) < 4:
        return _StepReading(None, None, None, 'track-too-short')
    split = _two_segment_split(lags)
    before = int(np.median(lags[:split]))
    after = int(np.median(lags[split:]))
    # Half the smallest step the config admits: below that the two segments are
    # one lag with the fit's own wobble on it, not a step.
    if abs(after - before) < int(0.01 * SR):
        return _StepReading(None, None, None, 'no-step-in-the-track')
    coarse = 0.5 * (centres[split - 1] + centres[split])
    low = int(max(max(before, after) + SWITCH_WINDOW // 2,
                  coarse - SWITCH_REACH))
    high = int(min(far.shape[-1] - SWITCH_WINDOW // 2, coarse + SWITCH_REACH))
    if high - low < 4 * SWITCH_STEP:
        return _StepReading(before, after, None, 'no-bracket')
    search = list(range(low, high, SWITCH_STEP))
    instant = _alignment_switch(far, echo, before, after, search)
    if instant is None:
        return _StepReading(before, after, None, 'no-crossing')
    return (_StepReading(before, after, instant, 'located')
            if _reference_is_continuous(
                far, [centre for centre in search
                      if abs(centre - instant) <= int(0.3 * SR)])
            else _StepReading(before, after, None, 'reference-gap'))


# What each render of the fixture below does, by index, taken from the
# locator's own reason for each of the 16. The fixture is deterministic, so
# this is the complete inventory rather than a sample.
#   'located'             the crossing was found in the audio and compared
#   'reference-gap'       the reference is silent around the crossing, so the
#                         crossing is only defined to the edge of the gap
#   'no-crossing'         no instant explains the two lags better than its
#                         neighbours. On this fixture the one render in this
#                         state has the strong excursions the wrong way round
#                         -- 45 windows strongly favour the pre-step alignment
#                         and none after the last of them favours the post-step
#                         one -- rather than two alignments that never separate
#   'no-step-in-the-track' the step lands too close to an edge for a 2 s window
#                         to sit on one side of it, so the lag track carries
#                         one segment
INSTANT_RENDERS = {
    0: 'located', 1: 'located', 2: 'no-step-in-the-track',
    3: 'located', 4: 'located', 5: 'no-crossing', 6: 'located',
    7: 'reference-gap', 8: 'located', 9: 'located',
    10: 'reference-gap', 11: 'reference-gap', 12: 'located', 13: 'located',
    14: 'no-step-in-the-track', 15: 'located',
}


def test_delay_step_shifts_the_echo_by_the_recorded_amount(delay_corpus):
    """The impairment's own metadata has to describe the audio.

    ⚠ Both halves of it. Comparing the lag long before the step with the lag
    long after verifies the SIZE and would pass unchanged with the recorded
    instant off by a second, so the instant is located independently -- from
    the audio alone, over the whole render -- and only then compared with
    `delay_step_at`.

    ⚠ Nothing about which renders are checked may depend on the value under
    test. Deciding analysability from the RECORDED instant (is it far enough
    from an edge, does the reference carry it) turns a regression that moves
    one instant into a silent skip: the render leaves the check instead of
    failing it, and a floor on the count cannot see the difference between "two
    renders were unanalysable" and "two renders were wrong". Here the coarse
    pass finds the step in the alignment track over the whole render, the fine
    pass refines it, and the reference is asked about the LOCATED instant.

    ⚠ The bulk delay is pinned LARGE (0.4-0.5 s) on purpose. The error this
    bound has to reject is an instant reported on the reference timeline
    instead of the microphone's, which is exactly one bulk delay -- 6400-8000
    samples here -- so INSTANT_BOUND_SEC sits at 0.16 s: twice the worst
    measurement error on an unmutated render (peak 1261 samples over the
    located renders) and a third of the error it discriminates. The range is a
    RANGE because a negative step is drawn from the headroom above
    `bulk_delay_ms_min`: with both ends pinned there is no headroom and every
    drawn step is positive, so half the impairment would go unexercised
    (measured here: 8 positive, 8 negative).

    ⚠ Which renders are analysable is a fixed, deterministic set, and it is
    asserted as one rather than as a floor: with a floor, a regression that
    takes exactly one render out of the check is silent. INSTANT_RENDERS says
    what the 16 renders of this fixture do, and any change to that fails here
    with the reason attached.

    ⚠ The recorded instant of a render that is not located is checked by
    nothing, here or anywhere: locating it from the audio is the only
    independent evidence there is, and where the audio cannot answer, the
    metadata cannot be contradicted. That is why the set is pinned -- the
    number of renders in that position must not grow quietly.
    """
    cfg = copy.deepcopy(delay_corpus['cfg'])
    cfg.set('echo_path', 'bulk_delay_ms_min', '400')
    cfg.set('echo_path', 'bulk_delay_ms_max', '500')

    with _without_linear_aec():
        reasons = {}
        signs = collections.Counter()
        for index in range(len(INSTANT_RENDERS)):
            rendered = _render_motion(delay_corpus, 'delay_step',
                                      index=index, cfg=cfg)
            meta = rendered.chunk_meta[0]
            step = meta['delay_step_samples']
            at = meta['delay_step_at']
            assert step != 0 and at > 0
            signs['negative' if step < 0 else 'positive'] += 1
            far, echo = _far_echo(rendered)
            reading = _locate_delay_step(far, echo)
            reasons[index] = reading.why
            if reading.before is None:
                continue
            assert reading.after - reading.before == pytest.approx(
                step, abs=int(0.002 * SR)), (
                f"render {index}: lag moved {reading.after - reading.before} "
                f"samples, recorded {step}")
            if reading.instant is None:
                continue
            assert abs(reading.instant - at) <= INSTANT_BOUND_SEC * SR, (
                f"render {index}: the echo's alignment switches at "
                f"{reading.instant / SR:.2f} s, recorded {at / SR:.2f} s")
    assert reasons == INSTANT_RENDERS, (
        "the analysable renders changed: "
        f"{ {index: why for index, why in reasons.items() if INSTANT_RENDERS[index] != why} }")
    assert signs['negative'] and signs['positive'], (
        f"the drawn steps are one-signed ({dict(signs)}), so the negative "
        "branch of the draw goes unexercised")


def test_a_jittered_delay_keeps_the_echo_behind_its_reference(delay_corpus):
    """The floor is on the delay the microphone actually receives.

    Two independent impairments move the same bulk delay: ``delay_step`` is
    baked into the played signal first and ``delay_jitter`` then walks it, from
    the PRE-step value. A walk clamped at zero therefore renders below the
    configured floor in two different ways -- on its own, by walking a short
    bulk delay down to nothing, and after a NEGATIVE step, where the net delay
    is the walk minus the step and goes negative outright, an echo arriving
    before the reference that caused it. Inside the frozen matched filter's
    first bin the delay estimate stops moving, so either one costs the rest of
    the sequence.

    ⚠ Measured from the audio, over 4 s windows. The composition is what the
    microphone hears and neither impairment's own draw shows it; the windows
    are long because the far end is conversational and the same speech files
    recur, so a 1 s window's cross-correlation can lock onto a repeat instead
    of the alignment.

    ⚠ The seeds are CHOSEN, and they cover both routes: 13, 22 and 23 draw a
    negative step, 30 draws a positive one and reaches the floor through the
    walk alone. With the walk clamped at zero instead, 13/23/30 render the echo
    137/104/81 samples behind a reference it should be at least 160 behind.
    """
    cfg = copy.deepcopy(delay_corpus['cfg'])
    cfg.set('echo_path', 'bulk_delay_ms_min', '10')
    cfg.set('echo_path', 'bulk_delay_ms_max', '60')
    floor = int(SR * cfg.getfloat('echo_path', 'bulk_delay_ms_min') / 1000)
    tolerance = int(0.001 * SR)

    renderer = AecSequenceRenderer(
        cfg, pools_for_split(delay_corpus['manifest'], UNIFIED_SPLIT),
        corpus_seed=SEED)
    chunk_samples = chunk_samples_from_config(
        cfg, linear_aec_contract_from_config(cfg).hop_size)
    n_chunks = int(cfg.getfloat('sequence', 'seq_sec_max') * SR) // chunk_samples
    span = 4 * SR

    negatives = 0
    with _without_linear_aec():
        for index in (0, 1, 13, 22, 23, 30):
            rendered = renderer.render(SequencePlan(
                sequence_id=index, n_chunks=n_chunks, scenario='far_only',
                seed=stable_seed(SEED, 'jitter-step', index),
                talk_mode='far_only', echo_mode='normal',
                impairments=('delay_step', 'delay_jitter')))
            meta = rendered.chunk_meta[0]
            negatives += meta['delay_step_samples'] < 0
            far, echo = _far_echo(rendered)
            active = 0.3 * float(far.pow(2).mean().sqrt())
            measured = 0
            for start in range(0, far.shape[-1] - span, span // 2):
                window = slice(start, start + span)
                if float(far[window].pow(2).mean().sqrt()) < active:
                    continue
                lag = _echo_lag(far, echo, window)
                measured += 1
                assert lag >= floor - tolerance, (
                    f"the echo is {lag} samples behind its reference over "
                    f"[{start / SR:.1f}, {(start + span) / SR:.1f}] s, inside "
                    f"the {floor}-sample floor; step "
                    f"{meta['delay_step_samples']}, bulk "
                    f"{meta['bulk_delay_samples']}")
            assert measured >= 8, (
                f"only {measured} window(s) of sequence {index} carried enough "
                "reference to measure a lag in")
    assert negatives, "no sequence drew a negative step; the test proves nothing"


def test_the_jitter_floor_is_raised_only_where_the_step_is(delay_corpus):
    """The floor follows the step into the signal, it does not precede it.

    A negative ``delay_step`` raises the walk's lower clamp by its own size,
    because after the step the net delay is the walk minus the step. Before the
    step instant there is no step in the signal: clamping there too would push
    the whole first part of every walk up by as much as 150 ms and make it
    one-sided -- a walk that can only go up is not a playout buffer glitch,
    and it would put a floor in the delay distribution that the config does not
    describe.

    Measured on the delayed audio rather than on the draw, because what has to
    hold is a property of the rendered signal: the walk stays above the plain
    floor everywhere, above the RAISED floor from the step onward, and reaches
    below the raised floor before it.
    """
    cfg = delay_corpus['cfg']
    generator = torch.Generator().manual_seed(11)
    reference = torch.randn(30 * SR, generator=generator)
    base = int(0.12 * SR)
    step = -int(0.1 * SR)
    step_at = 20 * SR
    floor = int(SR * cfg.getfloat('echo_path', 'bulk_delay_ms_min') / 1000)
    raised = floor - step
    fade = int(SR * cfg.getfloat('echo_path', 'jitter_fade_sec'))
    span = SR // 2

    lowest_before = raised
    for trial in range(40):
        rng = random.Random(stable_seed(SEED, 'jitter-floor', trial))
        walked = aec_dataset_module._apply_jittered_delay(
            reference, base, SR, rng, cfg, applied_step=step, step_at=step_at)
        for start in range(0, reference.shape[-1] - span, span):
            window = slice(start, start + span)
            lag = _echo_lag(reference, walked, window)
            assert lag >= floor, (
                f"the walk renders {lag} samples of delay over "
                f"[{start / SR:.1f}, {(start + span) / SR:.1f}] s, inside the "
                f"{floor}-sample floor")
            if start + span <= step_at:
                lowest_before = min(lowest_before, lag)
            elif start >= step_at + fade:
                assert lag >= raised, (
                    f"the walk renders {lag} samples after the step, so the "
                    f"net delay is {lag + step}, inside the {floor}-sample "
                    "floor")
    assert lowest_before < raised, (
        f"the lowest delay any walk reached before the step instant is "
        f"{lowest_before}, at or above the {raised}-sample floor that only "
        "applies once the step is in the signal: the walk is one-sided there")


def test_a_negative_delay_step_keeps_the_echo_inside_the_filter_reach(
        motion_corpus):
    """A step that lands the echo at 0 ms is not a shorter delay, it is a
    sequence whose linear error is the uncancelled capture.

    The frozen matched filter cannot resolve a delay inside its first bin, so
    it holds the pre-step estimate for the rest of the sequence. Clipping a
    negative step to the bulk delay does exactly that -- and, on the way,
    records steps below delay_step_ms_min while the metadata still says
    delay_step, which is indistinguishable from delay_jitter.
    """
    cfg = copy.deepcopy(motion_corpus['cfg'])
    cfg.set('echo_path', 'bulk_delay_ms_min', '10')
    cfg.set('echo_path', 'bulk_delay_ms_max', '40')
    # Read back, not restated: the floor the draw has to respect is the config
    # key, so moving that key moves what this test checks.
    floor = int(SR * cfg.getfloat('echo_path', 'bulk_delay_ms_min') / 1000)
    minimum = int(SR * cfg.getfloat('echo_path', 'delay_step_ms_min') / 1000)
    maximum = int(SR * cfg.getfloat('echo_path', 'delay_step_ms_max') / 1000)

    negatives = 0
    for trial in range(400):
        rng = random.Random(stable_seed(SEED, 'delay-step', trial))
        bulk = rng.randint(
            floor, int(SR * cfg.getfloat('echo_path', 'bulk_delay_ms_max')
                       / 1000))
        step, at = aec_dataset_module._draw_delay_step(
            30 * SR, bulk, SR, rng, cfg)
        assert minimum <= abs(step) <= maximum, (step, bulk)
        assert bulk + step >= floor, (step, bulk)
        margin = int(SR * DELAY_STEP_EDGE_MARGIN_SEC)
        assert margin <= at < 30 * SR - margin
        negatives += step < 0
    assert negatives, "the sign is never negative; the test proves nothing"


def test_a_moving_double_talk_render_is_reproducible(motion_corpus):
    """Every new draw -- trajectory, level walk, near path -- is seeded.

    A drifting sequence adds three rng consumers to a render that a corpus's
    --resume, its re-materialisation and every A/B against it all assume is a
    pure function of the sequence seed.
    """
    with _without_linear_aec():
        first = _render_motion(motion_corpus, 'movement',
                               talk_mode='double_talk')
        second = _render_motion(motion_corpus, 'movement',
                                talk_mode='double_talk')

    assert torch.equal(first.stems, second.stems)
    assert first.chunk_meta == second.chunk_meta
    for key in ('echo', 'echo_path_weights', 'echo_path_gain_db'):
        assert torch.equal(first.audit[key], second.audit[key]), key


def test_an_event_label_yields_to_a_dropout_and_a_switch_keeps_its_own(
        motion_corpus):
    """The three per-chunk event labels have a precedence and it is load-bearing.

    A zeroed reference inside a drifting sequence is a dropout chunk first --
    labelling it 'echo_path_moving' would put a chunk with no reference at all
    into the moving class and take it out of the idle supervision the dropout
    exists to provide. The one switch chunk of a one-shot change keeps its own
    label for the same reason.
    """
    with _without_linear_aec():
        drifting = _render_motion(motion_corpus, 'slow_drift', index=1,
                                  echo_mode='ref_dropout')
        switched = _render_motion(motion_corpus, 'echo_path_change', index=1)

    far, _echo = _far_echo(drifting)
    schedule = moving_chunks(
        drifting.audit['echo_path_weights'],
        drifting.chunk_samples, len(drifting.chunk_meta),
        motion_corpus['cfg'].getfloat('path_motion',
                                      'moving_label_weight_delta'))
    dropped = [meta['chunk_index'] for meta in drifting.chunk_meta
               if meta['scenario'] == 'ref_dropout']
    assert dropped
    for index in dropped:
        assert float(far[_chunk_window(drifting, index)].abs().max()) == 0.0
        assert not drifting.chunk_meta[index]['echo_path_moving']
    assert set(dropped) & schedule, (
        "no dropout chunk had a moving schedule; the precedence is untested")
    assert any(meta['scenario'] == 'echo_path_moving'
               for meta in drifting.chunk_meta), (
        "the sequence lost its moving chunks entirely")

    switch = (switched.chunk_meta[0]['echo_path_event_at']
              // switched.chunk_samples)
    assert switched.chunk_meta[switch]['scenario'] == 'echo_path_change'
    assert switched.chunk_meta[switch]['echo_path_change']


def _fixed_path_pair(seconds, rir, generator, level_db_per_sec=0.0):
    """A far/echo pair whose path is one fixed RIR, at an optional level ramp."""
    far = _speechlike(int(seconds * SR), generator)
    echo = aec_dataset_module.fftconvolve(far, rir)[:far.shape[-1]]
    if level_db_per_sec:
        ramp = level_db_per_sec * torch.arange(far.shape[-1]) / SR
        echo = echo * torch.pow(10.0, ramp / 20.0)
    return far, echo


def test_the_level_step_reads_a_pure_level_ramp_in_db(motion_corpus):
    """The level axis has to be measured in the units it is configured in."""
    generator = torch.Generator().manual_seed(5)
    rir = _rir(256, 0.05, generator, tail=CALIBRATION_TAIL)
    far, echo = _fixed_path_pair(20.0, rir, generator, level_db_per_sec=1.0)

    measured = path_drift_metrics(far, echo, SR)
    assert measured['gain_step_db'] == pytest.approx(1.0, abs=0.15)
    for lag in DEFAULT_LAGS_SEC:
        assert measured['correlation'][lag] >= 0.99, measured['correlation']


def test_the_level_step_counts_shape_change_the_way_the_reference_does():
    """``a = <H1,H2>/<H1,H1>`` in dB, not the ratio of the two norms.

    ⚠ The two are different statistics and the calibration targets were
    produced with the first. Two paths of equal norm and different shape have
    a norm ratio of exactly 1 -- 0 dB -- while |a| falls with their
    correlation, so a generator tuned to close the gap on the norm ratio would
    over-drive its level walk by everything the shape change contributes.
    """
    generator = torch.Generator().manual_seed(11)
    # Two INDEPENDENT paths, without the shared direct impulse _rir
    # plants: a common direct path would keep the two shapes correlated and
    # the level statistic would have nothing to disagree about.
    decay = torch.exp(-6.9078 * torch.arange(256) / SR / 0.05)
    first = torch.randn(256, generator=generator) * decay
    second = torch.randn(256, generator=generator) * decay
    second = second * float(first.pow(2).sum().sqrt()
                            / second.pow(2).sum().sqrt())
    far = _speechlike(24 * SR, generator)
    one = aec_dataset_module.fftconvolve(far, first)[:far.shape[-1]]
    other = aec_dataset_module.fftconvolve(far, second)[:far.shape[-1]]
    # Alternate the path every 2 s -- the estimator's own window -- so
    # consecutive windows see genuinely different shapes at the same level.
    select = ((torch.arange(far.shape[-1]) // (2 * SR)) % 2).float()
    echo = one * (1.0 - select) + other * select

    measured = path_drift_metrics(far, echo, SR)
    track, valid, step_sec = estimate_path_track(far, echo, SR)
    assert step_sec == 1.0
    norms = np.linalg.norm(track, axis=1)
    usable = np.where(valid & (norms > 0))[0]
    pairs = [(i, i + 1) for i in usable[:-1] if i + 1 in set(usable)]
    ratio_db = float(np.median([abs(20 * np.log10(norms[b] / norms[a]))
                                for a, b in pairs]))
    assert measured['correlation'][1.0] < 0.80, measured['correlation']
    assert measured['gain_step_db'] > ratio_db + 1.0, (
        f"level step {measured['gain_step_db']:.2f} dB vs norm ratio "
        f"{ratio_db:.2f} dB: the shape change is not being counted")


def test_delay_step_is_drawn_independently_of_path_motion(corpus):
    """A device event and a talker moving are different things.

    Delay jumps > 20 ms appear in 53% of moving and 63% of static captures, so
    a corpus in which the two co-occur would let a model infer either from the
    other.
    """
    cfg = copy.deepcopy(corpus['cfg'])
    for name in ('far_only', 'near_only', 'duplex_random'):
        cfg.set('talk_modes', f'p_{name}', '0')
    cfg.set('talk_modes', 'p_double_talk', '1')
    cfg.set('echo_modes', 'p_ref_dropout', '0')
    cfg.set('echo_modes', 'p_far_active_no_echo', '0')
    cfg.set('complex_cases', 'p_dt_stress_combo', '0')
    _only_impairments(cfg, delay_step='0.5', movement='0.5')

    seen = {(path_motion_mode(resolve_sequence_plan(plan)[2]),
             'delay_step' in resolve_sequence_plan(plan)[2])
            for plan in plan_sequences(cfg, 0.5, SEED, 'train')}
    assert seen == {(STATIC_PATH, False), (STATIC_PATH, True),
                    ('movement', False), ('movement', True)}, sorted(seen)


def test_path_motion_modes_are_one_mutually_exclusive_draw(corpus):
    """A path that both drifts and jumps describes no device."""
    cfg = copy.deepcopy(corpus['cfg'])
    # Near-only and no-echo sequences have no path to move, so the motion draw
    # is stripped from them; this test is about the draw itself.
    for name in ('far_only', 'near_only', 'duplex_random'):
        cfg.set('talk_modes', f'p_{name}', '0')
    cfg.set('talk_modes', 'p_double_talk', '1')
    cfg.set('echo_modes', 'p_ref_dropout', '0')
    cfg.set('echo_modes', 'p_far_active_no_echo', '0')
    _only_impairments(cfg, slow_drift='0.4', movement='0.3',
                      echo_path_change='0.3')

    modes = collections.Counter()
    for plan in plan_sequences(cfg, 0.3, SEED, 'train'):
        impairments = set(resolve_sequence_plan(plan)[2])
        assert len(impairments & set(PATH_MOTION_MODES)) <= 1, impairments
        modes[path_motion_mode(impairments)] += 1
    assert set(modes) == set(PATH_MOTION_MODES), modes
    assert modes[STATIC_PATH] == 0, "the three probabilities sum to 1"

    cfg.set('impairments', 'p_slow_drift', '0.8')
    with pytest.raises(ValueError, match='mutually exclusive draw'):
        plan_sequences(cfg, 0.001, SEED, 'train')


def test_pure_double_talk_movement_is_not_a_stress_combo(corpus):
    """Movement and the nonlinear/clipping bundle have to be countable apart."""
    cfg = copy.deepcopy(corpus['cfg'])
    for name in ('far_only', 'near_only', 'duplex_random'):
        cfg.set('talk_modes', f'p_{name}', '0')
    cfg.set('talk_modes', 'p_double_talk', '1')
    cfg.set('echo_modes', 'p_ref_dropout', '0')
    cfg.set('echo_modes', 'p_far_active_no_echo', '0')
    _only_impairments(cfg, movement='1')
    cfg.set('complex_cases', 'p_dt_stress_combo', '0')

    for plan in plan_sequences(cfg, 0.01, SEED, 'train'):
        assert set(resolve_sequence_plan(plan)[2]) == {'movement'}


def test_a_drift_plan_needs_a_room_with_enough_positions(tmp_path):
    """A sparse RIR pool must fail loudly, not render a shorter trajectory."""
    cfg, manifest = _sparse_rir_manifest(tmp_path, {'room_00': 2, 'room_01': 2})
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(manifest, UNIFIED_SPLIT), corpus_seed=SEED)
    with pytest.raises(RuntimeError, match=r'>= 3 RIR files'):
        renderer.render(SequencePlan(
            sequence_id=0, n_chunks=3, scenario='movement',
            seed=stable_seed(SEED, 'test', 'drift-sparse')))


def test_delay_step_range_is_kept_inside_the_frozen_matched_filter_reach(
        corpus):
    """The step stacks on the long-delay tail and the jitter, so it counts."""
    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('echo_path', 'delay_step_ms_max', '200')
    with pytest.raises(ValueError, match='worst-case long delay'):
        plan_sequences(cfg, 0.001, SEED, 'train')


# One value per check kind that the kind must refuse: a floor above its own
# ceiling, a ceiling below its own floor, a probability outside [0, 1], and so
# on. The point is coverage, not the message -- what is asserted is that the
# refusal NAMES the key, which is what makes it fixable.
_REFUSED_VALUE = {
    'positive_range': '-1',
    'range': '999999',
    'paired': '-1',
    'whole_range': '999999',
    'probability': '7',
    'open_unit': '1.0',
    'open_closed_unit': '2.0',
    'positive': '0',
    'non_negative': '-1',
    'waypoints': '1',
}


def test_the_planner_refuses_an_unrenderable_motion_model(corpus):
    """[path_motion] is read in a WORKER, so it has to be checked at plan time.

    A missing key or an out-of-range value otherwise survives the plan, the
    manifest and however many hours of rendering it takes for one sequence to
    draw path motion -- the exact failure the layered config's other
    preflight checks exist to prevent.
    """
    for section, option in (('path_motion', 'waypoints_min'),
                            ('path_motion', 'movement_position_correlation'),
                            ('path_motion', 'far_active_after_event_sec'),
                            ('echo_path', 'delay_step_ms_max')):
        cfg = copy.deepcopy(corpus['cfg'])
        assert cfg.remove_option(section, option)
        with pytest.raises(ValueError, match=re.escape(f'[{section}] {option}')):
            plan_sequences(cfg, 0.001, SEED, 'train')

    cfg = copy.deepcopy(corpus['cfg'])
    cfg.remove_section('path_motion')
    with pytest.raises(ValueError, match=re.escape('[path_motion]')):
        plan_sequences(cfg, 0.001, SEED, 'train')

    for option, value, message in (
            ('waypoints_min', '5', 'waypoints_min <= waypoints_max'),
            ('waypoints_min', '1', 'waypoints_min <= waypoints_max'),
            # A count that is not a count reports like every other bad value,
            # not with int()'s own message, which names nothing.
            ('waypoints_min', '2.5', '[path_motion] waypoints_min'),
            ('movement_position_correlation', '1.5',
             'movement_position_correlation'),
            # Both ends of the open interval: 1.0 is a path that never moves
            # and 0 is below what any mixture of one room's positions reaches,
            # so neither is a target the solver may be handed.
            ('movement_position_correlation', '1.0',
             'movement_position_correlation must be inside (0, 1)'),
            ('slow_drift_position_correlation', '0',
             'slow_drift_position_correlation must be inside (0, 1)'),
            ('slow_drift_dwell_p', '7', 'slow_drift_dwell_p'),
            ('movement_segment_sec_min', '0', 'movement_segment_sec_min'),
            ('slow_drift_segment_sec_min', '20', 'slow_drift_segment_sec_min'),
            ('gain_update_sec', '0', 'gain_update_sec'),
            ('gain_recentre_sec', '-1', 'gain_recentre_sec'),
            ('near_slowdown', '0', 'near_slowdown'),
            ('moving_label_weight_delta', '0', 'moving_label_weight_delta'),
    ):
        cfg = copy.deepcopy(corpus['cfg'])
        cfg.set('path_motion', option, value)
        with pytest.raises(ValueError, match=re.escape(message)):
            plan_sequences(cfg, 0.001, SEED, 'train')

    # And every OTHER key of both sections, driven off the spec tables the
    # validator itself reads. The cases above are the interesting messages,
    # hand-written; this is the statement that no key is merely LISTED -- a
    # spec added without a check, or a check that names a key the table does
    # not carry, leaves a value a worker reads and nothing refuses.
    for section, specs in (
            ('path_motion', aec_dataset_module.DRIFT_MODE_OPTION_SPECS),
            ('path_motion', aec_dataset_module.SHARED_PATH_MOTION_SPECS),
            ('echo_path', aec_dataset_module.ECHO_PATH_OPTION_SPECS)):
        drift = specs is aec_dataset_module.DRIFT_MODE_OPTION_SPECS
        for prefix in (sorted(aec_dataset_module.DRIFT_MODES) if drift
                       else ('',)):
            for stem, spec in specs.items():
                if spec.check is None:
                    continue
                key = f'{prefix}_{stem}' if prefix else stem
                cfg = copy.deepcopy(corpus['cfg'])
                cfg.set(section, key, _REFUSED_VALUE[spec.check])
                with pytest.raises(ValueError, match=re.escape(key)):
                    plan_sequences(cfg, 0.001, SEED, 'train')

    # A key that is not part of the trajectory model describes a DIFFERENT
    # model and reads as a plausible number in the same range, so it is named
    # rather than ignored -- ignoring it would render a corpus whose movement
    # axis is not the one its config asks for.
    for mode, stem in itertools.product(
            sorted(aec_dataset_module.DRIFT_MODES),
            aec_dataset_module.RETIRED_DRIFT_MODE_OPTION_STEMS):
        cfg = copy.deepcopy(corpus['cfg'])
        cfg.set('path_motion', f'{mode}_{stem}', '0.7')
        with pytest.raises(ValueError,
                           match=re.escape(f'[path_motion] {mode}_{stem}')):
            plan_sequences(cfg, 0.001, SEED, 'train')


def test_the_planner_refuses_a_config_missing_any_echo_path_key(corpus):
    """[echo_path] carries no planner probability, so nothing else looks at it.

    Every key in it is read inside a render WORKER -- the bulk delay for every
    far-capable sequence, the crossfade for a one-shot switch, the step and the
    jitter walk for their impairments -- and a key omitted from the section
    survives the plan and the manifest and kills the first worker that draws
    the impairment reading it, which for the shipped probabilities is a few per
    cent of the way into a multi-hour run.
    """
    for option in aec_dataset_module.ECHO_PATH_OPTIONS:
        cfg = copy.deepcopy(corpus['cfg'])
        assert cfg.remove_option('echo_path', option), option
        with pytest.raises(ValueError,
                           match=re.escape(f'[echo_path] {option}')):
            plan_sequences(cfg, 0.001, SEED, 'train')

    cfg = copy.deepcopy(corpus['cfg'])
    cfg.set('echo_path', 'delay_step_ms_min',
            cfg.get('echo_path', 'jitter_ms_max'))
    with pytest.raises(ValueError, match='must be > jitter_ms_max'):
        plan_sequences(cfg, 0.001, SEED, 'train')

    for option in ('jitter_steps_min', 'jitter_ms_min'):
        cfg = copy.deepcopy(corpus['cfg'])
        cfg.set('echo_path', option, '0')
        with pytest.raises(ValueError, match=re.escape(f'{option} must be > 0')):
            plan_sequences(cfg, 0.001, SEED, 'train')


class _ReadLoggingConfig(configparser.ConfigParser):
    """A config that remembers which options were actually read from it.

    Every typed accessor funnels through ``get``, so recording there catches
    ``getfloat``/``getint``/``getboolean`` as well.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reads = set()

    def get(self, section, option, **kwargs):        # noqa: A003
        self.reads.add((section, option))
        return super().get(section, option, **kwargs)


def test_the_presence_contract_names_every_key_a_worker_reads(motion_corpus):
    """The contract has to come from the RENDERER, not from itself.

    The two tests above delete each key the contract lists and check the
    planner refuses -- which is a statement about the keys already in it. The
    defect they cannot see is the opposite one: a key the renderer reads that
    nobody added to the list, which is precisely how a worker comes to die
    hours into a run. So this drives the renderer through every impairment with
    a config that records what it was asked for, and requires the contract to
    cover it.
    """
    cfg = _ReadLoggingConfig()
    cfg.read_dict(motion_corpus['cfg'])
    corpus = {'cfg': cfg, 'manifest': motion_corpus['manifest']}
    # ⚠ ONE renderer, and the log is cleared after it exists. Constructing a
    # renderer hashes the config, which reads every key of every section: with
    # a renderer per render the log is saturated before a single sequence is
    # drawn, and `read <= declared` degenerates into "the example config's keys
    # are all declared" -- true whatever the workers read.
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(motion_corpus['manifest'], UNIFIED_SPLIT),
        corpus_seed=SEED)
    cfg.reads.clear()
    with _without_linear_aec():
        for index, impairment in enumerate(IMPAIRMENTS):
            _render_motion(corpus, impairment, index=index, cfg=cfg,
                           talk_mode='double_talk', renderer=renderer)
        for index, echo_mode in enumerate(('normal', 'ref_dropout',
                                           'far_active_no_echo')):
            _render_motion(corpus, STATIC_PATH, index=index, cfg=cfg,
                           talk_mode='duplex_random', echo_mode=echo_mode,
                           renderer=renderer)

    for section, declared in (
            ('echo_path', aec_dataset_module.ECHO_PATH_OPTIONS),
            ('path_motion', aec_dataset_module.PATH_MOTION_OPTIONS)):
        read = {option for read_section, option in cfg.reads
                if read_section == section}
        assert read, f"the sweep never read [{section}] at all"
        assert read <= set(declared), (
            f"a render worker reads [{section}] "
            f"{sorted(read - set(declared))}, which the planner does not "
            "check for; a config missing it plans a whole corpus and then "
            "kills the worker that draws it")


def test_the_compatibility_planner_checks_the_same_ranges(corpus):
    """The legacy [scenarios] branch plans sequences a worker renders too.

    It is the branch a config copied from an older tree lands on, so it is
    exactly where an unrenderable trajectory model is most likely to arrive --
    and it reads the same [path_motion] and [echo_path] values in the same
    worker.
    """
    cfg = copy.deepcopy(corpus['cfg'])
    for section in ('talk_modes', 'echo_modes', 'impairments',
                    'acoustic_tails', 'complex_cases'):
        cfg.remove_section(section)
    cfg.add_section('scenarios')
    cfg.set('scenarios', 'p_movement', '1')
    assert plan_sequences(cfg, 0.001, SEED, 'train'), (
        'the compatibility branch must still plan a legacy config')

    for section, option, value, message in (
            ('path_motion', 'movement_position_correlation', '1.5',
             '[path_motion] movement_position_correlation'),
            ('path_motion', 'waypoints_min', '9', 'waypoints_min <= waypoints_max'),
            ('echo_path', 'delay_step_ms_min', '999',
             'delay_step_ms_min <= delay_step_ms_max'),
    ):
        broken = copy.deepcopy(cfg)
        broken.set(section, option, value)
        with pytest.raises(ValueError, match=re.escape(message)):
            plan_sequences(broken, 0.001, SEED, 'train')

    # The PRESENCE half, on this branch too. Two of these keys are read by no
    # validator -- only by a worker, at the moment it renders a crossfade -- so
    # a legacy config missing one plans a whole corpus and dies hours later,
    # which is exactly the failure the contract exists to remove.
    for section, options in (('echo_path', aec_dataset_module.ECHO_PATH_OPTIONS),
                             ('path_motion',
                              aec_dataset_module.PATH_MOTION_OPTIONS)):
        for option in options:
            broken = copy.deepcopy(cfg)
            assert broken.remove_option(section, option), option
            with pytest.raises(ValueError,
                               match=re.escape(f'[{section}] {option}')):
                plan_sequences(broken, 0.001, SEED, 'train')

    # And the section itself: an older config predating either of them must be
    # told which section to copy, not handed configparser's NoSectionError from
    # somewhere inside the validator.
    for section in ('echo_path', 'path_motion'):
        broken = copy.deepcopy(cfg)
        assert broken.remove_section(section)
        with pytest.raises(ValueError,
                           match=re.escape(f'section [{section}] is required')):
            plan_sequences(broken, 0.001, SEED, 'train')


def test_a_moving_sequence_is_drawn_from_the_rooms_that_can_hold_it(tmp_path):
    """The room draw pays for the trajectory, not the other way round.

    Every waypoint of a trajectory is in the SAME room (a waypoint elsewhere
    would leave the near talker behind in the old room -- the acoustic "this is
    echo" leak the same-room invariant exists to prevent), so a room with fewer
    than `waypoints_min` positions cannot host one. A sequence that draws such
    a room draws again among the rooms that can, which is what keeps the
    rendered motion share equal to the planned one.

    ⚠ The cue that leaves is one-sided and inherent: moving sequences come only
    from rooms rich enough to hold a trajectory, and no draw rule can change
    that while every waypoint stays in one room. What CAN be changed is
    hiding it -- `room_rir_count` is recorded per chunk so the cue is
    countable, and the CLI refuses a split where fewer than two rooms qualify.
    Still sequences keep drawing from the whole pool, so a sparse room is not
    evidence of the sequence's own mode, only of what it could not have been.
    """
    cfg, manifest = _sparse_rir_manifest(
        tmp_path, {'room_00': 1, 'room_01': 4, 'room_02': 4})
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(manifest, UNIFIED_SPLIT), corpus_seed=SEED)
    needed = cfg.getint('path_motion', 'waypoints_min')

    moving_rooms = collections.Counter()
    still_rooms = collections.Counter()
    with _without_linear_aec():
        for sequence_id in range(24):
            rendered = renderer.render(SequencePlan(
                sequence_id=sequence_id, n_chunks=2, scenario='far_only',
                seed=stable_seed(SEED, 'test', f'sparse-drift-{sequence_id}'),
                talk_mode='far_only', impairments=('slow_drift',)))
            meta = rendered.chunk_meta[0]
            assert 'slow_drift' in meta['impairments']
            assert meta['path_motion'] == 'slow_drift', (
                "a planned trajectory was downgraded, so the corpus's motion "
                "share is not the planned one")
            assert len(meta['echo_path_waypoints']) >= needed
            assert meta['room_rir_count'] >= needed
            moving_rooms[meta['room_id']] += 1

            rendered = renderer.render(SequencePlan(
                sequence_id=sequence_id, n_chunks=2, scenario='far_only',
                seed=stable_seed(SEED, 'test', f'sparse-still-{sequence_id}'),
                talk_mode='far_only', impairments=()))
            still_rooms[rendered.chunk_meta[0]['room_id']] += 1
            assert rendered.chunk_meta[0]['path_motion'] == STATIC_PATH
    assert set(moving_rooms) == {'room_01', 'room_02'}, moving_rooms
    assert set(still_rooms) == {'room_00', 'room_01', 'room_02'}, still_rooms


def test_the_generator_refuses_a_split_whose_motion_lives_in_one_room(
        tmp_path):
    """One eligible room makes the trajectory identify the room.

    With a single room able to hold a trajectory, every moving sequence in the
    corpus carries that room's RT60 and early-reflection signature and the
    motion mode is readable off the reverberation alone. Refused before any
    worker starts, because it cannot be repaired once the audio exists.
    """
    cfg, _manifest = _sparse_rir_manifest(
        tmp_path, {'room_00': 1, 'room_01': 4})
    _only_impairments(cfg, slow_drift='1')
    # The message has to name the mode: a split can be rich enough for one
    # motion mode and too sparse for another, and "add more RIRs per room" is
    # not actionable without knowing which requirement was missed.
    with pytest.raises(ValueError,
                       match='slow_drift sequences but only 1 of the 2 rooms'):
        _run_generator(cfg, tmp_path, 'one_room')


def test_a_split_with_one_room_is_exempt_from_the_two_room_rule(tmp_path):
    """A constant says nothing about any sequence.

    The rule above refuses a split where one of SEVERAL rooms is the only one
    that can host the mode, because the trajectory is then readable off the
    room. Where the split has a single room altogether there is nothing to
    read: every sequence, moving or still, carries that room.
    """
    cfg, _manifest = _sparse_rir_manifest(tmp_path, {'room_00': 4})
    _only_impairments(cfg, slow_drift='1')
    cfg.set('talk_modes', 'p_near_only', '0')
    _run_generator(cfg, tmp_path, 'one_room_only')


def test_the_generator_refuses_a_motion_mode_a_short_run_did_not_draw(
        tmp_path):
    """The preflight is about the CONFIG, not about one plan's draws.

    Room eligibility is a property of the config and the source inventory, so
    checking only the modes a plan happens to contain makes the refusal depend
    on ``--hours``: a handful of sequences may draw none of the sparse mode and
    run to completion, and the full run then dies on the same manifest. Here
    the mode with the room requirement is drawn by neither planned sequence.
    """
    cfg, _manifest = _sparse_rir_manifest(
        tmp_path, {'room_00': 2, 'room_01': 2, 'room_02': 4})
    # A one-shot switch needs two positions, which every room here has; a
    # trajectory needs waypoints_min, which only one room has.
    _only_impairments(cfg, echo_path_change='0.9', slow_drift='0.1')
    cfg.set('talk_modes', 'p_near_only', '0')
    planned = {path_motion_mode(resolve_sequence_plan(plan)[2])
               for plan in plan_sequences(cfg, 0.002, SEED, UNIFIED_SPLIT)}
    assert 'slow_drift' not in planned, (
        f"this plan drew {sorted(planned)}, so it cannot show that a mode the "
        "plan does NOT contain is still checked")
    # The message says what is true: the CONFIGURATION enables the mode. The
    # plan in front of it contains none, which is the whole point.
    with pytest.raises(
            ValueError,
            match='the configuration enables slow_drift sequences but only 1 '
                  'of the 3 rooms'):
        _run_generator(cfg, tmp_path, 'sparse_modes')


def test_the_preflight_reads_the_probabilities_the_planner_reads(tmp_path):
    """A section the planner ignores must not refuse the run.

    ``plan_sequences`` takes the layered branch whenever the layer sections
    exist, so a stale ``[scenarios]`` left in a layered config is dead to it --
    every probability in it. A preflight that unions both sections would refuse
    a perfectly renderable split over a mode that can never be drawn.
    """
    cfg, _manifest = _sparse_rir_manifest(
        tmp_path, {'room_00': 1, 'room_01': 4})
    _only_impairments(cfg)
    cfg.add_section('scenarios')
    # One eligible room: live, this would be refused.
    cfg.set('scenarios', 'p_slow_drift', '1')
    planned = {path_motion_mode(resolve_sequence_plan(plan)[2])
               for plan in plan_sequences(cfg, 0.002, SEED, UNIFIED_SPLIT)}
    assert planned == {STATIC_PATH}, (
        f"the layered planner drew {sorted(planned)} from a config whose only "
        "motion probability is in the dead section")
    _run_generator(cfg, tmp_path, 'stale_scenarios')


def test_the_census_counts_the_motion_that_was_rendered(tmp_path, capsys):
    """A corpus summary describes the corpus, not the plan.

    A sequence whose drawn motion did not fit renders still, so counting the
    planned impairments would report an axis the audio does not carry. Here
    every planned one-shot switch is downgraded -- the guaranteed far-active
    window is longer than the sequence can hold behind any switch point -- and
    the summary has to say so.
    """
    cfg, _manifest = _sparse_rir_manifest(
        tmp_path, {'room_00': 2, 'room_01': 2, 'room_02': 2})
    _only_impairments(cfg, echo_path_change='1')
    cfg.set('talk_modes', 'p_near_only', '0')
    cfg.set('echo_modes', 'p_far_active_no_echo', '0')
    cfg.set('path_motion', 'far_active_after_event_sec', '2.5')
    planned = [path_motion_mode(resolve_sequence_plan(plan)[2])
               for plan in plan_sequences(cfg, 0.002, SEED, UNIFIED_SPLIT)]
    assert planned and set(planned) == {'echo_path_change'}

    _run_generator(cfg, tmp_path, 'downgrade')
    printed = capsys.readouterr().out
    census = [line for line in printed.splitlines()
              if line.startswith('  Path motion')]
    assert census and 'echo_path_change' not in census[0], census
    assert f"'{STATIC_PATH}': {len(planned)}" in census[0], census
    assert 'rendered still' in printed


def test_the_reach_audit_is_scoped_to_this_run_and_counts_the_re_draws(
        tmp_path, capsys):
    """Two rows about the trajectory's positions, both scoped to this run.

    Nothing about a finished sequence is persisted beyond its chunk WAVs, so a
    resumed run cannot read back what an earlier one reached. The audit
    therefore covers the sequences THIS run rendered -- and a fully resumed run
    has to say the audit is missing rather than print nothing, which reads
    exactly like a corpus whose reach was never in question.

    The position counters are the other half, and there are two of them
    because a re-draw and a fallback cost different things: a pool whose
    eligible rooms are mostly near-duplicate positions renders correctly and
    slowly, while the sequences that run out of draws render their room's ONE
    certified trajectory and are correct without being varied. A single sum
    cannot tell an operator which of the two a corpus has.
    """
    cfg, _manifest = _mixed_rir_manifest(tmp_path)
    _only_impairments(cfg, movement='1')
    cfg.set('talk_modes', 'p_near_only', '0')
    cfg.set('echo_modes', 'p_far_active_no_echo', '0')
    def run(resume):
        _run_generator(cfg, tmp_path, 'reach', resume=resume)
        return capsys.readouterr().out

    printed = run(False)
    assert 'Path correlation : movement reached' in printed, printed
    assert 'rendered this run' in printed, printed
    assert 'drawn again' in printed, (
        "a pool of near-duplicate positions rendered without a single "
        "re-drawn set, so the counter is measuring nothing here")
    assert 'exhausted their draws' in printed, (
        "no trajectory on this pool exhausted its draws, so the fallback row "
        "is measuring nothing here")

    resumed = run(True)
    assert 'Path correlation : not audited' in resumed, resumed
    assert 'Position draws   : not audited' in resumed, (
        "a fully resumed run printed no position rows at all, which reads "
        "as a corpus without a single re-draw")
    assert 'drawn again' not in resumed and 'exhausted' not in resumed


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
