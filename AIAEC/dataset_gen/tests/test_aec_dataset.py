"""End-to-end tests for the AEC generator.

These render a tiny synthetic corpus through the REAL pipeline -- manifest,
generator CLI, packer, packed dataset -- rather than unit-testing the pieces in
isolation.  Every invariant checked here is one that a consumer cannot detect
being broken: a swapped stem channel trains a model that cancels the talker and
converges beautifully, a leaked speaker produces a validation curve that looks
excellent, and a sequence packed out of order looks like slow convergence.
"""

import argparse
import copy
import configparser
import dataclasses
import math
import pathlib
import re

import numpy as np
import pytest
import torch
import torchaudio

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
    AecSequenceRenderer,
    SequencePlan,
    check_rate_dependent_values,
    chunk_samples_from_config,
    plan_sequences,
    resample_by_ratio,
    stable_seed,
)
from AIAEC.dataset_gen import gen_aec_dataset as gen_aec_dataset_module
from AIAEC.dataset_gen.gen_aec_dataset import build_parser, gen_aec_dataset
from AIAEC.dataset_gen.linear_aec import linear_aec_contract_from_config
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


def _rir(n_samples, rt60, generator):
    t = torch.arange(n_samples, dtype=torch.float32) / SR
    decay = torch.exp(-6.9078 * t / rt60)
    out = torch.randn(n_samples, generator=generator) * decay
    out[0] = 1.0                      # a clear direct path
    return out * 0.5


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
    for room in range(4):
        for index in range(2):
            _write(root / 'rir' / f'room_{room:02d}' / f'rir_{index}.wav',
                   _rir(int(0.35 * SR), 0.3 + 0.1 * room, generator))

    cfg = _base_cfg(root)
    cfg.set('split', 'val_fraction', '0.25')
    # Boosted so the small corpus reliably contains the load-bearing scenario.
    cfg.set('scenarios', 'p_ref_dropout', '0.30')

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

    Rendered directly so the scenario is forced: a 40 s ref_dropout sequence is
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
        'sro_ppm', 'nonlinear', 'clipped', 'scenario',
    }
    checked = 0
    for _plan, rendered in _render_plans(
            corpus['cfg'], corpus['manifest'], 'train'):
        for meta in rendered.chunk_meta:
            checked += 1
            assert required <= set(meta), f"missing {required - set(meta)}"
            assert meta['scenario'] in SCENARIOS
            assert meta['sequence_scenario'] in SCENARIOS
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


def _sparse_rir_manifest(tmp_path, rooms):
    """A minimal corpus whose room -> RIR-file-count is fully controlled.

    ``rooms`` maps a room name to how many RIR files it gets, e.g.
    ``{'room_00': 1, 'room_01': 2}``.
    """
    generator = torch.Generator().manual_seed(29)
    root = tmp_path / 'sparse_rir_corpus'
    for speaker in range(3):
        _write(root / 'speech' / f'reader_{speaker:03d}' / 'take_0.wav',
              _speechlike(4 * SR, generator))
    for index in range(3):
        _write(root / 'noise' / f'noise_{index:02d}.wav',
              torch.randn(3 * SR, generator=generator) * 0.05)
    for room, count in rooms.items():
        for index in range(count):
            _write(root / 'rir' / room / f'rir_{index}.wav',
                  _rir(int(0.35 * SR), 0.3, generator))

    cfg = _base_cfg(root)
    return cfg, build_unified_manifest(cfg, seed=SEED, progress=False)


def test_echo_path_change_never_leaves_a_room_with_only_one_rir(tmp_path):
    """The post-change RIR must stay in the SAME room as the near talker's.

    room_00 has only 1 RIR file and must never be drawn for this scenario --
    otherwise _pick_path_change_rir would have to cross into a different room,
    reintroducing the acoustic "this is echo" leak the same-room invariant
    exists to prevent.
    """
    cfg, manifest = _sparse_rir_manifest(tmp_path, {'room_00': 1, 'room_01': 2})
    renderer = AecSequenceRenderer(
        cfg, pools_for_split(manifest, UNIFIED_SPLIT), corpus_seed=SEED)

    checked = 0
    for sequence_id in range(20):
        rendered = renderer.render(SequencePlan(
            sequence_id=sequence_id, n_chunks=3, scenario='echo_path_change',
            seed=stable_seed(SEED, 'test', f'epc-{sequence_id}')))
        for meta in rendered.chunk_meta:
            assert meta['room_id'] == 'room_01'
            checked += 1
    assert checked > 0


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


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
