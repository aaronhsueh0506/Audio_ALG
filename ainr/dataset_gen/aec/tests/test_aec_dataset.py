"""End-to-end tests for the AEC generator.

These render a tiny synthetic corpus through the REAL pipeline -- manifest,
generator CLI, packer, packed dataset -- rather than unit-testing the pieces in
isolation.  Every invariant checked here is one that a consumer cannot detect
being broken: a swapped stem channel trains a model that cancels the talker and
converges beautifully, a leaked speaker produces a validation curve that looks
excellent, and a sequence packed out of order looks like slow convergence.
"""

import argparse
import configparser
import math
import pathlib
import sys

import pytest
import torch

AINR = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(AINR))

from dataset_gen.aec import (  # noqa: E402
    STEM_ORDER,
    AecGrid,
    AecStems,
    SequenceChunkSampler,
    alpha_from_tau,
    assert_source_disjoint,
    istft,
    lane_reset_mask,
    stft,
)
from dataset_gen.aec.aec_dataset import (  # noqa: E402
    AecSequenceRenderer,
    SequencePlan,
    plan_sequences,
    resample_by_ratio,
    stable_seed,
)
from dataset_gen.aec.gen_aec_dataset import gen_aec_dataset  # noqa: E402
from dataset_gen.aec.manifest import (  # noqa: E402
    build_manifest,
    load_manifest,
    pools_for_split,
)
from dataset_gen.aec.pack_aec_dataset import pack  # noqa: E402
from dataset_gen.aec.packed_aec_dataset import PackedAecDataset  # noqa: E402


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

    cfg = configparser.ConfigParser()
    cfg.read(pathlib.Path(__file__).parents[1] / 'config.example.ini')
    cfg.set('signal', 'sr', str(SR))
    cfg.set('paths', 'speech_dir', str(root / 'speech'))
    cfg.set('paths', 'noise_dir', str(root / 'noise'))
    cfg.set('paths', 'rir_dir', str(root / 'rir'))
    cfg.set('split', 'val_fraction', '0.25')
    cfg.set('sequence', 'seq_sec_min', '2.0')
    cfg.set('sequence', 'seq_sec_max', '3.0')
    cfg.set('sequence', 'chunk_sec', '1.0')
    cfg.set('rir', 'rt60_min', '0.05')
    cfg.set('rir', 'rt60_max', '2.0')
    # Boosted so the small corpus reliably contains the load-bearing scenario.
    cfg.set('scenarios', 'p_ref_dropout', '0.30')

    config_path = root / 'config.ini'
    with open(config_path, 'w') as handle:
        cfg.write(handle)

    manifest = build_manifest(cfg, seed=SEED, progress=False)
    return {'root': root, 'cfg': cfg, 'config_path': config_path,
            'manifest': manifest}


@pytest.fixture(scope='module')
def packed(corpus, tmp_path_factory):
    """Render and pack a small corpus through the real CLI entry points."""
    output = tmp_path_factory.mktemp('aec_data')
    gen_aec_dataset(argparse.Namespace(
        config=str(corpus['config_path']), output=str(output), hours=0.012,
        workers=0, resume=False, seed=SEED, sample_rate=None, split='train',
        manifest=None, rebuild_manifest=False, wav_encoding='float32',
    ))
    gen_aec_dataset(argparse.Namespace(
        config=str(corpus['config_path']), output=str(output), hours=0.004,
        workers=0, resume=False, seed=SEED, sample_rate=None, split='val',
        manifest=None, rebuild_manifest=False, wav_encoding='float32',
    ))
    for split in ('train', 'val'):
        pack(argparse.Namespace(
            input=str(output / split), output=str(output / 'packed' / split),
            shard_clips=8, dtype='float32'))
    return {
        'output': output,
        'train': PackedAecDataset(str(output / 'packed' / 'train'), verbose=False),
        'val': PackedAecDataset(str(output / 'packed' / 'val'), verbose=False),
    }


# ============================================================
# Stem layout
# ============================================================

def test_stem_channel_order_matches_declared_list(packed):
    """The shard's declared order must be THE order, in every shard."""
    assert list(STEM_ORDER) == [
        'far_render', 'echo', 'near_speech', 'local_noise',
        'mic_preclip', 'mic_postclip',
    ]
    for split in ('train', 'val'):
        dataset = packed[split]
        assert tuple(dataset.stems) == STEM_ORDER
        stems, _meta = dataset[0]
        assert stems.shape[0] == len(STEM_ORDER)
        # And the named view must reach the same channels by name.
        view = AecStems(stems)
        for position, name in enumerate(STEM_ORDER):
            assert torch.equal(view.stem(name), stems[position])


def test_named_view_rejects_a_wrong_order():
    """A shard claiming a different order must fail loudly, not be reordered."""
    with pytest.raises(ValueError):
        AecStems(torch.zeros(6, 16), ('echo',) + STEM_ORDER[1:5])
    with pytest.raises(ValueError):
        AecStems(torch.zeros(5, 16))


def test_stems_recombine(packed):
    """mic_preclip == near_speech + local_noise + echo, for every clip.

    This is the corpus's central invariant.  If it fails, the stems have been
    scaled independently somewhere and no consumer can trust that ``echo`` is
    the echo that is actually in the microphone signal.
    """
    for split in ('train', 'val'):
        dataset = packed[split]
        for index in range(len(dataset)):
            view = dataset.stems_of(index)
            recombined = view.near_speech + view.local_noise + view.echo
            error = (view.mic_preclip - recombined).abs().max().item()
            assert error < 1e-5, (
                f"{split}[{index}] stems do not sum: max error {error:.2e}")


def test_postclip_differs_exactly_where_the_metadata_says(packed):
    """`clipped` / `agc` must describe the data, not sit alongside it.

    Both directions matter.  A flag that is set when nothing happened would
    poison an ablation; a flag that is clear when the mic path WAS altered
    would make mic_postclip silently untrustworthy.
    """
    for split in ('train', 'val'):
        dataset = packed[split]
        for index in range(len(dataset)):
            view = dataset.stems_of(index)
            meta = dataset.meta(index)
            altered = not torch.allclose(view.mic_postclip, view.mic_preclip,
                                         atol=1e-6)
            flagged = meta['clipped'] or meta['agc']
            assert altered == flagged, (
                f"{split}[{index}]: mic altered={altered} but "
                f"clipped={meta['clipped']} agc={meta['agc']}")
            if meta['sequence_scenario'] == 'clipping_agc':
                assert meta['clipped'] and meta['agc']


# ============================================================
# Reference dropout
# ============================================================

def test_ref_dropout_clips_have_a_silent_reference(packed):
    """Every chunk LABELLED ref_dropout really has X == 0.

    ⚠ This is what the idle-loss term and the "ref == 0 implies output ~= mic"
    gate are trained on.  If a dropout-labelled chunk still carried far-end
    audio, the gate would be supervised by contradictory examples.
    """
    found = 0
    for split in ('train', 'val'):
        dataset = packed[split]
        for index in dataset.indices_where(scenario='ref_dropout'):
            view = dataset.stems_of(index)
            assert float(view.far_render.abs().max()) == 0.0
            # With the default ref_dropout_echo_continues_p = 0 the far end is
            # silent end to end, so the mic is exactly S + N.
            assert float(view.echo.abs().max()) == 0.0
            found += 1
    assert found > 0, "corpus contains no ref_dropout chunks to check"


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


def test_a_leaked_source_is_detected(corpus):
    """The disjointness check must actually be able to fail."""
    import copy
    leaky = copy.deepcopy(corpus['manifest'])
    leaky['splits']['val']['speakers'].append(
        leaky['splits']['train']['speakers'][0])
    with pytest.raises(ValueError, match='source leak'):
        assert_source_disjoint(leaky)


def test_generated_clips_only_use_their_split_sources(packed, corpus):
    """The claim must hold in the DATA, not only in the manifest.

    A manifest can be perfectly disjoint while the renderer reaches past it.
    """
    manifest = corpus['manifest']
    for split in ('train', 'val'):
        allowed_rooms = set(manifest['splits'][split]['rooms'])
        allowed_devices = set(manifest['splits'][split]['devices'])
        allowed_speakers = set(manifest['splits'][split]['speakers'])
        dataset = packed[split]
        for index in range(len(dataset)):
            meta = dataset.meta(index)
            assert meta['room_id'] in allowed_rooms
            assert meta['device_id'] in allowed_devices
            if meta['speaker_id']:
                assert meta['speaker_id'] in allowed_speakers


def test_train_and_val_clips_share_no_room_or_device(packed):
    def observed(split, key):
        dataset = packed[split]
        return {dataset.meta(i)[key] for i in range(len(dataset))}

    for key in ('room_id', 'device_id'):
        assert observed('train', key) & observed('val', key) == set()


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
    reference, echo = view.X, view.D

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
    assert torch.allclose(view.mic_preclip, view.echo + view.local_noise, atol=1e-5)


def test_metadata_covers_the_declared_contract(packed):
    required = {
        'sequence_id', 'chunk_index', 'speaker_id', 'noise_id', 'rir_id',
        'ser_db', 'snr_db', 'erl_db', 'bulk_delay_samples', 'delay_jitter',
        'sro_ppm', 'nonlinear', 'clipped', 'scenario',
    }
    from dataset_gen.aec.aec_dataset import SCENARIOS
    dataset = packed['train']
    for index in range(len(dataset)):
        meta = dataset.meta(index)
        assert required <= set(meta), f"missing {required - set(meta)}"
        assert meta['scenario'] in SCENARIOS
        assert meta['sequence_scenario'] in SCENARIOS
        assert isinstance(meta['delay_jitter'], bool)
        assert isinstance(meta['clipped'], bool)
        # ⚠ +-inf is deliberate: it marks a ratio that is undefined because one
        # of its two signals is absent, rather than a fabricated number.
        assert not math.isnan(meta['ser_db'])
        assert not math.isnan(meta['snr_db'])


def test_shard_records_provenance(packed):
    dataset = packed['train']
    shard = torch.load(dataset.paths[0], map_location='cpu', weights_only=False)
    assert set(shard) >= {'stems', 'data', 'sr', 'meta',
                          'generator_commit', 'config_hash'}
    assert shard['sr'] == SR
    assert shard['data'].dtype == torch.float32


def test_manifest_round_trips(corpus, tmp_path):
    from dataset_gen.aec.manifest import save_manifest
    path = tmp_path / 'manifest.json'
    save_manifest(corpus['manifest'], str(path))
    assert load_manifest(str(path))['splits'] == corpus['manifest']['splits']


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
