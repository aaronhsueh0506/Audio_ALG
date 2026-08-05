"""Regression tests for dataset generation planning and RIR alignment."""

import configparser
import math
import random
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from dataset_gen.dataset import (
    DNS4Dataset,
    delay_signal,
    fftconvolve,
    parse_source_sr_values,
    parse_snr_values,
    prepare_rir,
    sample_mix_mode,
    sample_snr,
    simulate_upsampled_source,
    source_sr_candidates,
    validate_mix_probabilities,
)
from dataset_gen.gen_dataset import hours_to_sample_count, seed_worker
from dataset_gen.resample_dataset import resampled_num_frames


def _load_example_config() -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    cfg.read(Path(__file__).parents[1] / "config.example.ini")
    return cfg


def _build_dataset(cfg):
    """Construct a real DNS4Dataset from a config, without touching disk."""
    with (
        patch(
            "dataset_gen.dataset.glob.glob",
            side_effect=(["speech.wav"], ["noise.wav"]),
        ),
        patch("dataset_gen.dataset.os.path.isdir", return_value=False),
    ):
        return DNS4Dataset(cfg, return_raw=True)


def _stub_dataset(**overrides):
    """Build a DNS4Dataset with every _getitem_impl-required attribute set
    to a deterministic default, bypassing __init__ (no file I/O). Tests
    override only the fields they care about via kwargs.
    """
    dataset = DNS4Dataset.__new__(DNS4Dataset)
    dataset._indices = [0]
    dataset.speech_files = ["speech.wav"]
    dataset.segment_samples = 1600
    dataset.sr = 16000
    dataset.noise_only_p = 0.0
    dataset.speech_only_p = 0.0
    dataset.p_biquad = 0.0
    dataset.n_biquad_filters = 3
    dataset.biquad_gain_db = 15.0
    dataset.biquad_q_min = 0.5
    dataset.biquad_q_max = 1.5
    dataset.rir_files = []
    dataset.p_rir = 0.0
    dataset.rt60_min = 0.2
    dataset.rt60_max = 1.0
    dataset.early_rir_ms = 20.0
    dataset.pre_delay_keep_ms = 1.0
    dataset.drr = 0.3
    dataset.max_noise_mix = 1
    dataset.noise_files = ["noise.wav"]
    dataset.snr_values = [0.0]
    dataset.p_resample = 0.0
    dataset.source_sr_values = [8000]
    dataset.p_noise_clipping = 0.0
    dataset.p_mixture_clipping = 0.0
    dataset.clip_snr_min = 0.0
    dataset.clip_snr_max = 20.0
    dataset.level_mode = 'dns_target_level'
    dataset.target_level_min_db = -40.0
    dataset.target_level_max_db = -10.0
    dataset.return_raw = True
    dataset.return_metadata = False
    dataset._load_and_crop = lambda _path, length: torch.linspace(
        -0.2, 0.2, length
    )
    dataset._load_noise = lambda length: (
        torch.linspace(-0.05, 0.05, length), "noise.wav"
    )
    dataset._load_rir = lambda: (
        torch.tensor([1.0, 0.5, 0.2] + [0.0] * 20), 0.4, "rir.wav"
    )
    for key, value in overrides.items():
        setattr(dataset, key, value)
    return dataset


class DatasetConfigScopeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = configparser.ConfigParser()
        cls.cfg.read(Path(__file__).parents[1] / "config.example.ini")

    def test_example_contains_generation_sections_only(self):
        self.assertEqual(
            set(self.cfg.sections()),
            {
                "signal",
                "paths",
                "audio",
                "gen",
                "mixing",
                "rir",
                "noise",
                "augmentation",
            },
        )
        for section in ("model", "training", "perceptual_loss"):
            self.assertFalse(self.cfg.has_section(section))

    def test_raw_generator_needs_no_model_feature_or_training_keys(self):
        with (
            patch(
                "dataset_gen.dataset.glob.glob",
                side_effect=(["speech.wav"], ["noise.wav"]),
            ),
            patch("dataset_gen.dataset.os.path.isdir", return_value=False),
        ):
            dataset = DNS4Dataset(self.cfg, return_raw=True)

        self.assertFalse(hasattr(dataset, "n_fft"))
        self.assertFalse(hasattr(dataset, "bin_edges"))
        self.assertEqual(len(dataset), 1)
        self.assertEqual(
            dataset.source_sr_values,
            [8000, 12000, 16000, 22050, 24000, 32000, 44100],
        )
        self.assertEqual(dataset.p_resample, 0.1)


class HoursPlanningTest(unittest.TestCase):
    def test_decimal_hours_do_not_add_a_spurious_segment(self):
        self.assertEqual(hours_to_sample_count(8.3, 3.0), 9960)
        self.assertEqual(hours_to_sample_count(25.0, 3.0), 30000)

    def test_partial_segment_rounds_up_once(self):
        self.assertEqual(hours_to_sample_count(0.001, 3.0), 2)

    def test_invalid_duration_is_rejected(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                hours_to_sample_count(value, 3.0)


class RandomSeedTest(unittest.TestCase):
    def test_worker_seed_controls_python_random(self):
        torch.manual_seed(123)
        seed_worker(0)
        first = random.random()

        torch.manual_seed(123)
        seed_worker(1)
        self.assertEqual(random.random(), first)

        torch.manual_seed(124)
        seed_worker(0)
        self.assertNotEqual(random.random(), first)


class UpsampledSourceTest(unittest.TestCase):
    def test_source_rates_are_discrete_unique_and_sr_dependent(self):
        values = parse_source_sr_values(
            "8000, 12000, 16000, 22050, 24000, 32000, 44100, 16000"
        )
        self.assertEqual(
            values,
            [8000, 12000, 16000, 22050, 24000, 32000, 44100],
        )
        self.assertEqual(source_sr_candidates(values, 48000), values)
        self.assertEqual(
            source_sr_candidates(values, 16000),
            [8000, 12000],
        )

    def test_invalid_source_rates_are_rejected(self):
        for value in ("", "0", "-8000, 16000"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_source_sr_values(value)
        with self.assertRaises(ValueError):
            source_sr_candidates([8000], 0)

    def test_downsample_upsample_preserves_length_and_removes_high_band(self):
        algorithm_sr = 48000
        samples = torch.arange(algorithm_sr // 10) / algorithm_sr
        high_band = torch.sin(2 * torch.pi * 12000 * samples)

        output = simulate_upsampled_source(
            high_band,
            algorithm_sr=algorithm_sr,
            source_sr=16000,
        )

        self.assertEqual(output.shape, high_band.shape)
        self.assertLess(output.square().mean(), high_band.square().mean() * 0.01)

    def test_native_or_higher_source_rate_is_a_no_op(self):
        audio = torch.randn(100)
        self.assertIs(simulate_upsampled_source(audio, 48000, 48000), audio)
        self.assertIs(simulate_upsampled_source(audio, 48000, 96000), audio)


class MixingPolicyTest(unittest.TestCase):
    def test_dfn_discrete_snr_values_are_parsed_and_sampled(self):
        values = parse_snr_values("-5, 0, 5, 10, 20, 40")
        self.assertEqual(values, [-5.0, 0.0, 5.0, 10.0, 20.0, 40.0])

        random.seed(123)
        self.assertTrue(all(sample_snr(values) in values for _ in range(100)))

    def test_invalid_snr_values_are_rejected(self):
        for value in ("", "nan", "0, inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_snr_values(value)

    def test_special_modes_are_mutually_exclusive(self):
        validate_mix_probabilities(0.05, 0.05)
        with patch(
            "dataset_gen.dataset.random.random",
            side_effect=(0.049, 0.05, 0.099, 0.10),
        ):
            self.assertEqual(sample_mix_mode(0.05, 0.05), "noise_only")
            self.assertEqual(sample_mix_mode(0.05, 0.05), "speech_only")
            self.assertEqual(sample_mix_mode(0.05, 0.05), "speech_only")
            self.assertEqual(sample_mix_mode(0.05, 0.05), "mixed")

    def test_invalid_special_mode_probabilities_are_rejected(self):
        for noise_p, speech_p in ((-0.1, 0.0), (0.0, 1.1), (0.6, 0.5)):
            with (
                self.subTest(noise_p=noise_p, speech_p=speech_p),
                self.assertRaises(ValueError),
            ):
                validate_mix_probabilities(noise_p, speech_p)

    def test_speech_only_pair_stays_identity_through_resampling(self):
        # p_resample=1.0 and p_mixture_clipping=1.0 (would-be certain) both
        # exercise the "does speech_only actually skip this" question, not
        # just "is the probability wired up" -- level normalization
        # (dns_target_level, always on in _stub_dataset's defaults) is
        # exercised too: it must scale noisy/target by the identical factor
        # or this identity breaks.
        dataset = _stub_dataset(
            noise_only_p=0.05,
            speech_only_p=0.05,
            p_resample=1.0,
            source_sr_values=[8000],
            p_mixture_clipping=1.0,
        )

        with (
            patch("dataset_gen.dataset.sample_mix_mode", return_value="speech_only"),
            patch("dataset_gen.dataset.apply_clipping") as clipping,
        ):
            noisy, target = dataset._getitem_impl(0)

        self.assertTrue(torch.equal(noisy, target))
        clipping.assert_not_called()


class RirAlignmentTest(unittest.TestCase):
    def test_dry_target_and_reverbed_direct_paths_align(self):
        rir = torch.zeros(24)
        rir[10] = 1.0
        rir[13] = 0.3
        target_rir, full_rir = prepare_rir(
            rir,
            sr=1000,
            late_offset_ms=20,
            pre_delay_keep_ms=2,
            rt60=0.5,
        )
        direct_delay = int(full_rir.abs().argmax())
        self.assertEqual(direct_delay, 2)
        self.assertAlmostEqual(full_rir.norm().item(), 1.0, places=6)
        self.assertAlmostEqual(target_rir.norm().item(), 1.0, places=6)

        speech = torch.zeros(32)
        speech[0] = 1.0
        reverbed = fftconvolve(speech, full_rir)
        aligned_dry = delay_signal(speech, direct_delay)
        target = (
            0.3 * aligned_dry
            + 0.7 * fftconvolve(speech, target_rir)
        )

        self.assertTrue(torch.allclose(
            reverbed[:direct_delay],
            torch.zeros(direct_delay),
            atol=1e-7,
        ))
        self.assertTrue(torch.allclose(
            target[:direct_delay],
            torch.zeros(direct_delay),
            atol=1e-7,
        ))
        self.assertEqual(int(reverbed.abs().argmax()), direct_delay)
        self.assertEqual(int(target.abs().argmax()), direct_delay)

    def test_early_peak_keeps_its_actual_position(self):
        rir = torch.zeros(12)
        rir[1] = 1.0
        target_rir, full_rir = prepare_rir(
            rir,
            sr=1000,
            late_offset_ms=20,
            pre_delay_keep_ms=5,
            rt60=0.5,
        )
        self.assertEqual(int(full_rir.abs().argmax()), 1)
        self.assertEqual(int(target_rir.abs().argmax()), 1)

    def test_delay_preserves_length(self):
        signal = torch.tensor([1.0, 2.0, 3.0])
        self.assertTrue(torch.equal(
            delay_signal(signal, 1),
            torch.tensor([0.0, 1.0, 2.0]),
        ))
        self.assertTrue(torch.equal(
            delay_signal(signal, 3),
            torch.zeros_like(signal),
        ))


class ResampleMetadataTest(unittest.TestCase):
    def test_output_length_matches_ceil_contract(self):
        self.assertEqual(resampled_num_frames(4800, 48000, 16000), 1600)
        self.assertEqual(resampled_num_frames(4801, 48000, 24000), 2401)


class LevelNormalizationConfigTest(unittest.TestCase):
    def test_invalid_level_mode_is_rejected(self):
        cfg = _load_example_config()
        cfg.set('mixing', 'level_mode', 'bogus_mode')
        with self.assertRaises(ValueError):
            _build_dataset(cfg)

    def test_inverted_level_range_is_rejected(self):
        cfg = _load_example_config()
        cfg.set('mixing', 'target_level_min_db', '-10')
        cfg.set('mixing', 'target_level_max_db', '-40')
        with self.assertRaises(ValueError):
            _build_dataset(cfg)

    def test_non_finite_level_bounds_are_rejected(self):
        for value in ('nan', 'inf'):
            with self.subTest(value=value):
                cfg = _load_example_config()
                cfg.set('mixing', 'target_level_max_db', value)
                with self.assertRaises(ValueError):
                    _build_dataset(cfg)

    def test_defaults_apply_when_mixing_level_keys_are_omitted(self):
        cfg = _load_example_config()
        cfg.remove_option('mixing', 'level_mode')
        cfg.remove_option('mixing', 'target_level_min_db')
        cfg.remove_option('mixing', 'target_level_max_db')
        dataset = _build_dataset(cfg)
        self.assertEqual(dataset.level_mode, 'dns_target_level')
        self.assertEqual(dataset.target_level_min_db, -40.0)
        self.assertEqual(dataset.target_level_max_db, -10.0)


class ClippingConfigTest(unittest.TestCase):
    def test_example_config_splits_noise_and_mixture_clipping(self):
        dataset = _build_dataset(_load_example_config())
        self.assertAlmostEqual(dataset.p_noise_clipping, 0.10)
        self.assertEqual(dataset.p_mixture_clipping, 0.0)

    def test_out_of_range_clipping_probabilities_are_rejected(self):
        for key, value in (
            ('p_noise_clipping', '1.5'),
            ('p_mixture_clipping', '-0.1'),
        ):
            with self.subTest(key=key):
                cfg = _load_example_config()
                cfg.set('augmentation', key, value)
                with self.assertRaises(ValueError):
                    _build_dataset(cfg)


class LevelNormalizationBehaviorTest(unittest.TestCase):
    def test_level_normalization_targets_requested_dbfs(self):
        # min == max pins the uniform draw to an exact value without
        # needing to mock random.uniform.
        dataset = _stub_dataset(
            return_metadata=True,
            target_level_min_db=-20.0, target_level_max_db=-20.0,
        )
        with patch("dataset_gen.dataset.sample_mix_mode", return_value="mixed"):
            noisy, target, metadata = dataset._getitem_impl(0)
        self.assertAlmostEqual(metadata['requested_level_dbfs'], -20.0)
        achieved = 20.0 * math.log10(noisy.pow(2).mean().sqrt().item())
        self.assertAlmostEqual(achieved, -20.0, places=3)
        self.assertAlmostEqual(metadata['effective_level_dbfs'], achieved, places=3)

    def test_level_normalization_measures_after_resample_simulation(self):
        dataset = _stub_dataset(
            return_metadata=True,
            p_resample=1.0, source_sr_values=[8000],
            target_level_min_db=-20.0, target_level_max_db=-20.0,
        )
        with (
            patch("dataset_gen.dataset.sample_mix_mode", return_value="mixed"),
            patch(
                "dataset_gen.dataset.simulate_upsampled_source",
                side_effect=lambda audio, algorithm_sr, source_sr: audio * 0.1,
            ),
        ):
            noisy, _target, _metadata = dataset._getitem_impl(0)
        # If the level were measured/applied BEFORE this (mocked) resample-
        # simulation step, the trailing *0.1 would knock the result ~20 dB
        # below the requested target instead of landing on it -- proving
        # the RMS measurement runs on the POST-resample-simulation signal.
        achieved = 20.0 * math.log10(noisy.pow(2).mean().sqrt().item())
        self.assertAlmostEqual(achieved, -20.0, places=3)

    def test_level_normalization_preserves_target_to_noisy_ratio(self):
        def run_at(level_db):
            dataset = _stub_dataset(
                target_level_min_db=level_db, target_level_max_db=level_db,
            )
            with patch("dataset_gen.dataset.sample_mix_mode", return_value="mixed"):
                noisy, target = dataset._getitem_impl(0)
            return noisy, target

        noisy_a, target_a = run_at(-30.0)
        noisy_b, target_b = run_at(-15.0)
        ratio_a = target_a.pow(2).mean().sqrt() / noisy_a.pow(2).mean().sqrt()
        ratio_b = target_b.pow(2).mean().sqrt() / noisy_b.pow(2).mean().sqrt()
        self.assertAlmostEqual(ratio_a.item(), ratio_b.item(), places=5)

    def test_noise_only_target_stays_exactly_zero(self):
        dataset = _stub_dataset(
            return_metadata=True,
            target_level_min_db=-25.0, target_level_max_db=-25.0,
        )
        with patch("dataset_gen.dataset.sample_mix_mode", return_value="noise_only"):
            _noisy, target, metadata = dataset._getitem_impl(0)
        self.assertTrue(torch.equal(target, torch.zeros_like(target)))
        self.assertEqual(metadata['mix_mode'], 'noise_only')
        self.assertIsNone(metadata['snr_db'])


class ClippingBehaviorTest(unittest.TestCase):
    def test_noise_clipping_is_independent_of_mixture_clipping(self):
        dataset = _stub_dataset(
            return_metadata=True,
            p_noise_clipping=1.0, p_mixture_clipping=0.0,
        )
        with patch("dataset_gen.dataset.sample_mix_mode", return_value="mixed"):
            _noisy, _target, metadata = dataset._getitem_impl(0)
        self.assertTrue(metadata['noise_clipping_applied'])
        self.assertIsNotNone(metadata['noise_clip_snr_db'])
        self.assertFalse(metadata['mixture_clipping_applied'])
        self.assertIsNone(metadata['mixture_clip_snr_db'])

    def test_mixture_clipping_is_independent_of_noise_clipping(self):
        dataset = _stub_dataset(
            return_metadata=True,
            p_noise_clipping=0.0, p_mixture_clipping=1.0,
        )
        with patch("dataset_gen.dataset.sample_mix_mode", return_value="mixed"):
            _noisy, _target, metadata = dataset._getitem_impl(0)
        self.assertFalse(metadata['noise_clipping_applied'])
        self.assertTrue(metadata['mixture_clipping_applied'])
        self.assertIsNotNone(metadata['mixture_clip_snr_db'])

    def test_noise_clipping_applies_to_noise_only_mode(self):
        dataset = _stub_dataset(return_metadata=True, p_noise_clipping=1.0)
        with patch("dataset_gen.dataset.sample_mix_mode", return_value="noise_only"):
            _noisy, _target, metadata = dataset._getitem_impl(0)
        self.assertTrue(metadata['noise_clipping_applied'])

    def test_both_clipping_knobs_skip_speech_only(self):
        dataset = _stub_dataset(
            return_metadata=True,
            p_noise_clipping=1.0, p_mixture_clipping=1.0,
        )
        with patch("dataset_gen.dataset.sample_mix_mode", return_value="speech_only"):
            noisy, target, metadata = dataset._getitem_impl(0)
        self.assertFalse(metadata['noise_clipping_applied'])
        self.assertFalse(metadata['mixture_clipping_applied'])
        self.assertTrue(torch.equal(noisy, target))


class MetadataApiTest(unittest.TestCase):
    def test_return_metadata_requires_return_raw(self):
        cfg = _load_example_config()
        with (
            patch(
                "dataset_gen.dataset.glob.glob",
                side_effect=(["speech.wav"], ["noise.wav"]),
            ),
            patch("dataset_gen.dataset.os.path.isdir", return_value=False),
            self.assertRaises(ValueError),
        ):
            DNS4Dataset(cfg, return_raw=False, return_metadata=True)

    def test_getitem_shape_toggles_on_return_metadata(self):
        dataset = _stub_dataset(return_metadata=False)
        with patch("dataset_gen.dataset.sample_mix_mode", return_value="mixed"):
            two_tuple = dataset._getitem_impl(0)
        self.assertEqual(len(two_tuple), 2)

        dataset2 = _stub_dataset(return_metadata=True)
        with patch("dataset_gen.dataset.sample_mix_mode", return_value="mixed"):
            three_tuple = dataset2._getitem_impl(0)
        self.assertEqual(len(three_tuple), 3)
        _noisy, _target, metadata = three_tuple
        self.assertIsInstance(metadata, dict)
        for field in (
            'mix_mode', 'speech_file', 'snr_db',
            'requested_level_dbfs', 'effective_level_dbfs',
        ):
            self.assertIn(field, metadata)

    def test_metadata_records_provenance_and_augmentation_decisions(self):
        dataset = _stub_dataset(
            return_metadata=True,
            rir_files=["rir.wav"], p_rir=1.0,
            p_resample=1.0, source_sr_values=[8000],
            max_noise_mix=2,
        )
        with (
            patch("dataset_gen.dataset.sample_mix_mode", return_value="mixed"),
            patch("dataset_gen.dataset.random.randint", return_value=2),
        ):
            _noisy, _target, metadata = dataset._getitem_impl(0)
        self.assertEqual(metadata['speech_file'], "speech.wav")
        self.assertTrue(metadata['rir_applied'])
        self.assertEqual(metadata['rir_file'], "rir.wav")
        self.assertEqual(metadata['rt60'], 0.4)
        self.assertEqual(metadata['n_noises_mixed'], 2)
        self.assertEqual(metadata['noise_files'], ["noise.wav", "noise.wav"])
        self.assertTrue(metadata['resample_simulated'])
        self.assertEqual(metadata['source_sr'], 8000)
        self.assertEqual(metadata['snr_db'], 0.0)


if __name__ == '__main__':
    unittest.main()
