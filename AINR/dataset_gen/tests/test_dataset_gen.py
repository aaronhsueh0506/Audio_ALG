"""Regression tests for dataset generation planning and RIR alignment."""

import configparser
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
        dataset = DNS4Dataset.__new__(DNS4Dataset)
        dataset._indices = [0]
        dataset.speech_files = ["speech.wav"]
        dataset.segment_samples = 1600
        dataset.sr = 16000
        dataset.noise_only_p = 0.05
        dataset.speech_only_p = 0.05
        dataset.p_biquad = 0.0
        dataset.rir_files = []
        dataset.p_resample = 1.0
        dataset.source_sr_values = [8000]
        dataset.p_clipping = 1.0
        dataset.clip_snr_min = 0.0
        dataset.clip_snr_max = 20.0
        dataset.return_raw = True
        dataset._load_and_crop = lambda _path, length: torch.linspace(
            -0.2, 0.2, length
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


if __name__ == '__main__':
    unittest.main()
