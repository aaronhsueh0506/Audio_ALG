"""Regression tests for pack_dataset.py's effective_rms_dbfs recording.

pack_dataset.py's --target-sr resample does not exactly preserve the level/
SNR a WAV pair was generated at (see README.md's "48 kHz source, 16 kHz
pack" caveat) -- these tests pin that the packed payload records the
ACTUAL post-resample RMS, not a value silently carried forward from
generation-time metadata.
"""

import math
import os
import tempfile
import types
import unittest
from pathlib import Path

import torch
import torchaudio

from dataset_gen.dataset import rms_dbfs
from dataset_gen.pack_dataset import pack


def _write_pair(path, noisy, clean, sr):
    pair = torch.stack([noisy, clean], dim=0)
    torchaudio.save(path, pair, sr, bits_per_sample=16)


def _pack_args(input_dir, output_path, target_sr=None, quality='best',
               dtype='float32'):
    return types.SimpleNamespace(
        input=input_dir, output=output_path, dtype=dtype,
        target_sr=target_sr, quality=quality,
    )


class EffectiveRmsDbfsTest(unittest.TestCase):
    def test_present_and_correctly_shaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, 'pairs')
            os.makedirs(in_dir)
            sr = 16000
            t = torch.linspace(0, 1.0, sr, dtype=torch.float32)
            noisy = 0.5 * torch.sin(2 * math.pi * 200 * t)
            clean = 0.3 * torch.sin(2 * math.pi * 200 * t)
            _write_pair(os.path.join(in_dir, '000000.wav'), noisy, clean, sr)
            _write_pair(os.path.join(in_dir, '000001.wav'), noisy, clean, sr)

            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path))

            payload = torch.load(out_path, weights_only=True)
            self.assertIn('effective_rms_dbfs', payload)
            self.assertEqual(tuple(payload['effective_rms_dbfs'].shape), (2, 2))

    def test_matches_manual_rms_when_no_resample(self):
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, 'pairs')
            os.makedirs(in_dir)
            sr = 16000
            t = torch.linspace(0, 1.0, sr, dtype=torch.float32)
            noisy = 0.5 * torch.sin(2 * math.pi * 200 * t)
            clean = 0.3 * torch.sin(2 * math.pi * 200 * t)
            _write_pair(os.path.join(in_dir, '000000.wav'), noisy, clean, sr)

            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path))

            payload = torch.load(out_path, weights_only=True)
            # Read back through the same torchaudio.load() path pack() itself
            # uses -- 16-bit WAV round-trip quantization means this is not
            # bit-exact against the original float tensor.
            reloaded, _ = torchaudio.load(os.path.join(in_dir, '000000.wav'))
            expected_noisy = rms_dbfs(reloaded[0])
            expected_clean = rms_dbfs(reloaded[1])
            self.assertAlmostEqual(
                payload['effective_rms_dbfs'][0, 0].item(), expected_noisy, places=2)
            self.assertAlmostEqual(
                payload['effective_rms_dbfs'][0, 1].item(), expected_clean, places=2)

    def test_reflects_post_resample_level_not_pre_resample(self):
        # A narrowband high-frequency tone: most of its energy sits above
        # 8 kHz (the new Nyquist after a 48k->16k downsample), so the
        # post-resample RMS is measurably lower than the pre-resample RMS --
        # exactly the drift this field exists to catch (see README.md).
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, 'pairs')
            os.makedirs(in_dir)
            source_sr = 48000
            t = torch.linspace(0, 1.0, source_sr, dtype=torch.float32)
            high_tone = 0.5 * torch.sin(2 * math.pi * 18000 * t)
            _write_pair(os.path.join(in_dir, '000000.wav'), high_tone, high_tone, source_sr)

            pre_resample_rms = rms_dbfs(high_tone)

            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path, target_sr=16000))

            payload = torch.load(out_path, weights_only=True)
            self.assertEqual(payload['sr'], 16000)
            post_resample_rms = payload['effective_rms_dbfs'][0, 0].item()
            # The 18 kHz tone is almost entirely removed by the anti-alias
            # filter ahead of the 16 kHz Nyquist (8 kHz) -- a large, easily
            # asserted drop, not a rounding-noise-level difference.
            self.assertLess(
                post_resample_rms, pre_resample_rms - 20.0,
                "effective_rms_dbfs must reflect the ACTUAL post-resample "
                "level, not the pre-resample value carried forward"
            )

    def test_omitted_target_sr_still_records_effective_rms(self):
        # No --target-sr at all (pack-as-is): still recorded, since nothing
        # else in the packed payload carries per-sample level information
        # forward from generation time.
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, 'pairs')
            os.makedirs(in_dir)
            sr = 48000
            t = torch.linspace(0, 1.0, sr, dtype=torch.float32)
            noisy = 0.4 * torch.sin(2 * math.pi * 300 * t)
            clean = 0.4 * torch.sin(2 * math.pi * 300 * t)
            _write_pair(os.path.join(in_dir, '000000.wav'), noisy, clean, sr)

            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path, target_sr=None))

            payload = torch.load(out_path, weights_only=True)
            self.assertIn('effective_rms_dbfs', payload)
            self.assertTrue(torch.isfinite(payload['effective_rms_dbfs']).all())


if __name__ == '__main__':
    unittest.main()
