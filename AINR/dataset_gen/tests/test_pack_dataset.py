"""Regression tests for pack_dataset.py's contract enforcement and
effective_rms_dbfs recording.

pack_dataset.py used to accept ANY *.wav found by a recursive glob --
including a tmp.NNNNNN.wav left behind by a crashed gen_dataset.py write,
or an orphan WAV with no metadata sidecar -- silently treating incomplete/
crash artifacts as real training samples. It also never validated the
source batch's meta.json contract, never re-measured level after a
--target-sr resample against the actually-stored (post-dtype-cast) audio,
and wrote its output non-atomically. These tests pin the fixes for all of
that -- see the pack_dataset.py commit for the full list.
"""

import json
import math
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torchaudio

from dataset_gen.dataset import rms_dbfs
from dataset_gen.gen_dataset import (
    AINR_DATASET_CONTRACT_VERSION,
    _sample_paths,
    _save_metadata_sidecar_atomic,
    _save_pair_atomic,
)
from dataset_gen.pack_dataset import DatasetContractError, pack


def _make_batch(tmp, indices_and_audio, contract_version=AINR_DATASET_CONTRACT_VERSION,
                 config_hash='testhash0123456789', write_meta=True):
    """Build a `tmp/pairs/` directory of complete NNNNNN.wav+NNNNNN.json
    samples plus a parent `tmp/meta.json`, mirroring gen_dataset.py's own
    on-disk layout. `indices_and_audio` is a list of (index, noisy, clean,
    sr) tuples. Returns the pairs_dir path."""
    pairs_dir = os.path.join(tmp, 'pairs')
    os.makedirs(pairs_dir, exist_ok=True)
    for index, noisy, clean, sr in indices_and_audio:
        _save_pair_atomic(pairs_dir, index, noisy, clean, sr)
        _save_metadata_sidecar_atomic(pairs_dir, index, {'snr_db': 0.0})
    if write_meta:
        meta_path = os.path.join(tmp, 'meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'contract_version': contract_version,
                'config_hash': config_hash,
                'sr': indices_and_audio[0][3] if indices_and_audio else None,
            }, f)
    return pairs_dir


def _tone(sr, freq=200, amp=0.5, seconds=1.0):
    t = torch.linspace(0, seconds, int(sr * seconds), dtype=torch.float32)
    return amp * torch.sin(2 * math.pi * freq * t)


def _pack_args(input_dir, output_path, target_sr=None, quality='best',
               dtype='float32', allow_unversioned_input=False,
               allow_index_gaps=False):
    return types.SimpleNamespace(
        input=input_dir, output=output_path, dtype=dtype,
        target_sr=target_sr, quality=quality,
        allow_unversioned_input=allow_unversioned_input,
        allow_index_gaps=allow_index_gaps,
    )


class EffectiveRmsDbfsTest(unittest.TestCase):
    def test_present_and_correctly_shaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200, 0.5), _tone(sr, 200, 0.3)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr), (1, noisy, clean, sr)])

            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path))

            payload = torch.load(out_path, weights_only=True)
            self.assertIn('effective_rms_dbfs', payload)
            self.assertEqual(tuple(payload['effective_rms_dbfs'].shape), (2, 2))

    def test_matches_manual_rms_when_no_resample(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200, 0.5), _tone(sr, 200, 0.3)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)])

            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path))

            payload = torch.load(out_path, weights_only=True)
            # Read back through the same torchaudio.load() path pack() itself
            # uses -- 16-bit WAV round-trip quantization means this is not
            # bit-exact against the original float tensor.
            reloaded, _ = torchaudio.load(_sample_paths(in_dir, 0)[0])
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
            source_sr = 48000
            high_tone = _tone(source_sr, 18000, 0.5)
            in_dir = _make_batch(tmp, [(0, high_tone, high_tone, source_sr)])

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
            sr = 48000
            noisy = clean = _tone(sr, 300, 0.4)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)])

            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path, target_sr=None))

            payload = torch.load(out_path, weights_only=True)
            self.assertIn('effective_rms_dbfs', payload)
            self.assertTrue(torch.isfinite(payload['effective_rms_dbfs']).all())

    def test_measured_on_packed_dtype_not_pre_cast_float32(self):
        # float16 has ~3 decimal digits of precision; measuring on the
        # pre-cast float32 tensor instead of data[i] (what's ACTUALLY
        # stored) would describe a signal the packed payload doesn't
        # contain. Recompute RMS directly from the packed float16 data and
        # compare against effective_rms_dbfs -- if the field were measured
        # pre-cast, this comparison would (rarely) disagree past float16's
        # ~3-decimal-digit precision.
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy = clean = _tone(sr, 200, 0.5)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)])
            out_path = os.path.join(tmp, 'packed.pt')

            pack(_pack_args(in_dir, out_path, dtype='float16'))
            payload = torch.load(out_path, weights_only=True)
            # Recompute RMS directly from the packed (float16) data and
            # compare -- this must match to float32 precision even though
            # data itself was rounded to float16, proving the measurement
            # was taken AFTER, not before, that rounding.
            expected = rms_dbfs(payload['data'][0, 0].float())
            self.assertAlmostEqual(
                payload['effective_rms_dbfs'][0, 0].item(), expected, places=4)


class CompletePairScanTest(unittest.TestCase):
    """The release-blocking bug this fixes: a naive recursive glob('*.wav')
    also matched tmp.NNNNNN.wav (an in-progress/crashed write) and any
    orphan half-pair, silently packing crash artifacts as real samples."""

    def test_temp_file_from_a_crashed_write_is_never_packed(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)])
            # Reproduce a crash mid-write: a tmp.NNNNNN.wav left behind by
            # _save_pair_atomic's temp-file step, never renamed into place.
            wav1, _ = _sample_paths(in_dir, 1)
            from dataset_gen.gen_dataset import _tmp_path
            torchaudio.save(_tmp_path(wav1), torch.stack([noisy, clean]), sr,
                             bits_per_sample=16)

            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path))

            payload = torch.load(out_path, weights_only=True)
            self.assertEqual(payload['n_samples'], 1)
            self.assertEqual(payload['sample_indices'], [0])

    def test_orphan_wav_without_sidecar_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)])
            _save_pair_atomic(in_dir, 1, noisy, clean, sr)  # no sidecar for index 1
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path))

    def test_orphan_json_without_wav_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)])
            _save_metadata_sidecar_atomic(in_dir, 1, {})  # no audio for index 1
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path))

    def test_index_gap_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [
                (0, noisy, clean, sr), (1, noisy, clean, sr),
                (3, noisy, clean, sr),   # index 2 missing
            ])
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path))

    def test_index_gap_accepted_with_allow_index_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [
                (0, noisy, clean, sr), (1, noisy, clean, sr),
                (3, noisy, clean, sr),
            ])
            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path, allow_index_gaps=True))
            payload = torch.load(out_path, weights_only=True)
            self.assertEqual(payload['sample_indices'], [0, 1, 3])
            self.assertEqual(payload['n_samples'], 3)

    def test_no_complete_pairs_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            # A valid meta.json but zero samples -- isolates "no complete
            # pairs" from the (separately tested) "no meta.json" rejection.
            in_dir = _make_batch(tmp, [])
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(FileNotFoundError):
                pack(_pack_args(in_dir, out_path))


class BatchContractValidationTest(unittest.TestCase):
    def test_missing_meta_json_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)], write_meta=False)
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path))

    def test_missing_meta_json_accepted_with_allow_unversioned_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)], write_meta=False)
            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path, allow_unversioned_input=True))
            payload = torch.load(out_path, weights_only=True)
            self.assertIsNone(payload['contract_version'])
            self.assertIsNone(payload['config_hash'])

    def test_contract_version_mismatch_is_rejected_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(
                tmp, [(0, noisy, clean, sr)],
                contract_version=AINR_DATASET_CONTRACT_VERSION - 1)
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path))

    def test_contract_version_mismatch_accepted_with_allow_unversioned_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(
                tmp, [(0, noisy, clean, sr)],
                contract_version=AINR_DATASET_CONTRACT_VERSION - 1,
                config_hash='old_hash')
            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path, allow_unversioned_input=True))
            payload = torch.load(out_path, weights_only=True)
            self.assertEqual(payload['contract_version'], AINR_DATASET_CONTRACT_VERSION - 1)
            self.assertEqual(payload['config_hash'], 'old_hash')

    def test_matching_contract_carries_version_and_hash_into_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(
                tmp, [(0, noisy, clean, sr)], config_hash='deadbeef' * 8)
            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path))
            payload = torch.load(out_path, weights_only=True)
            self.assertEqual(payload['contract_version'], AINR_DATASET_CONTRACT_VERSION)
            self.assertEqual(payload['config_hash'], 'deadbeef' * 8)


class SampleRateConsistencyTest(unittest.TestCase):
    def test_mismatched_native_rate_without_target_sr_is_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            noisy16, clean16 = _tone(16000, 200), _tone(16000, 300)
            noisy48, clean48 = _tone(48000, 200), _tone(48000, 300)
            in_dir = _make_batch(tmp, [
                (0, noisy16, clean16, 16000),
                (1, noisy48, clean48, 48000),  # different native rate, no --target-sr
            ])
            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path))
            payload = torch.load(out_path, weights_only=True)
            # index 1 excluded (rate mismatch, no forced resample to reconcile it)
            self.assertEqual(payload['sample_indices'], [0])
            self.assertEqual(payload['sr'], 16000)

    def test_mismatched_native_rate_is_fine_when_target_sr_forces_resample(self):
        with tempfile.TemporaryDirectory() as tmp:
            noisy16, clean16 = _tone(16000, 200), _tone(16000, 300)
            noisy48, clean48 = _tone(48000, 200), _tone(48000, 300)
            in_dir = _make_batch(tmp, [
                (0, noisy16, clean16, 16000),
                (1, noisy48, clean48, 48000),
            ])
            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path, target_sr=16000))
            payload = torch.load(out_path, weights_only=True)
            self.assertEqual(payload['sample_indices'], [0, 1])
            self.assertEqual(payload['sr'], 16000)


class SampleIndicesPayloadTest(unittest.TestCase):
    def test_sample_indices_trace_packed_rows_to_source_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            samples = [(i, _tone(sr, 200 + i), _tone(sr, 300 + i), sr) for i in range(4)]
            in_dir = _make_batch(tmp, samples)
            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path))
            payload = torch.load(out_path, weights_only=True)
            self.assertEqual(payload['sample_indices'], [0, 1, 2, 3])
            self.assertEqual(len(payload['sample_indices']), payload['n_samples'])


class AtomicPackedWriteTest(unittest.TestCase):
    def test_success_leaves_no_tmp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)])
            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path))
            self.assertTrue(Path(out_path).exists())
            self.assertFalse(Path(out_path + '.tmp').exists())

    def test_failure_leaves_no_final_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)])
            out_path = os.path.join(tmp, 'packed.pt')
            with patch('dataset_gen.pack_dataset.torch.save',
                       side_effect=RuntimeError('disk full')):
                with self.assertRaises(RuntimeError):
                    pack(_pack_args(in_dir, out_path))
            self.assertFalse(Path(out_path).exists())


if __name__ == '__main__':
    unittest.main()
