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

import hashlib
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

# A real, well-formed 64-hex-char sha256 digest -- pack_dataset.py now
# validates config_hash's FORMAT (not just its value) whenever
# contract_version matches, so any fixture meant to exercise the normal
# "matching, trustworthy contract" path needs a hash that actually looks
# like one, not a short placeholder string.
_VALID_CONFIG_HASH = hashlib.sha256(b'test-fixture-config').hexdigest()


def _make_batch(tmp, indices_and_audio, contract_version=AINR_DATASET_CONTRACT_VERSION,
                 config_hash=_VALID_CONFIG_HASH, write_meta=True,
                 batch_start_idx=None, batch_n_samples=None, sr_override=None):
    """Build a `tmp/pairs/` directory of complete NNNNNN.wav+NNNNNN.json
    samples plus a parent `tmp/meta.json`, mirroring gen_dataset.py's own
    on-disk layout. `indices_and_audio` is a list of (index, noisy, clean,
    sr) tuples. Returns the pairs_dir path.

    `batch_start_idx`/`batch_n_samples` default to exactly what
    `indices_and_audio` puts on disk (a self-consistent, "meta.json
    accurately describes what's on disk" fixture) -- pack_dataset.py's
    fail-closed inventory check requires the on-disk complete-index set to
    equal EXACTLY `range(batch_start_idx, batch_start_idx + batch_n_samples)`
    when contract_version matches. Pass explicit, deliberately WRONG values
    to build a fixture that fails that check on purpose."""
    pairs_dir = os.path.join(tmp, 'pairs')
    os.makedirs(pairs_dir, exist_ok=True)
    for index, noisy, clean, sr in indices_and_audio:
        _save_pair_atomic(pairs_dir, index, noisy, clean, sr)
        _save_metadata_sidecar_atomic(pairs_dir, index, {'index': index, 'snr_db': 0.0})
    if write_meta:
        indices = [t[0] for t in indices_and_audio]
        meta_path = os.path.join(tmp, 'meta.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'contract_version': contract_version,
                'config_hash': config_hash,
                'sr': sr_override if sr_override is not None else (
                    indices_and_audio[0][3] if indices_and_audio else None),
                'batch_start_idx': (
                    batch_start_idx if batch_start_idx is not None
                    else (min(indices) if indices else 0)),
                'batch_n_samples': (
                    batch_n_samples if batch_n_samples is not None
                    else len(indices_and_audio)),
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

    def test_index_gap_rejected_even_with_allow_index_gaps_in_versioned_mode(self):
        # In the default (contract_version-matching) mode, the fail-closed
        # inventory check requires the on-disk complete indices to equal
        # EXACTLY meta.json's declared range -- there is no "curated
        # subset" escape hatch for a batch that's supposed to be trusted.
        # --allow-index-gaps alone must NOT be enough to bypass this.
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [
                (0, noisy, clean, sr), (1, noisy, clean, sr),
                (3, noisy, clean, sr),   # index 2 missing
            ])
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path, allow_index_gaps=True))

    def test_index_gap_accepted_with_allow_index_gaps_in_unversioned_mode(self):
        # The plain internal-contiguity check + --allow-index-gaps escape
        # hatch is a legacy fallback for when there's no trustworthy
        # meta.json to check an exact range against (--allow-unversioned-
        # input, e.g. an old/foreign contract_version).
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(
                tmp, [
                    (0, noisy, clean, sr), (1, noisy, clean, sr),
                    (3, noisy, clean, sr),
                ],
                contract_version=AINR_DATASET_CONTRACT_VERSION - 1)
            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path, allow_unversioned_input=True,
                             allow_index_gaps=True))
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


class DeclaredInventoryMismatchRejectionTest(unittest.TestCase):
    """Release blocker: pack_dataset.py used to trust whatever complete
    pairs it found on disk, never cross-checking that count/range against
    what meta.json itself declares this batch to contain (batch_start_idx/
    batch_n_samples). A batch that's still mid-generation (interrupted
    before its last round's meta.json save) or was corrupted/tampered with
    after the fact would pack silently either way."""

    def test_disk_has_more_samples_than_meta_declares_is_rejected(self):
        # meta.json declares only 1 sample, but 2 complete pairs actually
        # exist on disk (e.g. generation continued past the last
        # successful meta.json save, or two runs wrote into one directory).
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(
                tmp, [(0, noisy, clean, sr), (1, noisy, clean, sr)],
                batch_start_idx=0, batch_n_samples=1)
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path))
            self.assertFalse(os.path.exists(out_path))

    def test_disk_has_fewer_samples_than_meta_declares_is_rejected(self):
        # meta.json declares 2 samples, but only 1 complete pair actually
        # exists (e.g. a sample was deleted/corrupted after meta.json was
        # last saved).
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(
                tmp, [(0, noisy, clean, sr)],
                batch_start_idx=0, batch_n_samples=2)
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path))
            self.assertFalse(os.path.exists(out_path))

    def test_allow_unversioned_input_bypasses_the_exact_range_check(self):
        # The escape hatch is --allow-unversioned-input (an old/foreign
        # contract_version), not a dedicated flag for this check alone --
        # a mismatched-but-present meta.json falls back to the plain
        # internal-contiguity check instead.
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(
                tmp, [(0, noisy, clean, sr), (1, noisy, clean, sr)],
                contract_version=AINR_DATASET_CONTRACT_VERSION - 1,
                batch_start_idx=0, batch_n_samples=1)
            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path, allow_unversioned_input=True))
            payload = torch.load(out_path, weights_only=True)
            self.assertEqual(payload['sample_indices'], [0, 1])


class SidecarIntegrityRejectionTest(unittest.TestCase):
    """A sidecar that doesn't parse, or whose own 'index' field disagrees
    with its filename, is basic file-integrity corruption -- checked
    upfront (before any WAV is loaded) and always a hard error, in every
    mode, unlike a contract/config mismatch."""

    def test_corrupt_sidecar_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)])
            _, json_path = _sample_paths(in_dir, 0)
            Path(json_path).write_text('{not valid json')
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path))

    def test_sidecar_index_field_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)])
            _, json_path = _sample_paths(in_dir, 0)
            json_path_obj = Path(json_path)
            sidecar = json.loads(json_path_obj.read_text())
            sidecar['index'] = 999
            json_path_obj.write_text(json.dumps(sidecar))
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path))

    def test_no_bypass_via_allow_unversioned_input(self):
        # Even the legacy/unversioned escape hatch must not tolerate
        # corrupt metadata -- it relaxes contract/config trust, not basic
        # file integrity.
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(
                tmp, [(0, noisy, clean, sr)],
                contract_version=AINR_DATASET_CONTRACT_VERSION - 1)
            _, json_path = _sample_paths(in_dir, 0)
            Path(json_path).write_text('{not valid json')
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path, allow_unversioned_input=True))


class ConfigHashFormatValidationTest(unittest.TestCase):
    def test_malformed_config_hash_is_rejected_when_contract_version_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)],
                                  config_hash='not-a-real-sha256')
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path))

    def test_malformed_config_hash_tolerated_under_allow_unversioned_input(self):
        # A mismatched contract_version already means the batch can't be
        # trusted as-is -- --allow-unversioned-input's whole point is to
        # pack it anyway, so an also-malformed config_hash on that same
        # untrusted meta.json doesn't add a new failure mode.
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(
                tmp, [(0, noisy, clean, sr)],
                contract_version=AINR_DATASET_CONTRACT_VERSION - 1,
                config_hash='not-a-real-sha256')
            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path, allow_unversioned_input=True))
            payload = torch.load(out_path, weights_only=True)
            self.assertEqual(payload['config_hash'], 'not-a-real-sha256')


class MalformedMetaJsonRejectionTest(unittest.TestCase):
    """Found by adversarial review: a meta.json that's corrupt in ways the
    developer didn't anticipate must fail closed with a clean
    DatasetContractError, not crash with a raw json.JSONDecodeError/
    TypeError -- exactly like a genuinely alien contract_version does."""

    def test_meta_json_body_is_not_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)])
            meta_path = os.path.join(tmp, 'meta.json')
            Path(meta_path).write_text('{not valid json')
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path))

    def test_batch_start_idx_wrong_type_is_rejected_not_crashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)])
            meta_path = os.path.join(tmp, 'meta.json')
            meta = json.loads(Path(meta_path).read_text())
            meta['batch_start_idx'] = "0"  # string, not int
            Path(meta_path).write_text(json.dumps(meta))
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path))

    def test_batch_n_samples_wrong_type_is_rejected_not_crashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)])
            meta_path = os.path.join(tmp, 'meta.json')
            meta = json.loads(Path(meta_path).read_text())
            meta['batch_n_samples'] = 1.5  # float, not int
            Path(meta_path).write_text(json.dumps(meta))
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path))

    def test_sr_wrong_type_is_rejected_not_crashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            sr = 16000
            noisy, clean = _tone(sr, 200), _tone(sr, 300)
            in_dir = _make_batch(tmp, [(0, noisy, clean, sr)])
            meta_path = os.path.join(tmp, 'meta.json')
            meta = json.loads(Path(meta_path).read_text())
            meta['sr'] = [16000]  # list, not int
            Path(meta_path).write_text(json.dumps(meta))
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
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
    def test_mismatched_native_rate_without_target_sr_is_hard_rejected_in_versioned_mode(self):
        # A validated/contract-versioned batch should never contain a
        # sample rate mismatch -- pack() now stops immediately rather than
        # silently excluding it (silent exclusion would also reopen an
        # index gap the earlier inventory check already closed).
        with tempfile.TemporaryDirectory() as tmp:
            noisy16, clean16 = _tone(16000, 200), _tone(16000, 300)
            noisy48, clean48 = _tone(48000, 200), _tone(48000, 300)
            in_dir = _make_batch(tmp, [
                (0, noisy16, clean16, 16000),
                (1, noisy48, clean48, 48000),  # different native rate, no --target-sr
            ])
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path))

    def test_mismatched_native_rate_without_target_sr_is_excluded_in_unversioned_legacy_mode(self):
        # --allow-unversioned-input (no trustworthy meta.json/contract)
        # falls back to the pre-fail-closed behavior: soft-exclude and warn
        # instead of stopping the whole pack.
        with tempfile.TemporaryDirectory() as tmp:
            noisy16, clean16 = _tone(16000, 200), _tone(16000, 300)
            noisy48, clean48 = _tone(48000, 200), _tone(48000, 300)
            in_dir = _make_batch(
                tmp, [
                    (0, noisy16, clean16, 16000),
                    (1, noisy48, clean48, 48000),
                ],
                contract_version=AINR_DATASET_CONTRACT_VERSION - 1)
            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path, allow_unversioned_input=True))
            payload = torch.load(out_path, weights_only=True)
            # index 1 excluded (rate mismatch, no forced resample to reconcile it)
            self.assertEqual(payload['sample_indices'], [0])
            self.assertEqual(payload['sr'], 16000)

    def test_mismatched_native_rate_is_hard_rejected_in_versioned_mode_even_with_target_sr(self):
        # A contract-versioned batch is by construction single-native-rate
        # (gen_dataset.py only ever generates one rate per invocation) --
        # --target-sr reconciles the OUTPUT rate, it must not become a
        # blanket excuse to stop verifying a file's native rate actually
        # matches what this batch is supposed to contain (exactly the
        # "different run's file mixed in" tampering this fix targets).
        with tempfile.TemporaryDirectory() as tmp:
            noisy16, clean16 = _tone(16000, 200), _tone(16000, 300)
            noisy48, clean48 = _tone(48000, 200), _tone(48000, 300)
            in_dir = _make_batch(tmp, [
                (0, noisy16, clean16, 16000),
                (1, noisy48, clean48, 48000),
            ])
            out_path = os.path.join(tmp, 'packed.pt')
            with self.assertRaises(DatasetContractError):
                pack(_pack_args(in_dir, out_path, target_sr=16000))

    def test_mismatched_native_rate_is_fine_with_target_sr_in_unversioned_legacy_mode(self):
        # The legacy/unversioned fallback path (no trustworthy meta.json to
        # check a native rate against) may still legitimately combine
        # heterogeneous native rates when --target-sr reconciles them.
        with tempfile.TemporaryDirectory() as tmp:
            noisy16, clean16 = _tone(16000, 200), _tone(16000, 300)
            noisy48, clean48 = _tone(48000, 200), _tone(48000, 300)
            in_dir = _make_batch(
                tmp, [
                    (0, noisy16, clean16, 16000),
                    (1, noisy48, clean48, 48000),
                ],
                contract_version=AINR_DATASET_CONTRACT_VERSION - 1)
            out_path = os.path.join(tmp, 'packed.pt')
            pack(_pack_args(in_dir, out_path, target_sr=16000,
                             allow_unversioned_input=True))
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
