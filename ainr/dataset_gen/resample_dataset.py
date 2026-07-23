# -*- coding: utf-8 -*-
"""
Model-side resample: "pack-stage resample once" from a dataset produced by
gen_dataset.py down/up to a specific model's target sample rate, so per-epoch
training dataloaders never pay a resample cost. See README.md for the overall
generation-rate design.

Anti-aliasing: uses torchaudio.functional.resample with the Kaiser-windowed
sinc-interpolation kernel ("sinc_interp_kaiser") — the "kaiser_best" preset
from torchaudio's own resampling tutorial/benchmark:
    lowpass_filter_width = 64
    rolloff              = 0.9475937167399596
    beta                 = 14.769656459379492
This is deliberately the highest-quality preset torchaudio ships (vs. the
faster/cheaper "kaiser_fast" or the plain sinc_interp_hann default): a wide,
steep, well-behaved lowpass is applied before decimation so this step is
explicitly anti-aliased. Since it runs once offline (not per epoch), the
extra compute cost relative to a cheaper kernel is irrelevant — quality was
prioritized over speed. --quality fast (kaiser_fast) is offered only for
quick smoke tests.

Usage:
    python3 resample_dataset.py --input data_48k/ --output data_16k/ \
        --target-sr 16000 --workers 4

Preserves:
    - directory structure (relative to --input) and filenames
    - 2-channel (noisy, clean) pair layout produced by gen_dataset.py
    - meta.json (sr field updated to --target-sr; original rate recorded as
      source_sr)
"""

import argparse
import glob
import json
import os

import torch.utils.data as data
import torchaudio
import tqdm


# torchaudio "kaiser_best" resampling preset (see module docstring).
KAISER_BEST = dict(
    lowpass_filter_width=64,
    rolloff=0.9475937167399596,
    resampling_method="sinc_interp_kaiser",
    beta=14.769656459379492,
)
# Cheaper preset, offered via --quality fast for quick iteration/smoke-testing.
KAISER_FAST = dict(
    lowpass_filter_width=16,
    rolloff=0.85,
    resampling_method="sinc_interp_kaiser",
    beta=8.555504641634386,
)
CLIP_GUARD = 0.999


def resampled_num_frames(num_frames: int, source_sr: int, target_sr: int) -> int:
    """Match torchaudio resampling's ceil-based output-length contract."""
    return (num_frames * target_sr + source_sr - 1) // source_sr


class _ResampleJob(data.Dataset):
    """One item = one input WAV file → resample → write to mirrored output path.

    Loading, resampling, clip-guarding, and writing all happen inside
    __getitem__ so the heavy lifting is spread across DataLoader workers
    (mirrors gen_dataset.py's worker-does-the-work / main-process-aggregates
    pattern, except here the write itself also happens in the worker since
    output path is a pure function of input path — no shared file-name
    counter to serialize on).
    """

    def __init__(self, in_paths, out_paths, target_sr, resample_kwargs,
                clip_guard: float = CLIP_GUARD):
        self.in_paths = in_paths
        self.out_paths = out_paths
        self.target_sr = target_sr
        self.resample_kwargs = resample_kwargs
        self.clip_guard = clip_guard

    def __len__(self):
        return len(self.in_paths)

    def __getitem__(self, idx):
        in_path = self.in_paths[idx]
        out_path = self.out_paths[idx]
        audio, sr = torchaudio.load(in_path)   # (C, T)

        peak_before = audio.abs().max().item()

        if sr != self.target_sr:
            audio = torchaudio.functional.resample(
                audio, sr, self.target_sr, **self.resample_kwargs)

        peak_after = audio.abs().max().item()
        clipped = False
        if peak_after > self.clip_guard:
            # Kaiser-sinc resampling can slightly overshoot the original peak
            # (Gibbs-like ringing near sharp transients/near-full-scale
            # content). Scale the whole (noisy, clean) pair down together —
            # preserves their relative level / SNR — rather than let
            # torchaudio.save silently clip on the int16 write.
            audio = audio * (self.clip_guard / peak_after)
            peak_after = self.clip_guard
            clipped = True

        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        torchaudio.save(out_path, audio, self.target_sr, bits_per_sample=16)

        return {
            'file': in_path,
            'sr_in': sr,
            'peak_before': peak_before,
            'peak_after': peak_after,
            'rescaled_for_clip_guard': clipped,
        }


def _scan_wavs(input_dir):
    """Recursively find WAV pairs under a generated dataset directory."""
    return sorted(glob.glob(os.path.join(input_dir, '**', '*.wav'), recursive=True))


def _identity_collate(batch):
    """batch_size=1 passthrough. Must be a module-level (picklable) function,
    not a lambda — DataLoader workers are spawned via multiprocessing on
    macOS (spawn, not fork), which cannot pickle a local/lambda collate_fn."""
    return batch[0]


def resample_dataset(args):
    in_files = _scan_wavs(args.input)
    if not in_files:
        raise FileNotFoundError(f"No .wav files found under {args.input}")

    out_files = [
        os.path.join(args.output, os.path.relpath(p, args.input))
        for p in in_files
    ]

    resample_kwargs = KAISER_FAST if args.quality == 'fast' else KAISER_BEST
    print(f"Resampling {len(in_files)} file(s) → {args.target_sr} Hz "
          f"(quality={args.quality}, workers={args.workers})")
    print(f"  {args.input} → {args.output}")

    os.makedirs(args.output, exist_ok=True)

    job = _ResampleJob(in_files, out_files, args.target_sr, resample_kwargs)

    n_clipped = 0
    peak_max = 0.0

    if args.workers > 0:
        loader = data.DataLoader(
            job, batch_size=1, shuffle=False,
            num_workers=args.workers, collate_fn=_identity_collate,
        )
        results = tqdm.tqdm(loader, total=len(job), desc="Resampling")
    else:
        results = (job[idx] for idx in tqdm.tqdm(range(len(job)), desc="Resampling"))

    for result in results:
        peak_max = max(peak_max, result['peak_after'])
        if result['rescaled_for_clip_guard']:
            n_clipped += 1

    # Carry over / update meta.json if present at the input root
    meta_path_in = os.path.join(args.input, 'meta.json')
    if os.path.isfile(meta_path_in):
        with open(meta_path_in) as f:
            meta = json.load(f)
        meta['source_sr'] = meta.get('sr')
        meta['sr'] = args.target_sr
        if meta.get('segment_samples') is not None and meta['source_sr']:
            meta['segment_samples'] = resampled_num_frames(
                int(meta['segment_samples']),
                int(meta['source_sr']),
                args.target_sr,
            )
        elif meta.get('segment_sec') is not None:
            meta['segment_samples'] = int(
                round(float(meta['segment_sec']) * args.target_sr)
            )
        meta['resampled_from'] = os.path.abspath(args.input)
        meta['resample_quality'] = args.quality
        with open(os.path.join(args.output, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"  meta.json: sr {meta['source_sr']} -> {meta['sr']} (carried over, updated)")

    clip_note = f", {n_clipped} file(s) rescaled to avoid clipping" if n_clipped else ""
    print(f"\nDone. {len(in_files)} file(s) -> {args.output}/ "
          f"(peak level max={peak_max:.4f}{clip_note})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Resample a generated (noisy, clean) WAV-pair dataset to a '
                    'model-specific target sample rate (pack-stage, resample-once)')
    parser.add_argument('--input', required=True,
                        help='Input dataset root (e.g. data_48k); scanned '
                             'recursively for *.wav')
    parser.add_argument('--output', required=True,
                        help='Output directory for the resampled copy (mirrors input '
                            'directory structure and filenames)')
    parser.add_argument('--target-sr', type=int, required=True,
                        help='Target sample rate in Hz (e.g. 16000 for RNNoise-ERB)')
    parser.add_argument('--workers', type=int, default=4,
                        help='DataLoader workers (default: 4, 0=single process)')
    parser.add_argument('--quality', choices=['best', 'fast'], default='best',
                        help='Kaiser-window resampling quality preset (default: best). '
                            '"fast" trades some anti-aliasing sharpness for speed — '
                            'only useful for quick smoke tests, since this step runs '
                            'once offline, not per epoch.')
    args = parser.parse_args()
    resample_dataset(args)
