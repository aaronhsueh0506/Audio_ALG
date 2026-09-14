#!/usr/bin/env python3
"""Experiment 0: is DFN2 usable on 16 kHz material upsampled to 48 kHz?

DFN2 is trained full band at 48 kHz.  A 16 kHz product path would feed it
signals whose top two thirds of the ERB bands are empty (the house
resampler's stopband), so before any resampled pipeline is built this
measures, on VCTK+DEMAND (native 48 kHz, so the true-48 kHz ceiling is free):

    dfn2_48k        DFN2 on the true 48 kHz noisy signal, output decimated
                    to 16 kHz for scoring                            (ceiling)
    dfn2_16k_rt     noisy decimated to 16 kHz (the deployment input), house
                    resampler up to 48 kHz, DFN2, house resampler down  (deployed)
    mmse_16k        the NR C runner (MMSE-LSA balanced) on the 16 kHz input
    noisy_16k       the 16 kHz input, unprocessed                    (floor)
    rt_only         the clean 16 kHz signal through the resampler round
                    trip alone: what the resampler itself costs

Everything is scored at 16 kHz against the clean reference decimated to
16 kHz: PESQ (wide band), STOI, and DNSMOS P.835 (non-intrusive).  The corpus
decimation 48 k -> 16 k is torchaudio's sinc resampler (corpus preparation);
the up/down legs inside dfn2_16k_rt and rt_only are the house resampler, the
thing under test.  The H4 gate: dfn2_16k_rt must beat mmse_16k on DNSMOS
OVRL/BAK and not lose PESQ.

Outputs <out>/<arm>/<stem>.wav (16 kHz) and <out>/scores.json + <out>/summary.md.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402
import torchaudio  # noqa: E402

from pipelines.dfn2_rate_adapter import CResampler  # noqa: E402
from pipelines.dfn2_stage import (  # noqa: E402
    DF_BINS, HOP, N_FFT, SAMPLE_RATE, dfn2_modules, load_dfn2,
)

ARMS = ('dfn2_48k', 'dfn2_16k_rt', 'mmse_16k', 'noisy_16k', 'rt_only')
SR_LOW = 16000
DEFAULT_CORPUS = os.path.join(os.path.dirname(_ROOT), 'NR', 'test_wav', 'vctk_demand')
DEFAULT_CHECKPOINT = os.path.join(_ROOT, 'AINR', 'DeepFilterNet2', 'output', 'dfn2_best.pth')


def _sinc_resample(signal, rate_in, rate_out):
    if rate_in == rate_out:
        return signal.astype(np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(signal, np.float32))
    return torchaudio.functional.resample(tensor, rate_in, rate_out).numpy().astype(np.float32)


class Dfn2Offline:
    """Whole-utterance DFN2 (the model's own inference path): STFT, features,
    heads and compose in one pass, ISTFT."""

    def __init__(self, checkpoint):
        self.assets = load_dfn2(checkpoint=checkpoint)
        self.train, _model, _export = dfn2_modules()
        self.window = torch.hann_window(N_FFT).pow(0.5)

    @torch.no_grad()
    def __call__(self, audio48):
        audio = torch.from_numpy(np.ascontiguousarray(audio48, np.float32))
        spec = torch.stft(audio.unsqueeze(0), N_FFT, HOP, N_FFT, window=self.window,
                          return_complex=True, normalized=True)
        spec, feat_erb, feat_spec, _ = self.train.extract_dfn2_features(
            spec, self.assets.erb_fb, DF_BINS, feature_cfg=self.assets.feature_cfg)
        enhanced, _mask = self.assets.model(spec, feat_erb, feat_spec)
        out = torch.istft(enhanced, N_FFT, HOP, N_FFT, window=self.window,
                          length=audio.shape[-1], normalized=True)
        return out.squeeze(0).numpy().astype(np.float32)


def house_round_trip(signal16):
    up = CResampler(SR_LOW, SAMPLE_RATE)
    down = CResampler(SAMPLE_RATE, SR_LOW)
    high = up.process(signal16)
    return high, down


def find_nr_runner():
    for candidate in sorted(glob.glob(os.path.join(os.path.dirname(_ROOT), 'NR', 'c_impl',
                                                   'bin', 'ne10-*', 'denoise_wav'))):
        return candidate
    raise FileNotFoundError('build NR/c_impl first (make -C NR/c_impl)')


def score_pair(clean16, enhanced16, path16):
    from pesq import pesq
    from pystoi import stoi
    import speechmos.dnsmos as dnsmos
    n = min(len(clean16), len(enhanced16))
    clean16 = clean16[:n]
    enhanced16 = enhanced16[:n]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = {
            'pesq': float(pesq(SR_LOW, clean16, enhanced16, 'wb')),
            'stoi': float(stoi(clean16, enhanced16, SR_LOW, extended=False)),
        }
        mos = dnsmos.run(path16, SR_LOW, return_df=False)
    result.update(sig=float(mos['sig_mos']), bak=float(mos['bak_mos']),
                  ovrl=float(mos['ovrl_mos']), p808=float(mos['p808_mos']))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--corpus', default=DEFAULT_CORPUS)
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    ap.add_argument('--arms', default=','.join(ARMS))
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--nr-mode', default='balanced')
    ap.add_argument('-o', '--output-dir', required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)

    arms = [a.strip() for a in args.arms.split(',') if a.strip()]
    for arm in arms:
        if arm not in ARMS:
            ap.error(f'unknown arm {arm}')
    noisy_files = sorted(glob.glob(os.path.join(args.corpus, 'noisy_testset_wav', '*.wav')))
    if args.limit:
        noisy_files = noisy_files[:args.limit]
    if not noisy_files:
        ap.error('no noisy files found')
    os.makedirs(args.output_dir, exist_ok=True)
    for arm in arms + ['clean_16k', 'noisy_16k']:
        os.makedirs(os.path.join(args.output_dir, arm), exist_ok=True)

    dfn = Dfn2Offline(args.checkpoint) if any(a.startswith('dfn2') for a in arms) else None
    runner = find_nr_runner() if 'mmse_16k' in arms else None
    scores = {arm: {} for arm in arms}
    band_clamp = []
    started = time.time()
    for index, noisy_path in enumerate(noisy_files):
        stem = os.path.splitext(os.path.basename(noisy_path))[0]
        clean_path = os.path.join(args.corpus, 'clean_testset_wav', stem + '.wav')
        noisy48, sr = sf.read(noisy_path, dtype='float32')
        clean48, _ = sf.read(clean_path, dtype='float32')
        if noisy48.ndim > 1:
            noisy48 = noisy48[:, 0]
        if clean48.ndim > 1:
            clean48 = clean48[:, 0]
        if sr != SAMPLE_RATE:
            raise ValueError(f'{stem}: corpus is not 48 kHz')
        clean16 = _sinc_resample(clean48, SAMPLE_RATE, SR_LOW)
        noisy16 = _sinc_resample(noisy48, SAMPLE_RATE, SR_LOW)
        clean16_path = os.path.join(args.output_dir, 'clean_16k', stem + '.wav')
        noisy16_path = os.path.join(args.output_dir, 'noisy_16k', stem + '.wav')
        sf.write(clean16_path, clean16, SR_LOW)
        sf.write(noisy16_path, noisy16, SR_LOW)

        rendered = {}
        if 'noisy_16k' in arms:
            rendered['noisy_16k'] = noisy16
        if 'dfn2_48k' in arms:
            rendered['dfn2_48k'] = _sinc_resample(dfn(noisy48), SAMPLE_RATE, SR_LOW)
        if 'dfn2_16k_rt' in arms:
            high, down = house_round_trip(noisy16)
            enhanced48 = dfn(high)
            rendered['dfn2_16k_rt'] = down.process(enhanced48)
        if 'rt_only' in arms:
            high, down = house_round_trip(clean16)
            rendered['rt_only'] = down.process(high)
        if 'mmse_16k' in arms:
            out_path = os.path.join(args.output_dir, 'mmse_16k', stem + '.wav')
            subprocess.run([runner, noisy16_path, out_path, '--nr-mode', args.nr_mode],
                           check=True, capture_output=True)
            rendered['mmse_16k'], _ = sf.read(out_path, dtype='float32')

        for arm, signal in rendered.items():
            out_path = os.path.join(args.output_dir, arm, stem + '.wav')
            if arm != 'mmse_16k':
                sf.write(out_path, signal.astype(np.float32), SR_LOW)
            # The two resampler legs delay the signal by 32 samples; the
            # intrusive metrics are aligned before scoring.
            delay = 32 if arm in ('dfn2_16k_rt', 'rt_only') else 0
            aligned = signal[delay:] if delay else signal
            scores[arm][stem] = score_pair(clean16, aligned, out_path)
        print(f'[{index + 1}/{len(noisy_files)}] {stem} {time.time() - started:.0f}s', flush=True)

    summary = {}
    for arm in arms:
        rows = list(scores[arm].values())
        summary[arm] = {k: float(np.mean([r[k] for r in rows]))
                        for k in ('pesq', 'stoi', 'sig', 'bak', 'ovrl', 'p808')}
        summary[arm]['n'] = len(rows)
    with open(os.path.join(args.output_dir, 'scores.json'), 'w') as handle:
        json.dump({'summary': summary, 'scores': scores, 'checkpoint': args.checkpoint,
                   'nr_mode': args.nr_mode, 'files': len(noisy_files)}, handle, indent=2)
    lines = ['| arm | n | PESQ | STOI | SIG | BAK | OVRL | P808 |', '|---|---|---|---|---|---|---|---|']
    for arm in arms:
        s = summary[arm]
        lines.append(f"| {arm} | {s['n']} | {s['pesq']:.3f} | {s['stoi']:.4f} | {s['sig']:.3f} "
                     f"| {s['bak']:.3f} | {s['ovrl']:.3f} | {s['p808']:.3f} |")
    with open(os.path.join(args.output_dir, 'summary.md'), 'w') as handle:
        handle.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
