#!/usr/bin/env python3
"""Compare DFN2 feature inputs on the checked-in four-microphone recordings.

All arms share one AEC/BF/RES context stream on the recordings' own rate
(16 kHz); only the DFN2 stage runs at 48 kHz, through the rate bridge that
the C product path uses. The report also carries
the original MMSE-LSA+RES tail and an AEC/BF/RES-only identity baseline. Only
the spectrum used to estimate the DFN2 heads changes between DFN arms:

  raw_capture     raw/near spectrum from the capture-proxy microphone
  linear_capture  linear-AEC spectrum from that microphone, before BF/GSC
  beamformed      the actual post-beam linear spectrum

Every arm applies the resulting heads to the same post-beam, post-RES
spectrum. Each source is evaluated with both the full complex deep-filter
head and a mask-only transfer, because only the beamformed source is phase
coherent with the application spectrum.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
AEC_PY = ROOT / 'lib' / 'aec' / 'python'
if str(AEC_PY) not in sys.path:
    sys.path.insert(0, str(AEC_PY))

from pipelines.aec_dfn2_res_pipeline import make_stage, run_dfn2_res  # noqa: E402
from pipelines.dfn2_stage import load_dfn2  # noqa: E402
from lib.aec.python.aec import AecConfig  # noqa: E402


def _load_dfn_post_module():
    path = ROOT / 'pipelines' / '4ch_aec_bf_dfn_res' / 'pipeline.py'
    spec = importlib.util.spec_from_file_location('dfn_eval_four_post', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dfn_post = _load_dfn_post_module()
four = dfn_post.four_channel   # the one loaded copy of the 4ch reference
VARIANTS = ('raw_capture', 'linear_capture', 'beamformed')
COMPOSE_MODES = ('full', 'mask_only')


def _offset(capture, source):
    a = np.asarray(capture, np.float64)
    b = np.asarray(source, np.float64)
    a -= np.mean(a)
    b -= np.mean(b)
    corr = signal.correlate(a, b, mode='full', method='fft')
    lags = signal.correlation_lags(a.size, b.size, mode='full')
    return int(lags[int(np.argmax(np.abs(corr)))])


def _place(source, lag, length):
    out = np.zeros(length, np.float32)
    dst = max(0, lag)
    src = max(0, -lag)
    count = min(length - dst, len(source) - src)
    if count > 0:
        out[dst:dst + count] = source[src:src + count]
    return out


def _metrics(clean, out, far, native_rate):
    from pesq import pesq
    from pystoi import stoi
    n = min(len(clean), len(out), len(far))
    clean = np.asarray(clean[:n], np.float32)
    out = np.asarray(out[:n], np.float32)
    far = np.asarray(far[:n], np.float32)
    hop = native_rate // 100
    frames = n // hop
    clean_p = np.mean(clean[:frames * hop].reshape(frames, hop) ** 2, axis=1)
    far_p = np.mean(far[:frames * hop].reshape(frames, hop) ** 2, axis=1)
    out_p = np.mean(out[:frames * hop].reshape(frames, hop) ** 2, axis=1)
    clean_thr = max(float(np.percentile(clean_p, 95)) * 1e-4, 1e-10)
    far_thr = max(float(np.percentile(far_p, 95)) * 1e-4, 1e-10)
    near_on = clean_p > clean_thr
    far_on = far_p > far_thr

    def dbfs(mask):
        if not np.any(mask):
            return None
        return float(10.0 * np.log10(float(np.mean(out_p[mask])) + 1e-20))

    return {
        'pesq': float(pesq(native_rate, clean, out, 'wb')),
        'stoi': float(stoi(clean, out, native_rate, extended=False)),
        'far_only_dbfs': dbfs(far_on & ~near_on),
        'near_only_dbfs': dbfs(near_on & ~far_on),
        'double_talk_dbfs': dbfs(near_on & far_on),
    }


def evaluate(case_dir: Path, checkpoint: Path, seconds: float,
             output_dir: Path, identity_model: bool = False,
             compose_modes=COMPOSE_MODES):
    mics, native_rate = sf.read(case_dir / 'unprocessed_4ch.wav',
                                always_2d=True, dtype='float32')
    far_path = case_dir / 'woman(ref).wav'
    far_source, _ = sf.read(far_path, always_2d=True, dtype='float32')
    near_source, _ = sf.read(case_dir / 'man.wav', always_2d=True,
                             dtype='float32')
    far_source = far_source[:, 0]
    near_source = near_source[:, 0]
    timeline_lag = _offset(mics[:, 0], near_source)
    limit = len(mics) if seconds <= 0 else min(len(mics), int(seconds * native_rate))
    mics = mics[:limit]
    far_native = _place(far_source, timeline_lag, limit)

    config = four.FourChannelAecConfig(sample_rate=native_rate)
    front = four.FourChannelAecPipeline(config)
    hop = front.hop_size
    count = min(len(mics), len(far_native))
    count = count // hop * hop
    mics = mics[:count]
    far_native = far_native[:count]

    beamformer = four.EqualWeightBeamformer()
    contexts = []
    candidates = {name: [] for name in VARIANTS}
    linear = np.zeros(count, np.float32)
    for start in range(0, count, hop):
        pre = front.process_pre_beamformer(mics[start:start + hop],
                                           far_native[start:start + hop])
        bf = beamformer.process(pre.linear_hops, pre.contexts)
        frame = front.process_post_beamformer(pre, bf)
        contexts.append(frame.context)
        candidates['raw_capture'].append(
            np.asarray(pre.contexts[config.capture_proxy_channel].near_spec).copy())
        candidates['linear_capture'].append(
            np.asarray(pre.contexts[config.capture_proxy_channel].error_spec).copy())
        candidates['beamformed'].append(np.asarray(frame.context.error_spec).copy())
        linear[start:start + hop] = frame.beamformed

    assets = (load_dfn2(seed=0) if identity_model
              else load_dfn2(checkpoint=str(checkpoint)))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # These two anchors make the experiment answer both questions: whether
    # DFN beats the pipeline it replaces, and what it adds beyond RES alone.
    outputs['original_nr_res'] = dfn_post.post_conventional(
        contexts, native_rate, nr_preset='balanced', enable_cng=False)[:limit]
    post_config = AecConfig(sample_rate=native_rate, frame_size=2 * hop,
                            hop_size=hop, enable_cng=False)
    identity_stage = make_stage(assets, identity=True)
    outputs['aec_bf_res_only'] = run_dfn2_res(
        contexts, post_config, identity_stage,
        enable_cng=False, output_length=count)[:limit]

    dfn_output_names = []
    for name in VARIANTS:
        for compose in compose_modes:
            stage = make_stage(assets, compose=compose,
                               identity=identity_model)
            out = run_dfn2_res(
                contexts, post_config, stage,
                estimate_spectra=candidates[name],
                enable_cng=False, output_length=count)[:limit]
            # Preserve the original full-head name for compatibility and add
            # an explicit suffix only to the diagnostic mask-only arm.
            output_name = (f'dfn_{name}' if compose == 'full'
                           else f'dfn_{name}_{compose}')
            outputs[output_name] = out
            dfn_output_names.append(output_name)

    for name, out in tuple(outputs.items()):
        if len(out) < limit:
            out = np.pad(out, (0, limit - len(out)))
            outputs[name] = out
        sf.write(output_dir / f'{case_dir.name}_{name}.wav', out, native_rate)

    identity_max_abs = None
    if identity_model:
        anchor = outputs['aec_bf_res_only']
        identity_max_abs = max(
            float(np.max(np.abs(outputs[name] - anchor)))
            for name in dfn_output_names)
        if identity_max_abs != 0.0:
            raise RuntimeError(
                'identity smoke failed: a DFN candidate moved the RES-only '
                f'path (max abs {identity_max_abs:.9g})')

    linear = linear[:limit]
    # Use the shared linear-BF path to locate the clean source once. All DFN
    # arms are aligned to the host's own timeline (the bridge delay is
    # compensated offline), so they are scored against the same target.
    output_lag = _offset(linear, near_source)
    clean = _place(near_source, output_lag, limit)
    far = _place(far_source, output_lag, limit)
    return {
        'case': case_dir.name,
        'seconds': limit / native_rate,
        'native_rate': native_rate,
        'model_rate': 48000,
        'rate_bridge': 'pipelines/dfn_rate_bridge.c twin',
        'model_mode': 'identity-smoke' if identity_model else 'checkpoint',
        'compose_modes': list(compose_modes),
        'identity_max_abs_vs_res_only': identity_max_abs,
        'timeline_lag_input': timeline_lag,
        'timeline_lag_output': output_lag,
        'metrics': {name: _metrics(clean, out, far, native_rate)
                    for name, out in outputs.items()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--datasets-root', type=Path,
                    default=ROOT.parent / 'datasets')
    ap.add_argument('--checkpoint', type=Path,
                    default=ROOT / 'AINR' / 'DeepFilterNet2' / 'output' /
                            'dfn2_best.pth')
    ap.add_argument('--cases', default='aec_take_turn,aec_together')
    ap.add_argument('--seconds', type=float, default=12.0,
                    help='0 processes each complete recording')
    ap.add_argument('--identity-model', action='store_true',
                    help='pipeline/resampler smoke test; all DFN arms must '
                         'equal the RES-only baseline, no quality claim')
    ap.add_argument('--compose-modes', default=','.join(COMPOSE_MODES),
                    help='comma-separated subset of full,mask_only')
    ap.add_argument('-o', '--output-dir', type=Path, required=True)
    args = ap.parse_args()
    compose_modes = tuple(mode.strip() for mode in args.compose_modes.split(',')
                          if mode.strip())
    if not compose_modes or any(mode not in COMPOSE_MODES
                                for mode in compose_modes):
        ap.error(f'--compose-modes must be a non-empty subset of {COMPOSE_MODES}')
    results = [evaluate(args.datasets_root / name.strip(), args.checkpoint,
                        args.seconds, args.output_dir, args.identity_model,
                        compose_modes)
               for name in args.cases.split(',') if name.strip()]
    summary = {
        'checkpoint': None if args.identity_model else str(args.checkpoint),
        'identity_model': args.identity_model,
        'results': results,
    }
    with open(args.output_dir / 'summary.json', 'w') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
