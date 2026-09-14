#!/usr/bin/env python3
"""Experiment 1: render the DFN2 pipeline variants over the 90-case AEC manifest.

Every variant shares one linear-AEC pass per case (the contexts are computed
once), so the only thing that differs between the rendered files is the
post-RES stage:

    mmse             conventional min(G_nr, G_res) MMSE-LSA          (a)
    dfn2_bb          DFN2 estimated on P = E*G_res + CNG, applied on P (b)
    dfn2_ab          DFN2 estimated on E, applied on P               (c)
    dfn2_ab_maskonly (c) with the deep filter disabled (alpha = 0)   (c')
    dfn2_mic         DFN2 estimated on the mic spectrum, applied on P (d)

The blind corpus is 16 kHz and the product host runs at 16 kHz: the AEC and
RES stay on the 16 kHz grid for every variant, and only the DFN2 stage runs
at 48 kHz, through the rate bridge (the twin of the C product path). The
conventional arm is the shipped 16 kHz MMSE-LSA pipeline.

Outputs:  <out>/<variant>/<stem>_ours.wav   (16 kHz, for the benches)
          <out>/run.json                     (parameters, provenance)
Then score with
    python3 ../AEC/python/bench_aecmos.py <out>/<variant> <results>/<variant> \
        --dataset ../AEC/wav/aec_challenge_blind --label <variant>
    python3 pipelines/tools/bench_dnsmos.py <out>/<variant> <results>/<variant>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_AEC_PY = os.path.join(_ROOT, 'lib', 'aec', 'python')
if _AEC_PY not in sys.path:
    sys.path.insert(0, _AEC_PY)

import torch  # noqa: E402

from lib.aec.python.aec import AecConfig, AecMode, AecPreset  # noqa: E402
from pipelines.aec_dfn2_res_pipeline import make_stage, run_dfn2_res  # noqa: E402
from pipelines.aec_nr_pipeline import (  # noqa: E402
    _project_grid, run_aec_linear, run_nr_spectrum, run_res,
)
from pipelines.dfn2_stage import load_dfn2  # noqa: E402

VARIANTS = ('mmse', 'dfn2_bb', 'dfn2_ab', 'dfn2_ab_maskonly', 'dfn2_mic')
SR_NATIVE = 16000
DEFAULT_MANIFEST = os.path.join(os.path.dirname(_ROOT), 'AEC', 'eval', 'manifest_90case.json')
DEFAULT_DATASET = os.path.join(os.path.dirname(_ROOT), 'AEC', 'wav', 'aec_challenge_blind')
DEFAULT_CHECKPOINT = os.path.join(_ROOT, 'AINR', 'DeepFilterNet2', 'output', 'dfn2_best.pth')


def load_manifest(path):
    with open(path) as handle:
        manifest = json.load(handle)
    cases = []
    for scenario, splits in manifest.items():
        for split in ('static', 'movement'):
            for stem in splits.get(split, []):
                cases.append((stem, scenario, split))
    return cases


def render_case(stem, scenario, dataset, variants, out_root, preset, filter_ms,
                assets, atten_lim_db, enable_cng):
    mic_path = os.path.join(dataset, scenario, stem + '_mic.wav')
    lpb_path = os.path.join(dataset, scenario, stem + '_lpb.wav')
    targets = {v: os.path.join(out_root, v, stem + '_ours.wav') for v in variants}
    pending = [v for v in variants if not os.path.exists(targets[v])]
    if not pending:
        return 'cached'
    mic, sr = sf.read(mic_path, dtype='float32')
    ref, _ = sf.read(lpb_path, dtype='float32')
    if mic.ndim > 1:
        mic = mic[:, 0]
    if ref.ndim > 1:
        ref = ref[:, 0]
    n = min(len(mic), len(ref))
    mic, ref = mic[:n], ref[:n]
    if sr != SR_NATIVE:
        raise ValueError(f'{stem}: expected {SR_NATIVE} Hz, got {sr}')
    frame_size, hop_size, _fft = _project_grid(SR_NATIVE)

    filter_length = int(round(filter_ms * SR_NATIVE / 1000.0))
    config = AecConfig.from_preset(
        preset, sample_rate=SR_NATIVE, frame_size=frame_size, hop_size=hop_size,
        mode=AecMode.PBFDKF, filter_length=filter_length,
        enable_shadow=True, enable_res=False, return_res_context=True,
        enable_cng=enable_cng)
    aec_out, contexts = run_aec_linear(mic, ref, config)

    def write(variant, signal):
        os.makedirs(os.path.dirname(targets[variant]), exist_ok=True)
        out = np.asarray(signal, np.float32)[:len(mic)]
        if len(out) < len(mic):
            out = np.pad(out, (0, len(mic) - len(out)))
        sf.write(targets[variant], out, SR_NATIVE)

    if 'mmse' in pending:
        config_res = AecConfig.from_preset(
            preset, sample_rate=SR_NATIVE, frame_size=frame_size, hop_size=hop_size,
            mode=AecMode.PBFDKF, filter_length=filter_length,
            enable_res=True, enable_cng=enable_cng)
        nr_gains = run_nr_spectrum(contexts, SR_NATIVE, nr_preset='balanced',
                                   inject_echo_psd=True)
        tail = np.zeros(len(mic), np.float32)
        tail[:len(aec_out)] = aec_out
        write('mmse', run_res(tail, nr_gains, contexts, config_res,
                              use_res=True, combine='min'))

    dfn_variants = {
        'dfn2_bb': dict(estimate_source='post_res', compose='full'),
        'dfn2_ab': dict(estimate_source='pre_res', compose='full'),
        'dfn2_ab_maskonly': dict(estimate_source='pre_res', compose='mask_only'),
        'dfn2_mic': dict(estimate_source='mic', compose='full'),
    }
    for variant, spec in dfn_variants.items():
        if variant not in pending:
            continue
        stage = make_stage(assets, compose=spec['compose'], atten_lim_db=atten_lim_db)
        write(variant, run_dfn2_res(contexts, config, stage,
                                    estimate_source=spec['estimate_source'],
                                    enable_cng=enable_cng,
                                    output_length=len(mic)))
    return 'rendered'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', default=DEFAULT_MANIFEST)
    ap.add_argument('--dataset', default=DEFAULT_DATASET)
    ap.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--preset', default='balanced', choices=['mild', 'balanced', 'aggressive'])
    ap.add_argument('--filter-ms', type=float, default=52.0,
                    help='PBFDKF filter length in ms (832 taps at 16 kHz = 52 ms)')
    ap.add_argument('--atten-lim', type=float, default=0.0)
    ap.add_argument('--cng', action='store_true',
                    help='enable the comfort-noise fill for every arm (off by default, '
                         'as in the C DFN pipelines; the conventional arm follows suit '
                         'so the comparison stays matched)')
    ap.add_argument('--limit', type=int, default=None, help='first N cases only')
    ap.add_argument('-o', '--output-dir', required=True)
    args = ap.parse_args()

    torch.set_num_threads(1)
    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    for v in variants:
        if v not in VARIANTS:
            ap.error(f'unknown variant {v}')
    cases = load_manifest(args.manifest)
    if args.limit:
        cases = cases[:args.limit]
    os.makedirs(args.output_dir, exist_ok=True)
    preset = AecPreset(args.preset)
    assets = load_dfn2(checkpoint=args.checkpoint) if any(v.startswith('dfn2') for v in variants) else None
    run_info = {
        'variants': variants, 'preset': args.preset, 'filter_ms': args.filter_ms,
        'atten_lim_db': args.atten_lim, 'cng': args.cng, 'checkpoint': args.checkpoint,
        'cases': len(cases), 'sr_host': SR_NATIVE, 'sr_dfn': 48000,
        'sr_scoring': SR_NATIVE, 'no_prealign': True,
        'dfn_rate_bridge': 'pipelines/dfn_rate_bridge.c twin',
    }
    with open(os.path.join(args.output_dir, 'run.json'), 'w') as handle:
        json.dump(run_info, handle, indent=2)

    started = time.time()
    for index, (stem, scenario, split) in enumerate(cases):
        t0 = time.time()
        status = render_case(stem, scenario, args.dataset, variants, args.output_dir,
                             preset, args.filter_ms, assets, args.atten_lim, args.cng)
        print(f'[{index + 1}/{len(cases)}] {scenario}/{split} {stem} {status} '
              f'{time.time() - t0:.1f}s', flush=True)
    print(f'done in {(time.time() - started) / 60:.1f} min -> {args.output_dir}')


if __name__ == '__main__':
    main()
