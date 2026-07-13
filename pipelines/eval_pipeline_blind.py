#!/usr/bin/env python3
"""Run AEC+NR+RES pipeline on AEC Challenge blind test and score with AECMOS."""
import sys, os, argparse
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor, as_completed

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)                                       # lib.* + pipelines.*
sys.path.insert(0, os.path.join(_ROOT, 'lib', 'aec', 'python')) # aec + modules
from lib.aec.python.aec import AEC, AecConfig, AecMode, AecPreset
from pipelines.aec_nr_pipeline import run_aec_linear, run_nr, run_res


def process_case(mic_path, lpb_path, out_path, preset, fl, nr_preset):
    mic, sr = sf.read(mic_path, dtype='float32')
    ref, _ = sf.read(lpb_path, dtype='float32')
    if mic.ndim > 1: mic = mic[:, 0]
    if ref.ndim > 1: ref = ref[:, 0]
    n = min(len(mic), len(ref))
    mic, ref = mic[:n], ref[:n]
    # No offline pre-align: the AEC's online matched-filter delay estimator
    # (enable_delay_est) handles alignment, matching production.

    # AEC config
    config = AecConfig.from_preset(preset,
        sample_rate=sr, mode=AecMode.PBFDKF,
        filter_length=fl,
        enable_shadow=True, enable_res=False,
        return_res_context=True, enable_cng=True)

    # Stage 1: AEC (linear only)
    aec_out, contexts = run_aec_linear(mic, ref, config)

    # Stage 2: NR
    nr_out, nr_gains = run_nr(aec_out, sr, nr_preset=nr_preset, return_gain=True)

    # Stage 3: RES (A_min_pl: g_total = min(G_nr, G_res))
    config_res = AecConfig.from_preset(preset,
        sample_rate=sr, mode=AecMode.PBFDKF,
        filter_length=fl, enable_res=True, enable_cng=True)
    final_out = run_res(nr_out, nr_gains, contexts, config_res,
                        use_res=True, combine='min', ne_floor=0.4, ne_gate='both')

    sf.write(out_path, final_out, sr)
    return out_path


def run_scenario(base_dir, subdir, out_dir, preset, fl, nr_preset):
    sc_dir = os.path.join(base_dir, subdir)
    if not os.path.isdir(sc_dir):
        return f"No {subdir} directory"

    mic_files = sorted([f for f in os.listdir(sc_dir) if f.endswith('_mic.wav')])
    results = []
    for mf in mic_files:
        lpb_f = mf.replace('_mic.wav', '_lpb.wav')
        out_suffix = mf.replace('_mic.wav', '')
        mic_path = os.path.join(sc_dir, mf)
        lpb_path = os.path.join(sc_dir, lpb_f)
        out_path = os.path.join(out_dir, f"{out_suffix}_ours.wav")

        if not os.path.exists(lpb_path):
            continue
        if os.path.exists(out_path):
            results.append(out_path)
            continue

        process_case(mic_path, lpb_path, out_path, preset, fl, nr_preset)
        results.append(out_path)

    return f"{subdir}: {len(results)} cases"


def main():
    parser = argparse.ArgumentParser(description='AEC+NR+RES pipeline blind test')
    parser.add_argument('dataset_dir', help='aec_challenge_blind/ directory')
    parser.add_argument('--preset', default='balanced',
                        choices=['gentle', 'balanced', 'aggressive'])
    parser.add_argument('--filter', type=int, default=512)
    parser.add_argument('--nr-preset', default='balanced',
                        choices=['mild', 'balanced', 'aggressive'])
    parser.add_argument('-o', '--output-dir', default=None)
    args = parser.parse_args()

    base_dir = os.path.abspath(args.dataset_dir)
    out_dir = args.output_dir or os.path.join(base_dir, 'output_pipeline')
    os.makedirs(out_dir, exist_ok=True)

    preset = AecPreset(args.preset)

    for subdir in ['farend_singletalk', 'nearend_singletalk', 'doubletalk']:
        print(f"\n=== {subdir} ===")
        result = run_scenario(base_dir, subdir, out_dir, preset, args.filter, args.nr_preset)
        print(result)

    print(f"\nOutput saved to {out_dir}")


if __name__ == '__main__':
    main()
