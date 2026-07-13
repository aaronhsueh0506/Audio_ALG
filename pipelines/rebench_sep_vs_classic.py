#!/usr/bin/env python3
"""800-case A/B: separated freq pipeline vs classic, matched front-end.

Renders every blind-corpus case through BOTH arms with identical load (no
offline pre-align — the AEC's online matched-filter delay estimator handles
alignment, matching production), writing <stem>_ours.wav into two dirs so the
AEC repo's bench_aecmos.py can score each:

  classic   : AEC(enable_res=True) -> NR(time)              (RES inside AEC)
  separated : AEC(linear) -> NR(E(f)) -> RES (freq-domain)  (item-12 topology)

Usage:
  python3 pipelines/rebench_sep_vs_classic.py <classic_dir> <separated_dir> [limit_per_scenario]
Then:
  python3 ../AEC/python/bench_aecmos.py <classic_dir>   <res_classic>
  python3 ../AEC/python/bench_aecmos.py <separated_dir> <res_separated> --baseline <res_classic>/scores.json
"""
import contextlib
import io
import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Audio_ALG/
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'lib', 'aec', 'python'))

from aec import AEC, AecConfig, AecMode                       # noqa: E402
from pipelines.aec_nr_pipeline import (                       # noqa: E402
    run_aec_classic, run_nr, run_aec_linear, run_nr_spectrum, run_res,
)

CORPUS = os.path.join(ROOT, 'lib', 'aec', 'wav', 'aec_challenge_blind')
if not os.path.isdir(CORPUS) or not os.listdir(CORPUS):
    CORPUS = os.path.join(os.path.dirname(ROOT), 'AEC', 'wav', 'aec_challenge_blind')
SCENARIOS = ['doubletalk', 'farend_singletalk', 'nearend_singletalk']
SR = 16000
FL = 832
NR_PRESET = 'balanced'


def _cfg(enable_res):
    return AecConfig.from_preset(
        'balanced', sample_rate=SR, mode=AecMode.PBFDKF, filter_length=FL,
        enable_shadow=True, enable_res=enable_res, enable_cng=True)


def process_case(mic_path, lpb_path, out_classic, out_sep):
    mic, sr = sf.read(mic_path, dtype='float32')
    ref, _ = sf.read(lpb_path, dtype='float32')
    if mic.ndim > 1:
        mic = mic[:, 0]
    if ref.ndim > 1:
        ref = ref[:, 0]
    n = min(len(mic), len(ref))
    mic, ref = mic[:n], ref[:n]
    # No offline pre-align: the AEC's online matched-filter delay estimator
    # (enable_delay_est) handles alignment, matching production.

    # ---- Arm A: classic (AEC internal RES -> NR) ----
    np.random.seed(0)
    aec_out = run_aec_classic(mic, ref, _cfg(enable_res=True))
    nr_out = run_nr(aec_out, sr, nr_preset=NR_PRESET).astype(np.float32)
    m = min(len(nr_out), n)
    sf.write(out_classic, nr_out[:m], sr, subtype='FLOAT')

    # ---- Arm B: separated freq (AEC-linear -> NR(E) -> RES) ----
    np.random.seed(0)
    _, contexts = run_aec_linear(mic, ref, _cfg(enable_res=True))  # run_aec_linear flips res off
    nr_gains = run_nr_spectrum(contexts, sr, nr_preset=NR_PRESET)
    sep_out = run_res(np.zeros(n, dtype=np.float32), nr_gains, contexts,
                      _cfg(enable_res=False)).astype(np.float32)
    m = min(len(sep_out), n)
    sf.write(out_sep, sep_out[:m], sr, subtype='FLOAT')


def _job(args):
    mic_path, lpb_path, out_c, out_s, stem = args
    if os.path.exists(out_c) and os.path.exists(out_s):
        return (stem, 'skip')
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            process_case(mic_path, lpb_path, out_c, out_s)
        return (stem, 'ok')
    except Exception as e:  # noqa: BLE001
        return (stem, f'FAIL {type(e).__name__} {e}')


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    cls_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/cls800'
    sep_dir = sys.argv[2] if len(sys.argv) > 2 else '/tmp/sep800'
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    workers = int(os.environ.get('REBENCH_WORKERS', '6'))
    os.makedirs(cls_dir, exist_ok=True)
    os.makedirs(sep_dir, exist_ok=True)
    jobs = []
    for sc in SCENARIOS:
        sc_dir = os.path.join(CORPUS, sc)
        if not os.path.isdir(sc_dir):
            continue
        mics = sorted(f for f in os.listdir(sc_dir) if f.endswith('_mic.wav'))
        if limit:
            mics = mics[:limit]
        for mf in mics:
            stem = mf[:-len('_mic.wav')]
            lpb = os.path.join(sc_dir, stem + '_lpb.wav')
            if os.path.exists(lpb):
                jobs.append((os.path.join(sc_dir, mf), lpb,
                             os.path.join(cls_dir, stem + '_ours.wav'),
                             os.path.join(sep_dir, stem + '_ours.wav'), stem))
    print(f"rendering {len(jobs)} cases x2 arms with {workers} workers", flush=True)
    ok = skip = fail = done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_job, j): j[4] for j in jobs}
        for fut in as_completed(futs):
            _, status = fut.result()
            done += 1
            if status == 'ok':
                ok += 1
            elif status == 'skip':
                skip += 1
            else:
                fail += 1
                print(f"  {status}", flush=True)
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} (ok={ok} skip={skip} fail={fail})",
                      flush=True)
    print(f"DONE: {ok} ok + {skip} skip + {fail} fail", flush=True)


if __name__ == '__main__':
    main()
