#!/usr/bin/env python3
"""Render the AEC-alone arm over the 800-case corpus, in the SAME integrated
harness (same cfg/seed/FL, no offline pre-align) as rebench_joint.py, so the
two are directly comparable for an "AEC vs AEC+NR" table.

  aec_only : AEC(enable_res=True) full output  — its own AEC3 post-filter RES,
             NO NR stage. The baseline for "what NR adds".

Usage: python3 pipelines/tools/rebench_aec_only.py <out_dir> [limit]
Then:  python3 ../AEC/python/bench_aecmos.py <out_dir> <res_dir>
"""
import contextlib
import io
import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'lib', 'aec', 'python'))

from aec import AecConfig, AecMode                              # noqa: E402
from pipelines.aec_nr_pipeline import run_aec_classic           # noqa: E402
from pipelines.tools.rebench_sep_vs_classic import (            # noqa: E402
    CORPUS, SCENARIOS, SR, FL,
)


def _cfg():
    return AecConfig.from_preset('balanced', sample_rate=SR, mode=AecMode.PBFDKF,
                                 filter_length=FL, enable_shadow=True,
                                 enable_res=True, enable_cng=True)


def process(mic_path, lpb_path, out_path):
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
    with contextlib.redirect_stdout(io.StringIO()):
        np.random.seed(0)
        out = run_aec_classic(mic, ref, _cfg())            # AEC full output, no NR
    sf.write(out_path, out[:n].astype(np.float32), sr, subtype='FLOAT')


def _job(a):
    mic, lpb, out, stem = a
    if os.path.exists(out):
        return (stem, 'skip')
    try:
        process(mic, lpb, out)
        return (stem, 'ok')
    except Exception as e:  # noqa: BLE001
        return (stem, f'FAIL {type(e).__name__} {e}')


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/aec_only800'
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    workers = int(os.environ.get('REBENCH_WORKERS', '8'))
    os.makedirs(out_dir, exist_ok=True)
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
                             os.path.join(out_dir, stem + '_ours.wav'), stem))
    print(f"aec-only render: {len(jobs)} cases, {workers} workers", flush=True)
    ok = fail = done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_job, j): j[3] for j in jobs}
        for fut in as_completed(futs):
            _, st = fut.result()
            done += 1
            ok += st in ('ok', 'skip')
            if st.startswith('FAIL'):
                fail += 1
                print(' ', st, flush=True)
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    print(f"DONE ok={ok} fail={fail}", flush=True)


if __name__ == '__main__':
    main()
