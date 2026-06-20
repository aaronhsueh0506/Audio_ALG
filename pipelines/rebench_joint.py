#!/usr/bin/env python3
"""Render the A_min_pl freq pipeline over the 800-case corpus (production default).

  A_min_pl : AEC(linear) -> noise-only NR(E) -> g_total = min(G_nr, G_res)
             + per-bin echo-gated NE floor (ne_floor=0.4, gate='both')

Usage: python3 pipelines/rebench_joint.py <out_dir> [ne_floor] [ne_gate] [limit]
Then:  python3 ../AEC/python/bench_aecmos.py <out_dir> <res_dir> --baseline <classic>/scores.json
"""
import contextlib
import io
import os
import sys

import numpy as np
import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'lib', 'aec', 'python'))

from aec import AecConfig, AecMode                              # noqa: E402
from pipelines.aec_nr_pipeline import (                         # noqa: E402
    run_aec_linear, run_nr_spectrum, run_res,
)
from pipelines.rebench_sep_vs_classic import (                  # noqa: E402
    estimate_delay, CORPUS, SCENARIOS, SR, FL, NR_PRESET,
)

NE_FLOOR = float(os.environ.get('NE_FLOOR', '0.4'))
NE_GATE = os.environ.get('NE_GATE', 'both')
# Env override lets one renderer cover any NR preset (mild/balanced/aggressive);
# defaults to the shipped 'balanced' so the production render is unchanged.
NR_PRESET = os.environ.get('NR_PRESET', NR_PRESET)


def _cfg(enable_res):
    return AecConfig.from_preset('balanced', sample_rate=SR, mode=AecMode.PBFDKF,
                                 filter_length=FL, enable_shadow=True,
                                 enable_res=enable_res, enable_cng=True)


def process(mic_path, lpb_path, out_path):
    mic, sr = sf.read(mic_path, dtype='float32')
    ref, _ = sf.read(lpb_path, dtype='float32')
    if mic.ndim > 1:
        mic = mic[:, 0]
    if ref.ndim > 1:
        ref = ref[:, 0]
    n = min(len(mic), len(ref))
    mic, ref = mic[:n], ref[:n]
    if os.environ.get('PREALIGN'):                 # opt-in: offline pre-align before AEC
        d = estimate_delay(mic, ref, sr)
        if d > 0:
            ref = np.concatenate([np.zeros(d, dtype=np.float32), ref[:n - d]])
    with contextlib.redirect_stdout(io.StringIO()):
        np.random.seed(0)
        _, ctx = run_aec_linear(mic, ref, _cfg(True))
        g = run_nr_spectrum(ctx, sr, nr_preset=NR_PRESET)
        out = run_res(np.zeros(n, dtype=np.float32), g, ctx, _cfg(False),
                      use_res=True, combine='min', ne_floor=NE_FLOOR, ne_gate=NE_GATE)
    sf.write(out_path, out[:n], sr, subtype='FLOAT')


def _job(a):
    mic, lpb, out, stem = a
    if os.path.exists(out):
        return (stem, 'skip')  # WARNING: stale render may be from an older code version
    try:
        process(mic, lpb, out)
        return (stem, 'ok')
    except Exception as e:  # noqa: BLE001
        return (stem, f'FAIL {type(e).__name__} {e}')


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out_dir = sys.argv[1] if len(sys.argv) > 1 else '/tmp/joint800'
    if len(sys.argv) > 2:
        os.environ['NE_FLOOR'] = sys.argv[2]      # must set env before spawning workers
        globals()['NE_FLOOR'] = float(sys.argv[2])
    if len(sys.argv) > 3:
        os.environ['NE_GATE'] = sys.argv[3]
        globals()['NE_GATE'] = sys.argv[3]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
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
    print(f"joint render: {len(jobs)} cases, ne_floor={NE_FLOOR}, gate={NE_GATE}, "
          f"{workers} workers", flush=True)
    ok = fail = skip = done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_job, j): j[3] for j in jobs}
        for fut in as_completed(futs):
            _, st = fut.result()
            done += 1
            if st == 'ok':
                ok += 1
            elif st == 'skip':
                skip += 1
            elif st.startswith('FAIL'):
                fail += 1
                print(' ', st, flush=True)
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    if skip:
        print(f"WARNING: {skip} cases skipped (output already exists) — "
              f"stale renders may be from an older code version; delete {out_dir} to re-render",
              flush=True)
    print(f"DONE ok={ok} skip={skip} fail={fail}", flush=True)


if __name__ == '__main__':
    main()
