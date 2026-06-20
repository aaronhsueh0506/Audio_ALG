#!/usr/bin/env python3
"""DNSMOS bench analyzer — score enhanced outputs (P.835 SIG/BAK/OVRL + P.808).

Usage:
    python3 pipelines/bench_dnsmos.py <output_dir> <result_dir>

Walks <output_dir> for `<stem>_<scenario>[_with_movement]_ours.wav` files,
buckets them exactly like bench_aecmos.py (FS_static / FS_movement / DT_static /
DT_movement / NE), runs the local speechmos DNSMOS ONNX on each enhanced file
(no reference needed — DNSMOS is non-intrusive), and aggregates per-bucket means.

Outputs (in <result_dir>):
  dnsmos.json — {label, summary{bucket:{n,sig,bak,ovrl,p808}}, scores{stem:{...}}}
"""
import os
import sys
import json
import re
import argparse
import warnings
from collections import defaultdict

warnings.filterwarnings('ignore')
import numpy as np
import speechmos.dnsmos as dnsmos

# Same bucket grammar as bench_aecmos.py.
_BUCKET_RE = re.compile(
    r'^(?P<stem>.+?)_(?P<scenario>farend_singletalk|nearend_singletalk|doubletalk)'
    r'(?P<mv>_with_movement)?_ours\.wav$'
)


def classify(filename):
    m = _BUCKET_RE.match(filename)
    if not m:
        return None
    scenario = m.group('scenario')
    is_mv = bool(m.group('mv'))
    stem = m.group('stem') + '_' + scenario + (m.group('mv') or '')
    if scenario == 'farend_singletalk':
        bucket = 'FS_movement' if is_mv else 'FS_static'
    elif scenario == 'nearend_singletalk':
        bucket = 'NE'
    else:
        bucket = 'DT_movement' if is_mv else 'DT_static'
    return bucket, stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('output_dir', help='Directory containing <stem>_ours.wav files')
    ap.add_argument('result_dir', help='Where to write dnsmos.json')
    ap.add_argument('--label', default='current')
    args = ap.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    files = []
    for f in sorted(os.listdir(args.output_dir)):
        if not f.endswith('_ours.wav'):
            continue
        c = classify(f)
        if c:
            files.append((os.path.join(args.output_dir, f), c[0], c[1]))

    print(f"Scoring {len(files)} cases (DNSMOS)...", flush=True)
    scores = {}
    acc = defaultdict(lambda: {'sig': [], 'bak': [], 'ovrl': [], 'p808': []})
    for i, (path, bucket, stem) in enumerate(files):
        # Pass the file PATH (not an ndarray): speechmos enforces the [-1,1]
        # range only on ndarray input; the path branch loads via librosa and
        # feeds raw (possibly over-unity) audio — the standard DNSMOS behavior.
        r = dnsmos.run(path, 16000, return_df=False)
        rec = {'bucket': bucket, 'sig': float(r['sig_mos']), 'bak': float(r['bak_mos']),
               'ovrl': float(r['ovrl_mos']), 'p808': float(r['p808_mos'])}
        scores[stem] = rec
        for k in ('sig', 'bak', 'ovrl', 'p808'):
            acc[bucket][k].append(rec[k])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    summary = {}
    for bucket, d in acc.items():
        summary[bucket] = {'n': len(d['sig'])}
        for k in ('sig', 'bak', 'ovrl', 'p808'):
            summary[bucket][k] = float(np.mean(d[k]))

    out = {'label': args.label, 'summary': summary, 'scores': scores}
    json.dump(out, open(os.path.join(args.result_dir, 'dnsmos.json'), 'w'), indent=2)

    order = ['FS_static', 'FS_movement', 'DT_static', 'DT_movement', 'NE']
    print(f"\n=== DNSMOS {args.label} ===")
    print(f"{'bucket':<13} {'n':>4} {'SIG':>6} {'BAK':>6} {'OVRL':>6} {'P808':>6}")
    for b in order:
        if b in summary:
            s = summary[b]
            print(f"{b:<13} {s['n']:>4} {s['sig']:>6.3f} {s['bak']:>6.3f} "
                  f"{s['ovrl']:>6.3f} {s['p808']:>6.3f}")
    print(f"\nWrote {args.result_dir}/dnsmos.json")


if __name__ == '__main__':
    main()
