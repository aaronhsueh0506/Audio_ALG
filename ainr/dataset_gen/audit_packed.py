"""Pre-flight audit of a packed dataset: is the corpus itself a suspect?

WHY THIS EXISTS
---------------
``PackedDataset`` validates shape and sample rate and nothing else, and
``pack_dataset.py`` never checked for non-finite samples. So "the corpus is
clean" was an assumption, not a measurement -- and a single non-finite sample, or
a clip whose target is numerically silent, is enough to produce a gradient spike
that looks like a model or optimizer problem.

⚠ This is a READ-ONLY audit and it modifies nothing. Run it before a long
training run; the cost is one pass over the shards.

⚠ It reports the corpus AS PACKED. Editing generator config comments after the
fact changes nothing here -- an already-generated dataset carries whatever
augmentation was in force when it was written, so the numbers below are the only
statement about what the model will actually see.

WHAT IT LOOKS FOR
-----------------
non-finite      any NaN or inf. Any hit at all is disqualifying: it will reach
                the loss and, with gradient clipping active, turn into a NaN
                that poisons the optimizer moments irrecoverably.
silent target   clean RMS below --silent-rms. These are the full-suppression
                examples; their fraction should match the generator's
                ``noise_only_p``, and a large mismatch means the corpus is not
                what the config claims.
near-silent     clean RMS in a band just above silence. These are the ones that
                drive a prediction magnitude down through the region where
                DeepFilterNet's complex-angle gradient is most amplified, so
                they are more interesting than the exactly-silent ones.
dc offset       mean/RMS ratio above --max-dc. Upstream applies a random DC
                remover during training; this port does not, so a corpus with a
                real DC component is a genuine distribution difference.
clipped         fraction of samples at |x| >= --clip-thresh, i.e. already at
                full scale before the model sees them.

USAGE
-----
    python3 audit_packed.py --packed-dir /path/to/data_48k/packed.pt
    python3 audit_packed.py --packed-dir /path/to/shards/ --expected-sr 16000

Exit status is 1 if any disqualifying condition is found, so it can gate a
training job.
"""

import argparse
import sys

import torch

from .loader import load_packed_dataset


def audit_tensor(data, *, silent_rms, near_silent_rms, max_dc, clip_thresh,
                 batch=256):
    """Scan a packed (N, 2, T) tensor. Returns a dict of findings."""
    n_pairs = data.shape[0]
    stats = {
        'n_pairs': n_pairs,
        'non_finite_clips': [],
        'silent_target': [],
        'near_silent_target': [],
        'dc_offset': [],
        'clipped': [],
        'noisy_rms': [],
        'clean_rms': [],
        'peak': 0.0,
    }
    for start in range(0, n_pairs, batch):
        chunk = data[start:start + batch]
        chunk = torch.as_tensor(chunk).float()
        noisy, clean = chunk[:, 0], chunk[:, 1]

        finite = torch.isfinite(chunk).all(dim=-1).all(dim=-1)
        for offset in (~finite).nonzero(as_tuple=True)[0].tolist():
            idx = start + offset
            bad = chunk[offset]
            stats['non_finite_clips'].append({
                'index': idx,
                'nan': int(bad.isnan().sum()),
                'inf': int(bad.isinf().sum()),
            })

        # Everything below is meaningless on a non-finite clip, so mask it out
        # rather than letting one bad clip poison every aggregate.
        keep = finite
        if not bool(keep.any()):
            continue
        noisy, clean = noisy[keep], clean[keep]
        kept_idx = (
            keep.nonzero(as_tuple=True)[0] + start
        ).tolist()

        n_rms = noisy.pow(2).mean(dim=-1).sqrt()
        c_rms = clean.pow(2).mean(dim=-1).sqrt()
        stats['noisy_rms'].append(n_rms)
        stats['clean_rms'].append(c_rms)
        stats['peak'] = max(
            stats['peak'], float(torch.stack([noisy, clean]).abs().max())
        )

        for local, idx in enumerate(kept_idx):
            if float(c_rms[local]) < silent_rms:
                stats['silent_target'].append(idx)
            elif float(c_rms[local]) < near_silent_rms:
                stats['near_silent_target'].append(idx)

        for tag, wave, rms in (('noisy', noisy, n_rms), ('clean', clean, c_rms)):
            dc = wave.mean(dim=-1).abs()
            ratio = dc / rms.clamp_min(1e-12)
            for local in (ratio > max_dc).nonzero(as_tuple=True)[0].tolist():
                stats['dc_offset'].append(
                    (kept_idx[local], tag, float(ratio[local]))
                )
            at_full = (wave.abs() >= clip_thresh).float().mean(dim=-1)
            for local in (at_full > 0.0).nonzero(as_tuple=True)[0].tolist():
                stats['clipped'].append(
                    (kept_idx[local], tag, float(at_full[local]))
                )

    for key in ('noisy_rms', 'clean_rms'):
        stats[key] = (
            torch.cat(stats[key]) if stats[key] else torch.zeros(0)
        )
    return stats


def _describe(name, values):
    if values.numel() == 0:
        return f'  {name:<12} (no finite clips)'
    quantiles = torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0])
    q = torch.quantile(values.double(), quantiles.double()).tolist()
    return (f'  {name:<12} min={q[0]:.3e}  p1={q[1]:.3e}  '
            f'med={q[2]:.3e}  p99={q[3]:.3e}  max={q[4]:.3e}')


def report(stats, *, silent_rms, near_silent_rms, max_dc, clip_thresh):
    """Print the audit and return True if the corpus is fit to train on."""
    n = stats['n_pairs']
    print(f'\npacked pairs: {n}')
    print(_describe('noisy RMS', stats['noisy_rms']))
    print(_describe('clean RMS', stats['clean_rms']))
    print(f'  {"peak":<12} {stats["peak"]:.6f}')

    print('\nfindings')
    fatal = False

    bad = stats['non_finite_clips']
    if bad:
        fatal = True
        print(f'  ⚠ NON-FINITE SAMPLES in {len(bad)} clip(s) '
              f'-- disqualifying, these WILL produce a NaN loss')
        for entry in bad[:20]:
            print(f'      clip {entry["index"]}: '
                  f'{entry["nan"]} NaN, {entry["inf"]} inf')
        if len(bad) > 20:
            print(f'      ... and {len(bad) - 20} more')
    else:
        print('  ✓ no NaN or inf anywhere')

    silent = stats['silent_target']
    print(f'  silent targets      : {len(silent)} '
          f'({100.0 * len(silent) / max(n, 1):.2f}%, '
          f'clean RMS < {silent_rms:.0e})')
    print('      compare against the generator\'s noise_only_p; a large '
          'mismatch means')
    print('      the corpus is not what the config claims')

    near = stats['near_silent_target']
    print(f'  near-silent targets : {len(near)} '
          f'({100.0 * len(near) / max(n, 1):.2f}%, '
          f'{silent_rms:.0e} <= RMS < {near_silent_rms:.0e})')

    dc = stats['dc_offset']
    print(f'  DC offset > {max_dc:.2f}    : {len(dc)} clip/channel pairs')
    for idx, tag, ratio in dc[:5]:
        print(f'      clip {idx} {tag}: |mean|/rms = {ratio:.3f}')

    clipped = stats['clipped']
    print(f'  at full scale       : {len(clipped)} clip/channel pairs '
          f'(|x| >= {clip_thresh})')
    for idx, tag, frac in clipped[:5]:
        print(f'      clip {idx} {tag}: {100.0 * frac:.3f}% of samples')

    print()
    if fatal:
        print('VERDICT: DO NOT TRAIN on this corpus -- fix the non-finite '
              'clips first.')
    else:
        print('VERDICT: no disqualifying condition found.  The corpus is not '
              'the NaN source.')
    return not fatal


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Read-only audit of a packed dataset for non-finite '
                    'samples and distribution surprises')
    parser.add_argument('--packed-dir', required=True,
                        help='packed.pt file or directory of *.pt shards')
    parser.add_argument('--expected-sr', type=int, default=None,
                        help='fail if the shard sample rate differs')
    parser.add_argument('--mmap', action='store_true',
                        help='memory-map instead of loading into RAM')
    parser.add_argument('--silent-rms', type=float, default=1e-8,
                        help='clean RMS below this counts as a silent target')
    parser.add_argument('--near-silent-rms', type=float, default=1e-4,
                        help='upper edge of the near-silent band')
    parser.add_argument('--max-dc', type=float, default=0.1,
                        help='report |mean|/rms above this')
    parser.add_argument('--clip-thresh', type=float, default=0.999,
                        help='|x| at or above this counts as full scale')
    args = parser.parse_args(argv)

    dataset = load_packed_dataset(
        args.packed_dir, expected_sr=args.expected_sr, mmap=args.mmap,
    )
    shards = getattr(dataset, 'datasets', [dataset])

    ok = True
    for shard in shards:
        stats = audit_tensor(
            shard.data,
            silent_rms=args.silent_rms,
            near_silent_rms=args.near_silent_rms,
            max_dc=args.max_dc,
            clip_thresh=args.clip_thresh,
        )
        ok = report(
            stats,
            silent_rms=args.silent_rms,
            near_silent_rms=args.near_silent_rms,
            max_dc=args.max_dc,
            clip_thresh=args.clip_thresh,
        ) and ok
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
