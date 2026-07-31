# -*- coding: utf-8 -*-
"""Offline rendering of AEC scenario sequences into 7-channel WAV chunks.

Mirrors ``AINR/dataset_gen/gen_dataset.py``'s flags so the two generators are driven
the same way, with two additions that the NR generator has no need for:

    --split       which side of the SOURCE-DISJOINT split to draw from
    --manifest    where that split decision lives

⚠ The split is decided BEFORE generation, over the source lists, and both runs
must use the SAME manifest file.  See manifest.py for why splitting after
generation (as the NR loader does) is wrong here.

Usage:
    python3 gen_aec_dataset.py --config config.ini --output data_aec \\
        --hours 40 --split train --workers 4 --seed 42
    python3 gen_aec_dataset.py --config config.ini --output data_aec \\
        --hours 4  --split val   --workers 4 --seed 42

Output layout:
    data_aec/
      manifest.json                 the split decision, shared by both runs
      train/meta.json               run summary
      train/seqs/000000.json        chunk metadata for one parent sequence
      train/seqs/000000_000.wav     7-channel chunk, channels = STEM_ORDER
      val/...

Then pack with pack_aec_dataset.py.
"""

import argparse
import configparser
import json
import os
import subprocess
import sys
import time
from typing import List, Optional

import torch
import torch.utils.data as data
import torchaudio
import tqdm

if __package__ in (None, ''):
    # Direct-file compatibility without creating a second top-level
    # ``dataset_gen`` package that could be confused with ``AINR.dataset_gen``.
    _AUDIO_ALG = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    sys.path.insert(0, _AUDIO_ALG)
    __package__ = 'AIAEC.dataset_gen'

from .aec_dataset import (  # noqa: E402
    AecSequenceRenderer,
    SequencePlan,
    plan_sequences,
)
from .aec_features import STEM_ORDER  # noqa: E402
from .manifest import (  # noqa: E402
    build_manifest,
    config_hash,
    load_manifest,
    pools_for_split,
    save_manifest,
    summarise,
)


WAV_ENCODINGS = {
    # ⚠ float32 is the default because mic_preclip == near_speech +
    # local_noise + echo is the corpus's central invariant, checked at
    # generation time against the renderer's un-quantised audit tensors (see
    # aec_dataset.py's RenderedSequence.audit). Quantising the PERSISTED
    # stems to int16 would still degrade any downstream arithmetic that
    # combines them (e.g. an SER/SNR recomputed from the stored near_speech/
    # local_noise) by ~1e-4. int16 halves the disk cost and is fine for
    # listening, not for arithmetic.
    'float32': dict(encoding='PCM_F', bits_per_sample=32),
    'int16': dict(encoding='PCM_S', bits_per_sample=16),
}


def git_commit() -> str:
    """Best-effort commit id for provenance; 'unknown' outside a checkout."""
    try:
        out = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.decode().strip()
    except Exception:
        pass
    return 'unknown'


def verify_wav_io(tmp_dir: str, encoding: str) -> None:
    """Prove the requested WAV precision actually survives a round trip.

    ⚠ torchaudio warns that a future release will ignore ``encoding`` and
    ``bits_per_sample``.  If that lands, a float32 request would silently write
    int16 and the stem-sum identity would quietly degrade by four orders of
    magnitude with no error anywhere.  Failing loudly here is cheap; discovering
    it after a 40-hour render is not.
    """
    if encoding != 'float32':
        return
    os.makedirs(tmp_dir, exist_ok=True)
    path = os.path.join(tmp_dir, '.wav_io_check.wav')
    probe = torch.randn(len(STEM_ORDER), 512) * 0.1
    torchaudio.save(path, probe, 16000, **WAV_ENCODINGS['float32'])
    back, _sr = torchaudio.load(path)
    error = float((back - probe).abs().max())
    os.remove(path)
    if error > 1e-6:
        raise RuntimeError(
            f"float32 WAV round trip lost {error:.2e} -- this torchaudio build "
            f"is ignoring the requested encoding and writing a quantised file. "
            f"Use --wav-encoding int16 knowingly, or pin an older torchaudio."
        )


class _RenderJobs(data.Dataset):
    """One item per parent sequence.  Workers write the audio themselves.

    Returning the rendered tensors through the DataLoader queue instead would
    push ~7 MB per 60 s sequence through IPC for no reason; the chunks have to
    land on disk either way.
    """

    def __init__(self, cfg, manifest, split, plans, seqs_dir, encoding):
        self.cfg = cfg
        self.manifest = manifest
        self.split = split
        self.plans = plans
        self.seqs_dir = seqs_dir
        self.encoding = encoding
        self._renderer: Optional[AecSequenceRenderer] = None

    def _get_renderer(self) -> AecSequenceRenderer:
        # Built lazily, so each worker process constructs its own rather than
        # unpickling cached filter tensors from the parent.
        if self._renderer is None:
            self._renderer = AecSequenceRenderer(
                self.cfg, pools_for_split(self.manifest, self.split),
                corpus_seed=self.manifest['seed'])
        return self._renderer

    def __len__(self):
        return len(self.plans)

    def __getitem__(self, index):
        plan = self.plans[index]
        meta_path = os.path.join(self.seqs_dir, f"{plan.sequence_id:06d}.json")
        rendered = self._get_renderer().render(plan)

        write = WAV_ENCODINGS[self.encoding]
        for chunk_index in range(len(rendered.chunk_meta)):
            at = chunk_index * rendered.chunk_samples
            chunk = rendered.stems[:, at:at + rendered.chunk_samples].contiguous()
            torchaudio.save(
                os.path.join(self.seqs_dir,
                             f"{plan.sequence_id:06d}_{chunk_index:03d}.wav"),
                chunk, self.cfg.getint('signal', 'sr'), **write)

        # The sidecar is written LAST and is what --resume looks for, so a run
        # killed mid-sequence leaves an incomplete sequence that is simply
        # re-rendered rather than silently kept with missing chunks.
        with open(meta_path, 'w') as handle:
            json.dump(rendered.chunk_meta, handle)
        return {
            'sequence_id': plan.sequence_id,
            'n_chunks': plan.n_chunks,
            'scenario': plan.scenario,
        }


def _identity_collate(batch):
    """Module-level (not a lambda) so DataLoader workers can pickle it."""
    return batch[0]


def _pending(plans: List[SequencePlan], seqs_dir: str,
             resume: bool) -> List[SequencePlan]:
    if not resume:
        return list(plans)
    pending = []
    for plan in plans:
        if not os.path.isfile(os.path.join(seqs_dir, f"{plan.sequence_id:06d}.json")):
            pending.append(plan)
    return pending


def gen_aec_dataset(args):
    cfg = configparser.ConfigParser()
    if not cfg.read(args.config):
        raise FileNotFoundError(f"config not found: {args.config}")

    # One generation rate per run; CLI overrides config, exactly as the NR
    # generator does.
    cfg_sr = cfg.getint('signal', 'sr')
    generation_sr = args.sample_rate if args.sample_rate is not None else cfg_sr
    if generation_sr <= 0:
        raise ValueError(f"Sample rate must be positive, got {generation_sr}")
    cfg.set('signal', 'sr', str(generation_sr))
    print(f"Sample rate: {generation_sr} Hz "
          f"({'CLI' if args.sample_rate is not None else 'config.ini [signal] sr'})")

    split_dir = os.path.join(args.output, args.split)
    seqs_dir = os.path.join(split_dir, 'seqs')
    os.makedirs(seqs_dir, exist_ok=True)

    verify_wav_io(split_dir, args.wav_encoding)

    # --- the split decision -------------------------------------------------
    manifest_path = args.manifest or os.path.join(args.output, 'manifest.json')
    if os.path.isfile(manifest_path) and not args.rebuild_manifest:
        manifest = load_manifest(manifest_path)
        print(f"Manifest: reusing {manifest_path}")
        if manifest['config_hash'] != config_hash(cfg):
            # ⚠ Not fatal: --hours and output paths legitimately change between
            # the train run and the val run.  But a changed corpus definition
            # with an unchanged split is worth seeing.
            print("  ⚠ config_hash differs from the one the manifest was built "
                  "with; the SPLIT is still the manifest's, not this config's")
    else:
        print(f"Manifest: building from source lists (seed {args.seed}) ...")
        manifest = build_manifest(cfg, args.seed)
        save_manifest(manifest, manifest_path)
        print(f"  written to {manifest_path}")
    print(summarise(manifest))

    if manifest['sr'] != generation_sr:
        # Only RT60 filtering depends on the rate, but a corpus whose RIR set
        # was selected at a different rate is not the corpus the manifest
        # describes.
        raise ValueError(
            f"manifest was built at sr={manifest['sr']} but this run generates "
            f"at {generation_sr}; rebuild it with --rebuild-manifest")

    # --- the plan -----------------------------------------------------------
    plans = plan_sequences(cfg, args.hours, args.seed, args.split)
    planned_sec = sum(p.n_chunks for p in plans) * cfg.getfloat('sequence', 'chunk_sec')
    pending = _pending(plans, seqs_dir, args.resume)

    chunk_sec = cfg.getfloat('sequence', 'chunk_sec')
    chunk_samples = int(round(chunk_sec * generation_sr))
    bytes_per = 4 if args.wav_encoding == 'float32' else 2
    disk = sum(p.n_chunks for p in plans) * len(STEM_ORDER) * chunk_samples * bytes_per

    print(f"\nRequested {args.hours:g} h of '{args.split}' "
          f"-> {len(plans)} sequences / {planned_sec / 3600:.3f} h")
    print(f"  Chunk            : {chunk_sec:g} s ({chunk_samples} samples)")
    print(f"  Stems            : {len(STEM_ORDER)} x {args.wav_encoding}")
    print(f"  Estimated disk   : {disk / 1024 ** 3:.1f} GB")
    print(f"  Workers          : {args.workers}")
    if args.resume:
        print(f"  Resume           : {len(plans) - len(pending)} already rendered, "
              f"{len(pending)} to go")
    print(f"  Output           : {seqs_dir}/\n")

    jobs = _RenderJobs(cfg, manifest, args.split, pending, seqs_dir,
                       args.wav_encoding)
    started = time.time()
    scenario_counts = {}

    if pending:
        if args.workers > 0:
            loader = data.DataLoader(jobs, batch_size=1, shuffle=False,
                                     num_workers=args.workers, prefetch_factor=2,
                                     collate_fn=_identity_collate,
                                     persistent_workers=False)
            iterator = tqdm.tqdm(loader, total=len(pending), desc=f"render/{args.split}")
        else:
            iterator = tqdm.tqdm((jobs[i] for i in range(len(pending))),
                                 total=len(pending), desc=f"render/{args.split}")
        for done in iterator:
            scenario_counts[done['scenario']] = scenario_counts.get(done['scenario'], 0) + 1

    # Sequence-level scenario counts across the WHOLE plan, not just this run's
    # pending slice, so a resumed run reports the corpus and not the remainder.
    plan_counts = {}
    for plan in plans:
        plan_counts[plan.scenario] = plan_counts.get(plan.scenario, 0) + 1

    meta = {
        'split': args.split,
        'sr': generation_sr,
        'stems': list(STEM_ORDER),
        'chunk_sec': chunk_sec,
        'chunk_samples': chunk_samples,
        'n_sequences': len(plans),
        'n_chunks': sum(p.n_chunks for p in plans),
        'hours': planned_sec / 3600,
        'requested_hours': args.hours,
        'seed': args.seed,
        'wav_encoding': args.wav_encoding,
        'config': os.path.abspath(args.config),
        'config_hash': config_hash(cfg),
        'generator_commit': git_commit(),
        'manifest': os.path.abspath(manifest_path),
        'manifest_config_hash': manifest['config_hash'],
        'sequence_scenarios': plan_counts,
        'rendered_this_run': len(pending),
    }
    with open(os.path.join(split_dir, 'meta.json'), 'w') as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

    print(f"\nDone. {len(plans)} sequences ({planned_sec / 3600:.3f} h) in "
          f"{seqs_dir}/, {(time.time() - started) / 60:.1f} min this run.")
    print(f"  Sequence scenarios: {plan_counts}")
    print(f"  Next: python3 pack_aec_dataset.py --input {split_dir} "
          f"--output {os.path.join(args.output, 'packed', args.split)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Render AEC scenario sequences as 7-channel stem WAVs')
    parser.add_argument('--config', default='config.ini', help='Config file path')
    parser.add_argument('--output', default='data_aec', help='Output directory')
    parser.add_argument('--hours', type=float, default=8.0,
                        help='Target audio hours for the selected split')
    parser.add_argument('--workers', type=int, default=4,
                        help='Render workers (default: 4, 0 = single process)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip sequences whose metadata sidecar already exists')
    parser.add_argument('--seed', type=int, default=42,
                        help='Corpus seed. Fixes BOTH the source split and every '
                             'sequence; the train and val runs must share it.')
    parser.add_argument('--sample-rate', type=int, default=None,
                        help='Generation sample rate in Hz, overriding [signal] sr')
    parser.add_argument('--split', default='train', choices=('train', 'val'),
                        help='Which source-disjoint split to draw from')
    parser.add_argument('--manifest', default=None,
                        help='Manifest path (default: <output>/manifest.json). '
                             'The train and val runs MUST use the same file.')
    parser.add_argument('--rebuild-manifest', action='store_true',
                        help='⚠ Redraw the source split. Invalidates every corpus '
                             'already generated from the old manifest.')
    parser.add_argument('--wav-encoding', default='float32',
                        choices=sorted(WAV_ENCODINGS),
                        help='float32 keeps the stem-sum identity exact; int16 '
                             'halves the disk cost and degrades it to ~1e-4')
    return parser


if __name__ == '__main__':
    gen_aec_dataset(build_parser().parse_args())
