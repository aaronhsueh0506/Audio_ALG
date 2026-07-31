# -*- coding: utf-8 -*-
"""Offline rendering of AEC scenario sequences into 6-channel WAV chunks.

Mirrors ``AINR/dataset_gen/gen_dataset.py``'s flags so the two generators are driven
the same way, with two additions that the NR generator has no need for:

    --split       'all' (the selected protocol) or a SOURCE-DISJOINT side
                  ('train'/'val'), for a separate held-out generalisation corpus
    --manifest    where that split decision lives

⚠ The split is decided BEFORE generation, over the source lists, and both runs
must use the SAME manifest file.  See manifest.py for why splitting after
generation (as the NR loader does) is wrong here.

Usage (selected training protocol -- one unified pool, then a deterministic
random chunk split with AIAEC.training_common.split_dataset_by_sample at load
time; this is also --split's default):
    python3 gen_aec_dataset.py --config config.ini --output data_aec \\
        --hours 100 --split all --workers 4 --seed 42

Usage (source-disjoint -- two separate runs, same --seed/--manifest; use this
for a held-out generalisation corpus, not the main training/validation pool):
    python3 gen_aec_dataset.py --config config.ini --output data_aec \\
        --hours 40 --split train --workers 4 --seed 42
    python3 gen_aec_dataset.py --config config.ini --output data_aec \\
        --hours 4  --split val   --workers 4 --seed 42

Output layout:
    data_aec/
      manifest.json                 the split decision ('all' mode: just
                                     source-list provenance)
      all/meta.json                 run summary ('all' mode)
      all/seqs/000000.json          chunk metadata for one parent sequence
      all/seqs/000000_000.wav       6-channel chunk, channels = STEM_ORDER
      train/..., val/...            source-disjoint mode instead of all/,
                                     shared by both runs

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
    rooms_eligible_for_path_change,
)
from .aec_features import STEM_ORDER  # noqa: E402
from .linear_aec import linear_aec_contract_from_config  # noqa: E402
from .manifest import (  # noqa: E402
    ALL_SPLIT_NAMES,
    UNIFIED_SPLIT,
    build_manifest,
    build_unified_manifest,
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


def _sequence_is_complete(plan: SequencePlan, seqs_dir: str, *,
                          sample_rate: int, chunk_samples: int,
                          contract_hash: str, config_hash: str,
                          wav_encoding: str = 'float32') -> bool:
    """A sidecar alone is not enough to prove a six-channel sequence exists.

    ``config_hash`` closes the gap ``linear_aec_contract_hash`` alone leaves
    open: two configs can render the SAME chunk count/geometry for a given
    sequence_id (scenario weights, levels, RIR/device settings...) while
    producing entirely different audio. Without this check, --resume after
    such an edit would keep the stale sequence and meta.json would still
    claim the whole run reflects the new config.

    ``plan.scenario``/``plan.seed`` close a THIRD gap that config_hash cannot:
    ``--seed`` lives outside config.ini, so a --seed change reshuffles which
    scenario/seed plan_sequences() assigns each sequence_id without touching
    config_hash at all. Whenever the reshuffled plan happens to want the same
    chunk count for a sequence_id (routine at this corpus's small
    chunks_min/chunks_max spread), the old checks alone would resume it as
    complete under the NEW plan's scenario label while the audio on disk is
    still the OLD plan's.
    """
    meta_path = os.path.join(seqs_dir, f"{plan.sequence_id:06d}.json")
    if not os.path.isfile(meta_path):
        return False
    try:
        with open(meta_path, "r") as handle:
            chunk_meta = json.load(handle)
        if len(chunk_meta) != plan.n_chunks:
            return False
        for chunk_index, meta in enumerate(chunk_meta):
            if (
                meta.get("chunk_index") != chunk_index
                or meta.get("linear_aec_contract_hash") != contract_hash
                or meta.get("config_hash") != config_hash
                or meta.get("sequence_scenario") != plan.scenario
                or meta.get("sequence_seed") != plan.seed
            ):
                return False
            wav_path = os.path.join(
                seqs_dir, f"{plan.sequence_id:06d}_{chunk_index:03d}.wav"
            )
            if not os.path.isfile(wav_path):
                return False
            info = torchaudio.info(wav_path)
            expected_encoding = WAV_ENCODINGS[wav_encoding]
            if (
                info.sample_rate != sample_rate
                or info.num_frames != chunk_samples
                or info.num_channels != len(STEM_ORDER)
                or info.encoding != expected_encoding['encoding']
                or info.bits_per_sample != expected_encoding['bits_per_sample']
            ):
                return False
    except (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _pending(plans: List[SequencePlan], seqs_dir: str, resume: bool, *,
             sample_rate: int, chunk_samples: int,
             contract_hash: str, config_hash: str,
             wav_encoding: str = 'float32') -> List[SequencePlan]:
    if not resume:
        return list(plans)
    return [
        plan for plan in plans
        if not _sequence_is_complete(
            plan, seqs_dir, sample_rate=sample_rate,
            chunk_samples=chunk_samples, contract_hash=contract_hash,
            config_hash=config_hash, wav_encoding=wav_encoding,
        )
    ]


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
    # Computed once and reused below (the manifest-reuse comparison and the
    # --resume check both need it, and cfg does not change in between).
    cfg_hash = config_hash(cfg)

    split_dir = os.path.join(args.output, args.split)
    seqs_dir = os.path.join(split_dir, 'seqs')
    os.makedirs(seqs_dir, exist_ok=True)

    verify_wav_io(split_dir, args.wav_encoding)

    # --- the split decision -------------------------------------------------
    # One builder per manifest shape -- the caller (here) only decides WHICH
    # shape this --split wants, not how to build/save/announce it, so the
    # "reuse vs build" branch below stays two-way instead of three-way.
    expected_mode = 'unified' if args.split == UNIFIED_SPLIT else 'disjoint'
    manifest_builders = {'disjoint': build_manifest, 'unified': build_unified_manifest}

    manifest_path = args.manifest or os.path.join(args.output, 'manifest.json')
    if os.path.isfile(manifest_path) and not args.rebuild_manifest:
        manifest = load_manifest(manifest_path)
        print(f"Manifest: reusing {manifest_path}")
        if int(manifest['seed']) != int(args.seed):
            raise ValueError(
                f"manifest seed={manifest['seed']} but this run requested "
                f"--seed {args.seed}. The corpus seed owns both the source "
                "split and sequence plan; use the manifest's seed or pass "
                "--rebuild-manifest explicitly."
            )
        if manifest['config_hash'] != cfg_hash:
            # ⚠ Not fatal: --hours and output paths legitimately change between
            # the train run and the val run.  But a changed corpus definition
            # with an unchanged split is worth seeing.
            print("  ⚠ config_hash differs from the one the manifest was built "
                  "with; the SPLIT is still the manifest's, not this config's")
    else:
        description = ("UNIFIED pool -- random per-chunk split at training time"
                       if expected_mode == 'unified'
                       else "from source lists")
        print(f"Manifest: building {description} (seed {args.seed}) ...")
        manifest = manifest_builders[expected_mode](cfg, args.seed)
        save_manifest(manifest, manifest_path)
        print(f"  written to {manifest_path}")

    manifest_mode = manifest.get('split_mode', 'disjoint')   # pre-'all' manifests predate the field
    if manifest_mode != expected_mode:
        raise ValueError(
            f"--split {args.split!r} needs a {expected_mode!r}-mode manifest, "
            f"but {manifest_path} is {manifest_mode!r}. Either change --split to "
            f"match the existing manifest, or pass --rebuild-manifest to replace "
            f"it (⚠ invalidates every corpus already generated from the old one)."
        )
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

    if any(p.scenario == 'echo_path_change' for p in plans):
        # Fail before any worker starts rendering, not partway through a
        # multi-hour run whenever one happens to draw the first
        # echo_path_change sequence (AecSequenceRenderer checks this too, but
        # only once a worker actually reaches that sequence).
        if not rooms_eligible_for_path_change(pools_for_split(manifest, args.split)):
            raise ValueError(
                "the plan includes 'echo_path_change' sequences but no room "
                "in this split has >= 2 RIR files; add more RIRs per room or "
                "set [scenarios] p_echo_path_change = 0"
            )

    chunk_sec = cfg.getfloat('sequence', 'chunk_sec')
    chunk_samples = int(round(chunk_sec * generation_sr))
    linear_aec_contract = linear_aec_contract_from_config(cfg)
    pending = _pending(
        plans, seqs_dir, args.resume,
        sample_rate=generation_sr,
        chunk_samples=chunk_samples,
        contract_hash=linear_aec_contract.fingerprint(),
        config_hash=cfg_hash, wav_encoding=args.wav_encoding,
    )
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
        'linear_aec': linear_aec_contract.as_dict(),
        'linear_aec_contract_hash': linear_aec_contract.fingerprint(),
        'chunk_sec': chunk_sec,
        'chunk_samples': chunk_samples,
        'n_sequences': len(plans),
        'n_chunks': sum(p.n_chunks for p in plans),
        'hours': planned_sec / 3600,
        'requested_hours': args.hours,
        'seed': args.seed,
        'wav_encoding': args.wav_encoding,
        'config': os.path.abspath(args.config),
        'config_hash': cfg_hash,
        'generator_commit': git_commit(),
        'manifest': os.path.abspath(manifest_path),
        'manifest_config_hash': manifest['config_hash'],
        'manifest_seed': int(manifest['seed']),
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
        description='Render AEC scenario sequences as 6-channel stem WAVs')
    parser.add_argument('--config', default='config.ini', help='Config file path')
    parser.add_argument('--output', default='data_aec', help='Output directory')
    parser.add_argument('--hours', type=float, default=8.0,
                        help='Target audio hours for the selected split')
    parser.add_argument('--workers', type=int, default=4,
                        help='Render workers (default: 4, 0 = single process)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip sequences whose metadata sidecar already exists')
    parser.add_argument('--seed', type=int, default=42,
                        help="Corpus seed. Fixes both the source split (moot for "
                             "--split all, a single unified pool) and every "
                             "sequence; multiple runs sharing a manifest (the "
                             "source-disjoint train/val pair) must share it.")
    parser.add_argument('--sample-rate', type=int, default=None,
                        help='Generation sample rate in Hz, overriding [signal] sr')
    parser.add_argument('--split', default='all', choices=ALL_SPLIT_NAMES,
                        help="'all' (default, the selected protocol): one "
                             "unified pool, no source split; pair with "
                             "AIAEC.training_common.split_dataset_by_sample at "
                             "load time. 'train'/'val': which side of a "
                             "SOURCE-DISJOINT pool to draw from instead (two "
                             "runs, same --seed/--manifest) -- use this for a "
                             "separate held-out generalisation corpus, or real "
                             "recordings, not the main training/validation pool.")
    parser.add_argument('--manifest', default=None,
                        help='Manifest path (default: <output>/manifest.json). '
                             'The train and val runs MUST use the same file '
                             "('all' only ever needs one run).")
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
