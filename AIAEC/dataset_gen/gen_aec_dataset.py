# -*- coding: utf-8 -*-
"""Offline rendering of AEC scenario sequences into 5-channel WAV chunks.

Mirrors ``AINR/dataset_gen/gen_dataset.py``'s flags so the two generators are driven
the same way, with two additions that the NR generator has no need for:

    --split       'all' (the selected protocol) or a SOURCE-DISJOINT side
                  ('train'/'val'), for a separate held-out generalisation corpus
    --manifest    optional persisted source-split decision

⚠ The split is decided BEFORE generation, over the source lists. With the same
config, source inventory and seed, it is rebuilt deterministically in memory;
pass the SAME explicit ``--manifest`` to both source-disjoint runs if the
source directories may change between them. See manifest.py for why splitting
after generation (as the NR loader does) is wrong here.

Usage (selected training protocol -- one unified pool, then a deterministic
random chunk split with AIAEC.training_common.split_dataset_by_sample at load
time; this is also --split's default):
    python3 gen_aec_dataset.py --config config.ini --output data_aec \\
        --hours 100 --split all --workers 4 --seed 42

Usage (source-disjoint -- two separate runs with the same config/seed; use this
for a held-out generalisation corpus, not the main training/validation pool):
    python3 gen_aec_dataset.py --config config.ini --output data_aec \\
        --hours 40 --split train --workers 4 --seed 42
    python3 gen_aec_dataset.py --config config.ini --output data_aec \\
        --hours 4  --split val   --workers 4 --seed 42

Output layout -- rendered audio and nothing else:
    data_aec/
      all/seqs/000000_000.wav       5-channel chunk, channels = STEM_ORDER
      all/seqs/000000_001.wav
      train/..., val/...            source-disjoint mode instead of all/,
                                     shared by both runs

By default there is no JSON output at all: `SSSSSS_CCC.wav` is the whole
on-disk contract, and pack_aec_dataset.py rebuilds everything it needs from
those filenames, the WAV headers and config.ini. ``--manifest PATH`` is an
explicit opt-in for persisting a source-disjoint split; the packer never reads
that control file.

What that costs, stated plainly, because these WERE real checks:
  * --resume no longer detects a config.ini/--seed edit between runs. It
    checks that each planned sequence's chunks exist with the right rate,
    length and channel count -- not that they were rendered by the config now
    in effect. Resume into a directory only with the config that started it.
  * The renderer's per-chunk labels (scenario, ser_db, snr_db, rir_id, ...) are
    no longer persisted. No trainer read them; a curriculum that needs them
    must measure the stems, which is possible precisely because they are
    stored separately.
What survives without any JSON, because it never depended on one:
  * A visible chunk WAV is complete -- every chunk is written to a `tmp.` name
    and renamed into place (see _save_chunk_atomic), so a killed run can never
    leave a truncated chunk that looks finished.
  * pack_aec_dataset.py re-checks channel count, length, rate and finiteness
    on every chunk it packs.

Then pack with pack_aec_dataset.py.
"""

import argparse
import configparser
import os
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
from .linear_aec import linear_aec_contract_from_config  # noqa: E402
from .seq_layout import (  # noqa: E402
    chunk_indices,
    chunk_path,
    parse_chunk_name,
    save_chunk_atomic,
    scan_chunks,
    sequence_chunk_paths,
    stale_temp_files,
)
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
    # combines them (e.g. D_hat = mic_postclip - linear_error) by ~1e-4.
    # int16 halves the disk cost and is fine for listening, not for
    # arithmetic.
    'float32': dict(encoding='PCM_F', bits_per_sample=32),
    'int16': dict(encoding='PCM_S', bits_per_sample=16),
}


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
        rendered = self._get_renderer().render(plan)
        n_chunks = len(rendered.chunk_meta)

        # A shorter re-render of a sequence that already exists on disk must
        # not leave the old run's surplus chunks behind: with no sidecar to
        # declare the chunk count, a stray SSSSSS_007.wav would simply be
        # packed as a seventh chunk.
        for stale in sequence_chunk_paths(self.seqs_dir, plan.sequence_id):
            if parse_chunk_name(stale)[1] >= n_chunks:
                os.remove(stale)

        write = WAV_ENCODINGS[self.encoding]
        for chunk_index in range(n_chunks):
            at = chunk_index * rendered.chunk_samples
            chunk = rendered.stems[:, at:at + rendered.chunk_samples].contiguous()
            save_chunk_atomic(
                self.seqs_dir, plan.sequence_id, chunk_index, chunk,
                self.cfg.getint('signal', 'sr'), write)
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
                          wav_encoding: str = 'float32') -> bool:
    """Does this sequence's audio already exist, in full and in the right shape?

    Every chunk 0..n_chunks-1 must be present with the expected rate, length,
    channel count and encoding, and there must be no chunk BEYOND that range
    (a leftover from a longer earlier render, which the packer would otherwise
    pack as a real chunk).

    ⚠ What this cannot check, now that nothing records how a chunk was made:
    whether the audio on disk was rendered by the config and --seed now in
    effect. A config.ini edit, or a --seed change that reshuffles which
    scenario each sequence_id draws, leaves same-shaped chunks that resume
    happily accepts. Resume into a directory only with the run that started
    it; use a new --output otherwise.
    """
    present = chunk_indices(seqs_dir, plan.sequence_id)
    if present != list(range(plan.n_chunks)):
        return False
    expected_encoding = WAV_ENCODINGS[wav_encoding]
    for chunk_index in present:
        try:
            info = torchaudio.info(chunk_path(seqs_dir, plan.sequence_id, chunk_index))
        except (OSError, RuntimeError, ValueError):
            return False
        if (
            info.sample_rate != sample_rate
            or info.num_frames != chunk_samples
            or info.num_channels != len(STEM_ORDER)
            or info.encoding != expected_encoding['encoding']
            or info.bits_per_sample != expected_encoding['bits_per_sample']
        ):
            return False
    return True


def _pending(plans: List[SequencePlan], seqs_dir: str, resume: bool, *,
             sample_rate: int, chunk_samples: int,
             wav_encoding: str = 'float32') -> List[SequencePlan]:
    if not resume:
        return list(plans)
    return [
        plan for plan in plans
        if not _sequence_is_complete(
            plan, seqs_dir, sample_rate=sample_rate,
            chunk_samples=chunk_samples, wav_encoding=wav_encoding,
        )
    ]


def _validate_existing_output(plans: List[SequencePlan], seqs_dir: str,
                              resume: bool) -> None:
    """Prevent a rerun from silently mixing two WAV inventories.

    Without a persisted run record, the filenames are the complete inventory.
    A plain rerun used to overwrite the planned prefix but leave any old
    higher sequence ids in place, and the packer would then include both.
    Resume may extend a corpus, but every existing sequence id must be part of
    the current plan so lowering ``--hours`` cannot leave an invisible tail.
    """
    existing = scan_chunks(seqs_dir)
    if not existing:
        return
    if not resume:
        raise FileExistsError(
            f"{seqs_dir} already contains {sum(map(len, existing.values()))} "
            "chunk WAV(s). Re-run with --resume to continue the same corpus, "
            "or choose an empty --output directory for a new render."
        )
    planned_ids = {int(plan.sequence_id) for plan in plans}
    unexpected = sorted(set(existing) - planned_ids)
    if unexpected:
        raise ValueError(
            f"--resume refused: {seqs_dir} contains sequence id(s) outside "
            f"the current --hours plan: {unexpected[:20]}"
            f"{'...' if len(unexpected) > 20 else ''}. Increase --hours to at "
            "least the original total or use a new output directory; otherwise "
            "the packer would silently include this stale tail."
        )


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
    # Only the manifest-reuse comparison needs this now (the --resume check has
    # no config record to compare against any more); cfg does not change after
    # the --sample-rate override above.
    cfg_hash = config_hash(cfg)

    split_dir = os.path.join(args.output, args.split)
    seqs_dir = os.path.join(split_dir, 'seqs')
    os.makedirs(seqs_dir, exist_ok=True)

    # Refuse an unsafe rerun before scanning the potentially large source/RIR
    # trees. The sequence plan depends only on config + seed, not the manifest.
    plans = plan_sequences(cfg, args.hours, args.seed, args.split)
    _validate_existing_output(plans, seqs_dir, args.resume)

    verify_wav_io(split_dir, args.wav_encoding)

    # --- the split decision -------------------------------------------------
    # One builder per manifest shape; persistence is explicit and optional.
    expected_mode = 'unified' if args.split == UNIFIED_SPLIT else 'disjoint'
    manifest_builders = {'disjoint': build_manifest, 'unified': build_unified_manifest}

    manifest_path = args.manifest
    if manifest_path is None:
        if args.rebuild_manifest:
            raise ValueError("--rebuild-manifest requires an explicit --manifest PATH")
        print(f"Source inventory: building {expected_mode} split in memory "
              f"(seed {args.seed}; no JSON written) ...")
        manifest = manifest_builders[expected_mode](cfg, args.seed)
    elif os.path.isfile(manifest_path) and not args.rebuild_manifest:
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
        source = manifest_path or 'the in-memory source inventory'
        raise ValueError(
            f"--split {args.split!r} needs a {expected_mode!r}-mode manifest, "
            f"but {source} is {manifest_mode!r}. Either change --split to "
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
    planned_sec = sum(p.n_chunks for p in plans) * cfg.getfloat('sequence', 'chunk_sec')

    if any(p.scenario == 'echo_path_change' for p in plans):
        # Fail before any worker starts rendering, not partway through a
        # multi-hour run whenever one happens to draw the first
        # echo_path_change sequence (AecSequenceRenderer checks this too, but
        # only once a worker actually reaches that sequence).
        rooms_to_rirs = manifest['splits'][args.split]['rooms_to_rirs']
        if not any(len(rirs) >= 2 for rirs in rooms_to_rirs.values()):
            raise ValueError(
                "the plan includes 'echo_path_change' sequences but no room "
                "in this split has >= 2 RIR files; add more RIRs per room or "
                "set [scenarios] p_echo_path_change = 0"
            )

    chunk_sec = cfg.getfloat('sequence', 'chunk_sec')
    chunk_samples = int(round(chunk_sec * generation_sr))
    # Constructed for its validation side effect: it raises if config.ini's
    # signal/linear_aec grid is not one the frozen PBFDKF supports, which must
    # fail before rendering, not at pack time. pack_aec_dataset.py rebuilds the
    # same contract from the same config.
    linear_aec_contract_from_config(cfg)
    pending = _pending(
        plans, seqs_dir, args.resume,
        sample_rate=generation_sr,
        chunk_samples=chunk_samples,
        wav_encoding=args.wav_encoding,
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
        for _done in iterator:
            pass

    # Sequence-level scenario counts across the WHOLE plan, not just this run's
    # pending slice, so a resumed run reports the corpus and not the remainder.
    plan_counts = {}
    for plan in plans:
        plan_counts[plan.scenario] = plan_counts.get(plan.scenario, 0) + 1

    print(f"\nDone. {len(plans)} sequences ({planned_sec / 3600:.3f} h) in "
          f"{seqs_dir}/, {(time.time() - started) / 60:.1f} min this run.")
    print(f"  Sequence scenarios: {plan_counts}")
    stale = stale_temp_files(seqs_dir)
    if stale:
        print(f"  ⚠ {len(stale)} leftover tmp.*.wav from an interrupted run "
              f"(ignored by every scan; delete them to reclaim the space)")
    print(f"  Next: python3 pack_aec_dataset.py --config {args.config} "
          f"--input {split_dir} "
          f"--output {os.path.join(args.output, 'packed', args.split)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Render AEC scenario sequences as 5-channel stem WAVs')
    parser.add_argument('--config', default='config.ini', help='Config file path')
    parser.add_argument('--output', default='data_aec', help='Output directory')
    parser.add_argument('--hours', type=float, default=8.0,
                        help='Target audio hours for the selected split')
    parser.add_argument('--workers', type=int, default=4,
                        help='Render workers (default: 4, 0 = single process)')
    parser.add_argument('--resume', action='store_true',
                        help='Skip sequences whose chunk WAVs are all already '
                             'present with the expected rate/length/channels. '
                             '⚠ Nothing records WHICH config rendered them, so '
                             'resume only into a directory started by this same '
                             'config and --seed')
    parser.add_argument('--seed', type=int, default=42,
                        help="Corpus seed. Fixes both the source split (moot for "
                             "--split all, a single unified pool) and every "
                             "sequence; source-disjoint train/val runs must "
                             "use the same config, source inventory and seed.")
    parser.add_argument('--sample-rate', type=int, default=None,
                        help='Generation sample rate in Hz, overriding [signal] sr')
    parser.add_argument('--split', default='all', choices=ALL_SPLIT_NAMES,
                        help="'all' (default, the selected protocol): one "
                             "unified pool, no source split; pair with "
                             "AIAEC.training_common.split_dataset_by_sample at "
                             "load time. 'train'/'val': which side of a "
                             "SOURCE-DISJOINT pool to draw from instead (two "
                             "runs, same config/seed) -- use this for a "
                             "separate held-out generalisation corpus, or real "
                             "recordings, not the main training/validation pool.")
    parser.add_argument('--manifest', default=None,
                        help='Optional path used to persist/reuse the source '
                             'split. Omit for WAV-only output; train/val then '
                             'rebuild the same split from config + source '
                             'inventory + seed.')
    parser.add_argument('--rebuild-manifest', action='store_true',
                        help='⚠ With an explicit --manifest PATH, redraw that '
                             'persisted source split. Invalidates every corpus '
                             'already generated from the old manifest.')
    parser.add_argument('--wav-encoding', default='float32',
                        choices=sorted(WAV_ENCODINGS),
                        help='float32 keeps the stem-sum identity exact; int16 '
                             'halves the disk cost and degrades it to ~1e-4')
    return parser


if __name__ == '__main__':
    gen_aec_dataset(build_parser().parse_args())
