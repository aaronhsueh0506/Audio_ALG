"""Source-disjoint train/validation split, decided BEFORE any audio is rendered.

WHY THIS IS NOT THE NR SPLIT
----------------------------
``AINR/dataset_gen/loader.py`` splits the *generated* corpus: it draws a permutation
over finished clips and holds 5% out.  That is fine when every clip is an
independent draw, and it is wrong here.

Two AEC clips rendered from the same speaker, the same echo RIR and the same
loudspeaker model are not independent -- they share the exact nonlinearity, the
exact room response and the exact voice.  Split after generation and both halves
of that pair land on opposite sides of the fence, so validation measures how
well the model memorised a room rather than how well it cancels echo.  The
number that comes out is high, stable, reproducible, and meaningless.

So the split happens over the SOURCE LISTS, before a single sample is rendered:

    speaker      disjoint   (which also makes speech FILES disjoint)
    noise        disjoint
    room / RIR   disjoint   (a room is split whole; its RIRs never straddle)
    device       disjoint   (loudspeaker EQ + nonlinearity model)

⚠ Device disjointness is the aggressive one.  It means validation is scored on
loudspeaker nonlinearities the model has never seen, which is the honest
question for a shipped product and a materially harder task than the usual AEC
benchmark.  ``[split] device_split = shared`` relaxes it, and then the val
score answers a different, easier question -- say so if you use it.

The manifest is written to disk so that the train run and the val run, which
are separate invocations, provably draw from the same decision.
"""

import configparser
import hashlib
import json
import os
import random
import re
import time
from typing import Dict, List, Optional, Sequence

import torchaudio

from AINR.dataset_gen.dataset import estimate_rt60


__all__ = [
    'MANIFEST_VERSION',
    'SourcePools',
    'assert_source_disjoint',
    'build_manifest',
    'config_hash',
    'load_manifest',
    'pools_for_split',
    'save_manifest',
    'summarise',
]


# Bumped when the manifest's meaning changes, so a stale file is rejected
# rather than reinterpreted.
MANIFEST_VERSION = 'aec_manifest_v1'

SPLITS = ('train', 'val')

# Every axis that must not straddle the split.  Kept as one tuple so the
# builder, the checker and the test cannot disagree about what "disjoint" covers.
DISJOINT_AXES = ('speakers', 'speech_files', 'noise_ids', 'noise_files',
                 'rooms', 'rir_files', 'devices')


def config_hash(cfg: configparser.ConfigParser) -> str:
    """Stable hash of a config's full contents.

    Recorded in the manifest and in every shard, so a corpus generated before a
    config edit can be told apart from one generated after it.  Sorted, because
    configparser's section order depends on file order and would otherwise make
    two identical configs hash differently.
    """
    items = []
    for section in sorted(cfg.sections()):
        for key in sorted(cfg[section]):
            items.append(f"{section}.{key}={cfg[section][key]}")
    return hashlib.sha256("\n".join(items).encode('utf-8')).hexdigest()[:16]


# ============================================================
# Source identity
# ============================================================

def _scan_wavs(root: str) -> List[str]:
    """Recursively list .wav files, sorted, as paths relative to ``root``.

    Relative on purpose: an absolute path pins the manifest to one machine's
    directory layout, and the corpus is meant to be regenerable elsewhere.
    """
    if not os.path.isdir(root):
        raise FileNotFoundError(f"source directory does not exist: {root}")
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.lower().endswith('.wav'):
                found.append(os.path.relpath(os.path.join(dirpath, name), root))
    if not found:
        raise FileNotFoundError(f"no .wav files under {root}")
    return sorted(found)


def _grouping_key(rel_path: str, pattern: Optional[str], fallback: str) -> str:
    """Extract a speaker/noise/room id from a relative path.

    ``pattern`` is a regex whose first group is the id; corpora differ too much
    for a single hardcoded rule (DNS4 read_speech encodes the reader in the
    filename, most RIR sets encode the room in a directory).  When no pattern
    matches, ``fallback`` selects a structural rule instead of silently
    producing one giant group -- a single group would make a disjoint split
    impossible and the failure would surface as an unexplained error much later.
    """
    if pattern:
        match = re.search(pattern, rel_path.replace(os.sep, '/'))
        if match:
            return match.group(1) if match.groups() else match.group(0)
    if fallback == 'parent_dir':
        parent = os.path.dirname(rel_path)
        return parent.replace(os.sep, '/') if parent else '.'
    if fallback == 'top_dir':
        head = rel_path.replace(os.sep, '/').split('/')[0]
        return head if head != rel_path else '.'
    if fallback == 'stem':
        return os.path.splitext(os.path.basename(rel_path))[0]
    raise ValueError(f"unknown grouping fallback {fallback!r}")


def _group(rel_paths: Sequence[str], pattern: Optional[str],
           fallback: str) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for rel in rel_paths:
        groups.setdefault(_grouping_key(rel, pattern, fallback), []).append(rel)
    return groups


def _split_groups(groups: Dict[str, List[str]], val_fraction: float,
                  rng: random.Random, axis: str) -> Dict[str, List[str]]:
    """Assign whole groups to train/val.  Never splits a group."""
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError(
            f"{axis}: only {len(keys)} distinct group(s) found, so a "
            f"source-disjoint split is impossible. Check the grouping pattern "
            f"in [split], or point at a corpus with more variety."
        )
    rng.shuffle(keys)
    n_val = max(1, int(round(len(keys) * val_fraction)))
    n_val = min(n_val, len(keys) - 1)   # train must keep at least one group
    return {'val': sorted(keys[:n_val]), 'train': sorted(keys[n_val:])}


# ============================================================
# RT60
# ============================================================

def _rir_rt60(root: str, rel_paths: Sequence[str], sr: int,
              rt60_min: float, rt60_max: float,
              progress: bool = True) -> Dict[str, float]:
    """RT60 per RIR, filtered to the configured range.

    Computed here rather than at render time so the manifest is the single
    record of which RIRs the corpus may use. ``AINR/dataset_gen/dataset.py`` caches
    this in a side file keyed by a config hash; the manifest already is that
    record, so no second cache is introduced.
    """
    kept: Dict[str, float] = {}
    iterator = rel_paths
    if progress:
        try:
            import tqdm
            iterator = tqdm.tqdm(rel_paths, desc="RT60 scan")
        except ImportError:
            pass
    for rel in iterator:
        try:
            audio, file_sr = torchaudio.load(os.path.join(root, rel))
            audio = audio[0]
            if file_sr != sr:
                audio = torchaudio.functional.resample(audio, file_sr, sr)
            rt60 = estimate_rt60(audio, sr)
        except Exception:
            continue
        if rt60_min <= rt60 <= rt60_max:
            kept[rel] = float(rt60)
    if not kept:
        raise ValueError(
            f"no RIR passed the RT60 filter [{rt60_min}, {rt60_max}] s under {root}"
        )
    return kept


# ============================================================
# Manifest
# ============================================================

def build_manifest(cfg: configparser.ConfigParser, seed: int,
                   progress: bool = True) -> dict:
    """Decide the split over source lists.  Renders nothing."""
    sr = cfg.getint('signal', 'sr')
    speech_dir = cfg.get('paths', 'speech_dir')
    noise_dir = cfg.get('paths', 'noise_dir')
    rir_dir = cfg.get('paths', 'rir_dir')

    val_fraction = cfg.getfloat('split', 'val_fraction', fallback=0.1)
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"[split] val_fraction must be in (0, 1), got {val_fraction}")
    device_split = cfg.get('split', 'device_split', fallback='disjoint')
    if device_split not in ('disjoint', 'shared'):
        raise ValueError(
            f"[split] device_split must be 'disjoint' or 'shared', got {device_split!r}"
        )

    speaker_pattern = cfg.get('split', 'speaker_id_regex', fallback='') or None
    noise_pattern = cfg.get('split', 'noise_id_regex', fallback='') or None
    room_pattern = cfg.get('split', 'room_id_regex', fallback='') or None

    rt60_min = cfg.getfloat('rir', 'rt60_min', fallback=0.1)
    rt60_max = cfg.getfloat('rir', 'rt60_max', fallback=1.2)

    device_ids = [d.strip() for d in cfg.get('devices', 'device_ids').split(',') if d.strip()]
    if len(set(device_ids)) != len(device_ids):
        raise ValueError("[devices] device_ids contains duplicates")

    # The split RNG is dedicated and seeded only from `seed`, so the manifest
    # depends on (sources, seed) alone.  Re-running the generator with a
    # different --hours must not move a single speaker across the fence.
    rng = random.Random(seed)

    speech_rel = _scan_wavs(speech_dir)
    noise_rel = _scan_wavs(noise_dir)
    rir_rel_all = _scan_wavs(rir_dir)

    rt60_map = _rir_rt60(rir_dir, rir_rel_all, sr, rt60_min, rt60_max, progress)
    rir_rel = sorted(rt60_map)

    speakers = _group(speech_rel, speaker_pattern, 'parent_dir')
    noises = _group(noise_rel, noise_pattern, 'stem')
    rooms = _group(rir_rel, room_pattern, 'parent_dir')

    speaker_split = _split_groups(speakers, val_fraction, rng, 'speaker')
    noise_split = _split_groups(noises, val_fraction, rng, 'noise')
    room_split = _split_groups(rooms, val_fraction, rng, 'room')

    if device_split == 'disjoint':
        device_groups = {d: [d] for d in device_ids}
        dev_split = _split_groups(device_groups, val_fraction, rng, 'device')
    else:
        # ⚠ Sharing devices makes the val score answer "can it cancel a KNOWN
        # loudspeaker's echo in an unseen room" instead of "can it cancel an
        # unseen loudspeaker".  Recorded in the manifest so a reader of the
        # score knows which question was asked.
        dev_split = {'train': sorted(device_ids), 'val': sorted(device_ids)}

    manifest = {
        'version': MANIFEST_VERSION,
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'seed': int(seed),
        'sr': sr,
        'val_fraction': val_fraction,
        'device_split': device_split,
        'config_hash': config_hash(cfg),
        'roots': {
            # Relative source lists plus the roots they were taken under, so a
            # corpus can move without invalidating the split decision.
            'speech_dir': speech_dir,
            'noise_dir': noise_dir,
            'rir_dir': rir_dir,
        },
        'rt60': rt60_map,
        'splits': {},
    }

    for split in SPLITS:
        manifest['splits'][split] = {
            'speakers': speaker_split[split],
            'speech_files': sorted(
                f for spk in speaker_split[split] for f in speakers[spk]
            ),
            'noise_ids': noise_split[split],
            'noise_files': sorted(
                f for nid in noise_split[split] for f in noises[nid]
            ),
            'rooms': room_split[split],
            'rir_files': sorted(
                f for room in room_split[split] for f in rooms[room]
            ),
            'devices': dev_split[split],
            'rooms_to_rirs': {room: sorted(rooms[room]) for room in room_split[split]},
        }

    assert_source_disjoint(manifest)
    return manifest


def assert_source_disjoint(manifest: dict) -> None:
    """Raise unless every axis really is disjoint between train and val.

    Called at build time AND available to tests, because "we split by speaker"
    is a claim about code that was true when it was written.  A regex change
    that makes two readers collapse into one id would otherwise reintroduce the
    leak with no visible symptom.
    """
    shared_devices = manifest.get('device_split') == 'shared'
    for axis in DISJOINT_AXES:
        if axis == 'devices' and shared_devices:
            continue
        train = set(manifest['splits']['train'].get(axis, ()))
        val = set(manifest['splits']['val'].get(axis, ()))
        overlap = train & val
        if overlap:
            sample = sorted(overlap)[:5]
            raise ValueError(
                f"source leak on axis {axis!r}: {len(overlap)} shared entries "
                f"(e.g. {sample}). Validation would measure memorisation."
            )
        if not train or not val:
            raise ValueError(f"axis {axis!r} is empty on one side of the split")


def save_manifest(manifest: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)


def load_manifest(path: str) -> dict:
    with open(path, 'r') as handle:
        manifest = json.load(handle)
    if manifest.get('version') != MANIFEST_VERSION:
        raise ValueError(
            f"{path} is manifest version {manifest.get('version')!r}, this code "
            f"writes {MANIFEST_VERSION!r}. Rebuild it rather than reinterpreting "
            f"a split whose meaning may have changed."
        )
    assert_source_disjoint(manifest)
    return manifest


class SourcePools:
    """The absolute paths one split may draw from.

    ⚠ The renderer must never look outside these lists.  This object is the
    only place that turns a manifest into paths, so there is exactly one way
    for a source to reach a render.
    """

    __slots__ = ('split', 'speech_files', 'noise_files', 'devices',
                 'rooms', 'rirs_by_room', 'rt60', 'speaker_of', 'noise_of',
                 'rir_id_of', 'room_of')

    def __init__(self, split: str, manifest: dict):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        roots = manifest['roots']
        entry = manifest['splits'][split]

        self.split = split
        self.speech_files = [
            os.path.join(roots['speech_dir'], rel) for rel in entry['speech_files']
        ]
        self.noise_files = [
            os.path.join(roots['noise_dir'], rel) for rel in entry['noise_files']
        ]
        self.devices = list(entry['devices'])
        self.rooms = list(entry['rooms'])
        self.rirs_by_room = {
            room: [os.path.join(roots['rir_dir'], rel) for rel in rels]
            for room, rels in entry['rooms_to_rirs'].items()
        }
        self.rt60 = {
            os.path.join(roots['rir_dir'], rel): value
            for rel, value in manifest.get('rt60', {}).items()
        }

        # Source-id lookups, so the renderer can record WHICH source produced a
        # clip without re-deriving the grouping rule and getting it subtly
        # different from the rule the split was actually made with.
        self.speaker_of = {
            absolute: _group_lookup(rel, entry['speakers'])
            for rel, absolute in zip(entry['speech_files'], self.speech_files)
        }
        self.noise_of = {
            absolute: _group_lookup(rel, entry['noise_ids'])
            for rel, absolute in zip(entry['noise_files'], self.noise_files)
        }
        self.rir_id_of = {}
        self.room_of = {}
        for room, rels in entry['rooms_to_rirs'].items():
            for rel in rels:
                absolute = os.path.join(roots['rir_dir'], rel)
                self.rir_id_of[absolute] = rel.replace(os.sep, '/')
                self.room_of[absolute] = room

        for name, pool in (('speech', self.speech_files),
                           ('noise', self.noise_files),
                           ('device', self.devices),
                           ('room', self.rooms)):
            if not pool:
                raise ValueError(f"split {split!r} has an empty {name} pool")

    def __repr__(self):
        return (f"SourcePools(split={self.split!r}, speech={len(self.speech_files)}, "
                f"noise={len(self.noise_files)}, rooms={len(self.rooms)}, "
                f"devices={len(self.devices)})")


def _group_lookup(rel_path: str, group_ids: Sequence[str]) -> str:
    """Match a relative source path back to the group id it was split under.

    The manifest stores both the group list and the file list, so the id can be
    recovered by substring match without re-running the regex.  Longest first,
    so 'reader_067' cannot shadow 'reader_0671'.
    """
    normalised = rel_path.replace(os.sep, '/')
    for group_id in sorted(group_ids, key=len, reverse=True):
        if group_id in normalised:
            return group_id
    return os.path.splitext(os.path.basename(normalised))[0]


def pools_for_split(manifest: dict, split: str) -> SourcePools:
    return SourcePools(split, manifest)


def summarise(manifest: dict) -> str:
    lines = [
        f"manifest {manifest['version']} seed={manifest['seed']} "
        f"config_hash={manifest['config_hash']} device_split={manifest['device_split']}"
    ]
    for split in SPLITS:
        entry = manifest['splits'][split]
        lines.append(
            f"  {split:<5} speakers={len(entry['speakers'])} "
            f"speech={len(entry['speech_files'])} "
            f"noise={len(entry['noise_files'])} "
            f"rooms={len(entry['rooms'])} rirs={len(entry['rir_files'])} "
            f"devices={len(entry['devices'])}"
        )
    return "\n".join(lines)
