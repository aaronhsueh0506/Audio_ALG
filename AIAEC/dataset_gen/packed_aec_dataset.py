"""Torch Dataset over packed AEC shards, returning ``(stems, meta)``.

Unlike ``AINR/dataset_gen/packed_dataset.py``, this does not hand a directory to
``ConcatDataset``. It owns the global index/metadata map so checkpoints can
record exact random-chunk splits and evaluation can reconstruct
``(sequence_id, chunk_index)`` order.
"""

import bisect
import glob
import hashlib
import json
import os
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .aec_features import STEM_ORDER, AecStems
from .linear_aec import LinearAecContract


__all__ = ['PackedAecDataset', 'aec_collate']


class PackedAecDataset(Dataset):
    """Open one ``.pt`` shard, or every shard in a directory, in order.

    ``__getitem__`` returns ``(stems, meta)`` where ``stems`` is the raw
    ``(5, T)`` tensor in ``STEM_ORDER``.  Wrap it in
    :class:`~AIAEC.dataset_gen.aec_features.AecStems` to read channels by name --
    :meth:`stems_of` does that for you.
    """

    def __init__(self, path: str, expected_sr: Optional[int] = None,
                 mmap: bool = False, verbose: bool = True):
        self.paths = self._resolve(path)
        self.mmap = mmap

        self._shards: List[dict] = []
        self._offsets: List[int] = []
        self._meta: List[dict] = []
        total = 0
        chunk_samples = None
        linear_aec_contract = None
        shard_identity = None

        for shard_path in self.paths:
            # ⚠ weights_only=False, unlike AINR/dataset_gen/packed_dataset.py. The
            # shard's 'meta' is a list of dicts of str/float/bool, which the
            # weights-only unpickler refuses.  That makes a shard as trusted as
            # any pickle: load only shards this pipeline produced.
            obj = torch.load(shard_path, map_location='cpu', mmap=mmap,
                             weights_only=False)
            self._validate(obj, shard_path, expected_sr)
            shard_contract = LinearAecContract.from_dict(obj['linear_aec'])
            if obj['linear_aec_contract_hash'] != shard_contract.fingerprint():
                raise ValueError(
                    f"{shard_path}: linear_aec_contract_hash does not match "
                    "the recorded linear_aec contract"
                )
            if linear_aec_contract is None:
                linear_aec_contract = shard_contract
            elif shard_contract != linear_aec_contract:
                raise ValueError(
                    f"{shard_path}: linear_aec contract differs from an earlier shard"
                )
            data = obj['data']
            current_identity = {
                'sr': int(obj['sr']),
                'dtype': str(data.dtype),
                'linear_aec_contract_hash': obj['linear_aec_contract_hash'],
            }
            if shard_identity is None:
                shard_identity = current_identity
            elif current_identity != shard_identity:
                changed = [
                    key for key in shard_identity
                    if current_identity[key] != shard_identity[key]
                ]
                raise ValueError(
                    f"{shard_path}: packed-corpus identity differs from an "
                    f"earlier shard in {changed}; do not mix shards from "
                    "different generation/packing runs"
                )
            if chunk_samples is None:
                chunk_samples = data.shape[2]
            elif data.shape[2] != chunk_samples:
                # ⚠ Shards of different chunk lengths cannot be batched, and a
                # sequence spanning both would silently change length mid-lane.
                raise ValueError(
                    f"{shard_path}: T={data.shape[2]} but an earlier shard has "
                    f"T={chunk_samples}; these were generated with different "
                    f"[sequence] chunk_sec and must not be mixed")
            self._shards.append(obj)
            self._offsets.append(total)
            self._meta.extend(obj['meta'])
            total += data.shape[0]

        self.sr = int(self._shards[0]['sr'])
        self.chunk_samples = int(chunk_samples)
        self.stems = tuple(self._shards[0]['stems'])
        self.dtype = self._shards[0]['data'].dtype
        self.linear_aec_contract = linear_aec_contract
        self.linear_aec_contract_hash = linear_aec_contract.fingerprint()
        self._total = total

        if verbose:
            size_mb = sum(s['data'].nbytes for s in self._shards) / 1024 ** 2
            storage = "disk-backed" if mmap else "in RAM"
            print(f"PackedAecDataset: {len(self.paths)} shard(s), {total} chunks, "
                  f"{self.n_sequences()} sequences, T={self.chunk_samples}, "
                  f"SR={self.sr}, dtype={self.dtype}, {size_mb:.0f} MB ({storage})")

    # ---------------- construction helpers ----------------

    @staticmethod
    def _resolve(path: str) -> List[str]:
        if os.path.isdir(path):
            # Every shard_*.pt in the directory, which is why pack_aec_dataset.py
            # refuses to write into a directory that already holds shards: a
            # leftover from an earlier, differently-configured pack would
            # otherwise silently join this corpus. The per-shard identity
            # check in __init__ is the second line of defence.
            shards = sorted(glob.glob(os.path.join(path, 'shard_*.pt')))
            if not shards:
                raise FileNotFoundError(f"no .pt shards under {path}")
            return shards
        if not os.path.isfile(path):
            raise FileNotFoundError(f"packed shard not found: {path}")
        return [path]

    @staticmethod
    def _validate(obj: dict, path: str, expected_sr: Optional[int]) -> None:
        for key in (
            'stems', 'data', 'sr', 'meta', 'linear_aec',
            'linear_aec_contract_hash',
        ):
            if key not in obj:
                raise ValueError(f"{path} is missing required key {key!r}")
        # ⚠ The channel order is the one property a consumer cannot notice being
        # wrong: swapping echo and near_speech trains a model that cancels the
        # talker and it converges perfectly well.
        if list(obj['stems']) != list(STEM_ORDER):
            raise ValueError(
                f"{path} declares stem order {list(obj['stems'])}, this code "
                f"expects {list(STEM_ORDER)}")
        data = obj['data']
        if data.ndim != 3 or data.shape[1] != len(STEM_ORDER):
            raise ValueError(
                f"{path}: data must be (N, {len(STEM_ORDER)}, T), got "
                f"{tuple(data.shape)}")
        if len(obj['meta']) != data.shape[0]:
            raise ValueError(
                f"{path}: {len(obj['meta'])} metadata entries for "
                f"{data.shape[0]} clips")
        # Each entry only carries where the clip came from -- the sampler and
        # full-sequence reconstruction need exactly these two fields, and they
        # are the two a filename can prove.
        for index, meta in enumerate(obj['meta']):
            for key in ('sequence_id', 'chunk_index'):
                if not isinstance(meta.get(key), int):
                    raise ValueError(
                        f"{path}: metadata entry {index} has no integer {key!r}")
        if expected_sr is not None and int(obj['sr']) != expected_sr:
            raise ValueError(
                f"{path}: sr={obj['sr']}, but the config requires {expected_sr}")

    # ---------------- Dataset ----------------

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, dict]:
        shard_index, local = self._locate(index)
        return self._shards[shard_index]['data'][local], self._meta[index]

    def _locate(self, index: int) -> Tuple[int, int]:
        if index < 0:
            index += self._total
        if not 0 <= index < self._total:
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._offsets, index) - 1
        return shard_index, index - self._offsets[shard_index]

    # ---------------- what the sampler needs ----------------

    def sequence_ids(self) -> List[int]:
        return [int(m['sequence_id']) for m in self._meta]

    def chunk_indices(self) -> List[int]:
        return [int(m['chunk_index']) for m in self._meta]

    def n_sequences(self) -> int:
        return len(set(self.sequence_ids()))

    def meta(self, index: int) -> dict:
        return self._meta[index]

    def stems_of(self, index: int) -> AecStems:
        """``AecStems`` view, so channels are read by name."""
        return AecStems(self[index][0], self.stems)

    # ---------------- convenience ----------------

    def indices_where(self, **conditions) -> List[int]:
        """Dataset indices whose metadata matches every keyword.

        Only ``sequence_id``/``chunk_index`` are recorded, so this now answers
        "which chunks belong to sequence N", not "which chunks are dropouts" --
        the renderer's per-chunk labels are no longer persisted (see
        pack_aec_dataset.py). A subset that needs an acoustic property has to
        measure the stems, which is possible precisely because they are stored
        separately.
        """
        return [
            i for i, m in enumerate(self._meta)
            if all(m.get(key) == value for key, value in conditions.items())
        ]

    def describe(self) -> str:
        return json.dumps({
            'chunks': len(self),
            'sequences': self.n_sequences(),
            'sr': self.sr,
            'chunk_samples': self.chunk_samples,
            'linear_aec_contract_hash': self.linear_aec_contract_hash,
            'dataset_fingerprint': self.fingerprint(),
        }, indent=2, sort_keys=True)

    def fingerprint(self) -> str:
        """Stable corpus identity used by checkpoint/data-resume contracts.

        ⚠ This identifies the packed corpus's SHAPE and inventory -- rate,
        geometry, dtype, linear-AEC contract, and every (sequence, chunk) it
        contains -- not the audio itself. Two corpora rendered from different
        configs or seeds into the same shape now fingerprint identically,
        because nothing on disk records which config rendered a chunk. A
        checkpoint resumed against the wrong corpus of the same shape will not
        be caught here.
        """
        digest = hashlib.sha256()
        header = {
            'sr': self.sr,
            'chunk_samples': self.chunk_samples,
            'stems': list(self.stems),
            'dtype': str(self.dtype),
            'linear_aec_contract_hash': self.linear_aec_contract_hash,
        }
        digest.update(json.dumps(
            header, sort_keys=True, separators=(',', ':'),
        ).encode('utf-8'))
        for meta in self._meta:
            identity = (int(meta['sequence_id']), int(meta['chunk_index']))
            digest.update(repr(identity).encode('utf-8'))
            digest.update(b'\0')
        return digest.hexdigest()


def aec_collate(batch):
    """Stack stems, keep metadata as a list of dicts.

    torch's default collate turns a list of dicts into a dict of lists, which
    silently converts ``ser_db = inf`` into a tensor and loses which clip each
    value came from once anything is filtered.  Keeping the dicts intact costs
    nothing and keeps ``meta[lane]`` meaning "lane's metadata".
    """
    stems = torch.stack([item[0] for item in batch])
    return stems, [item[1] for item in batch]
