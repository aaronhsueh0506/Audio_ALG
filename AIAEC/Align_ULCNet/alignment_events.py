"""Replayable per-hop alignment events for Align-ULCNet (Python side).

The deployed C stream (stream/aiaec/ulcnet/ulcnet_stream.c) receives one
``(changed, solid)`` pair per model hop from the linear AEC's delay state and
turns it into a restart decision with ONE rule, carried across packets:

    event = changed or (an earlier hop exists and previous solid == 0 and solid == 1)

An event hop resets the model's recurrent state before that hop and the hop is
a residual bypass (enhanced frame = error spectrum unchanged). The Python
linear AEC has no delay-state accessor, so Python cannot derive these pairs
from audio; it replays them from a record file. This module holds the rule
once (``EdgeTracker``) and the record format; the WAV inference and the
calibration recorder both read events only through ``AlignmentEvents``.

Record file, one record per real model hop (hop k = STFT frame k, 0-based
list order; the number of records must equal ceil(samples / hop_len) at the
model rate):

  .json  ``{"schema": "aiaec-align-events-v1", "events": [[changed, solid], ...]}``
         or a bare list; an item may also be ``{"changed": 0, "solid": 1}``
  .csv   header ``changed,solid`` (an optional ``hop`` column must count 0,1,..)
  .npz   1-D arrays ``changed`` and ``solid`` (bool, integer or float)

Values are exactly 0/1 (JSON may also use true/false); anything else is refused, never cast. The raw pairs go through ``EdgeTracker``; no
file carries a pre-computed restart flag, so the edge rule cannot be bypassed.
"""

from __future__ import annotations

import csv
import json
import math
import os
from typing import List, Optional, Sequence

EVENTS_SCHEMA = 'aiaec-align-events-v1'
EVENT_SUFFIXES = ('.json', '.csv', '.npz')
EVENTS_HELP = (
    'Replay per-hop linear-AEC alignment events from a record file (one '
    '(changed, solid) pair per real model hop, JSON/CSV/NPZ; see '
    'Align_ULCNet/README.md). An event hop resets the model state at that hop '
    'and bypasses it (enhanced frame = error spectrum); at N > 1 it is the '
    'graph\'s frame_reset control. Without --events no event ever occurs. '
    'Single-file runs take a file; calib also takes a directory holding '
    '<relative wav name without .wav>.events.json|csv|npz per pair')


class EdgeTracker:
    """The C stream's alignment-edge rule, one hop at a time
    (ulcnet_stream.c ``apply_alignment_edge`` / ``packet_events``)."""

    def __init__(self):
        self.had_hop = False
        self.prev_solid = False

    def step(self, changed, solid) -> int:
        edge = bool(changed) or (self.had_hop and not self.prev_solid
                                 and bool(solid))
        self.had_hop = True
        self.prev_solid = bool(solid)
        return int(edge)


def edge_flags(changed: Sequence, solid: Sequence) -> List[int]:
    """Restart flags for a whole stream, one tracker pass (state carried
    across every packet boundary by construction)."""
    tracker = EdgeTracker()
    return [tracker.step(c, s) for c, s in zip(changed, solid)]


def real_hop_count(samples: int, hop_len: int) -> int:
    """Model hops that hold real samples: the last one may be partial."""
    return int(math.ceil(samples / hop_len))


def _bit(value, where, index=None):
    if isinstance(value, bool) or value in (0, 1):
        return int(value)
    raise ValueError('%s: event values must be 0/1, got %r%s' % (
        where, value, '' if index is None else ' at record %d' % index))


def _csv_bit(text, where, index, column):
    try:
        number = float(text)
    except (TypeError, ValueError):
        raise ValueError('%s: event values must be 0/1, got %r in column %s '
                         'at record %d' % (where, text, column, index)
                         ) from None
    return _bit(number, where, index)


def _npz_column(data, name, path):
    """A 1-D numeric array whose values are exactly 0/1, as a list of ints."""
    import numpy as np
    try:
        array = data[name]
    except ValueError as error:
        raise ValueError('%s: NPZ array %s cannot be read (%s)'
                         % (path, name, error)) from None
    if array.ndim != 1:
        raise ValueError('%s: NPZ array %s must be 1-D, got shape %s'
                         % (path, name, tuple(array.shape)))
    if array.dtype.kind not in 'biuf':
        raise ValueError('%s: NPZ array %s must be bool, integer or float, '
                         'got dtype %s' % (path, name, array.dtype))
    bad = np.flatnonzero(~np.isin(array, (0, 1)))
    if bad.size:
        raise ValueError(
            '%s: NPZ array %s: event values must be 0/1, got %s at records %s'
            % (path, name, array[bad[:5]].tolist(),
               bad[:5].tolist() + (['...'] if bad.size > 5 else [])))
    return array.astype(int).tolist()


class AlignmentEvents:
    """Edge flags for one clip. Frames past the record (tail frames the
    analyser emits after the last real hop) and packet padding are
    never events and never touch the tracker."""

    def __init__(self, changed: Sequence, solid: Sequence,
                 source: Optional[str] = None):
        if len(changed) != len(solid):
            raise ValueError('changed/solid record lengths differ (%d vs %d)'
                             % (len(changed), len(solid)))
        where = source or 'events'
        self.source = source
        self.changed = [_bit(v, where, i) for i, v in enumerate(changed)]
        self.solid = [_bit(v, where, i) for i, v in enumerate(solid)]
        self.flags = edge_flags(self.changed, self.solid)

    @property
    def hops(self) -> int:
        return len(self.flags)

    def flag(self, index: int) -> int:
        return self.flags[index] if 0 <= index < len(self.flags) else 0

    def packet_control(self, first: int, real: int, size: int) -> List[int]:
        """``frame_reset`` values of a packet of ``size`` slots whose first
        ``real`` slots are frames ``first..``; padding slots are never events."""
        return [self.flag(first + k) if k < real else 0 for k in range(size)]

    def require_hops(self, samples: int, hop_len: int) -> None:
        want = real_hop_count(samples, hop_len)
        if self.hops != want:
            raise ValueError(
                '%s holds %d hop records, the clip has %d real hops (%d '
                'samples at hop %d); the record must cover exactly the real '
                'hops (no padding hops)' % (self.source or 'event record',
                                            self.hops, want, samples, hop_len))

    def event_frames(self) -> List[int]:
        return [i for i, flag in enumerate(self.flags) if flag]

    @classmethod
    def load(cls, path: str) -> 'AlignmentEvents':
        path = str(path)
        suffix = os.path.splitext(path)[1].lower()
        if suffix == '.json':
            with open(path, 'r', encoding='utf-8') as stream:
                data = json.load(stream)
            if isinstance(data, dict):
                if data.get('schema', EVENTS_SCHEMA) != EVENTS_SCHEMA:
                    raise ValueError('%s: schema %r, expected %s'
                                     % (path, data.get('schema'),
                                        EVENTS_SCHEMA))
                data = data.get('events')
            if not isinstance(data, list):
                raise ValueError('%s: expected a list of event records' % path)
            changed, solid = [], []
            for item in data:
                if isinstance(item, dict):
                    pair = (item.get('changed'), item.get('solid'))
                else:
                    pair = tuple(item) if isinstance(item, (list, tuple)) else ()
                if len(pair) != 2 or None in pair:
                    raise ValueError('%s: every record needs changed and '
                                     'solid, got %r' % (path, item))
                changed.append(pair[0]), solid.append(pair[1])
        elif suffix == '.csv':
            with open(path, 'r', encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            if not rows or not {'changed', 'solid'} <= set(rows[0]):
                raise ValueError('%s: CSV needs a header with changed and '
                                 'solid columns and at least one row' % path)
            if 'hop' in rows[0]:
                try:
                    hops = [int(r['hop']) for r in rows]
                except (TypeError, ValueError):
                    hops = None
                if hops != list(range(len(rows))):
                    raise ValueError('%s: the hop column must count 0,1,2,..'
                                     % path)
            changed = [_csv_bit(r['changed'], path, i, 'changed')
                       for i, r in enumerate(rows)]
            solid = [_csv_bit(r['solid'], path, i, 'solid')
                     for i, r in enumerate(rows)]
        elif suffix == '.npz':
            import numpy as np
            loaded = np.load(path)
            if not hasattr(loaded, 'files'):
                raise ValueError('%s: not an NPZ archive' % path)
            with loaded as data:
                if not {'changed', 'solid'} <= set(data.files):
                    raise ValueError('%s: NPZ needs arrays changed and solid'
                                     % path)
                changed = _npz_column(data, 'changed', path)
                solid = _npz_column(data, 'solid', path)
        else:
            raise ValueError('%s: event record must be .json, .csv or .npz'
                             % path)
        return cls(changed, solid, source=path)


def find_events_file(root: str, relative_wav: str) -> str:
    """The record of one pair inside an events directory:
    ``<relative path without .wav>.events.{json,csv,npz}``, exactly one."""
    stem = os.path.splitext(relative_wav)[0]
    hits = [os.path.join(root, stem + '.events' + suffix)
            for suffix in EVENT_SUFFIXES
            if os.path.isfile(os.path.join(root, stem + '.events' + suffix))]
    if len(hits) != 1:
        raise ValueError(
            'events directory %s: expected exactly one of %s.events%s for '
            '%s, found %d' % (root, stem, '|'.join(EVENT_SUFFIXES),
                              relative_wav, len(hits)))
    return hits[0]
