"""Calibration artifact formats shared by stateless model exporters."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

import numpy as np


CALIBRATION_FORMATS = ('bin', 'npz')


def capture_calibration_inputs(captured, names, tensors):
    """Append one complete accelerator invocation without changing shape.

    Every saved sample includes the graph's batch dimension.  Only the
    leading dimension added later by ``np.stack`` is the calibration-frame
    axis; dropping ``tensor[0]`` here would make each BIN file disagree with
    the exported ONNX input shape.
    """
    if len(names) != len(tensors):
        raise ValueError('calibration input names/tensors length mismatch')
    values = []
    for name, tensor in zip(names, tensors):
        if hasattr(tensor, 'detach'):
            value = tensor.detach().cpu().numpy()
        else:
            value = np.asarray(tensor)
        if not np.isfinite(value).all():
            raise RuntimeError(
                'non-finite calibration input in %s' % name
            )
        values.append(value.copy())
    for name, value in zip(names, values):
        captured.setdefault(name, []).append(value)


def resolve_calibration_format(output_path, requested=None):
    """Resolve an explicit format or infer NPZ from its filename suffix."""
    if requested is not None:
        if requested not in CALIBRATION_FORMATS:
            raise ValueError('unsupported calibration format: %s' % requested)
        result = requested
    else:
        result = 'npz' if str(output_path).lower().endswith('.npz') else 'bin'
    if result == 'npz' and not str(output_path).lower().endswith('.npz'):
        raise ValueError('NPZ calibration output must end in .npz')
    if result == 'bin' and str(output_path).lower().endswith('.npz'):
        raise ValueError('BIN calibration output must be a directory name')
    return result


def calibration_output(output_path, requested=None, written=()):
    """``output_path`` without trailing separators, and its format.

    A run checks its output here before any work, instead of failing a write
    after it: a BIN directory that already exists is refused, and so is a
    file of ``written`` (the graph, the page: what else the run writes) at
    the output path or inside a BIN output."""
    path = _without_trailing_separators(output_path)
    artifact_format = resolve_calibration_format(path, requested)
    target = os.path.abspath(path)
    if artifact_format == 'bin':
        _refuse_existing_bin(target, ValueError)
    for other in filter(None, written):
        other_path = os.path.abspath(other)
        if other_path == target or (artifact_format == 'bin'
                                    and other_path.startswith(target + os.sep)):
            raise ValueError('%s would be written at or inside the '
                             'calibration output %s' % (other, target))
    return path, artifact_format


def _without_trailing_separators(path):
    path = str(path)
    return path.rstrip('/' + os.sep) or path


def _refuse_existing_bin(target, error):
    if os.path.lexists(target):
        raise error('binary calibration output already exists: %s' % target)


def calibration_report_path(output_path, artifact_format):
    """Where the report of a calibration artifact lives.

    BIN keeps ``manifest.json`` inside its directory. NPZ writes
    ``<stem>.calib.json``: a calibration run also exports ``<stem>.onnx``,
    whose exporter owns ``<stem>.json``, so the report must not take that name.
    """
    output_path = os.path.abspath(output_path)
    if artifact_format == 'npz':
        return os.path.splitext(output_path)[0] + '.calib.json'
    return os.path.join(output_path, 'manifest.json')


def sibling_onnx_path(output_path, override=None):
    """Where a calibration run writes the graph its tensors bind to.

    Beside the artifact, dropping trailing separators and a ``.npz`` suffix
    first, so ``calib/foo``, ``calib/foo/`` and ``calib/foo.npz`` all pair
    with ``calib/foo.onnx``.
    """
    if override:
        return override
    output_path = _without_trailing_separators(output_path)
    base = output_path[:-4] if output_path.endswith('.npz') else output_path
    return base + '.onnx'


def _validate_arrays(arrays):
    if not arrays:
        raise ValueError('calibration requires at least one input tensor')
    frame_count = None
    for name, value in arrays.items():
        if name != os.path.basename(name) or name in ('.', '..'):
            raise ValueError('invalid calibration tensor name: %r' % name)
        if not isinstance(value, np.ndarray) or value.ndim < 1:
            raise ValueError('%s must be an ndarray with a frame axis' % name)
        if frame_count is None:
            frame_count = int(value.shape[0])
        elif int(value.shape[0]) != frame_count:
            raise ValueError('calibration tensors have different frame counts')
    if not frame_count:
        raise ValueError('calibration requires at least one frame')
    return frame_count


def write_calibration_artifact(output_path, arrays, report, artifact_format):
    """Write NPZ or per-frame/per-tensor little-endian BIN calibration.

    ``arrays`` maps graph input names to arrays whose leading dimension is the
    captured invocation index. BIN creates one directory per input and writes
    zero-padded 0-based ``<name>_0000.bin``, ``<name>_0001.bin``, ... beneath
    it (width grows past 10000 frames so a lexicographic listing always equals
    frame order).
    """
    artifact_format = resolve_calibration_format(output_path, artifact_format)
    frame_count = _validate_arrays(arrays)
    report = dict(report)
    if int(report.get('frames', frame_count)) != frame_count:
        raise ValueError('report frame count disagrees with calibration arrays')
    report['frames'] = frame_count

    if artifact_format == 'npz':
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report['artifact_format'] = 'numpy_npz'
        np.savez_compressed(output_path, **arrays)
        report_path = calibration_report_path(output_path, 'npz')
        with open(report_path, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write('\n')
        return report_path

    output_path = os.path.abspath(output_path)
    _refuse_existing_bin(output_path, FileExistsError)
    parent = os.path.dirname(output_path)
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(
        prefix='.%s.tmp-' % os.path.basename(output_path), dir=parent
    )
    width = max(4, len(str(frame_count - 1)))
    tensor_manifest = {}
    try:
        for name, value in arrays.items():
            tensor_dir = os.path.join(temporary, name)
            os.makedirs(tensor_dir)
            little_dtype = value.dtype.newbyteorder('<')
            tensor_manifest[name] = {
                'dtype': little_dtype.name,
                'frame_shape': [int(size) for size in value.shape[1:]],
                'files': frame_count,
            }
            for frame_index in range(frame_count):
                frame = np.ascontiguousarray(
                    value[frame_index], dtype=little_dtype
                )
                filename = '%s_%0*d.bin' % (name, width, frame_index)
                with open(os.path.join(tensor_dir, filename), 'wb') as stream:
                    stream.write(frame.tobytes(order='C'))
        report.update({
            'artifact_format': 'per_frame_per_tensor_bin',
            'binary_byte_order': 'little',
            'binary_frame_index_base': 0,
            'binary_file_pattern': '<tensor>/<tensor>_%%0%dd.bin' % width,
            'binary_tensors': tensor_manifest,
        })
        report_path = os.path.join(temporary, 'manifest.json')
        with open(report_path, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write('\n')
        os.replace(temporary, output_path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return os.path.join(output_path, 'manifest.json')



def stream_stft(wave, n_fft, hop, win_len, window, normalized=False):
    """``torch.stft`` framed as the deployed streams frame it (audiolib
    ``AcStft`` at its default, center 0): each hop pushed yields the frame
    over the last ``n_fft`` samples, zeros before the first sample, so the
    first frame's first ``n_fft - hop`` samples are 0. A file's last partial
    hop is completed with zeros and pushed, as a host reading the file does,
    so a file of S samples gives ceil(S / hop) frames. (``torch.stft
    (center=True)`` reflects the start into the first frame and the end into
    the last; ``center=False`` alone starts a hop late.) ``wave`` is
    ``[..., samples]``; the result ``[..., bins, frames]``."""
    import torch

    lead = wave.shape[:-1]
    flat = wave.reshape(-1, wave.shape[-1])
    if not flat.shape[-1]:
        return torch.zeros(*lead, n_fft // 2 + 1, 0, dtype=torch.complex64)
    flat = torch.nn.functional.pad(flat, (n_fft - hop, -flat.shape[-1] % hop))
    spec = torch.stft(flat, n_fft, hop, win_len, window=window, center=False,
                      normalized=normalized, return_complex=True)
    return spec.reshape(*lead, *spec.shape[-2:])

# --------------------------------------------------------------------------
# Where a calibration run's frames go
# --------------------------------------------------------------------------

_PTQ_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))), 'PTQ')
SELECT_HELP = (
    'scan every recording and write only COUNT chosen frames: the frames '
    'holding each tensor\'s minimum and maximum, inside the graph too, then '
    'k-means picks (PTQ/calib_analyze.py). The frames are not all held in '
    'memory, and an analysis page is written beside the output')
DEFAULT_FRAMES = 256        # invocations a capture takes without --select


def add_selection_arguments(parser, jobs=True):
    """``--select`` and ``--report`` of a capture and, when it may split its
    recordings between processes (``jobs``), ``--jobs`` and the hidden
    ``--shard`` those processes run with."""
    import argparse

    parser.add_argument('--select', type=int, default=None, metavar='COUNT',
                        help=SELECT_HELP)
    parser.add_argument('--report', default=None, metavar='HTML',
                        help='with --select: the analysis page (default: '
                             '<output>.report.html)')
    if jobs:
        parser.add_argument('--jobs', type=int, default=1, metavar='N',
                            help='with --select: scan N recordings at a time, '
                                 'each in its own process; the result does '
                                 'not depend on N')
        # Set on the processes a --jobs run starts: K/N:FOLDER.
        parser.add_argument('--shard', default=None, help=argparse.SUPPRESS)


def check_selection_arguments(parser, args, attr='frames', flag='--frames'):
    """Refuse a bad combination; without --select, ``args.<attr>`` falls
    back to ``DEFAULT_FRAMES`` and must be positive."""
    jobs = getattr(args, 'jobs', 1)
    if args.select is None:
        if jobs != 1 or args.report:
            parser.error('--jobs and --report go with --select')
        if getattr(args, attr) is None:
            setattr(args, attr, DEFAULT_FRAMES)
        if getattr(args, attr) <= 0:
            parser.error('%s must be positive' % flag)
        return
    if args.select < 1:
        parser.error('--select must be at least 1')
    if jobs < 1:
        parser.error('--jobs must be at least 1')
    if getattr(args, attr) is not None:
        parser.error('--select scans every recording; leave out %s' % flag)


def _report_path(output, report):
    if report:
        return report
    graph = sibling_onnx_path(output)
    return graph[:-len('.onnx')] + '.report.html'


def _stored(value):
    """A captured tensor as it is written: floating tensors as float32, the
    others (Align-CRUSE's int64 frame counter) exact."""
    return (value.astype(np.float32, copy=False)
            if np.issubdtype(value.dtype, np.floating) else value)


def _without(argv, flags, valued):
    """``argv`` without the options ``flags`` (with their value when the flag
    is in ``valued``). An abbreviation argparse accepted is a prefix of one
    option only, so a prefix of one of ``flags`` is that flag."""
    def flag_of(item):
        name = item.split('=', 1)[0]
        if name in flags:
            return name
        hits = [flag for flag in flags
                if name.startswith('--') and len(name) > 2
                and flag.startswith(name)]
        return hits[0] if len(hits) == 1 else None

    out, skip = [], False
    for item in argv:
        if skip:
            skip = False
            continue
        flag = flag_of(item)
        if flag:
            skip = flag in valued and '=' not in item
            continue
        out.append(item)
    return out


class CalibrationFrames:
    """Where a calibration run's frames go.

    Without ``--select`` every frame is kept and ``write`` stacks and writes
    them as captures always have. With it each frame goes to a
    ``calib_analyze.Scan`` over the graph, which keeps only the frames a
    selection can pick; ``write`` chooses ``--select`` of them, writes them
    and the analysis page. ``facts`` is what the analysis knows of the run
    (``model_family``, ``checkpoint_sha256``, ``state_precision_policy``).
    With ``--jobs N`` the recordings are dealt to N processes running
    ``command`` (this capture, given ``argv`` with the options ``rewrite``
    replaces and the flags ``drop`` removes) on their share (``sources``),
    whose scans are merged in recording order."""

    def __init__(self, args, names, graph, facts=None, command=None,
                 argv=None, rewrite=None, drop=()):
        self.names = list(names)
        self.output = args.output
        self.select = args.select
        self.jobs = getattr(args, 'jobs', 1)
        self.page = args.report
        self.graph = graph
        self.facts = dict(facts or {})
        self.command = command
        self.argv = list(argv or [])
        self.rewrite = dict(rewrite or {})
        self.drop = tuple(drop)
        self.shard = None
        if getattr(args, 'shard', None):
            share, folder = args.shard.split(':', 1)
            k, total = (int(v) for v in share.split('/'))
            self.shard = (k, total, folder)
        self.values = ({name: [] for name in self.names}
                       if self.select is None else None)
        self.scan = None
        self.parts = None
        self.source = 0
        self.produced = 0
        if self.select is not None:
            # The --jobs processes run torch on one thread; every --select run
            # does, so the recorded states are the same whatever --jobs is.
            try:
                import torch
            except ImportError:
                pass
            else:
                torch.set_num_threads(1)

    def sources(self, items):
        """``items`` (the recordings, in order) this process captures. A
        ``--jobs`` run starts its processes here and captures none itself."""
        items = list(items)
        if self.select is not None and self.jobs > 1 and self.shard is None:
            self.parts = self._run(len(items))
            return
        for index, item in enumerate(items):
            if self.shard and index % self.shard[1] != self.shard[0]:
                continue
            self.source = index
            yield item

    def add(self, tensors):
        """One invocation's tensors, in ``names`` order, of the recording
        ``sources`` last gave."""
        if self.values is not None:
            capture_calibration_inputs(self.values, self.names, tensors)
            return
        if len(tensors) != len(self.names):
            raise ValueError('calibration input names/tensors length mismatch')
        # The scan copies each frame into its block and checks it is finite.
        frame = {name: _stored(tensor.detach().cpu().numpy()
                               if hasattr(tensor, 'detach')
                               else np.asarray(tensor))
                 for name, tensor in zip(self.names, tensors)}
        if self.scan is None:
            analyze = self._analyze()
            self.scan = analyze.Scan(
                [(name, value.shape, value.dtype)
                 for name, value in frame.items()],
                self.facts.get('state_precision_policy'), self.facts,
                self.graph, reserve=analyze.fill_reserve(self.select),
                progress=analyze.progress_line)
        self.scan.add(frame, self.source)
        self.produced += 1

    def __len__(self):
        if self.values is not None:
            return len(self.values[self.names[0]])
        if self.parts is not None:
            return int(sum(len(part['src']) for part in self.parts))
        return self.produced

    def full(self, limit):
        """Whether ``limit`` invocations are captured (never with --select:
        every recording is scanned)."""
        return self.values is not None and len(self) >= limit

    def empty(self):
        """Whether the run captured nothing. A --jobs process whose share
        gave no frame is not empty: it hands back nothing, and the merged
        run is checked."""
        return self.shard is None and not len(self)

    @staticmethod
    def _analyze():
        if _PTQ_DIR not in sys.path:
            sys.path.insert(0, _PTQ_DIR)
        import calib_analyze
        return calib_analyze

    def _run(self, count):
        """Run the --jobs processes; their parts, in recording order once
        merged. Their folder is gone when this returns or raises (the kept
        frames stay memory-mapped)."""
        import subprocess

        jobs = min(self.jobs, max(1, count))
        print('scanning %d recordings in %d processes' % (count, jobs),
              flush=True)
        parent = os.path.dirname(os.path.abspath(str(self.output)))
        os.makedirs(parent, exist_ok=True)
        root = tempfile.mkdtemp(prefix='.calib-jobs-', dir=parent)
        # A process writes no artifact; its --output only has to parse.
        valued = ('--jobs', '--report', '--output', '--onnx', '--shard',
                  '--format') + tuple(self.rewrite)
        base = _without(self.argv, valued + self.drop, valued)
        for flag, value in self.rewrite.items():
            base += [flag, str(value)]
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        running = []
        try:
            for k in range(jobs):
                folder = os.path.join(root, 'part%d' % k)
                os.makedirs(folder)
                with open(os.path.join(folder, 'log.txt'), 'w') as log:
                    running.append((folder, subprocess.Popen(
                        self.command + base + [
                            '--output', os.path.join(folder, 'frames.npz'),
                            '--onnx', os.path.join(folder, 'graph.onnx'),
                            '--shard', '%d/%d:%s' % (
                                k, jobs, os.path.join(folder, 'part'))],
                        stdout=log, stderr=subprocess.STDOUT, env=env)))
            failed = []
            for folder, process in running:
                if process.wait():
                    with open(os.path.join(folder, 'log.txt')) as stream:
                        failed.append(stream.read()[-2000:])
            if failed:
                raise RuntimeError('a --jobs process failed:\n' + failed[0])
            analyze = self._analyze()
            return [analyze.load_part(os.path.join(folder, 'part'))
                    for folder, _ in running
                    if os.path.isdir(os.path.join(folder, 'part'))]
        finally:
            for _, process in running:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            shutil.rmtree(root, ignore_errors=True)

    def write(self, output, fmt, make_report, limit=None, source_names=None):
        """Write the artifact and return its report; ``make_report(arrays)``
        gives the report and ``source_names`` names each recording ``sources``
        was given. A --jobs process hands its scan back and exits here."""
        if self.values is not None:
            arrays = {name: _stored(np.stack(values[:limit]))
                      for name, values in self.values.items()}
            report = make_report(arrays)
            write_calibration_artifact(output, arrays, report, fmt)
            return report
        analyze = self._analyze()
        if self.shard is not None:
            if self.scan is not None:
                analyze.save_part(self.scan.part(), self.shard[2])
            raise SystemExit(0)
        parts = self.parts if self.parts is not None else (
            [self.scan.part()] if self.scan is not None else [])
        self.parts = self.scan = None
        if not parts:
            raise RuntimeError('no calibration frames were produced')
        pool = analyze.assemble(
            parts, self.facts, str(output), fmt, graph=self.graph,
            source_names=source_names, source_note='recorded by the capture',
            reserve=analyze.fill_reserve(self.select))
        del parts
        analysis = analyze.analysis_of(pool, self.select, 'asym',
                                       analyze.DEFAULT_BITS)
        keep = np.asarray(analysis['recommend']['keep_frames'],
                          dtype=np.int64)
        recordings = len(pool.sources['starts'])
        # The artifact first: a page that cannot be written loses no scan.
        report = analyze.write_selection(
            pool, keep, str(output), fmt, make_report,
            parent='%d recordings scanned by the capture' % recordings)
        analyze.report_conclusions(analysis, _report_path(output, self.page))
        print('chose %d of %d frames from %d recordings' % (
            keep.size, pool.count, recordings))
        return report
