"""Calibration artifact formats shared by stateless model exporters."""

from __future__ import annotations

import json
import os
import shutil
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


def sibling_onnx_path(output_path, override=None):
    """Where a calibration run writes the graph its tensors bind to.

    Beside the artifact, dropping a ``.npz`` suffix first, so ``calib/foo``
    and ``calib/foo.npz`` both pair with ``calib/foo.onnx``.
    """
    if override:
        return override
    output_path = str(output_path)
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
    captured invocation index. NPZ keeps those arrays intact. BIN creates one
    directory per input and writes zero-padded 0-based
    ``<name>_0000.bin``, ``<name>_0001.bin``, ... beneath it (width grows
    past 10000 frames so a lexicographic listing always equals frame
    order).
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
        report_path = os.path.splitext(output_path)[0] + '.json'
        with open(report_path, 'w', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write('\n')
        return report_path

    output_path = os.path.abspath(output_path)
    if os.path.exists(output_path):
        raise FileExistsError(
            'binary calibration output already exists: %s' % output_path
        )
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
