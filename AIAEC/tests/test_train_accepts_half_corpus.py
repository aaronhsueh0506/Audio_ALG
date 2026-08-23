"""A corpus packed with ``--dtype float16`` must reach every model.

``pack_aec_dataset.py --dtype float16`` halves what the corpus costs on disk,
and nothing between the shard and the model widened it: ``__getitem__`` hands
back the stored tensor, ``aec_collate`` stacks it, and each ``forward_batch``
moved it to the device without naming a dtype. Training then stopped in
``torch.stft`` (``aec_features.py``), which has no half kernel on CPU and
falls back to a float64 path: ``expected scalar type Double but found Half``.
Reports of ``expected scalar type Half but found Float`` are the same cause
reaching a different op first. Half is not a precision problem here -- the
transform carries fp16's eleven mantissa bits faithfully -- it is a dtype one.

The fix belongs on the device move, which is where AINR's trainers already put
it: half survives the loader and is widened once, on the way in. So this
asserts the property at ``forward_batch`` rather than at the dataset --
widening earlier would work too, and would carry the wider dtype across the
loader for no gain. (AINR also pins memory and passes ``non_blocking``; AIAEC's
loaders do not, so only the dtype half of that pattern applies here.)

No corpus is rendered here. The batch is fabricated at the packed layout,
because what broke was the dtype of the tensor handed to ``forward_batch``,
not anything about how the samples were produced.
"""
import importlib
import math

import pytest
import torch

from AIAEC.aiaec_common import SignalGrid
from AIAEC.dataset_gen.aec_features import AecGrid
from AIAEC.dataset_gen.packed_aec_dataset import PACKED_STEM_ORDER
from AIAEC.Align_CRUSE.model import AlignCRUSE
from AIAEC.Align_ULCNet.model import AlignULCNet
from AIAEC.CAGCRN.model import CAGCRN
from AIAEC.DeepVQE_S.model import DeepVQES

# The models are built on SignalGrid and the views on AecGrid -- the same
# geometry named twice, exactly as test_export_tools.py pairs them.
GRID = SignalGrid(16000, 512, 512, 256)
AEC_GRID = AecGrid(16000, 512, 512, 256)

# One frame of hop-aligned audio is not enough for the models that keep a
# lookahead; a second of it is, and still runs in well under a second on CPU.
T = 16000

# Named here rather than read off the train module: none of them exposes its
# class, and test_export_tools.py already pairs them this way.
MODELS = (
    ('Align_ULCNet', AlignULCNet),
    ('DeepVQE_S', DeepVQES),
    ('Align_CRUSE', AlignCRUSE),
    ('CAGCRN', CAGCRN),
)


def _batch(dtype):
    torch.manual_seed(11)
    return torch.randn(2, len(PACKED_STEM_ORDER), T).clamp(-1.0, 1.0).to(dtype)


# The views hand the models spectra, so the single-precision family here is
# float32 AND complex64 -- complex64 is a float32 pair. What must not survive
# is anything half: a half corpus that reached the model unwidened would show
# up as float16, or as the complex32 an STFT of it produces.
HALF = (torch.float16, torch.bfloat16, torch.complex32)

# Derived, not tuned to whatever the first run happened to produce. Packing to
# float16 quantizes samples in [-1, 1] to a step of 2**-11, and an n_fft-point
# STFT sums that many of those errors, which for independent errors grows as
# sqrt(n_fft). The strict worst case -- every error aligned -- is n_fft * 2**-11
# = 0.25, too loose to catch anything. Measured on this fixed seed: 5.6e-3
# against the 1.1e-2 this gives, so roughly a factor of two of headroom.
SPECTRUM_ATOL = 2.0 ** -11 * math.sqrt(GRID.n_fft)


@pytest.mark.parametrize('name,model_cls', MODELS)
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_forward_batch_accepts_the_packed_dtype(name, model_cls, dtype):
    """Both storage dtypes must reach the model at single precision.

    float32 is exercised alongside float16 on purpose: it is what proves the
    cast is a widening rather than something that only happens to work for the
    dtype the bug was about.
    """
    train = importlib.import_module(f'AIAEC.{name}.train')
    model = model_cls(GRID).eval()
    with torch.no_grad():
        output, spectral = train.forward_batch(
            model, _batch(dtype), AEC_GRID, torch.device('cpu'))

    offenders = {
        key: tensor.dtype
        for key, tensor in spectral.inputs.items()
        if tensor.dtype in HALF
    }
    assert not offenders, (
        f"{name}: forward_batch handed the model half-precision inputs "
        f"{offenders} for a {dtype} corpus")
    assert output is not None, f"{name}: forward_batch produced no output"


@pytest.mark.parametrize('name,model_cls', MODELS)
def test_half_and_float_corpora_agree(name, model_cls):
    """The widening must not change what the model sees beyond half's precision.

    float16 -> float32 is exact, so the only difference between the two runs
    is the rounding the pack itself applied. Asserting they agree to that
    tolerance is what distinguishes "the cast works" from "the cast silently
    replaced the batch with something else".
    """
    train = importlib.import_module(f'AIAEC.{name}.train')
    model = model_cls(GRID).eval()
    device = torch.device('cpu')

    reference = _batch(torch.float32)
    with torch.no_grad():
        _, from_float = train.forward_batch(model, reference, AEC_GRID, device)
        _, from_half = train.forward_batch(
            model, reference.to(torch.float16), AEC_GRID, device)

    for key, expected in from_float.inputs.items():
        torch.testing.assert_close(
            from_half.inputs[key], expected, rtol=0.0, atol=SPECTRUM_ATOL,
            msg=lambda m, key=key: f"{name}: input {key!r} diverged: {m}")
