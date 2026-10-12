"""Python mirror of pipelines/common/se_stage.h.

One borrowed view of a hop's spectra, exchanged by the frequency-domain stages
of the AEC pipelines. The C applications write their combination explicitly --
min of the gains, apply, comfort noise, inverse transform -- and so do the
stage functions of aec_nr_pipeline.py and of the 4-channel pipeline, which
call the same four steps in the same order.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SeSpecFrame:
    """Spectra of one hop (audio scale for ``e``/``y``/``x``, int16^2 power
    scale for ``r2``/``cn``; gains are amplitudes).

    ``e`` is the linear-AEC error E(f) (for four channels, the beamformed
    error); ``y`` the matching capture spectrum; ``x`` the far-end spectrum;
    ``r2`` the residual-echo PSD; ``cn`` the comfort-noise PSD; ``g_res_ctx``
    the AEC's own residual gain (None when its lanes compute none).  Any field
    the producer does not keep is None.
    """

    n_freqs: int
    hop: int
    e: np.ndarray
    y: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    r2: Optional[np.ndarray] = None
    cn: Optional[np.ndarray] = None
    g_res_ctx: Optional[np.ndarray] = None
    far_hop: Optional[np.ndarray] = None
    formed_hop: Optional[np.ndarray] = None
    far_power: float = 0.0
    max_dt: float = 0.0
    max_sat: float = 0.0
    all_converged: bool = False
    clock_drift: bool = False
    delay_changed: bool = False
    delay_locked: bool = False


def frame_from_context(context, hop: int) -> SeSpecFrame:
    """The SeSpecFrame view of one AecResContext (the mono ``process_spec``)."""
    e = context.error_spec
    return SeSpecFrame(
        n_freqs=int(e.shape[0]),
        hop=int(hop),
        e=e,
        y=getattr(context, 'near_spec', None),
        x=getattr(context, 'far_spec', None),
        r2=context.r2,
        cn=context.comfort_noise,
        g_res_ctx=context.res_gain,
        formed_hop=getattr(context, 'formed_output', None),
        far_power=float(getattr(context, 'far_power', 0.0) or 0.0),
        max_dt=float(getattr(context, 'dt_indicator', 0.0) or 0.0),
        max_sat=float(getattr(context, 'saturation_level', 0.0) or 0.0),
        all_converged=bool(getattr(context, 'filter_converged', False)),
    )
