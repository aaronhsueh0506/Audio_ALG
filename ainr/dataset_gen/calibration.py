"""Fitting helpers shared by the per-model normaliser-init calibrators.

WHY THESE ARE SHARED
--------------------
RNNoise-ERB and DeepFilterNet2 both initialise their per-band EMA normalisers
from a two-point ``torch.linspace`` ramp across frequency, and both inherited
those two points from DeepFilterNet's libDF.  libDF aggregates each ERB band as
an energy MEAN over a rectangular, non-overlapping filterbank; both of these
ports use a triangular, overlapping, partition-of-unity bank whose band value
is a weighted SUM, so band energy grows with band width and the imported
constants start the EMA in the wrong place by a band-dependent amount.

The two models need SEPARATE numbers -- different sample rate, FFT size and
band count give different band widths, hence a different offset per band -- but
they need the SAME fitting procedure.  Keeping that procedure in one place is
the same reasoning that put the train/val split in ``loader.py``: the previous
per-project copies of shared logic drifted.
"""

import torch


__all__ = ['fit_ramp', 'robust_quantile']


# torch.quantile refuses inputs beyond roughly this many elements.  A default
# calibration run (thousands of clips x hundreds of frames x hundreds of bins)
# is comfortably past it, so the flat quantiles are taken by sorting instead.
_TORCH_QUANTILE_LIMIT = 16_000_000


def robust_quantile(x, q):
    """``torch.quantile`` without its input-size ceiling."""
    flat = x.reshape(-1)
    if flat.numel() <= _TORCH_QUANTILE_LIMIT:
        return torch.quantile(flat, q).item()
    return flat.sort().values[min(int(q * flat.numel()), flat.numel() - 1)].item()


def fit_ramp(x, axis_name='band', unit='', verbose=True):
    """Endpoints of the least-squares line through the per-column MEAN.

    ``init_lo``/``init_hi`` are the two ends of a ramp ACROSS FREQUENCY, so they
    have to be fitted along that axis.  Pooling every column and every frame
    into one distribution and reporting its median and 5th percentile -- which
    is what these calibrators originally did -- measures how levels vary over
    TIME and LOUDNESS instead, and yields two numbers that describe no ramp at
    all.  On a synthetic case with a known 16 dB frequency ramp and 12 dB of
    frequency-flat level jitter, the pooled version reports a 20.8 dB span and
    misses both endpoints by 8-13 dB; this recovers them to about 0.1 dB.

    The statistic is the MEAN because that is what the EMA converges to:
    ``mean = alpha*mean + (1-alpha)*x`` has expectation E[x], not its median.

    Args:
        x: (frames, columns) real tensor -- per-band dB, or per-bin magnitude.

    Returns:
        (value_at_first_column, value_at_last_column, rms_residual).  The
        residual is the part of the frequency profile a two-parameter ramp
        cannot represent; report it so the reader can judge whether the ramp is
        expressive enough before trusting the two numbers.
    """
    if x.ndim != 2:
        raise ValueError(f'expected (frames, columns), got {tuple(x.shape)}')
    per_col = x.mean(dim=0).double()
    n = per_col.numel()
    if n < 2:
        raise ValueError('need at least two columns to fit a ramp')

    i = torch.arange(n, dtype=torch.float64)
    i_mean, y_mean = i.mean(), per_col.mean()
    slope = ((i - i_mean) * (per_col - y_mean)).sum() / (i - i_mean).pow(2).sum()
    intercept = y_mean - slope * i_mean
    fitted = intercept + slope * i
    residual = (per_col - fitted).pow(2).mean().sqrt().item()

    if verbose:
        print(f"\n--- {axis_name} ramp fit ---")
        print(f"  measured mean spans {per_col[0].item():.4g} .. "
              f"{per_col[-1].item():.4g}{unit}"
              f"  (min {per_col.min().item():.4g}, max {per_col.max().item():.4g})")
        print(f"  fitted ramp          {fitted[0].item():.4g} .. "
              f"{fitted[-1].item():.4g}{unit}")
        print(f"  RMS residual vs the fitted line: {residual:.4g}{unit}"
              f"   <- what the 2-parameter ramp cannot express")
    return fitted[0].item(), fitted[-1].item(), residual
