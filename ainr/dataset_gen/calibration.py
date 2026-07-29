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


__all__ = ['describe_bands', 'fit_ramp', 'robust_quantile']


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


def describe_bands(erb_db, band_weight_sums, scale_db):
    """Per-band table that separates the filterbank artefact from the signal.

    The measured level of a band is the signal's level in that frequency range
    PLUS 10*log10(sum of the band's filter weights), because this filterbank
    sums where libDF averages.  That second term is deterministic -- it depends
    only on the band borders -- so printing it lets you subtract it and check
    whether what remains looks like the corpus you actually fed in.  If the
    corrected column does not resemble a plausible spectrum (for speech+noise,
    falling with frequency), the measurement is wrong somewhere upstream of the
    fit and the endpoints should not be trusted.

    ``spread`` is p95-p5 WITHIN each band across time.  That is the quantity
    ``scale_db`` (the /40 divisor) has to cover: after mean subtraction the
    feature is (level - running_mean)/scale_db, so a band whose spread is far
    below scale_db reaches the network compressed into a narrow range.  The
    across-band span is irrelevant to that divisor -- the mean subtraction
    removes it.
    """
    import torch

    q = torch.quantile(erb_db, torch.tensor([0.05, 0.5, 0.95]), dim=0)
    means = erb_db.mean(dim=0)
    fb_db = 10.0 * torch.log10(torch.as_tensor(band_weight_sums).double()).float()

    print(f"\n{'band':>5}{'mean':>9}{'p05':>9}{'p95':>9}{'spread':>9}"
          f"{'fb_off':>9}{'corrected':>11}")
    for b in range(erb_db.shape[1]):
        spread = (q[2, b] - q[0, b]).item()
        print(f"{b:>5}{means[b]:>9.1f}{q[0, b]:>9.1f}{q[2, b]:>9.1f}"
              f"{spread:>9.1f}{fb_db[b]:>9.1f}{means[b] - fb_db[b]:>11.1f}")

    spreads = (q[2] - q[0])
    print(f"\n  fb_off    = 10*log10(band weight sum): the part of `mean` that is "
          f"this filterbank, not the signal")
    print(f"  corrected = mean - fb_off: should look like your corpus's spectrum")
    print(f"  in-band spread over time: min {spreads.min().item():.1f} dB, "
          f"median {spreads.median().item():.1f} dB, max {spreads.max().item():.1f} dB")
    print(f"  vs erb_norm_scale_db = {scale_db:g} dB", end='')
    med = spreads.median().item()
    if med < scale_db * 0.5:
        print(f"  <- median spread is under half the divisor; features reach the "
              f"network in roughly +-{med / (2 * scale_db):.2f}, consider lowering it")
    elif med > scale_db * 1.5:
        print(f"  <- median spread exceeds the divisor; features will regularly "
              f"leave +-0.5, consider raising it")
    else:
        print("  <- reasonable")
    return spreads
