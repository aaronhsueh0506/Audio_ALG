# DeepFilterNet3 port — upstream alignment

This folder composes its output as a band split: the deep filter runs on the raw
low-band spectrum and the ERB mask owns the bins from `nb_df` up; there is no
alpha blend. `MODEL_VERSION` v5 identifies these checkpoints and rejects
cascade/alpha ones. The rest of the network (encoder/decoder widths,
frequency-major flattening, pathway-add skips, libDF feature normalisation,
AdamW/AMSGrad, MRSL, checkpoint contract, non-finite guards) follows upstream
except where listed below.

This file records which differences from upstream are deliberate, so that
"aligning the port to upstream" does not turn it into something that no longer
reconciles against the released checkpoint. Citations are by symbol, not by line.

⚠ Anything that touches numerics goes through a `FEATURE_VERSION` /
`MODEL_VERSION` bump, batched with other numeric changes. The feature contract is
serialised into every checkpoint (`make_checkpoint_contract`) and gated by
`FEATURE_VERSION` (`require_checkpoint_contract`).

## Deliberate divergences

| Item | Upstream | This port | Why |
|---|---|---|---|
| ERB filterbank | rectangular, non-overlapping, energy-MEAN | triangular, overlapping, energy-SUM | shared with the RNNoise-ERB lineage |
| Lookahead | `conv_lookahead = df_lookahead = 2` | `mask_lookahead = df_lookahead = 1` | one-hop budget: 10.7 ms vs 21.3 ms at hop 512 / 48 kHz; the quality cost is unverified |
| Local-SNR head / `[localsnrloss]` | trained | removed | removes an untrained head and its loss term; the 2,135,484 upstream count minus the head's 513 parameters is this port's 2,134,971 |
| Batch-size ramp | 16/24/32/64 per epoch | fixed `batch_size` | only LR and weight decay vary |
| Grid | `n_fft/hop = 960/480` | `1024/1024/512` | 960 is not radix-2 |
| `df_bins` | 96 bins = 4800 Hz | 96 bins = 4500 Hz | same bin count on a different grid; matching the Hz needs 102–103 bins and a feature-contract change |
| Noise clipping | per noise lane, train split | once on the combined noise (`p_noise_clipping`); `generation_split = validation` disables it | similar exposure, not distribution-identical |
| `noise_only_p` | 14.3% silent-target examples | `0.05` | tuning choice |
| ERB `+1e-10` log floor | applied to a band-energy mean on a `1/960`-normalised spectrum | applied to a weighted sum over a non-normalised triangular bank of a `normalized=True` spectrum | a silent band reaches −1.00 (band 0) / −0.25 (band 31) under libDF's norm-init; a recalibrated init shifts this |
| Model-input STFT | libDF zero-primes the analysis memory and drops the primed frame | `torch.stft` defaults `center=True, pad_mode='reflect'` | train and inference are consistent; steady state differs only by a fixed half-window frame-timing offset, and a streaming implementation reproduces it exactly |
| `spec_norm_eps` | libDF divides by a bare `sqrt(state)` | no eps; `read_feature_config` rejects the key | see below |
| MRSL `factor = 500` | upstream's MRSL sees signals at `1/sqrt(960)` amplitude | the sqrt-Hann round trip is unity gain, so the nominal objective is `960^0.3` = 7.85× upstream's | do not rescale mechanically; that is a tuning experiment with a `grad_clip` interaction |
| `p_resample = 0.1` | time-scale/pitch augmentation | low-SR-source simulation, noisy and target treated identically | same key, different augmentation |
| `seed` | 43 | 42 | all standalone NR trainers share one seed so their train/validation split stays stable (`tests/test_bakeoff_protocol.py::test_seed_defaults_match`) |
| `mask_pf`, `pf_beta` | post-filter shipped off | post-filter on the final complex spectrum after composition, mask `\|spec_e\|/\|spec\|`, beta from config, not gated on `training`; off by default | |
| `df_gru_skip = identity` assert | compares `emb_hidden_dim == df_hidden_dim` | compares `emb_in_dim == df_hidden` | upstream's guard passes and then fails to broadcast |
| `n_erb` divisibility | asserts `% 8` | `validate_signal_config` checks `% 8` (the stride-2 stages would only need `% 4`) | one guard shared by the trainer and `inference.py` |
| Streaming stage gating | the lsnr head lets the runtime skip the ERB/DF stages on clean or lightly-noisy frames | no head | a C/streaming port cannot reproduce that per-frame skipping |

Added on top of upstream: a non-finite guard that halts on the first NaN/inf
(upstream skips and tolerates 50), `error_if_nonfinite=True` on gradient
clipping, a pre-clip gradient-norm CSV trace with finite-spike batch dumps, a
refusal to overwrite a checkpoint with non-finite state, and
`dataset_gen/audit_packed.py`. All are inert at the shipped config.

## Settled questions

**GRU depth is 1 encoder + 2 ERB decoder + 2 DF decoder = 5.** Upstream code:
`num_layers=1` for the encoder `emb_gru`, `emb_num_layers - 1` for the ERB
decoder, `df_num_layers` for the DF decoder; the shipped config has
`emb_num_layers = 3` (a **total**, split 1 + (n−1)) and `df_num_layers = 2`. The
code defaults differ (`deepfilternet2.py` defaults 2 and 3), so quote the shipped
config, not the defaults. Per-site parameter counts match the released
checkpoint (`erb_dec.emb_gru` and `df_dec.df_gru` 805,888 each).

**`df_gru` runs at `linear_groups = 8`, not the config's 16**, because upstream
never passes `linear_groups` at that one site, so `SqueezedGRU_S`'s module
default applies (`df_dec.df_gru.linear_in.0.weight` is `(8, 64, 32)`).

**`spec_norm_eps` is removed.** libDF's `band_unit_norm` divides by a bare
`s.sqrt()`, and that Rust path trained the released model. The state update is
`mu = a*mu + (1-a)*|x|` from a strictly positive init, so `mu >= a^t * mu_0 > 0`;
in float32 the worst case (unbounded exact-zero input) settles at 6.31e-44, so
`x/sqrt(mu)` is always `0/positive`, never `0/0`. The argument assumes IEEE
round-to-nearest with subnormals enabled: under AMP/bf16/fp16, or on a C/DSP/NEON
target with FTZ or denormals-are-zero, `mu` can reach exactly 0. Guard that in the
C port, not by diverging the trainer.

## Do-not-touch list

Each item is a deliberate, verified divergence.

### Grid and latency

1. **`n_fft = 1024`, `win_len = 1024`, `hop_len = 512`.** Everything downstream
   is derived from this grid: 281 frames per 3 s segment, the norm-init
   calibration and the 10.7 ms latency budget.
2. **`mask_lookahead = 1`, `df_lookahead = 1`.** This is the latency budget. Do
   not align the lookahead **mechanism** either: the port's in-conv centred pad is equivalent to
   upstream's feature-shift scheme at equal L, except for the first `kt-1 = 2`
   frames of a stream.
3. **`df_bins = 96`.** Changing it to 102–103 for Hz parity changes the feature
   contract and requires retraining.

### Normalisation

4. **Norm-init literals are libDF's: `erb_norm_init = -60/-90` and
   `spec_norm_init = 0.001/0.0001`.** A recalibrated pair (`-15/-60`,
   `0.06/0.012`) is not used; it is tied to `normalized=True`, the triangular sum
   bank and the corpus level regime, so do not port it to another rate, bank or
   corpus.
5. **`erb_norm_tau_sec` / `spec_norm_tau_sec = 1.0`:** keep τ in seconds and
   re-derive α for the grid; do not hardcode upstream's α.
6. **The triangular, overlapping, energy-SUM forward ERB bank** (`mode=0`).
7. **`erb_inv` needs no row normalisation.** All 513 row sums are exactly 1.0
   (partition of unity), so `gain=1 → bin_gain=1`; upstream's inverse is a plain
   0/1 transpose.
8. **The `[:, df_bins:]` slice of `erb_inv`** instead of a full matmul then
   overwrite is mathematically identical and cheaper.

### Architecture

9. **`df_gru` at `linear_groups = 8`** (above); 16 puts the model 8,192 params
   off the checkpoint.
10. **Transposed convs stay separable/depthwise.** Upstream's shipped
    `convt_depthwise = False` is declared and never read; obeying it builds
    dense transposed convs (12,288 vs 192+4,096 params each) that do not
    reconcile against the checkpoint.
11. **`enc_channels = 64` and `enc_lin_groups = 32`** come from upstream's shipped
    config, not the code defaults (16 and 16).
12. **`convt3` and `conv0_out` are ordinary convs, not transposed**;
    `conv_kernel` (not `convt_kernel`) applies to them.
13. **No lsnr head, no `[localsnrloss]`.** The 2,135,484 − 513 = 2,134,971
    reconciliation depends on its absence.
14. **`validate_signal_config` is the one divisibility guard** (`n_erb % 8`);
    `model.py` carries no equivalent assert. Enforced by
    `test_n_erb_divisibility_follows_upstreams_stricter_rule`.

### Loss

15. **Plain (not sqrt) Hann + `hop = n_fft // 4` + `normalized=True` in the loss
    STFT, while the model path uses sqrt-Hann.** This asymmetry is
    upstream-correct; the loss STFT keeps `center=True`.
16. **`fft_sizes = 256,512,1024,2048`** are absolute sample counts at 48 kHz.
17. **`istft(enhanced)` is compared against the raw `clean` waveform.** Safe
    because the port's round trip is unity-gain to 8e-17.
18. **`factor = 500` / `factor_complex = 500`:** do not rescale by `960^-0.3`.

### Data and process

19. **`p_resample = 0.1` means low-SR-source simulation here**, not upstream's
    time-scale resample.
20. **`FEATURE_VERSION` / `MODEL_VERSION` gating and the checkpoint contract.**
    Changes to `spec_norm_eps`, the `1e-10` ERB floor, `erb_power`'s expression
    or the calibrator's floor go through a version bump, not a silent edit.
