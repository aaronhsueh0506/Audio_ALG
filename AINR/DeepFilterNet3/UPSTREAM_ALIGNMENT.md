# DeepFilterNet3 port vs upstream DeepFilterNet3 — alignment record

**Audited 2026-07-30.** 34 independent agents across five areas (feature
normalisation, GRU/embedding architecture, conv stack, loss, training schedule),
every claimed divergence then handed to an adversarial verifier whose default was
to refute. Only claims that survived verification appear below, at their
**post-verification** severity.

This file exists because "align the port to upstream" is not a mechanical
operation here. Roughly a third of the differences are deliberate, and several of
the port's own comments assert alignment where there is none — so the next person
to compare the two needs a record of which is which, or they will "fix" the port
into something that no longer reconciles against the released checkpoint.

Severity:

| | meaning |
|---|---|
| **S1** | can kill a whole training run |
| **S2** | changes numerics or the training distribution |
| **S3** | latent — bites only if a currently-shipped knob changes |
| **S4** | documentation only |

⚠ Anything marked **align** that touches numerics must go through a
`FEATURE_VERSION` / `MODEL_VERSION` bump, batched with the other numeric changes
— not landed as a silent edit. The feature contract is serialised into every
checkpoint (`make_checkpoint_contract`) and gated by `FEATURE_VERSION`
(`require_checkpoint_contract`), so a numeric change invalidates existing
checkpoints — as the v5 bump did.

⚠ **Citations are by SYMBOL, not by line.** An earlier revision cited
`file:line` throughout and every one of ~25 citations was invalidated by a single
day's edits. If a claim below still carries a line number, distrust it.

---

## 0a. CURRENT STATE — read this before §1

⚠ **§1 is the audit record, not a to-do list.** Its "Action" column says what the
audit recommended; most of it has since been applied. This section is what the code
actually does now.

The port is aligned to upstream **everywhere except four operator decisions and two
target constraints**.

### Deliberately NOT aligned — operator decisions, do not "fix"

| | what | why |
|---|---|---|
| **ERB filterbank** | triangular, overlapping, energy-**SUM** (upstream: rectangular, non-overlapping, energy-MEAN) | confirmed correct for this project; shared with the RNNoise-ERB / `bandERB.ipynb` lineage |
| **lookahead 1/1** | upstream ships 2/2 in all three releases | latency: 10.7 ms vs 21.3 ms at hop 512 / 48 kHz |
| **no lsnr head, no `[localsnrloss]`** | upstream trains both | removes an untrained head and its loss term |
| **no batch-size ramp** | upstream ramps 16/24/32/64 per epoch | fixed batch; only LR and WD vary |

### Fixed by the embedded target, not open for alignment

| | what | why |
|---|---|---|
| **`n_fft/win/hop = 1024/1024/512`** | upstream 960/480 | 960 is not radix-2 |
| **`df_bins = 96`** | same bin *count* as upstream | the Hz crossover (4500 vs 4800) differs only because the grid does; matching it needs 102–103 bins and a feature-contract change |

### Aligned to upstream in this pass

`erb`/`spec` normaliser inits → libDF literals · `spec_norm_eps` **removed** (libDF
divides by a bare sqrt) · `erb_power` → `re*re + im*im` (trainer **and**
calibrator) · `amsgrad` → true · `epochs` → 120 · `early_stop_patience` → 25 ·
`n_erb % 8` · `read_loss_config` rejects `factor==0` with
`factor_complex>0` · `SqueezedGRU_S` skip takes the raw input · `enc_concat` widens
only the embedding GRU's INPUT, leaving one immutable output bus (`emb_dim`) ·
`max(1, emb_num_layers-1)` clamp removed (fails loudly, as upstream does) ·
`mask_pf` + `pf_beta` implement **DFN3's** post-filter — applied to the final
complex spectrum after both stages, mask derived as `|spec_e|/|spec|`, beta
config-driven, **not** gated on `training` (DFN2's was an ERB-mask post-filter with
beta hardcoded; the two are not interchangeable) · libDF's `wnorm` analysis scale
(`analysis_scale`, `erb_band_db`) · the lookahead relation enforced once in the
constructor · `validate_signal_config` shared by the trainer and `denoise.py`.

### One alignment that was tried and REJECTED

**`--seed` stays 42, not upstream's 43.** A seed is arbitrary — matching upstream's
reproduces nothing of upstream's run (different corpus, data order and RNG), so the
alignment buys zero. All standalone NR trainers in `AINR/` sharing one seed is
load-bearing because it keeps their train/validation comparison split stable.
`tests/test_bakeoff_protocol.py::test_seed_defaults_match` caught the change and is
what keeps it reverted.

⚠ The general lesson: *align mechanisms and values that carry meaning; do not align
arbitrary constants at the cost of a real invariant.*

### Measured as NOT a one-line alignment

**`center=False`.** `torch.istft` refuses it outright at win=1024/hop=512 —
*"window overlap add min: 1"* — because the first and last frames lack OLA
coverage. Aligning means replicating libDF's zero-prime-then-drop framing, not
flipping a flag. Left at `center=True`, documented at the call site.

### Deliberate additions (not upstream, not divergences to undo)

A non-finite guard that **halts** on the first NaN/inf (upstream skips and tolerates
50), `error_if_nonfinite=True`, a pre-clip gradient-norm CSV trace with
finite-spike batch dumps, a refusal to overwrite a checkpoint with non-finite
state, and `dataset_gen/audit_packed.py`. All inert at the shipped config.

---

## 0. Checked and WRONG — do not re-raise

| Claim | Why it is wrong |
|---|---|
| "The port is missing upstream's `nb_erb % 8` divisibility assert." | The port **does** enforce it, at the config-read boundary rather than in the module: `validate_signal_config`. It checks **`% 8`**, upstream's own stricter rule — the two stride-2 stages would only need `% 4`, and 20 and 36 pass that while upstream rejects them. Enforced by `test_n_erb_divisibility_follows_upstreams_stricter_rule`. |
| "The parameter-count reconciliation comment omits `mask.erb_inv_fb`; the 2,135,484 anchor is off by 15,392." | 2,135,484 **is** upstream's own number under upstream's own convention (`sum(p.numel() for p in model.parameters() if p.requires_grad)`, which excludes buffers — `erb_fb` and `mask.erb_inv_fb` are both buffers). The port's arithmetic is correct as written. |

---

## 1. Surviving divergences, ranked

Within a tier, cheapest fix first.

### S1

| # | Item | Upstream | Port | Consequence | Action |
|---|---|---|---|---|---|
| **1** | **Non-finite loss/grad guard** | `clip_grad_norm_(..., 1.0, error_if_nonfinite=True)` inside try/except → dump the offending batch, `continue`, tolerate up to `MAX_NANS = 50` (`df/train.py:43,380-419`) | bare `clip_grad_norm_(model.parameters(), grad_clip)` (`train.py:779`); no `isnan`/`isfinite` anywhere in the file | One inf/NaN gradient → `clip_coef = 1.0/(inf+1e-6) = 0` → `0*inf = NaN` → the next `opt.step()` writes NaN into the weights **and into Adam's `exp_avg`/`exp_avg_sq`**. Unrecoverable: no diagnostic, no batch dump, and the run continues burning GPU hours producing NaN. | **align** (~10 lines) |

### S2 — numerics / training distribution

| # | Item | Upstream | Port | Consequence | Action |
|---|---|---|---|---|---|
| 2 | `amsgrad` | `amsgrad=True`, hardcoded on the adamw branch (`df/train.py:495`) | PyTorch default `False`, undocumented | No max-clamp on the second-moment denominator (β₂ = 0.999 ≈ 1000-step memory), so a gradient spike is followed by oversized steps until `v` recovers. Different trajectory from step 1 onward. | **align** (1 kwarg) |
| 3 | Noises mixed per example | `n_noises = rng.uniform(2,6)` → always 2–5, unconditional (`dataset.rs:1253-1254`) | `max_noise_mix = 1` (`dataset.py:896`) | Narrower, more stationary noise distribution than upstream trained on. ⚠ The port's justification cites `p_interfer_sp`, which controls **interfering speakers**, not noise mixing — the rationale is a misreading. | **align** |
| 4 | Noise clipping, train split | `RandClipping::default_with_prob(0.1).with_c(0.01..0.5)` attached to the **noise** chain whenever `Split::Train` (`dataset.rs:705-709`). ⚠ `p_clipping = 0.0` does **not** disable this — it bypasses config entirely | no noise-clipping path; `p_clipping = 0.0` was read as "DFN3 does none" | 10% of upstream's training noise is hard-clipped before mixing (broadband harmonics, target unchanged). The port has no robustness to clipped/overloaded noise. | **align** |
| 5 | `RandRemoveDc` (p=0.25) + `RandLFilt` (p=0.25) | both active on the speech **and** noise chains (`dataset.rs:644-651`) | neither exists | Two augmentations that **are** on in the shipped run are absent. The port's substitute (`p_biquad`) is one upstream turns off — so the spectral-augmentation distribution is unrelated to upstream's, not merely scaled. | **align** |
| 6 | `p_biquad` | realised **0.0** (`DF_P_BIQUAD` never set; `dataset.rs:652,697`) | `0.5` on speech and independently on noise, 3 filters, ±15 dB, labelled `(ref: DeepFilterNet)` | Up to ~45 dB of extra per-example spectral tilt that upstream does not apply, widening the range the ERB/spec EMA normalisers must absorb — which interacts directly with the recalibrated norm-init endpoints (do-not-touch item 4). The `(ref: ...)` attribution is **wrong**. | **document, or set 0.0** |
| 7 | `noise_only_p` | `snr = -100` sentinel, uniform 1-of-7 → **14.3%** silent-target examples (`dataset.rs:1218,1227-1228`) | `0.05`, documented | 2.9× fewer full-suppression examples. Also feeds §3 — but ⚠ **not** because a zero target is itself a gradient hazard (it is not; `_SafeAngle`'s gain at `x = 0` is exactly **0**). A silent target drives the prediction magnitude down **through** the peak-gain band at `\|x\| = 1e-5`, which is where the 1e5 amplification lives. | keep (tuning) / consider 0.14 |
| 8 | `+1e-10` ERB log floor | applied to a band-energy **mean** (`k = 1/band_size`) on a `1/960`-normalised spectrum (`lib.rs:206-231,288`) → feature bounded at **−1.00** | same literal applied to a weighted **sum** over a non-normalised triangular bank of a `normalized=True` spectrum (`train.py:476-477`) | Measured composite scale shift **+32.83 dB (band 0) … +48.40 dB (band 31), mean +38.97 dB**. The −100 dB floor therefore sits 102 dB below nominal band level instead of 67 dB, so on a numerically-silent band the feature reaches **−2.13** where upstream is hard-bounded at −1.00. The +39 dB shift was documented for the norm-init but nobody noted it also changes what `1e-10` *means*. Only bites when the floor is actually reached: top ERB bands of anything resampled from ≤44.1 kHz, and digital-silence segments. | document; re-derive only inside a `FEATURE_VERSION` bump |

### S3 — latent

| # | Item | Upstream | Port | Consequence | Action |
|---|---|---|---|---|---|
| 9 | Calibrator vs trainer ERB expression | one expression for both (`lib.rs:229`); no separate calibration script exists | `clamp_min(1e-16)` + `re²+im²` (`calibrate_norm_init.py:119-121`) vs the trainer's `(erb_power+1e-10).log10()*10` + `abs().pow(2)` (`train.py:476-477`) | The calibrator observes band levels 0–60 dB below anything the trainer can render; those frames drag the least-squares ramp's HF endpoint down, biasing `erb_norm_init_hi_db` low. | align (make the calibrator use the trainer's expression) |
| 10 | Model-input STFT priming | libDF zero-primes `analysis_mem` (`lib.rs:118-119`) and the offline front-end **drops** the primed frame (`transforms.rs:162,187`) | `torch.stft(...)` defaults `center=True, pad_mode='reflect'` — never passed, never commented (`train.py:746-749`, `:799-802`, `denoise.py:117-120`) | Clip onset contains a **time-reversed copy of the signal** where libDF has silence, and that lands inside the init-dominated EMA transient (3τ = 273 frames vs 281 frames per 3 s segment). Steady state is only a fixed half-window frame-timing offset, which a streaming implementation reproduces exactly. Train and inference are mutually consistent, so training converges. | document the offset + the onset difference; consider zero-priming |
| 11 | `conv_lookahead >= df_lookahead` invariant | asserted (`deepfilternet3.py:357-358`) | validated **independently** at three sites (`model.py:636-644`, `train.py:597-601`, `denoise.py:51-54`); `config.ini:26` states the rule in prose, nothing enforces it | Configs upstream refuses to build are accepted. Latent only — shipped 1/1 satisfies it. | align (2 lines) |
| 12 | `SqueezedGRU_S` skip target | `x = x + self.gru_skip(input)` — the **raw** input (`modules.py:732-738`) | `y = y + self.gru_skip(x)` where `x` was already rebound by `linear_in` (`model.py:300-306`) | Dormant: all three sites pass `gru_skip_op=None` and the released checkpoint has zero `gru_skip` keys. If ever enabled it **crashes loudly** (256 vs 512), not silently. Still a wrong reusable building block whose docstring reads as correct. | align (1 line) |
| 13 | `enc_concat=True` bus width | separate `emb_in_dim` / `emb_out_dim`; only the **input** doubles (`deepfilternet3.py:125-136,152-155`) | one attribute, so `output_size` doubles too (`model.py:336,348-350,356-357`) | Dormant at shipped `enc_concat = false`; setting it true emits a 1024-wide bus the ERB decoder rejects. `enc_concat` **is** an exposed knob. | align |

### S4 — documentation

| # | Item | Reality | What the port says | Action |
|---|---|---|---|---|
| 14 | `erb_power` | `re*re + im*im` (`lib.rs:290`) | `abs().pow(2)` (`train.py:476`) — a sqrt-then-square round trip: extra rounding plus an unnecessary hypot per bin per frame, and it disagrees in the last bits with the port's **own** calibrator, which uses the direct form | align (bit-changing → version bump) |
| 15 | `spec_norm_eps = 1e-12` | libDF: bare `*x /= s.sqrt()` (`lib.rs:253-259`). But upstream's **own PyTorch reference** of the same function guards: `clamp_min(1e-14)` on the **magnitude before the EMA** (`df/modules.py:287,296`, self-test only) | eps added to the **state** before sqrt (`train.py:398-402,427`), described as a locally-invented guard | keep + document — see §2b |
| 16 | Batch-size ramp | per-epoch `min(b, bs)` → realised **16/24/32/64** (epochs 20/40 are no-ops); LR **not** rescaled | fixed batch, no scheduling | keep (or add) — but see §3, this is *not* the epoch-10 trigger |
| 17 | `epochs = 100` vs upstream `max_epochs = 120` | compresses the LR and WD horizons 1.2× | — | keep. At epoch 10: LR 9.872e-4 vs 9.912e-4 (nil); WD 2.45e-4 vs 8.6e-5 (2.8×, but both ≪ the 0.01 endpoint) |
| 18 | MRSL `factor = 500` | Upstream's MRSL sees signals at `1/sqrt(960)` = 0.0323× true amplitude (verified to 1e-16). The port's matched sqrt-Hann round trip is **unity gain** (1.0 to 8e-17) | same `factor` ⇒ nominal objective `960^0.3` = **7.85×** upstream's. ⚠ The *effect* is unestablished: AdamW is scale-invariant in the relevant regime and `clip_grad_norm_` rescales without changing direction | document only. Do **not** rescale as a "fix" — that is a tuning experiment with a `grad_clip` interaction |
| 19 | `df_bins = 96` at n_fft 1024 | upstream 96 bins @ 50.0 Hz = 4800 Hz; port 96 bins @ 46.875 Hz = **4500 Hz** (exactly 300.0 Hz lower, 6 bins) | `config.ini:12` states the port's own coverage but not that it diverges. A reader cannot tell whether 96 was re-derived or copied | keep + document |
| 20 | `mask_lookahead / df_lookahead = 1/1` | upstream ships `conv_lookahead = 2`, `df_lookahead = 2` | 1/1, with a 10-line rationale. Mask sees `t+1` not `t+2`; DF support `[t-3..t+1]` instead of `[t-2..t+2]` (same 5 taps). ~10.7 ms algorithmic latency vs upstream's ~20 ms — the stated goal. ⚠ Quality cost **unverified**: no DFN3 ablation at lookahead 1 exists | keep (by design) |
| 21 | `alpha = 0.989` | 0.989 appears in **no** upstream config. Upstream derives α from `norm_tau` with a rounding loop (`utils.py:111-128`) | `config.ini:74-75` asserts identity with "the original-DFN rounded alpha" — a false *aligned* marker, which is exactly what stops a reviewer looking. The mechanism itself is correct (`train.py:175-183`). Side effect: effective τ = 0.9644 s vs 0.9950 s (3% shorter), because 3-decimal rounding is coarser at hop 512 | fix comment |
| 22 | `causal_ema_db_norm` docstring | code default is `(-15,-60)` (`train.py:363`) | the docstring three lines later still says `MEAN_NORM_INIT = [-60,-90]` and that `normalized=True` is what calibrates it (`train.py:368-369`). A reader trusting it concludes the port is upstream-aligned exactly where it deliberately is not. The unit-norm path's docstring gets this right (`train.py:403-405`) | fix comment |
| 23 | `deep_filter_apply` docstring + default | shipped `df_lookahead = 1` (`config.ini:17`); DFN3 filters the **unmasked** spectrum | docstring says "the deployed configuration uses `df_lookahead=0` … four history frames" and "masked spectra"; signature default is `df_lookahead=0` (`model.py:553-556,621`). Wrong twice, and actively misleading for the C/streaming port that is its stated audience. At 1/1 the ring buffer needs 3 history + current + 1 future and a one-frame output delay | fix comment + default |
| 24 | `GroupedLinearEinsum` docstring | ground truth: `df_gru` runs at **G=8** (`modules.py:712` default, never overridden at `deepfilternet3.py:297-304`) | `model.py:243-244` claims "G=16 everywhere except the encoder's DF projection (G=32)", contradicting the warning 250 lines later at `model.py:493-496`. ⚠ Wrong in precisely the direction that makes a reviewer "fix" `df_gru` to 16 and land 8,192 params off the checkpoint | fix comment |
| 25 | `max(1, emb_num_layers - 1)` | upstream: bare `p.emb_num_layers - 1` (fails loudly at 1) | clamped, uncommented (`model.py:435`). At `emb_num_layers = 1` the port silently builds 1+1 = 2 layers while the config says 1, breaking the documented total-depth invariant. Nil at shipped 3 | document |
| 26 | `df_gru_skip='identity'` assert | upstream compares `emb_hidden_dim == df_hidden_dim` (256 == 256 — passes, then fails to broadcast) | port compares `emb_in_dim == df_hidden` (512 vs 256 — fails) (`model.py:506-508`). The port's guard is arguably **correct** and upstream's is arguably wrong, but with no comment a reader cannot tell a fix from a slip | document as a deliberate upstream-bug fix |
| 27 | `min_bins_per_band`; `n_erb % 4` | upstream: `min_nb_erb_freqs = 2` is a first-class knob, and it asserts `nb_erb % 8` | port: `min_bins_per_band=2` reachable only by editing `model.py:40/114`; `train.py:596` checks `% 4`, which admits values (20, 36) upstream rejects. No behavioural difference at shipped 32/2, but this is the one filterbank parameter that can drift outside the checkpoint contract | document |
| 28 | `pf_beta` | `mask_pf = False` shipped upstream; the post-filter exists but is off | `config.ini:113-114 pf_beta = 0.0` is consumed by **nothing**. Setting `pf_beta = 0.02` silently does nothing — the kind of dead knob later mistaken for a lever | document or delete the key |
| 29 | `convt_depthwise = False` | declared in upstream's shipped config and **never read** (`deepfilternet3.py:34-39`); DFN3 hardcodes `separable=True`, and the checkpoint proves the transposed convs are depthwise | the port has no such key — which is correct. Obeying that shipped key would build dense transposed convs (12,288 vs 192+4,096 params each) and break checkpoint reconciliation | add a one-line "do not restore" note |
| 30 | lsnr head removal → lost runtime stage gating | `tract.rs:658-672` skips the ERB stage on clean speech and the whole DF stage on lightly-noisy frames, gated on lsnr | `model.py:361-364` explains the *training* rationale only. No training or offline-denoise effect, but a C/streaming port cannot reproduce upstream's per-frame DF skipping (compute saving + "don't touch clean speech" behaviour) | document the inference consequence |
| 31 | `p_resample = 0.1` | upstream: random resample = time-scale/pitch augmentation on speech and noise (`augmentations.rs:458-465`) | port: low-SR-source simulation, noisy and target treated identically. **Same key, same probability, different augmentation.** Net: the port has no time-scale/pitch aug; upstream has no bandwidth-limiting aug. Documented in the port's own terms | keep + one clarifying line |
| 32 | `read_loss_config` validation | upstream disables MRSL entirely when `factor == 0`, even if `factor_complex > 0` (`loss.py:718-722`) | port rejects odd/empty `fft_sizes`, γ range, and `factor + factor_complex == 0` (`train.py:162-170`). Diverges only at `factor=0, factor_complex=500`, which the recipe never uses | keep — defensive hardening |
| 33 | `seed 42` vs 43; `early_stop_patience 20` vs 25 | upstream 43 / 25 | cosmetic. ⚠ One real consequence: with no NaN guard, once the loss is NaN `val_loss < best` is never true, so early stopping fires 20 epochs later — **a NaN at epoch 13 surfaces as a stop near epoch 33** | no action |

### Confirmed clean — no entry needed

The whole conv/convT stack (shape-for-shape against the released checkpoint), the
deep-filter operator (max abs diff **5e-7** vs upstream `MF.DF` at lookahead
0/1/2), the band-split composition (DF on the raw spectrum, ERB owns bins ≥
`nb_df`), `MultiResSpecLoss` body / `_LossStft` / `_SafeAngle`, the LR and
weight-decay schedule shapes, and every `[optim]` scalar.

---

## 2. Two questions settled

### (a) GRU depth — 1 encoder + 2 ERB decoder + 2 DF decoder = 5

**Correct. Confirmed three independent ways.**

1. Upstream code: `deepfilternet3.py:156` `num_layers=1` (encoder `emb_gru`),
   `:216` `num_layers=p.emb_num_layers - 1` (ERB decoder), `:288/:300`
   `df_n_layers = p.df_num_layers` (DF decoder).
2. Shipped config: `emb_num_layers = 3` (a **total**, split 1 + (n−1)),
   `df_num_layers = 2` → 1 + 2 + 2 = 5.
3. The released checkpoint's only GRU key sets: `enc.emb_gru.gru.*_l0`,
   `erb_dec.emb_gru.gru.*_l{0,1}`, `df_dec.df_gru.gru.*_l{0,1}` = 1 + 2 + 2.

The port builds exactly 5 (verified by instantiating with the real `config.ini`),
and per-site parameter counts are **byte-identical** to the checkpoint —
`enc.emb_gru` 411,136 / `erb_dec.emb_gru` 805,888 / `df_dec.df_gru` 805,888 — so
**width** is confirmed, not only depth.

Released DFN1 and DFN2 ship the same `emb_num_layers = 3` / `df_num_layers = 2`,
so the 1/2/2 split is identical across all three upstream releases. ⚠ The *code
defaults* differ (`deepfilternet2.py` defaults 2 and 3 → 1+1+3); quoting them
instead of the shipped config is the standard way to get this wrong.

Two attached traps, both of which the port already handles:

- `emb_num_layers` is a **total**, not a per-site count (`config.ini:35-38`).
- `df_gru` runs at `linear_groups = 8`, not the config's 16, because
  `deepfilternet3.py:297-304` never passes `linear_groups` at that one site, so
  `SqueezedGRU_S`'s module default applies. The checkpoint confirms it:
  `df_dec.df_gru.linear_in.0.weight` is `(8, 64, 32)`. The same omission exists
  in `deepfilternet2.py:328-334`, so it is a long-standing upstream pattern, not
  a DFN3 typo. **Do not touch** — see do-not-touch item 9.

Nothing in this path depends on the 960/480 grid (bus width is
`conv_ch*nb_erb//4`, the DF projection is `conv_ch*nb_df//2`, both driven by the
unchanged `nb_erb=32` / `nb_df=96`; every group divisibility still holds), and
nothing depends on the removed lsnr head (a leaf off `emb`, exactly the 513
params the reconciliation subtracts).

### (b) `spec_norm_eps` — RESOLVED: removed

libDF's `band_unit_norm` divides by a bare `s.sqrt()` (`lib.rs:253-259`) and that
Rust path is what trained the released model, so the port's added
`spec_norm_eps = 1e-12` is gone — from the config, the reader, the validation, the
checkpoint contract, the function signature and the division.

Why removal is safe in this training path, measured rather than argued: the state
update is `mu = a*mu + (1-a)*|x|` from a strictly positive init, so
`mu >= a^t * mu_0 > 0`; and in float32 the worst case (unbounded exact-zero input)
settles at **6.31e-44** — 45x the smallest subnormal — first reached after **8358
frames (89 s)**, so `x/sqrt(mu)` is always `0/positive`, never `0/0`. Over a 3 s
segment `mu` only falls to **4.47e-05**, 38.9 orders above that floor.

⚠ **The one case where the argument does not hold**, and the reason a guard exists
at all: it assumes IEEE round-to-nearest **with subnormals enabled**. Under
AMP/bf16/fp16, or on a C/DSP/NEON target with FTZ or denormals-are-zero, `mu` can
reach exactly 0 and `x/sqrt(0)` is NaN. That is a **deployment** concern — guard it
in the C port, not by diverging the trainer.

⚠ Upstream's own PyTorch reference of this function *does* guard
(`df/modules.py`, `clamp_min(1e-14)` on the magnitude before the EMA), but it is
instantiated only in a self-test, so it is not what trained the model.

`read_feature_config` now **rejects** the key outright, so a stale copy in an old
config cannot read as though the guard were honoured.

## 3. The NaN (train loss rising from epoch 10, NaN at 13, fixed batch 32)

⚠ **EVIDENCE CAVEAT — read before using anything in this section.** The
"epoch 10 rise, epoch 13 NaN" account is a **verbal report with no corroborating
artifact in this workspace**. The only checkpoints present are `epoch 100 /
global_step 1100 / best_val_loss 79.18 / amsgrad=False`, with **no non-finite
value in either the weights or the optimizer state** — 11 steps per epoch, i.e.
throwaway smoke-run output, not the failing run. The failing run's log and its
last-good checkpoint live on the training machine. The `_SafeAngle` gain figure
was also wrong on the first pass (1e10 → **1e5**, corrected in row 1 and measured
against autograd in `tests/test_dfn3_contract.py`).

### ⚠ THE DECISIVE CONSTRAINT: this is a regression, not a latent pathology

Two facts, from the user, change the shape of the whole problem:

1. **The previous configuration trained fine.**
2. **The dataset is not being regenerated** — the corpus is byte-identical
   between the working run and the failing run.

`37db9df` is HEAD and is verified to change **only** the two feature-normaliser
init pairs (plus their fallbacks and a `FEATURE_VERSION` v2→v3 bump, which forces
a fresh run — so the failing run started here):

| key | v2 — trained fine | v3 — diverges |
|---|---|---|
| `erb_norm_init_lo/hi_db` | −60.0 / −90.0 | **−15 / −60** |
| `spec_norm_init_lo/hi` | 0.001 / 0.0001 | **0.06 / 0.012** |

**Therefore: any candidate that was identical in both runs cannot be the
trigger.** That eliminates most of the list below as *causes*:

| candidate | status under the constraint |
|---|---|
| Corpus properties (rows 3–7: `p_biquad`, `noise_only_p`, `max_noise_mix`, noise clipping) | **Cannot be the trigger** — identical in both runs. Demoted from *cause* to *pre-existing condition the change may have exposed*. |
| `amsgrad=False` (row 2) | **Cannot be the trigger** — identical in both runs. Remains an amplifier. |
| Missing non-finite guard (row 1) | **Never was the cause** — it is the permanence mechanism. |
| Batch-size ramp (row 16) | **Dead.** Already ruled out; the port never ran the ramp. |
| **The norm-init recalibration** | **The only difference between a run that worked and a run that does not.** |

⚠ **And row 8 is not an independent finding — it is the same regression.** The
`+1e-10` ERB log floor pins a numerically-silent band at −100 dB. Normalised:

```
v2 init:  band 0 = -1.000    band 31 = -0.250     (init span 30 dB)
v3 init:  band 0 = -2.125    band 31 = -1.000     (init span 45 dB)
```

The −2.13 excursion the audit reported as a standalone latent issue **exists only
under the v3 init**. Under v2 the same floor produced exactly **−1.00** — upstream's
own bound. So the init change did not introduce a new bug so much as break a
coincidence that was holding the floor harmless. That distinction decides the fix:
**reverting the init and fixing the floor are different repairs, and one may make
the other unnecessary.**

Complex path, same comparison — `x/sqrt(mu)` amplification at t=0:

```
v2 init:  bin 0 = 31.62x    bin 95 = 100.00x
v3 init:  bin 0 =  4.08x    bin 95 =   9.13x      (7.8-11.0x smaller)
```

⚠ The ranking that follows was written **before** the regression constraint was
known, so it over-weights corpus-side causes. Read it as a catalogue of
mechanisms, not as a ranking of triggers.

Two things that get conflated:

- **The terminal mechanism** — why the NaN is permanent. **Settled** (row 1).
- **The trigger** — why the loss started rising at epoch 10. **Not settled** by
  the evidence in hand.

A rising **train** loss over ~3 epochs followed by NaN is a divergence signature,
not overfitting, and it rules out "one bad batch instantly NaN'd everything" as
the whole story: something degraded progressively first.

| Rank | Candidate | Explains the rise? | Explains 10 → 13? | Verdict |
|---|---|---|---|---|
| **1** | **Rare pathological batch × 1e5 amplification.** `_SafeAngle.backward` divides by `clamp_min(1e-10)`, so its gradient gain is `\|x\|/max(\|x\|²,1e-10)` — peaking at **1e5, at `\|x\| = 1e-5`**. ⚠ **Not 1e10, and exactly 0 at `x = 0`** (see the correction below). The hazard is a **prediction** whose STFT magnitude sits near 1e-5; a silent target (row 7) matters only because it drives `\|y\|` down *through* that band, and with γ=0.3 it is pulled toward `1e-12^0.3 = 2.51e-4`, not toward zero. `p_biquad = 0.5` adds up to ±45 dB tilt (row 6); the un-re-derived `1e-10` ERB floor lets the feature reach −2.13 on silent bands (row 8). One oversized step degrades the model → more bins land in the hazard band → more amplification: **positive feedback** | ✅ the feedback loop *is* a multi-epoch rise | ✅ and in the right way — it predicts an **arbitrary-looking** epoch, which is what was reported | **Best combined fit, but a HYPOTHESIS** — see the evidence caveat below. ⚠ Upstream has the identical clamp (`utils.py:74`) and survives **only** because of its skip guard |
| **2** | **`amsgrad=False`** (row 2) | ✅ this is the amplifier that turns one spike into a 3-epoch runaway instead of a one-step blip | ❌ predicts a random onset; LR is ~99% of peak throughout epochs 3–30, so no distinguished epoch | **Strong amplifier, weak trigger.** Almost certainly a co-factor with #1 |
| **3** | **Missing non-finite guard** (row 1) | ❌ silent until the very end | explains "13", not "10" | **The terminal mechanism, not the cause.** Still the #1 *fix*: ~10 lines, and it converts a dead run into a logged, skipped batch |
| **4** | Missing batch ramp (32 fixed vs upstream 64@10) | weakly — higher gradient variance at fixed LR does raise blow-up risk | ❌ **demoted.** The epoch-10 coincidence only means something if the port shared upstream's schedule, and it does not: **the port ran batch 32 from epoch 0, so nothing changes at epoch 10 inside its own run.** The realised upstream ramp is also a doubling (16/24/32/64), not a quadrupling | **Likely a spurious coincidence.** Real recipe gap, not the trigger |
| 5 | MRSL 7.85× scale + `grad_clip = 1.0` (row 18) | no | no — would bite hardest during warmup, epochs 0–3 | background |
| 6 | 100- vs 120-epoch WD compression (row 17) | marginally | no — 2.45e-4 vs 8.6e-5, both ≪ the 0.01 endpoint | background |

⚠ **The evidence does not single out one cause.** #1 and #2 are naturally
*serial* (spike, then runaway), so the answer may be both, with #3 supplying the
permanence. What it *does* rule out is #4 as the primary explanation.

### The one discriminating experiment

**Instrumented resume from the last good checkpoint (epoch 9). One run.**

Three lines of instrumentation:

1. Log the value `clip_grad_norm_` **returns** (the pre-clip `total_norm`) at
   every step — free, it is already computed.
2. Install upstream's guard: `error_if_nonfinite=True` inside try/except → dump
   the batch to wav, `continue`, tolerate up to 50.
3. Dump any batch whose pre-clip `total_norm` exceeds ~100× the running median,
   even when finite.

Read-out:

- **Isolated enormous spikes** (10³–10⁶× median) at a handful of steps, and
  skipping those batches lets the run continue with flat/falling loss →
  **cause #1**. The dumped wavs carry the signature (zero/near-zero target, heavy
  biquad tilt, silent top bands), which also says whether to fix `p_biquad`,
  `noise_only_p`, the `1e-10` floor, or the `_SafeAngle` clamp.
- **Grad norms bounded** (clipped at 1.0 throughout, no outliers) yet the loss
  still rises from epoch 10 → the step size/direction is the problem, not spikes
  → **cause #2**. Confirm with a same-seed rerun at `amsgrad=True`.

**Apply regardless, ~15 lines, no effect at the shipped config:** the non-finite
guard (row 1), `amsgrad=True` (row 2), pre-clip grad-norm logging.

⚠ **Possibly no rerun needed.** The per-**step** (not per-epoch) train-loss trace
for epochs 10–13, plus whether val loss rose in step with train, may already
discriminate: a single-step inflection → #1; a smooth ramp → #2.

---

## 4. Do-not-touch list

Anyone aligning mechanically will break the port with these. Each is a
deliberate, verified divergence.

### Grid and latency

1. **`n_fft = 1024`, `win_len = 1024`, `hop_len = 512`.** Do not restore
   960/480 — it is not radix-2. Everything downstream is derived from this grid:
   281 frames per 3 s segment, the 3τ = 273-frame transient argument, the
   norm-init calibration, the 10.7 ms latency budget.
2. **`mask_lookahead = 1`, `df_lookahead = 1`** (not upstream's 2/2). This *is*
   the latency budget (`config.ini:15-27`). ⚠ Do not align the lookahead
   **mechanism** either: the port's in-conv centred pad is provably equivalent to
   upstream's feature-shift scheme at equal L — verified identical except the
   first `kt-1 = 2` frames of a stream.
3. **`df_bins = 96`** stays 96 (crossover 4500 Hz, not upstream's 4800 Hz).
   Changing it to 102–103 for "parity" changes the feature contract and requires
   retraining. Document the divergence; do not silently fix it.

### Normalisation

4. ⚠ **SUPERSEDED — this item now says the opposite of what the code does.**
   It used to read: *"`erb_norm_init = -15/-60` and `spec_norm_init = 0.06/0.012`;
   do not restore libDF's `-60/-90` and `0.001/0.0001`."*

   **The calibrated values were reverted.** Training diverged under them and
   converged under libDF's literals, on an identical corpus, both runs from
   scratch, with `37db9df` (norm-init only) as the sole behavioural change
   between them. The code now carries libDF's `-60/-90` and `0.001/0.0001`.

   The calibration measurements all still reproduce — STFT power ratio +29.82 dB /
   magnitude 30.98×, filterbank tilt +3.01…+18.58 dB, composite mean +38.97 dB,
   482 of 513 shared bins, endpoint-rounding cost 0.13 dB RMS — so the calibrated
   pair is *more accurate* and *less trainable*. That is a result, not a mistake to
   explain away.

   **The live hypothesis for why**, and the reason this may be recoverable: the
   `+1e-10` erb_db log floor pins a numerically-silent band at −100 dB, and
   `(erb_db − mu)/40` gives **−1.00 at band 0 under libDF's init** — exactly
   upstream's own bound — versus **−2.125 under the calibrated init**. libDF's
   values keep the floor harmless by coincidence; the calibrated ones do not.
   Correcting the floor for this port's +39 dB scale would restore the bound *and*
   keep the calibration. Until that is measured, the literals stand.

   ⚠ What survives from the original item: if the calibrated pair is ever
   restored, it is tied to `normalized=True`, the triangular sum bank, **and this
   corpus's `target_rms_min/max` level regime** — all three. Do not port those
   numbers to another rate, bank, or corpus.
5. **`erb_norm_tau_sec` / `spec_norm_tau_sec = 1.0`** — keep **τ in seconds** and
   re-derive α. Do not hardcode upstream's α: α is grid-dependent, and 0.989 is
   the port's own value. (Fix the comment that claims otherwise — row 21 — but
   keep the mechanism.)
6. **The triangular, overlapping, energy-SUM forward ERB bank**
   (`model.py:87-96`, `mode=0`) — deliberate, shared with this project's
   RNNoise-ERB / `bandERB.ipynb` lineage, and the reason for the recalibrated
   init.
7. **`erb_inv` needs NO row normalisation.** All 513 row sums are exactly 1.0 —
   verified. Do not "add the missing normalisation" that upstream's
   `erb_fb(inverse=True)` appears to have; upstream's inverse is a plain 0/1
   transpose and the port's is a partition of unity. Adding normalisation breaks
   `gain=1 → bin_gain=1`.
8. **The `[:, df_bins:]` slice of `erb_inv`** instead of a full matmul then
   overwrite — mathematically identical (matmul is column-independent) and
   cheaper.

### Architecture

9. **`df_gru` runs at `linear_groups = 8`.** Do not "fix" it to the config's 16;
   16 puts the model 8,192 params off the checkpoint. `model.py:493-496` already
   warns — fix the *contradicting docstring* at `model.py:243-244` (row 24), not
   the code.
10. **Transposed convs stay separable/depthwise.** Do **not** restore upstream's
    shipped `convt_depthwise = False`; that key is declared and never read, and
    obeying it builds dense 12,288-param transposed convs that do not reconcile
    against the checkpoint.
11. **`enc_channels = 64` and `enc_lin_groups = 32`** come from upstream's
    *shipped* config, not the code defaults (16 and 16). Shipped-config wins; do
    not "restore defaults".
12. **`convt3` and `conv0_out` are ordinary convs, not transposed**;
    `conv_kernel` (not `convt_kernel`) applies to them. Verified against the
    checkpoint.
13. **The removed lsnr head and absent `[localsnrloss]`.** Do not re-add for
    parity. Measured contribution is 0.2–4% of total loss (0.001–0.228 vs MRSL
    2.5–33 in upstream units), most of it landing on the head's own 513 params;
    `lsnr_dropout = False` means nothing in the forward path consumes it, and the
    2,135,484 − 513 = 2,134,971 reconciliation depends on its absence. Document
    the *inference* consequence instead (row 30).
14. **`df_gru_skip = 'identity'`'s assert compares `emb_in_dim` vs `df_hidden`.**
    This is the correct guard; upstream's is arguably wrong. Do not align it —
    document it (row 26).
15. **`validate_signal_config` is the one divisibility guard**, and it checks
    `n_erb % 8` — upstream's stricter rule, not the `% 4` the two stride-2 stages
    would need. `model.py` carries no equivalent assert, so do not delete it on
    the assumption that it does. Both the trainer and `denoise.py` call it.
    Enforced by `test_n_erb_divisibility_follows_upstreams_stricter_rule`.

### Loss

16. **Plain (not sqrt) Hann + `hop = n_fft // 4` + `normalized=True` in the loss
    STFT, while the model path uses sqrt-Hann.** This asymmetry is
    upstream-correct (`df/loss.py:42-61` uses plain Hann; the pipeline uses
    Vorbis/sqrt-Hann). Do not unify them. Relatedly, the loss STFT legitimately
    keeps `center=True` — row 10 is scoped to the **model-input** STFT only.
17. **`fft_sizes = 256,512,1024,2048`** are absolute sample counts at 48 kHz and
    need no re-derivation for the 1024/512 grid.
18. **Comparing `istft(enhanced)` against the raw `clean` waveform** (upstream
    round-trips its target). Safe because the port's round trip is unity-gain to
    8e-17. Do not "fix".
19. **`factor = 500` / `factor_complex = 500`.** Do not rescale by `960^-0.3` as
    a mechanical alignment (row 18).

### Data / process

20. **`p_resample = 0.1` means low-SR-source simulation here**, not upstream's
    time-scale resample. Same key, same probability, different augmentation. Do
    not repurpose it without noting the port then loses its bandwidth-limiting
    aug.
21. ⚠ **SUPERSEDED — `target_rms_min/max` no longer exists in dataset_gen.**
    It used to read: *"`target_rms_min/max = -35/-15 dBFS` — coupled to item 4.
    Changing the level regime invalidates the norm-init calibration."* That
    range was traced to the DNS-Challenge `noisyspeech_synthesizer` convention,
    not a DeepFilterNet setting, and was removed from `AINR/dataset_gen`
    (verified against libDF's actual `dataset.rs`/`augmentations.rs`, which use
    discrete `[-6, 0, 6]` dB gains and no continuous target-level step). Since
    the calibrated norm-init pair is already reverted (item 4) this does not
    change today's active config, but if the calibrated pair is ever restored,
    `calibrate_norm_init.py` must be re-run against a corpus generated *without*
    `target_rms` — its old measurements assumed the −35/−15 dBFS bound and are
    now stale.
22. **`FEATURE_VERSION` / `MODEL_VERSION` gating and the checkpoint contract.**
    Every "align" row that touches numerics — `spec_norm_eps` removal, the
    `1e-10` ERB floor, `erb_power`'s expression, the calibrator's floor — goes
    through a version bump, batched together, not as silent edits.
