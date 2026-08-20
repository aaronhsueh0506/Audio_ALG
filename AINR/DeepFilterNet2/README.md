# DeepFilterNet2

Current 48 kHz DFN2 branch with restored cascade/alpha output composition.

## Architecture contract

The network predicts a full-band ERB mask, low-band five-tap complex
deep-filter coefficients, and a sigmoid alpha:

```text
input spectrum
    → full-band ERB mask
    → apply the five-tap complex DF to masked bins 0..95
    → alpha * deep-filtered + (1 - alpha) * masked
```

Bins above the DF range keep the ERB-masked spectrum. This is a cascade; it is
not the DFN3 low/high parallel band split.

Default signal contract:

- 48 kHz;
- FFT/window/hop `1024/1024/512`;
- 32 ERB bands;
- `df_bins=96`, `df_order=5`;
- mask/DF lookahead `1/1`;
- optional post-filter disabled.

Each one-frame lookahead is a deliberate model choice. Upstream releases use
2/2; this port uses 1/1. In DFN2's cascade those two dependencies are serial,
so the deployed total is two hops. Changing either value requires retraining
and a new checkpoint contract.

## Checkpoints

Current model version:

```text
dfn2_fmajor_pathway_cascade_alpha_no_lsnr_v6
```

The feature version is unchanged from the earlier v5 band-split lineage, but
the model version is not. Old v5 band-split checkpoints must not be renamed or
force-loaded here.

`train.py --resume` and `inference.py --model` validate the serialized signal,
feature, and model contract before loading.

## Train and infer

```bash
python3 train.py --config config.ini --packed-dir /path/to/data_48k

python3 inference.py --config config.ini \
    --model /path/to/dfn2_checkpoint.pt \
    --input input_48k.wav --output output_48k.wav
```

Use `AINR/dataset_gen` to produce or pack the 48 kHz noisy/clean pairs.

## C deployment boundary

`dfn2_process.h/.c` owns the streaming STFT, ERB/complex normalization,
parameter-free cascade/alpha composition, attenuation limit, and WOLA. The
accelerator only needs to emit `(erb_mask, coefs, alpha)`. Run
`make -C .. test-simd` to compare the default NEON path against the scalar
reference; `make -C .. SIMD=0 test` forces the latter.

Use `dfn2_compose_stream()` for the accelerator boundary, not the aligned
`dfn2_compose()` reference. At wall frame `n`, the returned head describes
`n-mask_lookahead`; the cascade can emit only after that head has also masked
the future DF source. Therefore the shipped `mask_lookahead=1` and
`df_lookahead=1` have a total streaming delay of **two hops** (21.33 ms at
48 kHz), even though each individual knob is one. The C test uses changing
masks and nontrivial future taps so an off-by-one head/spectrum pairing cannot
pass as a merely finite output.

## ONNX, ERB tables and calibration

Install the optional export dependencies from `../requirements-export.txt`.
The ONNX graph contains only the learned heads; STFT, feature normalization,
head composition and WOLA remain in `dfn2_process.c/.h`.

```bash
python3 export_onnx.py --model output/dfn2_best.pth \
  --output output/dfn2_stream.onnx --verify
python3 export_erb_matrix.py --model output/dfn2_best.pth \
  --output-dir output/erb --format all
python3 inference.py calib --model output/dfn2_best.pth \
  --wav-dir /path/to/noisy_wavs --frames 8192 --format bin \
  --output calib/dfn2
```

The graph is stateless from the accelerator's perspective. Each invocation
receives exactly three feature frames `[t-1,t,t+1]`, emits the heads for frame
`t`, and returns one encoder GRU tensor, two independently quantizable ERB GRU
layer tensors, two independently quantizable DF GRU layer tensors, plus the
four-frame `df_convp` history for the next invocation. The C state keeps each
pair contiguous; only the graph boundary splits the layers so a per-tensor PTQ
tool does not force both recurrent layers to share one activation scale. The
extra history is required because the
input kernel sees three frames while the DF residual path has a causal
five-frame kernel; omitting it would silently reduce the trained receptive
field. CPU-side window/state storage is defined by
`dfn2_model_io.c/.h`, which also exports the window-slide and state-commit
helpers so an external dual-input consumer with its own state struct reuses
this window/commit discipline instead of inlining a second copy of the
memmove/memcpy pair. State commit is transactional: a null or non-finite
accelerator output returns `-1` before any persistent state is overwritten.
The exported metadata carries `state_layout_version`,
kept numerically equal to `DFN2_MODEL_IO_LAYOUT_VERSION` in that header, so an
integrator can refuse a graph whose state layout no longer matches the struct
it allocated. `inference.py calib --frames N` captures `N`
streaming invocations with real non-zero state. `calibrate_norm_init.py` is a
different tool: it estimates the feature EMA initialization constants used by
the C frontend.
Use `--format npz --output calib/dfn2.npz` when a NumPy archive is needed.

## Debugging excessive low-frequency suppression

Check these in order:

1. confirm the checkpoint reports the v6 cascade/alpha model version;
2. log ERB mask, alpha, and low-band DF output separately;
3. compare speech-only, noise-only, and steady low-frequency-noise cohorts;
4. verify error/far or clip-to-clip EMA state is not shared or leaked.

If the ERB-masked low band is healthy but the alpha-composed output collapses,
the issue is in the DF/alpha path rather than the shared dataset. If the
ERB-masked low band is already collapsed, audit low-frequency corpus coverage,
speech-only sampling, SNR distribution, and target contamination.

The full upstream-alignment and training-stability record is in
[`UPSTREAM_ALIGNMENT.md`](UPSTREAM_ALIGNMENT.md).
