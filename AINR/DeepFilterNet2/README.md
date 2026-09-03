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

`dfn2_prepost.h/.c` is the integrator's entry point: one opaque object that
composes `dfn2_process.c` and `dfn2_model_io.c` (neither changes, so their
standalone parity builds keep linking) behind the same lifecycle and per-hop
shape as the AIAEC classes, described once in `../../AIAEC/README.md`
("C pre/post-processing"). Specific to this one: `pre_process` returns 0 on
the first hop (the graph needs its right-hand neighbour) and 1 after; the
spectra at the `DFN2_IO_FREQ` boundary are torch.stft normalized=True on this
48 kHz/1024 grid, so chaining an AIAEC spectrum in is a 32x scale error (the
header's warning block); and the window is copied rather than borrowed
because `DFN2State` embeds its table by value. Its gate is
`../tests/test_dfn2_prepost_c.py`; `make -C .. lib` ships it in
`libainr_prepost.a` (`make -C .. print-lib-path` prints the configuration-keyed
location).

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
`t`, and returns one hidden tensor per GRU -- `h_encoder` `(1,1,256)`, `h_erb`
`(2,1,256)` and `h_df` `(2,1,256)` -- plus the four-frame `df_convp` history
for the next invocation. Each hidden is PyTorch's own stacked
`(num_layers, 1, hidden)` shape, which is exactly how `DFN2ModelIOState`
already lays its `[layers][hidden]` arrays out, so a runtime binds one tensor
to one contiguous field. The extra history is required because the
input kernel sees three frames while the DF residual path has a causal
five-frame kernel; omitting it would silently reduce the trained receptive
field. CPU-side window/state storage is defined by
`dfn2_model_io.c/.h`, which also exports the window-slide and state-commit
helpers so an external dual-input consumer with its own state struct reuses
this window/commit discipline instead of inlining a second copy of the
memmove/memcpy pair. `dfn2_model_io_commit_state()` and
`dfn2_model_io_commit_arrays()` take one array per GRU stack --
`erb_hidden_next[DFN2_MODEL_ERB_GRU_LAYERS][DFN2_MODEL_GRU_HIDDEN]` and
`df_hidden_next[DFN2_MODEL_DF_GRU_LAYERS][DFN2_MODEL_GRU_HIDDEN]` -- matching
the graph tensors element for element. State commit is transactional: a null
or non-finite accelerator output returns `-1` before any persistent state is
overwritten. The exported metadata carries `state_layout_version`,
kept numerically equal to `DFN2_MODEL_IO_LAYOUT_VERSION` in that header, so an
integrator can refuse a graph whose state layout no longer matches the struct
it allocated. `inference.py calib --frames N` captures `N`
streaming invocations with real non-zero state. `calibrate_norm_init.py` is a
different tool: it estimates the feature EMA initialization constants used by
the C frontend.
Use `--format npz --output calib/dfn2.npz` when a NumPy archive is needed.

### Fixed-batch NPU profiling

`inference_batch.py` exports a static batch graph and its matching calibration
inputs.  Batch elements are independent streaming lanes: they share weights,
but each has its own GRU and `df_convp` state.  They are not consecutive audio
frames packed together.

```bash
python3 inference_batch.py export \
  --model output/dfn2_best.pth --batch-size 4 \
  --gru-state-layout combined \
  --output output/dfn2_stream_b4.onnx --verify

python3 inference_batch.py calib \
  --model output/dfn2_best.pth --wav-dir /path/to/noisy_wavs \
  --batch-size 4 --batches 1000 --gru-state-layout combined \
  --format bin --output calib/dfn2_b4
```

Each tensor BIN file is one complete NPU invocation and therefore contains
all four lanes.  The combined recurrent input is `(B,5,1,256)`; the split
layout keeps PyTorch's native `(layers,B,256)` hidden tensors.  The generated
JSON records `fixed_batch_size`, exact tensor shapes, and the number of source
frame snapshots.  The C model-I/O helper remains the shipped batch-one/split
contract; fixed batch and combined state are profiling contracts only.

## Recurrent-state layouts

`--gru-state-layout` selects how the recurrent state is presented at the graph
boundary, on `export_onnx.py` and on `inference.py calib` alike (calibration
exports the graph from the same model instance in the same process, so pass
the flag there rather than exporting twice). The maths is identical either
way; only the boundary and the published `state_layout_version` differ.

| layout | version | state inputs | status |
| --- | --- | --- | --- |
| `split` (default) | 5 | `h_encoder`, `h_erb`, `h_df`, `df_convp_history` | shipped; `dfn2_model_io.h` binds it |
| `combined` | 7 | `h_gru` `(1,5,1,256)`, `df_convp_history` | experimental; no C runtime binds it |

Versions 4 and 6 are retired (`RETIRED_LAYOUT_VERSIONS`); the contract test asserts
`DFN2_MODEL_IO_LAYOUT_VERSION` against both the retired numbers and the
combined one, so a C-side bump onto either fails a test rather than a review.

ONNX has no stacked GRU op, so a two-layer stack still becomes two GRU nodes
fed by per-layer `Slice`s -- five GRU nodes in both layouts. What the split
layout trades is therefore PTQ scale granularity, per GRU rather than per
layer: the two layers of `h_erb` share one input tensor and one quantization
scale, in exchange for the native shape and an exact match to the C struct.

The combined layout exists to measure what a single shared scale costs, which
`inference.py calib` reports per GRU under `gru_state_slices`:

```bash
python3 inference.py calib --model output/dfn2_best.pth \
  --wav-dir /path/to/noisy_wavs --frames 8192 --format bin \
  --gru-state-layout combined --output calib/dfn2_combined
```

Adopting it would be a contract change, not a flag flip: `dfn2_model_io.h`,
its commit API, the I/O tables and the C tests would all have to move with it.

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
