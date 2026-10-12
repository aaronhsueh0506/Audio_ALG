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
header's warning block); the window is borrowed (a `cfg.window` of NULL
makes the pool carve the sqrt-Hann table once for the analysis and
synthesis objects); and the two recurrent state
tensors (`h_gru`, `df_convp_history`) live in the pool at addresses that never
change. The runtime binds each state output (`gru_hidden_next`,
`df_convp_window_next`) to the same address as its input, so the output of one
invocation is the input of the next with no copy; it must read all state inputs
before it writes any state output there. A commit that finds a non-finite head
or state value zeroes the two recurrent tensors, because the previous state was
overwritten and cannot be restored. `dfn2_prepost_state_generation()` is 1 after
init and advances on every `dfn2_prepost_reset()` and every refused commit
(modulo 2^32): the accelerator's fixed-point copy of the state is not reached
by either event, so a runtime that sees a different value than at the previous
frame refills its state tensors with quantized zeros. Its gate is
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
`t`, and returns two state tensors: `h_gru_out` `(1,5,1,256)` and
`df_convp_history_out` `(1,64,5,96)`. Each state input and its output have the
same shape and are bound to one address. `h_gru` holds the five GRU hidden
rows in `DFN2ModelIOState` memory order -- encoder (row 0), ERB decoder (rows
1-2), DF decoder (rows 3-4; `DFN2_MODEL_GRU_ROW_*`). `df_convp_history` is the
whole causal window of the five-frame DF residual path: the graph drops the
oldest frame with a Slice on the input, appends the new frame and returns the
whole concatenation. That window is required because the input kernel sees
three frames while the DF residual path has a causal five-frame kernel;
omitting it would silently reduce the trained receptive field. CPU-side
window/state storage is defined by
`dfn2_model_io.c/.h`, which also exports the window-slide and state-validate
helpers so an external dual-input consumer with its own state struct reuses
this window/validate discipline instead of inlining a second copy of the
memmove/memcpy pair. `dfn2_model_io_validate_state()` and
`dfn2_model_io_validate_arrays()` take `gru_hidden[DFN2_MODEL_GRU_ROWS]
[DFN2_MODEL_GRU_HIDDEN]` and `df_convp_window[...][DFN2_MODEL_DF_PATHWAY_WINDOW]
[...]`, matching the graph tensors element for element. Validation is a finite
check over the state itself: a null argument returns `-1`, and a non-finite
element zeroes the two arrays and returns `-1` (`dfn2_model_io_validate_state()`
also advances `dfn2_model_io_state_generation()`). A finite but partial or mixed
accelerator write is not detectable. The exported metadata carries
`state_layout_version`, kept numerically equal to `DFN2_MODEL_IO_LAYOUT_VERSION`
in that header, so an integrator can refuse a graph whose state layout no longer
matches the struct it allocated. `inference.py calib --frames N` captures `N`
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
  --output output/dfn2_stream_b4.onnx --verify

python3 inference_batch.py calib \
  --model output/dfn2_best.pth --wav-dir /path/to/noisy_wavs \
  --batch-size 4 --batches 1000 \
  --format bin --output calib/dfn2_b4
```

Each tensor BIN file is one complete NPU invocation and therefore contains
all four lanes.  The recurrent input is `(B,5,1,256)` and the pathway window
`(B,64,5,96)`.  The generated
JSON records `fixed_batch_size`, exact tensor shapes, and the number of source
frame snapshots.  The C model-I/O helper remains the shipped batch-one
contract; fixed batch is a profiling contract only.

## State boundary

State layout version 8. Both state tensors are rank 4 with the same shape at
the input and at the `*_out` output, bound to one address; the metadata records
`state_binding = in_place_same_address`, `state_handoff` and `state_slicing`.
Versions 4, 5, 6 and 7 are retired (`RETIRED_LAYOUT_VERSIONS`); the contract
test asserts `DFN2_MODEL_IO_LAYOUT_VERSION` against them, so a C-side bump onto
one fails a test rather than a review.

| tensor | shape | meaning |
| --- | --- | --- |
| `h_gru` | `(1,5,1,256)` | encoder row 0, ERB decoder rows 1-2, DF decoder rows 3-4 |
| `df_convp_history` | `(1,64,5,96)` | full causal window of the DF residual convolution |

Each GRU reads its rows through a Slice of the graph input on dim 1 followed by
a squeeze, and the next `h_gru` is one Concat on dim 1. ONNX has no stacked GRU
op, so a two-layer stack still becomes two GRU nodes: five GRU nodes in all.
The three GRUs therefore share one input tensor and one quantization scale;
`inference.py calib` reports under `gru_state_slices` what that costs each GRU
(bits lost under a symmetric max-abs int8 scale). The window input has exactly
one consumer, a Slice of axis 2 with the explicit range `[1:5]` and step 1; no
state output is produced by a Slice. `../onnx_streaming_contract.py`
(`validate_inplace_state_graph`) checks this at export time, and
`../check_state_quantization.py` repeats it on an exported graph and, given a
quantization report, aborts when a state input and its output differ in dtype,
scale, zero point or shape. A window row is 96 elements, so on an int8 target
the axis-2 Slice starts at a multiple of 96 bytes; how the toolchain aligns it
is not verified on a board.

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
