# DeepFilterNet2

DFN2 branch with cascade/alpha output composition. The shipped grid is 48 kHz; other grids are chosen at build time
(see [Choosing the grid](#choosing-the-grid)).

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

Shipped signal contract (the grid of the default build; other grids are in
[Choosing the grid](#choosing-the-grid)):

- 48 kHz;
- FFT/window/hop `1024/1024/512`;
- 32 ERB bands;
- `df_bins=96`, `df_order=5`;
- mask/DF lookahead `1/1`;
- optional post-filter disabled.

Each one-frame lookahead is a deliberate model choice. Upstream releases use
2/2; this port uses 1/1. In DFN2's cascade those two dependencies are serial,
so the deployed total is two hops. Changing either value requires retraining
and a new checkpoint contract; the other legal pairs export and run too (see
[Lookahead pairs](#lookahead-pairs)).

## Checkpoints

Current model version:

```text
dfn2_fmajor_pathway_cascade_alpha_no_lsnr_v6
```

`train.py --resume` and `inference.py --model` validate the serialized signal,
feature, and model contract before loading.

## Train and infer

```bash
python3 train.py --config config.ini --packed-dir /path/to/data_48k

python3 inference.py --config config.ini \
    --model /path/to/dfn2_checkpoint.pt \
    --input input_48k.wav --output output_48k.wav
```

Use `AINR/dataset_gen` to produce or pack the noisy/clean pairs at the grid's sample rate (48 kHz for the shipped grid).

## C deployment boundary

`dfn2_process.h/.c` owns the streaming STFT, ERB/complex normalization,
parameter-free cascade/alpha composition, attenuation limit, and WOLA. The
accelerator only needs to emit `(erb_mask, coefs, alpha)`. Run
`make -C .. test-simd` to compare the default NEON path against the scalar
reference; `make -C .. SIMD=0 test` forces the latter.

`dfn2_prepost.h/.c` is the integrator's entry point: one opaque object that
composes `dfn2_process.c` and `dfn2_model_io.c` behind the same lifecycle and per-hop
shape as the AIAEC classes, described once in `../../AIAEC/README.md`
("C pre/post-processing"). Specific to this one: `pre_process` returns 0 on
the first hop (the graph needs its right-hand neighbour) and 1 after (every
N-th hop in a build for N frames per call, see
[Multi-frame export](#multi-frame-export)); the
spectra at the `DFN2_IO_FREQ` boundary are torch.stft normalized=True on the
build's grid (1/sqrt(n_fft); 1024 points on the shipped 48 kHz grid), so chaining
an AIAEC spectrum in is a sqrt(n_fft) scale error (32x on the shipped grid; the
header's warning block); the window is borrowed (a `cfg.window` of NULL
makes the pool carve the sqrt-Hann table for the analysis object); in
`DFN2_IO_TIME` the synthesis is the application's audiolib `AcIstft`, built
from `dfn2_framing_config()` and handed in as `cfg.istft` (the class pushes
into it and resets it with its own state; with `DFN2_IO_FREQ` the caller
pushes the spectrum from `post_process_freq` itself); and the two recurrent state
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
`df_lookahead=1` have a total streaming delay of **two hops** (21.33 ms on the
shipped 48 kHz grid), even though each individual knob is one. The C test uses changing
masks and nontrivial future taps so an off-by-one head/spectrum pairing cannot
pass as a merely finite output.

## Lookahead pairs

`mask_lookahead` (M, 0..2) and `df_lookahead` (D, 0..4, i.e. below `df_order`)
are independent and part of the trained contract (`config.ini [model]`; a
checkpoint is refused under another pair). All 15 pairs export and run; (1,1)
is the shipped one. The shared rules (window, build flags, guards, tests) are in
[`../README.md`](../README.md#lookahead-pairs-deepfilternet2-and-deepfilternet3).
Specific to DFN2:

- DFN2 is a cascade: the deep filter reads the ERB-masked spectrum, so the
  output is `M + D` frames behind the input and the end-to-end latency is
  `1 + M + D` hops (three for the shipped pair). The first M hops only fill the
  window; the following D hops run the accelerator but emit nothing.
- The rings hold `max(DF_ORDER, M + D + 1)` frames (`DFN2_DF_RING`): five up to
  `M + D = 4`, six for `M + D = 5`, seven for `M + D = 6` (2/4). The state
  struct, the pool size and the pool hash follow the pair.

## Choosing the grid

The signal grid and the model geometry are build-time constants in `dfn2_grid.h` (included by `dfn2_process.h`). The C
library, the trained model and its ERB matrices must belong to the same grid. With no flag the build is the shipped
48 kHz grid: its objects (machine code, constant sections and symbols, full and `-DDFN2_MASK_ONLY`) are bit-identical to
those built before the header existed, and its `DFN2_IO_FREQ` pool is still 219,536 bytes with hash `d2d3bc45`.
The lookahead pair, `DFN2_MASK_ONLY` and `DFN2_FRAMES_PER_CALL` are separate build choices and do not depend on the grid.

**Built-in presets**, `-DDFN2_GRID_KHZ=16|32|48` (48 when nothing is chosen):

| preset | sr | n_fft / hop | bins | n_erb | df_bins | DF cutoff | alpha | analysis scale | state |
|---|---|---|---|---|---|---|---|---|---|
| `48` (default) | 48000 | 1024 / 512 | 513 | 32 | 96 | 4500 Hz | 0.989 (`0x3f7d2f1b`) | 0.03125 | shipped |
| `32` | 32000 | 512 / 256 | 257 | 28 | 96 | 6000 Hz | 0.992 (`0x3f7df3b6`) | 0.044194173 | provisional: no model is trained yet (28 is a multiple of 4, which `train.py` and the C library both ask for) |
| `16` | 16000 | 256 / 128 | 129 | 24 | 96 | 6000 Hz | 0.992 (`0x3f7df3b6`) | 0.0625 | matches `train.py`'s derivation bit for bit; quality unmeasured |

A preset is complete: defining one more value next to it (`-DDFN2_GRID_KHZ=16 -DDFN2_N_ERB=20`) is a `#error`, and so is
mixing a preset with the macros below.

**A grid defined outside the library**, for a model trained on another grid. Define these on the build line
(`-DDFN2_SR=24000 ...`) or in a header named by `-DDFN2_GRID_HEADER='"my_grid.h"'`:

| macro | meaning | `config.ini` / `train.py` |
|---|---|---|
| `DFN2_SR` | sample rate in Hz (defining it selects this path) | `[signal] sr` |
| `DFN2_N_FFT` | FFT size, a power of two in 16..8192 | `[signal] n_fft` |
| `DFN2_N_ERB` | ERB bands | `[model] n_erb` |
| `DFN2_DF_BINS` | bins owned by the deep filter | `[model] df_bins` |
| `DFN2_ERB_NORM_ALPHA` | ERB-feature EMA coefficient, a float literal such as `0.992f` | `make_norm_alpha` |
| `DFN2_SPEC_NORM_ALPHA` | complex-feature EMA coefficient | `make_norm_alpha` |

The six macros have no default (a missing one stops the build and names itself), so no value of another grid is
applied silently. hop = n_fft/2, window = n_fft and the bin count follow from n_fft (audiolib `audio_stft.h`: 50 % overlap,
sqrt-Hann, no zero padding); they are derived, not configured. Example, a 24 kHz test grid:

```bash
GRID='-DDFN2_SR=24000 -DDFN2_N_FFT=512 -DDFN2_N_ERB=24 -DDFN2_DF_BINS=96 -DDFN2_ERB_NORM_ALPHA=0.989f -DDFN2_SPEC_NORM_ALPHA=0.989f'
make -C .. lib CFLAGS="-O2 -std=c11 $GRID"
```

**Guards** (each is a `#error` at compile time): n_fft a power of two in [16, 8192]; `n_erb` positive and a multiple of 4;
`2 * n_erb <= n_bins`; `df_bins` in [2, n_bins], and below `n_bins` for the full variant (bins above the deep filter keep
the masked spectrum). `DFN2_DF_ORDER` is fixed at 5, because the graph's pathway window and the state layout are built for
it, and defining it is a `#error`. The `n_erb` rule is the C path's: the ERB kernel runs bands in fours and the
encoder halves the band axis twice, so only `% 4` is needed. `train.py`'s `validate_signal_config` asks for the same `% 4`
(upstream DeepFilterNet asserts `% 8`; this port does not keep it), so the 32 kHz preset's 28 bands can be trained.

**Where the constants come from.** Both alphas are `make_norm_alpha(sr, hop, tau)` of `train.py`: `exp(-(hop/sr)/tau)`
rounded to three decimals (one more digit if that reaches 1.0), tau = 1.0 s (`[feature] erb_norm_tau_sec` and
`spec_norm_tau_sec`). hop 512 at 48 kHz gives 0.989; hop 128 at 16 kHz and hop 256 at 32 kHz are both 8 ms and give 0.992.
They are written out as literals so they cannot depend on the compiler; `tests/test_dfn2_grid.py` compares them with
`train.py` bit for bit. Change tau, and they, with the checkpoint. The analysis scale is `train.py`'s
`analysis_scale()` = `2*hop/win^2*sqrt(n_fft)`, which is exactly `1/sqrt(n_fft)` for win = n_fft and hop = n_fft/2. It is
the same number as `DFN2_NORMALIZED_STFT_SCALE` (the `torch.stft(normalized=True)` factor, with `DFN2_NORMALIZED_STFT_UNSCALE`
as its inverse; both are computed from n_fft, so a new grid needs no table entry) but a different thing: it only scales the
copy that feeds the normalisers, and it is derived, not defined. The normaliser starting points (`DFN2_ERB_NORM_INIT_LO_DB` and
the others in `dfn2_process.h`) are libDF's literals, the same for every grid; a model trained with other ones defines them on
the build line, and that build is not the shipped grid, even on the 48 kHz preset.

**Using a grid.**

- Build the library and the application (the board sample too) with the same flags. A `-D` written only in the
  application's own file does not reach the library objects, which are compiled separately. Where a `.c` -> `.o` rule
  builds the sources, objects built with other flags (another grid, lookahead pair or variant) have the same names
  (`dfn2_process.o` and so on): keep one object directory per flag set, or clean between builds. A stale object fails to
  link with an undefined `dfn2_variant_*` or `dfn2_lookahead_*` symbol; `make -C AINR lib` already keeps a directory
  per flag set.
- Objects of different grids do not link: `dfn2_process.c` defines `dfn2_variant_full` / `dfn2_variant_mask_only` on the
  shipped grid (the 48 kHz preset) and `<that name>_s<sr>_f<n_fft>_e<n_erb>_d<df_bins>` on any other, a grid defined on the
  build line with the shipped integers included (for example
  `dfn2_variant_full_s16000_f256_e24_d96`), and the other units, and an application that runs
  `DFN2_REQUIRE_LOOKAHEAD_PAIR_LINK()` in `main()`, reference the symbol of their own grid.
- The pool identity (`build_flags_hash`) always covers sr, n_fft, hop, n_erb, df_bins and df_order; off the shipped grid it
  also folds in both alphas and the five normaliser starting points (seven floats), so two models on the same integers with
  a different tau are not interchangeable. The shipped grid's hash is unchanged.
- "Shipped" means the 48 kHz preset itself, with the normaliser starting points left alone. A build that overrides one of them,
  and a grid defined on the build line even with the shipped integers, is never the shipped one: its alphas and starting points
  are the definer's, so it gets its own symbol name and its own pool hash.
- Pair it with the accelerator model and the ERB matrices exported from a checkpoint of that grid: `export_onnx.py` for the
  graph, `export_erb_matrix.py --runtime-bins` for `erb_fwd.bin` / `erb_inv.bin` (n_bins x n_erb float32 values each: 65,664
  bytes at 48 kHz, 12,384 at 16 kHz). The board sample's deployed `model.bin` and `erb_matrix.bin` change with the grid too.
  `DFN2ModelIoDescriptor` still refuses a graph exported for other geometry at run time.

Sizes per grid (this host, 64-bit, shipped pair 1/1; bytes):

| grid | `sizeof(DFN2State)` | FREQ pool | TIME pool | mask-only FREQ pool | hop | latency of the identity path (1 + M + D hops) |
|---|---|---|---|---|---|---|
| 48 kHz, erb 32 | 67,368 | 219,536 | 233,936 | 34,816 | 512 = 10.67 ms | 3 hops = 1536 samples = 32 ms |
| 32 kHz, erb 28 | 43,752 | 189,680 | 196,912 | 21,344 | 256 = 8 ms | 768 samples = 24 ms |
| 16 kHz, erb 24 | 31,912 | 174,672 | 178,320 | 14,528 | 128 = 8 ms | 384 samples = 24 ms |
| outside, 24 kHz / 512 / erb 24 | 43,688 | 189,520 | 196,752 | not measured | 256 = 10.67 ms | 768 samples = 32 ms |

The TIME pool is the pool of `DFN2_IO_TIME` with the analysis object carved inside; the application's FFT handle and
`AcIstft` are not in either pool. The model's own lookahead is `M + D` = 2 hops (21.33 ms at 48 kHz, 16 ms at 32 and 16 kHz).

**Limits.** `pipelines/dfn/dfn_res_stage.h` takes its scaling constants from the grid (`DFN_RES_STAGE_IN_SCALE` and
`_OUT_SCALE` are `DFN2_NORMALIZED_STFT_SCALE` and `_UNSCALE`, 2^-5 and 2^5 on the shipped grid). The rate bridge
(`dfn_rate_bridge.c`) is written for the 48 kHz model, so its `resolve()` returns -1 when `DFN2_SR != 48000` or
`DFN2_N_FFT != 1024`; the pipelines and their Python twin are verified on the shipped grid only. Tests:
`../tests/test_dfn2_grid.py` (constants against `train.py`, features and compose against Python including `SIMD=0`, the C
stream against the offline model, the `#error` refusals, the link failures in both directions and between an application
and a library of different grids, the pool identity) and `tests/test_dfn2_contract.py` (reads `N_ERB` and `DF_BINS` from
the first `#define` in `dfn2_grid.h`, the 48 kHz branch). Shapes quoted elsewhere in this README (`(1,1,3,32)` and so on)
are the shipped grid's.

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

The C runtime loads the files written by `export_erb_matrix.py --runtime-bins DIR`
(`erb_fwd.bin` bin-major, `erb_inv.bin` band-major); the `--output-dir` formats
are for toolchain validation.

The graph is stateless from the accelerator's perspective. Each invocation
receives exactly three feature frames `[t-(2-M) .. t+M]` (`[t-1,t,t+1]` for the
shipped `mask_lookahead` M = 1), emits the heads for frame `t`, and returns
two state tensors: `h_gru_out` `(1,5,1,256)` and `df_convp_history_out`
`(1,64,5,96)`. Each state input and its output have the
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

## Multi-frame export

`export_onnx.py --frames-per-call N` (default 1, at most 64 as in the C host) writes a graph that runs N
frames per invocation, for an accelerator where one call per hop costs more
than the extra latency. `--frames-per-call 1`, or no option, writes the same
ONNX and JSON bytes as before the option existed.

| | one frame per call | N frames per call |
| --- | --- | --- |
| `erb`, `spec` | `(1,1,3,32)`, `(1,2,3,96)` | `(1,1,N+2,32)`, `(1,2,N+2,96)`, frames `[t-(2-M) .. t+N-1+M]` |
| `erb_mask`, `df_coefs`, `df_alpha` | heads of frame `t` | heads of frames `t .. t+N-1`, oldest first |
| `h_gru`, `df_convp_history` | `(1,5,1,256)`, `(1,64,5,96)` | unchanged |
| metadata | — | `frames_per_call`, `chunk_window_alignment`; `input_feature_frames` = N+2, `output_frames_per_invocation` = N |

Head `i` reads the three window frames starting at slot `i`, exactly the
one-frame window of frame `t+i`. The three GRUs run over the N frames as one
sequence (one GRU node per layer, as for N = 1) and `h_gru_out` is the hidden
state after the last frame. `df_convp_history_out` is the last five frames of
the pathway input: the kept history and the new frames for N < 5, the new
frames for N = 5; for N > 5 the last five frames of the DF input convolution
are computed from their own slice of the `spec` window, so the state output is
a convolution result and not a Slice (the in-place rule above; the same
multiply-adds as one convolution). Every other layer works on one frame at a
time. Fed in chunks of N with the state carried between calls, the graph equals
the one-frame graph fed frame by frame. `onnx_streaming_contract.py`
(`validate_frames_per_call`) refuses a graph whose `frames_per_call` disagrees
with its window and head shapes, and a one-frame graph that records one.

`inference.py calib --frames-per-call N` exports the N-frame graph and records
its invocations (`--frames` counts invocations, the report records
`frames_per_call`).

The C host takes N as `-DDFN2_FRAMES_PER_CALL=<N>` (1..64, plain decimal; the
AINR Makefile variable `DFN2_FRAMES_PER_CALL` adds it) on every translation
unit, with the lookahead pair:

- the window holds `N + 2` frames; the first call is ready after `M + N`
  pushed frames, every later one after N more (`pre_process` returns 1 on hops
  `M+N-1`, `M+2N-1`, ...); `dfn2_model_io_frame_slot()` names the slot of the
  last pushed frame;
- `frame_inputs()` publishes N heads per tensor, commit composes the N frames
  in one burst, and `post_process` returns one hop per call from that burst,
  so the output trails by N - 1 more hops: latency `1 + M + D + N - 1` hops
  (`DFN2_PREPOST_LATENCY_FRAMES` + 1; 10 hops, 106.7 ms, at the shipped pair
  with N = 8). A refused commit zeroes the state once and passes all N frames
  as the identity; flushing means pushing zero hops until the last real one
  has been emitted;
- `DFN2ModelIoDescriptor` keeps its 18 fields and 72 bytes: N is
  `input_frames - 2` (`input_frames` = N + 2, metadata `input_feature_frames`),
  so `dfn2_model_io_descriptor_validate()` and `dfn2_model_run_frame()` refuse
  another N, and an adapter compiled against the one-frame header
  (`input_frames` = 3) works with an N = 1 library and is refused by an N > 1
  one;
- N > 1 is folded into the pool hash (a pool of another N is refused) and
  objects built for different N do not link (`dfn2_frames_per_call<N>`);
- N > 1 changes the per-hop contract, so a unit written for one frame per call
  does not compile against it (`#error` in `dfn2_model_io.h`); a unit that
  drives N frames defines `DFN2_MULTI_FRAME_HOST`. The pipelines DFN stage and
  its Python twin are one-frame units and stay at N = 1.

`DFN2_MODEL_IO_LAYOUT_VERSION` stays 8: it names the state tensors, which N
does not change, the window and head shapes carry N through `input_frames` in
the descriptor and `frames_per_call` in the metadata, and a bump would change
the one-frame graph's metadata. With N = 1 the pool size, the pool hash and
every compile line are those of a build without the option.

Tests: `../tests/test_dfn2_frames_per_call.py` (wrapper and exported graph
streamed in chunks against the one-frame graph, all pairs; a wrong state carry
fails; metadata and validator; calibration) and
`../tests/test_dfn2_frames_per_call_c.py` (the C host against the one-frame
host, byte for byte, with a stand-in accelerator; reset, refusal, guards;
the one-frame header's 72-byte descriptor against libraries of N = 1 and 3).

## Mask-only (stage 1)

The cascade is stage 1 (ERB mask over the full band) then stage 2 (deep filter on the low bins, blended by `alpha`). The
mask head does not depend on the deep-filter decoder, so the same checkpoint can run stage 1 alone:

- `python3 inference.py --model ... --input a.wav --output out.wav --mask-only` writes `out.wav` (full) and
  `out_mask_only.wav` (stage 1, `model.compose_mask_only`). `--input-dir/--output-dir` does the same per file.
- `python3 export_onnx.py --model ... --output m.onnx --mask-only` and `inference.py calib ... --mask-only` write the
  stage-1 graph: inputs `erb`, `spec`, `h_gru` `(1, 3, 1, 256)`; outputs `erb_mask`, `h_gru_out`. The deep-filter decoder
  (two GRU layers, `df_skip`, `df_out`, `df_fc_a`, `df_convp`) and `df_convp_history` are gone: 5 GRU nodes become 3. The
  encoder keeps `df_conv0`, `df_conv1` and `df_fc_emb` (their output feeds the embedding the ERB decoder reads), so the
  grouped linear of `df_fc_emb` stays in the graph. Metadata: `variant: mask_only`, `state_layout_version` 9,
  `df_lookahead` 0.
- C: build every unit with `-DDFN2_MASK_ONLY` (see `dfn2_model_io.h`). The state, the accelerator boundary structs and the
  descriptor shrink accordingly, and the output is `noisy * (mask @ erb_inv)` of the head frame, `DFN2_MASK_LOOKAHEAD`
  frames behind the input. The final attenuation-limit mix (`--atten-lim`, `dfn2_prepost_set_atten_lim`) stays and acts on
  the stage-1 output in Python and in C. `../tests/test_dfn2_mask_only_c.py` compares it with numpy.
- Board sample: `alg_ai3_dfn.c` (`audio_sample/`, outside this repo) is the sample for this graph too. Build the sample and
  every library unit with `-DDFN2_MASK_ONLY` and it binds I/O layout 9: inputs `erb, spec, h_gru` (h_gru is `(1, 3, 1, 256)`),
  outputs `erb_mask, h_gru_out`. The model is then read from `/mnt/sd/audio_model/<quant>/dfn2_lite/` and the output goes to
  `/mnt/sd/audio/output/DFN2_LITE/`; the input and the ERB matrices are the full variant's. It takes the build's grid like the
  full variant does, and objects of the other variant do not link with it (`dfn2_variant_mask_only`); the two builds need
  object directories of their own (see "Using a grid").

No retraining: the quality of stage 1 alone is whatever the jointly trained mask gives (not measured here).

## State boundary

State layout version 8. Both state tensors are rank 4 with the same shape at
the input and at the `*_out` output, bound to one address; the metadata records
`state_binding = in_place_same_address`, `state_handoff` and `state_slicing`.
Versions 4, 5, 6 and 7 are retired (`RETIRED_LAYOUT_VERSIONS`); the contract
test asserts `DFN2_MODEL_IO_LAYOUT_VERSION` is not one of them.

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

The upstream-alignment record is in
[`UPSTREAM_ALIGNMENT.md`](UPSTREAM_ALIGNMENT.md).
