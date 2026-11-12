# DeepFilterNet2

DFN2 branch with cascade/alpha output composition. The shipped grid is 48 kHz; the C library holds no grid, the
application builds one and passes it to every object (see [Choosing the grid](reference/grid.md)).

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

Shipped signal contract (the 48 kHz grid of the shipped checkpoint; other grids are in
[Choosing the grid](reference/grid.md)):

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
[Lookahead pairs](#lookahead-pairs)). In C the pair is a field of the grid the
application passes in, not a build setting.

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

The library is frequency domain only and holds no transform, window or grid. The application frames with audiolib (`AcStft` /
`AcIstft`, `ac_stft_push_planar()` / `ac_istft_push_planar()`) and passes one spectrum per hop, framed as in training:
audiolib's default periodic sqrt-Hann window, center 0, hop `n_fft / 2`, plain overlap-add (`AC_OLA_ACCUMULATE`),
`spec_scale = dfn2_stft_scale(grid.n_fft)` (1/sqrt(n_fft)) and `ispec_scale = dfn2_stft_unscale(grid.n_fft)` (sqrt(n_fft)). A wrong window or scale is not
detectable inside the library. Against the training graph's `center=True` padding only the first frame of a clip differs.

`dfn2_process.h/.c` owns the ERB/complex normalization, parameter-free cascade/alpha composition and the attenuation limit. The
accelerator only needs to emit `(erb_mask, coefs, alpha)`. Run `make -C .. test-simd` to compare the default NEON path against the
scalar reference; `make -C .. SIMD=0 test` forces the latter.

`dfn2_prepost.h/.c` is the integrator's entry point: one opaque object that wraps `dfn2_process.c` and `dfn2_model_io.c` behind
the same lifecycle and per-hop shape as the AIAEC classes, described once in `../../AIAEC/README.md` ("C pre/post-processing").
Specific to this one:

- the `DFN2Grid` in its config decides everything that depends on the signal grid or the lookahead pair;
- with `frames_per_call` 1, `pre_process_freq` returns 1 on every hop. The first `mask_lookahead` hops (M) are warm-up frames (the graph's
  right-hand neighbours are not there yet): the accelerator runs on them, `frame_finish()` / `frame_bypass()` discard their heads,
  zero the registered state, advance the state generation by one and return 0, so the first real frame starts from zero state and
  the audio is as if the accelerator had not run. `dfn2_model_run_frame()` still returns 1 for a warm-up frame whose `infer()`
  succeeded. For N > 1 frames per call it keeps the older flow, see [Multi-frame export](reference/multiframe.md); with M = 0 there is no
  warm-up frame;
- the spectra are `torch.stft(normalized=True)` on the instance's grid (`dfn2_stft_scale(grid.n_fft)` = 1/sqrt(n_fft); 1024 points on
  the shipped 48 kHz grid), so chaining an AIAEC spectrum in is a sqrt(n_fft) scale error (`dfn2_stft_unscale(grid.n_fft)`, 32x on
  the shipped grid; see `dfn2_prepost.h`);
- the pool is the caller's (`dfn2_prepost_get_mem_size()`, memory, `dfn2_prepost_init_ex()`) and holds the feature windows, rings,
  normaliser state and output staging, not the recurrent state;
- the two recurrent state tensors (`h_gru`, `df_convp_history`) belong to the application. A float host registers a (pointer, element
  count) pair for each in `DFN2PrepostConfig` (all pairs or none): the instance zeroes them at init and reset, finite-checks them at
  every commit and hands out those very addresses, so the runtime binds each state output (`gru_hidden_next`,
  `df_convp_window_next`) to the same address as its input and the output of one invocation is the input of the next with no copy;
  it must read all state inputs before it writes any state output. A board registers nothing (NULL and 0 in the views): its
  accelerator holds the state;
- a commit that finds a non-finite head or state value zeroes the registered tensors, advances the state generation and returns -1
  with the frame still open; `dfn2_prepost_frame_bypass()` then passes the noisy spectrum through;
- `dfn2_prepost_state_generation()` is 1 after init and advances on every `dfn2_prepost_reset()`, every warm-up frame and every refused
  commit of a real frame (modulo 2^32), whether or not state is registered: the accelerator's fixed-point copy of the state is not reached by either event, so a
  runtime that sees a different value than at the previous frame refills its state tensors with quantized zeros.

Its gate is `../tests/test_dfn2_prepost_c.py`; `make -C .. lib` ships it in `libainr_prepost.a` (`make -C .. print-lib-path` prints
the configuration-keyed location).

Use `dfn2_enhance_spectrum_stream()` for the accelerator boundary, not the aligned `dfn2_enhance_spectrum()` reference. At wall frame
`n`, the returned head describes `n-mask_lookahead`; the cascade can emit only after that head has also masked the future DF source.
Therefore the shipped `mask_lookahead=1` and `df_lookahead=1` have a total streaming delay of **two hops** (21.33 ms on the shipped
48 kHz grid), even though each individual knob is one. The `DFN2PrepostConfig` carries the `DFN2Grid` by value,
`dfn2_prepost_grid()` returns the instance's copy and `dfn2_prepost_latency_frames(grid)` (`mask_lookahead + df_lookahead +
frames_per_call - 1`) the model's delay in hops. The C test uses changing masks and nontrivial future taps so an off-by-one
head/spectrum pairing cannot pass as a merely finite output.

## Graph port order

Every exported graph lists its ports in one order, and the board samples bind by position, never by name (`stream/ainr` keeps no
port list: `AinrWork` carries the block's signals and heads, the graph's wiring and state shapes are the sample's model configuration,
`stream/ainr/ai3_model_io_dfn2.h`, shared helper `stream/common/ai3_model_io.h`): inputs are the signal inputs, then the state inputs; outputs are the head outputs, then the state outputs; state `k`
is input `signal_in + k` and output `head_out + k`, with equal shapes at both ends. Full: `erb`, `spec` | `h_gru`,
`df_convp_history` in; `erb_mask`, `df_coefs`, `df_alpha` | `h_gru_out`, `df_convp_history_out` out (2 signals, 3 heads, 2 states).
Mask-only: `erb`, `spec` | `h_gru` in; `erb_mask` | `h_gru_out` out (2 signals, 1 head, 1 state). Fixed-point state ports need
`zero_point` 0 at both ends (the sample's `stream_check_ports()` refuses a graph that does not). `../tests/test_dfn2_port_order.py` asserts
this order on a fresh export (full and mask-only, N = 1 and 8) and compares the `AinrWork` views of real pre calls with the exported
signal inputs and heads at the registered grids.

## Lookahead pairs

`mask_lookahead` (M, 0..2) and `df_lookahead` (D, 0..4, i.e. below `df_order`)
are independent and part of the trained contract (`config.ini [model]`; a
checkpoint is refused under another pair). All 15 pairs export and run; (1,1)
is the shipped one. The shared rules (window, guards, tests) are in
[shared DFN lookahead contract](../reference/training.md#dfn-lookahead).
Specific to DFN2:

- In C the pair is two fields of `DFN2GridSpec` (`mask_lookahead`, `df_lookahead`), set when the application calls
  `dfn2_grid_make()`. Nothing is selected at build time and there is no link-time pair symbol: a graph description
  of another pair is refused by the model-I/O descriptor, a pool of another pair by the pool hash, and a pair outside the
  ranges by `DFN2_GRID_ERR_MASK_LOOKAHEAD` / `DFN2_GRID_ERR_DF_LOOKAHEAD`.
- DFN2 is a cascade: the deep filter reads the ERB-masked spectrum, so the
  output is `M + D` frames behind the input and the end-to-end latency is
  `1 + M + D` hops (three for the shipped pair). The first M hops are warm-up
  frames (the accelerator runs, its heads are discarded and the state restarts); the first M + D hops emit nothing (`valid` 0).
- The rings hold `grid.ring = max(df_order, M + D + 1)` frames: five up to
  `M + D = 4`, six for `M + D = 5` (1/4, 2/3), seven for `M + D = 6` (2/4). The state is carved to that ring at init, so a
  deeper pair only makes the pool larger (92,608 bytes at 1/1, 104,640 at 2/3, 116,672 at 2/4 on the 48 kHz grid; 64-bit host).
  Mask-only keeps only the noisy ring, `M + 1` frames.

## Integration reference

- [PCM stream SDK, state synchronization and one-hop calls](../../docs/html/models/dfn2/index.html)
- [Grid selection](reference/grid.md) and [frequency-domain C integration](reference/frequency-api.md)
- [ONNX, ERB files and calibration](reference/export.md)
- [Multi-frame export and mask-only graphs](reference/multiframe.md)
- [State ownership and diagnosing low-frequency suppression](reference/state.md)
