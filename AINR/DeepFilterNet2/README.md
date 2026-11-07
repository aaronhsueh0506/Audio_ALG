# DeepFilterNet2

DFN2 branch with cascade/alpha output composition. The shipped grid is 48 kHz; the C library holds no grid, the
application builds one and passes it to every object (see [Choosing the grid](#choosing-the-grid)).

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
`spec_scale = grid.stft_scale` (1/sqrt(n_fft)) and `ispec_scale = grid.stft_unscale` (sqrt(n_fft)). A wrong window or scale is not
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
  succeeded. For N > 1 frames per call it keeps the older flow, see [Multi-frame export](#multi-frame-export); with M = 0 there is no
  warm-up frame;
- the spectra are `torch.stft(normalized=True)` on the instance's grid (`grid.stft_scale` = 1/sqrt(n_fft); 1024 points on the shipped
  48 kHz grid), so chaining an AIAEC spectrum in is a sqrt(n_fft) scale error (`grid.stft_unscale`, 32x on the shipped grid; see
  `dfn2_prepost.h`);
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

## Lookahead pairs

`mask_lookahead` (M, 0..2) and `df_lookahead` (D, 0..4, i.e. below `df_order`)
are independent and part of the trained contract (`config.ini [model]`; a
checkpoint is refused under another pair). All 15 pairs export and run; (1,1)
is the shipped one. The shared rules (window, guards, tests) are in
[`../README.md`](../README.md#lookahead-pairs-deepfilternet2-and-deepfilternet3).
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

## Choosing the grid

**The library holds no grid.** What a trained model fixes is a `DFN2GridSpec`; the application builds the `DFN2Grid` with
`dfn2_grid_make()` and passes it to every object it creates (`dfn2_state_init`, `dfn2_model_io_init`,
`dfn2_prepost_config_defaults`, the model-I/O descriptor). The same binary runs any grid the maker accepts: there is no
compile-time capacity, the pool is sized from the grid at run time. The grid, the trained model and its ERB matrices must
belong together.

The shipped model's spec is the `SHIPPED` initializer of the example under [Calling the library](#calling-the-library);
`dfn2_grid_make(&grid, &spec)` returns `DFN2_GRID_OK` or the first failing status (`dfn2_grid_status_text()` names it).

**Given and derived.** The spec holds what a trained model fixes; the rest follows by formula (`dfn2_grid.h`):

| spec field | source |
|---|---|
| `sample_rate`, `n_fft`, `n_erb`, `df_bins`, `df_order`, `mask_lookahead`, `df_lookahead` | `config.ini` `[signal]` (`sr`, `n_fft`) and `[model]` (the same names) |
| `frames_per_call` | heads per accelerator call (`export_onnx.py --frames-per-call`, 1 for the shipped graph) |
| `erb_norm_tau_sec`, `erb_norm_init_lo_db`, `erb_norm_init_hi_db`, `erb_norm_scale_db`, `spec_norm_tau_sec`, `spec_norm_init_lo`, `spec_norm_init_hi` | `config.ini` `[feature]` (the same names); `calibrate_norm_init.py` estimates them for another grid; they must equal the training values |

| derived field | value |
|---|---|
| `hop`, `n_bins` | `n_fft / 2`, `n_fft / 2 + 1` (audiolib 50 % overlap sqrt-Hann framing) |
| `input_frames` | `frames_per_call + 2` (`DFN2_MODEL_INPUT_FRAMES` is 3) |
| `stft_scale`, `stft_unscale` | float32 `1 / sqrt(n_fft)` and `sqrt(n_fft)`: the `torch.stft(normalized=True)` factor, also `train.py`'s `analysis_scale()` for win = n_fft, hop = n_fft / 2 |
| `erb_norm_alpha`, `spec_norm_alpha` | `dfn2_norm_alpha(sample_rate, hop, tau_sec)` = `train.py` `make_norm_alpha`: `exp(-(hop / sr) / tau)` rounded to three decimals, one more digit while that reaches 1.0. 0.989 for hop 512 at 48 kHz, 0.992 for hop 128 at 16 kHz and hop 256 at 32 kHz (tau 1.0 s). Derived, never given |
| `df_history`, `ring` | `df_order - df_lookahead - 1` past deep-filter taps; ring depth `max(df_order, mask + df + 1)` (mask-only `mask + 1`) |

**Validity** (`dfn2_grid_make()` returns the first failing check, in this order, and leaves the grid untouched;
`dfn2_grid_validate()` runs the same checks on an existing grid and also refuses a hand-edited derived field):

| status | rule |
|---|---|
| `DFN2_GRID_ERR_NULL` | NULL pointer |
| `DFN2_GRID_ERR_SAMPLE_RATE` | `sample_rate > 0` |
| `DFN2_GRID_ERR_N_FFT` | a size audiolib's framing builds (`ac_stft_get_mem_size()` is not 0: a power of two in [16, 8192]) with `n_bins <= 65535` (16-bit ERB spans) |
| `DFN2_GRID_ERR_N_ERB` | a positive multiple of 4 (the ERB kernel runs bands in fours and the encoder halves the band axis twice; `train.py` asks for the same), every band at least two bins: `2 * n_erb <= n_bins` |
| `DFN2_GRID_ERR_DF_BINS` | in [2, n_bins), up to n_bins with `-DDFN2_MASK_ONLY` (bins above the deep filter keep the masked spectrum) |
| `DFN2_GRID_ERR_DF_ORDER` | in [1, `DFN2_GRID_DF_ORDER_MAX` = 16]; with `-DDFN2_MASK_ONLY` any value in [0, 16], stored as 0 |
| `DFN2_GRID_ERR_MASK_LOOKAHEAD` | in [0, `DFN2_GRID_MASK_LOOKAHEAD_MAX` = 2] |
| `DFN2_GRID_ERR_DF_LOOKAHEAD` | in [0, `df_order` - 1]; 0 with `-DDFN2_MASK_ONLY` |
| `DFN2_GRID_ERR_FRAMES_PER_CALL` | in [1, `DFN2_MODEL_MAX_FRAMES_PER_CALL` = 64] |
| `DFN2_GRID_ERR_ERB_NORM`, `DFN2_GRID_ERR_SPEC_NORM` | tau finite and > 0, starts finite (spec starts and the ERB scale > 0), and the alpha it gives strictly between 0 and 1 as float32: a rate or tau that rounds the alpha to 0 or 1 is refused |
| `DFN2_GRID_ERR_INCONSISTENT` | (`dfn2_grid_validate()`) a derived field differs from what `dfn2_grid_make()` derives, or an alpha is not in (0, 1) |

Status -8 is unassigned and is never reused. Every library entry point that takes a grid runs `dfn2_grid_validate()` and refuses
an invalid one (-1 or NULL).

**RAM.** The pool follows the grid and the pair (`dfn2_prepost_get_mem_size()`, 64-bit host, bytes). The application's `AcStft` /
`AcIstft`, the ERB matrices and the registered recurrent state are not in the pool:

| grid (board table pair) | full | mask-only |
|---|---|---|
| 16 kHz, n_fft 256, n_erb 24, pair 0/0 | 47,744 | 10,912 |
| 32 kHz, n_fft 512, n_erb 28, pair 0/0 | 62,784 | 16,736 |
| 48 kHz, n_fft 1024, n_erb 32, pair 1/1 (mask-only 1/0) | 92,608 | 32,224 |

On the 48 kHz grid a deeper ring grows the full pool: 104,640 bytes at 2/3 (ring 6) and 116,672 at 2/4 (ring 7); `frames_per_call` 4
gives 144,112. The recurrent state the shipped graph needs is 1,280 + 30,720 floats (`h_gru` and `df_convp_history`), the application's.

**What stays compile time**: `-DDFN2_MASK_ONLY` (every translation unit; the variant of the state and of the accelerator boundary
structs) and `DFN2_FEATURE_VERSION`. The graph geometry (GRU rows and width, encoder channels, pathway window) is not in the
library: it is what the application registers or the board's accelerator holds, and the descriptor's six interior fields are
zero from `dfn2_model_io_descriptor_default()` and never read.

**Link symbol and pool identity.**

- `dfn2_process.c` defines one symbol and the other units reference it through `DFN2_REQUIRE_VARIANT_LINK()`:
  `dfn2_variant_full` or `dfn2_variant_mask_only`. Objects built with a different variant (`-DDFN2_MASK_ONLY` on some units only)
  do not link; the grid and the pair are not in it. Under a `.c` -> `.o` rule objects of other flags share names: keep one object
  directory per flag set (`make -C AINR lib` does), or clean between builds.
- `DFN2_PREPOST_CARVE_VERSION` is 11. The pool hash (`build_flags_hash`, FNV-1a-32) folds the whole grid: sample rate, n_fft, hop,
  n_bins, n_erb, df_bins, `df_order`, both lookaheads, history, ring, `frames_per_call`, `input_frames`, both time constants and
  alphas, the two scales and the five normaliser starting points, plus the variant, the layout and carve versions and the feature
  version. A `DFN2PrepostMemReq` recorded by another grid or build is refused by `dfn2_prepost_init_ex`; query the size again.

**Board samples.** The board samples (`audio_sample/`, outside this repo) own the table from (sample rate, FFT size) to a
`DFN2GridSpec` row, designated initialisers passed to `dfn2_grid_make()`; a row carries the model's numbers (`df_order`,
`frames_per_call` 1 and the seven normaliser values). The library only validates what they pass. The three rows below are the table of
the standalone samples; `alg_ai3_aec_dfn.c` and the pipeline example mains (`pipelines/dfn/dfn_example_args.h`) keep two of them,
without 32000 / 512, because the AEC in front of the model has no 32 kHz grid; the mask-only sample differs, in its df lookahead (see
its row in the second table):

| model sample rate | FFT size | n_erb | df_bins | mask / df lookahead | state |
|---|---|---|---|---|---|
| 16000 | 256 | 24 | 96 | 0 / 0 | quality unmeasured |
| 32000 | 512 | 28 | 96 | 0 / 0 | provisional, no model trained |
| 48000 | 1024 | 32 | 96 | 1 / 1 | shipped |

`--n-erb` and `--df-bins` are never flags (the row gives them), and a pair outside the table is refused with the list of accepted
ones. Defaults are 48000 and 1024.

| sample | how the grid is chosen |
|---|---|
| `alg_ai3_dfn.c` | `--sample-rate <Hz>` and `--fft-size <n>`: the stream's rate is the model's rate. A token that begins `--sample` or `--fft` and is not one of these (`--sample-rate=16000`, `--samplerate`) prints the usage and exits 2 instead of running the default grid |
| `alg_ai3_dfn_lite.c` | the same flags and the same rule, with the mask-only table: `df_order` is 0 and `df_lookahead` is 0 in every row (48000 / 1024 is 1 / 0). The full model and the mask-only model are two separate samples; the lite one builds every library unit with `-DDFN2_MASK_ONLY`, in an object directory of its own |
| `alg_ai3_aec_dfn.c` | `--ai-sample-rate <Hz>` and `--ai-fft-size <n>` choose the model grid; `--sr` stays the host rate (when absent it follows the model's rate). Host rate equal to the model's: `DfnResStage` drives the model directly. Any other rate goes through `DfnRateBridge`, which needs the 48000 / 1024 model and a host at 8000 or 16000, and is otherwise refused with a message before the NPU is set up (exit 1). Its table has no 32000 / 512 row (its AEC has no 32 kHz grid), so that pair is refused with the accepted list (exit 2). A misspelt token that begins `--ai` exits 2 |
| `uvc_ai_nr_stream.c` | 48 kHz only (USB audio is 48 kHz; a native 16 kHz UVC path is out of scope) and no flags. It builds its grid with `dfn2_grid_make()` from its own one-row constants (48000, 1024, 32, 96, 1 / 1) and drives the `DFN2Prepost` class directly: the grid goes to `dfn2_prepost_config_defaults()`, the tensor tables and the ERB file name, and its static per-hop buffers are checked against `grid.hop` and `grid.n_bins` at run time |
| `uvc_audio_live2a.c` | 48 kHz only, no flags, the same one-row grid built with `dfn2_grid_make()`; it goes into the `DfnResStage` / `DfnRateBridge` configs, the model-I/O descriptor and the tensor tables (the host rate is the model's or goes through the bridge) |

The samples own the recurrent-state shapes (constants of the sample, the library knows none): `h_gru` `[1,5,1,256]` (mask-only
`[1,3,1,256]`) and `df_convp_history` `[1,64,5,df_bins]`. They register nothing: the accelerator holds the state, each state input
and its `*_out` output are bound to one buffer. Before every accelerator run a sample compares the library's state generation
(`dfn2_prepost_state_generation()`; `dfn_rate_bridge_state_generation()` or `dfn_res_stage_state_generation()` when a stage or
bridge hosts the model) with the last value it saw and, when it differs, reloads quantized zeros into the state input buffers. A
sample refuses to start when the element count the loaded model reports for a state tensor differs from its own table.

**The ERB matrices.** The library reads no file. It borrows two `const float` pointers, forward bin-major `[n_bins][n_erb]` and
inverse band-major `[n_erb][n_bins]` (raw float32, `n_bins * n_erb * 4` bytes each: 65,664 on the 48000 / 1024 / 32 grid), which the
application loads, owns and keeps alive while an object uses them; the inverse is a partition of unity. The producer is
`export_erb_matrix.py`:

| output | content |
|---|---|
| `--runtime-bins DIR` | `erb_fwd.bin` (bin-major) and `erb_inv.bin` (band-major), raw little-endian float32 without a header, plus `erb_matrices.json`: the two layouts the library takes |
| `--output-dir DIR --format bin` (also written by `--format all`) | `erb_matrix.bin`: `int32 n_bins`, `int32 n_bands`, `float32` forward `[n_bins][n_bands]`, `float32` inverse `[n_bins][n_bands]`, little-endian, 8 + 2 * 65,664 bytes on the 48000 / 1024 / 32 grid; both matrices are bin-major, so a loader transposes the inverse to band-major |

The pipeline example mains take the `--runtime-bins` pair as `--erb-fwd FILE` and `--erb-inv FILE` (both or neither; each file holds exactly
`n_bins * n_erb` floats; without them a unit ERB partition runs, which serves the fail-open smoke run only, and the main prints a notice).
Every DFN2 board sample reads `/mnt/sd/dfn2_blobs/fb_<sr>_<n_erb>.bin` (for example `fb_48000_32.bin`; `<sr>` is the model grid's rate,
not the host's), one deployment set for all of them: the exporter's `erb_matrix.bin` renamed for its grid. A sample checks the header
against the grid's `n_bins` and `n_erb` and transposes the inverse to band-major once at init. This repo holds no board run of these
samples.

**Tests and pipelines.** The tests choose their grid when they are built, through a header of their own, not through the library:
`AINR/tests/dfn2_test_grid.h` and `pipelines/dfn/tests/dfn_test_grid.h` (`-DDFN2_GRID_KHZ=16|32|48`, with `-DDFN2_MASK_LOOKAHEAD` /
`-DDFN2_DF_LOOKAHEAD`; `dfn2_test_grid()` and `dfn_test_grid()` return the `DFN2Grid`). They give the macros that size the tests'
fixed arrays and sweep grids and pairs; they are test scaffolding only, and the pair they default to is 1/1 for every preset, which is
not the board table's 0 / 0 for 16 and 32 kHz. The pipeline objects (`dfn_res_stage`, `dfn_rate_bridge`, `dfn_hop`) and the example
mains hold no build-time grid: the `DFN2Grid` travels in `DfnResStageConfig.grid`, and the example mains build it from
`--ai-sample-rate` / `--ai-fft-size` and the table above.

**Limits.** The rate bridge (`pipelines/dfn/dfn_rate_bridge.c`) is written for the 48 kHz model and refuses another (see
`pipelines/dfn/README.md`); the pipelines and their Python twin are verified on the shipped grid only. Shapes quoted elsewhere in
this README (`(1,1,3,32)` and so on) are the shipped grid's. Tests: `../tests/test_dfn2_grid.py` (every refusal and status code, the
derived fields, `make` agreeing with `validate` over a sweep of sample rates, `dfn2_norm_alpha` against `train.py` bit for bit, the
overflow-free `n_erb` check under UBSan, C against Python from one binary on every grid), `../tests/test_dfn2_runtime_grid.py` (the
whole chain on every grid and pair, pool identity against an independent Python mirror of the layout, the normaliser fields, no
grid in the library, the variant link symbol), `../tests/test_dfn2_prepost_c.py`, `../tests/test_c_prepost.c`,
`../tests/test_nn_chains.c`.

## Calling the library

The order, for one mono stream (hop in, hop out; the application frames with audiolib). Nothing in the library knows a grid until
the application gives it one, so step 1 comes first and its `DFN2Grid` goes to every later call.

1. `dfn2_grid_make()` from the application's table row (or `dfn2_grid_validate()` on a grid it already holds).
2. Load the ERB matrices of that model (the layouts are described under [Choosing the grid](#choosing-the-grid); the library never
   reads files and borrows the pointers).
3. The framing: `ac_stft_config_default(&sc, grid.n_fft)` with `sc.spec_scale = grid.stft_scale` and `sc.ispec_scale =
   grid.stft_unscale`, then `ac_stft_get_mem_size()` / `ac_stft_init()` and `ac_istft_get_mem_size()` / `ac_istft_init()`.
4. `dfn2_prepost_config_defaults(&cfg, &grid)`, set `erb_fwd`, `erb_inv` (and `atten_lim_db` when wanted) and, on a float host, the
   state registration; `dfn2_prepost_get_mem_size()`, a 16-byte aligned pool of `req.bytes`, `dfn2_prepost_init_ex()` (a stale `req`
   is refused).
5. `dfn2_model_io_descriptor_default(&grid, &d)`: the graph contract of this build for the grid, to compare with what the accelerator
   runtime reports (`dfn2_model_io_descriptor_validate(&grid, &d)`); a model exported for another grid or lookahead pair fails it.
6. Per hop, the same flow on every hop (`frames_per_call` 1): `ac_stft_push_planar()`, `dfn2_prepost_pre_process_freq()` (returns 1),
   `dfn2_prepost_frame_open()`, the model, `dfn2_prepost_frame_finish()` (`dfn2_prepost_frame_bypass()` when the model failed or the
   commit was refused); then `dfn2_prepost_post_process_freq()` and `ac_istft_push_planar()`; `valid` is informational (the spectrum
   is all zero while it is 0) and the argument may be NULL. `dfn2_model_run_frame(&model, p)` does the middle three for a `DFN2Model`
   that carries the descriptor. With `frames_per_call` > 1 the flow is the older one, see [Multi-frame export](#multi-frame-export).
7. Flush: push `M + D + 1` zero audio hops (`M + D` zero spectra, one more for the overlap-add) with the accelerator running on all of them.

```c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "audio_stft.h"
#include "dfn2_prepost.h"

/* The shipped model: config.ini [signal] [model] [feature]. */
static const DFN2GridSpec SHIPPED = {
    .sample_rate = 48000, .n_fft = 1024, .n_erb = 32, .df_bins = 96, .df_order = 5,
    .mask_lookahead = 1, .df_lookahead = 1, .frames_per_call = 1,
    .erb_norm_tau_sec = 1.0f, .erb_norm_init_lo_db = -60.0f, .erb_norm_init_hi_db = -90.0f,
    .erb_norm_scale_db = 40.0f, .spec_norm_tau_sec = 1.0f,
    .spec_norm_init_lo = 0.001f, .spec_norm_init_hi = 0.0001f,
};

/* State shapes belong to the application: the exported graph's h_gru [1,5,1,256] and
 * df_convp_history [1,64,5,df_bins]. */
enum { GRU_ELEMENTS = 5 * 256, CONVP_CHANNELS = 64, CONVP_WINDOW = 5 };

static void *alloc16(size_t bytes) { return aligned_alloc(16, (bytes + 15) & ~(size_t)15); }

/* fb_<sr>_<n_erb>.bin = the exporter's erb_matrix.bin: int32 n_bins, int32 n_bands, float32
 * fwd[n_bins][n_bands], float32 inv[n_bins][n_bands] (both bin-major). The library wants the
 * inverse band-major [n_erb][n_bins]: transpose it once. Returns 0, or -1. */
static int load_erb(const char *path, const DFN2Grid *g, float *fwd, float *inv_band_major)
{
    size_t n = (size_t)g->n_bins * (size_t)g->n_erb;
    int32_t hdr[2];
    float *inv = malloc(n * sizeof *inv);
    FILE *f = fopen(path, "rb");
    int ok = f && inv && fread(hdr, sizeof hdr[0], 2, f) == 2 &&
             hdr[0] == g->n_bins && hdr[1] == g->n_erb &&
             fread(fwd, sizeof *fwd, n, f) == n && fread(inv, sizeof *inv, n, f) == n;
    for (int k = 0; ok && k < g->n_bins; ++k)
        for (int b = 0; b < g->n_erb; ++b)
            inv_band_major[(size_t)b * g->n_bins + k] = inv[(size_t)k * g->n_erb + b];
    if (f) fclose(f);
    free(inv);
    return ok ? 0 : -1;
}

/* The accelerator: reads the windows in `in`, fills out->erb_mask / coefs / alpha and updates the
 * two state tensors in place (the state outputs alias the state inputs: read every state input
 * before writing any state output). 0 when everything was written. */
typedef int (*RunModel)(const DFN2PrepostInputs *in, DFN2PrepostOutputs *out);

/* One mono stream, `hops` hops of grid.hop samples from `in` to `out`. Returns 0, or -1. */
static int run_dfn2(const char *erb_path, RunModel run_model, const float *in, float *out, long hops)
{
    DFN2_REQUIRE_VARIANT_LINK();           /* units built with another -DDFN2_MASK_ONLY do not link */

    /* 1. the grid: what the model fixes, the rest derived and validated */
    DFN2Grid grid;
    int status = dfn2_grid_make(&grid, &SHIPPED);
    if (status != DFN2_GRID_OK) {
        fprintf(stderr, "grid refused: %s\n", dfn2_grid_status_text(status));
        return -1;
    }

    int rc = -1;
    size_t erb_n = (size_t)grid.n_bins * (size_t)grid.n_erb;
    size_t convp_elements = (size_t)CONVP_CHANNELS * CONVP_WINDOW * grid.df_bins;
    float *erb_fwd = malloc(erb_n * sizeof *erb_fwd), *erb_inv = malloc(erb_n * sizeof *erb_inv);
    float *gru = calloc(GRU_ELEMENTS, sizeof *gru);                      /* the application's state */
    float *convp = calloc(convp_elements, sizeof *convp);
    float *re = malloc(4 * (size_t)grid.n_bins * sizeof *re);            /* spectra of one hop */
    float *im = re + grid.n_bins, *out_re = im + grid.n_bins, *out_im = out_re + grid.n_bins;
    void *stft_mem = NULL, *istft_mem = NULL, *pool = NULL;
    AcStft *stft = NULL;
    AcIstft *istft = NULL;
    DFN2Prepost *p = NULL;
    if (!erb_fwd || !erb_inv || !gru || !convp || !re) goto done;
    if (load_erb(erb_path, &grid, erb_fwd, erb_inv) != 0) goto done;   /* 2. the ERB matrices (caller-owned, borrowed) */

    /* 3. the framing: audiolib, the scales from the grid */
    AcStftConfig sc;
    ac_stft_config_default(&sc, grid.n_fft);
    sc.spec_scale = grid.stft_scale;
    sc.ispec_scale = grid.stft_unscale;
    size_t stft_bytes = ac_stft_get_mem_size(&sc), istft_bytes = ac_istft_get_mem_size(&sc);
    stft_mem = alloc16(stft_bytes);
    istft_mem = alloc16(istft_bytes);
    if (!stft_mem || !istft_mem) goto done;
    stft = ac_stft_init(stft_mem, stft_bytes, &sc);
    istft = ac_istft_init(istft_mem, istft_bytes, &sc);
    if (!stft || !istft) goto done;

    /* 4. config (copies the grid), exact pool size, init inside caller memory */
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    if (dfn2_prepost_config_defaults(&cfg, &grid) != 0) goto done;
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;
    cfg.atten_lim_db = 0.0f;               /* 0 = no attenuation limit */
    cfg.gru_hidden = gru;                  /* a float host registers its state; a board registers nothing */
    cfg.gru_hidden_elements = GRU_ELEMENTS;
    cfg.df_convp_history = convp;
    cfg.df_convp_history_elements = convp_elements;
    if (dfn2_prepost_get_mem_size(&cfg, &req) != 0) goto done;
    pool = alloc16(req.bytes);
    p = pool ? dfn2_prepost_init_ex(pool, (size_t)req.bytes, &cfg, &req) : NULL;
    if (!p) goto done;

    /* 5. the graph the accelerator runtime was built against. A board fills it from the exported
     *    model's metadata; _default() is this build's contract for the grid, and _validate()
     *    refuses a description of another grid, pair or frame count. */
    DFN2ModelIoDescriptor desc;
    if (dfn2_model_io_descriptor_default(&grid, &desc) != 0 ||
        dfn2_model_io_descriptor_validate(&grid, &desc) != 0) goto done;

    /* 6. the same flow on every hop (N == 1): analysis, pre_process_freq, frame_open -> model ->
     *    frame_finish, post_process_freq, synthesis. The first mask_lookahead hops are warm-up frames:
     *    the model runs, its heads are discarded, the state restarts. dfn2_model_run_frame() does the
     *    middle three for a DFN2Model that carries the descriptor. */
    rc = 0;
    for (long h = 0; h < hops && rc == 0; ++h) {
        ac_stft_push_planar(stft, in + h * grid.hop, re, im);
        if (dfn2_prepost_pre_process_freq(p, re, im) != 1) { rc = -1; break; }   /* 1 on every hop */
        DFN2PrepostInputs xin;
        DFN2PrepostOutputs xout;
        if (dfn2_prepost_frame_open(p, &xin, &xout) != 0) { rc = -1; break; }
        /* dfn2_prepost_state_generation(p) changed since the last frame (a reset, a warm-up frame or a
         * refused commit zeroed the state): an accelerator with its own state copy refills it first. */
        if (run_model(&xin, &xout) != 0 || dfn2_prepost_frame_finish(p) != 0)
            dfn2_prepost_frame_bypass(p);              /* the noisy spectrum, clocks advance */
        if (dfn2_prepost_post_process_freq(p, out_re, out_im, NULL) != 0) { rc = -1; break; }   /* zeros for the first M + D hops */
        ac_istft_push_planar(istft, out_re, out_im, out + h * grid.hop);                         /* every hop: AC_OLA_ACCUMULATE */
    }
done:
    free(pool); free(istft_mem); free(stft_mem); free(re); free(convp); free(gru); free(erb_inv); free(erb_fwd);
    return rc;
}
```

The spectra are `torch.stft(normalized=True)` scaled: the `AcStft` / `AcIstft` of step 3 carry `grid.stft_scale` and
`grid.stft_unscale`, so the application passes audio-scale samples. Build every unit that includes the DFN2 headers with the same
`-DDFN2_MASK_ONLY` setting as the library; `DFN2_REQUIRE_VARIANT_LINK()` turns a mismatch into a link error. With the shipped 1/1
pair `valid` is 0 for the first two hops (the output is zeros) and the end-to-end delay is three hops. The example assumes
`frames_per_call` 1 (the table row above); the other pairs are in
[Lookahead pairs](#lookahead-pairs).

## ONNX, ERB tables and calibration

Install the optional export dependencies from `../requirements-export.txt`.
The ONNX graph contains only the learned heads; STFT and WOLA are the application's audiolib framing, feature
normalization and building the enhanced spectrum from the heads remain in `dfn2_process.c/.h`.

```bash
python3 export_onnx.py --model output/dfn2_best.pth \
  --output output/dfn2_stream.onnx --verify
python3 export_erb_matrix.py --model output/dfn2_best.pth \
  --output-dir output/erb --format all
python3 inference.py calib --model output/dfn2_best.pth \
  --wav-dir /path/to/noisy_wavs --frames 8192 --format bin \
  --output calib/dfn2
```

The library (the config's `erb_fwd` / `erb_inv`, `dfn2_set_erb_matrices`, `dfn2_prepost_set_erb_matrices`) takes pointers to the
two matrices and reads no file: forward bin-major, inverse band-major, the layouts of `export_erb_matrix.py --runtime-bins DIR`
(`erb_fwd.bin`, `erb_inv.bin`), which the pipeline example mains read through `--erb-fwd` and `--erb-inv`. The board samples read the
headered `erb_matrix.bin` of `--output-dir DIR --format bin`, renamed `fb_<sr>_<n_erb>.bin` (see "The ERB matrices" under
[Choosing the grid](#choosing-the-grid)); the other `--output-dir` formats are for toolchain validation.

The graph is stateless from the accelerator's perspective. Each invocation
receives exactly three feature frames `[t-(2-M) .. t+M]` (`[t-1,t,t+1]` for the
shipped `mask_lookahead` M = 1), emits the heads for frame `t`, and returns
two state tensors: `h_gru_out` `(1,5,1,256)` and `df_convp_history_out`
`(1,64,5,96)`. Each state input and its output have the
same shape and are bound to one address. `h_gru` holds the five GRU hidden
rows in graph order -- encoder (row 0), ERB decoder (rows
1-2), DF decoder (rows 3-4). `df_convp_history` is the
whole causal window of the five-frame DF residual path: the graph drops the
oldest frame with a Slice on the input, appends the new frame and returns the
whole concatenation. That window is required because the input kernel sees
three frames while the DF residual path has a causal five-frame kernel;
omitting it would silently reduce the trained receptive field. CPU-side
feature-window storage is defined by
`dfn2_model_io.c/.h` (the windows are carved from the caller's block at the grid's stride, `dfn2_model_io_mem_size(grid)` bytes);
the recurrent state is not: `dfn2_model_io_init()` registers the application's buffers, or none. The header also exports the
window-slide and state-validate helpers so an external dual-input consumer with its own state struct reuses
this window/validate discipline instead of inlining a second copy of the
memmove/memcpy pair. `dfn2_model_io_validate_arrays(gru_hidden, gru_count, df_convp_history, convp_count)` takes the flat buffers
and their element counts (`df_convp_history` count 0 = absent; `64 * 5 * df_bins` floats, the graph tensor element for element) and
`dfn2_model_io_validate_state()` the registered ones. Validation is a finite
check over the state itself: a null argument returns `-1`, and a non-finite
element zeroes the two arrays and returns `-1` (`dfn2_model_io_validate_state()`
also advances `dfn2_model_io_state_generation()`). A finite but partial or mixed
accelerator write is not detectable. The exported metadata carries
`state_layout_version`, kept numerically equal to `DFN2_MODEL_IO_LAYOUT_VERSION`
in that header, so an integrator can refuse a graph whose state layout no longer
matches the struct it allocated. `inference.py calib --frames N` captures `N`
streaming invocations with real non-zero state. `calibrate_norm_init.py` is a
different tool: it estimates the feature EMA initialization constants that
become the `erb_norm_init_*` and `spec_norm_init_*` fields of `DFN2GridSpec`.
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

The C host takes N as `DFN2GridSpec.frames_per_call` (1..64, a run-time field; `DFN2_GRID_ERR_FRAMES_PER_CALL` otherwise),
with the lookahead pair:

- the window holds `N + 2` frames; the first call is ready after `M + N`
  pushed frames, every later one after N more (`pre_process_freq` returns 1 on hops
  `M+N-1`, `M+2N-1`, ...); `dfn2_model_io_frame_slot()` names the slot of the
  last pushed frame. With N > 1 nothing of the one-flow-per-hop behaviour applies: the earlier hops return 0, there are no warm-up
  frames, and `frame_open()` is called only after a 1 (as are the model and `frame_finish()`);
- `frame_open()` publishes N heads per tensor, `frame_finish()` builds the N enhanced frames
  in one burst, and `post_process_freq` returns one hop per call from that burst,
  so the output trails by N - 1 more hops: latency `1 + M + D + N - 1` hops
  (`dfn2_prepost_latency_frames(grid)` + 1; 10 hops, 106.7 ms, at the shipped pair
  with N = 8). A refused commit zeroes the registered state and advances the
  generation, and `frame_bypass()` passes all N frames as the noisy spectrum;
  flushing means pushing zero hops until the last real one has been emitted (`M + D + N` of them);
- `DFN2ModelIoDescriptor` keeps its 18 fields and 72 bytes: N is
  `input_frames - 2` (`input_frames` = N + 2, metadata `input_feature_frames`),
  so `dfn2_model_io_descriptor_validate()` and `dfn2_model_run_frame()` refuse
  another N, and a descriptor made for the one-frame graph (`input_frames` = 3)
  is refused by a grid of N > 1;
- N is folded into the pool hash (a pool of another N is refused) and into the pool
  size (144,112 bytes at N = 4 on the shipped grid against 92,608 at N = 1; 64-bit host). The pipelines DFN stage and its Python twin are
  one-frame units, and the stage refuses N > 1.

`DFN2_MODEL_IO_LAYOUT_VERSION` stays 8: it names the state tensors, which N
does not change, and the window and head shapes carry N through `input_frames` in
the descriptor and `frames_per_call` in the metadata.

Tests: `../tests/test_dfn2_frames_per_call.py` (wrapper and exported graph
streamed in chunks against the one-frame graph, all pairs; a wrong state carry
fails; metadata and validator; calibration) and
`../tests/test_dfn2_frames_per_call_c.py` (the N-frame host against the
one-frame host, byte for byte, with a stand-in accelerator; reset, refusal, contract;
the 72-byte descriptor through a grid of either N).

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
  descriptor shrink accordingly, and the output is `noisy * (mask @ erb_inv)` of the head frame, `grid.mask_lookahead`
  frames behind the input. The grid's `df_lookahead` must be 0 (`DFN2_GRID_ERR_DF_LOOKAHEAD` otherwise), `df_order` is stored as 0, and `df_bins` may reach `n_bins`. The final attenuation-limit mix (`--atten-lim`, `dfn2_prepost_set_atten_lim`) stays and acts on
  the stage-1 output in Python and in C. `../tests/test_dfn2_mask_only_c.py` compares it with numpy.
- Board sample: `alg_ai3_dfn_lite.c` (`audio_sample/`, outside this repo) is the sample for this graph, separate from the
  full model's `alg_ai3_dfn.c`. Build it and every library unit with `-DDFN2_MASK_ONLY`, in an object directory of its own
  (objects of the other variant do not link: the link symbol is `dfn2_variant_mask_only`). It binds I/O layout 9: inputs
  `erb, spec, h_gru` (h_gru is `(1, 3, 1, 256)`), outputs `erb_mask, h_gru_out`. The model is read from
  `/mnt/sd/audio_model/<quant>/dfn2_lite/` and the output goes to `/mnt/sd/audio/output/DFN2_LITE/`. Its grid comes from
  `--sample-rate` and `--fft-size` through the sample's table (see [Choosing the grid](#choosing-the-grid)); not run on a board from this repo.

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

The library knows none of these shapes. The application registers buffers of these element counts (1,280 and 30,720 floats on the
shipped grid) or, on a board, its accelerator holds the state (see [C deployment boundary](#c-deployment-boundary)).

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

If the ERB-masked low band is healthy but the alpha-blended output collapses,
the issue is in the DF/alpha path rather than the shared dataset. If the
ERB-masked low band is already collapsed, audit low-frequency corpus coverage,
speech-only sampling, SNR distribution, and target contamination.

The upstream-alignment record is in
[`UPSTREAM_ALIGNMENT.md`](UPSTREAM_ALIGNMENT.md).
