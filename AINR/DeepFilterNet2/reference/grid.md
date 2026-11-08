# Choosing the grid

**The library holds no grid.** What a trained model fixes is a `DFN2GridSpec`; the application builds the `DFN2Grid` with
`dfn2_grid_make()` and passes it to every object it creates (`dfn2_state_init`, `dfn2_model_io_init`,
`dfn2_prepost_config_defaults`, the model-I/O descriptor). The same binary runs any grid the maker accepts: there is no
compile-time capacity, the pool is sized from the grid at run time. The grid, the trained model and its ERB matrices must
belong together.

The shipped model's spec is the `SHIPPED` initializer of the example under [Calling the library](frequency-api.md#calling-the-library);
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

[返回本主題](../README.md)
