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

`dfn2_process.h/.c` owns the streaming STFT, ERB/complex normalization,
parameter-free cascade/alpha composition, attenuation limit, and WOLA. The
accelerator only needs to emit `(erb_mask, coefs, alpha)`. Run
`make -C .. test-simd` to compare the default NEON path against the scalar
reference; `make -C .. SIMD=0 test` forces the latter.

`dfn2_prepost.h/.c` is the integrator's entry point: one opaque object that
composes `dfn2_process.c` and `dfn2_model_io.c` behind the same lifecycle and per-hop
shape as the AIAEC classes, described once in `../../AIAEC/README.md`
("C pre/post-processing"). Specific to this one: the `DFN2Grid` in its config
decides everything that depends on the signal grid or the lookahead pair;
`pre_process` returns 0 on the first `mask_lookahead` hops (the graph needs its
right-hand neighbours) and 1 after (every N-th hop in a build for N frames per
call, see [Multi-frame export](#multi-frame-export)); the
spectra at the `DFN2_IO_FREQ` boundary are torch.stft normalized=True on the
instance's grid (`grid.stft_scale` = 1/sqrt(n_fft); 1024 points on the shipped 48 kHz grid), so chaining
an AIAEC spectrum in is a sqrt(n_fft) scale error (`grid.stft_unscale`, 32x on the shipped grid; the
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
shipped 48 kHz grid), even though each individual knob is one. The
`DFN2PrepostConfig` carries the `DFN2Grid` by value, `dfn2_prepost_grid()` returns the
instance's copy and `dfn2_prepost_latency_frames(grid)` the model's delay in hops. The C test uses changing
masks and nontrivial future taps so an off-by-one head/spectrum pairing cannot
pass as a merely finite output.

## Lookahead pairs

`mask_lookahead` (M, 0..2) and `df_lookahead` (D, 0..4, i.e. below `df_order`)
are independent and part of the trained contract (`config.ini [model]`; a
checkpoint is refused under another pair). All 15 pairs export and run; (1,1)
is the shipped one. The shared rules (window, guards, tests) are in
[`../README.md`](../README.md#lookahead-pairs-deepfilternet2-and-deepfilternet3).
Specific to DFN2:

- In C the pair is two fields of the `DFN2Grid` (`mask_lookahead`, `df_lookahead`), set when the application calls
  `dfn2_grid_make()`. Nothing is selected at build time and there is no link-time pair symbol: a graph description
  of another pair is refused by the model-I/O descriptor, a pool of another pair by the pool hash, and a grid whose pair does
  not fit the capacities by `DFN2_GRID_ERR_CAPACITY`.
- DFN2 is a cascade: the deep filter reads the ERB-masked spectrum, so the
  output is `M + D` frames behind the input and the end-to-end latency is
  `1 + M + D` hops (three for the shipped pair). The first M hops only fill the
  window; the following D hops run the accelerator but emit nothing.
- The rings hold `grid.ring = max(DF_ORDER, M + D + 1)` frames: five up to
  `M + D = 4`, six for `M + D = 5`, seven for `M + D = 6` (2/4). The state is sized for the capacity
  `DFN2_MAX_DF_RING` (5 by default), so a pair that needs more (ring 6 or 7) is refused until the build raises it
  (`-DDFN2_MAX_DF_RING=7` on every unit; `tests/dfn_pairs.py` `defines()` adds it for the tests). Mask-only
  keeps only the noisy ring, `M + 1` frames, and its capacity is 2.

## Choosing the grid

**The library holds no grid.** The application builds a `DFN2Grid` with `dfn2_grid_make()` and passes it to every
object it creates (`dfn2_state_init`, `dfn2_model_io_init`, `dfn2_prepost_config_defaults`, the model-I/O descriptor,
`dfn2_framing_config`). The same binary runs any grid within the compile-time capacities below. The grid, the trained
model and its ERB matrices must belong together: the grid is the model's `config.ini [signal]` and `[model]`.

```c
DFN2Grid grid;                                   /* sample_rate n_fft n_erb df_bins mask_la df_la */
int status = dfn2_grid_make(&grid, 48000, 1024, 32, 96, 1, 1);
if (status != DFN2_GRID_OK) { fprintf(stderr, "%s\n", dfn2_grid_status_text(status)); return -1; }

DFN2PrepostConfig cfg;
dfn2_prepost_config_defaults(&cfg, DFN2_IO_TIME, &grid);       /* copies the grid */
AcStftConfig sc;
dfn2_framing_config(&sc, &grid, fft, NULL);                    /* for the AcStft / AcIstft */
```

**Given and derived.** The six given values are what a trained model fixes. The rest follows by formula
(`dfn2_grid.h`):

| field | value |
|---|---|
| `hop`, `n_bins` | `n_fft / 2`, `n_fft / 2 + 1` (audiolib 50 % overlap sqrt-Hann framing) |
| `stft_scale`, `stft_unscale` | float32 `1 / sqrt(n_fft)` and `sqrt(n_fft)`: the `torch.stft(normalized=True)` factor, also `train.py`'s `analysis_scale()` for win = n_fft, hop = n_fft / 2 |
| `erb_norm_alpha`, `spec_norm_alpha` | `dfn2_norm_alpha(sample_rate, hop, 1.0)` = `train.py` `make_norm_alpha`: `exp(-(hop / sr) / tau)` rounded to three decimals, one more digit while that reaches 1.0, tau = 1.0 s. 0.989 for hop 512 at 48 kHz, 0.992 for hop 128 at 16 kHz and hop 256 at 32 kHz. A model trained with another tau overwrites the two fields (`dfn2_norm_alpha()` computes them) |
| `df_history`, `ring` | `DF_ORDER - df_lookahead - 1` past deep-filter taps; ring depth `max(DF_ORDER, mask + df + 1)` (mask-only `mask + 1`) |

**Validity** (`dfn2_grid_make()` returns the first failing check and leaves the grid untouched;
`dfn2_grid_validate()` runs the same checks on an existing grid and also refuses a hand-edited derived field):

| status | rule |
|---|---|
| `DFN2_GRID_ERR_NULL` | NULL pointer |
| `DFN2_GRID_ERR_SAMPLE_RATE` | `sample_rate > 0`, and the norm alpha it gives (`dfn2_norm_alpha(sample_rate, hop, 1.0)`) strictly between 0 and 1 as float32: a rate so low or so high that the rounded alpha reaches 0 or 1 is refused |
| `DFN2_GRID_ERR_N_FFT` | a power of two in [16, 8192] (audiolib `audio_stft.h`) |
| `DFN2_GRID_ERR_N_ERB` | a positive multiple of 4 (the ERB kernel runs bands in fours and the encoder halves the band axis twice; `train.py` asks for the same), every band at least two bins: `2 * n_erb <= n_bins` |
| `DFN2_GRID_ERR_DF_BINS` | in [2, n_bins), up to n_bins with `-DDFN2_MASK_ONLY` (bins above the deep filter keep the masked spectrum) |
| `DFN2_GRID_ERR_MASK_LOOKAHEAD` | in [0, 2] |
| `DFN2_GRID_ERR_DF_LOOKAHEAD` | in [0, `DF_ORDER` - 1]; 0 with `-DDFN2_MASK_ONLY` |
| `DFN2_GRID_ERR_CAPACITY` | `n_fft`, `n_erb`, `df_bins` or the ring above the capacities below |
| `DFN2_GRID_ERR_INCONSISTENT` | (`dfn2_grid_validate()`) a derived field differs from what `dfn2_grid_make()` derives, or an alpha is not in (0, 1) |

Every library entry point that takes a grid runs `dfn2_grid_validate()` and refuses an invalid one (-1). `dfn2_framing_config()` returns nothing, so it zeroes the config it was given (`fft_size` 0): `ac_stft_get_mem_size()` and `ac_istft_get_mem_size()` of it are 0 and the framing objects refuse it, and the error surfaces at the next call that needs the objects.

**Capacities (compile time).** The state and the model-I/O arrays are plain structs sized for the largest grid the
build serves; a grid runs in the first part of each array (`DFN2ModelIOState` arrays are packed at the grid's stride:
`erb_window[f * n_erb + b]`, `spec_window[(c * WF + f) * df_bins + k]`, `df_convp_window[(ch * 5 + w) * df_bins + k]`).

| macro | default | meaning |
|---|---|---|
| `DFN2_MAX_N_FFT` | 1024 | largest `n_fft` (`DFN2_MAX_N_BINS` follows) |
| `DFN2_MAX_N_ERB` | 32 | most ERB bands |
| `DFN2_MAX_DF_BINS` | 96 | most deep-filter bins |
| `DFN2_MAX_DF_RING` | 5 (mask-only 2) | deepest ring: every pair with M + D <= 4 (ring 5); mask-only mask lookahead up to 1 |

A grid above one is refused (`DFN2_GRID_ERR_CAPACITY`). Raise a capacity with one `-D` on **every** translation unit of the
build, e.g. `-DDFN2_MAX_DF_RING=7` for the pair 2/4 or `-DDFN2_MAX_DF_RING=3` for mask-only with mask lookahead 2. They are
capacities, never the grid a program runs.

**RAM.** Because the arrays are sized for the capacity, a smaller grid wastes the difference. Measured on this host (64-bit,
`dfn2_prepost_get_mem_size`, default capacities 1024 / 32 / 96 / 5; bytes; the application's FFT handle and `AcIstft` are not in the pool):

| grid (board table pair) | `DFN2_IO_FREQ` | `DFN2_IO_TIME` | mask-only `DFN2_IO_FREQ` | mask-only `DFN2_IO_TIME` |
|---|---|---|---|---|
| 16 kHz, n_fft 256, n_erb 24, pair 0/0 | 210,336 | 213,984 | 25,632 | 29,280 |
| 32 kHz, n_fft 512, n_erb 28, pair 0/0 | 213,456 | 220,688 | 28,752 | 35,984 |
| 48 kHz, n_fft 1024, n_erb 32, pair 1/1 (mask-only 1/0) | 219,648 | 234,048 | 34,944 | 49,344 |

`sizeof(DFN2State)` is 67,424 bytes and `sizeof(DFN2ModelIOState)` 130,712 for every grid of the default build (15,360 and
5,784 mask-only); `sizeof(DFN2Grid)` is 56. A build that only serves a smaller grid lowers the capacities and gets that grid's size
back: `-DDFN2_MAX_N_FFT=256 -DDFN2_MAX_N_ERB=24` makes the 16 kHz pool 174,784 / 178,432 (FREQ / TIME, 35,552 bytes less) and
`-DDFN2_MAX_N_FFT=512 -DDFN2_MAX_N_ERB=28` the 32 kHz pool 189,792 / 197,024; mask-only at 16 kHz with the first pair of flags
14,656 / 18,304. A lowered build refuses the larger grids with `DFN2_GRID_ERR_CAPACITY`.

**What stays compile time**: `DFN2_DF_ORDER` (5; defining it is a `#error`), the capacities above, `-DDFN2_MASK_ONLY`,
`DFN2_FRAMES_PER_CALL`, the graph geometry (`DFN2_MODEL_*`: GRU hidden 256, 64 encoder channels, three input frames),
`DFN2_FEATURE_VERSION` and the normaliser starting points (`DFN2_ERB_NORM_INIT_LO_DB` and the others in `dfn2_process.h`, libDF's
literals; a model trained with others defines them on the build line, on every unit). The pool hash reads the values compiled into
`dfn2_process.c` (`dfn2_norm_starts[]`), the unit that initialises the state, so a unit built with other values does not change the hash.

**Link symbol and pool identity.**

- `dfn2_process.c` defines one symbol and the other units reference it through `DFN2_REQUIRE_VARIANT_LINK()`:
  `dfn2_variant_{full|mask_only}_n<MAX_N_FFT>_e<MAX_N_ERB>_d<MAX_DF_BINS>_r<MAX_DF_RING>`, for example
  `dfn2_variant_full_n1024_e32_d96_r5`. Objects built with a different variant (`-DDFN2_MASK_ONLY` on some units only) or different
  capacities do not link; the grid and the pair are not in it. Under a `.c` -> `.o` rule objects of other flags share names:
  keep one object directory per flag set (`make -C AINR lib` does), or clean between builds. `dfn2_frames_per_call<N>` is
  the other link symbol (see [Multi-frame export](#multi-frame-export)).
- `DFN2_PREPOST_CARVE_VERSION` is 9 (previously 8): the pool holds capacity-sized state and the state records the grid. The
  pool hash (`build_flags_hash`, FNV-1a-32) folds the whole grid for every grid: sample rate, n_fft, hop, n_erb, df_bins, `DF_ORDER`,
  both lookaheads, both alphas and the five normaliser starting points (from `dfn2_norm_starts[]` of `dfn2_process.c`), plus the
  layout and carve versions, the graph geometry, the feature version, the io_mode and, for N > 1, N. A `DFN2PrepostMemReq` recorded
  with carve 8 is refused by `dfn2_prepost_init_ex`; query the size again. The capacities are not in the hash: they change the pool
  size (`bytes`, compared by `_init_ex`) and the link symbol.
- Hashes of the board table (FREQ / TIME): 16 kHz 0/0 `3655a46b` / `1bb2ea3e`, 32 kHz 0/0 `18104ee0` / `68b1df35`,
  48 kHz 1/1 `5a69e185` / `f6a49ec8`; the same grid with another pair has another hash.

**Board samples.** The board samples (`audio_sample/`, outside this repo) own the table from (sample rate, FFT size) to the model
geometry; the library only validates what they pass. The three rows below are the same in every DFN2 sample and in the pipeline
example mains (`pipelines/dfn/dfn_example_args.h`); only the mask-only sample differs, in its df lookahead (see its row in the second table):

| model sample rate | FFT size | n_erb | df_bins | mask / df lookahead | state |
|---|---|---|---|---|---|
| 16000 | 256 | 24 | 96 | 0 / 0 | quality unmeasured |
| 32000 | 512 | 28 | 96 | 0 / 0 | provisional, no model trained |
| 48000 | 1024 | 32 | 96 | 1 / 1 | shipped |

Each sample builds its `DFN2Grid` with `dfn2_grid_make()` from a row and passes it down. `--n-erb` and `--df-bins` are never flags
(the row gives them), and a pair outside the table is refused with the list of accepted ones. Defaults are 48000 and 1024.

| sample | how the grid is chosen |
|---|---|
| `alg_ai3_dfn.c` | `--sample-rate <Hz>` and `--fft-size <n>`: the stream's rate is the model's rate. A token that begins `--sample` or `--fft` and is not one of these (`--sample-rate=16000`, `--samplerate`) prints the usage and exits 2 instead of running the default grid |
| `alg_ai3_dfn_lite.c` | the same flags and the same rule, with the mask-only table: `df_lookahead` is 0 in every row (48000 / 1024 is 1 / 0). The full model and the mask-only model are two separate samples; the lite one builds every library unit with `-DDFN2_MASK_ONLY`, in an object directory of its own |
| `alg_ai3_aec_dfn.c` | `--ai-sample-rate <Hz>` and `--ai-fft-size <n>` choose the model grid; `--sr` stays the host rate (when absent it follows the model's rate). Host rate equal to the model's: `DfnResStage` drives the model directly. Any other rate goes through `DfnRateBridge`, which needs the 48000 / 1024 model and a host at 8000 or 16000, and is otherwise refused with a message before the NPU is set up (exit 1). Its AEC has no 32 kHz grid, so the 32000 / 512 row is refused there. A misspelt token that begins `--ai` exits 2 |
| `uvc_ai_nr_stream.c` | 48 kHz only (USB audio is 48 kHz; a native 16 kHz UVC path is out of scope) and no flags. It builds its grid with `dfn2_grid_make(48000, 1024, 32, 96, 1, 1)` from its own one-row constants and drives the `DFN2Prepost` class (`DFN2_IO_FREQ`) directly: the grid goes to `dfn2_framing_config()`, `dfn2_prepost_config_defaults()`, the tensor tables and the ERB file name; the build-time guard on the sample rate is now compile-time checks of those constants |
| `uvc_audio_live2a.c` | 48 kHz only, no flags, the same one-row grid built with `dfn2_grid_make()`; it goes into the `DfnResStage` / `DfnRateBridge` configs, the model-I/O descriptor and the tensor tables (the host rate is the model's or goes through the bridge) |

**The ERB file.** Every DFN2 sample reads the ERB matrices from `/mnt/sd/dfn2_blobs/fb_<sr>_<n_erb>.bin` (for example
`fb_48000_32.bin`; `<sr>` is the model grid's rate, not the host's), one deployment set for all of them. The file is the exporter's
`erb_matrix.bin` renamed for its grid (`export_erb_matrix.py --output-dir DIR --format bin`, which `--format all` also writes; the
grid comes from the model's `config.ini`). Its layout: `int32 n_bins`, `int32 n_bands`, `float32 erb_fb[n_bins * n_bands]`,
`float32 erb_inv[n_bins * n_bands]`, both bin-major. It is not the `--runtime-bins` pair: `erb_fwd.bin` (bin-major) and `erb_inv.bin`
(band-major, the layout the library takes). A sample therefore checks the header against the grid's `n_bins` and `n_erb` and
transposes `erb_inv` to band-major once at init. None of these samples was run on a board from this repo.

**Tests and pipelines.** The tests choose their grid when they are built, through a header of their own, not through the library:
`AINR/tests/dfn2_test_grid.h` and `pipelines/dfn/tests/dfn_test_grid.h` (`-DDFN2_GRID_KHZ=16|32|48`, or `-DDFN2_SR` / `-DDFN2_N_FFT` /
`-DDFN2_N_ERB` / `-DDFN2_DF_BINS`, with `-DDFN2_MASK_LOOKAHEAD` / `-DDFN2_DF_LOOKAHEAD`; `dfn2_test_grid()` and `dfn_test_grid()` return the
`DFN2Grid`). They give the `DFN2_*` size macros that the tests' fixed arrays use and sweep grids and pairs; they are test scaffolding
only, and the pair they default to is 1/1 for every preset, which is not the board table's 0 / 0 for 16 and 32 kHz. The pipeline
objects (`dfn_res_stage`, `dfn_rate_bridge`, `dfn_hop`) and the example mains hold no build-time grid: the `DFN2Grid` travels in
`DfnResStageConfig.grid`, and the example mains build it from `--ai-sample-rate` / `--ai-fft-size` and the table above.

**Limits.** The rate bridge (`pipelines/dfn/dfn_rate_bridge.c`) is written for the 48 kHz model and refuses another (see
`pipelines/dfn/README.md`); the pipelines and their Python twin are verified on the shipped grid only. Shapes quoted elsewhere in
this README (`(1,1,3,32)` and so on) are the shipped grid's. Tests: `../tests/test_dfn2_grid.py` (every refusal and status code, the
derived fields, `make` agreeing with `validate` over a sweep of sample rates, `dfn2_norm_alpha` against `train.py` bit for bit, the
overflow-free `n_erb` check under UBSan, the `#error` on `DFN2_DF_ORDER`, C against Python from one binary on every grid),
`../tests/test_dfn2_runtime_grid.py` (capacity independence, pool identity, the normaliser starting points, no grid in the library,
the variant link symbol), `../tests/test_dfn2_prepost_c.py`, `../tests/test_c_prepost.c`, `../tests/test_nn_chains.c`.

## Calling the library

The order, for one mono stream in `DFN2_IO_TIME` (hop in, hop out). Nothing in the library knows a grid until the application
gives it one, so step 1 comes first and its `DFN2Grid` goes to every later call.

1. `dfn2_grid_make()` from the application's table row (or `dfn2_grid_validate()` on a grid it already holds).
2. Load the ERB matrices of that model (the file is described under [Choosing the grid](#choosing-the-grid); the library never reads files).
3. `fft_create(grid.n_fft)`, `dfn2_framing_config(&sc, &grid, fft, NULL)`, then `ac_istft_get_mem_size()` / `ac_istft_init()`: the
   synthesis object the class pushes into (in `DFN2_IO_FREQ` there is none and the application runs its own transforms).
4. `dfn2_prepost_config_defaults(&cfg, mode, &grid)`, set `fft`, `istft`, `erb_fwd`, `erb_inv` (and `window`, `atten_lim_db` when
   wanted), `dfn2_prepost_get_mem_size()`, a 16-byte aligned pool of `req.bytes`, `dfn2_prepost_init_ex()` (a stale `req` is refused).
5. `dfn2_model_io_descriptor_default(&grid, &d)`: the graph contract of this build for the grid, to compare with what the accelerator
   runtime reports (`dfn2_model_io_descriptor_validate(&grid, &d)`); a model exported for another grid or lookahead pair fails it.
6. Per hop: `dfn2_prepost_pre_process()`; when it returns 1, `dfn2_prepost_frame_inputs()`, the model, `dfn2_prepost_frame_commit()`
   (`dfn2_prepost_frame_skip()` when the model failed or the commit was refused); then `dfn2_prepost_post_process()`.
   `dfn2_model_run_frame(&model, p)` does the middle three for a `DFN2Model` that carries the descriptor.

```c
#include <stdio.h>
#include <stdlib.h>
#include "dfn2_prepost.h"

/* The application's table: (sample rate, FFT size) -> what the trained model fixes. */
typedef struct { int sr, n_fft, n_erb, df_bins, mask_la, df_la; } GridRow;
static const GridRow TABLE[] = {
    { 16000,  256, 24, 96, 0, 0 },
    { 32000,  512, 28, 96, 0, 0 },    /* provisional: no model trained */
    { 48000, 1024, 32, 96, 1, 1 },    /* shipped */
};

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

/* The accelerator: reads the windows in `in`, fills out->erb_mask / coefs / alpha and writes the
 * two next-state tensors. The state outputs alias the state inputs (one address each), so read
 * every state input before writing any state output. 0 when everything was written. */
typedef int (*RunModel)(const DFN2PrepostInputs *in, DFN2PrepostOutputs *out);

/* One mono stream, `hops` hops of grid.hop samples from `in` to `out`. Returns 0, or -1. */
static int run_dfn2(int sr, int n_fft, const char *erb_path, RunModel run_model,
                    const float *in, float *out, long hops)
{
    DFN2_REQUIRE_VARIANT_LINK();        /* units built with other flags or capacities do not link */

    const GridRow *row = NULL;
    for (size_t i = 0; i < sizeof TABLE / sizeof TABLE[0]; ++i)
        if (TABLE[i].sr == sr && TABLE[i].n_fft == n_fft) row = &TABLE[i];
    if (!row) return -1;

    /* 1. the grid: the six values the table gives, the rest derived and validated */
    DFN2Grid grid;
    int status = dfn2_grid_make(&grid, row->sr, row->n_fft, row->n_erb, row->df_bins,
                                row->mask_la, row->df_la);
    if (status != DFN2_GRID_OK) {
        fprintf(stderr, "grid refused: %s\n", dfn2_grid_status_text(status));
        return -1;
    }

    int rc = -1;
    size_t n = (size_t)grid.n_bins * (size_t)grid.n_erb;
    float *erb_fwd = malloc(n * sizeof(float)), *erb_inv = malloc(n * sizeof(float));
    FftHandle *fft = NULL;
    void *istft_mem = NULL, *pool = NULL;
    DFN2Prepost *p = NULL;

    /* 2. the ERB matrices of that model (caller-owned; the class borrows them) */
    if (!erb_fwd || !erb_inv || load_erb(erb_path, &grid, erb_fwd, erb_inv) != 0) goto done;

    /* 3. the framing: one FFT handle and the synthesis AcIstft (DFN2_IO_TIME) */
    AcStftConfig sc;
    fft = fft_create(grid.n_fft);
    dfn2_framing_config(&sc, &grid, fft, NULL);       /* sqrt-Hann, hop, torch normalized scaling */
    size_t istft_bytes = ac_istft_get_mem_size(&sc);
    istft_mem = aligned_alloc(16, (istft_bytes + 15) & ~(size_t)15);
    AcIstft *istft = istft_mem ? ac_istft_init(istft_mem, istft_bytes, &sc) : NULL;
    if (!fft || !istft) goto done;

    /* 4. config (copies the grid), exact pool size, init inside caller memory */
    DFN2PrepostConfig cfg;
    DFN2PrepostMemReq req;
    if (dfn2_prepost_config_defaults(&cfg, DFN2_IO_TIME, &grid) != 0) goto done;
    cfg.fft = fft;
    cfg.istft = istft;
    cfg.erb_fwd = erb_fwd;
    cfg.erb_inv = erb_inv;
    cfg.atten_lim_db = 0.0f;                          /* 0 = no attenuation limit */
    if (dfn2_prepost_get_mem_size(&cfg, &req) != 0) goto done;
    pool = aligned_alloc(req.alignment, ((size_t)req.bytes + 15) & ~(size_t)15);
    p = pool ? dfn2_prepost_init_ex(pool, (size_t)req.bytes, &cfg, &req) : NULL;
    if (!p) goto done;

    /* 5. the graph the accelerator runtime was built against. A board fills it from the exported
     *    model's metadata; dfn2_model_io_descriptor_default(&grid, &d) is this build's contract
     *    for the grid, and validate refuses a description of another grid, pair or frame count. */
    DFN2ModelIoDescriptor desc;
    if (dfn2_model_io_descriptor_default(&grid, &desc) != 0 ||
        dfn2_model_io_descriptor_validate(&grid, &desc) != 0) goto done;

    /* 6. per hop: pre_process, then (when it returns 1) frame_inputs -> model -> frame_commit,
     *    then post_process. dfn2_model_run_frame() does the middle three with a DFN2Model. */
    rc = 0;
    for (long h = 0; h < hops && rc == 0; ++h) {
        int calls = dfn2_prepost_pre_process(p, in + h * grid.hop);
        if (calls < 0) { rc = -1; break; }
        if (calls == 1) {                             /* 0 on the first mask_lookahead hops */
            DFN2PrepostInputs xin;
            DFN2PrepostOutputs xout;
            if (dfn2_prepost_frame_inputs(p, &xin, &xout) != 0) { rc = -1; break; }
            /* dfn2_prepost_state_generation(p) changed since the last frame (a reset or a
             * refused commit zeroed the state): refill the accelerator's own copy first. */
            if (run_model(&xin, &xout) != 0 || dfn2_prepost_frame_commit(p) != 0)
                dfn2_prepost_frame_skip(p);           /* identity for this frame, clocks advance */
        }
        int written = 0;                              /* 0 for the first mask + df hops: zeros out */
        if (dfn2_prepost_post_process(p, out + h * grid.hop, &written) != 0) rc = -1;
    }
done:
    free(pool);
    free(istft_mem);
    if (fft) fft_destroy(fft);
    free(erb_inv);
    free(erb_fwd);
    return rc;
}
```

In `DFN2_IO_FREQ` step 3 is dropped (the application runs its own transforms), `pre_process_freq()` / `post_process_freq()` replace
the hop calls, and the spectra are `torch.stft(normalized=True)` scaled: an unnormalised spectrum is multiplied by `grid.stft_scale`
on the way in and by `grid.stft_unscale` on the way out. Build every unit that includes the DFN2 headers with the same
`-DDFN2_MASK_ONLY` / `-DDFN2_MAX_*` / `-DDFN2_FRAMES_PER_CALL` flags as the library; `DFN2_REQUIRE_VARIANT_LINK()` turns a mismatch
into a link error. With the shipped 1/1 pair `written` is 0 for the first two hops (the output is zeros) and the end-to-end delay is
three hops; the other pairs are in [Lookahead pairs](#lookahead-pairs).

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

The library (`dfn2_set_erb_matrices`, `dfn2_prepost_set_erb_matrices`) and the pipeline example mains
take the files written by `export_erb_matrix.py --runtime-bins DIR` (`erb_fwd.bin` bin-major,
`erb_inv.bin` band-major). The board samples read the headered `erb_matrix.bin` of
`--output-dir DIR --format bin`, renamed `fb_<sr>_<n_erb>.bin` (see "The ERB file"); the other
`--output-dir` formats are for toolchain validation.

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
`dfn2_model_io.c/.h` (the three arrays with a grid dimension are packed at the grid's stride inside
capacity-sized storage), which also exports the window-slide and state-validate
helpers so an external dual-input consumer with its own state struct reuses
this window/validate discipline instead of inlining a second copy of the
memmove/memcpy pair. `dfn2_model_io_validate_state()` and
`dfn2_model_io_validate_arrays()` take `gru_hidden[DFN2_MODEL_GRU_ROWS]
[DFN2_MODEL_GRU_HIDDEN]` and the packed `float *df_convp_window` of the grid's
`df_bins` (`64 * 5 * df_bins` floats, the graph tensor element for element). Validation is a finite
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
  (`dfn2_prepost_latency_frames(grid)` + 1; 10 hops, 106.7 ms, at the shipped pair
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
  descriptor shrink accordingly, and the output is `noisy * (mask @ erb_inv)` of the head frame, `grid.mask_lookahead`
  frames behind the input. The grid's `df_lookahead` must be 0 (`DFN2_GRID_ERR_DF_LOOKAHEAD` otherwise), and `df_bins` may reach `n_bins`. The final attenuation-limit mix (`--atten-lim`, `dfn2_prepost_set_atten_lim`) stays and acts on
  the stage-1 output in Python and in C. `../tests/test_dfn2_mask_only_c.py` compares it with numpy.
- Board sample: `alg_ai3_dfn_lite.c` (`audio_sample/`, outside this repo) is the sample for this graph, separate from the
  full model's `alg_ai3_dfn.c`. Build it and every library unit with `-DDFN2_MASK_ONLY`, in an object directory of its own
  (objects of the other variant do not link: the link symbol starts `dfn2_variant_mask_only_`). It binds I/O layout 9: inputs
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
