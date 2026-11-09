# Changelog

## Unreleased

### stream/ainr/dfn2: the PCM stream accepts frames_per_call > 1

- `dfn2_stream_get_mem_size()`/`init`/`init_ex` no longer refuse `grid.frames_per_call != 1`
  (`cfg_valid()`'s check is gone); any N in [1, 64] now sizes and builds a pool, exactly as
  `dfn2_grid_make()` already allowed at the grid level. No struct layout changed (the pool
  already scaled with N through `dfn2_prepost_get_mem_size()`, and `build_flags_hash` already
  folded `frames_per_call` in), so no version bump.
- `DFN2HopIO` gains `job_ready`: 1 means `inputs`/`outputs` are open and the accelerator must run
  this hop (always true when N == 1, matching every prior hop exactly); 0 means N > 1 and this
  hop has no job -- `inputs`/`outputs` are stale, skip straight to `post_process`. `pre_process`
  now reads whatever `dfn2_prepost_pre_process_freq()` actually returns (0 or 1) instead of
  treating any non-1 return as a contract error, and only opens a frame on 1.
- `post_process` now runs every hop regardless of whether that hop had a job (it drains one hop
  of PCM from whatever the last job computed either way), calling `frame_finish`/`frame_bypass`
  only when a frame is actually open. `DFN2Stream` splits what used to be one `open` flag into
  `open` (a `DFN2Prepost` frame needs finishing) and `pending` (a `pre_process` without its
  matching `post_process` is outstanding) -- for N == 1 the two always flip together, so this is
  not a behavior change for existing N == 1 callers beyond adding the always-true `job_ready`
  check at their call site.
- One function body each (`pre_process`/`post_process`), no `#ifdef`, no separate N-aware file:
  the N == 1 case is a strict special case of the general N-hop cadence, not a different code
  path. `frames_per_call` is a run-time `DFN2Grid` field everywhere else in this codebase (unlike
  `DFN2_MASK_ONLY`, a real build-time variant); gating it with a stream-level build macro would
  have been the one inconsistent case.
- `tests/smoke_dfn2_stream.c` now runs the full init/hop-loop/reset cycle twice per build
  (N=1, N=4), cross-checking `io.job_ready` against the documented cadence
  (`AINR/DeepFilterNet2/reference/multiframe.md`) and the total accelerator-job count, not just
  that it doesn't crash.
- Docs: `docs/html/models/dfn2/index.html` and `reference/core.html` updated (the per-hop
  walkthrough there is still the N=1 case specifically; it links out to the N>1 contract instead
  of being redrawn for it). Two unrelated stale version numbers found while touching these pages
  (`DFN2_PREPOST_CARVE_VERSION` still read 11 in two places `pipelines/tests/test_doc_layout_versions.py`
  doesn't scan) fixed to 12 in passing.
- Verified: `make -C stream/ainr/dfn2 test` (default, `SIMD=0`, `MASK_ONLY=1`, `NO_STDIO=1`) and
  `audit-no-stdio` all exit 0; `pipelines/tests/test_doc_layout_versions.py` passes.

### DFN2Grid / DFN2GridSpec: the feature normaliser and the FFT scale become fixed, not run-time

- `DFN2GridSpec` and `DFN2Grid` lose the seven `config.ini [feature]` normaliser fields
  (`erb_norm_tau_sec`, `erb_norm_init_lo_db`, `erb_norm_init_hi_db`, `erb_norm_scale_db`,
  `spec_norm_tau_sec`, `spec_norm_init_lo`, `spec_norm_init_hi`); `DFN2Grid` also loses
  `stft_scale` / `stft_unscale`. They were already identical for every shipped grid (a new
  checkpoint needing different values is a new build of `dfn2_grid.h`, not a run-time choice),
  so they become fixed constants `DFN2_ERB_NORM_TAU_SEC`, `DFN2_ERB_NORM_INIT_LO_DB`,
  `DFN2_ERB_NORM_INIT_HI_DB`, `DFN2_ERB_NORM_SCALE_DB`, `DFN2_SPEC_NORM_TAU_SEC`,
  `DFN2_SPEC_NORM_INIT_LO`, `DFN2_SPEC_NORM_INIT_HI` plus two new inline helpers,
  `dfn2_stft_scale(n_fft)` / `dfn2_stft_unscale(n_fft)`, replacing the two removed grid fields
  with the same numeric result. `DFN2Grid` keeps `erb_norm_alpha` / `spec_norm_alpha` (the EMA
  coefficients derived from the now-fixed tau and the grid's own sample rate and hop).
- `DFN2_GRID_ERR_ERB_NORM` (-12) and `DFN2_GRID_ERR_SPEC_NORM` (-13) merge into one
  `DFN2_GRID_ERR_NORM_ALPHA` (-12); -13 is now permanently unassigned, like -8. The only way
  left to hit it is a sample rate whose fixed-tau EMA alpha rounds to exactly 0.0 or 1.0 in
  float32 -- the seven settings are no longer caller-supplied, so there is nothing left to
  corrupt at the spec level. `DFN2_PREPOST_CARVE_VERSION` bumps 11 -> 12.
- DFN3, RNNoise-ERB and the other AINR/AIAEC models are unchanged (their own, separate grid
  structs happen to share some of these field names, which is not shared state).

### stream/ainr/dfn2: file-loader merge, comment trim

- `dfn2_stream_erb_file.c` and `dfn2_stream_internal.h` are gone, folded into `dfn2_stream.c`:
  the file-loader entry points (`dfn2_stream_init_from_erb_file[_ex]`) are now guarded by
  `-DDFN2_STREAM_NO_STDIO` instead of living in a separate translation unit; `NO_STDIO=1` compiles
  them out rather than dropping a second `.o` from the archive. `audit-no-stdio` is unchanged in
  effect. `DFN2_STREAM_LAYOUT_VERSION` bumps 1 -> 2 (the dead `heap_base` field is also dropped
  from `DFN2Stream`, on top of the grid shrink above). `stream/aiaec/deepvqe_s` and
  `stream/aiaec/ulcnet` keep their existing internal-header split (a different, model-specific
  decision, not touched here).
- Comment pass across `stream/ainr/dfn2`, `stream/aiaec/deepvqe_s`, `stream/aiaec/ulcnet` and
  `stream/common`: dropped development-process narration ("confirmed:", "mirrors the board
  sample exactly") and condensed contract comments to the essential behavior, verified
  comment-only via `cc -E -P`.

### AI model libraries: run-time grid, caller memory, frequency domain only, application-owned state and ERB matrices

- Run-time grid for all four libraries. `dfn2_grid_make(&grid, const DFN2GridSpec *spec)` replaces the six-value form; the spec also
  carries `df_order` (1..16), `frames_per_call` (1..64) and the seven normaliser settings of `config.ini [feature]`
  (`erb_norm_tau_sec`, `erb_norm_init_lo_db`, `erb_norm_init_hi_db`, `erb_norm_scale_db`, `spec_norm_tau_sec`, `spec_norm_init_lo`,
  `spec_norm_init_hi`); `dfn2_norm_alpha()` takes the tau as a float. New status codes `DFN2_GRID_ERR_DF_ORDER` (-10),
  `_FRAMES_PER_CALL` (-11), `_ERB_NORM` (-12), `_SPEC_NORM` (-13); `DFN2_GRID_ERR_CAPACITY` and every `DFN2_MAX_*` capacity are gone (-8
  stays unassigned), and so are `DFN2_DF_ORDER`, `DFN2_FRAMES_PER_CALL` (make knob, link symbol, `DFN2_MULTI_FRAME_HOST`), the
  normaliser macros and `dfn2_norm_starts[]`, the `DFN2_MODEL_GRU_*`, `DFN2_MODEL_*_GRU_LAYERS`, `_ENCODER_CHANNELS` and `_DF_PATHWAY_WINDOW` state shapes and the ring-suffixed variant symbol
  (`DFN2_REQUIRE_VARIANT_LINK()` stays and checks `-DDFN2_MASK_ONLY` only). DFN3 gets the same object (`DFN3GridSpec`, `DFN3Grid`,
  `dfn3_grid_make()`, `dfn3_norm_alpha()`) in place of the `DFN3_*` geometry macros; its lookahead pair is now two spec fields, so
  the `DFN3_MASK_LOOKAHEAD` / `DFN3_DF_LOOKAHEAD` make variables, the `dfn3_lookahead_mask<M>_df<D>` link symbol and
  `DFN_PAIR_CFLAGS` are gone (a pair of another model is refused at run time; the only compile-time DFN3 pair is the test preset in
  `tests/test_dfn3_stream.c`). GTCRN (`GTCRNGridSpec`: `n_fft`, `erb_low_bins`, `erb_high_bands`) and RNNoise-ERB (`RNNoiseGridSpec`)
  get theirs; the `GTCRN_*` / `RNNOISE_*` geometry and normaliser macros are fields of the grid (GTCRN's sample rate is not one:
  it is a 16 kHz model).
- Caller memory for all four: `*_get_mem_size(cfg, &req)` -> memory -> `*_init()` / `_init_ex()`. GTCRN is now a pool-based class
  (`GTCRNProcess`, `gtcrn_process_config_defaults()`, 32-byte `GTCRNProcessMemReq`), RNNoise-ERB too (`rnnoise_init()`, 24-byte
  `RNNoiseMemReq`); `dfn2_state_init()`, `dfn3_state_init()` and the model-I/O `_init()` take a block and its size. The class pools no
  longer carry the framing, FFT, window or recurrent state: the DFN2 pool at the shipped 48 kHz test grid, pair 1/1 (64-bit host)
  goes 219,648 -> 92,608 bytes and the DFN3 pool 219,472 -> 90,384; GTCRN is 3,328 and RNNoise-ERB (16000/512, 22 bands) 5,600.
  `DFN2_PREPOST_CARVE_VERSION` 9 -> 11, `DFN3_PREPOST_CARVE_VERSION` 8 -> 9, new `GTCRN_PROCESS_CARVE_VERSION` 1 and
  `RNNOISE_CARVE_VERSION` 1; the layout versions (DFN2 8 / 9, DFN3 9, GTCRN 8, RNNoise 1) are unchanged.
- Frequency domain only. The time-domain entries and modes (`DFN2_IO_TIME` / `_IO_FREQ`, `io_mode`, `dfn2_prepost_pre_process()` /
  `post_process()` and the DFN3 twins), `*_framing_config()`, `*_analysis()` / `*_synthesis()` and the `fft` / `window` / `istft`
  config fields are removed; `rnnoise_synthesis()` becomes `rnnoise_apply_gains()` (the gain and the `sqrt(n_fft)` scale in place, no
  inverse FFT). The application runs audiolib's `AcStft` / `AcIstft` with the framing each header states (DFN2/DFN3 and RNNoise-ERB:
  `spec_scale = grid.stft_scale`; DFN2/DFN3 `ispec_scale = grid.stft_unscale`; GTCRN scales 1.0).
- Recurrent state belongs to the application. `GTCRNModelState`, `RNNoiseModelState` and their `*_state_*` functions are removed. A
  float host registers (pointer, element count) pairs in the config (`gru_hidden` / `df_convp_history`, `state[8]` by `GTCRNStateId`,
  `h1` / `h2` / `h3`); the library zeroes and finite-checks them. A board registers nothing and keeps the state in its accelerator;
  the state generation (`dfn2_prepost_state_generation()`, `dfn3_prepost_state_generation()`, `gtcrn_process_state_generation()`,
  `rnnoise_state_generation()`) tells it when to reload quantized zeros.
- Renames: `dfn2_prepost_frame_inputs()` / `_commit()` / `_skip()` -> `_frame_open()` / `_frame_finish()` / `_frame_bypass()` (DFN3
  likewise); `dfn2_compose()` / `dfn2_compose_stream()` -> `dfn2_enhance_spectrum()` / `dfn2_enhance_spectrum_stream()` (DFN3
  likewise). The same renames apply to the Align-ULCNet and DeepVQE-S classes of AIAEC.
- ERB matrices belong to the application. DFN2, DFN3, GTCRN and RNNoise-ERB take them as two borrowed `const float *` (forward
  bin-major `[n_bins][n_erb]`, inverse band-major `[n_erb][n_bins]`, float32; GTCRN: the high-band part, `[erb_high_bins][erb_high_bands]`
  and its transpose) in the config, and DFN2, DFN3 and GTCRN also through `*_set_erb_matrices()`; no library reads a file and the
  application keeps the matrices alive while they are used. RNNoise-ERB's compiled-in tables are removed (`rnnoise_tables_gen.h`,
  `gen_rnnoise_tables.c`, `-DRNN_TABLES_PORTABLE`, the two-layer drift guard); its config takes `erb_fwd` / `erb_inv`, and
  `make -C AINR/RNNoise-ERB test-tables` checks the matrices against an independent recomputation in C.
- The spec normaliser's start values (`spec_norm_init_lo` .. `spec_norm_init_hi`; DFN2, DFN3, RNNoise-ERB) are held between the two
  configured ends: with a span wide enough to cancel in float (1.0 to 1e-8) the last bin used to start at 0. The values of the shipped
  endpoint pairs are unchanged bit for bit (checked for 2 .. 4096 bins); the C tests build both orderings of a wide span and read the
  seeds after init and after reset.
- Integration examples (`GTCRN/README.md`, `RNNoise-ERB/README.md` and their usage pages): the GTCRN accelerator callback receives the
  registered state buffer; the RNNoise example synthesises the spectra that precede the first feature window with unit gains, so the first
  output hop has its overlap term.
- Tests: the DFN2 and GTCRN chain digests (`make -C AINR test-simd`) equal the ones before the change, NEON and scalar alike; the
  DFN3 shipped stream digest is `5b831894`; `make -C AINR test-pairs` runs 915 tests. Cases of retired modes (time mode, capacities,
  link variants, build-time pairs) are retired or replaced; pool bytes and hash are pinned against an independent Python layout
  mirror. Not run: a board, an ARM32 build, ASan.

### DFN2Prepost / DFN3Prepost: one flow on every hop (warm-up frames)

- With `frames_per_call` 1 and `mask_lookahead` M >= 1 (DFN3: its `mask_lookahead`), `*_prepost_pre_process_freq()` / `_dual()` returns 1
  on every hop, where it returned 0 on the first M hops after init or reset. The application runs the same flow on each hop:
  `pre_process_freq`, `frame_open`, the accelerator, `frame_finish` (`frame_bypass` on failure), `post_process_freq`, the synthesis
  push; no `ready` variable and no `valid` branch. The first M hops are warm-up frames: `frame_open()` publishes the window as it is
  (zero-filled at the start) with NaN heads; `frame_finish()` and `frame_bypass()` do not read the heads, advance the frame clock,
  zero the registered state, advance the state generation by exactly one (also when nothing is registered) and return 0.
  `dfn2_model_run_frame()` / `dfn3_model_run_frame()` return 1 on a warm-up frame whose `infer()` succeeded, although its heads are
  discarded. A refused `frame_finish()` of a real frame keeps its old meaning; `*_prepost_reset()` restarts the warm-up.
- The audio is bit-identical for every grid and every M / D pair: the first real frame still starts from zero state (the zeroing after
  each warm-up frame replaces the old "no run"). Changed on purpose: the return code of `pre_process_freq` and `frame_open` on those
  hops, one extra accelerator run per warm-up frame once per stream start (a board's `state_generation_sync` reloads quantized zeros
  after each, M times in all), and the state generation (1 after init, plus one per reset, per warm-up frame and per refused commit).
  The post stage is unchanged: `valid` is 0 and the spectrum all zero for the first M + D hops, so with an `AcIstft` in
  `AC_OLA_ACCUMULATE` mode (the default) the application may push the synthesis on every hop without looking at `valid`; flushing the DFN2
  tail takes M + D + 1 zero audio hops (DFN3: `output_delay` zero spectra plus the overlap-add hop), the accelerator running on all of them.
- Unchanged: `frames_per_call` > 1 (the earlier hops still return 0, then one call every N frames, and no frame is a warm-up frame) and M = 0 (no warm-up frame);
  no symbol, struct size or `DFN2_PREPOST_CARVE_VERSION` (11) / `DFN3_PREPOST_CARVE_VERSION` (9) change. The words "compose" and "composed" are gone from the DFN2 / DFN3
  public headers and READMEs (the Python `compose()` and `--dfn-compose` identifiers stay); the board samples call `fetch_heads`.
- Tests: the warm-up frame is open, the heads are discarded even when NaN or garbage, the registered state is zero after it, the
  generation advances by exactly one, and the next real frame sees the inputs it saw before; the chain digests and the DFN3 stream digest
  (`5b831894`) are unchanged. The pipelines side is in `pipelines/CHANGELOG.md`.

### DFN2 grid is a run-time value (`DFN2Grid`, `dfn2_grid.h`)

- The DFN2 C library holds no grid. `DeepFilterNet2/dfn2_grid.h` defines `DFN2Grid` (`sample_rate n_fft hop n_bins n_erb df_bins
  mask_lookahead df_lookahead df_history ring erb_norm_alpha spec_norm_alpha stft_scale stft_unscale`) and
  `dfn2_grid_make(&grid, sr, n_fft, n_erb, df_bins, mask_lookahead, df_lookahead)`: the application builds the grid from the six
  values its trained model fixes and passes it to every object it creates, so one binary runs any grid within the compile-time
  capacities. Also new: `dfn2_grid_validate()`, `dfn2_norm_alpha(sr, hop, tau)` and `dfn2_grid_status_text()`.
- The macros the library used to carry are gone: `DFN2_SR`, `_N_FFT`, `_N_BINS`, `_WIN_LEN`, `_HOP_LEN`, `_OVL_LEN`, `_N_ERB`,
  `_DF_BINS`, `_ERB_NORM_ALPHA`, `_SPEC_NORM_ALPHA`, `_ANALYSIS_SCALE`, `_NORMALIZED_STFT_SCALE` / `_UNSCALE`, `_MASK_LOOKAHEAD`,
  `_DF_LOOKAHEAD`, `_DF_RING*`, `_DF_HISTORY`, `_GRID_KHZ`, `_GRID_IS_SHIPPED`, `_GRID_HEADER`. They are fields of `DFN2Grid`. Product
  code (the library, the pipeline objects, the example mains, the board samples) has no build-time way to choose a grid: it takes the
  `DFN2Grid` from its caller or builds one with `dfn2_grid_make()`. The presets (`-DDFN2_GRID_KHZ=16|32|48`) and the explicit selectors
  live only in the two test headers (`AINR/tests/dfn2_test_grid.h`, `pipelines/dfn/tests/dfn_test_grid.h`), which give the test programs
  the macros for their fixed arrays and a `const DFN2Grid *`.
- Derived by formula: `hop` = n_fft/2, `n_bins` = n_fft/2 + 1, `stft_scale` / `stft_unscale` = float32 1/sqrt(n_fft) and sqrt(n_fft),
  both alphas = `train.py` `make_norm_alpha(sr, hop, tau = 1.0)` (a model trained with another tau overwrites the two fields),
  `df_history` = `DF_ORDER` - df_lookahead - 1, `ring` = max(`DF_ORDER`, mask + df + 1) (mask-only mask + 1). Validity: sr > 0; n_fft a
  power of two in [16, 8192]; n_erb a positive multiple of 4 with `2 * n_erb <= n_bins` (checked without overflow: `n_erb` 1 << 30 and
  0x7ffffffc are refused cleanly, UBSan included); df_bins in [2, n_bins) (up to n_bins mask-only); mask_lookahead in [0, 2];
  df_lookahead in [0, `DF_ORDER` - 1] (0 mask-only); sizes within the capacities. A sample rate whose norm alpha is not strictly between
  0 and 1 is refused with `DFN2_GRID_ERR_SAMPLE_RATE`, so what `dfn2_grid_make()` accepts `dfn2_grid_validate()` accepts. Status codes
  `DFN2_GRID_ERR_{NULL,SAMPLE_RATE,N_FFT,N_ERB,DF_BINS,MASK_LOOKAHEAD,DF_LOOKAHEAD,CAPACITY,INCONSISTENT}`.
- Capacities (compile time, one `-D` on every unit raises them): `DFN2_MAX_N_FFT` 1024, `DFN2_MAX_N_ERB` 32, `DFN2_MAX_DF_BINS` 96,
  `DFN2_MAX_DF_RING` 5 (mask-only 2). A grid above one is refused with `DFN2_GRID_ERR_CAPACITY`; a pair with ring > 5 (2/4 -> 7)
  needs `-DDFN2_MAX_DF_RING=7` (`tests/dfn_pairs.py` `defines()` adds it). Still compile time: `DFN2_DF_ORDER` 5, `DFN2_MASK_ONLY`,
  `DFN2_FRAMES_PER_CALL`, the graph geometry, `DFN2_FEATURE_VERSION`, the normaliser starting points. Ring 5 holds every pair with
  mask + df <= 4; 1/4 and 2/3 need 6, 2/4 needs 7.
- API: `dfn2_framing_config(cfg, grid, fft, window)` (zeroes `cfg` on an invalid grid, so `ac_stft_get_mem_size()` /
  `ac_istft_get_mem_size()` of it are 0); `int dfn2_state_init(st, grid, stft, istft)` (-1 on an invalid grid);
  `dfn2_apply_atten_lim(n_bins, ...)`; `int dfn2_model_io_init(state, grid)`; `dfn2_model_io_descriptor_default/_validate(grid, d)`;
  `dfn2_prepost_config_defaults(cfg, mode, grid)` and `DFN2PrepostConfig.grid`; `dfn2_prepost_grid(p)`. `DFN2_MODEL_READY_FRAMES`,
  `DFN2_PREPOST_LATENCY_FRAMES` and the `DFN2_PREPOST_*_ELEMENTS` size macros become `dfn2_model_io_ready_frames(state)`,
  `dfn2_prepost_latency_frames(grid)` and the inline `dfn2_prepost_{erb_window,spec_window,erb_mask,convp_window,coefs}_elements(grid)`.
  `DFN2_REQUIRE_LOOKAHEAD_PAIR_LINK()` and the pair symbol (`dfn2_lookahead_mask<M>_df<D>`, "DFN2 / DFN3 lookahead pairs" below) are
  gone: a pair mismatch is refused at run time by the descriptor, the pool hash and the grid check. `DFN2_DF_ORDER` stays fixed at 5:
  defining it on the build line is a `#error` in `dfn2_grid.h`.
- Accelerator boundary: `DFN2ModelIOState.erb_window`, `.spec_window` and `.df_convp_window` are flat capacity-sized arrays used
  packed at the grid's stride (`erb_window[f * n_erb + b]`, ...), the push helpers take `(float *window, const float *frame, n_erb |
  df_bins)`, `dfn2_model_io_validate_arrays(gru, convp, df_bins)`, `dfn2_model_io_push_features(state, erb, spec)`, and the
  `DFN2PrepostInputs/Outputs` tensor pointers are flat `const float *` / `float *` with unchanged `*_elements` counts. `DFN2State`
  holds a `DFN2Grid grid` first and capacity-sized arrays; `df_ring_re/im` and `hi_delay_re/im` are one full-band pair
  `masked_re/masked_im[ring][DFN2_MAX_N_BINS]`.
- Link symbol: `DFN2_REQUIRE_VARIANT_LINK()` references `dfn2_variant_{full|mask_only}_n<MAX_N_FFT>_e<MAX_N_ERB>_d<MAX_DF_BINS>_r<MAX_DF_RING>`
  (for example `dfn2_variant_full_n1024_e32_d96_r5`): objects built with a different variant or capacities do not link; the grid and the
  pair are not in it. `dfn2_frames_per_call<N>` is unchanged.
- Pool identity: `DFN2_PREPOST_CARVE_VERSION` goes 8 -> 9 (the state grew by `sizeof(DFN2Grid)` and holds capacity-sized arrays), and
  the pool hash now folds the whole grid for every grid (lookahead pair, both alphas, normaliser starts), so pool sizes and hashes
  changed and a `DFN2PrepostMemReq` recorded by an earlier build (carve 8) is refused by `_init_ex`. The normaliser starting points the
  hash reads are `dfn2_norm_starts[]`, exported by `dfn2_process.c` (the unit that initialises the state); units built with other
  starting points still link, and the hash is `dfn2_process.c`'s. Capacities are not in the hash; they change `bytes` and the link symbol. Measured on a 64-bit host with the default capacities: `sizeof(DFN2State)`
  67,424 B (mask-only 15,360), `sizeof(DFN2ModelIOState)` 130,712 B (5,784), `sizeof(DFN2Grid)` 56; `DFN2_IO_FREQ` / `DFN2_IO_TIME` pools
  210,336 / 213,984 B at 16 kHz (pair 0/0), 213,456 / 220,688 B at 32 kHz (0/0), 219,648 / 234,048 B at 48 kHz (1/1); mask-only
  25,632 / 29,280, 28,752 / 35,984 and 34,944 / 49,344 B (48 kHz 1/0). A smaller grid does not shrink the state: `-DDFN2_MAX_N_FFT=256
  -DDFN2_MAX_N_ERB=24` brings the 16 kHz pool to 174,784 / 178,432 B.
- The 48 kHz 1/1 chain is unchanged. Outputs of the run-time library were compared with the former build-time library through a
  stand-in accelerator (whole-chain digests, `DFN2_IO_FREQ` and `DFN2_IO_TIME`, tiny / huge / non-finite heads and state, skipped and
  unwritten frames) for 15 grid, lookahead pair and mask-only combinations (16, 32 and 48 kHz; full pairs 0/0, 0/1, 1/1, 1/3, 2/0, 2/4, mask-only masks 0..2) in
  NEON and scalar builds: all identical. The comparison used a scratch harness that is not in the repo.
- Board samples (outside this repo) own the (sample rate, FFT size) -> n_erb, df_bins, mask / df lookahead table (16000/256 -> 24, 96,
  0/0; 32000/512 -> 28, 96, 0/0, provisional, no model trained; 48000/1024 -> 32, 96, 1/1), build the grid with `dfn2_grid_make()` and
  pass it down (`alg_ai3_aec_dfn.c` and the pipeline example mains leave out the 32000/512 row: their AEC has no 32 kHz grid). `alg_ai3_dfn.c` and `alg_ai3_dfn_lite.c` (mask-only, library units built with `-DDFN2_MASK_ONLY`, object directories
  apart, `df_lookahead` 0 in every row) take `--sample-rate` and `--fft-size` (default 48000 and 1024) and refuse a misspelt token that
  begins `--sample` or `--fft` (usage, exit 2). `alg_ai3_aec_dfn.c` takes `--ai-sample-rate` and `--ai-fft-size` for the model grid and
  keeps `--sr` as the host rate: the stage directly when the rates are equal, otherwise the rate bridge, which needs the 48 kHz / 1024
  model and refuses anything else with a message. `uvc_ai_nr_stream.c` and `uvc_audio_live2a.c` stay 48 kHz only with no flags and build
  the grid from their own one-row constants. Every DFN2 sample reads the ERB matrices from `/mnt/sd/dfn2_blobs/fb_<sr>_<n_erb>.bin`, the
  exporter's `erb_matrix.bin` (`int32 n_bins`, `int32 n_bands`, `float32 erb_fb[]`, `float32 erb_inv[]`, both bin-major) renamed for
  its grid; this replaces the single `erb_matrix.bin` name the two UVC samples and `alg_ai3_aec_dfn.c` used. Not run on a board from here.
- `DeepFilterNet2/train.py` `validate_signal_config` now asks for `n_erb % 4` instead of `% 8` (shared by the trainer and `inference.py`).
  The encoder's two stride-2 stages need only `% 4`: 20, 24, 28, 32 and 36 bands build and run, 30 does not; upstream DeepFilterNet's
  stricter `% 8` is no longer kept, so a grid with 28 bands (32 kHz) can be trained. `UPSTREAM_ALIGNMENT.md` and the
  contract test follow (`test_n_erb_divisibility_is_the_architectures_own_rule` builds and runs the accepted widths).
- Tests: `tests/test_dfn2_grid.py` (every refusal and status code, the derived fields, `make` agreeing with `validate` over a sweep of
  sample rates, the `n_erb` overflow cases under UBSan, `dfn2_framing_config` on a refused grid, the `DFN2_DF_ORDER` `#error` in C11 and
  gnu99, `dfn2_norm_alpha` against `train.py` bit for bit, C against Python from one binary on every grid) and
  `tests/test_dfn2_runtime_grid.py` (capacity independence, pool identity, the normaliser starting points of `dfn2_process.c` (this
  replaces the former "mixed objects must not link" gate), no grid in the library, the variant link symbol, the standalone test header);
  `tests/test_dfn2_lookahead_pairs.py` builds the C suites for the shipped pair and 2/4 on the 48, 32 and 16 kHz grids. The other DFN2
  tests build their grid through `tests/dfn2_test_grid.h`, which also holds the one `dfn2_test_band_borders()`; `tests/dfn_pairs.py`
  holds the shared `dfn2_grid_config()` and `dfn2_write_stream_input()`. Documentation: the DFN2 and AINR READMEs ("Choosing the
  grid", "Calling the library", "DFN2 grid") and the HTML pages `ainr_dfn2_grids.html` and `ainr_dfn2_usage.html`.

### DFN2 stage 1 alone: `--mask-only` / `-DDFN2_MASK_ONLY`

- `inference.py --mask-only` (alias `--mask_only`): also writes the stage-1-only result (ERB mask, no deep filter, same
  checkpoint) beside each output as `<name>_mask_only.wav`; the normal output is still the full model. `model.py` gains
  `apply_erb_mask()` (the stage-1 mask expansion `compose()` already used) and `compose_mask_only()` (= `compose()` with
  alpha 0, checked bitwise).
- `export_onnx.py --mask-only` and `inference.py calib --mask-only` write the stage-1 graph: inputs `erb`, `spec`, `h_gru`
  (three rows: encoder + ERB decoder, no deep-filter decoder), outputs `erb_mask`, `h_gru_out`; no `df_convp_history`. It has
  its own `state_layout_version` 9 (the full graph stays 8), `variant: mask_only` and `df_lookahead` 0 in the metadata. Five
  GRU nodes become three; with `--frames-per-call N` the `df_conv0` split for the pathway tail is not needed. The mask head
  is the full model's (same numbers). The encoder's spec-side convolutions and `df_fc_emb` stay (they feed the embedding), so
  the saving is the deep-filter decoder only; no latency or size figure has been measured.
- C: `-DDFN2_MASK_ONLY` on every translation unit builds the same model: layout 9 (`DFN2_MODEL_IO_LAYOUT_VERSION_MASK_ONLY`;
  the full one is `_FULL`, 8), `DFN2_MODEL_GRU_ROWS` 3, no deep-filter rings, coefficient/alpha heads or pathway window (the
  `DFN2PrepostInputs/Outputs` fields and the `dfn2_model_io_validate_arrays` pathway argument do not exist), the grid's
  `df_lookahead` must be 0 (`dfn2_grid_make()` returns `DFN2_GRID_ERR_DF_LOOKAHEAD` otherwise) and `df_bins` may reach `n_bins`, so the
  latency is the grid's `mask_lookahead` frames. The ring capacity of a mask-only build is 2 (mask lookahead 2 needs
  `-DDFN2_MAX_DF_RING=3`). `dfn2_compose*()` keep their signatures and
  ignore `coefs`/`alpha`; the final attenuation-limit mix (`--atten-lim` in Python, `atten_lim_db` /
  `dfn2_prepost_set_atten_lim` in C) is kept and applies to the stage-1 output. Without the define nothing changes (`make test` digests identical before and after the stage-1 change itself).
  Not covered: the pipelines' DFN unit tests (their stand-in accelerator writes coefficient heads) and the Python pipeline mirror
  build the full model only.

### Prepost borrows the application's AcIstft

- The DFN2/DFN3 prepost classes (TIME path) take a borrowed `AcIstft` built from `dfn2_framing_config()` /
  `dfn3_framing_config()`; inverse FFT and overlap-add stay with the application, the FREQ path is unchanged. The TIME pools
  shrink by the `AcIstft`, so `DFN2_PREPOST_CARVE_VERSION` and `DFN3_PREPOST_CARVE_VERSION` go 7 -> 8 (outputs bit-identical); the DFN2
  one goes on to 9 with the run-time grid above.

### DFN2 multi-frame export: N frames per invocation

- `DeepFilterNet2/export_onnx.py --frames-per-call N` (default 1) and `inference.py calib --frames-per-call N`: the graph
  takes `N + 2` feature frames and returns the heads of N consecutive frames; the three GRUs run over the N frames as one
  sequence (5 GRU nodes, as before), `h_gru_out` is the state after the last frame, and `df_convp_history_out` is the last
  five frames of the pathway input (for N > 5 the last five frames of the DF input convolution are computed from their own
  slice of the `spec` window, because the in-place rule forbids a state output produced by a Slice). N = 1 writes the same
  ONNX and JSON bytes as before (compared under a fixed `PYTHONHASHSEED`, 0..3; the exporter's optimizer pass order comes
  from a set, so the bytes differ between runs with a random seed, as they did before this change).
  Both command lines, and `StatelessDFN2Heads`, refuse an N outside 1..64 (`MAX_FRAMES_PER_CALL`, the C host's limit).
- Export metadata of an N > 1 graph: `frames_per_call`, `chunk_window_alignment`; `input_feature_frames` = N + 2 and
  `output_frames_per_invocation` = N. `onnx_streaming_contract.validate_frames_per_call` refuses a graph whose metadata
  disagrees with its shapes and a one-frame graph that records `frames_per_call`.
- C: `DFN2_FRAMES_PER_CALL` (1..64, `#error` outside; AINR Makefile variable of the same name, the `-D` flag is added only
  when N != 1). Window of N + 2 frames, first call ready after `M + N` pushed frames and then every N,
  `dfn2_model_io_frame_slot()`, `DFN2ModelIoDescriptor.input_frames` = N + 2 (the struct stays 18 fields / 72 bytes),
  commit composes the N frames in one burst, `post_process` still returns one hop per call, latency
  `1 + M + D + N - 1` hops (`dfn2_prepost_latency_frames(grid)` + 1). N > 1 is folded into the pool hash and objects built for
  different N do not link (`dfn2_frames_per_call<N>`). A unit written for one frame per call does not compile for N > 1
  unless it defines `DFN2_MULTI_FRAME_HOST`; the pipelines DFN stage stays at N = 1.
- DFN2 C descriptor ABI: `DFN2ModelIoDescriptor` is the 18-field, 72-byte struct of 778dfed; there is no `frames_per_call`
  field and N is `input_frames - 2` (`dfn2_model_io_descriptor_default()` writes `DFN2_FRAMES_PER_CALL + 2`, `_validate()`
  accepts only that). With a 76-byte struct, an adapter compiled against the previous header and linked with the new library
  had 4 bytes written past its descriptor by `_default()` and read past it by `_validate()`. Now such an adapter
  (`input_frames` = 3) works with an N = 1 library and is refused by an N > 1 one. In C11 builds `_Static_assert` pins the
  size (72 bytes) and the field list (18). `DFN2_MODEL_IO_LAYOUT_VERSION` 8 and every N = 1
  output are unchanged by the option. Test: `tests/test_dfn2_frames_per_call_c.py::test_a_descriptor_of_the_one_frame_header_is_safe_for_any_frame_count`.
- `DFN2_MODEL_IO_LAYOUT_VERSION` stays 8 and the option does not move `DFN2_PREPOST_CARVE_VERSION`: the state tensors do not
  change, and with N = 1 the pool size, hash and compile lines are those of a build without the option (rationale in
  `DeepFilterNet2/README.md`, "Multi-frame export").
- Tests `tests/test_dfn2_frames_per_call.py` and `tests/test_dfn2_frames_per_call_c.py`, added to `make test-pairs`.
- Not measured: a target toolchain, GRU support for a sequence longer than one, quantized accuracy of an N-frame graph (all
  equivalence checks use random-initialised models and a stand-in accelerator).

### Calibration: the NPZ report no longer overwrites the graph sidecar

- A calibration run exports `<stem>.onnx` and the exporter writes that graph's metadata to `<stem>.json`; in NPZ mode the
  calibration report went to the same `<stem>.json` and replaced it (the graph's `input_schema` was lost). The NPZ report is
  now `<stem>.calib.json` (`calibration_io.calibration_report_path()`). The BIN `manifest.json` and the ONNX `<onnx>.json`
  keep their names. `PTQ/calib_analyze.py` reads `<stem>.calib.json` first and falls back to `<stem>.json` only when that
  file says `artifact_format` is `numpy_npz`, so archives written before still load.

### DFN2 / DFN3 lookahead pairs: 15 legal pairs, exported and run

- Legal pairs are `mask_lookahead` in 0..2 and `df_lookahead` in
  0..`df_order`-1, independent of each other (15 pairs); the former rule
  `df_lookahead <= mask_lookahead` is gone from the models, the C range checks
  and the tests. A checkpoint trained for one pair is still refused under
  another, with one shared instruction (`onnx_streaming_contract`).
- The graph, tensor names, shapes, state layouts (DFN2 8, DFN3 9) and the shipped
  1/1 graph are unchanged; `mask_lookahead` only selects the host window
  `[t-(2-M) .. t+M]`. `feature_windows()` takes the lookahead (no default) and
  validates it when called; the export metadata gains `mask_lookahead`,
  `df_lookahead` and a derived `input_window_alignment` (unchanged string for 1);
  the calibration reports (`inference.py calib`, `inference_batch.py calib`) name
  the pair.
- C: `DFN3_MASK_LOOKAHEAD` / `DFN3_DF_LOOKAHEAD` are build
  constants (defaults 1/1) range-checked by preprocessor `#error` in every
  language mode; objects built for different pairs fail to link (a one-byte
  symbol named after the pair, no code or allocation); model-io readiness is
  `MASK + 1` frames. DFN2's pair is two fields of the `DFN2Grid` (see "DFN2 grid is a
  run-time value" above, which supersedes this for DFN2: no build constants, no pair
  symbol, `dfn2_model_io_ready_frames()`, range checks in `dfn2_grid_make()`). DFN2 (cascade) delays
  `M + D` frames with a ring of `max(DF_ORDER, M + D + 1)` (`grid.ring`); DFN3 (parallel) delays `max(M, D)` frames with a
  ring of `DF_ORDER + max(M, D) - D`, and keeps the heads in the ring when
  `D > M` (`DFN3_HEAD_QUEUE`). For the shipped pair the code, state sizes, pool
  hash and test digests are unchanged. `dfn3_compose()` documents its
  same-frame head pairing.
- Tests: `tests/dfn_pairs.py` is the one source of the pair set (probed from the
  model constructors), the shipped pair (`config.ini`) and the compile flags
  (`make print-cflags`, `make print-sources`); `make test-pairs` runs the
  per-pair pytest files. New: `test_dfn_lookahead_link.py` (link and compile
  guards), `test_dfn_lookahead_calibration.py`, `test_dfn_lookahead_contract.py`,
  `test_dfn2_lookahead_pairs.py`; the DFN3 stream driver gained a pool-identity
  check across pairs. C builds run for the six pairs with `D <= M` plus (0,1),
  (1,3), (2,4).

### Source comments condensed

- Comments and docstrings in the AINR/AIAEC model C/H files (dfn2, dfn3, gtcrn,
  RNNoise-ERB `process.*`, ulcnet, deepvqe, aiaec) and their Python model and
  streaming modules were reduced to contract text (C/H comment lines 2,998 ->
  1,700; Python 1,283 -> 963). Code, identifiers, literals and generated object
  code are unchanged. The records below were removed from the sources.
- DFN2/DFN3 `model.py` upstream alignment: `df_dec.df_gru` takes the module
  default `linear_groups=8` (checkpoint weight `(8, 64, 32)`; 16 breaks loading);
  the `lsnr` head is omitted (513 parameters, loss term not trained), so a
  streaming port cannot skip the ERB/DF stages on clean frames; DFN3's 257-parameter
  `df_fc_a` is kept for the 2,135,484 reconciliation; the released DFN3
  `convt_depthwise` key is never read upstream; the DF decoder guard
  (`emb_in_dim == df_hidden`) intentionally differs from upstream's.
- DFN2/DFN3 `process.h`: the grid is 1024/512 (DF cutoff 4500 Hz instead of
  4800 Hz at 960/480); `ERB_NORM_ALPHA` is rounded to 3 decimals (effective tau
  0.9644 s); `ERB_NORM_SCALE_DB = |log floor - init_lo|`; `ANALYSIS_SCALE`
  is the libDF `wnorm` residual 1/32 and acts only on the feature branch;
  `SPEC_NORM_EPS` stays in C although the trainer dropped it (denormal flush);
  `torch.istft` rejects `center=False` at 1024/512, and the causal EMA carries a
  first-frame difference through a whole 3 s segment (3 tau = 273 of 281 frames).
- Latency figures: DFN2 identity path 3 hops (1536 samples, 32 ms at 48 kHz);
  DFN3 2 hops (1024 samples, 21.33 ms).
- RNNoise-ERB normaliser init values come from this project's 16 kHz corpus
  (libDF's -60/-90 dB would start the EMA 35..48 dB off for the triangular bank;
  0.001 against the measured 0.008161 would put the top bin 2.86x out).
- ULCNet/DeepVQE-S `prepost`: analysis is the rolling `center=False` schedule;
  centered frame k and rolling frame k-1 cover the same samples and window (bit
  identical when measured), and algorithmic latency stays one hop (synthesis
  half-window trim). Each `prepost.c` is a separate translation unit so that the
  process and model-io files stay independently linkable.
- DeepVQE-S `frame_skip()` fails closed (silence): its stream 0 is the raw
  microphone, so pass-through would emit the full echo; the cost is a one-frame
  notch. The ULCNet adapter cannot verify that the descriptor `delay_depth`
  equals the exported depth; there is no exporter-generated descriptor header yet.
- ULCNet model-io layout versions 3-17 and DeepVQE-S 1-3 are retired; numbers
  are never reused.

### Removed (2026-10-23)

- Dead C code with no caller in any supported configuration: `dfn2_post_filter`
  and `dfn3_post_filter` (with their `df_common_post_filter` helpers, the
  `DFN*_MASK_PF` / `DFN*_PF_BETA` macros and the guarded call sites; the macro
  was hard-defined to 0, so the post-filter was never compiled), the empty
  `rnnoise_tables_init`, and the unused `dfn3_model_io_init`. The Python
  `mask_pf` option and the shipped output are unchanged.

### Changed (2026-10-22)

- DFN2/DFN3 memory descriptor hashes distinguish the 32-bit target ABI while
  preserving 64-bit hashes and layouts. Pool-walk locals initialize explicitly
  for GCC ARM32; strict C11 test declarations and array qualifiers are portable.
- ARM32 model code selects scalar twins automatically; shared FFT dispatch is
  independent. Model geometry, state, presets and latency are unchanged.

### Added

- Restored the independent DeepFilterNet3 band-split comparison branch from
  commit `c0ddbf84aac9d7dbd66d5627690a6c03a68a293a`, adapted to current shared
  training, calibration, ONNX and audiolib C interfaces. DFN2 remains available.

### Changed (2026-10-21)

- RNNoise-ERB 的 C 前後處理改用呼叫端持有的共用 FFT／STFT／ISTFT；完整
  512-sample frame、512/512/256 格點、縮放與 gain 乘法順序、首 hop 和
  overlap-add 行為維持不變。前後處理 layout 為 2，GRU／ONNX state layout 仍為 1。
- 新增 framing 尺寸、窗、縮放與模式的拒絕檢查、獨立建立的測試物件、頻譜修改／
  延遲／重置對照，以及測試 header 依賴。文件明確說明模型窗逐位元要求與 NULL 行為。
- 本機獨立 static memory 由 15,328 增至 31,136 bytes；固定 frame 的前後處理
  （不含網路推論）在 SIMD=0／1 下分別加速約 2.12／3.02 倍。FFT 捨入差異已對照
  舊實作與 Python 量測；相同頻譜的特徵處理保持逐位元相同。尚無本地已訓練權重，
  感知品質未量測，固定係數網路測試僅作數值整合驗證。

### Restoration validation (2026-10-20)

- Integrated Python suite: **220 passed**, running
  `python -m pytest AINR/DeepFilterNet2/tests AINR/DeepFilterNet3/tests AINR/tests -q`
  from the repository root.
- `make -C AINR WERROR=1 lib test-simd` and
  `make -C AINR SIMD=0 WERROR=1 test` passed. The DFN2/GTCRN output
  digests of that date were unchanged; DFN3 scalar/NEON stream digest was `5b831894`. The frozen-digest
  checks were removed on 2026-10-23: the current tests check reproducibility and NEON/scalar agreement only.
- A tiny synthetic 48 kHz corpus exercised train, resume, checkpoint load,
  WAV inference, normalizer calibration, graph/calibration export and ERB
  export. Batch-1 / batch-3 ONNXRuntime replay errors were at most
  `2.98e-7` / `2.38e-7` absolute in that smoke run.
- All 17 pre-existing DFN2 source/header/config/documentation files retained
  their recorded SHA256 values. The restored DFN3 has independent names and
  C symbols; the build includes both models.

These checks validate integration and arithmetic, not trained audio quality.
Use accepted production checkpoints or train both models before measuring
noise suppression or speech preservation.
