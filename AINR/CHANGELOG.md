# Changelog

## Unreleased

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
