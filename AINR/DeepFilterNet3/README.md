# DeepFilterNet3

DFN3 band-split implementation alongside DFN2 cascade/alpha for controlled
comparison. This custom port uses the current AINR training, calibration, ONNX
and audiolib C interfaces. For deployment, see the [DFN3 frequency-domain core](../../docs/html/models/dfn3/index.html).

## Architecture and comparison contract

```text
DFN2: full-band ERB mask -> DF on masked low bins -> learned alpha blend
DFN3: raw low bins -> DF; high bins -> ERB mask; concatenate low/high
```

DFN3 does not consume the ERB gain in its low-frequency path and has no alpha
output. Unused alpha parameters remain in `state_dict` so compatible
DFN3 v5 weights can still load strictly; they do not affect inference.
Both models deliberately omit the upstream local-SNR head.

The shipped configs agree on signal grid, feature calculation, network sizes,
loss, optimizer, schedule and dataset split/seed. They use the same shared
`dataset_gen`, `training_common`, `calibration_io` and `onnx_streaming_contract`
modules. `../tests/test_dfn_comparison.py` checks the configs, exact feature
parity, matching learned heads under controlled weights and rejection of
cross-model checkpoints. No dataset regeneration is needed to switch models
when the existing noisy/clean corpus already satisfies the 48 kHz contract.

| Setting | DFN2 | DFN3 |
|---|---|---|
| Sampling rate | 48 kHz | 48 kHz |
| FFT / window / hop | 1024 / 1024 / 512 | 1024 / 1024 / 512 |
| ERB bands / DF bins / taps | 32 / 96 / 5 | 32 / 96 / 5 |
| Mask / DF lookahead (shipped) | 1 / 1 | 1 / 1 |
| Lookahead delay | 2 hops, 21.33 ms | 1 hop, 10.67 ms |
| Identity audio delay, including WOLA | 3 hops, 32 ms | 2 hops, 21.33 ms |
| Accelerator heads | mask, coefs, alpha | mask, coefs |
| State layout version | 8 | 9 |

The audio delay is the sample shift on the identity path, with zero-filled
streaming warm-up, unit ERB mask and a unit current-frame DF tap. It excludes
accelerator execution, external queues and resampling. Whole-clip Python
inference uses centered STFT at clip boundaries; its boundary framing is not
the streaming C timing reference.

## Lookahead pairs

`mask_lookahead` (M, 0..2) and `df_lookahead` (D, 0..4, i.e. below `df_order`)
are independent and part of the trained contract (`config.ini [model]`; a
checkpoint is refused under another pair). All 15 pairs export and run; (1,1)
is the shipped one. The shared rules (window, guards, tests) are in
[shared DFN lookahead contract](../reference/training.md#dfn-lookahead).
Specific to DFN3:

- DFN3 is a parallel split: the deep filter reads the raw spectrum, so output
  frame `t` needs the heads of `t` (input up to `t+M`) and the spectrum up to
  `t+D`. The output is `max(M, D)` frames behind the input and the end-to-end
  latency is `1 + max(M, D)` hops. The first M hops are warm-up frames (the
  accelerator runs, its heads are discarded and the state restarts); with `D > M` the next `D - M` hops run the accelerator
  and emit nothing, and the heads wait in the ring until their output frame.
- In C the pair is two fields of `DFN3GridSpec` (`mask_lookahead`, `df_lookahead`); `dfn3_grid_make()` refuses a pair outside the
  ranges with `DFN3_GRID_ERR_MASK_LOOKAHEAD` / `DFN3_GRID_ERR_DF_LOOKAHEAD`. Nothing is selected at build time and there is no
  link-time pair symbol: a graph description of another pair is refused by the model-I/O descriptor and a pool of another pair
  by the pool hash.
- The ring holds `grid.ring = df_order + max(M, D) - D` frames: `grid.df_history` before the delayed frame,
  `grid.output_delay = max(M, D)` after it, and the frame itself (five for the shipped pair, up to seven for `D = 0, M = 2`). The
  state is carved to the ring at init, so the pool follows the pair: 90,384 bytes for ring 5 (1/1), 102,416 for ring 6 (1/0) and
  114,448 for ring 7 (2/0) on the 48 kHz grid (64-bit host).
- `dfn3_enhance_spectrum()` (aligned reference) pairs a spectrum with the heads passed
  in the same call; the accelerator path is `dfn3_enhance_spectrum_stream()`.

## Checkpoints

- Model: `dfn3_fmajor_flatten_pathway_add_no_lsnr_v5`.
- Features: `dfn2_libdf_wnorm_upstream_init_no_eps_v5` (the shared feature name
  is intentional).
- Loss: `dfn_mrsl_mag_complex_gamma_v2`.
- Composition: `dfn3_parallel_split` in newly written contracts.

A checkpoint with the exact model/feature versions above may omit the
composition field, since the model version already identifies that graph. An
explicitly different composition is rejected. A checkpoint from another model,
feature or accelerator-layout version must not be relabeled or force-loaded.

## Train, infer and export

Run each model from its own directory so its relative `output/` and `calib/`
paths remain separate. Use the same packed corpus and seed for the comparison:

```bash
# Use SE/.venv or an equivalent environment.
source ../../../.venv/bin/activate
python3 train.py --config config.ini --packed-dir /path/to/data_48k --seed 42
python3 inference.py --config config.ini --model output/dfn3_best.pth \
    --input input_48k.wav --output enhanced_48k.wav
python3 export_onnx.py --config config.ini --model output/dfn3_best.pth \
    --output output/dfn3_stream.onnx --verify
python3 export_erb_matrix.py --config config.ini --model output/dfn3_best.pth \
    --runtime-bins calib/erb
python3 inference.py calib --config config.ini --model output/dfn3_best.pth \
    --wav-dir /path/to/noisy_48k --frames 8192 --format bin --output calib/dfn3
```

For C deployment, `--runtime-bins` writes `erb_fwd.bin` in bin-major layout and `erb_inv.bin` in
band-major layout (raw little-endian float32 without a header, plus `erb_matrices.json`). The C config
takes pointers to exactly these two matrices (`DFN3PrepostConfig.erb_fwd` / `erb_inv`) and reads no
file: the application loads them and keeps them alive while the library uses them. The separate
`--output-dir` mode is a toolchain matrix bundle and is not interchangeable with those runtime files:
its `erb_matrix.bin` is `int32 n_bins`, `int32 n_bands`, then forward and inverse, both bin-major, so a
loader transposes the inverse to band-major.
A supplied `--model` is contract-checked and its actual buffers are exported.

Optional ONNX dependencies are in `../requirements-export.txt`. Calibration
exports and verifies the graph from the same model instance as its tensors.
`inference_batch.py calib` also supports independent fixed-batch streaming lanes;
use its `--help` for the batch-specific options. A batch-N graph and its
calibration tensors are not the single-stream C API's graph.

## Accelerator and C streaming boundary

The batch-1 graph has inputs `erb`, `spec`, `h_gru`, `df_convp_history` and
outputs `erb_mask`, `df_coefs`, `h_gru_out`, `df_convp_history_out`.
The rank-4 recurrent tensors are bound in place, as in DFN2, but DFN3's graph
and C descriptors use layout **9**, so the DFN2 layout-8 graph is refused.

- `dfn3_process.c/.h`: `DFN3GridSpec` / `dfn3_grid_make()`, and the independent feature, enhancement and band-split reference
  on spectra.
- `dfn3_model_io.c/.h`: feature windows (carved from the caller's block), accelerator bindings and state checks.
- `dfn3_prepost.c/.h`: static-pool/host lifecycle, frequency domain only, delayed head handoff, warm-up frames and reset/bypass handling. The
  application frames with audiolib (`AcStft` / `AcIstft`, sqrt-Hann, center 0, `AC_OLA_ACCUMULATE`, `spec_scale = grid.stft_scale`,
  `ispec_scale = grid.stft_unscale`).

The grid is built from what the exported model fixes, and every later call takes it:

```c
DFN3GridSpec spec = {                      /* the shipped model: config.ini [signal] [model] [feature] */
    .sample_rate = 48000, .n_fft = 1024, .n_erb = 32, .df_bins = 96, .df_order = 5,
    .mask_lookahead = 1, .df_lookahead = 1,
    .erb_norm_tau_sec = 1.0f, .erb_norm_init_lo_db = -60.0f, .erb_norm_init_hi_db = -90.0f,
    .erb_norm_scale_db = 40.0f, .spec_norm_tau_sec = 1.0f,
    .spec_norm_init_lo = 0.001f, .spec_norm_init_hi = 0.0001f,
};
DFN3Grid grid;
if (dfn3_grid_make(&grid, &spec) != DFN3_GRID_OK) return -1;       /* dfn3_grid_status_text() names the failure */

DFN3PrepostConfig cfg;
DFN3PrepostMemReq req;
dfn3_prepost_config_defaults(&cfg, &grid);                          /* copies the grid */
cfg.erb_fwd = erb_fwd;                                              /* borrowed pointers into the ERB data */
cfg.erb_inv = erb_inv;
cfg.gru_hidden = h_gru;           cfg.gru_hidden_elements = 5 * 256;                 /* float host: register the state, */
cfg.df_convp_history = convp;     cfg.df_convp_history_elements = 64 * 5 * 96;       /* a board registers nothing */
dfn3_prepost_get_mem_size(&cfg, &req);                              /* then a 16-byte aligned pool of req.bytes */
DFN3Prepost *p = dfn3_prepost_init_ex(pool, (size_t)req.bytes, &cfg, &req);
```

The per-hop order is DFN2's (see [`../DeepFilterNet2/README.md`](../DeepFilterNet2/reference/frequency-api.md#calling-the-library)) with the `dfn3_`
names: `ac_stft_push_planar()`, `dfn3_prepost_pre_process_freq()` (1 on every hop), `frame_open()`, the model, `frame_finish()`
(`frame_bypass()` on a refusal), then `dfn3_prepost_post_process_freq()` and `ac_istft_push_planar()` on every hop (the zero
spectrum before the first output yields zeros); `dfn3_model_run_frame()` wraps the middle three. DFN3 has no `frames_per_call`. The
first `mask_lookahead` hops after init or reset are warm-up frames: the model runs on them, `frame_finish()` / `frame_bypass()`
discard their heads, zero the registered state and advance the generation by one, so the first real frame starts from zero state and
the audio is unchanged. The recurrent state (`h_gru`,
`df_convp_history`) is the application's under the same rules as DFN2: both pairs registered or none, zeroed at init and
reset, finite-checked at every commit, state output at the address of its input; the state generation
(`dfn3_prepost_state_generation()`, 1 after init) advances on every reset, warm-up frame and refused commit. `DFN3_PREPOST_CARVE_VERSION` is 9.

Use `dfn3_enhance_spectrum_stream()` at the accelerator boundary; `dfn3_enhance_spectrum()` is
the aligned arithmetic reference. The streaming API applies each head to the
frame it describes, returns its output frame index and validates frame order.
The low-band ring holds raw spectra; the high-band mask stays aligned to the
same output frame. To flush the tail the application pushes `output_delay` zero
spectra while the accelerator keeps running, draining the last analysed frame; a
caller running a full STFT/OLA from PCM needs one more zero audio hop to drain the
last real hop (`output_delay + 1` total), the extra one being the overlap-add tail.
See `dfn3_prepost.h` for the lifecycle and flush contract. Both DFN2 and DFN3 pass
normalized STFT spectra (`grid.stft_scale`
in, `grid.stft_unscale` out); the feature analysis scale is applied inside the feature
branch only and must not be applied to the audio path.

Build and test from AINR:

```bash
make WERROR=1 lib
make WERROR=1 test-simd
make SIMD=0 WERROR=1 test
python3 -m pytest DeepFilterNet2/tests DeepFilterNet3/tests \
    tests/test_dfn_comparison.py tests/test_dfn3_stream_c.py \
    tests/test_dfn3_prepost_c.py -q
```

`libainr_prepost.a` contains independent DFN2, DFN3, GTCRN and RNNoise-ERB symbols. The
DFN3 C tests cover changing masks, nonzero future taps, Python/C parity,
identity audio latency, reset/rejected-head behavior and scalar/SIMD agreement.
`tests/test_dfn3_prepost_c.py` pins the grid accessors and the between-hops
ERB matrix swap contract (including its refusals) through ctypes.
The upstream numerical alignment is in
[`UPSTREAM_ALIGNMENT.md`](UPSTREAM_ALIGNMENT.md).
