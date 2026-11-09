# ONNX, ERB tables and calibration

請在模型目錄執行本頁的 Python 指令。

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
[Choosing the grid](grid.md#choosing-the-grid)); the other `--output-dir` formats are for toolchain validation.

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
must equal the fixed `DFN2_ERB_NORM_INIT_*` / `DFN2_SPEC_NORM_INIT_*` constants
of `dfn2_grid.h` (no longer `DFN2GridSpec` fields -- they are compile-time
constants now, identical for every shipped grid).
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

[返回本主題](../README.md)
