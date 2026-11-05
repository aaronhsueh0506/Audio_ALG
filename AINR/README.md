# AINR

## ARM target selection

C deployment supports both ARM EABI32 and AArch64 builds. The target compiler's
predefined macros select the path: AArch64 and 32-bit ARM with NEON plus fused FMA
(`-mcpu=cortex-a32/a53/a55/a73`, `-mfpu=neon-vfpv4`, `-mfpu=neon-fp-armv8`) use the
guarded NEON model kernels (`SK_HAVE_NEON`); other ARM32 targets use their scalar
implementations. `SIMD=0` still forces scalar.
The shared FFT independently selects its available target implementation; select
the board compiler and ABI flags rather than editing source guards. Signal grids,
model state tensors and latency do not change with pointer width.

Public memory descriptors remain 32 bytes. Their build hashes add an ILP32 token
on 32-bit targets and preserve the existing hashes on 64-bit targets. Re-query
requirements for the target build; `_init_ex` rejects a descriptor from the other
ABI before touching the supplied pool.


Standalone neural speech-enhancement models. AINR datasets contain noisy/clean
pairs and do not include the AIAEC echo-stem contract.

## Models

| Directory | Rate / grid | Role | Checkpoint contract |
|---|---|---|---|
| `RNNoise-ERB` | 16 kHz, 512/512/256 | compact ERB recurrent suppressor | audiolib AcStft/AcIstft |
| `GTCRN` | 16 kHz, 512/512/256 | GTCRN baseline | model-specific |
| `DeepFilterNet2` | 48 kHz, 1024/1024/512 (shipped grid; the application chooses the grid at run time, see [DFN2 grid](#dfn2-grid)) | DFN2 cascade + learned alpha | `MODEL_VERSION` v6 |
| `DeepFilterNet3` | 48 kHz, 1024/1024/512 | DFN3 raw-low DF / high ERB split | `MODEL_VERSION` v5, feature v5 |

`DeepFilterNet2`'s final composition is a cascade:

```text
DFN2: full-band ERB mask → masked low-band DF → alpha blend
DFN3: raw low-band DF + high-band ERB mask → concatenate
```

Do not force-load a checkpoint from another composition into DFN2. The
contract gate should reject it; bypassing the gate creates an invalid
comparison even when many parameter names happen to match.

## Dataset

`dataset_gen/` is the single noisy/clean augmentation implementation:

- run at 16 kHz for RNNoise-ERB and GTCRN;
- run at 48 kHz for DeepFilterNet3 and for the shipped DeepFilterNet2 grid (a DFN2 model trained for another grid takes that grid's sample rate);
- keep separate output directories for separate sample-rate runs.

The generator applies the same waveform transformation to noisy and target
where required and can pack WAV pairs for training. The persisted corpus is
WAV-only; optional sample metadata exists only in the in-process API. See
[`dataset_gen/README.md`](dataset_gen/README.md).

## Current DFN interpretation

`DeepFilterNet2` is the branch to use for the DFN2 cascade/alpha
experiment. If it suppresses steady low-frequency speech, compare the output
composition, target/loss behavior, normalization state, and low-frequency
training coverage before attributing the result to the shared dataset.

`DeepFilterNet3` restores the independent band-split comparison branch. Both
shipped DFN configs share the signal grid, feature arithmetic, training recipe and
dataset seed/split (DFN2 can also run another grid, see [DFN2 grid](#dfn2-grid);
DFN3 stays at 48 kHz). With mask/DF lookahead 1/1, DFN2 waits two hops and DFN3
one hop; including identity-path WOLA the measured shifts on the shipped grid are 1536 and 1024
samples respectively. See `tests/test_dfn_comparison.py` and the DFN3 README.
Keep outputs in each model's own directory.

Detailed contracts:

- [`DeepFilterNet2/README.md`](DeepFilterNet2/README.md)
- [`DeepFilterNet2/UPSTREAM_ALIGNMENT.md`](DeepFilterNet2/UPSTREAM_ALIGNMENT.md)
- [`DeepFilterNet3/README.md`](DeepFilterNet3/README.md)

## Typical commands

Run from the selected model directory:

```bash
python3 train.py --config config.ini --packed-dir /path/to/packed_data
python3 inference.py --config config.ini --model /path/to/checkpoint.pt \
    --input input.wav --output output.wav
```

Use `--resume` only with a checkpoint accepted by that directory's serialized
feature/model contract.

## Shared training safety and AINR schedule

The generic safety/checkpoint machinery lives in `AINR/training_common.py` and
is reused, through `AIAEC/training_common.py`, by the AEC candidates. It covers
non-finite handling, finite gradient clipping, evidence dumps, resume helpers
and the per-tensor vanished-weight guard. The numerical training recipe is not
universal: the AIAEC models deliberately use model-specific optimizers, losses
and schedules documented in `AIAEC/README.md`.

**AINR learning rate** — the AINR trainers use a per-*optimizer-step* linear
warmup from `lr_warmup` to `lr` over `warmup_epochs`, then cosine annealing to
`min_lr`. It decays unconditionally; nothing about it depends on validation
loss. Do not infer an AIAEC model's schedule from this section.

**Resume for reconstructible schedules** — warmup/cosine scheduler state is
deliberately not restored from the checkpoint: it would bring back the
`T_max` of the run that wrote it, so a longer resumed run would never reach
`min_lr`. `global_step` is stored instead, and `fast_forward_scheduler()`
rebuilds for the current run and indexes the fresh schedule by step. Stateful schedules,
such as Align-ULCNet's validation plateau scheduler, have their own checkpoint
contract described in `AIAEC/README.md`.

**Non-finite values** — the loss and the gradient are checked separately, because a
non-finite loss is a forward-side fault and a finite loss with an exploding
gradient is a backward-side one, and those have opposite fixes. Both route to
`halt_on_non_finite()`, which halts on the FIRST hit and writes
`output/nan_halt/e<epoch>_b<batch>_s<step>/`: the offending batch as `.pt` and
wavs, a per-lane table naming the suspect lane, and `pre_step.pth` — weights and
optimizer moments as they were BEFORE the step, so the run can be resumed from
the last uncontaminated state.

⚠ `clip_grad_norm_(..., error_if_nonfinite=True)` is not optional. Without the
flag, `clip_coef = max_norm / (inf + 1e-6)` is `0.0` and `inf * 0.0` is `NaN`, so
the clip itself writes NaN into every gradient and `optimizer.step()` carries it
into the weights and into Adam's moments, where no later clean batch recovers it.
With the flag the raise lands before any scaling. Both branches are pinned by
`AIAEC/tests/test_training_common.py::test_clipping_a_nonfinite_norm_without_the_flag_creates_the_nan`.

**Gradient trace** — every step's pre-clip norm goes to `output/grad_norm.csv`.
Once clipping is in effect the loss curve cannot distinguish a few enormous
isolated spikes (a pathological batch) from bounded gradients pointing the wrong
way (an optimizer-state problem); the trace can.

**Vanished weights** — a PER-TENSOR guard (`WeightScaleGuard`) runs once per
epoch, immediately before the checkpoint write, and halts when a parameter
tensor's `max|w|` has fallen below both 1e-06 of its own running peak and an
absolute 1e-12. Nothing else above can see that failure: the weights stay
FINITE, so the loss curve, `scan_non_finite()`, an ONNX `--verify` parity check
and the GLOBAL norm in `grad_norm.csv` all look healthy while one branch dies.

The baseline is the running maximum of each tensor's own `max|w|`, not its
initial value, so a bias that initializes at exactly zero is skipped until it
has grown and is still caught if it later collapses; under `--resume` the
baseline starts from the loaded weights.
`WeightScaleGuard`'s docstring carries the threshold arithmetic.

## C pre/post-processing

RNNoise-ERB, GTCRN, DeepFilterNet2 and DeepFilterNet3 provide streaming C analysis/synthesis
code; the neural inference between those two boundaries is left to the target
accelerator. The production default enables NEON where `SK_HAVE_NEON` holds (AArch64, and 32-bit ARM with NEON plus fused FMA).
Use the same switch as the conventional stack to build the scalar reference:

```bash
make lib                # libainr_prepost.a: DeepFilterNet2 + DeepFilterNet3 + GTCRN
                        # (RNNoise-ERB is built from RNNoise-ERB/ directly, no
                        # archive); outputs are keyed by configuration --
                        # `make print-lib-path` prints the absolute path
make test-simd          # NEON and forced-scalar digest must match, the
                        # model-local NEON kernels equal their scalar twins,
                        # and the DFN2/GTCRN chains print identical digests in
                        # both builds (tests/test_nn_chains.c)
make SIMD=0 test        # scalar-only build, including DFN3 streaming
python3 -m pytest DeepFilterNet2/tests DeepFilterNet3/tests tests -q
make test-pairs         # only the lookahead-pair pytest files (rebuilds the DFN2/DFN3
                        # C hosts per pair)
make -C RNNoise-ERB SIMD=0 test
```

### Lookahead pairs (DeepFilterNet2 and DeepFilterNet3)

`mask_lookahead` M in 0..2 and `df_lookahead` D in 0..`df_order`-1 are
independent model contracts (15 pairs, `config.ini [model]`); a checkpoint
trained for one pair is refused under another. The ONNX graph does not depend on
them; they decide what the host does:

- the window for `heads[t]` is frames `[t-(2-M) .. t+M]` (zero before the
  stream start), the first heads are ready once frame M has been pushed, and
  `export_onnx.feature_windows(feature, mask_lookahead)`, the calibration
  captures and the export metadata (`mask_lookahead`, `df_lookahead`,
  `input_window_alignment`) all use it; the calibration report names the pair;
- the DFN3 C host takes the pair as `-DDFN3_MASK_LOOKAHEAD=<M> -DDFN3_DF_LOOKAHEAD=<D>`
  (plain decimals, defaults = `config.ini`) on every translation unit of the build. An
  out-of-range pair does not compile (preprocessor `#error`, so also under
  gnu99 and C++); objects built for different pairs do not link (each unit that
  depends on the pair references a symbol named after its pair that
  `dfn3_process.c` defines); a graph description or pool of another pair is
  refused at run time (descriptor validation, pool hash). The DFN2 C host has no
  build setting for the pair: it is two fields of the `DFN2Grid` the application passes
  (`dfn2_grid_make()` refuses an out-of-range pair with `DFN2_GRID_ERR_MASK_LOOKAHEAD` /
  `DFN2_GRID_ERR_DF_LOOKAHEAD`, and a ring deeper than `DFN2_MAX_DF_RING` with
  `DFN2_GRID_ERR_CAPACITY`), there is no link-time pair symbol, and a graph description or pool of
  another pair is refused at run time (descriptor validation, pool hash). The only build-time
  pair of DFN2 is the test programs' (`tests/dfn2_test_grid.h`, `pipelines/dfn/tests/dfn_test_grid.h`),
  which size fixed arrays; a pair whose ring `max(5, M + D + 1)` is deeper than 5 (1/4 and 2/3: 6, 2/4: 7)
  also needs `-DDFN2_MAX_DF_RING=<ring>` on every unit, because the library state is sized for a capacity;
- latency: DFN2 (cascade) `1 + M + D` hops, DFN3 (parallel) `1 + max(M, D)`.

The C hosts are exercised for the shipped pair by `make test` / `make test-simd`
and for every other pair by pytest, which rebuilds them per pair
(`make test-pairs`: `tests/test_dfn2_lookahead_pairs.py`,
`test_dfn2_prepost_c.py`, `test_dfn3_stream_c.py`, `test_dfn3_prepost_c.py`,
`test_dfn_lookahead_link.py`, `test_dfn_lookahead_calibration.py`, and the
multi-frame files `test_dfn2_frames_per_call.py`,
`test_dfn2_frames_per_call_c.py`). The Python
parts (export, replay, calibration, offline inference) run for all 15 pairs;
the C builds for the six pairs with `D <= M` and (0,1), (1,3), (2,4). The
pairs, the shipped pair and the compile flags the tests use come from
`tests/dfn_pairs.py` (constructor, `config.ini`, `make print-cflags`). No
checkpoint trained for a pair other than the shipped one exists here, so the
audio quality of the others is unmeasured.

### DFN2 grid

DeepFilterNet2 only. **The C library holds no grid.** The signal grid (sample rate, FFT size), the model geometry (ERB bands,
deep-filter bins) and the lookahead pair are a `DFN2Grid` (`DeepFilterNet2/dfn2_grid.h`) that the application builds with
`dfn2_grid_make(&grid, sample_rate, n_fft, n_erb, df_bins, mask_lookahead, df_lookahead)` and passes to every object it creates;
the same binary runs any grid within the compile-time capacities (`DFN2_MAX_N_FFT` 1024, `DFN2_MAX_N_ERB` 32,
`DFN2_MAX_DF_BINS` 96, `DFN2_MAX_DF_RING` 5, mask-only 2; a grid above one is refused with `DFN2_GRID_ERR_CAPACITY`, and one `-D`
on every unit raises it). The grid, the trained model and its ERB matrices must belong together. `hop`, `n_bins`, the two scales,
the two EMA coefficients (`train.py` `make_norm_alpha`, tau 1.0 s), the deep-filter history and the ring depth are derived by
formula. What stays compile time: `DFN2_DF_ORDER` (5), the capacities, `-DDFN2_MASK_ONLY`, `DFN2_FRAMES_PER_CALL`, the graph geometry,
`DFN2_FEATURE_VERSION` and the normaliser starting points.

The board samples own the table from (sample rate, FFT size) to the model geometry. `alg_ai3_dfn.c` and `alg_ai3_dfn_lite.c` take
`--sample-rate` and `--fft-size`, `alg_ai3_aec_dfn.c` and the two pipeline example mains `--ai-sample-rate` and `--ai-fft-size`, from the table without its 32000 row
(their AEC has no 32 kHz grid; their `--sr` / `--sample-rate` is the host's rate), the two UVC samples are 48 kHz only with the shipped row hard-wired, and all of them read
the ERB matrices from `fb_<sr>_<n_erb>.bin` (details in [`DeepFilterNet2/README.md`](DeepFilterNet2/README.md#choosing-the-grid)):

| model sample rate | FFT size | n_erb | df_bins | mask / df lookahead | state |
|---|---|---|---|---|---|
| 16000 | 256 | 24 | 96 | 0 / 0 | quality unmeasured |
| 32000 | 512 | 28 | 96 | 0 / 0 | provisional, no model trained (28 is a multiple of 4, which `train.py` and the C library both ask for) |
| 48000 | 1024 | 32 | 96 | 1 / 1 | shipped |

The state and the model-I/O arrays are sized for the capacities, so a smaller grid wastes the difference: measured pools
(`DFN2_IO_FREQ` / `DFN2_IO_TIME`, 64-bit host, default capacities) are 210,336 / 213,984 bytes at 16 kHz 0/0, 213,456 / 220,688
at 32 kHz 0/0 and 219,648 / 234,048 at 48 kHz 1/1; lowering the capacities to the grid (`-DDFN2_MAX_N_FFT=256 -DDFN2_MAX_N_ERB=24`)
brings the 16 kHz pool to 174,784 / 178,432. `DFN2_PREPOST_CARVE_VERSION` is 9 and the pool hash folds the whole grid, so a pool
size or `DFN2PrepostMemReq` recorded by an earlier build is refused. Only the tests choose a grid when they are built, through their
own header (`tests/dfn2_test_grid.h`, `pipelines/dfn/tests/dfn_test_grid.h`); the library and the pipeline objects take the
`DFN2Grid` from their caller (`DfnResStageConfig.grid` in `pipelines/dfn`). The rules, the derived fields, the status codes, the link
symbol, the sizes and the call order (a C skeleton under "Calling the library") are in
[`DeepFilterNet2/README.md`](DeepFilterNet2/README.md#choosing-the-grid); the tests are `tests/test_dfn2_grid.py` and
`tests/test_dfn2_runtime_grid.py`. The rate bridge of `pipelines/dfn` is written for the 48 kHz model and refuses any other.

### Multi-frame export

DeepFilterNet2 only. `export_onnx.py --frames-per-call N` (default 1) exports
a graph that takes `N + 2` feature frames and returns the heads of N
consecutive frames with the state after the last one; the state tensors keep
their shapes. N = 1 writes exactly the one-frame graph and JSON. The metadata of
an N-frame graph records `frames_per_call` and `chunk_window_alignment`
(`input_feature_frames` = N + 2, `output_frames_per_invocation` = N), and
`onnx_streaming_contract.validate_frames_per_call` refuses a graph whose
metadata and shapes disagree. The C host is built with
`make DFN2_FRAMES_PER_CALL=<N>` (1..64; `-DDFN2_FRAMES_PER_CALL=<N>` on every
unit): it runs the accelerator every N hops, composes the N frames in one
burst and still emits one hop per call, so the output trails by N - 1 more
hops (DFN2 latency `1 + M + D + N - 1` hops). The pool hash, the
descriptor's `input_frames` (N + 2) and a link-time symbol carry N; a unit
written for one frame per call does not compile for N > 1, so `make test` / `make test-simd`
(one-frame drivers) and the pipelines DFN stage stay at N = 1 while
`make lib` builds for any N. ABI: `DFN2ModelIoDescriptor` is 72 bytes (18
fields) for every N, pinned by `_Static_assert` (C11 builds); N is carried by
`input_frames` and the export metadata, so an adapter built against the
one-frame header neither overruns nor passes an N > 1 library. Details in
[`DeepFilterNet2/README.md`](DeepFilterNet2/README.md#multi-frame-export).

The DFN C API also performs ERB/complex feature normalization, mask expansion,
deep-filter composition, and attenuation limiting. In time-domain I/O mode it
synthesizes through a caller-borrowed audiolib `AcIstft`; inverse FFT and
overlap-add run inside that object. The post-filter (`mask_pf`) exists only in
the Python reference.
`dfn2_compose()` is the aligned/offline arithmetic reference. Hardware
integrations must use `dfn2_compose_stream()`: it explicitly pairs a delayed
network head with the spectrum frame it describes and returns the output frame
index. DFN2's cascade costs two hops because the future DF source first needs
its own lookahead mask. The API returns zero during warm-up; callers must not
send an invalid warm-up frame to synthesis. DFN3 uses its independent
`dfn3_compose_stream()`/`dfn3_prepost` APIs; its raw-low branch and high-band
mask need one hop of lookahead, and are aligned to the same output frame. GTCRN uses the configured
unnormalised complex `[F,2]` network boundary. DeepFilterNet2, DeepFilterNet3
and GTCRN use zero-padded streaming warm-up (`center=False` semantics), while
their offline PyTorch whole-utterance inference uses centered STFT framing at
clip boundaries. RNNoise-ERB uses `center=False` in both Python and C; its C
analysis reads a full caller-supplied 512-sample frame rather than a hop.
Its FFT/IFFT/overlap state belongs to caller-owned audiolib `AcStft`/`AcIstft`
objects, bound with pre/post API layout 2; its GRU/ONNX state layout stays 1.
See the model README for initialization and memory requirements.

DeepFilterNet2, DeepFilterNet3 and GTCRN also provide `export_onnx.py`,
`inference.py calib`, and `export_erb_matrix.py`. Install the optional
packages in `requirements-export.txt`; the model-specific READMEs define
whether recurrent state is explicit at the ONNX boundary and whether ERB is
inside or outside the graph.

RNNoise-ERB, DeepFilterNet2, DeepFilterNet3 and GTCRN calibration can emit `bin` or `npz`.
Every calibration run also exports and parity-checks the ONNX graph the
tensors bind to (default `<output>.onnx`, override with `--onnx`), from the
same model instance in the same process, so the two deployment artifacts
cannot drift apart; `export_onnx.py` remains available for a graph-only
export. The deployment BIN layout is identical for all four: one subdirectory per
ONNX input, one zero-padded `<tensor>_0000.bin`, `<tensor>_0001.bin`, ... per invocation, and a
`manifest.json`. Each file contains one complete graph input tensor, including
its batch dimension; only the filename index is the calibration-frame axis.
Keep generated data under each model's `calib/`; those
directories are ignored by Git and remain separate from `output/` weights and
ONNX graphs.

The active deployment exports are stateless accelerator graphs:

| Model | New signal context per call | Caller-owned model state |
|---|---:|---|
| RNNoise-ERB | 3 feature frames -> 1 gain frame | 3 GRU hidden tensors |
| DeepFilterNet2 | 3 feature frames -> 1 head frame (N + 2 -> N with `--frames-per-call N`) | `h_gru` (1,5,1,256) + 5-frame DF pathway window (1,64,5,96) |
| DeepFilterNet3 | 3 feature frames -> 1 two-head frame | same state tensor geometry as DFN2, independent layout 9, no alpha output |
| GTCRN | 1 complex STFT frame | 6 conv windows (1,16,3\|5\|11,33) + `h_tra` (1,6,1,16) + `h_dpgrnn` (1,4,33,8) |

“Stateless” describes the accelerator, not the algorithm. Each state tensor has
one host buffer owned by the caller.

DeepFilterNet2, DeepFilterNet3 and GTCRN state is rank 4 and has the same shape at the graph
input and at its `*_out` output; the runtime binds both to the same address, so
the updated state is the next call's input and nothing is copied. A sliding
window (DFN2 `df_convp_history`, the six GTCRN conv windows) is the full window:
the graph drops the oldest frame with a Slice on the input, appends the new
frame and returns the whole concatenation; `state_slicing` declares
`drop = first` for it and the validator accepts only that range. Recurrent states are stacked into
one tensor per model (`h_gru`, `h_tra`, `h_dpgrnn`), cut with Slices on dim 1
and joined with one Concat. `onnx_streaming_contract.py`
(`validate_inplace_state_graph`) checks this at export time and
`check_state_quantization.py` repeats it on an exported graph and on a
quantization report. The runtime must finish reading every state input before
it writes the output at that address, and the `*_validate` helpers
(`gtcrn_model_state_validate()`, `dfn2_model_io_validate_state()`) finite-check
the state and zero it on a non-finite value, because the previous state no
longer exists. The accelerator's fixed-point copy of the state is not reached by
a reset or by that zeroing, so `dfn2_prepost_state_generation()`,
`dfn2_model_io_state_generation()` and `gtcrn_model_state_generation()` count
them (1 after init, plus one per event): a runtime that sees a new value refills
its state tensors with quantized zeros.

RNNoise-ERB takes the ordinary runtime's own output tensors with
`rnnoise_model_state_inherit()` (finite-check, then copy), or is bound in place
and validated with `rnnoise_model_state_validate()`.

DeepFilterNet2 additionally publishes the host-side callback boundary a
pipeline binds a runtime to -- `DFN2Model` / `dfn2_model_run_frame()` in
`DeepFilterNet2/dfn2_prepost.h`, with `DFN2ModelIoDescriptor` in
`dfn2_model_io.h` as the graph contract a runtime must present;
`dfn2_model_run_frame()` returns -1 without calling `infer` for a NULL or invalid
descriptor. There is no
separate accelerator adapter for DFN2, unlike Align-ULCNet: the feature
windows and both recurrent state tensors already live inside the `DFN2Prepost`
pool, so the class itself is the adapter. The FREQ mode also offers
`dfn2_prepost_pre_process_freq_dual()`, which estimates the heads from one
spectrum and applies them to another on the same frame clock; the two
branches touch disjoint state inside `dfn2_process.c`.

Per-model call order (with error handling), sequence diagrams and the external/internal
parameter tables are in the HTML site, one page per model:
[`ainr_dfn2_usage`](../docs/html/ainr_dfn2_usage.html),
[`ainr_dfn3_usage`](../docs/html/ainr_dfn3_usage.html),
[`ainr_gtcrn_usage`](../docs/html/ainr_gtcrn_usage.html),
[`ainr_rnnoise_erb_usage`](../docs/html/ainr_rnnoise_erb_usage.html).
