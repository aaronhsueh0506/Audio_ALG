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

Public memory descriptors are 32 bytes (DFN2, DFN3, GTCRN; RNNoise-ERB's `RNNoiseMemReq` is 24). Their build hashes add an ILP32 token
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
DFN3 also takes its grid at run time but has only the 48 kHz / 1024 model and test grid). With mask/DF lookahead 1/1, DFN2 waits two hops and DFN3
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

RNNoise-ERB, GTCRN, DeepFilterNet2 and DeepFilterNet3 provide the streaming C code around
the network, in the frequency domain only: the application frames with audiolib (`AcStft` /
`AcIstft`; each model header states the window, scale and overlap-add mode it was trained with),
passes one spectrum per hop and takes one enhanced spectrum back; the neural inference between
those two boundaries is left to the target accelerator. Every library takes its grid at run time
and its memory from the caller (`get_mem_size` -> caller memory -> `init`), reads no file (the ERB
matrices are data the application loads and passes in) and holds none of the network's recurrent state.
The production default enables NEON where `SK_HAVE_NEON` holds (AArch64, and 32-bit ARM with NEON plus fused FMA).
Use the same switch as the conventional stack to build the scalar reference:

```bash
make lib                # libainr_prepost.a: DeepFilterNet2 + DeepFilterNet3 + GTCRN +
                        # RNNoise-ERB (member rnnoise_process.o); outputs are keyed by
                        # configuration -- `make print-lib-path` prints the absolute path
make test-simd          # NEON and forced-scalar digest must match, the
                        # model-local NEON kernels equal their scalar twins,
                        # and the DFN2/GTCRN chains print identical digests in
                        # both builds (tests/test_nn_chains.c)
make SIMD=0 test        # scalar-only build, including DFN3 streaming
python3 -m pytest DeepFilterNet2/tests DeepFilterNet3/tests tests -q
make test-pairs         # only the lookahead-pair pytest files (rebuilds the test
                        # programs that take the pair at build time)
make -C RNNoise-ERB SIMD=0 test
```

### Lookahead pairs (DeepFilterNet2 and DeepFilterNet3)

`mask_lookahead` M in 0..2 and `df_lookahead` D in 0..`df_order`-1 are
independent model contracts (15 pairs, `config.ini [model]`); a checkpoint
trained for one pair is refused under another. The ONNX graph does not depend on
them; they decide what the host does:

- the window for `heads[t]` is frames `[t-(2-M) .. t+M]` (zero before the
  stream start), the first heads are ready once frame M has been pushed (with one frame per call the host runs the
  accelerator on the M frames before that as warm-up frames and discards their heads), and
  `export_onnx.feature_windows(feature, mask_lookahead)`, the calibration
  captures and the export metadata (`mask_lookahead`, `df_lookahead`,
  `input_window_alignment`) all use it; the calibration report names the pair;
- in C the pair is two run-time fields of the spec the application gives
  `dfn2_grid_make()` / `dfn3_grid_make()` (`DFN2GridSpec` / `DFN3GridSpec`); nothing is
  selected at build time and there is no link-time pair symbol. The makers refuse an
  out-of-range pair (`DFN2_GRID_ERR_MASK_LOOKAHEAD` / `_DF_LOOKAHEAD`, `DFN3_GRID_ERR_*` likewise),
  and a graph description or pool of another pair is refused at run time (descriptor
  validation, pool hash). Each library carves the rings of its pair at init. The only
  build-time pairs are the test programs' (`tests/dfn2_test_grid.h`,
  `pipelines/dfn/tests/dfn_test_grid.h`, the preset at the top of `tests/test_dfn3_stream.c`);
- latency: DFN2 (cascade) `1 + M + D` hops, DFN3 (parallel) `1 + max(M, D)`.

The C hosts are exercised for the shipped pair by `make test` / `make test-simd`
and for every other pair by pytest (`make test-pairs`:
`tests/test_dfn2_lookahead_pairs.py`, `test_dfn2_prepost_c.py`,
`test_dfn3_stream_c.py`, `test_dfn3_prepost_c.py`, `test_dfn_lookahead_link.py`,
`test_dfn_lookahead_calibration.py`, and the multi-frame files
`test_dfn2_frames_per_call.py`, `test_dfn2_frames_per_call_c.py`). The Python
parts (export, replay, calibration, offline inference) and `test_dfn3_prepost_c.py`
(one binary, the pair in the grid) run all 15 pairs; the C programs that take
the pair at build time are rebuilt for the six pairs with `D <= M` and (0,1),
(1,3), (2,4). The pairs, the shipped pair and the compile flags the tests use come
from `tests/dfn_pairs.py` (constructor, `config.ini`, `make print-cflags`). No
checkpoint trained for a pair other than the shipped one exists here, so the
audio quality of the others is unmeasured.

### DFN2 grid

DeepFilterNet2 (the board table below is its). **The C library holds no grid.** What the exported model fixes (sample rate, FFT size, ERB bands,
deep-filter bins and order, the lookahead pair, the frames per call and the normaliser settings of `config.ini [feature]`) is a
`DFN2GridSpec` (`DeepFilterNet2/dfn2_grid.h`) that the application fills and turns into a `DFN2Grid` with
`dfn2_grid_make(&grid, &spec)`; the grid goes to every object the application creates, and one binary runs any grid the maker
accepts. There is no compile-time capacity: the pool is sized from the grid at run time by `dfn2_prepost_get_mem_size()`. The grid,
the trained model and its ERB matrices must belong together. `hop`, `n_bins`, the two scales, the two EMA coefficients (`train.py`
`make_norm_alpha`), the deep-filter history, the ring depth and `input_frames` are derived by formula. What stays compile time:
`-DDFN2_MASK_ONLY` and `DFN2_FEATURE_VERSION`. DeepFilterNet3 takes its grid at run time the same way (`DFN3GridSpec` /
`dfn3_grid_make()`), but no trained model or test program uses a grid other than 48000 / 1024.

The board samples own the table from (sample rate, FFT size) to the model geometry. `alg_ai3_dfn.c` and `alg_ai3_dfn_lite.c` take
`--sample-rate` and `--fft-size`, `alg_ai3_aec_dfn.c` and the two pipeline example mains `--ai-sample-rate` and `--ai-fft-size`, from the table without its 32000 row
(their AEC has no 32 kHz grid; their `--sr` / `--sample-rate` is the host's rate), the two UVC samples are 48 kHz only with the shipped row hard-wired. The samples read the ERB
matrices from `fb_<sr>_<n_erb>.bin` (the exporter's `erb_matrix.bin` renamed for its grid), the example mains from `--erb-fwd FILE` and `--erb-inv FILE` (details in [`DeepFilterNet2/README.md`](DeepFilterNet2/README.md#choosing-the-grid)):

| model sample rate | FFT size | n_erb | df_bins | mask / df lookahead | state |
|---|---|---|---|---|---|
| 16000 | 256 | 24 | 96 | 0 / 0 | quality unmeasured |
| 32000 | 512 | 28 | 96 | 0 / 0 | provisional, no model trained (28 is a multiple of 4, which `train.py` and the C library both ask for) |
| 48000 | 1024 | 32 | 96 | 1 / 1 | shipped |

Pools (`dfn2_prepost_get_mem_size()`, 64-bit host, bytes; the application's `AcStft` / `AcIstft`, the ERB matrices and the registered
recurrent state are not in them) follow the grid: 47,744 at 16 kHz 0/0, 62,784 at 32 kHz 0/0 and 92,608 at 48 kHz 1/1, 104,640 and 116,672
at 48 kHz 2/3 and 2/4 (a deeper ring). `DFN2_PREPOST_CARVE_VERSION` is 11 and the pool hash folds the whole grid, so a pool size or
`DFN2PrepostMemReq` recorded by an earlier build is refused. Only the tests choose a grid when they are built, through their
own header (`tests/dfn2_test_grid.h`, `pipelines/dfn/tests/dfn_test_grid.h`); the library and the pipeline objects take the
`DFN2Grid` from their caller (`DfnResStageConfig.grid` in `pipelines/dfn`). The rules, the derived fields, the status codes, the
sizes and the call order (a C skeleton under "Calling the library") are in
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
metadata and shapes disagree. The C host takes N as
`DFN2GridSpec.frames_per_call` (1..64, a run-time field): it runs the accelerator
every N hops, builds the N enhanced frames in one burst and still emits one hop per call,
so the output trails by N - 1 more hops (DFN2 latency `1 + M + D + N - 1` hops).
The pool hash and the descriptor's `input_frames` (N + 2) carry N; the pipelines DFN
stage (and the rate bridge built on it) refuse N > 1, and `make test` / `make test-simd` drive N = 1. Only N = 1 has the
one-flow-per-hop behaviour described below (warm-up frames); N > 1 keeps its older return values.
ABI: `DFN2ModelIoDescriptor` is 72 bytes (18 fields) for every N, pinned by
`_Static_assert` (C11 builds); N is carried by `input_frames` and the export
metadata, so a descriptor made for the one-frame graph (`input_frames` 3) is refused
by a grid of N > 1. Details in
[`DeepFilterNet2/README.md`](DeepFilterNet2/README.md#multi-frame-export).

The DFN C API also performs ERB/complex feature normalization, mask expansion,
the deep filter, and attenuation limiting, on spectra: the application runs
audiolib's `AcStft` / `AcIstft` around it. The post-filter (`mask_pf`) exists only in
the Python reference.
`dfn2_enhance_spectrum()` is the aligned/offline arithmetic reference. Hardware
integrations must use `dfn2_enhance_spectrum_stream()`: it explicitly pairs a delayed
network head with the spectrum frame it describes and returns the output frame
index. DFN2's cascade costs two hops because the future DF source first needs
its own lookahead mask. With one frame per call the first `mask_lookahead` hops are warm-up frames: the application runs the
same flow on every hop (`pre_process_freq` returns 1, `frame_open`, the accelerator, `frame_finish`, `post_process_freq`), the
heads of a warm-up frame are discarded, the registered state is zeroed and the state generation advances by one, so the audio is
unchanged. The output spectrum is zero while `valid` is 0 (the first `M + D` hops), so the application may push the synthesis on
every hop when its `AcIstft` is in `AC_OLA_ACCUMULATE` mode (the default; `AC_OLA_ENV_NORM` does not apply). DFN3 uses its independent
`dfn3_enhance_spectrum_stream()`/`dfn3_prepost` APIs; its raw-low branch and high-band
mask need one hop of lookahead, and are aligned to the same output frame. GTCRN uses the configured
unnormalised complex `[F,2]` network boundary. DeepFilterNet2, DeepFilterNet3
and GTCRN use zero-padded streaming warm-up (`center=False` semantics), while
their offline PyTorch whole-utterance inference uses centered STFT framing at
clip boundaries. RNNoise-ERB uses `center=False` in both Python and C; its FFT,
inverse FFT and overlap-add belong to the application's audiolib `AcStft` /
`AcIstft`, and its pre/post API layout is 2 (GRU/ONNX state layout 1).
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
one host buffer owned by the application: a float host registers (pointer, element count)
pairs in the model's config and the library zeroes and finite-checks them; a board registers
nothing, its accelerator holds the state, and the library only counts restarts.

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
it writes the output at that address. The registered state is finite-checked whole
after each invocation (`gtcrn_process_state_validate()`, `dfn2_prepost_frame_finish()`
and `dfn2_model_io_validate_state()`, `rnnoise_frame_finish()`) and zeroed on a
non-finite value, because the previous state no longer exists. The accelerator's
fixed-point copy of the state is not reached by a reset or by that zeroing, so
`dfn2_prepost_state_generation()`, `dfn2_model_io_state_generation()`,
`dfn3_prepost_state_generation()`, `gtcrn_process_state_generation()` and
`rnnoise_state_generation()` count them (1 after init, plus one per event; for DFN2 and DFN3 a warm-up frame is one): a runtime
that sees a new value refills its state tensors with quantized zeros. Registration is
all pairs or none (DFN2/DFN3: `h_gru` and `df_convp_history`, mask-only DFN2: `h_gru`; GTCRN: the eight tensors
in `GTCRNStateId` order; RNNoise-ERB: `h1`, `h2`, `h3`); a pointer with a count of 0 and
a count with a NULL pointer are refused, and the registered buffers are in no pool size or hash.

DeepFilterNet2 additionally publishes the host-side callback boundary a
pipeline binds a runtime to -- `DFN2Model` / `dfn2_model_run_frame()` in
`DeepFilterNet2/dfn2_prepost.h`, with `DFN2ModelIoDescriptor` in
`dfn2_model_io.h` as the graph contract a runtime must present;
`dfn2_model_run_frame()` returns -1 without calling `infer` for a NULL or invalid
descriptor (DFN3 has the same pair, `DFN3Model` / `dfn3_model_run_frame()`). There is no
separate accelerator adapter for DFN2, unlike Align-ULCNet: the feature
windows live inside the `DFN2Prepost` pool and the recurrent state is the
application's, so the class itself is the adapter. DFN2 and DFN3 also offer
`dfn2_prepost_pre_process_freq_dual()` / `dfn3_prepost_pre_process_freq_dual()`, which
estimate the heads from one spectrum and apply them to another on the same frame
clock; the two branches touch disjoint state inside `dfn2_process.c` / `dfn3_process.c`.

Per-model call order (with error handling), sequence diagrams and the external/internal
parameter tables are in the HTML site, one page per model:
[`ainr_dfn2_usage`](../docs/html/ainr_dfn2_usage.html),
[`ainr_dfn3_usage`](../docs/html/ainr_dfn3_usage.html),
[`ainr_gtcrn_usage`](../docs/html/ainr_gtcrn_usage.html),
[`ainr_rnnoise_erb_usage`](../docs/html/ainr_rnnoise_erb_usage.html).
