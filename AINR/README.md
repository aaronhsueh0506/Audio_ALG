# AINR

## ARM target selection

C deployment supports both ARM EABI32 and AArch64 builds. The target compiler's
predefined macros select the path: AArch64 uses the guarded NEON model kernels;
ARM32 uses their existing scalar implementations. `SIMD=0` still forces scalar.
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
| `DeepFilterNet2` | 48 kHz, 1024/1024/512 | DFN2 cascade + learned alpha | `MODEL_VERSION` v6 |
| `DeepFilterNet3` | 48 kHz, 1024/1024/512 | DFN3 raw-low DF / high ERB split | `MODEL_VERSION` v5, feature v5 |

`DeepFilterNet2`'s final composition is a cascade:

```text
DFN2: full-band ERB mask → masked low-band DF → alpha blend
DFN3: raw low-band DF + high-band ERB mask → concatenate
```

Do not force-load a checkpoint from another composition into DFN2. The
contract gate should reject it; bypassing the gate creates an invalid
comparison even when many parameter names happen to match.

The two files currently under `DeepFilterNet2/output/` predate this contract:
`dfn2_best.pth` and `dfn2_last.pth` identify model v5 / feature v3, while the
source above requires model v6 / feature v5. They remain historical artifacts,
not deployable weights; the AEC/BF/RES input-placement evaluation must wait for
a freshly trained compatible checkpoint.

## Dataset

`dataset_gen/` is the single noisy/clean augmentation implementation:

- run at 16 kHz for RNNoise-ERB and GTCRN;
- run at 48 kHz for DeepFilterNet2 and DeepFilterNet3;
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
DFN configs share the signal grid, feature arithmetic, training recipe and
dataset seed/split. With mask/DF lookahead 1/1, DFN2 waits two hops and DFN3
one hop; including identity-path WOLA the measured shifts are 1536 and 1024
samples respectively. See `tests/test_dfn_comparison.py` and the DFN3 README.
Historical model-v5 / feature-v3 weights are incompatible with both current
feature-v5 implementations. Keep outputs in each model's own directory.

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
deliberately not restored from the checkpoint. Restoring it brings back the
`T_max` of the run that wrote it: measured, a checkpoint from a 100-epoch run
resumed at 120 epochs ends on lr 1.02e-04 against a `min_lr` of 1e-06, 102x.
`global_step` is stored instead, and `fast_forward_scheduler()` rebuilds for
the current run and indexes the fresh schedule by step. Stateful schedules,
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
absolute 1e-12. Nothing else above can see that failure. A real run lost one of
two parallel subband GRU blocks to weights of 6.5e-25 while its twin stayed at
6e-01, and every value stayed FINITE the whole way down, so the loss curve,
`scan_non_finite()` and an ONNX `--verify` parity check all reported agreement —
PyTorch and ONNX produce the same garbage. `grad_norm.csv` could not see it
either: that is the GLOBAL norm, which the surviving branch and the rest of the
network held above 0.2 for the entire run.

The baseline is the running maximum of each tensor's own `max|w|`, not its
initial value, so a bias that initializes at exactly zero is skipped until it
has grown and is still caught if it later collapses; under `--resume` the
baseline starts from the loaded weights. Both thresholds have orders of
magnitude on either side: float32's smallest normal is 1.18e-38, so the 1e-12
floor sits 26 decades above the denormal boundary, 19x above even the least
collapsed tensor of that run (5.3e-14), and eight and a half decades below the
smallest `max|w|` any healthy tensor starts at (3.8e-02).
`WeightScaleGuard`'s docstring carries the rest of the arithmetic.

## C pre/post-processing

RNNoise-ERB, GTCRN, DeepFilterNet2 and DeepFilterNet3 provide streaming C analysis/synthesis
code; the neural inference between those two boundaries is left to the target
accelerator. The production default enables AArch64 NEON.
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
make -C RNNoise-ERB SIMD=0 test
```

The DFN C API also performs ERB/complex feature normalization, mask expansion,
deep-filter composition, optional post-filter/attenuation limiting, and WOLA.
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
| DeepFilterNet2 | 3 feature frames -> 1 head frame | `h_gru` (1,5,1,256) + 5-frame DF pathway window (1,64,5,96) |
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

RNNoise-ERB still takes the ordinary runtime's own output tensors with
`rnnoise_model_state_inherit()` (finite-check, then copy), or is bound in place
and validated with `rnnoise_model_state_validate()`.

DeepFilterNet2 additionally publishes the host-side callback boundary a
pipeline binds a runtime to -- `DFN2Model` / `dfn2_model_run_frame()` in
`DeepFilterNet2/dfn2_prepost.h`, with `DFN2ModelIoDescriptor` in
`dfn2_model_io.h` as the graph contract a runtime must present. There is no
separate accelerator adapter for DFN2, unlike Align-ULCNet: the feature
windows and both recurrent state tensors already live inside the `DFN2Prepost`
pool, so the class itself is the adapter. The FREQ mode also offers
`dfn2_prepost_pre_process_freq_dual()`, which estimates the heads from one
spectrum and applies them to another on the same frame clock; the two
branches touch disjoint state inside `dfn2_process.c`.
