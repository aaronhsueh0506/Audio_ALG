# AIAEC models

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


AIAEC has four candidates; Align-ULCNet covers the linear-AEC + RES/NR route.

| Route | Model | Model inputs | Training target | Status |
|---|---|---|---|---|
| linear AEC -> RES+NR | Align_ULCNet | linear error + far | early, clean near speech | selected |
| end-to-end AEC+RES+NR | Align_CRUSE | microphone + far | early, clean near speech | selected |
| end-to-end AEC+RES+NR | DeepVQE_S | microphone + far | early, clean near speech | primary |
| end-to-end AEC+RES+NR | CAGCRN | microphone + far | early, clean near speech | backup |

The model grid is zero-padding-free, sqrt-Hann and 50% overlap. Current model
configs use FFT/window/hop = 512/512/256 at 16 kHz. Checkpoint contracts store
the complete grid and inference rejects incompatible settings.

## Dataset

dataset_gen/ renders a complete stateful AEC scenario, runs the frozen Python
PBFDKF over the complete sequence, then cuts it into 10-second WAV chunks. WAV
generation stores five stems:

1. far_render
2. near_speech
3. near_target
4. mic_postclip
5. linear_error

Packing stores only the four tensors used by training: far_render,
mic_postclip, linear_error, and near_target. Existing generated WAV data does
not need to be regenerated when repacking; to recompute `linear_error` after a
frontend change see `dataset_gen/README.md`, "Recomputing `linear_error` after a
frontend change".

model_views.build_model_view() is the only waveform-to-model mapping:
Align-ULCNet receives linear_error + far_render; the other three receive
mic_postclip + far_render. Every model targets near_target.

See dataset_gen/README.md for generation, packing, contract migration and
resume rules.

## Train and inference

Every retained model directory owns these user-facing files:

- train.py
- config.ini
- inference.py
- export_onnx.py

Shared training/checkpoint and signal utilities remain under AIAEC/; modules
whose filename starts with an underscore are implementation details, not
command-line entry points.

Example:

    cd AIAEC/Align_ULCNet
    python3 train.py --config config.ini \
      --packed-dir /path/to/packed/all --gpu 0 --mmap
    python3 inference.py checkpoint.pth mic.wav far.wav enhanced.wav

Inference is streaming by default: it consumes one audio hop, advances the
STFT, calls ``forward_stream()`` for one frame with persistent state, and
advances WOLA.  ``--verify`` additionally runs the whole-utterance reference
and reports the numerical difference.  Both mono inputs are resampled to the
checkpoint sample rate first; output remains at that rate. Inputs must
describe the same capture timeline; a tail-length mismatch is cropped or
zero-padded, but a start-time offset is not estimated here.

Align-ULCNet also accepts --input-is-linear-error for evaluating a precomputed
linear-AEC residual. Normal inference advances the checkpoint-matched PBFDKF
one hop at a time and feeds its formed error plus the original raw far to the
model; PBFDKF is not precomputed over the whole WAV.

## Training contract

Generic safety machinery is shared through `AIAEC/training_common.py`:
non-finite evidence dumps, finite gradient clipping, the per-tensor
vanished-weight guard, data loading, and checkpoint contracts. Optimizer and
the optimizer remain **model-specific**; Align-ULCNet, Align-CRUSE and DeepVQE-S
share the PLCPA objective and CAGCRN keeps its own. The shipped schedule is
deliberately common: three epochs of linear warmup followed by per-step cosine decay.

| model | optimizer / LR | schedule | objective | shipped batch / epochs |
|---|---|---|---|---|
| Align-ULCNet | Adam / 4e-3 | 3-epoch warmup + cosine to 1e-6 | STFT-consistent PLCPA, c=0.3, beta=0.7 on complex term (the papers state no formula) | 16 / 50 |
| Align-CRUSE | Adam / 1.5e-4, decay 5e-6 | 3-epoch warmup + cosine to 1e-6 | STFT-consistent PLCPA, c=0.3, beta=0.7 on complex term | 16 / 50 |
| DeepVQE-S | AdamW / 1.2e-3, decay 5e-7 | 3-epoch warmup + cosine to 1e-6 | inherited Align-CRUSE PLCPA; the DeepVQE paper does not state its loss | 8 / 50 |
| CAGCRN | AdamW / 1.2e-3, decay 5e-7 | 3-epoch warmup + cosine to 1e-6 | MSE + SI-SNR + normalized parameter L1 | 32 / 50 |

### What a four-way comparison can and cannot conclude

The four candidates do not share an optimizer, and CAGCRN has its own loss, so a
ranking of their final numbers is not a ranking of their architectures. Two
independent reasons, both measurable from this directory:

**The losses are different functions.** PLCPA (Align-ULCNet, Align-CRUSE,
DeepVQE-S) and MSE+SI-SNR+L1 (CAGCRN) do not share a scale or a zero. Loss
values across that boundary are not comparable at all -- only a common
external metric is.

**The optimisation budgets differ.** Batch sizes, epoch definitions, peak
learning rates and published training durations differ. Sharing the shape of
the schedule does not equalise update count or audio exposure, so a win by one
model is not attributable to architecture alone.

This is a deliberate trade: per-paper fidelity is chosen over a shared
protocol, and the two cannot both hold (Align-ULCNet's loss is the exception:
its paper states none, so it uses the shared PLCPA objective). Whichever is
wanted, it should be an explicit decision rather than an artefact:

- **To rank architectures**, put all four on one optimiser, one schedule and
  one loss, and accept that none is then paper-faithful.
- **To reproduce each paper**, keep these recipes and judge each model against
  its own published numbers, not against its siblings here.

The epoch budget is a project decision. Align-ULCNet batch 16 and DeepVQE-S batch 8 are GPU-memory overrides. The
Align-ULCNet paper uses 64; DeepVQE-S and Align-CRUSE both publish batch 400
(for 250 and 150 epochs respectively). CAGCRN explicitly publishes batch 32.
Early stopping is disabled for all four shipped 50-epoch runs; validation still
selects the best checkpoint. The common cosine schedule is a project decision,
not a claim that all four papers published that schedule.
Changing a loss recipe changes `loss_version`, so a checkpoint trained under a different
recipe is rejected instead of silently resuming under a new objective.
The checkpoint contract records the resolved training recipe -- optimizer
name, learning rate, weight decay, amsgrad, schedule and its scalars, batch
size, epoch budget, warmup and LR floor -- read once and used both to build the
optimizer and to write the contract, so `require_checkpoint_contract` rejects a
changed trajectory instead of silently resuming it. All four shipped configs use
`warmup_cosine`, rebuilt from the global optimizer-step count on resume instead
of restoring a stale scheduler horizon.

The stored project examples are 10-second chunks and one epoch is one complete
DataLoader pass. Align-ULCNet's paper used random 3-second examples and 20,000
steps per epoch.

## ONNX export and calibration

The accelerator graphs are stateless: convolution windows, attention rings
and recurrent state are explicit rank-4 graph inputs and outputs. Each output
has exactly the shape of the input it continues: a window input is cut by a
Slice on the input side and the new frame is concatenated, so a runtime can
bind every `*_out` to the address of its input (`state_binding =
in_place_same_address`). `export_graph` checks that structure on every file it
writes (`validate_inplace_state_contract` in `_onnx_contract.py`, a wrapper over
`validate_inplace_state_graph` in `AINR/onnx_streaming_contract.py`). Each
window declares the end it drops in the graph metadata (`state_window_drop`:
`first` for a time-ordered window, `last` for a newest-first ring) and the
validator accepts exactly that Slice range. Complex values use real/imaginary
pairs.

One calibration run produces both deployment artifacts: it exports and
parity-checks the ONNX graph the tensors bind to (default `<output>.onnx`,
override with `--onnx`) from the same wrapper in the same process, then
records the calibration frames against it. Run it from the model directory:

    python3 inference.py calib \
      --checkpoint checkpoint.pth \
      --primary-dir /path/to/primary_wavs \
      --far-dir /path/to/far_wavs \
      --frames 8192 --format bin \
      --output calib/model

`export_onnx.py` remains available for a graph-only export:

    python3 export_onnx.py \
      --checkpoint checkpoint.pth \
      --output output/model.onnx --verify

For Align-ULCNet, --primary-dir contains materialized linear-error WAVs (see
"Align-ULCNet linear-error provenance" below), or, with --primary-is-mic, RAW
microphone WAVs. For the end-to-end models it contains microphone WAVs.
Relative WAV paths in the
primary and far trees must match; differently-tokened names
(mic_001.wav vs lpb_001.wav) pair via an explicit --pair-replace mic:lpb
rule.

Calibration formats:

- --format bin: tensor/tensor_0000.bin, tensor/tensor_0001.bin, and so on, plus
  manifest.json. Each file is one complete ONNX input tensor, including batch
  dimension. This is the board-facing format.
- --format npz: one NumPy archive plus `<name>.calib.json` beside it. The graph exported by the same run is
  `<name>.onnx` with its own `<name>.json`; the report has a different name so the two never share a file.

If --format is omitted, a .npz suffix selects NPZ; otherwise BIN is used.
Generated calib/ directories are ignored by Git.

Align-ULCNet supports --max-delay-frames D in both export and calibration.
The same D must be used for the ONNX graph, calibration artifact and CPU state
allocation. D changes state shapes, not learned weight shapes.

Align-ULCNet calibration uses the available training-domain
linear_error + raw_far WAV pair. Board deployment supplies the same raw_far;
the report records both fields so a mismatch is detectable.

Align-ULCNet linear-error provenance. The calibration frames describe the
error signal of one specific front end, the unified AEC's analysis seam
(`aiaec-linear-error-v4`, `error_input_seam = analysis_formed`). A pre-made
--primary-dir therefore has to come from that front end and carry a stamp:

- a `linear_error.contract.json` inside the directory (written by a
  --primary-is-mic run), or
- a `linear_error.done.json` corpus ledger next to it (the dataset
  generator's ledger) that records the contract it was written under; a
  legacy ledger with only a fingerprint does not count, and a ledger that does
  not claim every chunk sequence under the directory (a half re-rendered
  corpus) is refused.

The stamp must equal the checkpoint's own `linear_aec` contract. A retired
contract version or behaviour hash, a different contract, an internally
inconsistent stamp or a stamp taken from another seam is refused before the
ONNX graph or the artifact is written. A directory with no stamp is refused too,
unless --allow-unstamped-linear-error is passed (Align-ULCNet, without
--primary-is-mic only); the override emits a warning and the report records
`linear_error_stamp.origin = unstamped_override`.

--primary-is-mic is the self-stamping route: the checkpoint-matched frozen
PBFDKF derives the stems from RAW microphone WAVs in-process, writes them to
`<output>_linear_error/` with a `linear_error.contract.json` made from the
checkpoint's contract, and records `primary_source = raw_mic_via_frozen_pbfdkf`.
The sidecar is written last, so an interrupted run leaves stems without a
stamp.

The calibration report (`<name>.calib.json`, or `manifest.json` for BIN) is
stamped as well: `linear_aec_contract_version`, `aec_behavior_hash`,
`linear_aec_contract_hash`, `error_input_seam` and `linear_error_stamp`
(origin `sidecar`, `corpus_ledger`, `derived_in_process` or
`unstamped_override`). `PTQ/calib_analyze.py` refuses an Align-ULCNet artifact
with no stamp, a contract version other than the current one, or a retired
behaviour hash, both when loading it and when writing a pruned selection;
regenerate such an artifact with `inference.py calib`. Export
(`export_onnx.py`) and `inference.py` refuse a checkpoint whose recorded
behaviour hash is retired; `--input-is-linear-error` is an evaluation-only
bypass that does not look at the supplied WAV's origin. None of this is
enforced in C: the C libraries and pipelines never refuse a model by front-end
version, the contract travels as metadata (ONNX props, JSON sidecar, the
`ULCNET_MODEL_IO_FRONTEND_CONTRACT` identifier) and is checked on the Python
side and by committed tests.

## C pre/post-processing

    make -C AIAEC
    make -C AIAEC print-lib-path

This builds libaiaec_prepost.a under AIAEC/build/simd<N>-<sig>/ -- outputs
are keyed by configuration (SIMD, flags, compiler), so a SIMD=0 build never
reuses SIMD=1 objects; print-lib-path prints the archive's absolute path for
the current configuration. Members: ulcnet_process, ulcnet_model_io,
ulcnet_prepost, ulcnet_accelerator_adapter, aiaec_process, deepvqe_process
and deepvqe_prepost, one object each.

| Model | Accelerator output | Host composition |
|---|---|---|
| Align-ULCNet | complex mask (planar) + full next state windows (written in place) | `ulcnet_prepost` class (ulcnet_model_io_enhance_spectrum() multiplies the compressed error by the mask and inverts the power; the application's `AcIstft` follows) |
| Align-CRUSE | real mask | aiaec_apply_real_mask() |
| DeepVQE-S | 3x3 complex CCM taps | `deepvqe_prepost` class (deepvqe_ccm_process()) |
| CAGCRN | complex mask | aiaec_apply_complex_mask() |

Align-ULCNet and DeepVQE-S each ship a pre/post *class*
(`Align_ULCNet/ulcnet_prepost.h`, `DeepVQE_S/deepvqe_prepost.h`) that owns
everything between the caller's spectra and the accelerator's tensors behind
one opaque object. The classes are frequency domain only: the application
frames with audiolib (an `AcStft` per input stream, one `AcIstft` for the
output, with the framing each model was trained on, stated in
`ulcnet_process.h` and `deepvqe_prepost.h`), and the library holds no window,
FFT or STFT object. The lifecycle:

1. The grid is a run-time value. Fill a `UlcnetGridSpec` / `DeepVqeGridSpec`
   (`sample_rate`, `n_fft`, `delay_depth`) with what the graph was exported for
   and call `ulcnet_grid_make()` / `deepvqe_grid_make()`; one binary serves 16
   and 48 kHz.
2. `*_prepost_config_defaults()` copies the grid into the config. The recurrent
   state belongs to the application: a float host registers its buffers
   (borrowed, element counts, all or none: four for Align-ULCNet, the sixteen
   `DeepVqeStateId` entries for DeepVQE-S), a board registers nothing and its
   accelerator holds the state.
3. `_get_mem_size`/`_init` on a caller pool (16-byte aligned) or
   `_create`/`_destroy` on the heap. The pool holds no recurrent state, and a
   registered buffer is in neither its size nor its hash; `_init_ex` refuses a
   pool recorded for another grid, D or build.
4. Per hop `pre_process_freq` -> `frame_open` -> (accelerator, with every state
   output bound to the address of its input) -> `frame_finish` or
   `frame_bypass` -> `post_process_freq`.

```c
UlcnetGridSpec spec = { .sample_rate = 16000, .n_fft = 512, .delay_depth = 8 };
UlcnetGrid grid;
UlcnetPrepostConfig cfg;
UlcnetPrepostMemReq req;
UlcnetPrepost *p;
UlcnetModelIoInputs in;
UlcnetModelIoOutputs out;
AcStftConfig sc;
AcStft *stft_err, *stft_far;
AcIstft *istft;

ulcnet_grid_make(&grid, &spec);                 /* what the graph was exported for */
ulcnet_prepost_config_defaults(&cfg, &grid);
cfg.key_history = key;     cfg.key_history_elements = key_n;      /* float host: all four buffers, */
cfg.value_history = value; cfg.value_history_elements = value_n;  /* a board leaves them NULL/0    */
cfg.logit_history = logit; cfg.logit_history_elements = logit_n;
cfg.h_gru = gru;           cfg.gru_hidden_elements = gru_n;
ulcnet_prepost_get_mem_size(&cfg, &req);
p = ulcnet_prepost_init(pool, req.bytes, &cfg); /* pool: 16-byte aligned, req.bytes */

ac_stft_config_default(&sc, grid.n_fft);        /* sqrt-Hann, center 0, scales 1 */
stft_err = ac_stft_init(mem_a, ac_stft_get_mem_size(&sc), &sc);
stft_far = ac_stft_init(mem_b, ac_stft_get_mem_size(&sc), &sc);
sc.ola = AC_OLA_ENV_NORM;
istft = ac_istft_init(mem_s, ac_istft_get_mem_size(&sc), &sc);

/* per hop */
ac_stft_push_planar(stft_err, err_hop, err_re, err_im);
ac_stft_push_planar(stft_far, far_hop, far_re, far_im);
ulcnet_prepost_pre_process_freq(p, err_re, err_im, far_re, far_im);
ulcnet_prepost_frame_open(p, &in, &out);
if (run_accelerator(&in, &out) != 0 ||          /* each state *_out has its input's address */
    ulcnet_prepost_frame_finish(p) != 0)
    ulcnet_prepost_frame_bypass(p);             /* fail open; DeepVQE-S mutes */
ulcnet_prepost_post_process_freq(p, out_re, out_im);
ac_istft_push_planar(istft, out_re, out_im, hop_out);
```

Both classes also expose `*_state_generation()`, which counts the times the C
side restarted the state (reset, refused `frame_finish`; 0 after init, whether
or not state is registered): a board with fixed-point accelerator state loads
quantized zeros at start and whenever the value differs from the one it last
saw. Both are one inference per hop from the first hop. The Align-ULCNet error
spectrum is `AecResContext.error_spec` (the analysis seam, never
`aec_get_output_base().base_spec`, the rendering base) or, on the 4-channel
path, the sample beam helper's beamformed spectrum from
`mcab_sample_beam_process()` or the frame `mcab_fuse()` returns; those spectra
are already in this center-0 framing. `ulcnet_prepost.c` builds on
`ulcnet_model_io.c` (state and mask handling: Align_ULCNet/README.md) and
`deepvqe_prepost.c` on `deepvqe_process.c`; those keep their own standalone
parity tests, and the class gates are `tests/test_ulcnet_prepost_c.py` and
`tests/test_deepvqe_prepost_c.py`.
DeepFilterNet2's class, `../AINR/DeepFilterNet2/dfn2_prepost.h`, has the same
shape with two differences an integrator must know: its first `mask_lookahead`
hops return 0 (the graph needs its right-hand neighbours before it can run; none
for a grid with no lookahead), and its spectra are torch.stft normalized=True
(the application scales by `grid.stft_scale` = 1/sqrt(n_fft) in and
`grid.stft_unscale` out, see `dfn2_process.h`), so an AIAEC spectrum
(unnormalised, on the AIAEC grid) cannot be chained into it as it is.

Per-model call order (with error handling), sequence diagrams and the external/internal
parameter tables are in the HTML site, one page per model:
[`aiaec_align_ulcnet_usage`](../docs/html/aiaec_align_ulcnet_usage.html),
[`aiaec_deepvqe_s_usage`](../docs/html/aiaec_deepvqe_s_usage.html),
[`aiaec_align_cruse_usage`](../docs/html/aiaec_align_cruse_usage.html),
[`aiaec_cagcrn_usage`](../docs/html/aiaec_cagcrn_usage.html). Align-CRUSE and CAGCRN have no C
pre/post class; their pages describe the Python reference and the ONNX boundary.

## Tests

From Audio_ALG/:

    python3 -m pytest AIAEC
    make -C AIAEC test

AIAEC/wav_testset/ is a local, gitignored Align-ULCNet listening-demo clone;
it is not part of the release.
