# DeepFilterNet3

DFN3 band-split implementation alongside DFN2 cascade/alpha for controlled
comparison. This custom port uses the current AINR training, calibration, ONNX
and audiolib C interfaces. Release history is in
[`../CHANGELOG.md`](../CHANGELOG.md).

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
is the shipped one. The shared rules (window, build flags, guards, tests) are in
[`../README.md`](../README.md#lookahead-pairs-deepfilternet2-and-deepfilternet3).
Specific to DFN3:

- DFN3 is a parallel split: the deep filter reads the raw spectrum, so output
  frame `t` needs the heads of `t` (input up to `t+M`) and the spectrum up to
  `t+D`. The output is `max(M, D)` frames behind the input and the end-to-end
  latency is `1 + max(M, D)` hops. The first M hops only fill the window (no
  accelerator run); with `D > M` the next `D - M` hops run the accelerator and
  emit nothing, and the heads wait in the ring until their output frame.
- The ring holds `DF_ORDER + max(M, D) - D` frames (`DFN3_DF_RING`): `HISTORY`
  before the delayed frame, `max(M, D)` after it, and the frame itself (five for
  the shipped pair, up to seven for `D = 0, M = 2`). The state struct, the pool
  size and the pool hash follow the pair.
- `dfn3_compose()` (aligned reference) pairs a spectrum with the heads passed
  in the same call; the accelerator path is `dfn3_compose_stream()`.

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

For C deployment, `--runtime-bins` writes `erb_fwd.bin` in bin-major layout
and `erb_inv.bin` in band-major layout. The separate `--output-dir` mode is
a toolchain matrix bundle and is not interchangeable with those runtime files.
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

- `dfn3_process.c/.h`: independent feature, composition and framing reference.
- `dfn3_model_io.c/.h`: feature windows, accelerator bindings and state checks.
- `dfn3_prepost.c/.h`: static-pool/host lifecycle, frequency or time I/O (time I/O synthesises through the
  application's `AcIstft`, `cfg.istft`, built from `dfn3_framing_config()`), delayed
  head handoff, reset/skip handling and WOLA using current audiolib APIs.

Use `dfn3_compose_stream()` at the accelerator boundary; `dfn3_compose()` is
the aligned arithmetic reference. The streaming API applies each head to the
frame it describes, returns its output frame index and validates frame order.
The low-band ring holds raw spectra; the high-band mask stays aligned to the
same output frame. In time mode the accelerator still needs to run during
zero-hop tail flushing; see `dfn3_prepost.h` for the lifecycle and flush contract.
Both DFN2 and DFN3 frequency I/O use normalized STFT spectra. Their internal
feature analysis scale is separate and must not be applied to the audio path.

Build and test from AINR:

```bash
make WERROR=1 lib
make WERROR=1 test-simd
make SIMD=0 WERROR=1 test
python3 -m pytest DeepFilterNet2/tests DeepFilterNet3/tests \
    tests/test_dfn_comparison.py tests/test_dfn3_stream_c.py \
    tests/test_dfn3_prepost_c.py -q
```

`libainr_prepost.a` contains independent DFN2, DFN3 and GTCRN symbols. The
DFN3 C tests cover changing masks, nonzero future taps, Python/C parity,
identity audio latency, reset/rejected-head behavior and scalar/SIMD agreement.
`tests/test_dfn3_prepost_c.py` pins the grid accessors and the between-hops
ERB-matrix swap contract (including its refusals) through ctypes.
The upstream numerical alignment is in
[`UPSTREAM_ALIGNMENT.md`](UPSTREAM_ALIGNMENT.md).
