# AIAEC ONNX export and calibration

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

[返回本主題](../README.md)
