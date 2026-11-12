# Align-ULCNet

Hybrid `linear AEC -> joint RES + NR` reference candidate. Inputs are the
**frozen production linear-AEC error** and far-end reference. That frontend
runs on the context-only seam (`enable_res=0`, `return_res_context=1`) and the
error is its analysis seam (`get_formed_output()`, the C `formed_hop`): the real
selected linear residual of every hop, which can exceed the capture on a hop
where the rendering guard of `lib/aec` would publish the capture. The guard
selects the rendering base only and never alters this input; see
`dataset_gen/README.md`, "The `linear_error` front-end contract". The target is
the common denoised, echo-free, early/dereverberated near-end speech. It is
not a direct neural AEC. This target is the project's comparison contract and
should not be reported as an upstream checkpoint-equivalent setting.

## Grid and training

The historical 16 kHz dataset default uses FFT/window/hop 512/512/256; materialization, packing, training and Python inference also accept 256/256/128. Keep the two corpora/checkpoints separate and match the model grid to its materialized frontend. The streaming ONNX exporter accepts three deployed profiles: 16 kHz/256, 16 kHz/512 and 48 kHz/1024; any other grid is refused. The 48 kHz grid uses 1024/1024/512 and needs a checkpoint trained for its frequency dimensions. The checkpoint, ONNX and runtime must agree on attention history `max_delay_frames` (D). The matched-filter count in the upstream AEC is independent of D. [Delay profiles and memory](reference/delay.md)

The supplied config uses Adam, peak LR `4e-3`, and STFT-consistent PLCPA with power `0.3` and complex weight `0.7`. It selects the best checkpoint by validation. [Training and dataset](../reference/training.md)

## Release data kept out of the customer manual

The customer manual lists only `{sr, fft, hop_in}`. The rest is resolved by the compiled-in catalog of `stream/aiaec/aiaec.c` and the product binding in `stream/aiaec/aiaec_product.h` (`AIAEC_PRODUCT_MODEL_ID`, 10 = Align-ULCNet; a single-model build `AIAEC_MODEL=align` binds it by default): the graph path per `(sr, fft, N)` and the delay depth. The delay depth is `D = ceil(0.5 s * sr / model_hop)`: 63 at 16 kHz / FFT 256 (hop 128) and 32 at 16 kHz / FFT 512 (hop 256). The exported alignment depth must equal it; when the checkpoint's depth differs, pass `--max-delay-frames D` to export and calibration (for example `--max-delay-frames 32` for a 16 kHz / FFT 512 export). Quantization choices and the physical graph files are release data in the catalog, not manual content.

## Graph port order

Inputs are the signal inputs followed by the four state inputs; outputs are the head (`output`, the planar complex mask) followed by
the four state outputs named `<input>_out`; state `k` is input `signal_in + k` and output `1 + k` with equal shapes. The signal
inputs are `error_mag`, `far_mag`, `error_cos`, `error_sin` (layout 18, `--frames-per-call 1`: 8 inputs, 5 outputs); at
`--frames-per-call N > 1` (layout 19) `frame_reset` `[1, N]` is appended as the last signal input (9 inputs, 5 outputs). One
description generator produces both. `stream/aiaec` keeps no port list: `AiaecWork` carries the block's signals and the head, and the
graph's wiring (including the four state shapes) is the board sample's model configuration (`stream/aiaec/ai3_model_io_alignulcnet.h`, shared helper `stream/common/ai3_model_io.h`), which binds by
position, never by name. The stream computes `frame_reset` from the per-hop events (hop 3 of a packet alone: `0 0 1 0 0 0 0 0`) and the
sample moves it like any other signal input. `AIAEC/tests/test_aiaec_bind_info_pin.py` compares the `AiaecWork` views of real pre calls
with a fresh export for N = 1 and N = 8 and asserts this name order. Fixed-point state ports need `zero_point` 0 at both ends
(the sample's `stream_check_ports()` refuses a graph that does not).

## Alignment events in Python (`--events`)

The C stream restarts the model on an alignment event: the linear AEC reports `(changed, solid)` per model hop, and a hop is an event when `changed`, or when `solid` turns 0 -> 1 after an earlier hop (`prev_solid` is carried across packets). An event hop resets the model state at that hop and is a residual bypass (enhanced frame = error spectrum); at `--frames-per-call N > 1` it is the packet graph's `frame_reset[1, N]` control (1.0 on the event hop). The Python PBFDKF has no delay-state accessor, so Python cannot derive the pairs from the mic/far WAVs; it replays them from a record file. The whole rule lives once, in `alignment_events.py` (`EdgeTracker`); inference and calib both use it, so N = 1 and N = 8 see the same events.

```
python3 AIAEC/Align_ULCNet/inference.py checkpoint.pth mic.wav far.wav out.wav --frames-per-call 8 --events mic.events.json
python3 AIAEC/Align_ULCNet/inference.py calib ... --frames-per-call 8 --events RECORD_OR_DIR
```

Record file: one record per real model hop, in order (record k = STFT frame k = the k-th model hop; "hop 3" is index 2). The count must equal `ceil(samples / hop_len)` at the model rate; a padding hop never belongs in the record, and frames after the last record (the analyser's tail frame, packet padding) are never events.

| format | content |
|---|---|
| `.json` | `{"schema": "aiaec-align-events-v1", "events": [[changed, solid], ...]}`, a bare list, or items `{"changed": 0, "solid": 1}` |
| `.csv` | header `changed,solid` (an optional `hop` column must count 0, 1, 2, ...) |
| `.npz` | 1-D integer arrays `changed` and `solid` |

Values are 0/1. No file carries a precomputed restart flag. Behaviour:

- Inference prints `alignment events: ...` and, at N > 1, `packet K frame_reset 00100000` for the first packet and every packet holding an event. The streaming STFT/ISTFT keep running through an event (the C `_freq` flow); the PCM-family restart of the stream's own analysis is not mirrored. Not valid with `--mic-dir`.
- Calib takes a file (one pair) or a directory with `<relative wav name without .wav>.events.json|csv|npz` per pair; a missing or wrong-length record is an error. N > 1 captures `frame_reset` as exact 0.0/1.0 (policy `binary_control_no_ptq`, no range, no quantization). N = 1 has no control input: an event hop restarts the state and is not captured (it is a bypass the accelerator never runs) and does not count in `--frames`. The report gains `alignment_events` with the number of replayed event frames. With `--events` the last real hops are kept: a short final packet is zero-completed to N and its padding slots carry `frame_reset` 0, so an event on the last real hop is captured.
- Without `--events` nothing changes: no event ever occurs, only whole packets are captured (a short tail is dropped, as before) and the outputs are bit-identical to before.
- Verified with random weights against the C `_freq` stream and ONNX Runtime (`tests/test_aiaec_packet_ort_events.py`); there is no formal quantized graph or board run yet.

## Deployment reference

Connect the linear AEC’s float residual and the raw far-end hop with `ulcnet_stream_pre_process_f32()`. Each hop can return a real inference job or an alignment reprime. Follow the PCM SDK’s state-generation and error-recovery contract. [PCM stream and frequency-domain APIs](../../docs/html/models/align-ulcnet/index.html)

- [Inference parity and alignment events](reference/alignment.md)
- [Choosing a delay profile](reference/delay.md)
- [Stateless graph, state tensors and calibration](reference/deployment.md)
