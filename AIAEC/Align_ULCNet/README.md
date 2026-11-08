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

The 16 kHz grid uses FFT/window/hop 512/512/256. The 48 kHz grid uses 1024/1024/512 and needs a checkpoint trained for its frequency dimensions. The checkpoint, ONNX and runtime must agree on attention history `max_delay_frames` (D). The supplied training config uses D=32. The matched-filter count in the upstream AEC is independent of D. [Delay profiles and memory](reference/delay.md)

The supplied config uses Adam, peak LR `4e-3`, and STFT-consistent PLCPA with power `0.3` and complex weight `0.7`. It selects the best checkpoint by validation. [Training and dataset](../reference/training.md)

## Deployment reference

Connect the linear AEC’s float residual and the raw far-end hop with `ulcnet_stream_pre_process_f32()`. Each hop can return a real inference job or an alignment reprime. Follow the PCM SDK’s state-generation and error-recovery contract. [PCM stream and frequency-domain APIs](../../docs/html/models/align-ulcnet/index.html)

- [Inference parity and alignment events](reference/alignment.md)
- [Choosing a delay profile](reference/delay.md)
- [Stateless graph, state tensors and calibration](reference/deployment.md)
