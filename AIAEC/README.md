# AIAEC

Choose the model by the signal available at the application boundary. The accelerator returns a mask or complex filter taps together with its next state. The host C code composes the enhanced spectrum.

| Model and training | Inputs | C integration |
|---|---|---|
| [Align-ULCNet](Align_ULCNet/README.md) | Linear AEC residual and raw far end | [PCM SDK](../docs/html/models/align-ulcnet/index.html) accepts float AEC hops or int16. A caller with spectra can use the frequency core. |
| [DeepVQE-S](DeepVQE_S/README.md) | Microphone and far end | [PCM SDK and frequency core](../docs/html/models/deepvqe-s/index.html), using complex CCM taps |
| [Align-CRUSE](Align_CRUSE/README.md) | Microphone and far end | [Real-mask helper](../docs/html/models/align-cruse/index.html). The application owns framing, graph I/O and state. |
| [CAGCRN](CAGCRN/README.md) | Microphone and far end | [Complex-mask helper](../docs/html/models/cagcrn/index.html). The application owns framing, graph I/O and state. |

For C integration, initialize once per stream. A device-owned recurrent state must match the generation returned with a real job. If an accelerator run may have written state before reporting failure, the adapter reloads quantized zero before the next inference. Each model’s deployment page gives its return values and output policy.

- [Dataset, training commands and losses](reference/training.md)
- [ONNX export, checkpoint compatibility and calibration](reference/export.md)
- [Dataset generator](dataset_gen/README.md)
