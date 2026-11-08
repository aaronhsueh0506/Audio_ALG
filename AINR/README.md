# AINR

Standalone neural speech enhancement takes a microphone signal. Use the checkpoint, feature normalization, grid and output composition of the selected model.

| Model and training | Shipped sample rate / FFT / hop | C integration |
|---|---|---|
| [DeepFilterNet2](DeepFilterNet2/README.md) | 48 kHz / 1024 / 512 | [PCM SDK](../docs/html/models/dfn2/index.html) for one frame per invocation, or the [frequency API](DeepFilterNet2/reference/frequency-api.md) for single- and multi-frame graphs |
| [DeepFilterNet3](DeepFilterNet3/README.md) | 48 kHz / 1024 / 512 | [Frequency core](../docs/html/models/dfn3/index.html), raw low-band deep filtering plus high-band masking |
| [GTCRN](GTCRN/README.md) | 16 kHz / 512 / 256 | [Frequency core](../docs/html/models/gtcrn/index.html), complex mask and recurrent state |
| [RNNoise-ERB](RNNoise-ERB/README.md) | 16 kHz / 512 / 256 | [Frequency core](../docs/html/models/rnnoise-erb/index.html), ERB gains and GRU state |

DFN2 applies the ERB mask before low-band deep filtering and alpha blending. Each model README provides its training and export commands. The reusable C sources are in `lib/ainr/<model>/`.

- [Dataset preparation, training commands and shared schedule](reference/training.md)
- [Dataset generator](dataset_gen/README.md)
