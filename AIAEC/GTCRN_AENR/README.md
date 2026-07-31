# GTCRN-AENR

Project variant for `linear AEC -> joint RES + NR`. It takes the production
linear error and far reference and predicts a complex ratio mask applied to the
linear error.

The audited standalone GTCRN is reused unchanged after its first layer:
ERB band merge/split, SFE, ShuffleNetV2-style group temporal convolutions, TRA,
two DPGRNN blocks, decoder and CRM arithmetic are identical. The only topology
change is the first convolution: `[mag,re,im]` plus SFE gives 9 channels for one
spectrum, so error + far conditioning gives 18 instead of 9 channels.

This model is intentionally locked to upstream `16 kHz, FFT/window/hop =
512/512/256`. It is not an authored GTCRN AEC paper/checkpoint; comparisons
must label it **GTCRN-AENR project variant**. A 48 kHz extension would require a
new ERB design and is not represented as upstream-equivalent.
