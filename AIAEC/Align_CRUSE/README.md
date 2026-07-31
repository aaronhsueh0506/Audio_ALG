# Align-CRUSE

Direct neural AEC/RES candidate. There is no matched filter or linear AEC in
front of this network. Inputs are unaligned microphone and far-end spectra; the
output is a real magnitude mask applied to the microphone spectrum, preserving
microphone phase. For this project's **AEC-only** route the training target is
`near_speech + local_noise`, not clean speech.

Paper-aligned details:

- log-power features from microphone and far end;
- mic encoder channels `16,40,72,32`, far encoder `8,24`;
- causal `4x3`, frequency-stride-2 convolutions;
- causal-running soft delay distribution and weighted far-feature alignment;
- GRU bottleneck, trainable skip projections, decoder `32,48,48`;
- sigmoid magnitude mask with learned output gain.

The paper used 16 kHz, `320/320/160` and `dmax=100`. This repository preserves
the one-second delay span while using the project grid (`512/512/256` at 16 kHz
or `1024/1024/512` at 48 kHz), so the number of delay frames is derived from the
hop rather than copied as 100. The deployment default
`alignment_mode="causal_running"` emits `D[B,T,dmax]` using past evidence
only. Select `paper_global` explicitly to reproduce the paper's one
`D[B,dmax]` vector; that mode reads the complete utterance and is not causal.

All four frequency restorations use `1x3`, stride-2 transposed convolutions;
the fourth is the shape-restoring convolution in the mask block. The default
192-unit GRU keeps the 16 kHz project-grid graph in the paper's approximately
0.75 M parameter class. No author code/checkpoint was released, so exact
padding and projection width remain documented reconstruction choices.
