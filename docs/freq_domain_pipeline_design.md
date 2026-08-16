# Frequency-domain mono pipeline design

The current conventional mono path performs linear echo cancellation first,
then calculates one final complex spectrum from two suppression gains. RES is a
gain source inside the AEC seam, not a second time-domain filter after NR.

## Data flow

```text
mic/ref
  → linear AEC (PBFDKF + shadow + delay/EPC)
      ├─ E(f): linear residual
      └─ AecResContext
           ├─ G_res(f): AEC3 suppression gain
           ├─ R²(f): residual-echo PSD
           ├─ comfort-noise PSD
           └─ far-end activity

E(f), R²(f)
  → echo-aware MMSE-LSA
  → G_nr(f)

G_total(f) = min(G_nr(f), G_res(f))
  → far/near-gated near-end floor
  → E(f) × G_total(f) + CNG
  → one final iFFT/OLA
```

## Why the gains are fused

`G_nr` has the noise estimate and speech-presence model. `G_res` has AEC
convergence and residual-echo information that NR cannot reconstruct. Feeding
`R²` into the NR prior allows NR to help on echo-dominated bins, but does not
make `G_res` redundant.

The near-end floor lifts suppression toward unity only when the bin appears
safe. Its effective strength is gated by far- and near-end activity; it is not
a global minimum gain. Comfort noise is driven by the AEC suppression gain so
it fills echo-removal holes rather than every bin attenuated by NR.

## Signal grid

The mono pipeline uses an explicit power-of-two, zero-padding-free frame/FFT
with 50% overlap:

| Sample rate | Frame | Hop | FFT | Frequencies |
|---:|---:|---:|---:|---:|
| 8 kHz | 256 | 128 | 256 | 129 |
| 16 kHz (default) | 256 | 128 | 256 | 129 |
| 16 kHz (alternate) | 512 | 256 | 512 | 257 |
| 48 kHz | 1024 | 512 | 1024 | 513 |

AEC, final OLA, NR gains, and `AecResContext` are validated against the same
derived dimensions during initialization.

The `pipelines/4ch_aec_bf_nr_res/` shell uses the same three primary product grids
(16k/256/128, 16k/512/256, 48k/1024/512). AIAEC models remain a separate
training/checkpoint contract even when a numeric grid happens to match.

## Public implementation boundary

The reusable C API is implemented in:

- `pipelines/mono_aec_nr_res/audio_pipeline.h`;
- `pipelines/mono_aec_nr_res/audio_pipeline.c`;
- `libaudio_pipeline.a`.

It supports both a heap convenience constructor and a caller-owned,
16-byte-aligned pool initialized from a versioned memory descriptor. The two
CLI executables are reference wrappers around this implementation.

Low-level details, backend/release rules, memory sizing, presets, and tests are
maintained in [`../pipelines/README.md`](../pipelines/README.md). Superseded
pre-implementation proposals are retained in Git history.
