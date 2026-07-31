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

The mono pipeline uses a 20 ms frame, 10 ms hop, and the next power-of-two FFT:

| Sample rate | Frame | Hop | FFT | Frequencies |
|---:|---:|---:|---:|---:|
| 8 kHz | 160 | 80 | 256 | 129 |
| 16 kHz | 320 | 160 | 512 | 257 |
| 48 kHz | 960 | 480 | 1024 | 513 |

AEC, final OLA, NR gains, and `AecResContext` are validated against the same
derived dimensions during initialization.

The `pipelines/4ch_pipelines/` shell and AIAEC model candidates intentionally use
zero-padding-free 50%-overlap grids. They are separate contracts and should
not be inferred from this table.

## Public implementation boundary

The reusable C API is implemented in:

- `pipelines/audio_pipeline.h`;
- `pipelines/audio_pipeline.c`;
- `libaudio_pipeline.a`.

It supports both a heap convenience constructor and a caller-owned,
16-byte-aligned pool initialized from a versioned memory descriptor. The two
CLI executables are reference wrappers around this implementation.

Low-level details, backend/release rules, memory sizing, presets, and tests are
maintained in [`../pipelines/README.md`](../pipelines/README.md). The original
pre-implementation design is archived at
[`archive/freq_domain_pipeline_design_2026_06_07.md`](archive/freq_domain_pipeline_design_2026_06_07.md).
