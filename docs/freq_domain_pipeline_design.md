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
  → E(f) × G_total(f) + CNG
  → one final iFFT/OLA
```

## Why the gains are fused

`G_nr` has the noise estimate and speech-presence model. `G_res` has AEC
convergence and residual-echo information that NR cannot reconstruct. Feeding
`R²` into the NR prior allows NR to help on echo-dominated bins, but does not
make `G_res` redundant.

There is no near-end floor: `G_total` is applied as computed. Which bins
survive is decided per bin by the denoiser's own speech-presence model and by
`G_res`, so no broadband level can cap the denoiser or make the noise floor
follow the far talker. Comfort noise is driven by the AEC suppression gain so
it fills echo-removal holes rather than every bin attenuated by NR, and its
level is scaled by the denoiser gain of the bin, bounded below at -10 dB
(`CNG_NR_GAIN_FLOOR`): the denoiser never sees the comfort noise, so applying
its gain here keeps the fill near the denoised floor instead of the ambient
level, while the bound keeps enough of it to mask the residual echo.

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

There is no mono pipeline library. The reusable parts are separate libraries, each
with its own size query and `init`: lib/aec's `Aec` (`aec_process_context()`),
`common/se_stage.h` (`SeSpecFrame`, `se_frame_from_aec()`, `se_res_stage_gain()`), the denoiser
(`anr.h`) and `AcIstft`; the mono RES gain is the `Aec`'s own (`ctx.res_gain`, exposed as
`frame.g_res_ctx`, strength through `aec_set_preset()`). Each example `main` initialises the
libraries it uses and writes the per-hop sequence itself; the only shared example header is
`pipelines/samples/se_anr_stage.h` (no ABI, no layout version). The only wrapped unit is the
4-channel linear AEC + beamformer + post-beam RES, `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.h`
(caller-owned 16-byte-aligned pool initialized from a versioned memory descriptor, or a heap
convenience constructor; its own `AecRes` is reached through `audio_pipeline_4ch_res()` and
`audio_pipeline_4ch_fuse()` fills `frame.g_res_ctx`). The two mono CLI executables are example
programs that compose the libraries.

Low-level details, backend/release rules, memory sizing, presets, and tests are
maintained in [`../pipelines/README.md`](../pipelines/README.md).

## Stage chain

Each application writes its frequency-domain chain explicitly over a borrowed
view of the analysis frame (`pipelines/common/se_stage.h`, `SeSpecFrame`):

```text
frame = analysis stage           (mono: aec_process_context -> se_frame_from_aec,
                                  4ch: audio_pipeline_4ch_process_* -> audio_pipeline_4ch_fuse
                                       (or four_aec_front_process -> beamformer -> four_aec_fuse))
res_gain = frame.g_res_ctx                      mono: the AEC's own gain; 4ch unit: filled by fuse();
                                                a core-level example with its own AecRes calls
                                                se_res_stage_gain(res, frame) instead
nr_gain  = se_anr_stage_gain(nr, frame, scratch, use_r2)    gain stages return const float*
total    = sk_min_f32(nr_gain, res_gain)        a disabled source reads as the unity vector
spec     = sk_capply_gain_f32(frame.e, total)
spec    += aec_cng_fill(...)                    comfort noise on the cut bins
out      = ac_istft_push(spec)                  one iFFT/OLA per hop
```

The frame is valid until its producer runs again and is never written through.
Stages that run a neural model are split around the model call
(`ulcnet_nn_stage_begin*` / `dfn_hop_begin`, `nn_begin`, caller runs the NPU, `nn_end`, `end`);
a failed, non-finite or never-fetched model frame takes the identity, so the
output timing does not depend on the model's health. There is no function-pointer
table: the call order is part of each application's source.
