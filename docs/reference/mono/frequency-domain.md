# Frequency-domain mono pipeline design

The current conventional mono path performs linear echo cancellation first,
then calculates one final complex spectrum from two suppression gains. RES is a
gain source inside the AEC seam, not a second time-domain filter after NR.

## Data flow

```text
mic/ref
  → linear AEC (PBFDKF + shadow + delay/EPC)
      ├─ E(f): linear residual (analysis seam)
      └─ AecResidualView
           ├─ B(f): rendering base (base_spec; E on every hop the output guard leaves alone)
           ├─ G_res(f): AEC3 suppression gain
           ├─ R²(f): residual-echo PSD
           ├─ comfort-noise PSD
           └─ far-end activity

B(f), R²(f)
  → echo-aware MMSE-LSA
  → G_nr(f)

G_total(f) = min(G_nr(f), G_res(f))
  → B(f) × G_total(f) + CNG
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
(`AEC_CNG_NR_GAIN_FLOOR` of `aec.h`, applied inside `aec_apply_cng()`, in the mono and the 4-channel examples): the denoiser never sees the comfort noise, so applying
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

In the mono example `derive_lane()` resolves the rate's grid once with lib/aec's
`aec_resolve_signal_grid()`; the AEC lane, the denoiser (`anr_config_for_level_grid()`) and the synthesis
(`ac_stft_config_default()`) are all built from that grid's `fft_size`, so they share one set of
dimensions by construction.

The four-channel array (an `Aec` with `AecConfig.n_mics = 4`, `pipelines/4ch/`) uses the same three primary product grids
(16k/256/128, 16k/512/256, 48k/1024/512). AIAEC models remain a separate
training/checkpoint contract even when a numeric grid happens to match.

## Public implementation boundary

There is no mono pipeline library. The reusable parts are separate libraries, each
with its own size query and `init`: lib/aec's `Aec` (`aec_process_linear()` and `aec_process_residual()`),
the denoiser (`anr.h`) and `AcIstft`; the mono RES gain is the `Aec`'s own (`AecResidualView.res_gain`,
read with `aec_get_residual()`, applied with `aec_apply_gain()` / `aec_apply_cng()`, strength through `aec_set_level()`). Each example `main`
initialises the libraries it uses and writes the per-hop sequence itself; the example headers
(`pipelines/common/example_cli.h`, `pipelines/4ch/array_sample_post.h`) carry no ABI and
no layout version. The only multi-channel piece is
the 4-channel array instance of the same library (`AecConfig.n_mics = AEC_ARRAY_MICS`, `aec.h` "Array instance"): one
`aec_get_mem_requirements()` / `aec_init_ex()` places the four lanes, the shared delay estimator (MATCHED mode), the
alignment ring, the fusion buffers and the one gain stage in a caller-owned 16-byte-aligned pool with a versioned
memory descriptor (`AecMemReq`; `aec_create()` is the heap convenience). It holds no
beamformer: `aec_process_residual()` takes the caller's weights, the gain stage (`AEC_RESIDUAL_FULL`) is the array's,
and `aec_apply_gain()` / `aec_apply_cng()` render it exactly as in mono. The SRP-PHAT/BF glue of the examples
is the sample helper `array_sample_beam.{h,c}`, which is outside the library. The mono CLI executables
(`aec_nr_pipeline`, `aec_nr_pipeline_static`, one `main.c` built twice) are
example programs that compose the libraries.

Low-level details, backend/release rules, memory sizing, levels, and tests are
maintained in [`../pipelines/README.md`](../../../pipelines/README.md).

## Stage chain

Each application writes its frequency-domain chain explicitly over a borrowed
view of the analysis frame (lib/aec's `AecResidualView`, for mono and for the 4-channel array alike):

```text
v        = analysis stage             (mono: aec_process_linear + aec_process_residual(aec, NULL) -> aec_get_residual,
                                       4ch: aec_process_linear + aec_get_array -> beamformer -> aec_process_residual(aec, &beam)
                                            -> aec_get_residual
                                            (examples: array_sample_beam_process* -> array_sample_beam_fuse))
res_gain = v.res_gain                         the AEC's own gain (mono) or the array's one gain stage over the
                                              fused spectra (4ch); aec_get_residual().res_gain
nr_gain  = mono_nr_gain(...)                  gain stages return const float*
           / array_sample_anr_gain(nr, &v, scratch, use_r2)
spec     = aec_apply_gain(aec, nr_gain, spec) P = v.base_spec * min(nr_gain, res_gain);
                                              estimators read v.error_spec / v.base_spec, render v.base_spec
spec    += aec_apply_cng(aec, nr_gain, spec)  comfort noise on the cut bins, same extra gain to both,
                                              the array's / Aec's own generator
                                              (--no-res: aec_apply_gain_spec with the unity vector, no comfort noise)
out      = ac_istft_push(spec)                one iFFT/OLA per hop
```

The frame is valid until its producer runs again and is never written through.
Stages that run a neural model are split around the model call
(`ulcnet_nn_stage_begin_spec` / `dfn_hop_begin`, `nn_begin`, caller runs the NPU, `nn_end`, `ulcnet_nn_stage_end_spec` / `dfn_hop_end`; the ULCNet stage takes and returns spectra and the application runs its own `AcStft` / `AcIstft`);
a failed, non-finite or never-fetched model frame takes the identity, so the
output timing does not depend on the model's health. There is no function-pointer
table: the call order is part of each application's source.

[返回本主題](README.md)
