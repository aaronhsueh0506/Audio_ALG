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
(`CNG_NR_GAIN_FLOOR` in the mono example, `MCAB_SAMPLE_CNG_NR_GAIN_FLOOR` in
`pipelines/4ch_aec_bf_nr_res/mcab_sample_post.h`): the denoiser never sees the comfort noise, so applying
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

The four-channel flow `mcab` (`pipelines/4ch_aec_bf_nr_res/`) uses the same three primary product grids
(16k/256/128, 16k/512/256, 48k/1024/512). AIAEC models remain a separate
training/checkpoint contract even when a numeric grid happens to match.

## Public implementation boundary

There is no mono pipeline library. The reusable parts are separate libraries, each
with its own size query and `init`: lib/aec's `Aec` (`aec_process_context()`),
the denoiser (`anr.h`) and `AcIstft`; the mono RES gain is the `Aec`'s own (`AecResContext.res_gain`,
read with `aec_get_res_context()`, strength through `aec_set_level()`). Each example `main`
initialises the libraries it uses and writes the per-hop sequence itself; the example headers
(`pipelines/common/example_cli.h`, `pipelines/4ch_aec_bf_nr_res/mcab_sample_post.h`) carry no ABI and
no layout version. The only multi-channel piece is
`mcab`, the 4-channel flow layer, `pipelines/4ch_aec_bf_nr_res/mcab.h`: it is not a library that
builds its own parts. The application creates the four `Aec` lanes and, in MATCHED mode, the shared
`DelayAec`, and hands the pointers to `mcab_init_ex()`; mcab keeps only flow state (delay admission,
alignment ring, fuse buffers) in a caller-owned 16-byte-aligned pool initialized from a versioned
memory descriptor (or the heap convenience `mcab_create()` for that pool alone). It holds no
beamformer and no RES: `mcab_fuse()` takes the caller's weights and leaves `frame.g_res_ctx` NULL,
and a 4-channel application creates its own lib/aec `AecRes`. The SRP-PHAT/BF glue of the examples
is the sample helper `mcab_sample_beam.{h,c}`, which is outside the module. The mono CLI executables
(`aec_nr_pipeline`, `aec_nr_pipeline_static`, one `main.c` built twice) are
example programs that compose the libraries.

Low-level details, backend/release rules, memory sizing, levels, and tests are
maintained in [`../pipelines/README.md`](../pipelines/README.md).

## Stage chain

Each application writes its frequency-domain chain explicitly over a borrowed
view of the analysis frame (mono: lib/aec's `AecResContext`; 4ch: `McabFrame` of `mcab.h`):

```text
frame = analysis stage           (mono: aec_process_context -> aec_get_res_context,
                                  4ch: mcab_process_front -> beamformer -> mcab_fuse
                                       (examples: mcab_sample_beam_process* -> mcab_sample_beam_fuse))
res_gain = ctx.res_gain                         mono: the AEC's own gain (inside the AEC call);
                                                4ch (mcab): McabFrame.g_res_ctx is NULL, the application's
                                                own AecRes gives aec_res_process(res, &in) with
                                                in filled by mcab_sample_res_input(frame, &in)
nr_gain  = mono_nr_gain(...)                    gain stages return const float*
           / mcab_sample_anr_gain(nr, frame, scratch, use_r2)
total    = sk_min_f32(nr_gain, res_gain)        a disabled source reads as the unity vector
spec     = sk_capply_gain_f32(frame.b, total)      # estimators read frame.e; render frame.b
spec    += aec_cng_fill(...)                    comfort noise on the cut bins
out      = ac_istft_push(spec)                  one iFFT/OLA per hop
```

The frame is valid until its producer runs again and is never written through.
Stages that run a neural model are split around the model call
(`ulcnet_nn_stage_begin*` / `dfn_hop_begin`, `nn_begin`, caller runs the NPU, `nn_end`, `end`);
a failed, non-finite or never-fetched model frame takes the identity, so the
output timing does not depend on the model's health. There is no function-pointer
table: the call order is part of each application's source.
