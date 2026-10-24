# pipelines/samples

Examples, not library API: no ABI, no layout version, no descriptor, in no archive. Copy a
header into your project and change it. The pipeline example mains (`main.c`, `static_main.c`,
the raw and mcab drivers) include the headers of this directory for the stage calls they
chain; the production targets (the 4-channel unit `AudioPipeline4Ch` and its archives) do not. Built and tested on its own (`make -C samples test`, also wired into the
top-level `pipelines/Makefile`'s `make test`).

Only the multi-channel linear AEC + beamformer (with its post-beam RES) is wrapped as one unit.
Noise reduction, comfort noise, synthesis and the neural post-filters are separate libraries
that the integrator initialises (each with its own `mem_size` and `init`) and chains. There is
no shared chain header: each example `main` initialises the libraries it uses and writes the
per-hop sequence itself (RES gain, `se_anr_stage_gain()`, `sk_min_f32()`,
`sk_capply_gain_f32()`, `aec_cng_fill()`, `ac_istft_push()`), for instance `mono_form()` in
`mono_aec_nr_res/main.c` and `raw_chain()` in `4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c`.

| File | What it shows |
|---|---|
| `se_anr_stage.h` | The NR gain call that folds the AEC residual-echo powers into the denoiser's noise floor (`se_anr_stage_gain`), and the PSD scale (`se_psd_scale()`). The denoiser is the library preset (`anr_config_for_mode_grid()`). |
| `dfn_hop.h` | One DeepFilterNet2 hop: the 48 kHz stage with a caller-owned synthesis, or the rate bridge below 48 kHz; the phased begin / model / end calls. |
| `ulcnet_nn_stage.{h,c}` | The Align-ULCNet post-filter stage behind a linear AEC: far-end STFT, model call policy (reprime, alignment events, fail-open), WOLA. |
| `mcab_demo.c` | N linear AEC lanes wired to SRP-PHAT and the beamformer, without the production scheduling. |

`common/se_stage.h` carries `se_frame_from_aec()`, which fills the `SeSpecFrame` of a mono
`Aec` after `aec_process_context()` (its `g_res_ctx` is the AEC's own RES gain); the 4-channel
unit fills the same view in `audio_pipeline_4ch_fuse()` (its `g_res_ctx` is the unit's post-beam
RES gain). It also carries `se_hop_is_finite()`, the finite check on an emitted hop.

## mcab_demo (`mcab_*` = Multi-Channel AEC-BF)

A standalone demo wiring N independent AEC lanes directly to the SRP-PHAT DOA estimator and the
beamformer through their public APIs (`aec.h`, `srp.h`, `bf.h`), with no production scheduling
in between. Production code uses `AudioPipeline4Ch` (`4ch_aec_bf_nr_res`), not this file: the
demo omits duty-cycle scheduling, quarantine and shared-delay coordination, which
`4aec_nr_res.h` implements. See `mcab_demo.c`'s own top-of-file comment for the design
rationale.

## ulcnet_nn_stage

```c
size_t bytes = ulcnet_nn_stage_mem_size(0);          /* 1: also an STFT for begin_time() */
UlcnetNnStage* stage = ulcnet_nn_stage_init(pool, bytes, 0);
ulcnet_nn_stage_set_model(stage, &model);            /* UlcnetModel; identity when unset */

/* per hop: err = the linear-error spectrum (a 4-channel unit's beam.spectrum, or the STFT of
 * a mono lane's formed hop), far = the same call's raw far-end samples */
UlcnetNnDelay delay = { front.delay.changed, front.delay.solid };
ulcnet_nn_stage_begin_complex(stage, beam.spectrum, far, delay);
if (ulcnet_nn_stage_nn_begin(stage, &job) == 1)
    ulcnet_nn_stage_nn_end(stage, run_network(&job));
ulcnet_nn_stage_end(stage, out);                      /* hop #0 is zeros */
```

A mono lane maps `AecLinearContext.delay_state` to `{ state == CHANGED, state != UNLOCKED }`
and may pass `formed_hop` to `ulcnet_nn_stage_begin_time()` on a stage built with an error
analysis. One build serves one grid: `ULCNET_GRID=16k` (16 kHz / fft 512, default) or `48k`
(48 kHz / fft 1024).

## Build

```
make                 # mcab_demo, test_ulcnet_nn_stage, test_dfn_hop (SIMD=1 default)
make SIMD=0          # scalar spatial/SIMD kernels
make ULCNET_GRID=48k # the other Align-ULCNet grid
make test            # build + run; every program exits nonzero on a failed check
make clean
```
