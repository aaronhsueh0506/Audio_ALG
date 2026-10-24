# pipelines/samples

Examples, not library API: no ABI, no layout version, no descriptor, in no archive. Copy a
header into your project and change it. The pipeline example mains (`main.c`, `static_main.c`,
the raw and mcab drivers) include the headers of this directory for the stage calls they
chain; the production targets (the 4-channel unit `AudioPipeline4Ch` and its archives) do not. Built and tested on its own (`make -C samples test`, also wired into the
top-level `pipelines/Makefile`'s `make test`).

Only the multi-channel linear AEC + beamformer is wrapped as one unit; it holds no RES.
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
`Aec` after `aec_process_context()` (its `g_res_ctx` is the AEC's own RES gain; mono keeps the RES inside the
AEC call); the 4-channel unit fills the same view in `audio_pipeline_4ch_fuse()` with `g_res_ctx` NULL, and the
4-channel examples create their own libaec `AecRes` (`aec_res_init()`) and call `se_res_stage_gain()` on that view.
It also carries `se_hop_is_finite()`, the finite check on an emitted hop.

## How the examples are called

| Header | Sequence (app-side order) | Return values |
|---|---|---|
| `se_anr_stage.h` | `anr_config_for_mode_grid()` -> `anr_get_mem_size()` -> `anr_init()`; per hop `se_anr_stage_gain(nr, &frame, scratch, use_r2)`; on a refusal `anr_reset()` | gains owned by the denoiser, valid until its next call; `NULL` = a NULL argument or a refused hop |
| `dfn_hop.h` | `dfn_hop_init_stage()` (48 kHz, caller-owned `AcIstft`) or `dfn_hop_init_bridge()`; per hop `dfn_hop_begin(E, P)` -> `while (dfn_hop_nn_begin() == 1) dfn_hop_nn_end(rc)` -> `dfn_hop_end(out)`; `dfn_hop_reset()` on a refusal | `DFN_HOP_INVALID` (-1), `DFN_HOP_SEQUENCE_ERROR` (-2) |
| `ulcnet_nn_stage.{h,c}` | `ulcnet_nn_stage_mem_size()` -> `_init()` -> `_set_model()`; per hop `_begin()` (re/im arrays), `_begin_time()` or `_begin_complex()` -> `_nn_begin()` -> `_nn_end(rc)` -> `_end(out)`; `_reset()` on a refusal | `nn_begin` 1 = a frame to run, 0 = none (warm-up or reprime) |

All three are public only in the sense of "copy and adapt": external parameters are the arguments
shown (`use_r2`, the `AnrMode`, `atten_lim_db`, the ULCNet grid and `D`); the stage constants
(`SE_PSD_SCALE`, `ULCNET_NN_STAGE_REPRIME_FRAMES`, the DFN scale powers of two) are internal.

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
