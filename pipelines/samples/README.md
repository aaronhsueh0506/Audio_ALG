# pipelines/samples

Examples, not library API: no ABI, no layout version, no descriptor, in no archive. Copy a
header into your project and change it. The pipeline example mains (`main.c`, `static_main.c`,
the raw and mcab drivers) include the headers of this directory to show how the separate
libraries chain; the production targets (the 4-channel unit `AudioPipeline4Ch` and its
archives) do not. Built and tested on its own (`make -C samples test`, also wired into the
top-level `pipelines/Makefile`'s `make test`).

Only the multi-channel linear AEC + beamformer is wrapped as one unit. RES, noise reduction,
comfort noise, synthesis and the neural post-filters are separate libraries that the
integrator initialises (each with its own `mem_size` and `init`) and chains.

| File | What it shows |
|---|---|
| `se_sample_chain.h` | The gain sequence from a `SeSpecFrame` to the output hop: RES gain, NR gain, `min`, apply to E, comfort noise on the RES-cut bins, synthesis. One caller pool; every stage is optional. |
| `se_anr_stage.h` | The NR gain call that folds the AEC residual-echo powers into the denoiser's noise floor (`se_anr_stage_gain`). The denoiser is the library preset (`anr_config_for_mode_grid()`). |
| `dfn_hop.h` | One DeepFilterNet2 hop: the 48 kHz stage with a caller-owned synthesis, or the rate bridge below 48 kHz; the phased begin / model / end calls. |
| `ulcnet_nn_stage.{h,c}` | The Align-ULCNet post-filter stage behind a linear AEC: far-end STFT, model call policy (reprime, alignment events, fail-open), WOLA. |
| `mcab_demo.c` | N linear AEC lanes wired to SRP-PHAT and the beamformer, without the production scheduling. |

`common/se_stage.h` carries `se_frame_from_aec()`, which fills the `SeSpecFrame` of a mono
`Aec` after `aec_process_context()`; the 4-channel unit fills the same view in
`audio_pipeline_4ch_fuse()`.

## mcab_demo (`mcab_*` = Multi-Channel AEC-BF)

A standalone demo wiring N independent AEC lanes directly to the SRP-PHAT DOA estimator and the
beamformer through their public APIs (`aec.h`, `srp.h`, `bf.h`), with no production scheduling
in between. Production code uses `AudioPipeline4Ch` (`4ch_aec_bf_nr_res`), not this file: the
demo omits duty-cycle scheduling, quarantine and shared-delay coordination, which
`4aec_nr_res.h` implements. See `mcab_demo.c`'s own top-of-file comment for the design
rationale.

## se_sample_chain.h

```c
SeSampleChainConfig cfg = se_sample_chain_default_config(16000, 256);
cfg.res = SE_SAMPLE_RES_AEC_RES;      /* OFF, FRAME_GAIN (the front end's own gain), AEC_RES */
cfg.res_preset = AEC_PRESET_BALANCED; /* AEC_RES only */
cfg.enable_nr = 1;  cfg.nr_mode = ANR_MODE_BALANCED;
cfg.enable_cng = 1; cfg.with_synthesis = 1;

size_t bytes = se_sample_chain_mem_size(&cfg);            /* 0: invalid configuration */
SeSampleChain* chain = se_sample_chain_init(pool, bytes, &cfg);   /* 16-byte aligned pool */

/* per hop, after the front end filled a SeSpecFrame `frame` */
const Complex* P = se_sample_chain_form(chain, &frame);   /* NULL: refused hop */
if (!P || se_sample_chain_synthesize(chain, P, out) != 0 || !se_hop_is_finite(out, hop)) {
    /* reset the front end and the chain together */
    se_sample_chain_reset(chain);
}
```

* A stage that is off reads as the unity vector. Comfort noise runs only with a RES source and
  `enable_cng`, at the level of the RES gain before the `min`.
* The RES strength is `aec_res_set_preset(se_sample_chain_res(chain), ...)`, the NR strength
  `anr_set_mode(se_sample_chain_nr(chain), mode)`: the denoiser is the library preset for
  `nr_mode` on the chain's grid.
* `SeSpecFrame` pointers are borrowed from the front end and valid until its next call; `P`
  must be formed in the same hop.
* Failure is the integrator's: after any refused hop, reset the front end, the chain and any
  post-filter stage together (`se_sample_chain_reset` also restarts the comfort-noise
  generator).

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
