# 4-channel array C API

## C API

The deployable C seam is the array instance of the AEC library:

- [`aec.h`](../../../../AEC/c_impl/include/aec.h) ("Array instance" section, in the sibling AEC checkout): the public API --
  `AecConfig` with `n_mics = AEC_ARRAY_MICS`, `residual`, the delay fields; the lifecycle (`aec_get_mem_requirements()` /
  `aec_get_mem_size()`, `aec_init_ex()` / `aec_init()` / `aec_create()`, `aec_reset()`, `aec_destroy()`); the two per-hop
  calls (`aec_process_linear()` + `aec_get_array()`, `aec_process_residual(&AecBeam)` + `aec_get_residual()`); the output
  stage (`aec_apply_gain()`, `aec_apply_cng()`, `aec_post_reset_serial()`,
  `aec_synthesis_window()`); the strength setter `aec_set_level()`; and the structural audit accessors
  (`aec_array_matched_filter_count()`, `aec_array_linear_aec_count()`, `aec_far_fft_real_compute_count()`,
  `aec_array_realign_{warm,soft}_lane_count()`, `aec_array_get_timing()`). The array has no beamformer; the
  lane parameters are the public fields of the same `AecConfig` (12 fields: the filter length and every other tuning value are the library's, set from the rate and the level), the lanes' own level has no effect;
- `AEC/c_impl/src/aec_array.c` (private to the AEC library): the flow over one shared `DelayAec` and four linear `Aec`
  lanes **that it creates itself in the caller's pool**, the shared-delay admission, the alignment ring, the fusion
  (coherent context projection) and, with `AEC_RESIDUAL_FULL`, the residual gain stage. The MMSE-LSA instance and the
  inverse transform are carved and chained by the applications below, behind the array. The library holds
  `aec_array_internal.h`, the delay-admission helpers and the byte-equivalent scalar/NEON projection kernels;
  its own tests are in the AEC repo (`test/test_aec_array.c`, `test/test_aec_array2.c`, `test/test_aec_array_flow.c`);
- [`array_sample_beam.h`](../array_sample_beam.h) also carries `array_sample_default_config()` (balanced level, MATCHED,
  bank 5, FULL, 4 mics); the application-side init is `aec_get_mem_requirements()` + `aec_init_ex()` in one caller block
  (see [`array_sample_unit.h`](../array_sample_unit.h));
- [`array_sample_beam.h`](../array_sample_beam.h) and [`array_sample_beam.c`](../array_sample_beam.c): sample code (in no
  archive), the optional glue that inserts the reusable SRP-PHAT/DOA smoothing and BF of `audiolib/{doa,bf}`
  between `aec_process_linear()` and `aec_process_residual()`. Per hop `array_sample_beam_process_front()` /
  `array_sample_beam_process_beam()` (or `array_sample_beam_process_with_activity()` /
  `array_sample_beam_process()`) give the beamformed spectrum and weights; `array_sample_beam_fuse()` runs
  `aec_process_residual()` and fills the `AecResidualView` for the frequency-domain stages. NR, comfort noise and
  synthesis are not inside it. An application with its own beamformer calls the `aec_*` functions directly;
- [`array_sample_post.h`](../array_sample_post.h): header-only sample code (in no archive) between `aec_get_residual()` and the
  libraries behind it -- `array_sample_anr_gain()` runs the denoiser on a fused view (folding `r2` into the noise floor
  when asked), `array_sample_post_reset_due()` tells the synthesis to follow a lane's post-chain reset, and
  `ARRAY_SAMPLE_PSD_SCALE_INV`; the estimators read `error_spec`, the synthesis renders `base_spec` (README:
  "Analysis and rendering spectra"). The comfort-noise floor is `AEC_CNG_NR_GAIN_FLOOR` and the generator is the array's;
- [`array_sample_unit.h`](../array_sample_unit.h): header-only sample code shared by `4ch_raw` and `4ch_static` -- the array
  plus beam helper unit (`array_sample_unit_plan()` / `_init()` / `_destroy()`) and the stages behind the fuse
  (`array_sample_stages_plan()` / `_init()` / `_chain()` / `_reset()`: NR gain, `aec_apply_gain()`, `aec_apply_cng()`, synthesis);
- [`array_sample_driver.h`](../array_sample_driver.h) and [`array_sample_driver.c`](../array_sample_driver.c): the CLI
  parsing and the lifecycle/hop-loop skeleton shared by the four 4-channel samples (sample code);
- [`4ch_external_static.c`](../4ch_external_static.c): caller-owned-pool example that spells the application-side
  init out (`flow_plan()` / `flow_init()` of its own), drives the array with fixed equal weights and carves, behind it in one
  block, the NR, the synthesis and the gain buffers (its own `post_plan()` / `post_init()` / `post_hop()`; `--no-res` selects
  `AEC_RESIDUAL_ESTIMATE`), following the same query → allocate → `init` →
  process → destroy → release sequence as the mono `mono/aec_nr_res/main.c` built as `aec_nr_pipeline_static`;
- [`4ch_raw.c`](../4ch_raw.c): raw-float host CLI for running the array, the sample beam helper and the per-hop chain
  (`array_sample_stages_chain()`) on recorded four-channel fixtures (the array and helper share one `posix_memalign` block and the post
  stages a second -- a one-shot host tool, not a board-deployment demo);
- [`4ch_static.c`](../4ch_static.c): caller-owned-pool example for the array plus the sample beam helper (real SRP-PHAT
  + real BF, not an externally-supplied fixed-weight stand-in) and the chain
  (`array_sample_stages_chain()`), following the same query → allocate → `init_ex` → process → destroy → release sequence as
  `4ch_external_static.c` above;
- [`tests/test_4ch_chain.c`](../tests/test_4ch_chain.c): the chain-level contracts that stay in this repo -- NR bypass identity,
  RES bypass passthrough, comfort-noise contract and its following of the NR gain, pool-placement invariance and
  reset-equals-fresh-instance of the whole chain;
  [`tests/conv_chain_4ch.h`](../tests/conv_chain_4ch.h) writes the post-beam chain over a second array as the reference
  the application tests compare against, and [`tests/array_test_rig.h`](../tests/array_test_rig.h) builds and tears down
  the array for the tests;
- [`tests/test_array_sample_beam.c`](../tests/test_array_sample_beam.c): sample beam helper lifecycle, topology, reset,
  split calls and their sequence guards, `AEC_RESIDUAL_STATE`, finite-output tests, `array_sample_beam_fuse()`
  publishing the gain, the level ramp against the reference chain, and the
  bit-exact equality of array + helper + `array_sample_beam_fuse()` + the examples' chain with the reference chain;
- [`tests/test_array_sample_post.c`](../tests/test_array_sample_post.c): the acceptance test of `array_sample_post.h`; it
  links no array object and is built by `pipelines/Makefile` (`make -C .. test_array_sample_post`);
- [`../../../audiolib/tests/test_spatial_third_party.c`](../../../../audiolib/tests/test_spatial_third_party.c):
  scalar/SIMD PHAT, cached SRP, exported-BF-weight and VAD pool equivalence
  tests; `make test` here runs them first (`test-third-party` is `make -C ../../../audiolib test`).

For side-by-side reading, the mono example and its four-channel counterpart:

| Mono example | Four-channel counterpart |
|---|---|
| lib/aec `Aec` (`aec_get_mem_size`, `aec_init`) | the same `Aec` with `n_mics = AEC_ARRAY_MICS` (`aec_get_mem_requirements`, `aec_init_ex`): four lanes and one shared `DelayAec` inside the pool |
| `aec_process_linear` + `aec_process_residual(aec, NULL)` + `aec_get_residual()` (`AecResidualView`) | `aec_process_linear` + `aec_get_array` → external beamformer → `aec_process_residual(&AecBeam)` + `aec_get_residual()` (`AecResidualView`; `array_sample_beam_fuse()`) |
| `Aec` carries the RES gain (`v.res_gain`, strength `aec_set_level()`) | the array's one gain stage (`AEC_RESIDUAL_FULL`; `v.res_gain`, strength `aec_set_level()`) over the fused spectra |
| `mono_nr_gain()` / `mono_form()` in `mono/aec_nr_res/main.c` | `array_sample_anr_gain()` and `post_hop()` in `4ch_external_static.c` |
| `mono/aec_nr_res/main.c` built as `aec_nr_pipeline_static` | `4ch/4ch_external_static.c` |
| `mono/aec_nr_res/main.c` built as the host runner `aec_nr_pipeline` | `4ch/4ch_raw.c` raw host runner |
| `mono/aec_nr_res/tests/test_mono_chain.c` | `4ch/tests/test_4ch_chain.c`, `test_array_sample_beam.c` |

| Mono | Four-channel array |
|---|---|
| `derive_lane()` + `aec_get_mem_size()` (`mono/aec_nr_res/main.c`) | `array_sample_default_config` / `aec_config_from_level` + `n_mics`, then `aec_get_mem_requirements` |
| `aec_init` | `aec_init_ex` (one call; the lanes and the estimator are built inside) |
| (heap flavor) | `aec_create` (`Aec* aec_create(const AecConfig*)`; `aec_destroy` frees the block) |
| `aec_process_linear` + `aec_process_residual(aec, NULL)` + `aec_get_residual` | `aec_process_linear` + `aec_get_array` → external beamformer → `aec_process_residual(&AecBeam)` + `aec_get_residual` |
| `aec_reset` / `aec_destroy` | `aec_reset` (also resets the lanes, the estimator and the gain stage) / `aec_destroy` (releases the lanes with the pool) |

The sample beam helper adds the same lifecycle one level up (`array_sample_beam_get_mem_size` /
`_init` / `_reset` / `_destroy`) around the SRP-PHAT and BF; it borrows the `Aec` array.

The array (the application's object):

```c
AecConfig cfg;
AecMemReq req;

aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);   /* or array_sample_default_config(&cfg, 16000) */
cfg.n_mics = AEC_ARRAY_MICS;
cfg.residual = AEC_RESIDUAL_FULL;     /* STATE: lanes only (the front-only seam the Align-ULCNet application uses) */
cfg.delay_mode = AEC_DELAY_MATCHED;   /* the shared alignment of the four lanes; FIXED / EXTERNAL_ALIGNED also valid */
aec_get_mem_requirements(&cfg, &req); /* 0 = OK, -1 = invalid; the whole array pool */
```

Heap convenience (the array with an external beamformer; the caller writes the post-beam stages out):

```c
Aec *array;                                        /* one heap block (Aec and pool); aec_destroy() frees it */
AecArrayView front;
AecResidualView v;
const float *nr;

array = aec_create(&cfg);                          /* NULL = refused or out of memory */
if (!array) return -1;
memset(&front, 0, sizeof front);  front.struct_size = sizeof front;
memset(&v, 0, sizeof v);          v.struct_size = sizeof v;

/* microphones is interleaved [hop][4], reference is [hop]. */
aec_process_linear(array, microphones, reference);
aec_get_array(array, &front);

/* External SRP-PHAT/BF consumes front.lane_error_spec and produces
 * channel-major Complex[4][n_freqs] weights (and, optionally, the mono error).
 * Convention: out[k] = sum(weights[ch,k] * in[ch,k]); no conjugation. */
external_srp_bf_update(front.lane_error_spec, weights);

AecBeam beam = { weights, NULL };                  /* NULL: rebuild E from the weights */
aec_process_residual(array, &beam);
aec_get_residual(array, &v);

/* The application chains its own stages over v (4ch_external_static.c post_hop() is the
 * example, tests/conv_chain_4ch.h the independent reference): */
nr = array_sample_anr_gain(my_nr, &v, scratch, 1);  /* array_sample_post.h; scratch: v.n_freqs floats; on v.error_spec */
if (array_sample_post_reset_due(&handled, &v))      /* a lane had a post-chain reset */
    ac_istft_reset(my_istft);                      /* the array reseeds its own comfort noise */
aec_apply_gain(array, nr, spec);                   /* P = v.base_spec * min(G_nr, G_res) */
aec_apply_cng(array, nr, spec);                    /* comfort noise on the bins the gain cut */
ac_istft_push(my_istft, spec, mono_output);

aec_destroy(array);                                /* the block is freed (lanes, estimator, window); the pointer is dead */
anr_destroy(my_nr);
```

A product that keeps linear AEC, RES, CNG and synthesis but omits MMSE-LSA simply never initialises the denoiser and
passes `NULL` for `extra_gain` (the example's `--no-res` selects `ESTIMATE` and applies a unity vector with
`aec_apply_gain_spec()` instead). The array itself has no NR or comfort-noise setting beyond `AecConfig`
(`enable_cng`, `level`). `residual` selects whether the array fuses at all and whether it carves the gain stage
(`AEC_RESIDUAL_STATE` is the front-only seam the Align-ULCNet application uses).

The sample beam helper does not apply a gain either: `array_sample_beam_fuse()` runs `aec_process_residual()`, fills the
`AecResidualView` and the gain stays in the array; the application calls `aec_apply_gain()` / `aec_apply_cng()` over it
every hop. A refused `array_sample_beam_fuse()` resets the whole helper (array, SRP-PHAT, BF) and is returned, except for a
sequence refusal, the mode refusal of an `AEC_RESIDUAL_STATE` array and the helper's own argument refusal (NULL arguments),
which change nothing; the application resets its own post stages (`array_sample_beam_reset()` does not touch the denoiser
or the synthesis). A `NULL` from `array_sample_anr_gain()` or a non-OK `aec_apply_*` is a refused hop and the array and every
post stage are reset together. Without the gain stage (`ESTIMATE`) the RES gain is the unity vector and `aec_apply_cng()` is not
called (comfort noise fills only RES-cut bins). `4ch_alignulcnet` builds the array with `cfg.residual = AEC_RESIDUAL_STATE`
and has no gain stage.

The applied gain is `min(G_nr, G_res)` (`aec_apply_gain()`) as computed. The post path has no
near-end floor or other level-dependent lift, so a loud background cannot cap
the denoiser or make the gain follow the far talker; which bins survive is
decided per bin by the denoiser's own speech-presence model and by `G_res`.

Caller-owned pool (static/board path):

```c
Aec *array;
void *pool;

/* cfg and req as above */
pool = platform_alloc(req.bytes, req.alignment);
array = aec_init_ex(pool, (size_t)req.bytes, &cfg, &req);

/* The per-hop aec_process_linear -> external beamformer -> aec_process_residual calls are identical. */

aec_destroy(array);                    /* does not release caller-owned memory */
/* then release the pool */
platform_free(pool);
```

The sample beam helper is placed in a caller-owned block of its own: `array_sample_beam_get_mem_size()` composes the
SRP/BF sub-modules' `*_get_mem_size()` with the helper's own scratch (the beamformed spectrum and the weights),
and `array_sample_beam_init()` places all of it, zero-heap, in that block on top of the array. The denoiser, the synthesis
and the gain buffers are a further block, each sized by its own query:

```c
ArraySampleBeamParams bp = array_sample_beam_default_params();
size_t beam_bytes;
array_sample_beam_get_mem_size(&cfg, &bp, &beam_bytes);
void *beam_pool = platform_alloc(beam_bytes, 16);
ArraySampleBeam *beam = array_sample_beam_init(beam_pool, beam_bytes, &cfg, &bp, array);

AnrConfig nr_cfg = anr_config_for_level_grid(cfg.sample_rate, cfg.fft_size, ANR_LEVEL_BALANCED);
AcStftConfig sy_cfg;  ac_stft_config_default(&sy_cfg, cfg.fft_size);
size_t nr_b = anr_get_mem_size(&nr_cfg), sy_b = ac_istft_get_mem_size(&sy_cfg);
void *post_pool = platform_alloc(ALIGN16(nr_b) + ALIGN16(sy_b) + gain_buffers, 16);   /* NR, synthesis, buffers */
AnrDenoiser *nr_inst = anr_init(post_pool, nr_b, &nr_cfg);
AcIstft *istft = ac_istft_init((uint8_t *)post_pool + ALIGN16(nr_b), sy_b, &sy_cfg);

/* per hop: array_sample_beam_process_with_activity() (or array_sample_beam_process_front() + array_sample_beam_process_beam()),
 * array_sample_beam_fuse(beam, &view), then
 * array_sample_anr_gain(nr_inst, &view, ...) (on view.error_spec),
 * array_sample_post_reset_due() (ac_istft_reset when it returns 1), aec_apply_gain(), aec_apply_cng(), ac_istft_push()
 * as array_sample_stages_chain() writes them. A refused call:
 * array_sample_beam_reset(beam), anr_reset(nr_inst), ac_istft_reset(istft) together. Strength:
 * aec_set_level(array, level), anr_set_level(nr_inst, level). */

array_sample_beam_destroy(beam); /* first: it borrows the array; does not release caller-owned memory */
aec_destroy(array);             /* then the array */
anr_destroy(nr_inst);
platform_free(post_pool);
platform_free(beam_pool);
platform_free(pool);
```

`aec_process_residual()` requires a linear frame processed since construction or the
last reset (`AEC_ERR_SEQUENCE` otherwise); a linear frame stays
valid until the next `aec_process_linear()` or `aec_reset()`,
and there is no token to return. Invalid weights leave the linear frame usable,
so the caller may correct and retry. `aec_create()` allocates one 16-byte-aligned block (the `Aec` and the array pool) and `aec_destroy()` frees it; the pointer is dead
afterwards and must not be freed. `aec_init_ex()` uses only the caller's aligned pool. Neither per-hop call performs
any allocation.

Build the applications or run the complete C acceptance gate from
`Audio_ALG/pipelines/4ch`:

```bash
make 4ch_external_static  # build caller-pool reference executable (array + chain)
make 4ch_static           # build caller-pool reference executable (array + sample beam helper + chain)
make 4ch_raw              # build the recording-validation runner (array + sample beam helper + chain)
make libs                  # the producer archives and the sample objects (array_sample_beam.o)
make test_4ch_chain             # build only the chain test binary (needs lib/aec and lib/nr, not the spatial kernels)
make test_array_sample_beam # build only the sample beam helper test
make test_4ch_dfn test_4ch_ulcnet   # the DFN2 / Align-ULCNet compositions
make 4ch_alignulcnet       # the Align-ULCNet board skeleton
make 4ch_aec_bf_dfn_res    # the DFN2 composition example
make test_4ch_dfn DFN2_MASK_LOOKAHEAD=2 DFN2_DF_LOOKAHEAD=4   # the DFN2 composition test at another lookahead pair
make test                  # audiolib's tests, then the 4ch tests and example smoke runs (see the Makefile)
make NO_STDIO=1 audit-no-stdio

# Query the exact pool budget for the supported grids.
"$(make -s print-bin-dir)/4ch_external_static" --print-mem-size --sample-rate 16000 --fft-size 256
"$(make -s print-bin-dir)/4ch_external_static" --print-mem-size --sample-rate 16000
"$(make -s print-bin-dir)/4ch_external_static" --print-mem-size --sample-rate 48000
"$(make -s print-bin-dir)/4ch_static" --print-mem-size --sample-rate 16000 --fft-size 256
"$(make -s print-bin-dir)/4ch_static" --print-mem-size --sample-rate 16000
"$(make -s print-bin-dir)/4ch_static" --print-mem-size --sample-rate 48000
```

`DFN2_MASK_LOOKAHEAD` (0..2) and `DFN2_DF_LOOKAHEAD` (0..4) default to the shipped pair 1/1. The DFN2 stage, the bridge and the
example main hold no pair (the application puts a `DFN2Grid` in the hop's pool plan, `DfnHopPoolPlan.stream_cfg.grid`, or in `DfnResStageConfig.grid` for the stage; the main reads it from `--ai-sample-rate` /
`--ai-fft-size`), so a different pair is passed as `-D` flags to the test unit `test_4ch_dfn` alone. The pair is part of the build
signature, so pairs never share an object directory. Only the 1/1 pair has a trained checkpoint, and the audio quality of the other pairs is not measured.

There is no 4-channel archive: the array is in `libaec.a`. The sample objects (`array_sample_beam.o`,
`array_sample_driver.o`), the example chain headers and the ULCNet stage object
(`ulcnet_nn_stage.o`) are compiled into the application binaries that use them; no
per-application archive is produced. A consumer of the array alone links `libaec.a`,
`libaudio_common.a` and `-lm` (`libanr.a` is for the programs that chain NR behind the array). A consumer of the sample
beam helper links `array_sample_beam.o`, then `libdoa.a`, `libbf.a`, `libspatial_common.a`,
`libaec.a`, `libanr.a`, `libaudio_common.a`, and `-lm`, in the order demonstrated by this directory's Makefile.

Each goal resolves, and keys its `bin/ne10-<sig>/` and `obj/ne10-<sig>/` directories on, only the producers it
needs: `test_4ch_chain` needs common, lib/aec and lib/nr (its reference chain calls the NR; the spatial kernels may be absent),
every other goal all of them; a goal missing a producer it needs stops and names it. The
same goal built in a narrower scope lands in its own directory. `make -s print-bin-dir` / `print-obj-dir` print the
directory of `make all`, and `PRINT_FOR=<goal>` the one of that goal:

```bash
make test_4ch_chain && "$(make -s print-bin-dir PRINT_FOR=test_4ch_chain)/test_4ch_chain"
```

[返回本主題](../README.md)
