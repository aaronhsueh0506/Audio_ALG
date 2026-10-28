[Index](README.md) | previous: [Call sequence and parameter surface](README.md#call-sequence-and-parameter-surface) | next: [Python reference seam](README.md#python-reference-seam) <!-- split-nav -->
## C API

The deployable C seam is:

- [`mcab.h`](mcab.h): the public module API -- `McabConfig`, the configuration helpers
  (`mcab_default_config()`, `mcab_apply_lane_contract()`, `mcab_lane_config()`,
  `mcab_delay_estimator_config()`), the lifecycle, the two per-hop calls (`mcab_process_front()`,
  `mcab_fuse()`) and the queries and structural audit accessors. Its config carries only the flow's fields
  (grid, delay policy, `enable_post`); the module has no RES and no beamformer, and the lane parameters
  (level, `filter_length`) belong to the application;
- [`mcab.c`](mcab.c): the flow over one shared `DelayAec` and four linear `Aec` instances **that the application
  created** (passed in `McabLibs`), and the coherent context projection. The residual suppressor (`AecRes`,
  `aec_res.h`, libaec) is carved and chained by the applications below, behind the flow; so are the
  MMSE-LSA instance and the inverse transform. `libaudio_pipeline_mcab.a` contains `mcab.o` only;
- [`mcab_internal.h`](mcab_internal.h): private -- the delay-admission helpers and the byte-equivalent
  scalar/NEON complex projection, residual-vector, comfort-noise and weight-matrix gate kernels;
- [`mcab_sample_flow.h`](mcab_sample_flow.h) and [`mcab_sample_flow.c`](mcab_sample_flow.c): sample code (in no
  archive) that writes the application-side init sequence of `mcab.h` once -- four lanes
  (`mcab_lane_config()` + `aec_init()`), the shared estimator object and its pool, and mcab's pool, in one
  caller block (`mcab_sample_flow_get_mem_size()` / `_init()` / `_create()` / `_destroy()`);
- [`mcab_sample_beam.h`](mcab_sample_beam.h) and [`mcab_sample_beam.c`](mcab_sample_beam.c): sample code (in no
  archive), the optional glue that inserts the reusable SRP-PHAT/DOA smoothing and BF of `audiolib/{doa,bf}`
  between `mcab_process_front()` and `mcab_fuse()`. Per hop `mcab_sample_beam_process_front()` /
  `mcab_sample_beam_process_beam()` (or `mcab_sample_beam_process_with_activity()` /
  `mcab_sample_beam_process()`) give the beamformed spectrum and weights; `mcab_sample_beam_fuse()` fills the
  `McabFrame` for the frequency-domain stages (`frame.g_res_ctx` stays `NULL`). RES, NR, comfort noise and
  synthesis are not inside it. An application with its own beamformer calls the two `mcab` functions directly;
- [`mcab_sample_post.h`](mcab_sample_post.h): header-only sample code (in no archive) between `mcab_fuse()` and the
  libraries behind it -- `mcab_sample_res_input()` fills the libaec `AecResInput` of a fused frame,
  `mcab_sample_anr_gain()` runs the denoiser on it (folding `r2` into the noise floor when asked),
  `mcab_sample_post_reset_due()` tells the synthesis to follow a lane's post-chain reset, and the
  constants `MCAB_SAMPLE_CNG_NR_GAIN_FLOOR`, `MCAB_SAMPLE_CNG_RNG_SEED` (= `AEC_CNG_SEED`, 42) and
  `MCAB_SAMPLE_PSD_SCALE_INV`; the estimators read `frame.e`, the synthesis renders `frame.b` (README:
  "Analysis and rendering spectra");
- [`mcab_sample_driver.h`](mcab_sample_driver.h) and [`mcab_sample_driver.c`](mcab_sample_driver.c): the CLI
  parsing and the lifecycle/hop-loop skeleton shared by the four 4-channel samples (sample code);
- [`mcab_external_static.c`](mcab_external_static.c): caller-owned-pool example that spells the application-side
  init out (`flow_plan()` / `flow_init()`), drives the flow with fixed equal weights and carves, behind it in one
  block, its own post-beam `AecRes` (unless `--no-res`), the NR, the synthesis and the gain buffers
  (`post_plan()` / `post_init()` / `post_hop()`), following the same query → allocate → `init` →
  process → destroy → release sequence as the mono `mono_aec_nr_res/main.c` built as `aec_nr_pipeline_static`;
- [`mcab_raw.c`](mcab_raw.c): raw-float host CLI for running the flow, the sample beam helper and the per-hop chain
  (`raw_chain()`, with its own `AecRes`) on recorded four-channel fixtures (the lanes, estimator, flow and helper
  share one `posix_memalign` block and the post stages a second -- a one-shot host tool, not a board-deployment demo);
- [`mcab_static.c`](mcab_static.c): caller-owned-pool example for the flow plus the sample beam helper (real SRP-PHAT
  + real BF, not an externally-supplied fixed-weight stand-in), its own `AecRes` and the chain
  (`static_chain()`), following the same query → allocate → `init_ex` → process → destroy → release sequence as
  `mcab_external_static.c` above;
- [`tests/test_mcab.c`](tests/test_mcab.c): 16/48 kHz grid, lifecycle, sequence rules, invalid-config,
  lane-contract (`mcab_lane_config()` against the documented rules, an application-composed lane), init validation
  of the injected objects, pool-boundary, heap/static byte-parity, weight, delay-machine and finite-output
  acceptance tests of the flow, and the descriptor gate (the superseded layouts 17-24 are refused);
  [`tests/conv_chain_mcab.h`](tests/conv_chain_mcab.h) writes the post-beam chain over it as the reference the
  application tests compare against, and [`tests/mcab_test_rig.h`](tests/mcab_test_rig.h) builds and tears down
  the lanes and the estimator for the tests;
- [`tests/test_mcab_sample_beam.c`](tests/test_mcab_sample_beam.c): sample beam helper lifecycle, topology, reset,
  split calls and their sequence guards, `enable_post = 0`, finite-output tests, `mcab_sample_beam_fuse()`
  leaving `g_res_ctx` NULL, the caller-owned RES (reset and level changes on the chain's `AecRes`), and the
  bit-exact equality of flow + helper + `mcab_sample_beam_fuse()` + the examples' chain with the reference chain;
- [`tests/test_mcab_sample_post.c`](tests/test_mcab_sample_post.c): the acceptance test of `mcab_sample_post.h`; it
  links no mcab object and is built by `pipelines/Makefile` (`make -C .. test_mcab_sample_post`);
- [`../../../audiolib/tests/test_spatial_third_party.c`](../../../audiolib/tests/test_spatial_third_party.c):
  scalar/SIMD PHAT, cached SRP, exported-BF-weight and VAD pool equivalence
  tests; `make test` here runs them first (`test-third-party` is `make -C ../../../audiolib test`).

For side-by-side reading, the mono example and its four-channel counterpart:

| Mono example | Four-channel counterpart |
|---|---|
| lib/aec `Aec` (`aec_get_mem_size`, `aec_init`) | `4ch_aec_bf_nr_res/mcab.h` (the flow over four `Aec` lanes and one `DelayAec`, all application-created) |
| `aec_get_res_context()` (`AecResContext`) | `mcab_fuse()` / `mcab_sample_beam_fuse()` (`McabFrame`) |
| `Aec` carries the RES gain (`ctx.res_gain`, strength `aec_set_level()`) | an `AecRes` the application creates (`aec_res_init()`, strength `aec_res_set_level()`), `aec_res_process()` over `mcab_sample_res_input()` after `mcab_fuse()`; `frame.g_res_ctx` stays `NULL` |
| `mono_nr_gain()` / `mono_form()` in `mono_aec_nr_res/main.c` | `mcab_sample_anr_gain()` and `post_hop()` in `mcab_external_static.c` |
| `mono_aec_nr_res/main.c` built as `aec_nr_pipeline_static` | `4ch_aec_bf_nr_res/mcab_external_static.c` |
| `mono_aec_nr_res/main.c` built as the host runner `aec_nr_pipeline` | `4ch_aec_bf_nr_res/mcab_raw.c` raw host runner |
| `mono_aec_nr_res/tests/test_mono_chain.c` | `4ch_aec_bf_nr_res/tests/test_mcab.c`, `test_mcab_sample_beam.c` |

| Mono | Four-channel flow |
|---|---|
| `derive_lane()` + `aec_get_mem_size()` (`mono_aec_nr_res/main.c`) | `mcab_default_config` / `mcab_lane_config` / `mcab_get_mem_requirements` (plus the lanes' and estimator's own `get_mem_size`) |
| `aec_init` | `aec_init` x4, `delay_aec_init` (MATCHED), then `mcab_init_ex` / `mcab_init` |
| (heap flavor) | `mcab_create` (mcab's own pool only; the lane and estimator memory stays the caller's) |
| `aec_process_context` + `aec_get_res_context` | `mcab_process_front` → external beamformer → `mcab_fuse` |
| `aec_reset` / `aec_destroy` | `mcab_reset` (also resets the lanes and the estimator) / `mcab_destroy` (touches no lane), then `aec_destroy` of each lane |

The sample beam helper adds the same lifecycle one level up (`mcab_sample_beam_get_mem_size` /
`_init` / `_reset` / `_destroy`) around the SRP-PHAT and BF; it borrows the `Mcab` handle.

The application's objects (the init order of `mcab.h`; `mcab_sample_flow_init()` does the same in one block):

```c
McabConfig cfg = mcab_default_config(16000);
AecConfig lane;
McabDelayEstimatorConfig dc;
McabMemReq req;
McabLibs libs;
DelayAec delay;                       /* the estimator object's own storage; outlives the flow */
int ch;

memset(&libs, 0, sizeof(libs));
mcab_lane_config(&cfg, &lane);        /* or aec_config_from_level(&lane, level, 16000), your own
                                       * filter_length etc., then mcab_apply_lane_contract(&cfg, &lane) */
mcab_delay_estimator_config(&cfg, &dc);
mcab_get_mem_requirements(&cfg, &req);
for (ch = 0; ch < MCAB_CHANNELS; ++ch) {          /* four distinct, freshly initialised lanes */
    size_t bytes = aec_get_mem_size(&lane);
    libs.lanes[ch] = aec_init(platform_alloc(bytes, 16), bytes, &lane);
}
if (dc.required) {                                /* MATCHED only; otherwise libs.delay stays NULL */
    size_t bytes = delay_aec_get_mem_size(dc.sample_rate, dc.hop_size, dc.num_filters);
    delay_aec_init(&delay, platform_alloc(bytes, 16), bytes, dc.sample_rate, dc.hop_size,
                   dc.num_filters);
    libs.delay = &delay;
}
```

Heap convenience (the flow with an external beamformer; mcab has no RES, so this
caller builds its own `AecRes` and writes the post-beam stages out):

```c
Mcab *pipeline = mcab_create(&cfg, &libs);        /* a heap block for mcab's own pool only */
McabFront front;
McabFrame f;
AecResInput rin;

/* microphones is interleaved [hop][4], reference is [hop]. */
mcab_process_front(pipeline, microphones, reference, &front);

/* External SRP-PHAT/BF consumes front.linear_spectra and produces
 * channel-major Complex[4][n_freqs] weights (and, optionally, the mono error).
 * Convention: out[k] = sum(weights[ch,k] * in[ch,k]); no conjugation. */
external_srp_bf_update(front.linear_spectra, weights);

mcab_fuse(pipeline, weights, NULL, &f);           /* NULL: rebuild E from the weights */

/* The application chains its own stages over f (mcab_external_static.c post_hop() is the
 * example, tests/conv_chain_mcab.h the independent reference): */
mcab_sample_res_input(&f, &rin);                   /* mcab_sample_post.h; estimated on f.e */
res = aec_res_process(my_res, &rin);               /* my_res: the caller's own AecRes */
nr  = mcab_sample_anr_gain(my_nr, &f, scratch, 1); /* scratch: f.n_freqs floats; on f.e */
sk_min_f32(total_gain, nr, res, f.n_freqs);
if (mcab_sample_post_reset_due(&handled, &f)) {    /* a lane had a post-chain reset */
    ac_istft_reset(my_istft);
    rng = MCAB_SAMPLE_CNG_RNG_SEED;                 /* AEC_CNG_SEED */
}
sk_capply_gain_f32(spec, f.b, total_gain, f.n_freqs);   /* rendered on the rendering base */
aec_cng_fill(&rng, spec, f.cn, res, nr, MCAB_SAMPLE_CNG_NR_GAIN_FLOOR, f.n_freqs);
ac_istft_push(my_istft, spec, mono_output);

mcab_destroy(pipeline);                            /* touches no lane or estimator */
for (ch = 0; ch < MCAB_CHANNELS; ++ch) aec_destroy(libs.lanes[ch]);
anr_destroy(my_nr);
```

`my_res` is built as `mcab_external_static.c` builds it: `aec_config_from_level(&res_cfg, level,
sample_rate)` with `res_cfg.fft_size` set to the flow's FFT, then `aec_res_get_mem_size()` and
`aec_res_init()` on a 16-byte-aligned block; `aec_res_set_level(my_res, level, ramp_ms)` changes
its strength while it runs. A product that keeps linear AEC, RES, CNG and synthesis but omits
MMSE-LSA simply never initialises the denoiser and passes the unity vector for `nr` (the example's
`--no-res` and the unity vector do the same for the RES). The module itself has no RES, NR or
comfort-noise setting. `enable_post` selects whether the lanes keep the
frequency-domain post context and mcab carves the fuse buffers
(`enable_post = 0` is the front-only seam the Align-ULCNet application uses).

The sample beam helper does not do the RES either: `mcab_sample_beam_fuse()` leaves
`frame->g_res_ctx` `NULL`, like `mcab_fuse()`, and neither pool holds an `AecRes`. An application
on the helper builds `my_res` the same way (`res_cfg.fft_size = mcab_fft_size(flow)`), as
`mcab_static.c` does in its post pool, and calls `aec_res_process()` over `mcab_sample_res_input(&frame, &in)` every
hop; a `NULL` result is a refused hop and the flow and every post stage are reset together
(`mcab_sample_beam_reset()` does not touch the `AecRes`: call `aec_res_reset()` yourself, which keeps
the level set with `aec_res_set_level()`). A refused `mcab_sample_beam_fuse()` resets the whole
helper (flow, SRP-PHAT, BF) and is returned, except for a sequence refusal and the helper's own argument refusal
(NULL arguments, `enable_post = 0`), which change nothing; the application resets its own post stages. Without `my_res` the RES gain is the unity vector and `aec_cng_fill()` is not called
(comfort noise fills only RES-cut bins). `4ch_alignulcnet` builds the flow with `cfg.enable_post = 0` and has no RES.

`total_gain` is `min(G_nr, G_res)` applied as computed. The post path has no
near-end floor or other level-dependent lift, so a loud background cannot cap
the denoiser or make the gain follow the far talker; which bins survive is
decided per bin by the denoiser's own speech-presence model and by `G_res`.

Caller-owned pool (static/board path):

```c
Mcab *pipeline;
void *pool;

/* the application's objects as above (cfg, req, libs) */
pool = platform_alloc(req.bytes, req.alignment);
pipeline = mcab_init_ex(pool, (size_t)req.bytes, &cfg, &libs, &req);

/* The per-hop mcab_process_front -> external beamformer -> mcab_fuse calls are identical. */

mcab_destroy(pipeline);                /* does not release caller-owned memory, touches no lane */
/* then aec_destroy() of each lane, and release the lane / estimator / pool memory */
platform_free(pool);
```

The sample beam helper is placed in a caller-owned block of its own: `mcab_sample_beam_get_mem_size()` composes the
SRP/BF sub-modules' `*_get_mem_size()` with the helper's own scratch (the beamformed spectrum and the weights),
and `mcab_sample_beam_init()` places all of it, zero-heap, in that block on top of the flow. The post-beam `AecRes`, the
denoiser, the synthesis and the gain buffers are a further block, each sized by its own query:

```c
McabSampleBeamParams bp = mcab_sample_beam_default_params();
size_t beam_bytes;
mcab_sample_beam_get_mem_size(&cfg, &bp, &beam_bytes);
void *beam_pool = platform_alloc(beam_bytes, 16);
McabSampleBeam *beam = mcab_sample_beam_init(beam_pool, beam_bytes, &cfg, &bp, pipeline);

AecConfig res_cfg;   aec_config_from_level(&res_cfg, AEC_LEVEL_BALANCED, 16000);
res_cfg.fft_size = mcab_fft_size(pipeline);                                          /* the flow's grid */
AnrConfig nr_cfg = anr_config_for_level_grid(16000, mcab_fft_size(pipeline), ANR_LEVEL_BALANCED);
AcStftConfig sy_cfg;  ac_stft_config_default(&sy_cfg, mcab_fft_size(pipeline));
size_t res_b = aec_res_get_mem_size(&res_cfg), nr_b = anr_get_mem_size(&nr_cfg), sy_b = ac_istft_get_mem_size(&sy_cfg);
void *post_pool = platform_alloc(ALIGN16(res_b) + ALIGN16(nr_b) + ALIGN16(sy_b) + gain_buffers, 16);   /* RES, NR, synthesis, buffers */
AecRes *res = aec_res_init(post_pool, res_b, &res_cfg);
AnrDenoiser *nr = anr_init((uint8_t *)post_pool + ALIGN16(res_b), nr_b, &nr_cfg);
AcIstft *istft = ac_istft_init((uint8_t *)post_pool + ALIGN16(res_b) + ALIGN16(nr_b), sy_b, &sy_cfg);

/* per hop: mcab_sample_beam_process_with_activity() (or mcab_sample_beam_process_front() + mcab_sample_beam_process_beam()),
 * mcab_sample_beam_fuse(beam, &frame)  (frame.g_res_ctx is NULL), then
 * mcab_sample_res_input() + aec_res_process(res, ...), mcab_sample_anr_gain(nr, &frame, ...) (on frame.e), sk_min_f32(),
 * mcab_sample_post_reset_due() (ac_istft_reset + reseed when it returns 1), sk_capply_gain_f32() over frame.b,
 * aec_cng_fill(), ac_istft_push() as raw_chain() / static_chain() write them. A refused call:
 * mcab_sample_beam_reset(beam), aec_res_reset(res), anr_reset(nr), ac_istft_reset(istft) and
 * rng = MCAB_SAMPLE_CNG_RNG_SEED together. Strength: aec_res_set_level(res, level, ramp_ms),
 * anr_set_level(nr, level). */

mcab_sample_beam_destroy(beam); /* first: it borrows the flow; does not release caller-owned memory */
mcab_destroy(pipeline);         /* then the flow, then the lanes (aec_destroy) */
anr_destroy(nr);                /* AecRes has no destroy */
platform_free(post_pool);
platform_free(beam_pool);
platform_free(pool);
```

`mcab_fuse()` requires a front frame processed since construction or the
last reset (`MCAB_SEQUENCE_ERROR` otherwise); a front frame stays
valid until the next `mcab_process_front()` or `mcab_reset()`,
and there is no token to return. Invalid weights leave the front frame usable,
so the caller may correct and retry. `mcab_create()` allocates one block for mcab's own pool;
`mcab_init_ex()` uses only the caller's aligned pool. Neither per-hop call performs
any allocation.

Build the applications or run the complete C acceptance gate from
`Audio_ALG/pipelines/4ch_aec_bf_nr_res`:

```bash
make mcab_external_static  # build caller-pool reference executable (flow + chain)
make mcab_static           # build caller-pool reference executable (flow + sample beam helper + chain)
make mcab_raw              # build the recording-validation runner (flow + sample beam helper + chain)
make libaudio_pipeline_mcab.a  # the flow archive alone (needs only audiolib common and lib/aec)
make test_mcab             # build only the flow test binary (needs lib/aec and lib/nr, not the spatial kernels)
make test_mcab_sample_beam # build only the sample beam helper test
make test_4ch_dfn test_4ch_ulcnet   # the DFN2 / Align-ULCNet compositions
make 4ch_alignulcnet       # the Align-ULCNet board skeleton
make 4ch_aec_bf_dfn_res    # the DFN2 composition example
make 4ch_aec_bf_dfn_res DFN2_MASK_LOOKAHEAD=2 DFN2_DF_LOOKAHEAD=4   # another DFN2 lookahead pair
make test                  # audiolib's tests, then the 4ch tests and example smoke runs (see the Makefile)
make NO_STDIO=1 audit-no-stdio

# Query the exact pool budget for the supported grids.
"$(make -s print-bin-dir)/mcab_external_static" --print-mem-size --sample-rate 16000 --fft-size 256
"$(make -s print-bin-dir)/mcab_external_static" --print-mem-size --sample-rate 16000
"$(make -s print-bin-dir)/mcab_external_static" --print-mem-size --sample-rate 48000
"$(make -s print-bin-dir)/mcab_static" --print-mem-size --sample-rate 16000 --fft-size 256
"$(make -s print-bin-dir)/mcab_static" --print-mem-size --sample-rate 16000
"$(make -s print-bin-dir)/mcab_static" --print-mem-size --sample-rate 48000
```

`DFN2_MASK_LOOKAHEAD` (0..2) and `DFN2_DF_LOOKAHEAD` (0..4) default to the shipped pair 1/1; a different pair is passed
as `-D` flags to every unit this Makefile compiles and is part of the build signature, so pairs never share an object
directory. Only the 1/1 pair has a trained checkpoint, and the audio quality of the other pairs is not measured.

`libaudio_pipeline_mcab.a` contains `mcab.o` (the flow) only. The sample objects (`mcab_sample_flow.o`,
`mcab_sample_beam.o`, `mcab_sample_driver.o`), the example chain headers and the ULCNet stage object
(`ulcnet_nn_stage.o`) are compiled into the application binaries that use them; no
per-application archive is produced. A consumer of the flow alone links `libaudio_pipeline_mcab.a`, `libaec.a`,
`libaudio_common.a` and `-lm` (`mcab.o` references lib/aec and audiolib's common utilities only; `libanr.a` is for
the programs that chain NR behind the flow). A consumer of the sample beam helper links `mcab_sample_beam.o` and
`mcab_sample_flow.o`, then `libaudio_pipeline_mcab.a`, `libdoa.a`, `libbf.a`, `libspatial_common.a`, `libaec.a`,
`libanr.a`, `libaudio_common.a`, and `-lm`, in the order demonstrated by this directory's Makefile.

Each goal resolves, and keys its `bin/ne10-<sig>/` and `obj/ne10-<sig>/` directories on, only the producers it
needs: `libaudio_pipeline_mcab.a` lib/aec and audiolib common (lib/nr and audiolib's spatial kernels may be absent),
`test_mcab` adds lib/nr, every other goal all of them; a goal missing a producer it needs stops and names it. The
same goal built in a narrower scope lands in its own directory. `make -s print-lib-path` prints the flow archive of
its own scope, `make -s print-bin-dir` / `print-obj-dir` the directory of `make all`, and `PRINT_FOR=<goal>` the one
of that goal (`PRINT_FOR=all` locates the copy of the archive that `make all` builds):

```bash
make test_mcab && "$(make -s print-bin-dir PRINT_FOR=test_mcab)/test_mcab"
```

[Index](README.md) | previous: [Call sequence and parameter surface](README.md#call-sequence-and-parameter-surface) | next: [Python reference seam](README.md#python-reference-seam) <!-- split-nav -->
