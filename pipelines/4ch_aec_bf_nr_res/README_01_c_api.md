[Index](README.md) | previous: [Call sequence and parameter surface](README.md#call-sequence-and-parameter-surface) | next: [Python reference seam](README.md#python-reference-seam) <!-- split-nav -->
## C API

The deployable C seam is:

- [`4aec_nr_res.h`](4aec_nr_res.h): public config, lifecycle, the two per-hop
  calls (`four_aec_front_process()`, `four_aec_fuse()`) and structural audit
  accessors. Its config carries only the front end's fields (grid, delay policy,
  `enable_post`); the core has no RES. The unit's config adds only geometry, DOA, BF and fallback-VAD
  fields; the RES, NR and comfort-noise settings belong to the application's own chain;
- [`4aec_nr_res.c`](4aec_nr_res.c): one shared `DelayAec`, four linear `Aec`
  instances and the coherent context projection. The residual suppressor (`AecRes`,
  `aec_res.h`, libaec) is carved and chained by the applications below, behind the core or the unit; so are the
  MMSE-LSA instance and the inverse transform;
- [`4aec_projection_kernels.h`](4aec_projection_kernels.h): byte-equivalent
  scalar/NEON complex projection, residual-vector, comfort-noise and
  weight-matrix gate kernels;
- [`4aec_nr_res_internal.h`](4aec_nr_res_internal.h): private delay-admission
  helpers;
- [`4aec_nr_res_static.c`](4aec_nr_res_static.c): caller-owned-pool example
  that drives the core with fixed equal weights and carves, behind the front end in one block, its
  own post-beam `AecRes` (unless `--no-res`), the NR, the synthesis and the gain buffers
  (`post_plan()` / `post_init()` / `post_hop()`), following the same query → allocate → `init` →
  process → destroy → release sequence as the mono `mono_aec_nr_res/static_main.c`;
- [`audio_pipeline_4ch.h`](audio_pipeline_4ch.h) and
  [`audio_pipeline_4ch.c`](audio_pipeline_4ch.c): the one unit (pool-first
  `get_mem_requirements`/`init_ex`/`init`, plus a `create` heap convenience
  wrapper — same two-tier convention as `4aec_nr_res.h` above) that inserts
  the reusable SRP-PHAT/DOA smoothing and BF of `audiolib/{doa,bf}` between `four_aec_front_process()`
  and `four_aec_fuse()`. Per hop `process_front()` / `process_beam()` (or
  `process_with_activity()` / `process()`) give the beamformed spectrum and weights;
  `audio_pipeline_4ch_fuse()` fills the `SeSpecFrame` for the frequency-domain stages
  (`frame.g_res_ctx` stays `NULL`). RES, NR, comfort noise and synthesis are not inside it;
- [`audio_pipeline_4ch_raw.c`](audio_pipeline_4ch_raw.c): raw-float host CLI for running
  the unit plus the per-hop chain (`raw_chain()`, with its own `AecRes`) on recorded four-channel fixtures (heap
  `create()` for the unit — a one-shot host tool, not a board-deployment demo);
- [`audio_pipeline_4ch_static.c`](audio_pipeline_4ch_static.c): caller-owned-pool
  example for the unit (front end + real SRP-PHAT + real BF, not an
  externally-supplied fixed-weight stand-in), its own `AecRes` and the chain (`static_chain()`), following the same query →
  allocate → `init_ex` → process → destroy → release sequence as
  `4aec_nr_res_static.c` above;
- [`tests/test_4aec_nr_res.c`](tests/test_4aec_nr_res.c): 16/48 kHz grid, lifecycle,
  sequence rules, invalid-config, pool-boundary, heap/static byte-parity,
  weight, delay-machine and finite-output acceptance tests of the front end;
  [`tests/conv_chain_4ch.h`](tests/conv_chain_4ch.h) writes the post-beam chain over
  it as the reference the application tests compare against;
- [`tests/test_audio_pipeline_4ch.c`](tests/test_audio_pipeline_4ch.c): unit lifecycle,
  topology, reset, split calls and their sequence guards, `enable_post = 0`, the descriptor
  gate, finite-output tests, `fuse()` leaving `g_res_ctx` NULL, the caller-owned RES (reset and
  preset changes on the chain's `AecRes`), and the bit-exact equality of unit +
  `audio_pipeline_4ch_fuse()` + the examples' chain with the reference chain;
- [`../../../audiolib/tests/test_spatial_third_party.c`](../../../audiolib/tests/test_spatial_third_party.c):
  scalar/SIMD PHAT, cached SRP, exported-BF-weight and VAD pool equivalence
  tests; `make test` here delegates to `make -C ../../../audiolib test`.

For side-by-side reading, the mono example and its four-channel counterpart:

| Mono example | Four-channel counterpart |
|---|---|
| lib/aec `Aec` (`aec_get_mem_size`, `aec_init`) | `4ch_aec_bf_nr_res/4aec_nr_res.h` (front end) + `audio_pipeline_4ch.h` (unit) |
| `common/se_stage.h` `se_frame_from_aec()` | `audio_pipeline_4ch_fuse()` / `four_aec_fuse()` |
| `Aec` carries the RES gain (`ctx.res_gain`, strength `aec_set_preset()`) | an `AecRes` the application creates (`aec_res_init()`, strength `aec_res_set_preset()`), `se_res_stage_gain()` after `audio_pipeline_4ch_fuse()`; `frame.g_res_ctx` stays `NULL` |
| `mono_aec_nr_res/static_main.c` | `4ch_aec_bf_nr_res/4aec_nr_res_static.c` |
| `mono_aec_nr_res/main.c` host runner | `4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c` raw host runner |
| `mono_aec_nr_res/tests/test_mono_chain.c` | `4ch_aec_bf_nr_res/tests/test_4aec_nr_res.c`, `test_audio_pipeline_4ch.c` |

| Mono | Four-channel front end |
|---|---|
| `se_derive()` + `aec_get_mem_size()` | `four_aec_nr_res_default_config` / `four_aec_nr_res_get_mem_requirements` |
| `aec_init` | `four_aec_nr_res_init_ex` / `_init` |
| (heap flavor) | `four_aec_nr_res_create` |
| `aec_process_context` + `se_frame_from_aec` | `four_aec_front_process` → external beamformer → `four_aec_fuse` |
| `aec_reset` / `aec_destroy` | `four_aec_nr_res_reset` / `_destroy` |

The unit adds the same lifecycle one level up (`audio_pipeline_4ch_get_mem_requirements` /
`_init_ex` / `_init` / `_create` / `_reset` / `_destroy`) around the SRP-PHAT and BF.

Heap convenience (the core with an external beamformer; neither the core nor the unit has a RES, so this
caller builds its own `AecRes` and writes the post-beam stages out):

```c
FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);
FourAecNrRes *pipeline = four_aec_nr_res_create(&cfg);
FourAecFront front;
SeSpecFrame f;

/* microphones is interleaved [hop][4], reference is [hop]. */
four_aec_front_process(pipeline, microphones, reference, &front);

/* External SRP-PHAT/BF consumes front.linear_spectra and produces
 * channel-major Complex[4][n_freqs] weights (and, optionally, the mono error).
 * Convention: out[k] = sum(weights[ch,k] * in[ch,k]); no conjugation. */
external_srp_bf_update(front.linear_spectra, weights);

four_aec_fuse(pipeline, weights, NULL, &f);       /* NULL: rebuild E from the weights */

/* The application chains its own stages over f (4aec_nr_res_static.c post_hop() is the
 * example, tests/conv_chain_4ch.h the independent reference): */
res = se_res_stage_gain(my_res, &f);               /* my_res: the caller's own AecRes */
nr  = se_anr_stage_gain(my_nr, &f, scratch, 1);
sk_min_f32(total_gain, nr, res, f.n_freqs);
sk_capply_gain_f32(spec, f.e, total_gain, f.n_freqs);
aec_cng_fill(&rng, spec, f.cn, res, nr, SE_CNG_NR_GAIN_FLOOR, f.n_freqs);
ac_istft_push(my_istft, spec, mono_output);

four_aec_nr_res_destroy(pipeline);
anr_destroy(my_nr);
```

`my_res` is built as `4aec_nr_res_static.c` builds it: `aec_config_from_preset(&res_cfg, preset,
sample_rate)` with `res_cfg.fft_size` set to the core's FFT, then `aec_res_get_mem_size()` and
`aec_res_init()` on a 16-byte-aligned block; `aec_res_set_preset(my_res, preset, ramp_ms)` changes
its strength while it runs. A product that keeps linear AEC, RES, CNG and synthesis but omits
MMSE-LSA simply never initialises the denoiser and passes the unity vector for `nr` (the example's
`--no-res` and the unity vector do the same for the RES). The front end itself has no RES, NR or
comfort-noise setting. `enable_post` selects whether the lanes keep the
frequency-domain post context and the front end carves the fuse buffers
(`enable_post = 0` is the front-only seam the Align-ULCNet application uses).

The unit (`AudioPipeline4Ch`) does not do the RES either: `audio_pipeline_4ch_fuse()` leaves
`frame->g_res_ctx` `NULL`, like `four_aec_fuse()`, and the unit pool holds no `AecRes`. An application
on the unit builds `my_res` the same way (`res_cfg.fft_size = audio_pipeline_4ch_fft_size(p)`), as
`audio_pipeline_4ch_static.c` does in its post pool, and calls `se_res_stage_gain(my_res, &frame)` every
hop; a `NULL` result is a refused hop and the unit and every post stage are reset together
(`audio_pipeline_4ch_reset()` does not touch the `AecRes`: call `aec_res_reset()` yourself, which keeps
the preset set with `aec_res_set_preset()`). A refused `audio_pipeline_4ch_fuse()` resets the whole unit
(core, SRP-PHAT, BF) and is returned; the application resets its own post stages. Without
`my_res` the RES gain is the unity vector and `aec_cng_fill()` is not called (comfort noise fills only
RES-cut bins). `4ch_alignulcnet` runs the unit with `core.enable_post = 0` and has no RES.

`total_gain` is `min(G_nr, G_res)` applied as computed. The post path has no
near-end floor or other level-dependent lift, so a loud background cannot cap
the denoiser or make the gain follow the far talker; which bins survive is
decided per bin by the denoiser's own speech-presence model and by `G_res`.

Caller-owned pool (static/board path):

```c
FourAecNrResConfig cfg;
FourAecNrResMemReq req;
FourAecNrRes *pipeline;
void *pool;

cfg = four_aec_nr_res_default_config(16000);
four_aec_nr_res_get_mem_requirements(&cfg, &req);
pool = platform_alloc(req.bytes, req.alignment);
pipeline = four_aec_nr_res_init_ex(
    pool, (size_t)req.bytes, &cfg, &req);

/* The per-hop front_process -> external beamformer -> fuse calls are identical. */

four_aec_nr_res_destroy(pipeline); /* does not release caller-owned memory */
platform_free(pool);
```

The unit (front end + SRP-PHAT + BF) uses the same pattern one level up --
`audio_pipeline_4ch_get_mem_requirements()` composes the front end's own
requirement with the SRP/BF sub-modules' `*_get_mem_size()` and the unit's own
scratch (the beamformed spectrum and the weights),
and `audio_pipeline_4ch_init_ex()` places all of it, zero-heap, in one
caller-owned pool. The post-beam `AecRes`, the denoiser, the synthesis and the gain buffers are a second
block, each sized by its own query:

```c
AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(16000);
AudioPipeline4ChMemReq req;
audio_pipeline_4ch_get_mem_requirements(&cfg, &req);
void *pool = platform_alloc(req.bytes, req.alignment);
AudioPipeline4Ch *p =
    audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &req);

AecConfig res_cfg;   aec_config_from_preset(&res_cfg, AEC_PRESET_BALANCED, 16000);
res_cfg.fft_size = audio_pipeline_4ch_fft_size(p);                                    /* the unit's grid */
AnrConfig nr_cfg = anr_config_for_mode_grid(16000, audio_pipeline_4ch_fft_size(p), ANR_MODE_BALANCED);
AcStftConfig sy_cfg;  ac_stft_config_default(&sy_cfg, audio_pipeline_4ch_fft_size(p));
size_t res_b = aec_res_get_mem_size(&res_cfg), nr_b = anr_get_mem_size(&nr_cfg), sy_b = ac_istft_get_mem_size(&sy_cfg);
void *post_pool = platform_alloc(ALIGN16(res_b) + ALIGN16(nr_b) + ALIGN16(sy_b) + gain_buffers, 16);   /* RES, NR, synthesis, buffers */
AecRes *res = aec_res_init(post_pool, res_b, &res_cfg);
AnrDenoiser *nr = anr_init((uint8_t *)post_pool + ALIGN16(res_b), nr_b, &nr_cfg);
AcIstft *istft = ac_istft_init((uint8_t *)post_pool + ALIGN16(res_b) + ALIGN16(nr_b), sy_b, &sy_cfg);

/* per hop: audio_pipeline_4ch_process_with_activity() (or process_front() + process_beam()),
 * audio_pipeline_4ch_fuse(p, &frame)  (frame.g_res_ctx is NULL), then
 * se_res_stage_gain(res, &frame), se_anr_stage_gain(nr, &frame, ...), sk_min_f32(), sk_capply_gain_f32(),
 * aec_cng_fill(), ac_istft_push() as raw_chain() / static_chain() write them. A refused call:
 * audio_pipeline_4ch_reset(p), aec_res_reset(res), anr_reset(nr), ac_istft_reset(istft) and
 * rng = SE_CNG_RNG_SEED together. Strength: aec_res_set_preset(res, preset, ramp_ms),
 * anr_set_mode(nr, mode). */

audio_pipeline_4ch_destroy(p); /* does not release caller-owned memory */
anr_destroy(nr);               /* AecRes has no destroy */
platform_free(post_pool);
platform_free(pool);
```

`four_aec_fuse()` requires a front frame processed since construction or the
last reset (`FOUR_AEC_NR_RES_SEQUENCE_ERROR` otherwise); a front frame stays
valid until the next `four_aec_front_process()` or `four_aec_nr_res_reset()`,
and there is no token to return. Invalid weights leave the front frame usable,
so the caller may correct and retry. `create()` allocates one complete pool;
`init_ex()` uses only the caller's aligned pool. Neither per-hop call performs
any allocation.

Build the applications or run the complete C acceptance gate from
`Audio_ALG/pipelines/4ch_aec_bf_nr_res`:

```bash
make 4aec_nr_res_static     # build caller-pool reference executable (front end + chain)
make audio_pipeline_4ch_static    # build caller-pool reference executable (unit + chain)
make audio_pipeline_4ch_raw      # build the recording-validation runner (unit + chain)
make test_4aec_nr_res      # build only the front-end 4-channel test binary
make test_audio_pipeline_4ch     # build only the unit test
make test_4ch_dfn test_4ch_ulcnet   # the DFN2 / Align-ULCNet compositions
make 4ch_alignulcnet       # model callback remains a board TODO; fail-open smoke
make test                  # run only the isolated 4ch acceptance gate
make NO_STDIO=1 audit-no-stdio

# Query the exact pool budget for the supported grids.
"$(make -s print-bin-dir)/4aec_nr_res_static" --print-mem-size --sample-rate 16000 --fft-size 256
"$(make -s print-bin-dir)/4aec_nr_res_static" --print-mem-size --sample-rate 16000
"$(make -s print-bin-dir)/4aec_nr_res_static" --print-mem-size --sample-rate 48000
"$(make -s print-bin-dir)/audio_pipeline_4ch_static" --print-mem-size --sample-rate 16000 --fft-size 256
"$(make -s print-bin-dir)/audio_pipeline_4ch_static" --print-mem-size --sample-rate 16000
"$(make -s print-bin-dir)/audio_pipeline_4ch_static" --print-mem-size --sample-rate 48000
```

`libaudio_pipeline_4ch.a` contains `4aec_nr_res.o`, `audio_pipeline_4ch.o` (the unit) and the
shared `se_config.o` of `pipelines/common`. The example chain headers and the ULCNet stage object
(`ulcnet_nn_stage.o`) are compiled into the application binaries that use them; no
per-application archive is produced. A consumer of the unit links `libaudio_pipeline_4ch.a`,
`libdoa.a`, `libbf.a`, `libspatial_common.a`, `libaec.a`, `libanr.a`, `libaudio_common.a`, and
`-lm`, in the order demonstrated by this directory's Makefile.

[Index](README.md) | previous: [Call sequence and parameter surface](README.md#call-sequence-and-parameter-surface) | next: [Python reference seam](README.md#python-reference-seam) <!-- split-nav -->
