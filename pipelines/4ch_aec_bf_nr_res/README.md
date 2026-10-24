# 4-channel PBFDKF + beamforming + NR/RES application

This is the conventional four-channel product application directory. It owns
the reusable four-lane linear-AEC front end (`FourAecNrRes`), the one wrapped unit
`AudioPipeline4Ch` (front end + SRP-PHAT + BF + the post-beam RES), the example programs that chain
the remaining post-beam stages over them, host evaluation tools and their tests. The unit wraps the
linear AEC, the BF and one post-beam residual suppressor (`AecRes`: `enable_res`, `res_preset`);
NR, comfort noise, synthesis and neural post-filters are separate libraries that the integrator
initialises (each with its own `mem_size` and `init`) and chains, as each example `main` does.
The sibling `../4ch_alignulcnet/` and
`../4ch_aec_bf_dfn_res/` applications reuse the unit and the core; they do not duplicate AEC,
DOA, BF or NR implementations.

This shares the mono pipeline's zero-padding-free grid convention (frame ==
FFT, hop == frame/2 -- see the supported-grids table below), not an AIAEC
neural-model deployment path. The same boundary is available as a Python
reference and as a synchronous C library.

The core's boundary is intentionally narrow: the post-beam stages are not part of it, and the
unit adds only the RES to the linear path:

```text
raw/aligned far ──> one shared aligner (MATCHED, FIXED or EXTERNAL)
                              |
                              v (one aligned reference, identical to every lane)
mic 0 ──> linear AEC 0 ─┐ (computes the far-end FFT: aec_process_context())
mic 1 ──> linear AEC 1 ─┤ (normally borrows lane 0's far_spec: aec_process_context_shared_far_from();
mic 2 ──> linear AEC 2 ─┤  own FFT instead on a mixed warm/soft-realign hop -- see "Far-end FFT
mic 3 ──> linear AEC 3 ─┘  sharing" below)
                        │    four_aec_front_process()
                        └─>[external SRP-PHAT/BF: weights + beamformed error]
                              │    four_aec_fuse()   → SeSpecFrame
                              v
        unit: audio_pipeline_4ch_fuse() also runs its post-beam AecRes → frame.g_res_ctx
        core: the caller's own AecRes → se_res_stage_gain()
        G_res ─ se_anr_stage_gain ─ sk_min_f32 ─ sk_capply_gain_f32
              ─ aec_cng_fill ─ ac_istft_push → mono hop          (the application's own code)
```

Each `linear AEC` box still runs its own PBFDKF/shadow filter, residual-echo
(R2) estimate, and `DominantNearend` hold-state update every hop -- the echo
path genuinely differs per microphone, and the next hop's onset-gated R2
depends on that lane's own hold-state. What it does not do is compute its
own mono suppression gain: `AecConfig.spatial_linear_context=1` (set
internally by this pipeline, not caller-configurable) skips straight to the
`DominantNearend` update and leaves `res_gain` `NULL`, so the post-beam RES
gain (`se_res_stage_gain()` over the fused frame, called by the unit's
`audio_pipeline_4ch_fuse()`) is the only place a suppression gain is calculated.

**Far-end FFT sharing** (one far-end rfft per hop instead of four): all four
lanes see the byte-identical `p->aligned_ref` and realign on the same hop by the
same delta (see the hard invariants below), but
`aec_apply_external_realign()`'s warm-vs-soft outcome is per lane, and a soft
outcome zeros that lane's far_buffer OLA history. When every lane agrees (all
warm, all soft, or no realign), lane 0's far-end FFT is what any lane would
compute: lane 0 runs `aec_process_context()` and lanes 1-3 borrow its `far_spec`
through `aec_process_context_shared_far_from()`. On a mixed warm/soft hop
(`FourAecNrRes.far_spec_mixed_hop`) lanes 1-3 run their own
`aec_process_context()`, so each far_spec reflects the lane's own outcome.
`four_aec_nr_res_far_fft_real_compute_count()` exposes the per-hop FFT count
(1, or 4 on a mixed hop). `ref` is passed to every lane either way: the OLA
far-buffer history and the non-FFT uses of the raw far signal (saturation
detection, delay estimation, mu_scale) are per-lane state.

`fuse_contexts()`'s cross-lane far-spec agreement gate
(`four_aec_nr_res_far_spec_agrees()`, in `4aec_nr_res_internal.h`) is bypassed on
a mixed hop; the fused `X` takes lane 0's far_spec either way
(`frame->x = p->snapshots[0].far_spec` in `four_aec_fuse()`), as in `pipeline.py`'s
`_fuse_contexts()`.

Hard invariants:

- There is at most **one** shared matched-filter bank. MATCHED mode uses
  1..5 filters and a configurable capture proxy; FIXED uses only the exact
  configured-delay ring; EXTERNAL consumes caller-aligned far and allocates
  neither a matcher nor an alignment ring.
- Only the adaptive linear filter is replicated, exactly four times.
- All four filters consume the same delay-aligned reference and realign together
  if that shared alignment changes -- this is also the precondition the
  far-end-FFT-sharing scheme above depends on, not a separate guarantee.
- No NR, RES, neural model, or delay estimator is replicated per microphone --
  enforced structurally, not just left unused: each lane keeps only the
  R2/`DominantNearend` state its own echo path needs and never calculates its
  own suppression gain (`spatial_linear_context=1`; see above).
- `delay_num_filters`, each PBFDKF lane's `filter_length`, and an
  Align-ULCNet checkpoint's TA depth `D` are independent budgets.
- The Python reference and the C front end leave SRP-PHAT/BF at an explicit
  ownership seam. The C unit in `audio_pipeline_4ch.*` connects
  that seam to the reusable implementations under `../../../audiolib/{doa,bf}`
  (a sibling checkout of this repo, shared by every pipeline).
- `EqualWeightBeamformer` exists only as a deterministic offline/test adapter;
  it is never selected by default.

The external beamformer must expose its effective complex weights with shape
`[4, n_freqs]`. `FourAecFront.linear_spectra` are reconstructing 50%-overlap
sqrt-Hann STFT frames and can be consumed directly without four duplicate FFTs.
`FourAecFront.formed_hop[ch]` are the exact selected/crossfaded time-domain hops
underlying those spectra (borrowed from each lane, not copied) for integrations
whose beamformer owns its four analysis FFTs. The per-lane candidate set is
refined, coarse, or -- in this seam's context-only mode -- the lane's own mic
capture, on a hop the filtering-quality analyzer has not cleared; a beamformer
computing its own residual-echo estimate from these hops inherits that
substitution. The Python reference also accepts the external mono hop.

`four_aec_fuse()` takes the weights and, optionally, the mono spectrum the same
beamformer call produced. With `beamformed_error == NULL` it rebuilds the error
from the weights, so independently supplied output samples and context cannot
come from different beamformer states. A beamformer that atomically returns both
its mono spectrum and effective weights passes the spectrum and skips that
duplicate weighted sum; both arrays are validated for finiteness, while their
coherence remains an explicit caller contract. The unit's
`audio_pipeline_4ch_fuse()` uses this form with `bf_process_with_weights()`. The far-end spectrum remains the one
shared digital render reference -- it is verified equal across lanes and is not
spatially weighted. The result passes through one mono NR+RES path without a
fifth AEC or replicated post-filters. When `front.delay.changed` is non-zero, an
external beamformer that owns STFT/OLA state must clear that overlap state
before consuming the frame; the application's own synthesis is not restarted
(the lanes realign their filters in place).

## C API

The deployable C seam is:

- [`4aec_nr_res.h`](4aec_nr_res.h): public config, lifecycle, the two per-hop
  calls (`four_aec_front_process()`, `four_aec_fuse()`) and structural audit
  accessors. Its config carries only the front end's fields (grid, delay policy,
  `enable_post`); the core has no RES. The unit's config adds `enable_res` / `res_preset`; the NR and
  comfort-noise settings belong to the application's own chain;
- [`4aec_nr_res.c`](4aec_nr_res.c): one shared `DelayAec`, four linear `Aec`
  instances and the coherent context projection. The residual suppressor (`AecRes`,
  `aec_res.h`) is carved by the unit (below) or by a caller that drives the core directly; the
  MMSE-LSA instance and the inverse transform are carved and chained by the applications below;
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
  third-party SRP-PHAT/DOA smoothing and BF between `four_aec_front_process()`
  and `four_aec_fuse()`. Per hop `process_front()` / `process_beam()` (or
  `process_with_activity()` / `process()`) give the beamformed spectrum and weights;
  `audio_pipeline_4ch_fuse()` fills the `SeSpecFrame` for the frequency-domain stages and, with
  `enable_res`, runs the post-beam RES over it (`frame.g_res_ctx`). NR, comfort noise and
  synthesis are not inside it;
- [`audio_pipeline_4ch_raw.c`](audio_pipeline_4ch_raw.c): raw-float host CLI for running
  the unit plus the per-hop chain (`raw_chain()`) on recorded four-channel fixtures (heap
  `create()` — a one-shot host tool, not a board-deployment demo);
- [`audio_pipeline_4ch_static.c`](audio_pipeline_4ch_static.c): caller-owned-pool
  example for the unit (front end + real SRP-PHAT + real BF + RES, not an
  externally-supplied fixed-weight stand-in) and the chain (`static_chain()`), following the same query →
  allocate → `init_ex` → process → destroy → release sequence as
  `4aec_nr_res_static.c` above;
- [`tests/test_4aec_nr_res.c`](tests/test_4aec_nr_res.c): 16/48 kHz grid, lifecycle,
  sequence rules, invalid-config, pool-boundary, heap/static byte-parity,
  weight, delay-machine and finite-output acceptance tests of the front end;
  [`tests/conv_chain_4ch.h`](tests/conv_chain_4ch.h) writes the post-beam chain over
  it as the reference the application tests compare against;
- [`tests/test_audio_pipeline_4ch.c`](tests/test_audio_pipeline_4ch.c): unit lifecycle,
  topology, reset, split calls and their sequence guards, `enable_post = 0`, the descriptor
  gate, finite-output tests, the unit's RES (against a separate `AecRes` fed the same fused
  frame, reset, the accessor and its strength setter), and the bit-exact equality of unit +
  `audio_pipeline_4ch_fuse()` + the examples' chain with the reference chain;
- [`../../../audiolib/tests/test_spatial_third_party.c`](../../../audiolib/tests/test_spatial_third_party.c):
  scalar/SIMD PHAT, cached SRP, exported-BF-weight and VAD pool equivalence
  tests; `make test` here delegates to `make -C ../../../audiolib test`.

For side-by-side reading, the mono example and its four-channel counterpart:

| Mono example | Four-channel counterpart |
|---|---|
| lib/aec `Aec` (`aec_get_mem_size`, `aec_init`) | `4ch_aec_bf_nr_res/4aec_nr_res.h` (front end) + `audio_pipeline_4ch.h` (unit) |
| `common/se_stage.h` `se_frame_from_aec()` | `audio_pipeline_4ch_fuse()` / `four_aec_fuse()` |
| `Aec` carries the RES gain (`ctx.res_gain`, strength `aec_set_preset()`) | the unit's post-beam `AecRes` (`enable_res`, `audio_pipeline_4ch_res()`), `frame.g_res_ctx` after `audio_pipeline_4ch_fuse()` |
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

Heap convenience (the core with an external beamformer; the core has no RES, so this caller
builds its own `AecRes` and writes the post-beam stages out):

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

The unit (`AudioPipeline4Ch`) does the RES for you: with `enable_res = 1` (the default) and
`core.enable_post = 1` it carves one post-beam `AecRes` after `bf_weights`, built from
`aec_config_from_preset(res_preset, sample_rate)` on the unit's FFT. `audio_pipeline_4ch_fuse()`
runs it over the fused frame and sets `frame->g_res_ctx`; `audio_pipeline_4ch_res(p)` returns it for
`aec_res_set_preset(audio_pipeline_4ch_res(p), preset, ramp_ms)`. With `enable_res = 0` (or
`enable_post = 0`) no suppressor is built (its storage leaves the pool), `g_res_ctx` stays `NULL`,
the gain is the denoiser's alone, comfort noise, which fills only RES-cut bins, is skipped, and the
accessor returns `NULL`. `audio_pipeline_4ch_reset()` resets the suppressor and keeps its preset; a
RES refusal inside `fuse()` is `FOUR_AEC_NR_RES_DSP_ERROR` and resets the whole unit.

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
scratch (the beamformed spectrum and the weights) and, with `enable_res`, the post-beam `AecRes`,
and `audio_pipeline_4ch_init_ex()` places all of it, zero-heap, in one
caller-owned pool. The denoiser, the synthesis and the gain buffers are a second block, each sized
by its own query:

```c
AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(16000);
AudioPipeline4ChMemReq req;
audio_pipeline_4ch_get_mem_requirements(&cfg, &req);
void *pool = platform_alloc(req.bytes, req.alignment);
AudioPipeline4Ch *p =
    audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &req);

AnrConfig nr_cfg = anr_config_for_mode_grid(16000, audio_pipeline_4ch_fft_size(p), ANR_MODE_BALANCED);
AcStftConfig sy_cfg;  ac_stft_config_default(&sy_cfg, audio_pipeline_4ch_fft_size(p));
size_t nr_b = anr_get_mem_size(&nr_cfg), sy_b = ac_istft_get_mem_size(&sy_cfg);
void *post_pool = platform_alloc(ALIGN16(nr_b) + ALIGN16(sy_b) + gain_buffers, 16);   /* NR, synthesis, buffers */
AnrDenoiser *nr = anr_init(post_pool, nr_b, &nr_cfg);
AcIstft *istft = ac_istft_init((uint8_t *)post_pool + ALIGN16(nr_b), sy_b, &sy_cfg);

/* per hop: audio_pipeline_4ch_process_with_activity() (or process_front() + process_beam()),
 * audio_pipeline_4ch_fuse(p, &frame)  (frame.g_res_ctx is the unit's RES gain), then
 * se_anr_stage_gain(nr, &frame, ...), sk_min_f32(), sk_capply_gain_f32(), aec_cng_fill(),
 * ac_istft_push() as raw_chain() / static_chain() write them. A refused call:
 * audio_pipeline_4ch_reset(p), anr_reset(nr), ac_istft_reset(istft) and rng = SE_CNG_RNG_SEED
 * together. Strength: aec_res_set_preset(audio_pipeline_4ch_res(p), preset, ramp_ms),
 * anr_set_mode(nr, mode). */

audio_pipeline_4ch_destroy(p); /* does not release caller-owned memory */
anr_destroy(nr);
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

## Python reference seam

```python
from importlib import import_module

aec4 = import_module("pipelines.4ch_aec_bf_nr_res")
BeamformerFrame = aec4.BeamformerFrame
FourChannelAecPipeline = aec4.FourChannelAecPipeline

pipeline = FourChannelAecPipeline()      # no beamformer is created; the post-beam RES is
                                         # owned (FourChannelAecConfig.enable_res / aec_preset)

# 1. Our code: one shared aligner + four linear AEC lanes.
pre = pipeline.front_process(microphones_hop, far_hop)

# 2. External Python owner: SRP-PHAT + BF. The C unit supplies
#    this stage, but the Python reference intentionally keeps the seam open.
mono_hop, effective_weights = external_srp_bf.process(pre.linear_channels)

# 3. Our code again: project AEC context with those exact weights, then run the
#    pipeline's one post-beam RES (the one stateful gain calculator, as the C unit's
#    audio_pipeline_4ch_fuse() does).  post.context.res_gain is filled in, or None
#    with enable_res off.
post = pipeline.fuse(
    pre,
    BeamformerFrame(samples=mono_hop, weights=effective_weights),
)
frame = post.spec_frame()               # g_res_ctx = post.context.res_gain, like the C frame
#    Runtime strength: pipeline.post_beam_res.set_preset(...) (C: aec_res_set_preset() on
#    audio_pipeline_4ch_res()).  The existing mono NR+RES seam consumes post.beamformed
#    and post.context.
```

Python shapes and ordering are contractual:

- `pre.linear_channels`: `[hop, 4]`;
- returned `samples`: `[hop]`;
- returned effective `weights`: `[4, n_freqs]`, under
  `out[f] = sum(weights[ch,f] * in[ch,f])`;
- external results must resume in `frame_index` order because the pipeline's
  `PostBeamResidualSuppressor` has temporal state;
- a queued Python `FrontFrame` owns snapshots of all four contexts, so
  later linear-AEC calls cannot overwrite it;
- a queued frame is bound to the pipeline instance that created it and cannot
  resume on another instance;
- `reset()` invalidates every frame still in flight and resets the lanes, the
  shared delay and the pipeline's own `PostBeamResidualSuppressor`.

If the external BF is a multi-frame/time-domain filter, `weights` must mean
its exact current effective frequency response. Returning only a mono waveform
is insufficient for the traditional RES path.

The implementation never chooses one lane's RES and never takes the minimum
of four lane gains. It coherently projects error and near spectra, converts
each lane's R2 into an echo-phase-bearing residual vector and combines that
vector with the same spatial weights; `fuse()` then runs exactly one stateful
post-beam suppressor (the unit's `AecRes` in C, `PostBeamResidualSuppressor` in Python) over the
fused frame to calculate the mono gain used after NR (the core's `four_aec_fuse()` leaves that to the caller). It does
not calculate unused beamformed echo/far spectra. Residual normalization uses
one square root and one divide per channel/bin; the public weights-only path
performs 12 complex MACs per bin (error + near + residual across four lanes),
while the unit's `audio_pipeline_4ch_fuse()` reuses its atomic mono spectrum and performs 8
(near + residual). Comfort-noise power projection is a separate real SIMD
accumulation.

For the fixed four-channel BF, the internal RLS covariance is laid out as
`P[M][M][F]`. This makes frequency contiguous and lets AArch64 NEON update four
bins at once without changing the public API or recurrence order. Automated
tests compare the SIMD and scalar output, effective weights, `P` and `wa`
byte-for-byte after every hop, including reset, masked, tail-bin and non-finite
recovery cases.

`AecResContext` does not export the complete stationarity/AecState surface, so
the post-beam RES omits the stationary mask. R2 itself is **not** bounded. The phase-bearing residual
vector's intermediate is not clamped: a clamp would break the projection's
`|out|^2 == r2` contract — `r2` arrives int16^2-scaled while `echo_spec` is
audio-scaled, so the honest scale routinely exceeds any fixed bound and a
clamp would simply understate residual echo. Only the overflow
fallback guards the intermediates, and it preserves the same magnitude
contract for every finite float32 input.

The core C API does not expose an async multi-frame queue. The C
unit does execute SRP-PHAT/BF and remains synchronous; it supports both
the heap convenience constructor and the caller-owned, zero-heap static-pool
constructor shown above. Automated tests prove caller-pool and heap lifecycle,
byte-identical core construction paths, topology, supported grids, spatial
arithmetic equivalence,
sequencing, and finite DSP output; they do not replace objective speech
metrics or subjective quality sign-off on real array recordings.

DOA and BF follow the same selected no-padding grid as AEC/NR/RES:

| sample rate | frame | FFT | hop | bins | selection |
|---:|---:|---:|---:|---:|---|
| 16 kHz | 256 | 256 | 128 | 129 | default |
| 16 kHz | 512 | 512 | 256 | 257 | `cfg.core.fft_size = 512` |
| 48 kHz | 1024 | 1024 | 512 | 513 | default/only main grid |

Frame and hop are derived from the selected FFT rather than independently
configured: `frame == FFT`, `hop == frame/2`. SRP receives the matching
sample rate, FFT and bin count for steering; BF operates on the exact
`FFT/2+1` spectra exported by the four linear AEC lanes. The raw runner
selects the same tuple with `--fft-size 256|512|1024` and rejects cross-rate
combinations.

DOA, BF and AEC/NR/RES always use the same selected tuple, so changing the grid
changes the complete pipeline atomically; there is no second spatial timing
domain.

Reproduce the two checked-in recording tests from the `Audio_ALG` directory:

```bash
../.venv/bin/python -m pipelines.4ch_aec_bf_nr_res.evaluate_recordings

# Complete C SRP-PHAT/BF path:
make -C pipelines/4ch_aec_bf_nr_res audio_pipeline_4ch_raw
../.venv/bin/python -m pipelines.4ch_aec_bf_nr_res.evaluate_external_recordings

# Exercise both checked-in 16 kHz grids:
../.venv/bin/python -m pipelines.4ch_aec_bf_nr_res.evaluate_external_recordings \
  --fft-size 256
../.venv/bin/python -m pipelines.4ch_aec_bf_nr_res.evaluate_external_recordings \
  --fft-size 512
```

The command is an acceptance test, not only a report generator. It exits
nonzero unless both cases remain finite, preserve the `1 matcher / 4 linear /
1 post-beam RES` resource boundary, acquire a solid nonzero shared delay, and
finish within the declared delay tolerance of the independent file-level
measurement (the C low-latency evaluator uses `max(half hop, 5 ms)`).
Use `--no-contract-check` only when inspecting a deliberately changed fixture.
`evaluate_external_recordings` runs the C unit and the example chain (`audio_pipeline_4ch_raw`)—shared
matcher, four linear AEC lanes, SRP-PHAT, BF, one NR and one post-beam RES—and verifies
the reported DOA/BF grids. The separate Python `evaluate_recordings` command
uses `EqualWeightBeamformer` only to test the open Python pre/post seam.

The unit's default is a four-microphone UCA with 35 mm radius only so the API
has a runnable example. Product integration must set the measured UCA radius,
ULA spacing, or all four custom `(x, y)` coordinates and confirm channel order.
The checked-in recordings do not carry array-geometry metadata, so their DOA
angles and attenuation report are integration smoke evidence, not a beamformer
quality sign-off.

`woman(ref).wav` and `man.wav` are two stems on one source timeline, while that
timeline and `unprocessed_4ch.wav` have different recording starts. The
evaluator estimates the one fixture offset from the near stem and applies it to
both. It deliberately does **not** align the far stem directly to its echo in
the mic, because doing so would erase the acoustic/system delay that the one
live shared matched filter is supposed to estimate. The C evaluator uses the
same alignment rule, writes raw float hops for `audio_pipeline_4ch_raw`, and checks
the complete `1 matcher / 4 linear AEC / SRP-PHAT / BF / 1 NR / 1 post-RES`
resource and delay contract.
