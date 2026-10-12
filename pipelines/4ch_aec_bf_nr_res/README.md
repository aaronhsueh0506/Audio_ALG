# 4-channel PBFDKF + beamforming + NR/RES application

This is the conventional four-channel product application directory. It owns
the reusable four-lane linear-AEC front end, the SRP-PHAT/GSC integration, host
evaluation tools and their tests. The sibling `../4ch_alignulcnet/`
application reuses the public front-only core API; it does not duplicate AEC,
DOA, GSC or NR implementations.

This shares the mono pipeline's zero-padding-free grid convention (frame ==
FFT, hop == frame/2 -- see the supported-grids table below), not an AIAEC
neural-model deployment path. The same boundary is available as a Python
reference and as a synchronous C library.

The front end's boundary is intentionally narrow, and the post-beam stages are
not part of it:

```text
raw/aligned far ──> one shared aligner (MATCHED, FIXED or EXTERNAL)
                              |
                              v (one aligned reference, identical to every lane)
mic 0 ──> linear AEC 0 ─┐ (computes the far-end FFT: aec_process_context())
mic 1 ──> linear AEC 1 ─┤ (borrows lane 0's far_spec: aec_process_context_shared_far())
mic 2 ──> linear AEC 2 ─┤ (borrows lane 0's far_spec: aec_process_context_shared_far())
mic 3 ──> linear AEC 3 ─┘ (borrows lane 0's far_spec: aec_process_context_shared_far())
                        │    four_aec_front_process()
                        └─>[external SRP-PHAT/GSC: weights + beamformed error]
                              │    four_aec_fuse()   → SeSpecFrame
                              v
        se_res_stage_gain ─ se_anr_stage_gain ─ sk_min_f32 ─ sk_capply_gain_f32
                         ─ aec_cng_fill ─ ac_istft_push → mono hop     (the application's own code)
```

Each `linear AEC` box still runs its own PBFDKF/shadow filter, residual-echo
(R2) estimate, and `DominantNearend` hold-state update every hop -- the echo
path genuinely differs per microphone, and the next hop's onset-gated R2
depends on that lane's own hold-state. What it does not do is compute its
own mono suppression gain: `AecConfig.spatial_linear_context=1` (set
internally by this pipeline, not caller-configurable) skips straight to the
`DominantNearend` update and leaves `res_gain` `NULL`, so the post-beam RES
gain (`se_res_stage_gain()` over the fused frame) is the only place a
suppression gain is calculated.

**Far-end FFT sharing** (one far-end rfft/hop instead of four): all four
lanes see the byte-identical `p->aligned_ref` every hop and realign together on
any delay change (see the hard invariants below) -- the precondition for
safely sharing one lane's far-end spectrum across the other three was already
true before this existed. Lane 0 runs `aec_process_context()` (computes its
own far-end FFT, as always); `aec_get_res_context()`'s always-populated
`far_spec` field is then handed to lanes 1-3 via
`aec_process_context_shared_far()`, which borrows it instead of computing a
redundant, byte-identical transform. `four_aec_nr_res_far_fft_real_compute_count()`
exposes a running per-hop count for tests to confirm the 4x→1x drop directly,
rather than trusting the wiring by inspection alone. `ref` is still passed to
every lane even when its far-end spectrum is borrowed -- the OLA far-buffer
time-domain history and every non-FFT use of the raw far signal (saturation
detection, delay estimation, mu_scale) are per-lane state, untouched by this
sharing.

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
- The Python reference and the C front end leave SRP-PHAT/GSC at an explicit
  ownership seam. The complete C wrapper in `audio_pipeline_4ch.*` connects
  that seam to the reusable implementations under `pipelines/third_party/`
  (a sibling of this directory, shared by every pipeline).
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
coherence remains an explicit caller contract. The bundled complete wrapper uses
this form with `gsc_process_with_weights()`. The far-end spectrum remains the one
shared digital render reference -- it is verified equal across lanes and is not
spatially weighted. The result passes through one mono NR+RES path without a
fifth AEC or replicated post-filters. When `front.delay.changed` is non-zero, an
external beamformer that owns STFT/OLA state must clear that overlap state
before consuming the frame; the application's own synthesis is not restarted
(the lanes realign their filters in place).

## C API

The deployable C seam is:

- [`4aec_nr_res.h`](4aec_nr_res.h): public config, lifecycle, the two per-hop
  calls (`four_aec_front_process()`, `four_aec_fuse()`), the stage-config helper
  and structural audit accessors;
- [`4aec_nr_res.c`](4aec_nr_res.c): one shared `DelayAec3`, four linear `Aec`
  instances and the coherent context projection. The residual suppressor (`AecRes`,
  `aec_res.h`), the MMSE-LSA instance and the inverse transform are carved and
  chained by the applications below;
- [`4aec_projection_kernels.h`](4aec_projection_kernels.h): byte-equivalent
  scalar/NEON complex projection, residual-vector, and comfort-noise kernels;
- [`4aec_nr_res_internal.h`](4aec_nr_res_internal.h): private delay-admission
  helpers;
- [`4aec_nr_res_static.c`](4aec_nr_res_static.c): caller-owned-pool example
  that carves the front end and the post-beam stages from one block and chains
  them, following the same query → allocate → `init` → process → destroy →
  release sequence as the mono `mono_aec_nr_res/static_main.c`;
- [`audio_pipeline_4ch.h`](audio_pipeline_4ch.h) and
  [`audio_pipeline_4ch.c`](audio_pipeline_4ch.c): complete wrapper (pool-first
  `get_mem_requirements`/`init_ex`/`init`, plus a `create` heap convenience
  wrapper — same two-tier convention as `4aec_nr_res.h` above) that inserts
  third-party SRP-PHAT/DOA smoothing and GSC between `four_aec_front_process()`
  and `four_aec_fuse()` and chains the post-beam stages (`run_post_stages()`);
- [`audio_pipeline_4ch_raw.c`](audio_pipeline_4ch_raw.c): raw-float host CLI for running
  the complete wrapper on recorded four-channel fixtures (heap `create()` —
  a one-shot host tool, not a board-deployment demo);
- [`audio_pipeline_4ch_static.c`](audio_pipeline_4ch_static.c): caller-owned-pool
  example for the COMPLETE wrapper (front end + real SRP-PHAT + real GSC, not an
  externally-supplied fixed-weight stand-in), following the same query →
  allocate → `init_ex` → process → destroy → release sequence as
  `4aec_nr_res_static.c` above;
- [`tests/test_4aec_nr_res.c`](tests/test_4aec_nr_res.c): 16/48 kHz grid, lifecycle,
  sequence rules, invalid-config, pool-boundary, heap/static byte-parity,
  weight, delay-machine and finite-output acceptance tests of the front end;
  [`tests/conv_chain_4ch.h`](tests/conv_chain_4ch.h) writes the post-beam chain over
  it as the reference the application tests compare against;
- [`tests/test_audio_pipeline_4ch.c`](tests/test_audio_pipeline_4ch.c): complete-wrapper lifecycle,
  topology, reset, strength setters, stage sizing, finite-output tests, and the
  bit-exact equality of the wrapper's chain with the reference chain;
- [`../third_party/tests/test_spatial_third_party.c`](../third_party/tests/test_spatial_third_party.c):
  scalar/SIMD PHAT, cached SRP, exported-GSC-weight and VAD pool equivalence
  tests. These live with the modules they cover; `make test` here delegates to
  `make -C ../third_party test`.

For side-by-side reading, the files and public calls map directly:

| Original mono pipeline | Four-channel counterpart |
|---|---|
| `mono_aec_nr_res/audio_pipeline.h` | `4ch_aec_bf_nr_res/4aec_nr_res.h` (front end) + `audio_pipeline_4ch.h` |
| `mono_aec_nr_res/audio_pipeline.c` | `4ch_aec_bf_nr_res/4aec_nr_res.c` + `audio_pipeline_4ch.c` (chain) |
| `mono_aec_nr_res/static_main.c` | `4ch_aec_bf_nr_res/4aec_nr_res_static.c` |
| `mono_aec_nr_res/main.c` host runner | `4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c` raw host runner |
| `mono_aec_nr_res/tests/test_audio_pipeline.c` | `4ch_aec_bf_nr_res/tests/test_4aec_nr_res.c` |

The spatial layer keeps the same order as `audio_pipeline.c`: instance,
validation, default config/construction, per-hop processing, reset/teardown,
then read-only accessors. The front-end layer retains caller-pool APIs; the
complete spatial wrapper is a separate heap convenience layer.

| Mono `audio_pipeline_*` | Four-channel `four_aec_*` front end |
|---|---|
| `default_config` | `four_aec_nr_res_default_config` |
| `get_mem_requirements` | `four_aec_nr_res_get_mem_requirements` |
| `init_ex` / `init` | `four_aec_nr_res_init_ex` / `_init` |
| `create` | `four_aec_nr_res_create` |
| `process_spec` | `four_aec_front_process` → external beamformer → `four_aec_fuse` |
| `reset` / `destroy` | `four_aec_nr_res_reset` / `_destroy` |

Heap convenience (the front end, with the post-beam stages written out):

```c
FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);
FourAecNrRes *pipeline = four_aec_nr_res_create(&cfg);
FourAecFront front;
SeSpecFrame f;

/* microphones is interleaved [hop][4], reference is [hop]. */
four_aec_front_process(pipeline, microphones, reference, &front);

/* External SRP-PHAT/GSC consumes front.linear_spectra and produces
 * channel-major Complex[4][n_freqs] weights (and, optionally, the mono error).
 * Convention: out[k] = sum(weights[ch,k] * in[ch,k]); no conjugation. */
external_srp_gsc_update(front.linear_spectra, weights);

four_aec_fuse(pipeline, weights, NULL, &f);       /* NULL: rebuild E from the weights */

/* The application chains its own stages over f (tests/conv_chain_4ch.h is the
 * reference): */
res_gain = se_res_stage_gain(res, &f);
nr_gain  = se_anr_stage_gain(nr, &f, extra_noise, 1);
sk_min_f32(total_gain, nr_gain, res_gain, f.n_freqs);
sk_capply_gain_f32(spec, f.e, total_gain, f.n_freqs);
aec_cng_fill(&rng, spec, f.cn, res_gain, nr_gain, SE_CNG_NR_GAIN_FLOOR, f.n_freqs);
ac_istft_push(istft, spec, mono_output);

four_aec_nr_res_destroy(pipeline);
```

For a product that keeps linear AEC, shared RES, CNG and synthesis but omits
MMSE-LSA, set `cfg.enable_nr = 0` in the `AudioPipeline4ChConfig.core` before
querying memory or constructing the wrapper. The wrapper's static-memory
requirement then excludes the NR state. The front end itself reads none of
`enable_res` / `enable_nr` / `enable_cng` / `legacy_amin`: they configure the
application's stages (`four_aec_nr_res_stage_configs()` derives the `AecConfig`
and `AnrConfig`). `enable_post` selects whether the lanes keep the
frequency-domain post context and the front end carves the fuse buffers
(`enable_post = 0` is the front-only seam the Align-ULCNet application uses).

`enable_res` (default 1) selects the post-beam residual suppressor. With 0 no
suppressor is built (its storage leaves the pool), the gain is the denoiser's
alone and comfort noise, which fills only RES-cut bins, is skipped;
`audio_pipeline_4ch_set_aec_preset()` and `_post_split_floor()` then refuse.

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

The COMPLETE wrapper (front end + SRP-PHAT + GSC + the post-beam stages in one
instance) uses the same pattern one level up --
`audio_pipeline_4ch_get_mem_requirements()` composes the front end's own
requirement with the SRP/GSC sub-modules' `*_get_mem_size()`, the `AecRes`, the
denoiser, the synthesis object and this wrapper's own scratch, and
`audio_pipeline_4ch_init_ex()` places all of it, zero-heap, in one
caller-owned pool:

```c
AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(16000);
AudioPipeline4ChMemReq req;
audio_pipeline_4ch_get_mem_requirements(&cfg, &req);
void *pool = platform_alloc(req.bytes, req.alignment);
AudioPipeline4Ch *p =
    audio_pipeline_4ch_init_ex(pool, (size_t)req.bytes, &cfg, &req);

/* audio_pipeline_4ch_process()/_process_with_activity() per hop. */

audio_pipeline_4ch_destroy(p); /* does not release caller-owned memory */
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
make 4aec_nr_res_static     # build caller-pool reference executable (front end + stages)
make audio_pipeline_4ch_static    # build caller-pool reference executable (complete wrapper)
make audio_pipeline_4ch_raw      # build the complete recording-validation runner
make test_4aec_nr_res      # build only the front-end 4-channel test binary
make test_audio_pipeline_4ch     # build only the complete spatial-wrapper test
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

`libaudio_pipeline_4ch.a` contains `4aec_nr_res.o` and the shared
`se_config.o` / `se_anr_stage.o` of `pipelines/common`. Standard and ULCNet
wrapper objects are linked directly into their application binaries; no
per-application archive is produced. A standard consumer links the front end with
`audio_pipeline_4ch.o`, `libdoa.a`, `libgsc.a`,
`libspatial_common.a`, `libaec.a`, `libanr.a`, `libaudio_common.a`, and
`-lm`, in the order demonstrated by this directory's Makefile.

## Python reference seam

```python
from importlib import import_module

aec4 = import_module("pipelines.4ch_aec_bf_nr_res")
BeamformerFrame = aec4.BeamformerFrame
FourChannelAecPipeline = aec4.FourChannelAecPipeline

pipeline = FourChannelAecPipeline()      # no beamformer is created

# 1. Our code: one shared aligner + four linear AEC lanes.
pre = pipeline.front_process(microphones_hop, far_hop)

# 2. External Python owner: SRP-PHAT + GSC. The complete C wrapper supplies
#    this stage, but the Python reference intentionally keeps the seam open.
mono_hop, effective_weights = external_srp_gsc.process(pre.linear_channels)

# 3. Our code again: project AEC context with those exact weights and calculate
#    one mono residual-suppression gain. The existing mono NR+RES seam consumes
#    post.beamformed and post.context.
post = pipeline.fuse(
    pre,
    BeamformerFrame(samples=mono_hop, weights=effective_weights),
)
```

Python shapes and ordering are contractual:

- `pre.linear_channels`: `[hop, 4]`;
- returned `samples`: `[hop]`;
- returned effective `weights`: `[4, n_freqs]`, under
  `out[f] = sum(weights[ch,f] * in[ch,f])`;
- external results must resume in `frame_index` order because post-BF RES has
  temporal state;
- a queued Python `FrontFrame` owns snapshots of all four contexts, so
  later linear-AEC calls cannot overwrite it;
- a queued frame is bound to the pipeline instance that created it and cannot
  resume on another instance;
- `reset()` invalidates every frame still in flight.

If the external GSC is a multi-frame/time-domain filter, `weights` must mean
its exact current effective frequency response. Returning only a mono waveform
is insufficient for the traditional RES path.

The implementation never chooses one lane's RES and never takes the minimum
of four lane gains. It coherently projects error and near spectra, converts
each lane's R2 into an echo-phase-bearing residual vector and combines that
vector with the same spatial weights, then runs exactly one stateful
`PostBeamResidualSuppressor` to calculate the mono gain used after NR. It does
not calculate unused beamformed echo/far spectra. Residual normalization uses
one square root and one divide per channel/bin; the public weights-only path
performs 12 complex MACs per bin (error + near + residual across four lanes),
while the complete GSC wrapper reuses its atomic mono spectrum and performs 8
(near + residual). Comfort-noise power projection is a separate real SIMD
accumulation.

For the fixed four-channel GSC, the internal RLS covariance is laid out as
`P[M][M][F]`. This makes frequency contiguous and lets AArch64 NEON update four
bins at once without changing the public API or recurrence order. Automated
tests compare the SIMD and scalar output, effective weights, `P` and `wa`
byte-for-byte after every hop, including reset, masked, tail-bin and non-finite
recovery cases.

One parity limitation remains explicit: `AecResContext` does not yet export
the complete stationarity/AecState surface, so the post-beam RES omits the
stationary mask. R2 itself is **not** bounded. The phase-bearing residual
vector's intermediate is not clamped: a clamp would break the projection's
`|out|^2 == r2` contract — `r2` arrives int16^2-scaled while `echo_spec` is
audio-scaled, so the honest scale routinely exceeds any fixed bound and a
clamp would simply understate residual echo. Only the overflow
fallback guards the intermediates, and it preserves the same magnitude
contract for every finite float32 input.
This is structurally correct but **not bit-exact** to a future fully extracted
AEC3 post-beam RES API; it must be cohort-tuned before production sign-off.

The core C API does not expose an async multi-frame queue. The complete C
wrapper does execute SRP-PHAT/GSC and remains synchronous; it supports both
the heap convenience constructor and the caller-owned, zero-heap static-pool
constructor shown above. Automated tests prove caller-pool and heap lifecycle,
byte-identical core construction paths, topology, supported grids, spatial
arithmetic equivalence,
sequencing, and finite DSP output; they do not replace objective speech
metrics or subjective quality sign-off on real array recordings.

DOA and GSC now follow the same selected no-padding grid as AEC/NR/RES:

| sample rate | frame | FFT | hop | bins | selection |
|---:|---:|---:|---:|---:|---|
| 16 kHz | 256 | 256 | 128 | 129 | default |
| 16 kHz | 512 | 512 | 256 | 257 | `cfg.core.fft_size = 512` |
| 48 kHz | 1024 | 1024 | 512 | 513 | default/only main grid |

Frame and hop are derived from the selected FFT rather than independently
configured: `frame == FFT`, `hop == frame/2`. SRP receives the matching
sample rate, FFT and bin count for steering; GSC operates on the exact
`FFT/2+1` spectra exported by the four linear AEC lanes. The raw runner
selects the same tuple with `--fft-size 256|512|1024` and rejects cross-rate
combinations.

There is no legacy/downsample grid entry: DOA, GSC and AEC/NR/RES always use
the same selected tuple. Changing the grid therefore changes the complete
pipeline atomically rather than creating a second spatial timing domain.

Reproduce the two checked-in recording tests from the `Audio_ALG` directory:

```bash
../.venv/bin/python -m pipelines.4ch_aec_bf_nr_res.evaluate_recordings

# Complete C SRP-PHAT/GSC path:
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
`evaluate_external_recordings` runs the complete C wrapper—shared matcher,
four linear AEC lanes, SRP-PHAT, GSC, one NR and one post-beam RES—and verifies
the reported DOA/GSC grids. The separate Python `evaluate_recordings` command
uses `EqualWeightBeamformer` only to test the open Python pre/post seam.

The wrapper default is a four-microphone UCA with 35 mm radius only so the API
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
the complete `1 matcher / 4 linear AEC / SRP-PHAT / GSC / 1 NR / 1 post-RES`
resource and delay contract.
