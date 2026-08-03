# 4-ch shared-delay AEC + post-beam NR/RES

This shares the mono pipeline's zero-padding-free grid convention (frame ==
FFT, hop == frame/2 -- see the supported-grids table below), not an AIAEC
neural-model deployment path. The same boundary is available as a Python
reference and as a synchronous C library.

The core boundary is intentionally narrow:

```text
raw far reference ──> one shared matched-filter/delay estimator
                              |
                              v (one aligned reference)
mic 0 ──> linear AEC 0 ─┐                              ┌─> mono NR
mic 1 ──> linear AEC 1 ─┤  PRE HANDOFF   POST RESUME  ├─> gain fusion/OLA
mic 2 ──> linear AEC 2 ─┼─>[external SRP-PHAT/GSC]───┴─> mono RES
mic 3 ──> linear AEC 3 ─┘
```

Hard invariants:

- There is exactly **one** matched-filter instance. It uses a configurable
  capture proxy channel and the raw far reference.
- Only the adaptive linear filter is replicated, exactly four times.
- All four filters consume the same delay-aligned reference and reset together
  if that shared alignment changes.
- No NR, RES, neural model, or delay estimator is replicated per microphone.
- The Python reference and the core C API leave SRP-PHAT/GSC at an explicit
  pre/post ownership seam. The complete C wrapper in `audio_pipeline_4ch.*` connects
  that seam to the reusable implementations under this project's
  `third_party/` directory.
- `EqualWeightBeamformer` exists only as a deterministic offline/test adapter;
  it is never selected by default.

The external beamformer must expose its effective complex weights with shape
`[4, n_freqs]`. The Python reference also accepts the external mono hop. The C
core API synthesizes the mono spectrum internally from the supplied weights,
so independently supplied output samples and context cannot come from
different beamformer states. The complete C wrapper can safely skip that one
reconstruction because `gsc_process_with_weights()` produces its mono spectrum
and effective weights atomically in the same call; the internal-only trusted
seam is not exposed to external beamformers. The far-end spectrum remains the
one shared digital render reference—it is verified equal across lanes and is
not spatially weighted. The result passes through one mono NR+RES path without
a fifth AEC.

## C API

The deployable C seam is:

- [`4aec_nr_res.h`](4aec_nr_res.h): public config, lifecycle, pre/post
  calls, frame token, and structural audit accessors;
- [`4aec_nr_res.c`](4aec_nr_res.c): one shared `DelayAec3`, four linear
  `Aec` instances, coherent context projection, one `SuppressionGain`, one
  MMSE-LSA instance, and one final iFFT/OLA;
- [`4aec_projection_kernels.h`](4aec_projection_kernels.h): byte-equivalent
  scalar/NEON complex projection, residual-vector, and comfort-noise kernels;
- [`4aec_nr_res_internal.h`](4aec_nr_res_internal.h): internal-only trusted
  GSC-spectrum continuation used by the complete wrapper, not a public
  external-beamformer API;
- [`4aec_nr_res_static.c`](4aec_nr_res_static.c): caller-owned-pool example
  following the same query → allocate → `init` → process → destroy →
  release sequence as the mono `aec_nr_pipeline_static.c`;
- [`audio_pipeline_4ch.h`](audio_pipeline_4ch.h) and
  [`audio_pipeline_4ch.c`](audio_pipeline_4ch.c): complete heap wrapper that inserts
  third-party SRP-PHAT/DOA smoothing and GSC between `process_pre()` and
  `process_post()`;
- [`audio_pipeline_4ch_raw.c`](audio_pipeline_4ch_raw.c): raw-float host CLI for running
  the complete wrapper on recorded four-channel fixtures;
- [`test_4aec_nr_res.c`](test_4aec_nr_res.c): 16/48 kHz grid, lifecycle,
  ordering, token invalidation, invalid-config, pool-boundary, heap/static
  byte-parity, weight, and finite-output acceptance tests;
- [`test_audio_pipeline_4ch.c`](test_audio_pipeline_4ch.c): complete-wrapper lifecycle,
  topology, reset, and finite-output tests;
- [`test_spatial_third_party.c`](test_spatial_third_party.c): scalar/SIMD PHAT,
  cached SRP, and exported-GSC-weight equivalence tests.

For side-by-side reading, the files and public calls map directly:

| Original mono pipeline | Four-channel counterpart |
|---|---|
| `audio_pipeline.h` | `4ch_pipelines/4aec_nr_res.h` |
| `audio_pipeline.c` | `4ch_pipelines/4aec_nr_res.c` |
| `aec_nr_pipeline_static.c` | `4ch_pipelines/4aec_nr_res_static.c` |
| `aec_nr_pipeline.c` host runner | `4ch_pipelines/audio_pipeline_4ch_raw.c` raw host runner |
| `test_audio_pipeline.c` | `4ch_pipelines/test_4aec_nr_res.c` |

The spatial layer keeps the same order as `audio_pipeline.c`: instance,
validation, default config/construction, per-hop processing, reset/teardown,
then read-only accessors. The core layer retains caller-pool APIs; the complete
spatial wrapper is a separate heap convenience layer.

| Mono `audio_pipeline_*` | Four-channel `four_aec_nr_res_*` |
|---|---|
| `default_config` | `default_config` |
| `get_mem_requirements` | `get_mem_requirements` |
| `init_ex` / `init` | `init_ex` / `init` |
| `create` | `create` |
| `process` | `process_pre` → external beamformer → `process_post` |
| `reset` / `destroy` | `reset` / `destroy` |

Heap convenience:

```c
FourAecNrResConfig cfg;
FourAecNrResPreFrame pre;
FourAecNrRes *pipeline;

cfg = four_aec_nr_res_default_config(16000);
pipeline = four_aec_nr_res_create(&cfg);

/* microphones is interleaved [hop][4], reference is [hop]. */
four_aec_nr_res_process_pre(pipeline, microphones, reference, &pre);

/* External SRP-PHAT/GSC updates channel-major Complex[4][n_freqs].
 * Convention: out[k] = sum(weights[ch,k] * in[ch,k]); no conjugation. */
external_srp_gsc_update(pre.linear_interleaved, weights);

four_aec_nr_res_process_post(pipeline, &pre.token, weights, mono_output);
four_aec_nr_res_destroy(pipeline);
```

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

/* The per-hop pre -> external beamformer -> post calls are identical. */

four_aec_nr_res_destroy(pipeline); /* does not release caller-owned memory */
platform_free(pool);
```

Only one pre frame may be in flight. `process_post()` must receive the exact
token returned by `process_pre()`; replay, cross-instance use, and a token
invalidated by `reset()` are rejected. Invalid weights leave the frame pending
so the caller may correct and retry them. `create()` allocates one complete
pool; `init_ex()` uses only the caller's aligned pool. Both process calls
perform no allocation.

Build the archive or run the complete C acceptance gate from
`Audio_ALG/pipelines/4ch_pipelines`:

```bash
make libaudio_pipeline_4ch.a
make 4aec_nr_res_static     # build caller-pool reference executable
make audio_pipeline_4ch_raw      # build the complete recording-validation runner
make test_4aec_nr_res      # build only the core 4-channel test binary
make test_audio_pipeline_4ch     # build only the complete spatial-wrapper test
make test_spatial_third_party
make test                  # run only the isolated 4ch acceptance gate
make NO_STDIO=1 audit-no-stdio

# Query the exact pool budget for the supported grids.
"$(make -s print-bin-dir)/4aec_nr_res_static" --print-mem-size --sample-rate 16000 --fft-size 256
"$(make -s print-bin-dir)/4aec_nr_res_static" --print-mem-size --sample-rate 16000
"$(make -s print-bin-dir)/4aec_nr_res_static" --print-mem-size --sample-rate 48000
```

`libaudio_pipeline_4ch.a` contains only `4aec_nr_res.o` and
`audio_pipeline_4ch.o`. It deliberately does not absorb the algorithm
libraries. A consumer links it with `libdoa.a`, `libgsc.a`,
`libspatial_common.a`, `libaec.a`, `libmmse_lsa.a`, `libaudio_common.a`, and
`-lm`, in the order demonstrated by this directory's Makefile.

## Python reference seam

```python
from importlib import import_module

aec4 = import_module("pipelines.4ch_pipelines")
BeamformerFrame = aec4.BeamformerFrame
FourChannelAecPipeline = aec4.FourChannelAecPipeline

pipeline = FourChannelAecPipeline()      # no beamformer is created

# 1. Our code: one matcher + four linear AEC lanes.
pre = pipeline.process_pre_beamformer(microphones_hop, far_hop)

# 2. External Python owner: SRP-PHAT + GSC. The complete C wrapper supplies
#    this stage, but the Python reference intentionally keeps the seam open.
mono_hop, effective_weights = external_srp_gsc.process(pre.linear_channels)

# 3. Our code again: project AEC context with those exact weights and calculate
#    one mono residual-suppression gain. The existing mono NR+RES seam consumes
#    post.beamformed and post.context.
post = pipeline.process_post_beamformer(
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
- a queued Python `PreBeamformerFrame` owns snapshots of all four contexts, so
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

One parity limitation remains explicit: `AecResContext` does not yet export
unbounded R2 or the complete stationarity/AecState surface. The post-beam RES
therefore uses bounded R2 for both gain inputs and omits the stationary mask.
This is structurally correct but **not bit-exact** to a future fully extracted
AEC3 post-beam RES API; it must be cohort-tuned before production sign-off.

The core C API does not expose an async multi-frame queue. The complete C
wrapper does execute SRP-PHAT/GSC, but it remains synchronous and heap-backed.
Automated tests prove caller-pool and heap lifecycle, byte-identical core
construction paths, topology, supported grids, spatial arithmetic equivalence,
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
../.venv/bin/python -m pipelines.4ch_pipelines.evaluate_recordings

# Complete C SRP-PHAT/GSC path:
make -C pipelines/4ch_pipelines audio_pipeline_4ch_raw
../.venv/bin/python -m pipelines.4ch_pipelines.evaluate_external_recordings

# Exercise both checked-in 16 kHz grids:
../.venv/bin/python -m pipelines.4ch_pipelines.evaluate_external_recordings \
  --fft-size 256
../.venv/bin/python -m pipelines.4ch_pipelines.evaluate_external_recordings \
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
