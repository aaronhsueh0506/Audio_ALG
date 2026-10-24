# Board integration: separate libraries, one unit

## Scope and prerequisites

This document is the firmware-side contract for the delivered pipeline code: pool
sizing, the query -> init -> process -> destroy sequence, the memory descriptor of the one
wrapped unit, the board build, and the on-target verification checklist.

What is delivered, and how it is initialised:

- Only the multi-channel linear AEC + beamformer is wrapped as one unit
  (`AudioPipeline4Ch`, `../pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.h`: shared delay aligner, four
  linear AEC lanes, SRP-PHAT, BF and the `fuse()` projection; it holds no RES, so `fuse()` leaves
  `frame->g_res_ctx` NULL). It has a
  fixed 32-byte descriptor and an 8-point `init_ex()` gate.
- Every other stage is a separate library that the integrator initialises with its own size
  query and `init`, and chains: the mono `Aec` (lib/aec, whose own RES gain is `ctx.res_gain`),
  `AecRes` (lib/aec; the 4-channel post-beam RES, which the application creates itself after the unit or
  the core with `aec_res_get_mem_size()`/`aec_res_init()`, as `audio_pipeline_4ch_static.c` and
  `4aec_nr_res_static.c` do; the mono RES is not a separate object, it is inside the `Aec` call),
  `AnrDenoiser` (lib/nr), `AcIstft` (audiolib), the DFN2 stage and rate bridge (`dfn_res_stage.h`,
  `dfn_rate_bridge.h`, which carry their own descriptors). There is no mono unit and no "complete"
  wrapper object; the mono flow is `Aec` plus the libraries it chains per hop.
- Each example `main` initialises the libraries it uses and writes the per-hop chaining itself
  (res gain, `se_anr_stage_gain()`, `sk_min_f32()`, `sk_capply_gain_f32()`, `aec_cng_fill()`,
  `ac_istft_push()`). `../pipelines/samples/` holds the shared example code (`se_anr_stage.h`,
  `dfn_hop.h`, `ulcnet_nn_stage.{h,c}`). It has no ABI, no layout version and no
  descriptor; copy and adapt it. The example mains, `static_main.c` and `example_board_adapter.c`
  show the composition.

It assumes [`../pipelines/README.md`](../pipelines/README.md) has been read
first — that file owns the algorithm description, the parameter/grid
alignment, the measured memory budget ("Memory Budget"), the heap-versus-pool
statement ("Verification"), the heap-vs-pool overview ("Two Versions"), and
the tunable-parameter tables. References below of the form
`see "X" in ../pipelines/README.md` point back there.

Every relative path in this file is written from `Audio_ALG/docs/`; every
`make` invocation shown is run from `Audio_ALG/pipelines/`.

## At a glance: sequence, public/internal functions, external/internal parameters

Lifelines of the board path. The mono flow has three library objects, the 4-channel flow has the
one wrapped unit and the same two post libraries; every box is queried and initialised on its own.
The numbered sequences further down give the exact calls; the per-flow diagrams (with return values
and the swimlane over hops) are in section 0 of
[`integration_mono_zh_TW.md`](integration_mono_zh_TW.md),
[`integration_4ch_core_zh_TW.md`](integration_4ch_core_zh_TW.md) and
[`integration_4ch_spatial_zh_TW.md`](integration_4ch_spatial_zh_TW.md).

```
Firmware         Aec | AudioPipeline4Ch       AnrDenoiser          AcIstft
   |                  |                           |                   |
   |== boot (once; re-query every size, never cache) ==========================
   |-- *_get_mem_size / audio_pipeline_4ch_get_mem_requirements (0 / -1 = invalid)
   |-- platform_alloc(Sum ALIGN16(each block), 16)         pool need not be zeroed
   |-- aec_init | audio_pipeline_4ch_init_ex -->|-- aec_res_init (4ch only) -->|-- anr_init -->|-- ac_istft_init -->|   (NULL = stop)
   |== per hop ================================================================
   |-- aec_process_context | audio_pipeline_4ch_process_* (+ _fuse) -->|  frame{e,r2,cn,g_res_ctx}
   |      (mono: g_res_ctx is the Aec's own RES gain; 4ch: g_res_ctx = NULL)
   |-- se_res_stage_gain(res, &f) (4ch only) ---->|  G_res from your AecRes
   |-- se_anr_stage_gain ------------------------>|  G_nr
   |-- sk_min_f32, sk_capply_gain_f32, aec_cng_fill (firmware code)
   |-- ac_istft_push ------------------------------------------------>|  out[hop]   (1 hop behind the input)
   |== refused call or non-finite output =======================================
   |-- aec_reset | audio_pipeline_4ch_reset, aec_res_reset (4ch only), anr_reset, ac_istft_reset, rng = SE_CNG_RNG_SEED   (all together)
   |== teardown ================================================================
   |-- anr_destroy, aec_destroy | audio_pipeline_4ch_destroy, then platform_free(pool)
```

| | Public (the firmware calls these) | Internal (do not call or include) |
|---|---|---|
| Mono | `se_derive()`, `aec_get_mem_size()`/`aec_init()`, `anr_get_mem_size()`/`anr_init()`, `ac_istft_get_mem_size()`/`ac_istft_init()`, `aec_process_context()`, `se_frame_from_aec()`, `se_anr_stage_gain()`, `aec_cng_fill()`, `ac_istft_push()`, resets, destroys, `aec_set_preset()`, `anr_set_mode()` | AEC headers other than `aec.h`/`aec_res.h`, NR `noise_estimator.h`/`spp_estimator.h`/`anr_internal.h` |
| 4-channel | `audio_pipeline_4ch_*` of `audio_pipeline_4ch.h` (or `four_aec_*` of `4aec_nr_res.h` when the firmware owns the beamformer) plus the same post libraries | `4aec_nr_res_internal.h`, `4aec_projection_kernels.h`, `bf.h`/`srp.h` and `audiolib/{doa,bf}` |

Parameters: what a third party may tune (grid and rate, `filter_length`, delay mode and bank size,
presets and modes, array geometry, `num_angles`) and what stays inside
(time constants, step sizes, detector thresholds, DOA/BF recursion constants, NR tracker constants)
is classified field by field in section 0.3 of the three manuals above.

## Static APIs and pool sizing

Each library slices its own internals (HPF, PBFDKF x2, delay estimator, RES/post context,
internal FFTs for AEC; MCRA + SPP for NR) inside its single pool segment. The integrator queries
each size, aligns every segment to 16 bytes (`ALIGN16`) and places them in one block (or
several):

```c
size_t aec_sz   = aec_get_mem_size(&aec_cfg);       /* lib/aec; 0 = invalid config        */
size_t nr_sz    = anr_get_mem_size(&nr_cfg);         /* lib/nr; 0 = invalid config         */
size_t synth_sz = ac_istft_get_mem_size(&synth_cfg); /* audiolib; 0 = invalid config       */
/* + your gain/spectrum buffers and your own stages; every segment 16-byte aligned (ALIGN16) */

Aec*         aec   = aec_init(mem_aec, aec_sz, &aec_cfg);
AnrDenoiser* nr    = anr_init(mem_nr, nr_sz, &nr_cfg);
AcIstft*     istft = ac_istft_init(mem_synth, synth_sz, &synth_cfg);
/* destroy() on a static instance never frees the pool. aec_destroy() only drops the
 * instance's internal references and is NULL-safe and repeatable; anr_destroy() returns at
 * once for a pool-resident instance. Call them before releasing the pool. */
```

Query the exact pool budget for any configuration without running audio:

```bash
./aec_nr_pipeline_static --print-mem-size --sample-rate 16000
./audio_pipeline_4ch_static --print-mem-size --sample-rate 16000     # in 4ch_aec_bf_nr_res/
```

**Embedded-target integration:** allocate one contiguous, 16-byte-aligned
block of the summed total from the platform allocator, pass it in place of
the desktop `malloc` — no other change. Each pool base MUST be 16-byte
aligned (the libraries assert this).

## Sequence (mono: `Aec` plus the libraries it chains)

```
1. query    se_derive(&req, &rules, &grid, &aec_cfg);         // grid + AecConfig
            AnrConfig nr_cfg = anr_config_for_mode_grid(sample_rate, grid.fft_size, mode);
            AcStftConfig synth_cfg;  ac_stft_config_default(&synth_cfg, grid.fft_size);
            size_t aec_bytes   = aec_get_mem_size(&aec_cfg);       // 0 = invalid
            size_t nr_bytes    = anr_get_mem_size(&nr_cfg);        // 0 = invalid
            size_t synth_bytes = ac_istft_get_mem_size(&synth_cfg);// 0 = invalid
            size_t buf_bytes   = 3*ALIGN16(n_freqs*sizeof(float)) + ALIGN16(n_freqs*sizeof(Complex));
            // Query these fresh, immediately before every init below — never cache a size
            // across a build/backend/config change and replay it later. See "Warnings" below.

2. allocate void* pool = platform_alloc(ALIGN16(aec_bytes) + ALIGN16(nr_bytes) +
                                       ALIGN16(synth_bytes) + buf_bytes, 16);
            // pool need NOT be zeroed — see "Dirty-pool contract" below. Pool must stay STABLE
            // and EXCLUSIVE (nothing else reads/writes it, not shared with any other
            // instance) for the entire lifetime of the handles below.

3. init     uint8_t* cur = pool;
            Aec* aec = aec_init(cur, aec_bytes, &aec_cfg);           cur += ALIGN16(aec_bytes);
            AnrDenoiser* nr = anr_init(cur, nr_bytes, &nr_cfg);      cur += ALIGN16(nr_bytes);
            AcIstft* istft = ac_istft_init(cur, synth_bytes, &synth_cfg);  cur += ALIGN16(synth_bytes);
            // gain/spectrum buffers follow (unity, total gain, NR prior, output spectrum)
            // NULL on: a misaligned/undersized pool, an invalid config, or a sub-object init
            // failure (stderr has detail from the libraries, UNLESS they were built with
            // NO_STDIO=1 — see "NO_STDIO=1" below). A rejected init writes nothing outside
            // the pool and leaves the pool usable for a correct init.

4. process  float mic[hop], ref[hop], out[hop];   // hop = grid.hop_size
            while (have_audio()) {
                read_hop(mic, ref, hop);
                aec_process_context(aec, mic, ref);
                // per hop: G_res = f.g_res_ctx; nr = se_anr_stage_gain(nr, &f, extra, use_r2);
                //          sk_min_f32; sk_capply_gain_f32; aec_cng_fill; ac_istft_push
                if (se_frame_from_aec(aec, &f) != 0 || !(P = mono_form(&f)) ||
                    ac_istft_push(istft, P, out) != hop || !se_hop_is_finite(out, hop)) {
                    aec_reset(aec); anr_reset(nr); ac_istft_reset(istft); rng = SE_CNG_RNG_SEED;
                    continue;                                       // a refused hop: reset together
                }
                write_hop(out, hop);
            }

5. reset?   aec_reset(aec); anr_reset(nr); ac_istft_reset(istft); rng = SE_CNG_RNG_SEED;
            // optional: echo-path change, stream switch; re-zeros AEC/NR/synthesis state in place
            // and restarts the comfort-noise generator; no re-validation, no pool re-touch beyond that.

6. destroy  anr_destroy(nr); aec_destroy(aec);
            // NULL-safe; idempotent for these pool-resident instances.

7. release  platform_free(pool);   // only after step 6
```

## Sequence (4-channel: the unit plus the libraries it feeds)

The unit is the one object with a descriptor. Its query, init and teardown:

```
1. query    AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(sample_rate);
            cfg.geometry/uca_radius_m/num_angles/core.* = ...;
            AudioPipeline4ChMemReq req;
            audio_pipeline_4ch_get_mem_requirements(&cfg, &req);   // 0 = OK, -1 = invalid
            // fresh before every init_ex(); see "Warnings" below

2. allocate void* pool = platform_alloc(req.bytes, req.alignment);

3. init     AudioPipeline4Ch* p = audio_pipeline_4ch_init_ex(pool, req.bytes, &cfg, &req);
            // NULL on: any expected-descriptor mismatch (descriptor_version / layout_version /
            // backend_id / build_flags_hash / alignment / reserved / bytes), a
            // misaligned/undersized pool, an invalid cfg, or a core / SRP / BF init failure.
            // No diagnostic output at all (the unit links no stdio). audio_pipeline_4ch_init()
            // is init_ex(..., NULL): no descriptor check.

   (The post-beam RES, NR and synthesis, the neural stage or DFN2 stage and bridge are separate blocks with
    their own size queries and inits: aec_config_from_preset + aec_res_get_mem_size / aec_res_init (config
    fft_size = audio_pipeline_4ch_fft_size(p); no destroy), anr_get_mem_size / anr_init,
    ac_istft_get_mem_size / ac_istft_init, ulcnet_nn_stage_mem_size / _init,
    dfn_rate_bridge_get_mem_requirements / dfn_rate_bridge_init; the unit holds no RES)

4. process  audio_pipeline_4ch_process_with_activity(p, mics, far, vad, vad, NULL, &beam, &info);
            audio_pipeline_4ch_fuse(p, &frame);            // optional (core.enable_post = 1)
            // frame.g_res_ctx = NULL (the unit has no RES); then
            // se_res_stage_gain(res, &frame) -> se_anr_stage_gain -> sk_min_f32 -> sk_capply_gain_f32
            // -> aec_cng_fill -> ac_istft_push
            // or process_front() / process_beam() with your own mask and VAD in between

5. reset?   audio_pipeline_4ch_reset(p);  aec_res_reset(res);  anr_reset(nr);  ac_istft_reset(istft);  rng = SE_CNG_RNG_SEED;  [ulcnet_nn_stage_reset(stage);]
            // after ANY refused call too: the integrator resets the unit and its own stages together

6. destroy  audio_pipeline_4ch_destroy(p); ...      // NR, stage and bridge each have their own
7. release  platform_free(pool); ...                // only after step 6
```

### Running the model on the NPU (NN applications)

The ULCNet and DFN2 examples split the hop around the model call so the
accelerator runs between `nn_begin` and `nn_end` (the model config then needs
no `infer` callback; a `reset` callback is still called on an alignment
boundary and by the stage's `reset`):

```
// ULCNet (samples/ulcnet_nn_stage.h)
ulcnet_nn_stage_begin_complex(stage, beam.spectrum, far, delay);   // mono: ulcnet_nn_stage_begin_time(...)
if (ulcnet_nn_stage_nn_begin(stage, &job) > 0) {                   // 1 = a frame is open (UlcnetNnJob)
    rc = run_npu(&job);                                            // job outputs are NaN-prefilled; write every one
    ulcnet_nn_stage_nn_end(stage, rc);                             // rc != 0 or a non-finite output => identity
}
ulcnet_nn_stage_end(stage, out);                                   // WOLA, exactly one hop

// DFN2 (samples/dfn_hop.h): up to two frames per hop through the rate bridge
dfn_hop_begin(&hop, e, p);               // P = E.G_res formed first (by the example main)
while (dfn_hop_nn_begin(&hop, &job) == 1)
    dfn_hop_nn_end(&hop, run_npu(&job));
dfn_hop_end(&hop, out);
```

The 4-channel Align-ULCNet example splits the unit's hop once more for a board that derives a per-bin
mask and a VAD from the lane spectra of the same hop: `audio_pipeline_4ch_process_front(p, mics, far, &front)`
(lanes), the caller reads `front.linear_spectra[0]`, then
`audio_pipeline_4ch_process_beam(p, &front, vad_raw, vad_out, mask, &beam, &info)` (SRP, BF; `mask`
may be NULL). `info` returns the DOA (radians, NaN when none), the BF adaptation flag and the
alignment state (`info.delay`). `process_with_activity(v)` is `process_front` plus
`process_beam(front, v, v, NULL)`.

A frame that is never fetched, or left open at `end`, takes the
identity, so latency does not depend on the model. Every library's pool is
sized by its own query and carved from a caller-owned block; the per-hop stage chaining adds
no heap use in the hop path.

## Warnings

- **`audio_pipeline_4ch_create()` ALWAYS uses the heap** (`posix_memalign`
  under the hood) — it is the desktop/prototyping convenience path (see
  "Two Versions" in [`../pipelines/README.md`](../pipelines/README.md)),
  never the board path. **A board build must never call it** (nor `aec_create()`/`anr_create()`).
  The pool sequence above (`*_get_mem_requirements`/`*_mem_size` + `*_init_ex`/`*_init` +
  `*_destroy`) is the only sequence that touches zero heap.
- **Both the heap path and the pool path are always compiled in.** The entry point you call is the
  only switch, decided at runtime; there is no compile-time flag to set.
- **Re-query every size and descriptor after ANY build, backend, or config change** —
  a firmware rebuild, or a change to any config field that affects sizing. Never cache a
  bare byte count (or a whole `AudioPipeline4ChMemReq`) across one of those
  changes and replay it into a later build's `audio_pipeline_4ch_init_ex()` —
  that mismatch is exactly what the `expected` argument exists to catch
  (see "Descriptor semantics" below). This still holds with the fixed
  32-byte layout (below): "can be persisted byte-for-byte" is not the same
  claim as "is still valid after a rebuild/backend/config change": a
  persisted descriptor from a PREVIOUS build is exactly the stale-cache
  case `audio_pipeline_4ch_init_ex()` exists to catch. If a descriptor must be
  persisted across a restart (same board, same binary, same CPU — see
  "Descriptor semantics" for the same-endianness scope this is limited
  to), re-derive it fresh at the next boot from THAT boot's
  `audio_pipeline_4ch_get_mem_requirements()` and treat any persisted copy as
  advisory at best — never trust a persisted value as `expected` without
  also having a freshly-queried one to fall back to. The mono objects and the example programs have no
  descriptor, so re-querying the size on every build is the whole rule there.
- **Mixing `_create()` and pool APIs per submodule is not supported.**
  Don't hand-carve one of AEC/SRP/BF into a pool segment yourself while
  letting `audio_pipeline_4ch_create()` heap-allocate another. `audio_pipeline_4ch_init()`/`_init_ex()`
  own the ENTIRE unit pool layout (control block + core + SRP + BF + scratch) as one unit — there is
  no supported way to substitute a heap-obtained sub-module handle into a pool-resident
  `AudioPipeline4Ch`, or vice versa. The RES (`AecRes`), NR, synthesis and the NN stages are separate objects
  with their own pools.
- **Failure recovery is yours.** When the unit, the denoiser or a stage refuses a call, or the emitted
  hop is not finite, reset the unit (or `Aec`), the 4-channel `AecRes`, the denoiser, the synthesis (and the comfort-noise
  generator) and any neural stage together
  (`audio_pipeline_4ch_reset()` / `aec_reset()`, `aec_res_reset()`, `anr_reset()`, `ac_istft_reset()`, `ulcnet_nn_stage_reset()`).
  The DFN2 stage is not reset by an AEC delay change.

## Descriptor semantics (`AudioPipeline4ChMemReq`)

Every field is a fixed-width integer
(`uint32_t`/`uint64_t`, never `size_t` or a pointer), and the struct is
pinned to a stable, `_Static_assert`-enforced **32-byte** layout — see
"Serializing the descriptor" below for what that does (and does not) buy
you. The core `FourAecNrResMemReq`, the DFN stage and the rate bridge use the same shape with their
own version counters.

| Field | Meaning |
|-------|---------|
| `descriptor_version` | This STRUCT's own ABI version — `AUDIO_PIPELINE_4CH_DESCRIPTOR_VERSION` (currently `1`). Bumped only when `AudioPipeline4ChMemReq`'s field set/order/width changes, independent of `layout_version` below. `audio_pipeline_4ch_init_ex()` checks this FIRST, before interpreting any other field. |
| `layout_version` | `AUDIO_PIPELINE_4CH_LAYOUT_VERSION` (currently `21`). Bumped whenever the pool size, carve order or descriptor content of `../pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c` changes — including a control-block-only change of the composed core. A cached `bytes` alone is only a lower bound, so this counter is what refuses a descriptor persisted by another build. |
| `backend_id` | Compile-time FFT backend identity this build was made with, as a small integer — the core's `backend_id`, `FOUR_AEC_NR_RES_BACKEND_NE10` (2, `4aec_nr_res.h`), the only FFT backend (matches the `BACKEND=` of `../pipelines/4ch_aec_bf_nr_res/Makefile`). A process-local rodata pointer cannot be serialized safely, so `backend_id` is compared with a plain integer `==`. A descriptor carrying any other value is never valid, even at matching `bytes`. `0` is reserved for "unknown backend" and is never present in a descriptor this library actually returns. |
| `build_flags_hash` | FNV-1a-32 of a small fixed set of compile-time strings that affect the unit's carve STRUCTURE: the backend identity above, a literal carve-order token list, the alignment granularity, and the core's own hash — see `audio_pipeline_4ch_build_flags_hash()` in `../pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c`. **Covers:** a change to the unit's own carve order/buffer set/alignment, and (through the core's hash) a core-layer layout bump. **Does NOT cover:** config VALUES (`sample_rate`, `num_angles`, ...) — those change `bytes` but are config, not layout, so a caller re-querying `get_mem_requirements()` for its actual config already gets the right `bytes` regardless of this hash; AEC's/SRP's/BF's/an FFT backend's internal struct layouts (opaque blobs); the compiler/ABI/toolchain. |
| `alignment` | Required base alignment of the pool pointer, bytes. Always 16 today (the one alignment every module in this stack — AEC, NR, the NE10 FFT backend, `mem_align.h`'s `ALIGN16` — carves to). `uint32_t`, not `size_t`. |
| `reserved` | Always 0. Exists only so `bytes` (a `uint64_t`) lands on an 8-byte-aligned offset within the struct with no compiler-inserted padding — part of the fixed 32-byte layout, not a field to read or write. |
| `bytes` | Total pool size to allocate (includes the opaque `AudioPipeline4Ch` control block itself, carved at the front, plus the core, SRP-PHAT, BF, the unit's scratch buffers; it does NOT include the RES, NR, synthesis or any neural stage). `uint64_t`, not `size_t` — cast to `size_t` before passing to `malloc`/`posix_memalign`/`audio_pipeline_4ch_init*()` on a target where the two widths differ. |

`audio_pipeline_4ch_init_ex(mem, bytes, cfg, expected)` (see "Sequence" above)
automates exactly this comparison: passing the descriptor straight back in
as `expected` makes the library itself recompute the CURRENT
`descriptor_version`/`layout_version`/`backend_id`/`build_flags_hash`/
`alignment`/`bytes` and reject (`NULL`) unless every one still agrees with `expected`, so a board
integrator does not hand-write this comparison. The one discipline
this does NOT relieve you of: `expected` must itself have been queried
freshly (see "Warnings" above) — `audio_pipeline_4ch_init_ex()` can only compare
what it's given against what the CURRENT build computes; it has no way to
know whether the `expected` you passed in was itself stale relative to some
THIRD, even older, build.

## Serializing the descriptor

`AudioPipeline4ChMemReq` is a fixed-width, 32-byte POD (`_Static_assert`-pinned
in `../pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.h`, sizeof and every field's offset) — it
can be copied byte-for-byte (`memcpy`) to a file, a flash region, or a wire
message, and read back later, even by a different process, even after a
restart. This is deliberately scoped narrower than "fully portable
serialization":

- **Same-endianness only.** The board firmware and this library build run
  on the SAME CPU, so this struct provides no byte-swap helpers and makes
  no attempt to support a big-endian producer read by a little-endian
  consumer (or vice versa). If you ever need cross-endian interchange, add
  your own swap routine at the serialization boundary — it does not exist
  here today.
- **Not a substitute for re-querying.** A persisted/transmitted descriptor
  is still exactly the "stale cache" risk "Warnings" above describes — the
  fixed layout only means the BYTES survive the round trip intact, not that
  they still describe the CURRENT build. Always pass a freshly-queried
  descriptor as `expected` to `audio_pipeline_4ch_init_ex()`; treat a persisted
  copy as informational at best (e.g. a board bring-up log recording what
  the PREVIOUS boot's build looked like).

## Dirty-pool contract

None of the init functions requires a zero-filled pool. The unit clears the whole block it is
given, and AEC/NR/the FFT backend each zero
their own sub-region during their own `_init()` — so a pool filled with poison bytes inits and
processes identically to a freshly-zeroed one.
`../pipelines/4ch_aec_bf_nr_res/tests/test_audio_pipeline_4ch.c` (unit, `0xA5`-poisoned pool against heap) and
`../pipelines/mono_aec_nr_res/tests/test_mono_chain.c` (mono, pool carved from a `0xA5`-poisoned
block against the heap flavor) exercise exactly this: byte-for-byte the same output. There is no
need for a caller-side blanket `memset(pool, 0, bytes)` before `init`.

## `NO_STDIO=1` — building without libc stdio

The unit and the core print nothing at all on any path (no stdio symbol is linked), and the example
header `se_anr_stage.h` uses none. lib/aec, lib/nr and audiolib compile their init-time diagnostics out under
`NO_STDIO=1`; the DFN2 stage and rate bridge archive is audited for stdio symbols. Every failure is
signalled through the return value (`NULL`/`-1`/`0` size), so a board image that cannot or will not
link libc's stdio (no console, or a policy that forbids the stdio symbol set
outright) loses only the human-readable "why" the producers would otherwise write to `stderr`.

```bash
make BACKEND=ne10 NO_STDIO=1 libaudio_pipeline_dfn.a
make BACKEND=ne10 audit-no-stdio   # PASS/FAIL, non-zero exit on FAIL
make BACKEND=ne10 audit-no-stdio-stack
```

`audit-no-stdio` (in `pipelines/`) builds the DFN2 stage archive with `NO_STDIO=1` and asserts with
`nm` that it holds the stage, bridge and DFN2 pre/post members and none of
`fprintf`/`printf`/`puts`/`fputs`/`stderr`/`__stderrp` (pattern/style follows
`audiolib/common/scripts/audit_alloc_symbols.sh`); the same target in
`4ch_aec_bf_nr_res/` (run with `NO_STDIO=1`) scans the 4-channel archive, which includes the unit, the DFN archive and the producers. `audit-no-stdio-stack` also builds
AEC, NR and audiolib with `NO_STDIO=1`, scans each archive and links
`mono_aec_nr_res/tests/test_no_stdio_stack.c` (query -> init -> one hop -> reset -> destroy over
`Aec`, the denoiser and the synthesis, chained as the examples do) to prove the linked image has no stdio symbol. Run them after touching
any diagnostic.

`NO_STDIO` participates in `../pipelines/Makefile`'s `CFG_SIG`/`OBJ_DIR` so a `NO_STDIO=1` build
never shares an object directory with a default build. The example CLIs
(`aec_nr_pipeline`/`aec_nr_pipeline_static` — host tools that always do their own WAV-I/O stdio)
and the tests keep stdio regardless. `make BACKEND=<b> NO_STDIO=1` (without a specific target) still
builds them normally, linked against `NO_STDIO=1` producers — a supported combination: the library
half stays silent on rejection while the CLI half keeps its own WAV-path error messages.

A board build linking `NO_STDIO=1` producers must rely entirely on return values
(`NULL`/`-1`, `0` for a size query) — there is no error-callback or status-code-detail mechanism
beyond what each function already returns.

## Teardown order

`audio_pipeline_4ch_destroy()` tears down BF -> SRP-PHAT -> core — BF borrows the
steering table, so it goes first. For the mono flow destroy the denoiser and then the
`Aec`. None of these calls frees a caller-owned pool, and they are repeatable (matches the "Two
Versions" section in [`../pipelines/README.md`](../pipelines/README.md) — NE10's twiddle configs live
in the pool under vendored patch P0001); the order is kept anyway — a future backend/module may hold
something outside the pool that a destroy call needs to release, and it is exactly what the
heap-convenience path needs for real (`free()` on the pool `audio_pipeline_4ch_create()`
allocated).

## Board-side verification checklist

Before trusting a board integration in production, verify
each of the following on-target. Most of these are properties of the
**caller's** allocator/integration code, not of the libraries themselves — a library only checks what it
can observe from inside a single call
(alignment of the pointer it was handed, the `bytes` count it was told,
the `expected` descriptor if one was passed); it has no visibility into the
platform allocator, DMA engine, or power state behind that pointer.

- **16-byte alignment**: every pool pointer is aligned exactly as
  the size query reports (16 today). `audio_pipeline_4ch_init_ex()` and `aec_init()` reject a
  misaligned `mem` argument, but only if one is actually passed in
  misaligned — the platform allocator itself must honor the alignment it
  was asked for.
- **Exact-bytes accounting**: the block handed to each init
  really is at least the queried bytes USABLE — verify the allocator's own
  bookkeeping (e.g. an internal header carved out of the block it hands
  back) doesn't silently shrink the usable region below what was requested.
- **Region non-overlap**: the pools are not aliased with any other
  allocation, DMA buffer, or another instance's pool — every
  sub-module pointer inside a pool is a raw pointer into it, not a copy,
  so an overlapping region is silent memory corruption, not a caught error.
  The NR/synthesis block and the unit's block (or `Aec` block) must not overlap each other.
- **Exclusive lifetime**: nothing else reads or writes a pool for the
  entire lifetime of its handle, from `init` through
  the matching `destroy` — no other task, ISR, or DMA
  transfer touches it concurrently.
- **Cache coherence / DMA ownership**: if `mic`/`ref`/`out` (or the pool
  itself, on a non-cache-coherent platform) cross a DMA boundary, the
  board's own cache-maintenance operations (clean/invalidate) run at the
  correct points. These libraries have no notion of cache lines or DMA and
  perform none of their own — the per-hop calls assume `mic`/`ref`
  are already CPU-visible on entry and that `out` is fully written before
  any DMA-out on it begins.
- **Allocator-failure / partial-init rollback**: confirm the caller's own
  rollback path is correct for a `platform_alloc()` failure BEFORE any
  init is ever called, and for an init that fails after another succeeded (the AEC built, the denoiser
  refused: destroy the AEC and free the pool) — each init never partially
  commits (every rejection path returns `NULL` before writing into `mem`),
  so there is nothing on ITS side to roll back, but the caller's own
  pool-acquisition failure handling is not something a library can verify.
- **Reset / reconfigure / double-destroy / power-resume**: exercise
  the paired reset (`aec_reset()`/`audio_pipeline_4ch_reset()` with `aec_res_reset()` (4ch), `anr_reset()`, `ac_istft_reset()` and any
  neural stage reset — echo-path change), a full destroy-then-
  re-`init` on the SAME pool (config change), calling
  `destroy` twice in a row (must be a safe no-op the
  second time — see "Teardown order" above), and a power-suspend/resume
  cycle if the target platform has one (these libraries have no notion of
  power state themselves; confirm the pool's contents survive whatever the
  platform does across suspend, or re-`init` from scratch on resume
  if they don't).
- **Allocator-hook trace of init→destroy (zero-heap)**:
  with a runtime allocator-hook/interposer (the style `lib/aec`'s and
  `audiolib`'s own zero-heap tests use), confirm NO `malloc`/`calloc`/
  `realloc`/`free` call happens between the last `init` returning and the
  first `destroy` returning, ON YOUR
  ACTUAL TARGET BUILD — NE10 is zero-heap end-to-end
  (vendored patch P0001; see "Two Versions" in
  [`../pipelines/README.md`](../pipelines/README.md)), but that is a
  property of THIS repo's reference builds, not a substitute for verifying
  your own board's build.
  **Include the logging path in this trace**: on the default (stdio-
  enabled) producer builds, a rejected init only
  ever touches `stderr` (no heap symbol involved), so it does not itself
  break a zero-heap trace — but if the target forbids the stdio symbol set
  outright (not just heap use), build with `NO_STDIO=1` (above) so even
  that `stderr` reference is compiled out, and re-run the same audit
  against that build (`make BACKEND=<b> audit-no-stdio` is the static,
  `nm`-based version of this same check — see above).

## Cross-compile / board build

The board deliverable is built with `make publish BACKEND=ne10` and consumed
from the stable `dist/ne10/current/` path — immutable, content-addressed
release dirs where `MANIFEST.txt` is the deterministic build-config record
(flags, tool identities, per-file SHA-256; byte-verified on every
republish, and it deliberately carries no commit/date so the same release
id always has the same MANIFEST bytes) and the append-only `ATTEST/`
directory records provenance.

`ATTEST/` is **one-event-one-file**: exactly one new
`attest-<utc>-<commit>[-dirty]-<seq>.txt` per publish event, including
idempotent republishes of an already-published release. The file is
installed atomically with a write-temp + `link(2)` no-clobber step (the
kernel-level equivalent of `O_CREAT|O_EXCL`) — a same-second name collision
(same UTC second, same commit) regenerates the event id with the next
`<seq>` and retries, so an existing attestation is **never** overwritten.
Each attestation's `event_id` is embedded inside the file, and it records a
**full 40-hex git commit OID** plus a dirty flag for **this repo AND all
three producers** (audiolib, lib/aec, lib/nr) — "which exact checkout
of the whole stack published this" lives here, not in MANIFEST.

`publish` **refuses by default**, checking tracked and untracked state as
two independent dimensions, when this repo or any of the three
producers (a) has uncommitted **tracked** changes, (b) contains **any
untracked, non-ignored file** (gitignored build output is excluded by
design and is never part of this provenance), or (c) isn't a git checkout
at all (commit unknown) — `make publish` FATALs and names exactly the
offending repo(s) *and* dimension. Case (c) is refused
**unconditionally**: no combination of `ALLOW_DIRTY_PUBLISH`/
`ALLOW_UNTRACKED_PUBLISH` admits an identity-less checkout, for this repo
or any producer — without a commit OID neither the tracked diff nor the
untracked enumeration can be named for it, so there is no escape hatch to
offer. `ALLOW_DIRTY_PUBLISH=1` admits tracked changes on an
otherwise-identified checkout — pass it to publish anyway; the deviation
is then recorded in the attestation (`allow_dirty_publish=1`) together
with a `dirty_diff_sha256` per dirty repo (sha256 of that repo's `git diff
--binary HEAD`). `ALLOW_UNTRACKED_PUBLISH=1` separately admits untracked
files, recording `allow_untracked_publish=1` plus an
`untracked_tree_sha256` per repo that had one — sha256 over sorted,
FIXED-FIELD per-file records (the `hash_untracked()` helper in
`../pipelines/Makefile`) in which every variable-length value — path, symlink target,
file content — is itself hashed before being placed in the record, so two
records can never be confused by naive concatenation (a raw "path target"
join is collision-prone: `"a b"->"c"` vs `"a"->"b c"` would otherwise
encode identically). Any `stat`/`readlink`/`shasum` failure, or a path
that is neither a regular file nor a symlink (an embedded git checkout, a
fifo, ...), downgrades that entry to an unhashable record, which is
**always** a hard FATAL naming the path rather than an empty or
best-effort hash (fail-closed) — a publish never proceeds as though it
could account for a source it can't. The two flags are orthogonal —
either, both, or neither may be needed for a given publish.

This repo's own untracked check is **whitelisted by prefix**: any path
starting `AINR/GTCRN/` or `AINR/gtcrn_github/` never counts against a
publish here (neither the refusal nor the provenance hash). The three producers get **no such
whitelist**: an untracked file there is always a violation.

**Attestations are UNSIGNED**: they provide traceability, not authenticity
— anyone with filesystem access could forge one, so do not treat `ATTEST/`
as tamper-proof under an attacker model. `MANIFEST.txt` byte-verification
(performed automatically on every republish) is the actual integrity
check; `ATTEST/` is the provenance log.

`make -n|-q|-t ... publish` (dry-run / question / touch mode) is fully
zero-write — each flag takes a
different path, but none of the three ever create `dist/`, take the
publish lock, stage/attest anything, or change an artifact's mtime. This
also holds for **combined** flags (e.g. `-nt`, `-tq`, `-nqt`): `-t` is
checked **first**, so any invocation that includes it takes the touch
no-op path regardless of what else is set — recursing for `-n` first
would hand a child `make` both flags together, and GNU make then really
applies touch semantics to the child's own targets, which is exactly the
real write this ordering exists to prevent. `-n` **prints** what a real
publish would run (recurses into a print-only child, same as any other
target, when `-t` is not also set); `-q` **answers via its exit status
alone** (question mode's documented "needs updating" reply — `publish` is
phony, so it always would run — with no output and no recursion); `-t`
(alone or combined) is an **explicit no-op**: one note printed to stdout
and exit 0, with no recursion. `OBJ_ROOT=`/`BIN_ROOT=` relocate this
repo's own keyed obj/bin build trees (default: `obj/`/`bin/` under
`../pipelines/`) — like `DIST_ROOT=`, these
are isolation-test knobs for running scratch-directory builds, not part of
the build's config identity (CFG_SIG).

The `current` symlink is swapped via a rename(2)-atomic helper
(`audiolib/common/tools/atomic_symlink_swap.c`, built at publish time with the
host compiler `HOSTCC`), and `publish` takes its per-backend lock BEFORE
building anything, so concurrent publishes serialize fully. When
cross-compiling, pass
the **full toolchain, not just CC** — BACKEND=ne10 compiles one C++ TU
(NE10's generic-radix kernel), so a partial override would mix host-built
C++ objects into a cross build and then try to link ARM objects with the
host driver:

```bash
make publish BACKEND=ne10 \
     CC=aarch64-linux-gnu-gcc \
     CXX=aarch64-linux-gnu-g++ \
     AR=aarch64-linux-gnu-ar \
     RANLIB=aarch64-linux-gnu-ranlib \
     EXTRA_CFLAGS='-mcpu=cortex-a53'      # or -mcpu=cortex-a73
```

Guard rails (all four repos):

- **CC/CXX target-coherence check**: every BACKEND=ne10 build compares
  `$(CC) -dumpmachine` against `$(CXX) -dumpmachine` and hard-fails on
  mismatch (exactly the partial-toolchain mistake above). `TOOLCHAIN_CHECK=0`
  skips it (participates in the config signature).
- **`CFLAGS=`/`CXXFLAGS=`/`LDFLAGS=`/`CPPFLAGS=`/`FP_POLICY=` cannot be
  overridden on the make command line** — doing so would silently drop the
  repo-pinned flags (`-ffp-contract=off`, `-fno-math-errno`, backend defines, `NO_STDIO`); the
  build errors out and points at `EXTRA_CFLAGS`/`EXTRA_LDFLAGS`, the two
  supported hooks. `EXTRA_CFLAGS` containing `-Ofast`/`-ffast-math`/
  `-ffp-contract=` is likewise rejected by the FP-policy conflict gate.
- **C++ runtime comes from the C++ driver** (`libstdc++` on GNU/Linux gcc,
  `libc++` on macOS/clang) — there is no hardcoded `-lc++`, so GNU
  toolchains link cleanly.

## Compile flags (`EXTRA_CFLAGS`)

`../pipelines/Makefile` passes `EXTRA_CFLAGS` through to the `lib/aec` and
`lib/nr` sub-builds *and* this pipeline's own compile, so one invocation
reaches every `.o`. There are no optional performance flags.

No manual `make clean-libs` is needed when switching `EXTRA_CFLAGS` (or
`BACKEND`/`WERROR`): objects are flag-keyed — each distinct combination
of `BACKEND`/`CFLAGS`/`EXTRA_CFLAGS`/`WERROR` lands in its own hash-named
object directory (`obj/<backend>-<config-hash>/`, both under
`../pipelines/` and independently in `lib/aec`'s and `lib/nr`'s own
Makefiles), so a flag change
always compiles fresh objects into a fresh directory instead of reusing a
stale `.o`. Two builds with different flags/backends can even coexist or run
concurrently in the same worktree without stomping each other's objects.

## Unified FP-contraction policy

`-ffp-contract=off -fno-math-errno` is a **unified policy spanning all four repos**
(`audiolib`, `lib/nr`, `lib/aec`, the `../pipelines/` Makefile): every TU
each Makefile compiles — own sources and vendored NE10 alike — builds
with both flags. The second flag guarantees that guarded AArch64 `sqrtf`
lowers to `FSQRT` without a cold errno fallback; production code does not
read math `errno`.
They are appended LAST in the `CFLAGS` assembly (after
`-DAUDIO_PIPELINE_BACKEND_STR`, `EXTRA_CFLAGS`, `WERROR`) so nothing can
override it. `EXTRA_CFLAGS` (or a
`CFLAGS=` override) containing `-Ofast`/`-ffast-math`/`-ffp-contract=<any>`
is rejected at parse time:

```
$ make EXTRA_CFLAGS=-ffast-math
Makefile:217: *** FP policy conflict: CFLAGS/EXTRA_CFLAGS contains -ffast-math; this repo pins -ffp-contract=off; remove -ffast-math from EXTRA_CFLAGS.  Stop.
```

`../../audiolib/common/scripts/audit_fp_contract.sh` is the disassembly-level
proof the flag actually bites (audiolib's and NR's TUs —
`../pipelines/`'s own TUs, apart from the ones `pipelines/scripts/audit_fp_contract.sh` lists, are CLI/glue code with no
per-sample math loops of their own). See
`audiolib/common/README.md`'s "FP-contraction policy" section for the full
cross-repo writeup.
