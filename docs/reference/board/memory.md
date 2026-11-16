# Pool 與 descriptor

## Warnings

- **`aec_create()`, `anr_create()` and the other `*_create()` entry points ALWAYS use the heap**
  (`posix_memalign` or `malloc` under the hood) — they are the desktop/prototyping convenience path (see
  "Two Versions" in [建置與診斷](../../../pipelines/reference/build.md)),
  never the board path. **A board build must never call them.**
  The pool sequence ([above](sequence.md#static-apis-and-pool-sizing), `*_get_mem_requirements`/`*_mem_size` + `*_init_ex`/`*_init` +
  `*_destroy`) is the only sequence that touches zero heap.
- **Both the heap path and the pool path are always compiled in.** The entry point you call is the
  only switch, decided at runtime; there is no compile-time flag to set.
- **Re-query every size and descriptor after ANY build, backend, or config change** —
  a firmware rebuild, or a change to any config field that affects sizing. Never cache a
  bare byte count (or a whole `AecMemReq`) across one of those
  changes and replay it into a later build's `aec_init_ex()` —
  that mismatch is exactly what the `expected` argument exists to catch
  (see "Descriptor semantics" below). This still holds with the fixed
  32-byte layout (below): "can be persisted byte-for-byte" is not the same
  claim as "is still valid after a rebuild/backend/config change": a
  persisted descriptor from a PREVIOUS build is exactly the stale-cache
  case `aec_init_ex()` exists to catch. If a descriptor must be
  persisted across a restart (same board, same binary, same CPU — see
  "Descriptor semantics" for the same-endianness scope this is limited
  to), re-derive it fresh at the next boot from THAT boot's
  `aec_get_mem_requirements()` and treat any persisted copy as
  advisory at best — never trust a persisted value as `expected` without
  also having a freshly-queried one to fall back to. The NR denoiser, the synthesis and the example programs have no
  descriptor, so re-querying the size on every build is the whole rule there.
- **The array owns its lanes.** The 4-channel `Aec` (`AecConfig.n_mics = AEC_ARRAY_MICS`) builds the four lanes, the shared
  delay estimator, the alignment ring, the fusion buffers and the residual gain stage inside its own pool; nothing is
  injected and nothing needs to be initialised first, so a used lane or estimator cannot reach it. What stays outside are the
  NR denoiser, the synthesis, the sample beamformer helper and the NN stages: separate objects with their own pools,
  which must outlive nothing but their own use (the helper borrows the array and must be destroyed before it).
- **Failure recovery is yours.** When the array, the denoiser or a stage refuses a call, or the emitted
  hop is not finite, reset the `Aec` (`aec_reset()`: for the array also the four lanes, the estimator and the gain stage; the level setting survives), the
  denoiser, the synthesis and any neural stage together
  (`aec_reset()`, `anr_reset()`, `ac_istft_reset()`, `ulcnet_stream_reset()`), and refresh the `post_reset_serial` you track.
  An `AEC_ERR_DSP` from an array has already reset the array itself. The DFN2 model stream is not reset by an AEC delay change. A model
  that keeps its own recurrent state (the DFN2 stream and rate bridge register none) follows the state generation (a reset and a refused commit advance it), see
  ["Running the model on the NPU"](sequence.md#running-the-model-on-the-npu-nn-applications).

## Descriptor semantics (`AecMemReq`)

Every field is a fixed-width integer
(`uint32_t`/`uint64_t`, never `size_t` or a pointer), and the struct is
pinned to a stable **32-byte** layout (the AEC library's `test_memory_layout` checks it at run time) — see
"Serializing the descriptor" below for what that does (and does not) buy
you. One `Aec` descriptor covers the single-channel instance and the whole 4-channel array pool alike. The DFN2 model stream and the rate bridge use the same shape with their
own version counters.

| Field | Meaning |
|-------|---------|
| `descriptor_version` | This STRUCT's own ABI version — `AEC_DESCRIPTOR_VERSION` (currently `1`). Bumped only when `AecMemReq`'s field set/order/width changes, independent of `layout_version` below. `aec_init_ex()` checks this FIRST, before interpreting any other field. |
| `layout_version` | `AEC_LAYOUT_VERSION` (currently `4`). Bumped whenever the pool size, carve order or descriptor content of the `Aec` pool (`AEC/c_impl/src/aec.c`, `aec_array.c`) changes. A cached `bytes` alone is only a lower bound, so this counter is what refuses a descriptor persisted by another build. |
| `backend_id` | Compile-time FFT backend identity this build was made with, as a small integer — `AEC_BACKEND_NE10` (2, `aec.h`), the only FFT backend (matches the `BACKEND=` of `../pipelines/4ch/Makefile`). A process-local rodata pointer cannot be serialized safely, so `backend_id` is compared with a plain integer `==`. A descriptor carrying any other value is never valid, even at matching `bytes`. `1` is never returned. |
| `build_flags_hash` | FNV-1a-32 of a small fixed set of compile-time strings that affect the carve STRUCTURE of the pool. **Covers:** a change to the library's own carve order/buffer set/alignment. **Does NOT cover:** config VALUES (`sample_rate`, `residual`, `delay_mode`, ...) — those change `bytes` but are config, not layout, so a caller re-querying `aec_get_mem_requirements()` for its actual config already gets the right `bytes` regardless of this hash; the NR denoiser's, the synthesis' and the beamformer helper's internal struct layouts (separate objects); the compiler/ABI/toolchain. |
| `alignment` | Required base alignment of the pool pointer, bytes. Always 16 today (the one alignment every module in this stack — AEC, NR, the NE10 FFT backend, `mem_align.h`'s `ALIGN16` — carves to). `uint32_t`, not `size_t`. |
| `reserved` | Always 0 and validated (`aec_init_ex()` refuses a nonzero value). Exists only so `bytes` (a `uint64_t`) lands on an 8-byte-aligned offset within the struct with no compiler-inserted padding — part of the fixed 32-byte layout, not a field to read or write. |
| `bytes` | Size of the `Aec` pool to allocate. For the 4-channel array: the `Aec` struct, the control block, the mic lane, the aligned far, the fusion buffers and the gain stage (by `residual`), the alignment ring in MATCHED/FIXED, the shared estimator in MATCHED, and the four lanes; 666,992 B for 16 kHz/256 MATCHED with `AEC_RESIDUAL_FULL`. It does NOT include the beamformer helper, NR, synthesis or any neural stage. `uint64_t`, not `size_t` — cast to `size_t` before passing to `malloc`/`posix_memalign`/`aec_init*()` on a target where the two widths differ. |

`aec_init_ex(mem, bytes, cfg, expected)` (see ["Sequence"](sequence.md#sequence-4-channel-the-aec-array-instance-plus-the-libraries-it-feeds) above)
automates exactly this comparison: passing the descriptor straight back in
as `expected` makes the library itself recompute the CURRENT
`descriptor_version`/`layout_version`/`backend_id`/`build_flags_hash`/
`alignment` and reject (`NULL`) unless every one still agrees with `expected`, `reserved` is 0 and `expected->bytes` is not below the current `bytes`, so a board
integrator does not hand-write this comparison. The one discipline
this does NOT relieve you of: `expected` must itself have been queried
freshly (see "Warnings" above) — `aec_init_ex()` can only compare
what it's given against what the CURRENT build computes; it has no way to
know whether the `expected` you passed in was itself stale relative to some
THIRD, even older, build.

## Serializing the descriptor

`AecMemReq` is a fixed-width, 32-byte POD (pinned by the AEC
library's `test_memory_layout`: sizeof and the offset of `bytes`) — it
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
  descriptor as `expected` to `aec_init_ex()`; treat a persisted
  copy as informational at best (e.g. a board bring-up log recording what
  the PREVIOUS boot's build looked like).

## Dirty-pool contract

None of the init functions requires a zero-filled pool. `aec_init_ex()` clears the whole block it is
given, and NR/the FFT backend each zero
their own sub-region during their own `_init()` — so a pool filled with poison bytes inits and
processes identically to a freshly-zeroed one.
`../pipelines/4ch/tests/test_4ch_chain.c` (the array chain, `0xA5`-poisoned caller pool against the heap flavor; the library's own array tests are `test_aec_array*.c` in the AEC repo) and
`../pipelines/mono/aec_nr_res/tests/test_mono_chain.c` (mono, pool carved from a `0xA5`-poisoned
block against the heap flavor) exercise exactly this: byte-for-byte the same output. There is no
need for a caller-side blanket `memset(pool, 0, bytes)` before `init`.

## `NO_STDIO=1` — building without libc stdio

The array instance prints nothing at all on any path (no stdio symbol is linked), and the example
header `array_sample_post.h` uses none. lib/aec, lib/nr and audiolib compile their init-time diagnostics out under
`NO_STDIO=1`; the DFN2 model stream and rate bridge archive is audited for stdio symbols. Every failure is
signalled through the return value (`NULL`/`-1`/`0` size), so a board image that cannot or will not
link libc's stdio (no console, or a policy that forbids the stdio symbol set
outright) loses only the human-readable "why" the producers would otherwise write to `stderr`.

```bash
make BACKEND=ne10 NO_STDIO=1 libaudio_pipeline_dfn.a
make BACKEND=ne10 audit-no-stdio   # PASS/FAIL, non-zero exit on FAIL
make BACKEND=ne10 audit-no-stdio-stack
```

`audit-no-stdio` (in `pipelines/`) builds the DFN2 archive with `NO_STDIO=1` and asserts with
`nm` that it holds the model stream, bridge and DFN2 pre/post members and none of
`fprintf`/`printf`/`puts`/`fputs`/`stderr`/`__stderrp` (pattern/style follows
`audiolib/common/scripts/audit_alloc_symbols.sh`); the same target in
`4ch/` (run with `NO_STDIO=1`) scans the 4-channel build: the DFN archive, the producers (including the spatial archives) and the sample object `array_sample_beam.o`. `audit-no-stdio-stack` also builds
AEC, NR and audiolib with `NO_STDIO=1`, scans each archive and links
`mono/aec_nr_res/tests/test_no_stdio_stack.c` (query -> init -> one hop -> reset -> destroy over
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

For the 4-channel flow destroy in reverse order of construction: the sample helper
(`array_sample_beam_destroy()`: BF -> SRP-PHAT, BF borrows the steering table so it goes first), then
`aec_destroy()` of the array (the lanes and the estimator are inside its pool and need no call of their own). On a caller pool `aec_destroy()` is idempotent and frees nothing; on an `aec_create()` handle it frees the block and is called once. For the mono flow destroy the denoiser and then the
`Aec`. None of these calls frees a caller-owned pool, and on a caller pool they are repeatable (matches the "Two
Versions" section in [建置與診斷](../../../pipelines/reference/build.md) — NE10's twiddle configs live
in the pool under vendored patch P0001); the order is kept anyway — a future backend/module may hold
something outside the pool that a destroy call needs to release, and it is exactly what the
heap-convenience path needs for real (the block `aec_create()`
allocated, freed by `aec_destroy()`; the `Aec` pointer is dead afterwards and must not be `free()`d).

[返回本主題](README.md)
