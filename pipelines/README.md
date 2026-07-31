# C Pipeline: Linear AEC → NR → RES

## Architecture

```
mic ─┐                       ┌─ aec_out ──┐              ┌─ nr_out ──┐                  ┌─ output
     ├→ AEC (linear) ────────┤            ├→ NR (MMSE) ──┤           ├→ RES (post) ─────┤
ref ─┘   PBFDKF+Shadow      └─ context   ┘  LSA+MCRA    └─ gain[]   ┘  echo×nr_gain    └─ final
```

## Modules

| Module | Library | Header | Function |
|--------|---------|--------|----------|
| AEC | libaec.a | aec.h | PBFDKF adaptive filter + shadow filter |
| NR | libmmse_lsa.a | mmse_lsa_denoiser.h | MMSE-LSA + MCRA noise est + SPP |
| RES | libaec.a (included) | aec.h (`AecResContext`) | Residual echo suppression, folded into AEC's freq-domain seam |

RES is not a standalone module/library — it is exposed as the `AecResContext` seam on
the AEC object. With `AecConfig.return_res_context=1` and `enable_res=0`, `aec_process()`
(or the streaming `aec_analyze_render()` / `aec_process_capture()` pair) still computes the
AEC3 post-filter's residual-echo suppression internals but does not apply them to the time
output; `aec_get_res_context(a, &ctx)` then exposes `AecResContext` — `echo_spec`, `error_spec`,
`res_gain` (G_res(f)), `r2` (residual-echo PSD), `comfort_noise`, etc. — so an external caller
can run AEC(linear) → NR → RES itself. See `lib/aec/c_impl/include/aec.h` (`AecResContext`,
`aec_get_res_context()`) for the full field list.

## Parameter Alignment

All modules use unified 20ms frame / 10ms hop, auto-configured by sample rate:

| Parameter | 8 kHz | 16 kHz | 48 kHz | Formula |
|-----------|-------|--------|--------|---------|
| frame_size | 160 | 320 | 960 | sr × 20ms |
| hop_size | 80 | 160 | 480 | frame / 2 |
| fft_size | 256 | 512 | 1024 | next pow2 ≥ frame |
| n_freqs | 129 | 257 | 513 | fft/2 + 1 |
| filter_length | 416 | 832 | 3072 | ms-derived: sr × 52ms (64ms ≥44.1 kHz) |
| n_partitions | 6 | 6 | 7 | ceil(filter_length / hop) |

## Latency & Performance

| 項目 | 數值 | 說明 |
|------|------|------|
| **Algorithmic latency** | 10 ms | 1 hop（所有 sample rate 一致） |
| **NR OLA delay** | +10 ms | NR frame 處理引入額外 1 hop 延遲 |
| **Pipeline total latency** | **20 ms** | AEC hop + NR OLA delay |
| **Processing (per hop)** | < 0.5 ms | AEC + NR + RES 合計（ARM Cortex-A53 @ 1GHz 估計） |
| **RTF** | < 0.05 | 遠低於即時要求 |

### Memory Budget

Measured figures from `./aec_nr_pipeline_static --print-mem-size --sample-rate 16000`
(balanced presets). The AEC row is the composite `aec_get_mem_size()` pool — it
already contains HPF, PBFDKF ×2 (main+shadow), delay estimator, the RES/post
context and the AEC-internal FFTs. Since NE10 vendored patch P0001 the NE10
twiddle configs are carved from these pools too, so both columns are the
complete memory requirement (strict init→destroy zero-heap on both backends):

Pipeline bufs is now 12 scratch buffers, not 13 — the `g_aec` buffer (a
per-hop memcpy'd duplicate of `AecResContext.res_gain`) was removed; both
its former readers now read `ctx.res_gain` directly (stable for the whole
hop per `aec.h`'s own doc), shrinking each row's `Pipeline bufs`/`Total` by
`ALIGN16(n_freqs*4)` B — 528 B @ 8 kHz, 1,040 B @ 16 kHz, 2,064 B @ 48 kHz
(`AUDIO_PIPELINE_LAYOUT_VERSION` bumped 1→2 accordingly):

| Rate / Backend | AEC | FFT (OLA) | NR | Pipeline bufs | **Total** |
|--------|-----|-----------|-----|---------------|-----------|
| **8 kHz KISS** | 290,672 B | 8,784 B | 97,792 B | 6,736 B | **404,176 B (394.7 KB)** |
| **8 kHz NE10** | 288,848 B | 8,176 B | 97,792 B | 6,736 B | **401,744 B (392.3 KB)** |
| **16 kHz KISS** | 538,320 B | 16,976 B | 194,048 B | 13,392 B | **762,928 B (745.0 KB)** |
| **16 kHz NE10** | 534,192 B | 15,600 B | 194,048 B | 13,392 B | **757,424 B (739.7 KB)** |
| **48 kHz KISS** | 1,253,680 B | 33,360 B | 386,560 B | 31,824 B | **1,705,616 B (1,665.6 KB)** |
| **48 kHz NE10** | 1,244,944 B | 30,448 B | 386,560 B | 31,824 B | **1,693,968 B (1,654.3 KB)** |

> filter_length 是 ms-derived（52 ms；≥44.1 kHz 用 64 ms → 48 kHz 為 3072 taps、
> 7 partitions），加長會等比增加 AEC 記憶體；記憶體吃緊時先縮 `filter_length`
> 與 NR 的 `L`（48 kHz 也可用 `n_partitions` override 換較短尾巴）。
> 三個 rate 都由同一 hop=10 ms 規則自動推導（`pipeline_dims.h`），並在 init 以
> grid assert 驗證 pipeline/AEC/FFT/NR 四方一致。

## Integration Flow

1. **AEC (linear)**: Set `enable_res=0` and `return_res_context=1`, call `aec_process()` (or the
   streaming `aec_analyze_render()` / `aec_process_capture()` pair), then `aec_get_res_context(a, &ctx)`
   to read the `AecResContext` seam
2. **NR**: `mmse_lsa_process()` for denoising, `mmse_lsa_get_gain()` for per-bin gain
3. **RES**: Correct echo PSD with `echo_spec *= nr_gain`, then apply `ctx.res_gain` (AEC3
   `SuppressionGain` G_res(f)) to the NR output — there is no separate `res_process()` call

### Echo PSD Correction

```c
const float* gain = mmse_lsa_get_gain(nr, NULL);
for (int k = 0; k < n_freqs; k++) {
    corrected_echo[k].r = ctx.echo_spec[k].r * gain[k];
    corrected_echo[k].i = ctx.echo_spec[k].i * gain[k];
}
/* apply ctx.res_gain[k] (G_res(f)) to nr_out[k] to get the final RES-suppressed output */
```

NR already attenuated certain frequency bins. The echo PSD estimate must
reflect this, otherwise RES will over-suppress (seeing echo that NR already
removed). Multiplying by the NR gain corrects for this.

## NR OLA Delay

NR uses OLA (frame_size=320, hop=160), introducing 1-frame (10ms) delay.
The pipeline saves the previous AEC context and uses it when the
corresponding NR output becomes available.

## Build

```bash
# From Audio_ALG/pipelines/ — builds the submodule libs + BOTH binaries
make                # libs (BACKEND=kiss) + aec_nr_pipeline + aec_nr_pipeline_static

# Binaries land in a config-keyed dir (round-3 review B01):
#   bin/<backend>-<config-hash>/  — resolve it with `make print-bin-dir`
# (same flags as your build), or use `make publish` for the stable
# dist/<backend>/current/ handoff path.
BIN="$(make -s print-bin-dir)"

# Run Version A (malloc)
"$BIN"/aec_nr_pipeline mic.wav ref.wav output.wav balanced
"$BIN"/aec_nr_pipeline mic.wav ref.wav output.wav --aec-only
"$BIN"/aec_nr_pipeline mic.wav ref.wav output.wav aggressive --nr-preset aggressive

# Run Version B (static memory) — same CLI, plus a mem-size query mode
"$BIN"/aec_nr_pipeline_static mic.wav ref.wav output.wav balanced
"$BIN"/aec_nr_pipeline_static --print-mem-size --sample-rate 16000

# Run the audio_pipeline.h library's own acceptance tests (F20/R08/R09/B06/§7.3) —
# create-vs-init byte equality (incl. a poisoned pool), destroy idempotence,
# misaligned/undersized pool rejection, sample-rate whitelist rejection,
# AudioPipelineConfig reject-first validation (bad enum/bool fields),
# audio_pipeline_init_ex()'s `expected` descriptor gate (tampered
# descriptor_version/layout_version/backend_id/build_flags_hash/alignment/
# bytes each rejected) — each per-rate case runs once per supported rate
# (8000/16000/48000; 48 kHz uses a reduced hop count, see test_audio_pipeline.c)
# — AND builds + runs the example_board_adapter smoke test (see "Board
# Integration" below)
make test

# Build + run JUST the REFERENCE ONLY board-adapter example standalone
# (also runs as part of `make test` above):
make example-adapter

# Build libaudio_pipeline.a with no stdio linked in at all (board images that
# forbid the stdio symbol set), and audit that it holds:
make NO_STDIO=1 libaudio_pipeline.a
make audit-no-stdio
```

`make` also builds `libaudio_pipeline.a` (the linkable pool-sizing/carving/
processing library both CLIs above are now thin shells over) as a side
effect of building either binary. See "Board Integration" below for the API
this exposes to a firmware/board consumer, including the `NO_STDIO=1` build
knob and `audit-no-stdio` target above.

## Debugging & Performance Flags

Both `aec_nr_pipeline` and `aec_nr_pipeline_static` support the same debug CLI
option (mirrored, byte-for-byte identical wiring in both binaries). There are
no optional performance compile flags — the fast matched-filter arithmetic and
delay-estimator duty-cycling are built into `lib/aec` unconditionally.

### `--debug`

Once per second of processed audio, prints one compact status line to stderr
combining `aec_debug_status()` (lib/aec) and `mmse_lsa_debug_status()` (lib/nr) —
both are read-only snapshots of state the engines already maintain, so this adds
no per-frame cost when the flag is off and doesn't perturb the DSP output when on
(stdout/the output WAV are unaffected either way).

```
./aec_nr_pipeline mic.wav ref.wav out.wav --debug
[dbg   1.0s] aec: delay=-1 conf=0.5 upd=6 erle=0.0dB lin=0 conv=0 near=8.74e-04 out=8.08e-04 | nr: init=1 gain=-18.2/-23.4dB spp=0.50 noise=-1.2dB
[dbg   2.0s] aec: delay=320 conf=1.0 upd=18 erle=0.0dB lin=1 conv=0 near=4.79e-03 out=3.47e-03 | nr: init=1 gain=-18.8/-23.9dB spp=0.51 noise=-4.8dB
...
```

`aec:` fields are `AecDebugStatus` (delay in samples, `-1` = not yet acquired;
`conf`/`upd` = delay-estimator confidence/update count; `erle` = windowed ERLE dB;
`lin`/`conv` = usable-linear-estimate / filter-converged gates; `near`/`out` = EMA
power). `nr:` fields are `MmseLsaDebugStatus` (`init` = noise-floor initialized;
`gain` = mean/min linear gain dB; `spp` = mean speech-presence probability;
`noise` = mean noise-floor dB). With `--aec-only` the `nr:` half prints `n/a`
(no denoiser exists in that mode).

> **Caveat**: this pipeline always runs AEC in **linear mode**
> (`enable_res=0`, `return_res_context=1` — the external NR/RES seam), and
> `lib/aec/c_impl/src/aec.c` only caches `last_erle_windowed` when
> `cfg.enable_res` is true. So `erle=` in this pipeline's `--debug` output
> always reads `0.0dB` — that's expected here, not a broken query. (The field
> does move if you drive `aec_debug_status()` from a caller running with
> `enable_res=1`, e.g. `lib/aec/c_impl/example/aec_wav.c --debug`.)

### Delay-estimator duty-cycling (built in, always on)

The AEC3 matched-filter delay estimator duty-cycles itself — no flag or
config field: once the delay estimate is solid (confidence 1.0) and unchanged
for `delay_est_period_s` (default 0.5s), analysis drops to 1 hop in every K
(K=10 by default) instead of every hop — full-rate analysis resumes
immediately if the estimate changes, loses solidity, or ERLE drops >6dB off
its running peak. **Sampled-quality-verified ~zero cost** (60-case AECMOS:
≤+0.014 / worst −0.006). On a stable-delay clip the decimated schedule never
actually skips a *different* outcome; verified here on
`wav/aec_challenge_blind/doubletalk/0I0XMl3M0ECO0U1N0cJvpg_*`.

Note: in THIS pipeline the ERLE-watchdog resume leg is inert — the AEC runs
linear-mode (`enable_res=0`) so `last_erle_windowed` is never updated (same
root cause as the `--debug` `erle=0.0` caveat above). Full-rate analysis
still resumes on estimate change or lost solidity, which are the primary
resume paths.

### Compile flags (`EXTRA_CFLAGS`)

`pipelines/Makefile` passes `EXTRA_CFLAGS` through to the `lib/aec` and
`lib/nr` sub-builds *and* this pipeline's own compile, so one invocation
reaches every `.o`. The fast matched-filter arithmetic and delay-estimator
duty-cycling are built in unconditionally — there are no optional
performance flags at present.

No manual `make clean-libs` is needed when switching `EXTRA_CFLAGS` (or
`BACKEND`/`WERROR`): objects are now flag-keyed — each distinct combination
of `BACKEND`/`CFLAGS`/`EXTRA_CFLAGS`/`WERROR` lands in its own hash-named
object directory (`obj/<backend>-<config-hash>/`, both here and
independently in `lib/aec`'s and `lib/nr`'s own Makefiles), so a flag change
always compiles fresh objects into a fresh directory instead of reusing a
stale `.o`. Two builds with different flags/backends can even coexist or run
concurrently in the same worktree without stomping each other's objects.

### Unified FP-contraction policy (round-3 review B04)

`-ffp-contract=off` is a **unified policy spanning all four repos**
(`audio_common`, `lib/nr`, `lib/aec`, this `pipelines/` Makefile): every TU
each Makefile compiles — own sources and vendored KISS/NE10 alike — builds
with the flag, appended LAST in the `CFLAGS`/`LIB_CFLAGS` assembly (after
`-DAUDIO_PIPELINE_BACKEND_STR`, `EXTRA_CFLAGS`, `WERROR`) so nothing can
override it — this Makefile used to carry the flag as the *fourth* token of
the base `CFLAGS` assignment (before `EXTRA_CFLAGS` was folded in), which
this review moved to its current trailing position. `EXTRA_CFLAGS` (or a
`CFLAGS=` override) containing `-Ofast`/`-ffast-math`/`-ffp-contract=<any>`
is rejected at parse time:

```
$ make EXTRA_CFLAGS=-ffast-math
Makefile:217: *** FP policy conflict: CFLAGS/EXTRA_CFLAGS contains -ffast-math; this repo pins -ffp-contract=off; remove -ffast-math from EXTRA_CFLAGS.  Stop.
```

`../../audio_common/scripts/audit_fp_contract.sh` is the disassembly-level
proof the flag actually bites (audio_common's and NR's TUs — this
directory's own three TUs, being pure CLI/glue code with no per-sample math
loops of their own, are outside that script's audit list). See
`audio_common/README.md`'s "FP-contraction policy" section for the full
cross-repo writeup.

## Two Versions

### Version A: malloc (`aec_nr_pipeline.c`)
Each module uses `_create()` / `_destroy()` and manages its own memory internally.
Suitable for desktop testing and Linux servers.

### Version B: static memory (`aec_nr_pipeline_static.c`)

Built by default alongside Version A (both `lib/aec` and `lib/nr` track
`main` — each library ships the heap and static APIs side by side,
selected at runtime). One
caller-owned pool, no malloc after init, byte-identical output to Version A
(see Verification below). Since NE10 vendored patch P0001
(`audio_common/lib/ne10/VENDORED.md`), the NE10 backend's three R2C/C2R
twiddle configs are carved FROM the caller pool too
(`ne10_fft_init_r2c_float32_ext`: carve + twiddle-generate directly over
caller-supplied memory, no `malloc()` involved) and are already counted in
the `*_get_mem_size` figures — so "no malloc after init" is zero heap
allocation ever, end to end, on **both** backends, not just on the per-hop
audio path. (Pre-P0001 the configs were NE10-internal heap allocations
*outside* the pool, allocated during `aec_init`/`fft_init` and not counted
in `*_get_mem_size`; that exception is gone — see the `destroy()` note in
the code block below.)

The pipeline uses exactly THREE composite static APIs — there are no
per-submodule `_get_mem_size()` entry points; each library slices its own
internals (HPF, PBFDKF ×2, delay estimator, RES/post context, internal FFTs
for AEC; MCRA + SPP for NR) inside its single pool segment:

```c
size_t aec_sz = aec_get_mem_size(&aec_cfg);          /* lib/aec           */
size_t nr_sz  = mmse_lsa_get_mem_size(&nr_cfg);      /* lib/nr            */
size_t fft_sz = fft_get_mem_size(fft_size);          /* audio_common (OLA) */
/* + pipeline buffers; every segment 16-byte aligned (ALIGN16)            */

Aec*             aec = aec_init(mem_aec, aec_sz, &aec_cfg);
MmseLsaDenoiser* nr  = mmse_lsa_init(mem_nr, nr_sz, &nr_cfg);
FftHandle*       fft = fft_init(mem_fft, fft_sz, fft_size);
/* destroy() on static instances frees no pool memory (runtime is_static)
 * and, since P0001, releases no backend-owned handle OUTSIDE the pool
 * either: on BOTH backends every one of aec_destroy/mmse_lsa_destroy/
 * fft_destroy above is a genuine no-op for a pool-resident instance --
 * nothing lives outside mem_aec/mem_nr/mem_fft to leak or double-free, so
 * each is safe to call any number of times (including never, if the caller
 * is about to free/reuse the whole pool anyway). (Pre-P0001, NE10's
 * destroy calls were NOT idempotent/skippable this way: skipping one
 * leaked its twiddle config, calling it twice double-freed it.) */
```

Query the exact pool budget for any configuration without running audio:

```bash
./aec_nr_pipeline_static --print-mem-size --sample-rate 16000
```

**Embedded-target integration:** allocate one contiguous, 16-byte-aligned
block of the reported total from the platform allocator, pass it in place of
the desktop `malloc` — no other change. The pool base MUST be 16-byte
aligned (both libraries assert this).

## Board Integration

Review F20: the pool-sizing/carving/per-hop-processing logic both CLIs above
embed is also available as a standalone, linkable library —
[`audio_pipeline.h`](audio_pipeline.h) / [`audio_pipeline.c`](audio_pipeline.c),
built into `libaudio_pipeline.a`. A board's own memory manager consumes this
directly instead of copying `aec_nr_pipeline_static.c`'s file-local carve
code into firmware. Both CLIs are now thin shells over it (arg parsing + WAV
I/O + the `--print-mem-size`/`--debug`/`DUMP_CTX` diagnostics) — see
`aec_nr_pipeline_static.c` for the caller-pool flavor of the sequence below,
or `aec_nr_pipeline.c` for the heap-convenience flavor
(`audio_pipeline_create()`).

**Reference example** (round-3 review B02/R04):
[`example_board_adapter.c`](example_board_adapter.c) is a compilable,
runnable HOST SIMULATION of the sequence below — a `board_mem` module
standing in for a platform memory manager, the query→alloc→init_ex→
process→reset→process→destroy→free flow (run twice on the same static
arena, proving pool reusability), and the negative demonstrations
(undersized/misaligned pool, tampered descriptor, double-destroy, pool
usable after a rejected init). Build + run it with `make example-adapter`
(also wired into `make test`). **This example does NOT replace the real
board adapter review** — the actual platform adapter source, memory-manager
implementation, build command, and final link map for the real target must
still be authored and submitted for sign-off; every `board_mem_*` function
in that file is a plain host-array stand-in, marked `// BOARD:` wherever
real platform code belongs instead.

### Cross-compile / board build (round-4 review P1-2)

The board deliverable is built with `make publish BACKEND=ne10` and consumed
from the stable `dist/ne10/current/` path — immutable, content-addressed
release dirs where `MANIFEST.txt` is the deterministic build-config record
(flags, tool identities, per-file SHA-256; byte-verified on every
republish, and it deliberately carries no commit/date so the same release
id always has the same MANIFEST bytes) and the append-only `ATTEST/`
directory records provenance.

`ATTEST/` is **one-event-one-file** (round-6 review): exactly one new
`attest-<utc>-<commit>[-dirty]-<seq>.txt` per publish event, including
idempotent republishes of an already-published release. The file is
installed atomically with a write-temp + `link(2)` no-clobber step (the
kernel-level equivalent of `O_CREAT|O_EXCL`) — a same-second name collision
(same UTC second, same commit) regenerates the event id with the next
`<seq>` and retries, so an existing attestation is **never** overwritten.
Each attestation's `event_id` is embedded inside the file, and it records a
**full 40-hex git commit OID** plus a dirty flag for **this repo AND all
three producers** (audio_common, lib/aec, lib/nr) — "which exact checkout
of the whole stack published this" lives here, not in MANIFEST.

`publish` **refuses by default** (round-6 review, split into two
independent dimensions by round-7) when this repo or any of the three
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
FIXED-FIELD per-file records (the `hash_untracked()` helper in the
Makefile) in which every variable-length value — path, symlink target,
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
publish here (neither the refusal nor the provenance hash) — this working
tree permanently carries those two directories as user-owned content that
is never staged for a release. The three producers get **no such
whitelist**: an untracked file there is always a violation. (Splitting
tracked/untracked into two independent dimensions also fixed a round-6
asymmetry: the three producers used to fold untracked files into the
SAME dirty check as tracked changes — `git status --porcelain`, no `-uno`
— so an untracked producer file made `ALLOW_DIRTY_PUBLISH=1` the only
override, with no separate provenance trail for it. Round-7 gives every
one of the four repos the identical tracked-only `-uno` check plus its own
untracked check, so the two kinds of deviation are never conflated for
anyone, and only this repo's untracked dimension carries the whitelist.)

**Attestations are UNSIGNED**: they provide traceability, not authenticity
— anyone with filesystem access could forge one, so do not treat `ATTEST/`
as tamper-proof under an attacker model. `MANIFEST.txt` byte-verification
(performed automatically on every republish) is the actual integrity
check; `ATTEST/` is the provenance log.

`make -n|-q|-t ... publish` (dry-run / question / touch mode) is fully
zero-write (round-6 review, `-t` tightened by round-7) — each flag takes a
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
and exit 0, with no recursion (recursing here used to let plain `touch`
semantics bump this repo's own build-artifact mtimes — a real write —
before round-7 special-cased it). `OBJ_ROOT=`/`BIN_ROOT=` relocate this
repo's own keyed obj/bin build trees (default: `obj/`/`bin/` here,
byte-identical to the previous hardcoded paths) — like `DIST_ROOT=`, these
are isolation-test knobs for running scratch-directory builds, not part of
the build's config identity (CFG_SIG).

The `current` symlink is swapped via a rename(2)-atomic helper
(`audio_common/tools/atomic_symlink_swap.c`, built at publish time with the
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

Guard rails (all four repos, round-4 review):

- **CC/CXX target-coherence check**: every BACKEND=ne10 build compares
  `$(CC) -dumpmachine` against `$(CXX) -dumpmachine` and hard-fails on
  mismatch (exactly the partial-toolchain mistake above). `TOOLCHAIN_CHECK=0`
  skips it (participates in the config signature).
- **`CFLAGS=`/`CXXFLAGS=`/`LDFLAGS=`/`CPPFLAGS=`/`FP_POLICY=` cannot be
  overridden on the make command line** — doing so would silently drop the
  repo-pinned flags (`-ffp-contract=off`, backend defines, `NO_STDIO`); the
  build errors out and points at `EXTRA_CFLAGS`/`EXTRA_LDFLAGS`, the two
  supported hooks. `EXTRA_CFLAGS` containing `-Ofast`/`-ffast-math`/
  `-ffp-contract=` is likewise rejected (FP-policy conflict, round-3 B04).
- **C++ runtime comes from the C++ driver** (`libstdc++` on GNU/Linux gcc,
  `libc++` on macOS/clang) — there is no hardcoded `-lc++` anywhere any
  more, so GNU toolchains link cleanly.

### Sequence

```
1. query    AudioPipelineConfig cfg = audio_pipeline_default_config(sample_rate);
            cfg.aec_preset/nr_mode/aec_only/enable_cng/legacy_amin = ...;
            AudioPipelineMemReq req;
            audio_pipeline_get_mem_requirements(&cfg, &req);   // -> req.bytes/alignment/...
            // Query THIS SAME `req`, fresh, immediately before every
            // init_ex call below — never cache it (or just its `bytes`)
            // across a build/backend/config change and replay it later.
            // See "Warnings" below.

2. allocate void* pool = platform_alloc(req.bytes, req.alignment);
            // req.alignment is always 16 today; pool need NOT be zeroed —
            // see "Dirty-pool contract" below. Pool must stay STABLE and
            // EXCLUSIVE (nothing else reads/writes it, not shared with any
            // other instance) for the entire lifetime of the handle below.

3. init     AudioPipeline* p = audio_pipeline_init_ex(pool, req.bytes, &cfg, &req);
            // Passing `req` back in as `expected` is what makes this call
            // reject a STALE pool/descriptor instead of silently carving
            // into one sized/shaped for a different build — see
            // "Descriptor semantics" below for the seven-condition check
            // this performs. NULL on: any expected-descriptor mismatch
            // (descriptor_version / layout_version / backend_id /
            // build_flags_hash / alignment / bytes), a misaligned/undersized
            // pool, an invalid cfg, or a sub-module init/grid-agreement
            // failure (stderr has detail, UNLESS this library was built with
            // NO_STDIO=1 — see "NO_STDIO=1" below). `audio_pipeline_init(pool,
            // req.bytes, &cfg)` remains available and is EXACTLY
            // `audio_pipeline_init_ex(pool, req.bytes, &cfg, NULL)` — it
            // skips the descriptor check entirely (nothing to compare
            // against); prefer `_ex` whenever a descriptor was already
            // queried, which the board flow above always has at hand.

4. process  float mic[hop], ref[hop], out[hop];   // hop = audio_pipeline_hop_size(p)
            while (have_audio()) {
                read_hop(mic, ref, hop);
                audio_pipeline_process(p, mic, ref, out);
                write_hop(out, hop);
            }

5. reset?   audio_pipeline_reset(p);   // optional: echo-path change, stream switch
            // re-zeros pipeline/AEC/NR state in place; no re-validation, no
            // pool re-touch beyond that.

6. destroy  audio_pipeline_destroy(p);
            // NR -> pipeline FFT -> AEC, reverse of the init carve order.
            // NULL-safe; idempotent for THIS pool-resident instance (every
            // sub-destroy is already a genuine no-op on the pool path, on
            // BOTH backends — see "Two Versions" above). Call it exactly
            // once if `p` came from audio_pipeline_create() instead
            // (ordinary free() semantics — see "Warnings" below).

7. release  platform_free(pool);   // only after step 6 — the pool is dead once
            //                        audio_pipeline_init_ex/destroy have run on it.
```

### Warnings

- **`audio_pipeline_create()` ALWAYS uses the heap** (`posix_memalign`
  under the hood) — it is the desktop/prototyping convenience path (see
  "Two Versions" above), never the board path. **A board build must never
  call it.** The pool sequence above
  (`audio_pipeline_get_mem_requirements` + `audio_pipeline_init_ex` +
  `audio_pipeline_destroy`) is the only sequence that touches zero heap.
- **There is no `USE_EXT_MEM` macro.** Both the heap path and the pool
  path are always compiled into `libaudio_pipeline.a` — the *entry point
  you call* is the only switch, decided at runtime by which functions your
  code calls, not by any compile-time flag. See "`USE_EXT_MEM` — not a
  thing here" below; do not look for one to set, there is nothing to
  accidentally leave unset either.
- **Re-query the descriptor after ANY build, backend, or config change** —
  a firmware rebuild, a `BACKEND=kiss` ↔ `BACKEND=ne10` switch, or a change
  to any `AudioPipelineConfig` field that affects sizing. Never cache a
  bare byte count (or a whole `AudioPipelineMemReq`) across one of those
  changes and replay it into a later build's `audio_pipeline_init_ex()` —
  that mismatch is exactly what the `expected` argument exists to catch
  (see "Descriptor semantics" below). This still holds even though V2's
  fixed 32-byte layout (below) makes the struct meaningfully
  serializable now — "can be persisted byte-for-byte" is not the same
  claim as "is still valid after a rebuild/backend/config change": a
  persisted descriptor from a PREVIOUS build is exactly the stale-cache
  case `audio_pipeline_init_ex()` exists to catch. If a descriptor must be
  persisted across a restart (same board, same binary, same CPU — see
  "Descriptor semantics" for the same-endianness scope this is limited
  to), re-derive it fresh at the next boot from THAT boot's
  `audio_pipeline_get_mem_requirements()` and treat any persisted copy as
  advisory at best — never trust a persisted value as `expected` without
  also having a freshly-queried one to fall back to.
- **Mixing `_create()` and pool APIs per submodule is not supported.**
  Don't hand-carve one of AEC/NR/the FFT backend into a pool segment
  yourself while letting `audio_pipeline_create()` heap-allocate another,
  and don't call `aec_create()`/`mmse_lsa_create()`/`fft_create()` directly
  alongside this library's own carve. `audio_pipeline_init()`/`_init_ex()`
  own the ENTIRE pool layout (control block + AEC + FFT + NR + the 12
  pipeline buffers) as one unit — there is no supported way to substitute
  a heap-obtained sub-module handle into a pool-resident `AudioPipeline`,
  or vice versa.

### Descriptor semantics (`AudioPipelineMemReq`)

Descriptor V2 (review B06): every field is a fixed-width integer
(`uint32_t`/`uint64_t`, never `size_t` or a pointer), and the struct is
pinned to a stable, `_Static_assert`-enforced **32-byte** layout — see
"Serializing the descriptor" below for what that does (and does not) buy
you. This is a **BREAKING** change from the original (F20) descriptor shape
(`{size_t bytes; size_t alignment; uint32_t layout_version; const char*
backend; uint32_t build_flags_hash;}`); every caller in this repo has been
updated, there is no compatibility shim.

| Field | Meaning |
|-------|---------|
| `descriptor_version` | This STRUCT's own ABI version — `AUDIO_PIPELINE_DESCRIPTOR_VERSION` (currently `2`). Bumped only when `AudioPipelineMemReq`'s field set/order/width changes, independent of `layout_version` below (which tracks THIS FILE's carve layout, not the descriptor struct's own shape). `audio_pipeline_init_ex()` checks this FIRST, before interpreting any other field. |
| `layout_version` | Bumped whenever `audio_pipeline.c`'s OWN carve order/buffer set/sizing formula changes — i.e. whenever a `bytes` figure computed by an older build would misdescribe a newer build's actual carve, or vice versa. Starts at 1. Does **not** need bumping for a change purely inside AEC's/NR's/an FFT backend's own internal `_get_mem_size` layout (each is consumed as one opaque composite blob here, same as the pre-F20 static CLI already treated them — a stale cached `bytes` from an old submodule build is still caught by the undersized-pool rejection at init). |
| `backend_id` | Compile-time FFT backend identity this `audio_pipeline.o` was built with, as a small integer — `AUDIO_PIPELINE_BACKEND_KISS` (1) or `AUDIO_PIPELINE_BACKEND_NE10` (2) (matches this Makefile's `BACKEND=`). Replaces the F20 `const char* backend` field — a process-local rodata pointer can't be serialized, and comparing it required `strcmp` against caller-supplied data at a trust boundary; `backend_id` is compared with a plain integer `==` instead. The two backends are still not byte-identical to each other (pre-existing, expected — see `lib/aec/CLAUDE.md`); a descriptor from one is never valid for the other even at matching `bytes`. `0` is reserved for "unknown backend" and is never present in a descriptor this library actually returns — `audio_pipeline_get_mem_requirements()` rejects an unrecognized backend string outright. |
| `build_flags_hash` | FNV-1a-32 of a small fixed set of compile-time strings that affect the pipeline's own carve STRUCTURE: the backend identity above, a literal token list naming the 12 scratch buffers in carve order, and the alignment granularity — see `audio_pipeline_build_flags_hash()` in `audio_pipeline.c`. **Covers:** a change to this file's own carve order/buffer set/alignment. **Does NOT cover:** `AudioPipelineConfig` preset/tunable VALUES (`aec_preset`, `nr_mode`, `sample_rate`, `aec_only`, ...) — those change `bytes` but are config, not layout, so a caller re-querying `get_mem_requirements()` for its actual config already gets the right `bytes` regardless of this hash; AEC's/NR's/an FFT backend's internal struct layouts (opaque blobs, as above); the compiler/ABI/toolchain. |
| `alignment` | Required base alignment of the pool pointer, bytes. Always 16 today (the one alignment every module in this stack — AEC, NR, both FFT backends, `mem_align.h`'s `ALIGN16` — carves to). `uint32_t`, not `size_t`. |
| `reserved` | Always 0. Exists only so `bytes` (a `uint64_t`) lands on an 8-byte-aligned offset within the struct with no compiler-inserted padding — part of the fixed 32-byte layout, not a field to read or write. |
| `bytes` | Total pool size to allocate (includes the opaque `AudioPipeline` control block itself, carved at the front — a few hundred bytes — plus AEC + FFT(OLA) + NR + the 12 pipeline scratch buffers, same carve `aec_nr_pipeline_static.c`'s old file-local `pipeline_pool_size()` produced). `uint64_t`, not `size_t` — cast to `size_t` before passing to `malloc`/`posix_memalign`/`audio_pipeline_init*()` on a target where the two widths differ. |

`audio_pipeline_init_ex(mem, bytes, cfg, expected)` (see "Sequence" above)
automates exactly this comparison: passing the descriptor straight back in
as `expected` makes the library itself recompute the CURRENT
`descriptor_version`/`layout_version`/`backend_id`/`build_flags_hash`/
`alignment`/`bytes` and reject (`NULL`, with a diagnostic naming the
mismatched field) unless every one still agrees with `expected` — a board
integrator no longer has to hand-write this comparison. The one discipline
this does NOT relieve you of: `expected` must itself have been queried
freshly (see "Warnings" above) — `audio_pipeline_init_ex()` can only compare
what it's given against what the CURRENT build computes; it has no way to
know whether the `expected` you passed in was itself stale relative to some
THIRD, even older, build.

### Serializing the descriptor

`AudioPipelineMemReq` is a fixed-width, 32-byte POD (`_Static_assert`-pinned
in `audio_pipeline.h`, sizeof and every field's offset) — unlike the F20
shape, it can be copied byte-for-byte (`memcpy`) to a file, a flash region,
or a wire message, and read back later, even by a different process, even
after a restart. This is deliberately scoped narrower than "fully portable
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
  descriptor as `expected` to `audio_pipeline_init_ex()`; treat a persisted
  copy as informational at best (e.g. a board bring-up log recording what
  the PREVIOUS boot's build looked like).

### Dirty-pool contract

`audio_pipeline_init()` does **not** require a zero-filled pool. Every
pipeline-owned scratch buffer (the OLA accumulator, per-bin gain/spectrum
scratch, the mic/ref/output hop copies) is explicitly zeroed at carve time,
and AEC/NR/the FFT backend each zero their own sub-region during their own
`_init()` — so a pool filled with poison bytes inits and processes
identically to a freshly-zeroed one. `test_audio_pipeline.c`'s
create-vs-init parity case exercises exactly this: a `memset(pool, 0xA5,
bytes)`-poisoned pool run through `audio_pipeline_init()` produces
byte-for-byte the same 1000-hop output as `audio_pipeline_create()`'s
(unpoisoned) heap path. There is no need for a caller-side blanket
`memset(pool, 0, bytes)` before `audio_pipeline_init()` — it was only ever a
defensive habit carried over from the pre-F20 static CLI, not a requirement.

### `USE_EXT_MEM` — not a thing here

Both the heap path (`audio_pipeline_create`/`audio_pipeline_destroy`) and the
pool path (`audio_pipeline_get_mem_requirements`/`audio_pipeline_init`/
`audio_pipeline_destroy`) are always compiled into `libaudio_pipeline.a` —
which one you use is selected purely by which entry point you call, at
runtime, same as `lib/aec`'s and `lib/nr`'s own `_create` vs. `_get_mem_size`/
`_init` pairs. There is no `-DUSE_EXT_MEM`-style compile-time switch (that
pattern existed historically in `lib/nr` and was removed — see
`lib/nr/c_impl/CHANGELOG.md` `[v1.11.0]`/later entries); do not look for one,
and do not add one.

### `NO_STDIO=1` — building without libc stdio

`audio_pipeline.c`'s own diagnostics (init/build-time reject reasons only —
nothing on the per-hop `audio_pipeline_process()` path ever logs anything)
are advisory: every failure this file can hit is ALSO signalled through its
return value (`NULL`/`-1`), so a board image that cannot or will not link
libc's stdio (no console, or a policy that forbids the stdio symbol set
outright) still gets a fully-functional library — it only loses the
human-readable "why" that would otherwise go to `stderr`.

```bash
make BACKEND=ne10 NO_STDIO=1 libaudio_pipeline.a
```

compiles `audio_pipeline.o` with `-DAUDIO_PIPELINE_NO_STDIO`, which turns
every diagnostic in that file into a no-op and drops its `<stdio.h>`
include entirely — the resulting `libaudio_pipeline.a` references none of
`fprintf`/`printf`/`puts`/`fputs`/`stderr`/`__stderrp`.

```bash
make BACKEND=ne10 audit-no-stdio   # PASS/FAIL, non-zero exit on FAIL
```

builds exactly that and asserts it with `nm` over `audio_pipeline.o` itself
(pattern/style follows `audio_common/scripts/audit_alloc_symbols.sh`) — run
it after touching any diagnostic in `audio_pipeline.c` to confirm the gate
still holds.

`NO_STDIO` only ever changes `audio_pipeline.o`'s own compile flags (see the
`LIB_CFLAGS` comment in `Makefile`, and its `CFG_SIG`/`OBJ_DIR` participation
so a `NO_STDIO=1` build never shares an object directory with a default
build). Both CLIs (`aec_nr_pipeline`/`aec_nr_pipeline_static` — host tools
that always do their own WAV-I/O stdio) and `test_audio_pipeline` (prints
its own PASS/FAIL via stdio) keep stdio regardless — neither references
`AUDIO_PIPELINE_NO_STDIO`, so there is nothing to gain (and real CLI/test
diagnostics to lose) by gating them too. `make BACKEND=<b> NO_STDIO=1`
(without a specific target) still builds both CLIs normally, linked against
a `NO_STDIO=1` `libaudio_pipeline.a` — this is a supported combination, it
simply means the library half of that binary stays silent on rejection
while the CLI half keeps its own WAV-path error messages.

A board build linking a `NO_STDIO=1` `libaudio_pipeline.a` must rely
entirely on return values (`NULL`/`-1`, including the field-by-field
`audio_pipeline_init_ex()` rejection reasons documented under "Sequence"
above) — there is no error-callback or status-code-detail mechanism beyond
what each function already returns. The diagnostic strings exist for a
dev/desktop build's console only; do not build board firmware assuming
they will be there.

### Board-side verification checklist

Before trusting a board integration of this library in production, verify
each of the following on-target. Most of these are properties of the
**caller's** allocator/integration code, not of `audio_pipeline.c` itself —
this library only checks what it can observe from inside a single call
(alignment of the pointer it was handed, the `bytes` count it was told,
the `expected` descriptor if one was passed); it has no visibility into the
platform allocator, DMA engine, or power state behind that pointer.

- **16-byte alignment**: the `pool` pointer is aligned exactly as
  `req.alignment` reports (16 today). `audio_pipeline_init_ex()` rejects a
  misaligned `mem` argument, but only if one is actually passed in
  misaligned — the platform allocator itself must honor the alignment it
  was asked for.
- **Exact-bytes accounting**: the block handed to `audio_pipeline_init_ex()`
  really is at least `req.bytes` USABLE bytes — verify the allocator's own
  bookkeeping (e.g. an internal header carved out of the block it hands
  back) doesn't silently shrink the usable region below what was requested.
- **Region non-overlap**: the pool is not aliased with any other
  allocation, DMA buffer, or another `AudioPipeline`'s pool — every
  sub-module pointer inside the pool is a raw pointer into it, not a copy,
  so an overlapping region is silent memory corruption, not a caught error.
- **Exclusive lifetime**: nothing else reads or writes the pool for the
  entire lifetime of the handle, from `audio_pipeline_init_ex()` through
  the matching `audio_pipeline_destroy()` — no other task, ISR, or DMA
  transfer touches it concurrently.
- **Cache coherence / DMA ownership**: if `mic`/`ref`/`out` (or the pool
  itself, on a non-cache-coherent platform) cross a DMA boundary, the
  board's own cache-maintenance operations (clean/invalidate) run at the
  correct points. This library has no notion of cache lines or DMA and
  performs none of its own — `audio_pipeline_process()` assumes `mic`/`ref`
  are already CPU-visible on entry and that `out` is fully written before
  any DMA-out on it begins.
- **Allocator-failure / partial-init rollback**: confirm the caller's own
  rollback path is correct for a `platform_alloc()` failure BEFORE
  `audio_pipeline_init_ex()` is ever called — this library never partially
  commits (every rejection path returns `NULL` before writing into `mem`),
  so there is nothing on ITS side to roll back, but the caller's own
  pool-acquisition failure handling is not something this library can
  verify.
- **Reset / reconfigure / double-destroy / power-resume**: exercise
  `audio_pipeline_reset()` (echo-path change), a full destroy-then-
  re-`init_ex()` on the SAME pool (config change), calling
  `audio_pipeline_destroy()` twice in a row (must be a safe no-op the
  second time — see "Teardown order" below), and a power-suspend/resume
  cycle if the target platform has one (this library has no notion of
  power state itself; confirm the pool's contents survive whatever the
  platform does across suspend, or re-`init_ex()` from scratch on resume
  if they don't).
- **Allocator-hook trace of init→destroy (zero-heap, both backends)**:
  with a runtime allocator-hook/interposer (the style `lib/aec`'s and
  `audio_common`'s own zero-heap tests use), confirm NO `malloc`/`calloc`/
  `realloc`/`free` call happens between `audio_pipeline_init_ex()`
  returning and the matching `audio_pipeline_destroy()` returning, ON YOUR
  ACTUAL TARGET BUILD — KISS and NE10 are both zero-heap end-to-end since
  P0001 (see "Two Versions" above), but that is a property of THIS repo's
  reference builds, not a substitute for verifying your own board's build.
  **Include the logging path in this trace**: on the default (stdio-
  enabled) library build, a rejected `audio_pipeline_init_ex()` call only
  ever touches `stderr` (no heap symbol involved), so it does not itself
  break a zero-heap trace — but if the target forbids the stdio symbol set
  outright (not just heap use), build with `NO_STDIO=1` (above) so even
  that `stderr` reference is compiled out, and re-run the same audit
  against that build (`make BACKEND=<b> audit-no-stdio` is the static,
  `nm`-based version of this same check — see above).

### Teardown order

`audio_pipeline_destroy()` tears down NR → pipeline FFT (the OLA irfft
instance) → AEC — the reverse of `audio_pipeline_init()`'s carve order (AEC →
FFT → NR → scratch). Every one of those three calls is a genuine no-op for a
pool-resident instance today, on **both** backends (matches the "Two
Versions" section above — NE10's twiddle configs moved fully into the pool
under vendored patch P0001, closing the one case that used to need a real
release); the order is kept anyway as forward-compat insurance — a future
backend/module MAY hold something outside the pool that a destroy call
needs to release (see the P0001 history earlier in this file for what that
looked like pre-fix), and it is exactly what the heap-convenience path needs
for real (`free()` on the pool `audio_pipeline_create()` allocated).

## Tunable Parameters

### AEC (`AecConfig`, see `aec.h`)

**Presets**: `AEC_PRESET_GENTLE` / `AEC_PRESET_BALANCED`（default）/ `AEC_PRESET_AGGRESSIVE`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sample_rate` | 16000 | 8000 / 16000 / 48000，自動計算 frame/fft/hop |
| `filter_length` | sr×32ms | 自適應濾波器長度（256@8k, 512@16k, 1536@48k） |
| `enable_highpass` | 1 | 高通濾波器（移除 DC + 低頻） |
| `highpass_cutoff_hz` | 80.0 | HPF 截止頻率 (Hz) |

**RES / preset strength axis**：RES has no standalone `ResConfig` — it lives inside `AecConfig`
and the AEC3 post-filter chain (`SuppressionGain`, `ResidualEchoEstimator`, etc.), surfaced
externally through the `AecResContext` seam (see `## Modules` above). The three AEC presets
differ in exactly one field:

| Parameter | Gentle | Balanced | Aggressive | Description |
|-----------|--------|----------|------------|-------------|
| `min_gain_floor_far_active_db` | -20 | -28 | -38 | AEC3 `SuppressionGain` 遠端活躍時的最低增益下限 dB（最大抑制量）；其餘欄位（filter length、Kalman Q、delay buffer、CNG…）三個 preset 皆相同 |

### NR (`MmseLsaConfig`, see `mmse_lsa_types.h`)

**Modes**: `MMSE_LSA_NR_MILD` / `MMSE_LSA_NR_MODERATE` / `MMSE_LSA_NR_BALANCED`（default）/ `MMSE_LSA_NR_AGGRESSIVE`

> These are the library's mode enum, not this pipeline's CLI surface. `aec_nr_pipeline.c`'s
> `parse_nr_mode()` only recognizes `"mild"` / `"aggressive"` (anything else, including
> `"moderate"`, silently falls back to `MMSE_LSA_NR_BALANCED` — no error); the Python
> `aec_nr_pipeline.py` CLI likewise restricts `--nr-preset` to `choices=['mild', 'balanced',
> 'aggressive']`. `MODERATE` is only reachable by calling `mmse_lsa_config_for_mode()` directly.

`g_min_db` is in the amplitude-dB convention (`/20`, i.e. `g_min = 10^(g_min_db/20)`), not the
older power-dB (`/10`) convention:

| Parameter | Mild | Moderate | Balanced | Aggressive | Description |
|-----------|------|----------|----------|------------|-------------|
| `g_min_db` | -20 | -25 | -30 | -40 | 最小增益 dB（最大抑制量，amplitude dB, /20） |
| `q` | 0.60 | 0.55 | 0.50 | 0.35 | 語音先驗機率（低→積極抑噪） |
| `xi_min_db` | -15 | -18 | -20 | -25 | 先驗 SNR 下限 dB |
| `alpha_d` | 0.85 | 0.85 | 0.70 | 0.50 | 噪聲追蹤 IIR 係數 |
| `alpha_g` | 0.92 | 0.92 | 0.88 | 0.85 | 增益時間平滑（高→平滑） |
| `alpha_attack` | 0.40 | 0.40 | 0.30 | 0.15 | Attack 平滑（語音起始追蹤） |
| `alpha_decay` | 0.92 | 0.92 | 0.88 | 0.88 | Decay 平滑（噪聲釋放） |

**MCRA 噪聲估計**：

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_s` | 0.95 | 功率譜時間平滑 |
| `alpha_d` | 0.70 | 噪聲更新速率 |
| `L` | 32 | 最小值追蹤視窗（幀數，×10ms = 320ms） |
| `num_init_frames` | 20 | 初始化靜默幀數（200ms） |
| `scene_change_threshold_db` | 10.0 | 場景轉換偵測閾值 |

**SPP**：

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha_xi` | 0.88 | Decision Directed 先驗 SNR 平滑 |

---

## Troubleshooting & Tuning Guide

### AEC 相關

| 症狀 | 原因 | 調整方式 |
|------|------|----------|
| **殘留回聲明顯** | RES 抑制不足 | 改用更 aggressive preset，或直接覆寫 `min_gain_floor_far_active_db`（如 -28→-38） |
| **殘留回聲 + 遠端持續講話** | Filter 未完全收斂 | 增加 `filter_length`（如 832→1536），確認 mic-ref delay < filter_length |
| **近端語音被壓制（DT degradation）** | RES 過度抑制 | 改用 mild preset，或直接覆寫 `min_gain_floor_far_active_db`（如 -28→-20） |
| **輸出底噪不自然（突然靜音）** | CNG 未開啟 | 確認 `enable_cng=1`（preset 預設已開啟） |
| **收斂太慢** | Kalman Q 太保守 | 提高 `kalman_q_high`（如 1e-3→2e-3），減少 `warmup_frames`（如 100→50） |
| **Filter 發散（輸出爆音）** | Kalman Q 太激進或 echo path 劇變 | 降低 `kalman_q_high`（如 1e-3→5e-4） |
| **Echo path 變化後適應慢** | Shadow filter 太保守 | 提高 `shadow_mu_nlms`（如 0.5→0.7），降低 `shadow_err_alpha`（如 0.8→0.6） |

> `min_gain_floor_far_active_db` 是唯一在 mild/balanced/aggressive 三個 preset 間變動的欄位；
> 沒有獨立的 `res_*` tunable struct（見上方 `AEC (AecConfig, see aec.h)`）。

### NR 相關

| 症狀 | 原因 | 調整方式 |
|------|------|----------|
| **噪聲殘留太多** | 抑制量不夠 | 降低 `g_min_db`（如 -30→-40），降低 `q`（如 0.5→0.35） |
| **語音被吃掉** | 抑制太激進 | 提高 `g_min_db`（如 -30→-20），提高 `q`（如 0.5→0.6），提高 `alpha_g`（如 0.88→0.92） |
| **Musical noise（隨機顆粒噪聲）** | 增益抖動 | 提高 `alpha_g`（增益更平滑），提高 `alpha_decay`（釋放更慢） |
| **語音起始被截斷** | Attack 太慢 | 降低 `alpha_attack`（如 0.3→0.15），讓增益快速回升 |
| **噪聲環境切換後適應慢** | MCRA 追蹤窗太長 | 減小 `L`（如 32→16），但會增加噪聲底噪估計抖動 |
| **初始化期語音被壓** | 噪聲底噪估計偏高 | 減少 `num_init_frames`（如 20→10），但需確保前段有足夠噪聲 |
| **穩態噪聲殘留（風扇聲）** | 噪聲更新太慢 | 降低 `alpha_d`（如 0.7→0.5），讓噪聲估計更快跟上 |
| **語音段噪聲估計上升** | SPP 平滑不足 | 提高 `alpha_xi`（如 0.88→0.95），讓 SPP 更穩定判別語音 |

### Pipeline 整體

| 症狀 | 原因 | 調整方式 |
|------|------|----------|
| **回聲消了但底噪變大** | NR 沒開或太保守 | 確認 NR mode 非 MILD，或降低 `g_min_db` |
| **NR 把回聲當噪聲學進去** | AEC 殘留回聲被 MCRA 當底噪 | 先確保 AEC 收斂良好，再調 NR。提高 `num_init_frames` 讓 MCRA 避開 AEC 收斂期 |
| **整體語音品質差（悶、失真）** | 多階段過度處理 | 改用 MILD preset（AEC + NR 都放鬆），只在必要時加強個別模組 |
| **處理 48kHz 音訊記憶體不足** | 模組記憶體隨 fft_size 增長 | 縮短 `filter_length`、減小 NR `L`（主要記憶體佔用） |

### Verification

Both versions build from the default `make` and have been verified
**byte-identical** to each other (`cmp` on the full rendered WAV) at 16 kHz on
real doubletalk material (`aec_challenge_blind` case `0I0XMl3M`, balanced
presets, CNG on), and the static build's init asserts the 8 kHz / 16 kHz FFT
grids agree across AEC/NR/OLA (`n_freqs` cross-check at init).
