/**
 * audio_pipeline.h — linkable, pool-first library API for the AEC(linear) ->
 * echo-aware NR -> RES pipeline (review F20).
 *
 * Before this file, the pool-sizing/carving/per-hop-processing logic that now
 * lives here was file-local `static` code duplicated (byte-for-byte) inside
 * pipelines/aec_nr_pipeline_static.c's `main()` — a CLI could run it, but no
 * other binary could link against it. Board firmware wants exactly the
 * opposite shape: a caller-owned memory block (from the platform's own
 * allocator, not `malloc`), a `bytes` requirement it can query up front, and
 * a process() call it drives from its own audio ISR/task loop — no WAV I/O,
 * no argv, no stdio.
 *
 * This header is that shape. It intentionally borrows ONLY the naming
 * convention (`audio_pipeline_*`) from the older heap-era design note
 * pipelines/PLAN_audio_pipeline_api.md — not its feature set. That plan's
 * `PipelineMode` enum, per-frame debug callback, and hot/warm runtime
 * setters are OUT OF SCOPE here: this API wires exactly the ONE DSP chain
 * aec_nr_pipeline.c / aec_nr_pipeline_static.c already ship (AEC linear ->
 * echo-aware NR -> RES, i.e. the old plan's `PIPELINE_AEC_NR_RES` mode),
 * nothing else.
 *
 * ── Two ways to get an AudioPipeline* ───────────────────────────────────────
 *
 *   Pool-first (the board story — zero heap involvement on this path):
 *
 *     AudioPipelineConfig cfg = audio_pipeline_default_config(16000);
 *     AudioPipelineMemReq req;
 *     audio_pipeline_get_mem_requirements(&cfg, &req);
 *     void* pool = platform_alloc(req.bytes, req.alignment);  // 16-byte aligned
 *     AudioPipeline* p = audio_pipeline_init(pool, req.bytes, &cfg);
 *     ...
 *     audio_pipeline_destroy(p);       // no-op on the pool path (see below)
 *     platform_free(pool);
 *
 *   Heap convenience (desktop CLIs, quick prototyping):
 *
 *     AudioPipeline* p = audio_pipeline_create(&cfg);
 *     ...
 *     audio_pipeline_destroy(p);       // frees the pool `create()` allocated
 *
 * Per-hop:
 *
 *     float mic[hop], ref[hop], out[hop];
 *     while (have_audio()) {
 *         read_hop(mic, ref, hop);
 *         audio_pipeline_process(p, mic, ref, out);
 *         write_hop(out, hop);
 *     }
 *
 * See pipelines/README.md ("Board Integration") for the full sequence
 * (query -> allocate -> init -> process* -> reset? -> destroy -> release),
 * teardown-order rationale, and the layout_version/build_flags_hash contract.
 */
#ifndef AUDIO_PIPELINE_H
#define AUDIO_PIPELINE_H

#include <stddef.h>
#include <stdint.h>

#include "aec.h"                /* AecConfig/AecPreset, Aec (non-opaque)      */
#include "mmse_lsa_denoiser.h"  /* MmseLsaConfig/MmseLsaNrMode, MmseLsaDenoiser (opaque) */

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Memory descriptor
 * ========================================================================== */

/**
 * Richer-than-a-bare-size_t memory descriptor (review F20). A board memory
 * manager consuming ONLY `bytes` (the way the old CLI's bare
 * `pipeline_pool_size() -> size_t` did) has no way to tell whether a
 * previously-recorded byte count still describes today's library build, or
 * whether the block it's about to hand back is even legal to carve into.
 * The extra fields close that gap:
 *
 *   - alignment:        required base alignment of the `mem` pointer passed
 *                        to audio_pipeline_init(). Always 16 today (the one
 *                        alignment every module in this stack — AEC, NR,
 *                        the FFT backends, mem_align.h's ALIGN16 — carves
 *                        to); kept as a field rather than a hardcoded "16"
 *                        so a caller's code doesn't have to.
 *   - layout_version:   bumped whenever this file's OWN carve order, buffer
 *                        set, or per-buffer sizing formula changes (i.e.
 *                        whenever a `bytes` figure computed by an OLDER
 *                        build of this header/TU would misdescribe a NEWER
 *                        build's actual carve, or vice versa). Starts at 1.
 *                        Does NOT need bumping for a change purely inside
 *                        AEC's/NR's/the FFT backend's OWN internal
 *                        `_get_mem_size` layout — those already change
 *                        `bytes` itself (this pipeline treats each of the
 *                        three as an opaque composite blob, exactly as
 *                        pipelines/README.md's "Two Versions" section
 *                        documents), so a stale cached `bytes` from an old
 *                        submodule build is already caught by an undersized-
 *                        pool rejection at init, without needing a version
 *                        bump on ITS OWN axis. layout_version is specifically
 *                        this file's contract with a caller who might cache
 *                        the descriptor across a library upgrade.
 *   - backend:           compile-time FFT backend identity, "kiss" or
 *                        "ne10" (the `BACKEND=` a caller built THIS TU
 *                        with — see pipelines/Makefile). Informational: the
 *                        two backends are not byte-identical to each other
 *                        (pre-existing, expected — see lib/aec/CLAUDE.md),
 *                        so a descriptor computed against one is never a
 *                        valid stand-in for the other even at matching
 *                        `bytes`.
 *   - build_flags_hash:  FNV-1a-32 hash of a small, fixed set of compile-time
 *                        strings that affect this file's carve STRUCTURE:
 *                        the backend identity (above) plus a literal token
 *                        list naming the pipeline's own 13 scratch buffers
 *                        in carve order, plus the alignment granularity.
 *                        See audio_pipeline.c's audio_pipeline_build_flags_hash()
 *                        for the exact inputs. Covers: a change to this
 *                        file's own carve order/buffer set/alignment.
 *                        Does NOT cover: AecConfig/MmseLsaConfig preset or
 *                        tunable VALUES (g_min_db, min_gain_floor_far_active_db,
 *                        sample_rate, aec_only, ...) — those change `bytes`
 *                        but are config, not layout, so they're deliberately
 *                        excluded (a caller re-querying get_mem_requirements()
 *                        for its actual config already gets the right
 *                        `bytes`); AEC's/NR's/the FFT backend's internal
 *                        struct layouts (opaque composite blobs, as above);
 *                        the compiler/ABI/toolchain.
 */
typedef struct {
    size_t      bytes;             /* total pool size (>= this many bytes, 16-aligned) */
    size_t      alignment;         /* required base alignment of `mem`, bytes (16)     */
    uint32_t    layout_version;    /* this file's own carve-layout version (see above) */
    const char* backend;           /* "kiss" | "ne10" (compile-time)                   */
    uint32_t    build_flags_hash;  /* FNV-1a-32 of the layout-affecting compile flags   */
} AudioPipelineMemReq;

/* ============================================================================
 * Config
 * ========================================================================== */

/**
 * Everything the two CLIs (aec_nr_pipeline.c / aec_nr_pipeline_static.c)
 * currently read from argv to shape a run, minus the purely-CLI concerns
 * (WAV paths, --debug, DUMP_CTX) — those stay in each CLI, which now reads
 * the underlying Aec* / MmseLsaDenoiser* handles directly via
 * audio_pipeline_get_aec()/audio_pipeline_get_nr() for its own diagnostics.
 */
typedef struct {
    int           sample_rate;   /* 8000 | 16000 | 48000                              */
    AecPreset     aec_preset;    /* MILD | BALANCED | AGGRESSIVE                       */
    MmseLsaNrMode nr_mode;       /* MILD | MODERATE | BALANCED | AGGRESSIVE            */
    int           aec_only;      /* 1 = skip NR/RES entirely (linear AEC output only)  */
    int           enable_cng;    /* 1 = fill AEC-suppressed bins with comfort noise    */
    int           legacy_amin;   /* 1 = prior min-only A_min_pl (--legacy-amin): NR    *
                                   * gain computed WITHOUT folding R² into the noise    *
                                   * floor, and the far/near-gated near-end floor       *
                                   * strength collapses to the fixed scalar 0.4         */
} AudioPipelineConfig;

/** Sane defaults: BALANCED/BALANCED, full pipeline, CNG on, non-legacy. */
AudioPipelineConfig audio_pipeline_default_config(int sample_rate);

/* ============================================================================
 * Opaque handle
 * ========================================================================== */

typedef struct AudioPipeline AudioPipeline;

/* ============================================================================
 * Pool-first lifecycle
 * ========================================================================== */

/**
 * Query the memory descriptor for `cfg` WITHOUT allocating or touching any
 * audio state. Validates `cfg` via the same module validators
 * aec_get_mem_size()/mmse_lsa_get_mem_size() already gate on internally
 * (aec_validate_config / mmse_lsa_validate_config, both invalid-config ->
 * return 0) PLUS an explicit reject-first check up front, in
 * derive_dims_and_configs() (the one place every entry point in this file
 * funnels through): sample_rate against the {8000,16000,48000} whitelist
 * (aec_is_valid_sample_rate — e.g. sample_rate=44100 is rejected before any
 * size arithmetic runs, not just left to a downstream 0), aec_preset/nr_mode
 * against their defined enum values (rather than silently falling through
 * aec_config_from_preset's/mmse_lsa_config_for_mode's own balanced-default
 * fallback), and aec_only/enable_cng/legacy_amin against {0,1} (rather than
 * being treated as truthy by a stray nonzero value downstream).
 *
 * @return 0 on success (*out filled), -1 on NULL args or invalid cfg.
 */
int audio_pipeline_get_mem_requirements(const AudioPipelineConfig* cfg,
                                         AudioPipelineMemReq* out);

/**
 * Carve an AudioPipeline instance (control block + AEC + FFT(OLA) + NR + the
 * 13 pipeline scratch buffers, in that order) out of `mem`, verbatim-porting
 * the carve order/sizes the static CLI's file-local `pipeline_build` used.
 *
 * Requirements on `mem`/`bytes` (checked, NULL-returned on violation):
 *   - `mem` must be 16-byte aligned (MEM_IS_ALIGNED16).
 *   - `bytes` must be >= audio_pipeline_get_mem_requirements(cfg, ...)->bytes.
 *   - `mem` need NOT be zero-filled: every pipeline-owned scratch buffer
 *     (OLA accumulator, per-bin gain/spectrum scratch, the mic/ref/output
 *     hop copies) is explicitly zeroed here at carve time, and AEC/NR/the
 *     FFT backend each zero their own sub-region during their own
 *     `_init()` — so a pool filled with poison bytes (e.g. `memset(pool,
 *     0xA5, bytes)`, the pattern lib/aec's own zero-heap test uses) inits
 *     and processes identically to a freshly-zeroed one. See
 *     test_audio_pipeline.c's dirty-pool case.
 *
 * The pool must stay stable (nothing else writes into it) and exclusive
 * (not shared with any other AudioPipeline/AEC/NR/FFT instance) for the
 * entire lifetime of the returned handle — every sub-module and pipeline
 * buffer is a raw pointer into it, not a copy.
 *
 * @return a valid handle, or NULL (misaligned/undersized pool, invalid cfg,
 *         or a sub-module init/grid-agreement failure — see stderr).
 */
AudioPipeline* audio_pipeline_init(void* mem, size_t bytes,
                                    const AudioPipelineConfig* cfg);

/**
 * Process exactly one hop (audio_pipeline_hop_size(p) samples) of mic/ref
 * into `out`. Verbatim port of the static CLI's per-hop while-loop body:
 * AEC(linear) -> echo-aware NR gain -> g_total=min(g_nr,g_res) -> far/near
 * gated near-end floor lift -> S(f)=E(f)*g_total (+ CNG on the cut bins) ->
 * irfft -> sqrt-Hann OLA. `aec_only` short-circuits to the raw linear AEC
 * residual (mirrors the CLI's `--aec-only`).
 *
 * `mic`/`ref` are read-only and only for the duration of this call (they are
 * copied into pool-owned scratch before use — see audio_pipeline.c); `out`
 * is fully overwritten (never read). All three must be exactly
 * audio_pipeline_hop_size(p) floats.
 *
 * @return 0 on success, -1 if p/mic/ref/out is NULL.
 */
int audio_pipeline_process(AudioPipeline* p, const float* mic,
                            const float* ref, float* out);

/**
 * Re-zero all pipeline/AEC/NR state (OLA accumulator, comfort-noise RNG,
 * near-end-floor hangover counter, and each sub-module's own reset) without
 * touching the pool itself or re-validating cfg — equivalent to a fresh
 * audio_pipeline_init() on the SAME pool/cfg, but without the alignment/size
 * re-checks. Use after an echo-path change (speaker swap, AEC re-seat) or
 * between unrelated streams sharing one instance.
 */
void audio_pipeline_reset(AudioPipeline* p);

/**
 * Tear down in reverse carve order: NR -> pipeline FFT (the OLA irfft
 * instance) -> AEC. This is the mirror image of audio_pipeline_init's carve
 * order (AEC -> FFT -> NR -> scratch) and matches the teardown order
 * pipelines/README.md's "Two Versions" section already documents for the
 * static CLI (`mmse_lsa_destroy` / `fft_destroy` / `aec_destroy`, in that
 * order) — kept even though every one of those three calls is a genuine
 * no-op on a pool-resident (audio_pipeline_init'd) instance today: it is
 * forward-compat insurance (a future backend/module MAY hold something
 * outside the pool that a destroy call needs to release — see the NE10
 * twiddle-config caveat in aec.h/fft_wrapper.h) and is exactly what the
 * heap convenience path (audio_pipeline_create) needs for real.
 *
 * NULL-safe (destroy(NULL) is a no-op) and idempotent FOR A POOL-RESIDENT
 * INSTANCE — repeated calls are safe because each of the three underlying
 * destroy calls already promises that. For a HEAP instance (obtained via
 * audio_pipeline_create()), the SAME single free() this call performs on
 * the pool follows ordinary free() semantics: call exactly once. A second
 * call on a heap instance is a double-free, exactly as a second
 * free()/fft_destroy() on an already-freed heap handle would be — this
 * function cannot detect that case (the instance it would check is the
 * memory being freed).
 */
void audio_pipeline_destroy(AudioPipeline* p);

/* ============================================================================
 * Heap convenience (desktop CLIs / quick prototyping — NOT the board path)
 * ========================================================================== */

/**
 * audio_pipeline_get_mem_requirements() + posix_memalign(16, ...) +
 * audio_pipeline_init(), all in one call. The returned handle owns its pool;
 * audio_pipeline_destroy() frees it.
 *
 * @return a valid handle, or NULL (invalid cfg or allocation failure).
 */
AudioPipeline* audio_pipeline_create(const AudioPipelineConfig* cfg);

/* ============================================================================
 * Accessors
 * ========================================================================== */

int audio_pipeline_hop_size(const AudioPipeline* p);     /* -1 if p is NULL */
int audio_pipeline_n_freqs(const AudioPipeline* p);      /* -1 if p is NULL */
int audio_pipeline_sample_rate(const AudioPipeline* p);  /* -1 if p is NULL */

/**
 * Read-only access to the underlying module handles, for a caller's OWN
 * diagnostics (aec_debug_status/aec_get_res_context/mmse_lsa_debug_status/
 * mmse_lsa_get_gain/...) — this is how both CLIs keep their existing
 * `--debug` status line and `DUMP_CTX` per-hop dump working as thin
 * wrappers, without this header re-exposing every intermediate signal
 * itself. Do not call any _reset/_destroy/mutating entry point on these
 * directly — go through audio_pipeline_reset()/audio_pipeline_destroy() so
 * pipeline-owned state (OLA, RNG, hangover counter) stays in sync.
 */
Aec*             audio_pipeline_get_aec(const AudioPipeline* p);  /* never NULL for a valid p */
MmseLsaDenoiser* audio_pipeline_get_nr(const AudioPipeline* p);   /* NULL iff cfg.aec_only     */

/* ============================================================================
 * Diagnostic breakdown (backs --print-mem-size in both CLIs)
 * ========================================================================== */

/** Per-module byte breakdown, mirroring the static CLI's original
 * `print_mem_budget()` table (AEC / FFT / NR / pipeline-buffer columns) so
 * that diagnostic stays available without either CLI re-deriving
 * AecConfig/MmseLsaConfig/frame-dims itself. NOT part of
 * AudioPipelineMemReq: those bytes are already folded into `bytes` there;
 * this is purely for the human-readable table. */
typedef struct {
    size_t aec_bytes;
    size_t fft_bytes;         /* 0 when cfg.aec_only */
    size_t nr_bytes;          /* 0 when cfg.aec_only */
    size_t pipeline_bytes;    /* the 13 scratch buffers (fewer when aec_only) */
    int    hop, frame_sz, fft_sz, n_freqs;
} AudioPipelineMemBreakdown;

/**
 * @return 0 on success (*out filled), -1 on NULL args or invalid cfg (same
 *         validation as audio_pipeline_get_mem_requirements()).
 */
int audio_pipeline_get_mem_breakdown(const AudioPipelineConfig* cfg,
                                      AudioPipelineMemBreakdown* out);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_PIPELINE_H */
