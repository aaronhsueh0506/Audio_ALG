#ifndef GSC_H
#define GSC_H

#include <stdint.h>

#include "fft_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef GSC_USE_PROJECTION_BLOCKING
#define GSC_USE_PROJECTION_BLOCKING 1
#endif

typedef struct {
    int enable;                  /* 0: bypass GSC, 1: enable GSC */
    float lambda;
    float mu;

    /* fix beam config */
    int enable_fix_mode;          /* 0: AUTO DOA, 1: FIX BEAM */
    float fixed_doa_rad;          /* only used when enable_fix_mode == 1 */
    int fixed_align_notebook;     /* 1: fixed mode matches Blocking_DSP_GSC.ipynb */

    /* adaptive update interval */
    int adapt_interval;           /* update RLS every N frames, <=0 means 1 */
} GSC_Config;

typedef struct {
    int enable;

    int M;
    int F;
    int num_angles;

    float lambda;
    float mu;

    /* fix beam config */
    int enable_fix_mode;
    float fixed_doa_rad;
    int fixed_align_notebook;

    /* adaptive update interval */
    int adapt_interval;

    /* steering */
    Complex*** a_array;

    /* RLS. P is stored as P[M][M][F], so four adjacent frequency bins of
     * one matrix element are contiguous for the AArch64 NEON update. The
     * public processing API is unchanged; this is internal state and must
     * not be persisted or indexed by integrators. */
    Complex*** P;   // (M,M,F)
    Complex** wa;   // (M,F)

    /* Persistent per-hop work area. Keeping the F- and M*F-sized arrays
     * here avoids a ~29 KiB stack spike at the 48 kHz / 1024-FFT grid. */
    Complex* scratch;
    Complex* scratch_das;
    Complex* scratch_wu;
    Complex* scratch_spec;
    Complex* scratch_u;
#if !GSC_USE_PROJECTION_BLOCKING
    Complex* scratch_b;
#endif

    /* state */
    int initialized;
    int first_doa_found;
    int64_t first_doa_frame;  /* -1 sentinel ("not found yet") preserved --
                               * int64_t (not uint64_t) so that stays a true
                               * negative value; still effectively unbounded
                               * for any real frame_idx snapshot */
    float current_doa;
    uint64_t frame_idx;  /* was int: a per-frame monotonic counter would
                          * overflow INT_MAX (~199-265 days at typical hop
                          * rates); uint64_t is effectively unbounded for
                          * any real run and avoids signed-overflow UB */

    /* log */
    float doa_used;
    int adaptive;

    /* Lifetime count of per-bin RLS divergence recoveries (see
     * GSC_P_DIAG_FLOOR/CEIL in gsc.c) -- NOT cleared by gsc_reset(), so a
     * caller/test can observe whether numerical instability occurred at all
     * across this instance's whole run. Expected to stay at (or very near)
     * 0 in normal operation; the diagonal clamp is meant to keep this
     * reactive path rare. */
    long bin_resets;

    /* Non-NULL only on the gsc_create() heap path: the single
     * posix_memalign()'d block backing every array above, freed by
     * gsc_destroy(). NULL on the gsc_init() caller-pool path, where the
     * caller owns the memory and gsc_destroy() must not free it. Appended
     * at the end of the struct so every field above keeps its pre-existing
     * offset (same precedent as AEC/c_impl's Aec.heap_arena). */
    void* owned_heap;

} GSC;

/*
 * Single source of truth for the adapt_interval GSC will actually run at,
 * once fixed-notebook-mode forcing is taken into account.
 *
 * gsc_create() forces the RLS update cadence to 1 (every hop) whenever
 * enable_fix_mode && fixed_align_notebook, regardless of the caller's
 * requested adapt_interval (kept for baseline-matching against the
 * reference notebook). Any quantity that is derived from "how often GSC
 * actually adapts" -- e.g. a caller retiming lambda/mu for a slower
 * wall-clock update period -- MUST call this same function with the same
 * inputs instead of re-deriving the forcing rule, so the effective cadence
 * used for that derivation and the cadence gsc_create() actually configures
 * can never silently diverge.
 */
int gsc_effective_adapt_interval(
    int enable_fix_mode, int fixed_align_notebook, int adapt_interval);

/* create */
GSC* gsc_create(int M, int F, int num_angles,
                Complex*** a_array,
                const GSC_Config* cfg);

/*
 * Static-memory companions: place a GSC struct plus every backing array
 * (P/wa/scratch) in a single caller-owned pool. Mirrors AEC/c_impl's
 * aec_get_mem_size()/aec_init() pair (same simple, non-descriptor tier --
 * GSC is a single library-level module, not a composite like
 * 4aec_nr_res.c).
 *
 * gsc_get_mem_size(M, F) returns the total byte requirement; it does not
 * depend on num_angles or cfg (neither affects GSC's own storage --
 * num_angles only indexes into the caller-owned a_array, and cfg's fields
 * are plain scalars copied into the struct). Returns 0 for an invalid
 * (M <= 0 or F <= 0) shape.
 *
 * gsc_init() places the GSC struct at mem[0], carves every backing array out
 * of the same block (no malloc called), and applies the exact same
 * validation gsc_create() does. `a_array` remains a BORROWED pointer in both
 * gsc_create() and gsc_init() -- it is owned by the caller (typically SRP),
 * never carved from the pool or freed by GSC.
 */
size_t gsc_get_mem_size(int M, int F);
GSC*   gsc_init(void* mem, size_t mem_size, int M, int F,
                int num_angles, Complex*** a_array,
                const GSC_Config* cfg);

void gsc_process(GSC* g,
                 const Complex* const* X,
                 float doa_s,
                 int allow_adapt_in,
                 const int* mask,
                 Complex* gsc_out);

/*
 * Same GSC processing and state update as gsc_process(), plus the exact
 * pre-update effective coefficients under:
 *
 *   gsc_out[f] = sum(effective_weights[m,f] * X[m,f])
 *
 * (no conjugation at the call site).  The output buffer may be NULL only
 * when the corresponding result is not needed; gsc_out remains required.
 */
void gsc_process_with_weights(GSC* g,
                              const Complex* const* X,
                              float doa_s,
                              int allow_adapt_in,
                              const int* mask,
                              Complex* gsc_out,
                              Complex* effective_weights);

void gsc_reset(GSC* g);
void gsc_destroy(GSC* g);
float gsc_get_doa_used(const GSC* g);
int gsc_get_adaptive(const GSC* g);

/*
 * The per-hop leak factor applied to the adaptive weight state `wa` during
 * an active, unmasked RLS update (see gsc.c's GSC_WA_LEAK comment). Exposed
 * read-only so callers/tests can reference the authoritative constant
 * instead of duplicating its numeric value.
 */
float gsc_wa_leak_factor(void);

/*
 * Lifetime count of per-bin RLS divergence recoveries this instance has
 * performed (see GSC struct's bin_resets field). Read-only diagnostic --
 * expected to stay at 0 in normal operation.
 */
long gsc_get_bin_resets(const GSC* g);

/*
 * The P-diagonal clamp bounds applied every adapted frame (see gsc.c's
 * GSC_P_DIAG_FLOOR/CEIL comment). Exposed read-only so callers/tests can
 * reference the authoritative constants instead of duplicating their
 * numeric values (same rationale as gsc_wa_leak_factor()).
 */
float gsc_p_diag_floor(void);
float gsc_p_diag_ceil(void);

#ifdef __cplusplus
}
#endif

#endif
