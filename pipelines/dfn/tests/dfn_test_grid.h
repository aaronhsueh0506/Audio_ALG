/* The DeepFilterNet2 grid a pipeline test program runs.
 *
 * The library and the pipeline objects (dfn_res_stage, dfn_rate_bridge, dfn_hop) hold no grid:
 * they take a DFN2Grid built by the application with dfn2_grid_make(). A test program has to
 * decide one, and it is chosen when the program is built so that one test source covers every
 * grid and lookahead pair and its buffers can be plain arrays:
 *
 *   -DDFN2_GRID_KHZ=16 | 32 | 48          a preset below (48 when nothing is chosen)
 *   -DDFN2_SR=.. -DDFN2_N_FFT=.. -DDFN2_N_ERB=.. -DDFN2_DF_BINS=..
 *                                         a grid of another model (all four, no preset)
 *   -DDFN2_MASK_LOOKAHEAD=m -DDFN2_DF_LOOKAHEAD=d
 *                                         the lookahead pair (1/1 full, 1/0 mask-only);
 *                                         a pair with ring > 5 also needs
 *                                         -DDFN2_MAX_DF_RING=<ring> on every unit
 *
 *      preset   sr      n_fft  hop   bins  n_erb  df_bins
 *      48       48000   1024   512   513   32     96
 *      32       32000   512    256   257   28     96     (not trained yet)
 *      16       16000   256    128   129   24     96
 *
 * This is test scaffolding only. dfn_test_grid() hands the stage, the bridge and the library the
 * same grid as a DFN2Grid, built by dfn2_grid_make(), and the DFN2_* macros below size the test's
 * own buffers. Product code (the pipeline objects, the example mains) never includes this header:
 * it takes its grid from its caller or its arguments. pipelines/Makefile applies the selectors to
 * the test programs only (DFN2_TEST_GRID_KHZ, DFN2_MASK_LOOKAHEAD / DFN2_DF_LOOKAHEAD). */

#ifndef DFN_TEST_GRID_H
#define DFN_TEST_GRID_H

#include <stdio.h>
#include <stdlib.h>

#include "dfn2_prepost.h"

#ifdef DFN2_SR
#ifdef DFN2_GRID_KHZ
#error "define either DFN2_GRID_KHZ (a preset) or the grid macros (DFN2_SR, ...), not both"
#endif
#if !defined(DFN2_N_FFT) || !defined(DFN2_N_ERB) || !defined(DFN2_DF_BINS)
#error "a test grid defined by DFN2_SR needs DFN2_N_FFT, DFN2_N_ERB and DFN2_DF_BINS too"
#endif
#else
#ifndef DFN2_GRID_KHZ
#define DFN2_GRID_KHZ 48
#endif
#if defined(DFN2_N_FFT) || defined(DFN2_N_ERB) || defined(DFN2_DF_BINS)
#error "a DFN2_GRID_KHZ preset is complete: to change one value define the whole grid, starting with DFN2_SR"
#endif
#if DFN2_GRID_KHZ == 48
#define DFN2_SR                 48000
#define DFN2_N_FFT              1024
#define DFN2_N_ERB              32
#define DFN2_DF_BINS            96
#elif DFN2_GRID_KHZ == 32
#define DFN2_SR                 32000
#define DFN2_N_FFT              512
#define DFN2_N_ERB              28
#define DFN2_DF_BINS            96
#elif DFN2_GRID_KHZ == 16
#define DFN2_SR                 16000
#define DFN2_N_FFT              256
#define DFN2_N_ERB              24
#define DFN2_DF_BINS            96
#else
#error "DFN2_GRID_KHZ must be 16, 32 or 48 (or define DFN2_SR, DFN2_N_FFT, DFN2_N_ERB and DFN2_DF_BINS)"
#endif
#endif

#define DFN2_N_BINS         ((DFN2_N_FFT) / 2 + 1)
#define DFN2_WIN_LEN        (DFN2_N_FFT)
#define DFN2_HOP_LEN        ((DFN2_N_FFT) / 2)
#define DFN2_OVL_LEN        (DFN2_WIN_LEN - DFN2_HOP_LEN)

#ifndef DFN2_MASK_LOOKAHEAD
#define DFN2_MASK_LOOKAHEAD 1
#endif
#ifndef DFN2_DF_LOOKAHEAD
#ifdef DFN2_MASK_ONLY
#define DFN2_DF_LOOKAHEAD   0
#else
#define DFN2_DF_LOOKAHEAD   1
#endif
#endif
#define DFN2_DF_HISTORY     (DFN2_DF_ORDER - DFN2_DF_LOOKAHEAD - 1)
#define DFN2_DF_RING_NOISY  (DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD + 1)
#ifdef DFN2_MASK_ONLY
#define DFN2_DF_RING        DFN2_DF_RING_NOISY
#else
#define DFN2_DF_RING \
    (DFN2_DF_ORDER > DFN2_DF_RING_NOISY ? DFN2_DF_ORDER : DFN2_DF_RING_NOISY)
#endif

/* Frames the stage's output trails its input by: the cascade of the test grid's lookahead pair. */
#define DFN_TEST_LOOKAHEAD_FRAMES (DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD)

#if DFN2_DF_RING > DFN2_MAX_DF_RING
#error "this lookahead pair needs a deeper ring than the library capacity: pass -DDFN2_MAX_DF_RING=<DFN2_DF_RING> on every translation unit"
#endif

/* The grid of this program as the library's value; built once, aborts on a grid the library
 * refuses (the test would be running something else). */
static inline const DFN2Grid *dfn_test_grid(void)
{
    static DFN2Grid grid;
    static int ready;
    if (!ready) {
        int status = dfn2_grid_make(&grid, DFN2_SR, DFN2_N_FFT, DFN2_N_ERB, DFN2_DF_BINS,
                                    DFN2_MASK_LOOKAHEAD, DFN2_DF_LOOKAHEAD);
        if (status != DFN2_GRID_OK) {
            fprintf(stderr, "dfn_test_grid: %s\n", dfn2_grid_status_text(status));
            exit(2);
        }
        ready = 1;
    }
    return &grid;
}

/* The line `make test-pairs` and `make test-dfn-grids` read from a test binary (pipelines/Makefile):
 * the lookahead pair the program was built with, when the program measured it the stage latency it
 * observed in hops (measured_hops < 0: not measured), and last the grid (sample rate Hz / FFT size).
 * test-pairs compares the pair with the pair make asked for and the measured latency L with
 * mask + df, and the algorithmic latency (L + 1) * hop (the stage's cascade plus the framing hop)
 * with (mask + df + 1) * hop of the test grid; test-dfn-grids compares the grid with the one it
 * asked for, so a grid flag that never reached the test build fails there. */
static inline void dfn_test_report_pair(const char *who, int measured_hops)
{
    const DFN2Grid *g = dfn_test_grid();
    if (measured_hops < 0)
        printf("%s: lookahead pair %d/%d, grid %d Hz / %d\n", who, g->mask_lookahead,
               g->df_lookahead, g->sample_rate, g->n_fft);
    else
        printf("%s: lookahead pair %d/%d, measured stage latency %d hops, algorithmic latency "
               "(%d + 1) * %d = %d samples, grid %d Hz / %d\n", who, g->mask_lookahead,
               g->df_lookahead, measured_hops, measured_hops, g->hop,
               (measured_hops + 1) * g->hop, g->sample_rate, g->n_fft);
}

#endif /* DFN_TEST_GRID_H */
