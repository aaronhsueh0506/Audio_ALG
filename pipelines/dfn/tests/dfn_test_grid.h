/* The DeepFilterNet2 grid a pipeline test program runs.
 *
 * The library and the pipeline objects (dfn_res_stage, dfn_rate_bridge, dfn_hop) hold no grid:
 * they take a DFN2Grid built by the application with dfn2_grid_make(). A test program decides one
 * when it is built, so that one test source covers every grid and lookahead pair and its buffers
 * can be plain arrays:
 *
 *   -DDFN2_GRID_KHZ=16 | 32 | 48          a preset below (48 when nothing is chosen)
 *   -DDFN2_SR=.. -DDFN2_N_FFT=.. -DDFN2_N_ERB=.. -DDFN2_DF_BINS=..
 *                                         a grid of another model (all four, no preset)
 *   -DDFN2_MASK_LOOKAHEAD=m -DDFN2_DF_LOOKAHEAD=d
 *                                         the lookahead pair (default 1/1)
 *
 *      preset   sr      n_fft  hop   bins  n_erb  df_bins
 *      48       48000   1024   512   513   32     96
 *      32       32000   512    256   257   28     96     (not trained yet)
 *      16       16000   256    128   129   24     96
 *
 * The df_order, frames_per_call and normaliser numbers are the shipped checkpoint's
 * (AINR/DeepFilterNet2/config.ini [model], [feature]); test_dfn_test_grid_pin.py compares them.
 * Test scaffolding only: product code takes its grid from its caller or its arguments.
 * pipelines/Makefile applies the selectors to the test programs only (DFN2_TEST_GRID_KHZ,
 * DFN2_MASK_LOOKAHEAD / DFN2_DF_LOOKAHEAD). */

#ifndef DFN_TEST_GRID_H
#define DFN_TEST_GRID_H

#include <stdio.h>
#include <stdlib.h>

#include "dfn2_prepost.h"

#ifdef DFN2_MASK_ONLY
#error "the pipeline tests run the full DFN2 variant: mask-only is not driven through the pipelines"
#endif

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
#define DFN2_HOP_LEN        ((DFN2_N_FFT) / 2)

#ifndef DFN2_MASK_LOOKAHEAD
#define DFN2_MASK_LOOKAHEAD 1
#endif
#ifndef DFN2_DF_LOOKAHEAD
#define DFN2_DF_LOOKAHEAD   1
#endif

/* Model numbers every row of the test tables carries. */
#define DFN_TEST_DF_ORDER             5
#define DFN_TEST_FRAMES_PER_CALL      1
#define DFN_TEST_ERB_NORM_TAU_SEC     1.0f
#define DFN_TEST_ERB_NORM_INIT_LO_DB  (-60.0f)
#define DFN_TEST_ERB_NORM_INIT_HI_DB  (-90.0f)
#define DFN_TEST_ERB_NORM_SCALE_DB    40.0f
#define DFN_TEST_SPEC_NORM_TAU_SEC    1.0f
#define DFN_TEST_SPEC_NORM_INIT_LO    0.001f
#define DFN_TEST_SPEC_NORM_INIT_HI    0.0001f

/* Largest row of the test tables (48 kHz / 1024): the static matrices and spectra of the table
 * rows are sized from it, and dfn_test_table_fits() refuses a row beyond it. */
#define DFN_TEST_TABLE_MAX_BINS       513
#define DFN_TEST_TABLE_MAX_ERB        32

/* Frames the stage's output trails its input by: the cascade of the test grid's lookahead pair. */
#define DFN_TEST_LOOKAHEAD_FRAMES (DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD)

/* A complete spec for one table row: every field is named, so none can stay zero. */
static inline DFN2GridSpec dfn_test_spec(int sample_rate, int n_fft, int n_erb, int df_bins,
                                         int mask_lookahead, int df_lookahead)
{
    DFN2GridSpec spec = {
        .sample_rate          = sample_rate,
        .n_fft                = n_fft,
        .n_erb                = n_erb,
        .df_bins              = df_bins,
        .df_order             = DFN_TEST_DF_ORDER,
        .mask_lookahead       = mask_lookahead,
        .df_lookahead         = df_lookahead,
        .frames_per_call      = DFN_TEST_FRAMES_PER_CALL,
        .erb_norm_tau_sec     = DFN_TEST_ERB_NORM_TAU_SEC,
        .erb_norm_init_lo_db  = DFN_TEST_ERB_NORM_INIT_LO_DB,
        .erb_norm_init_hi_db  = DFN_TEST_ERB_NORM_INIT_HI_DB,
        .erb_norm_scale_db    = DFN_TEST_ERB_NORM_SCALE_DB,
        .spec_norm_tau_sec    = DFN_TEST_SPEC_NORM_TAU_SEC,
        .spec_norm_init_lo    = DFN_TEST_SPEC_NORM_INIT_LO,
        .spec_norm_init_hi    = DFN_TEST_SPEC_NORM_INIT_HI,
    };
    return spec;
}

/* The spec that makes `grid` again, every field copied from it. */
static inline DFN2GridSpec dfn_test_spec_of(const DFN2Grid *grid)
{
    DFN2GridSpec spec = {
        .sample_rate          = grid->sample_rate,
        .n_fft                = grid->n_fft,
        .n_erb                = grid->n_erb,
        .df_bins              = grid->df_bins,
        .df_order             = grid->df_order,
        .mask_lookahead       = grid->mask_lookahead,
        .df_lookahead         = grid->df_lookahead,
        .frames_per_call      = grid->frames_per_call,
        .erb_norm_tau_sec     = grid->erb_norm_tau_sec,
        .erb_norm_init_lo_db  = grid->erb_norm_init_lo_db,
        .erb_norm_init_hi_db  = grid->erb_norm_init_hi_db,
        .erb_norm_scale_db    = grid->erb_norm_scale_db,
        .spec_norm_tau_sec    = grid->spec_norm_tau_sec,
        .spec_norm_init_lo    = grid->spec_norm_init_lo,
        .spec_norm_init_hi    = grid->spec_norm_init_hi,
    };
    return spec;
}

/* 1 when the table's static buffers hold `grid`'s spectra and matrices. */
static inline int dfn_test_table_fits(const DFN2Grid *grid)
{
    return grid->n_bins <= DFN_TEST_TABLE_MAX_BINS && grid->n_erb <= DFN_TEST_TABLE_MAX_ERB;
}

/* The grid of this program as the library's value; built once, aborts on a grid the library
 * refuses (the test would be running something else). */
static inline const DFN2Grid *dfn_test_grid(void)
{
    static DFN2Grid grid;
    static int ready;
    if (!ready) {
        DFN2GridSpec spec = dfn_test_spec(DFN2_SR, DFN2_N_FFT, DFN2_N_ERB, DFN2_DF_BINS,
                                          DFN2_MASK_LOOKAHEAD, DFN2_DF_LOOKAHEAD);
        int status = dfn2_grid_make(&grid, &spec);
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
