/* The DeepFilterNet2 grid a test program runs.
 *
 * The library takes its grid at run time (DFN2Grid, dfn2_grid.h) and holds no grid of its
 * own, so a test program decides one. It is chosen when the program is built, which lets
 * one test source cover every grid and lookahead pair and lets the buffers be plain arrays:
 *
 *   -DDFN2_GRID_KHZ=16 | 32 | 48          a preset below (48 when nothing is chosen)
 *   -DDFN2_SR=.. -DDFN2_N_FFT=.. -DDFN2_N_ERB=.. -DDFN2_DF_BINS=..
 *                                         a grid of another model (all four, no preset)
 *   -DDFN2_MASK_LOOKAHEAD=m -DDFN2_DF_LOOKAHEAD=d
 *                                         the lookahead pair (1/1 full, 1/0 mask-only)
 *
 *      preset   sr      n_fft  hop   bins  n_erb  df_bins
 *      48       48000   1024   512   513   32     96
 *      32       32000   512    256   257   28     96     (not trained yet)
 *      16       16000   256    128   129   24     96
 *
 * dfn2_test_grid() hands the library this grid as a DFN2Grid and the DFN2_* macros size the
 * test's own buffers; nothing here is read by the library. The order, the normaliser
 * settings and the recurrent-state shapes are the shipped model's (config.ini [model],
 * [feature]; export_onnx.py), pinned by tests/test_dfn2_contract.py. The recurrent state
 * belongs to the application: a float host registers DFN2_TEST_GRU_ELEMENTS and
 * DFN2_TEST_CONVP_ELEMENTS floats per instance. Framing is the test's own: ainr_test_hop.h. */

#ifndef DFN2_TEST_GRID_H
#define DFN2_TEST_GRID_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__has_include)
#if __has_include("dfn2_prepost.h")
#include "dfn2_prepost.h"
#else
#include "DeepFilterNet2/dfn2_prepost.h"
#endif
#else
#include "DeepFilterNet2/dfn2_prepost.h"
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
/* config.ini [model] df_order and [feature]: the shipped model's values, a test table that
 * tests/test_dfn2_contract.py pins to the files. */
#define DFN2_DF_ORDER                  5

/* Shapes of the exported graph's recurrent state (export_onnx.py): h_gru is
 * [1, GRU_ROWS, 1, GRU_HIDDEN] (encoder row 0, ERB rows from 1, DF rows from 3) and
 * df_convp_history is [1, CONVP_CHANNELS, CONVP_WINDOW, df_bins]; the mask-only graph has
 * the first three rows and no pathway window. */
#define DFN2_TEST_GRU_ROW_ENCODER      0
#define DFN2_TEST_GRU_ROW_ERB          1
#define DFN2_TEST_GRU_HIDDEN           256
#define DFN2_TEST_CONVP_CHANNELS       64
#ifdef DFN2_MASK_ONLY
#define DFN2_TEST_GRU_ROWS             3
#define DFN2_TEST_CONVP_WINDOW         0
#else
#define DFN2_TEST_GRU_ROW_DF           3
#define DFN2_TEST_GRU_ROWS             5
#define DFN2_TEST_CONVP_WINDOW         5
#endif
#define DFN2_TEST_GRU_ELEMENTS         ((size_t)DFN2_TEST_GRU_ROWS * DFN2_TEST_GRU_HIDDEN)
#define DFN2_TEST_CONVP_ELEMENTS \
    ((size_t)DFN2_TEST_CONVP_CHANNELS * DFN2_TEST_CONVP_WINDOW * (DFN2_DF_BINS))

#define DFN2_DF_HISTORY     (DFN2_DF_ORDER - DFN2_DF_LOOKAHEAD - 1)
#define DFN2_DF_RING_NOISY  (DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD + 1)
#ifdef DFN2_MASK_ONLY
#define DFN2_DF_RING        DFN2_DF_RING_NOISY
#else
#define DFN2_DF_RING \
    (DFN2_DF_ORDER > DFN2_DF_RING_NOISY ? DFN2_DF_ORDER : DFN2_DF_RING_NOISY)
#endif

/* analysis and synthesis scale of the framing, the float32 values the library derives */
#define DFN2_NORMALIZED_STFT_SCALE    (1.0f / sqrtf((float)(DFN2_N_FFT)))
#define DFN2_NORMALIZED_STFT_UNSCALE  sqrtf((float)(DFN2_N_FFT))
#define DFN2_ANALYSIS_SCALE           DFN2_NORMALIZED_STFT_SCALE

/* The spec of this program's grid, one frame per call; a test that varies a field edits the
 * copy before dfn2_grid_make(). */
static inline DFN2GridSpec dfn2_test_grid_spec(void)
{
    DFN2GridSpec spec = {
        .sample_rate = DFN2_SR,
        .n_fft = DFN2_N_FFT,
        .n_erb = DFN2_N_ERB,
        .df_bins = DFN2_DF_BINS,
        .df_order = DFN2_DF_ORDER,
        .mask_lookahead = DFN2_MASK_LOOKAHEAD,
        .df_lookahead = DFN2_DF_LOOKAHEAD,
        .frames_per_call = 1,
    };
    return spec;
}

/* The grid of this program as the library's value; built once, aborts on a grid the library
 * refuses (the test would be running something else). */
static inline const DFN2Grid *dfn2_test_grid(void)
{
    static DFN2Grid grid;
    static int ready;
    if (!ready) {
        const DFN2GridSpec spec = dfn2_test_grid_spec();
        int status = dfn2_grid_make(&grid, &spec);
        if (status != DFN2_GRID_OK) {
            fprintf(stderr, "dfn2_test_grid: %s\n", dfn2_grid_status_text(status));
            exit(2);
        }
        ready = 1;
    }
    return &grid;
}

/* Band borders of the triangular ERB filters the test programs build their reference matrices
 * from: DFN2_N_ERB entries, from bin 0 to DFN2_N_BINS, each band at least two bins wide. The
 * shipped 48 kHz geometry (1024-point frames, 32 bands) uses its own border table. Any other
 * grid gets widths that grow with the band, in integer arithmetic so the table is the same on
 * every platform: 2 * b plus a quadratic share of the remaining bins, which ends exactly on
 * DFN2_N_BINS (dfn2_grid_make guarantees 2 * n_erb <= n_bins, so the share is not negative).
 * `borders` has room for DFN2_N_ERB ints. */
static inline void dfn2_test_band_borders(int *borders)
{
#if DFN2_N_BINS == 513 && DFN2_N_ERB == 32
    static const int shipped[32] = {
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 30, 36, 42,
        50, 59, 69, 81, 94, 110, 129, 151, 176, 205, 239, 279,
        325, 378, 440, 513
    };
    memcpy(borders, shipped, sizeof(shipped));
#else
    const int last = DFN2_N_ERB - 1;
    for (int b = 0; b <= last; ++b)
        borders[b] = 2 * b + (int)((long long)b * b *
                                   (DFN2_N_BINS - 2 * last) / ((long long)last * last));
#endif
}

#endif /* DFN2_TEST_GRID_H */
