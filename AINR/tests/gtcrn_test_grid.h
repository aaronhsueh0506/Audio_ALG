/* The GTCRN grid and recurrent-state shapes the test programs run: the shipped model's
 * (config.ini [signal] n_fft, [model] erb_subband_1 / erb_subband_2; the state tensors of
 * export_onnx.py), pinned by GTCRN/tests/test_gtcrn_export_contract.py. The library knows
 * none of the shapes: a float host registers one buffer per GTCRNStateId and its element
 * count, here as slices of one flat array in graph order. Framing: ainr_test_hop.h. */

#ifndef GTCRN_TEST_GRID_H
#define GTCRN_TEST_GRID_H

#include <stdio.h>
#include <stdlib.h>

#if defined(__has_include)
#if __has_include("gtcrn_process.h")
#include "gtcrn_process.h"
#else
#include "GTCRN/gtcrn_process.h"
#endif
#else
#include "GTCRN/gtcrn_process.h"
#endif

#define GTCRN_TEST_SR              16000
#define GTCRN_TEST_N_FFT           512
#define GTCRN_TEST_HOP_LEN         256
#define GTCRN_TEST_N_BINS          257
#define GTCRN_TEST_ERB_LOW_BINS    65
#define GTCRN_TEST_ERB_HIGH_BANDS  64
#define GTCRN_TEST_ERB_BANDS       129
#define GTCRN_TEST_ERB_HIGH_BINS   192

/* conv_enc0..conv_dec2: [1, CONV_CHANNELS, window, CONV_FREQ] with the windows below;
 * h_tra: [TRA_GRUS, 1, TRA_HIDDEN]; h_dpgrnn: [DPGRNN_GRUS, DPGRNN_FREQ, DPGRNN_HIDDEN]. */
#define GTCRN_TEST_CONV_CHANNELS   16
#define GTCRN_TEST_CONV_FREQ       33
#define GTCRN_TEST_TRA_GRUS        6
#define GTCRN_TEST_TRA_HIDDEN      16
#define GTCRN_TEST_DPGRNN_GRUS     4
#define GTCRN_TEST_DPGRNN_FREQ     33
#define GTCRN_TEST_DPGRNN_HIDDEN   8

static const int gtcrn_test_conv_window[6] = {3, 5, 11, 11, 5, 3};

/* Element count of each state tensor, indexed by GTCRNStateId. */
static const size_t gtcrn_test_state_elements[GTCRN_STATE_COUNT] = {
    (size_t)GTCRN_TEST_CONV_CHANNELS * 3u * GTCRN_TEST_CONV_FREQ,
    (size_t)GTCRN_TEST_CONV_CHANNELS * 5u * GTCRN_TEST_CONV_FREQ,
    (size_t)GTCRN_TEST_CONV_CHANNELS * 11u * GTCRN_TEST_CONV_FREQ,
    (size_t)GTCRN_TEST_CONV_CHANNELS * 11u * GTCRN_TEST_CONV_FREQ,
    (size_t)GTCRN_TEST_CONV_CHANNELS * 5u * GTCRN_TEST_CONV_FREQ,
    (size_t)GTCRN_TEST_CONV_CHANNELS * 3u * GTCRN_TEST_CONV_FREQ,
    (size_t)GTCRN_TEST_TRA_GRUS * GTCRN_TEST_TRA_HIDDEN,
    (size_t)GTCRN_TEST_DPGRNN_GRUS * GTCRN_TEST_DPGRNN_FREQ * GTCRN_TEST_DPGRNN_HIDDEN
};

/* All eight tensors back to back. */
#define GTCRN_TEST_STATE_FLOATS 21216u

static inline GTCRNGridSpec gtcrn_test_grid_spec(void)
{
    GTCRNGridSpec spec = {
        .n_fft = GTCRN_TEST_N_FFT,
        .erb_low_bins = GTCRN_TEST_ERB_LOW_BINS,
        .erb_high_bands = GTCRN_TEST_ERB_HIGH_BANDS,
    };
    return spec;
}

/* Built once; aborts on a grid the library refuses. */
static inline const GTCRNGrid *gtcrn_test_grid(void)
{
    static GTCRNGrid grid;
    static int ready;
    if (!ready) {
        const GTCRNGridSpec spec = gtcrn_test_grid_spec();
        int status = gtcrn_grid_make(&grid, &spec);
        if (status != GTCRN_GRID_OK) {
            fprintf(stderr, "gtcrn_test_grid: %s\n", gtcrn_grid_status_text(status));
            exit(2);
        }
        ready = 1;
    }
    return &grid;
}

/* Register the eight tensors as consecutive slices of `flat` (GTCRN_TEST_STATE_FLOATS floats). */
static inline void gtcrn_test_register(GTCRNProcessConfig *cfg, float *flat)
{
    size_t offset = 0;
    for (int i = 0; i < GTCRN_STATE_COUNT; ++i) {
        cfg->state[i] = flat + offset;
        cfg->state_elements[i] = gtcrn_test_state_elements[i];
        offset += gtcrn_test_state_elements[i];
    }
}

#endif /* GTCRN_TEST_GRID_H */
