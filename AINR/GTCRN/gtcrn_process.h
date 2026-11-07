/* GTCRN pre/post-processing around the graph, for the GTCRNGrid in its config. Per hop:
 * model_input, graph, mask_validate, state_validate, model_output (pass the spectrum through on
 * -1). Spectra in, spectrum out: the STFT / ISTFT are audiolib's (AcStft / AcIstft), configured
 * by ac_stft_config_default(&sc, grid.n_fft) to GTCRN's training framing: periodic
 * sqrt-Hann, center 0, hop = grid.hop = n_fft / 2, spec_scale = ispec_scale = 1.0f
 * (unnormalised, not torch normalized=True), ola = AC_OLA_ACCUMULATE. One frame per graph call
 * (its history is in the state tensors): do not pad it to several frames. */

#ifndef GTCRN_PROCESS_H
#define GTCRN_PROCESS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Layout of the graph's state tensors (export_onnx.py). Layout versions 3-7 are never reused. */
#define GTCRN_MODEL_LAYOUT_VERSION 8

/* Bumped on any change of pool size, carve order or MemReq content; CARVE enters the hash. */
#define GTCRN_PROCESS_DESCRIPTOR_VERSION 1u
#define GTCRN_PROCESS_CARVE_VERSION      1u

#define GTCRN_PROCESS_ALIGNMENT 16u

/* The eight state tensors belong to the application: each state input and its *_out output
 * (same shape) are bound to one buffer (read every state input before writing any state output),
 * zero-loaded (quantised) at start and whenever gtcrn_process_state_generation() changes. */
typedef enum GTCRNStateId {   /* graph order */
    GTCRN_STATE_CONV_ENC0 = 0,
    GTCRN_STATE_CONV_ENC1,
    GTCRN_STATE_CONV_ENC2,
    GTCRN_STATE_CONV_DEC0,
    GTCRN_STATE_CONV_DEC1,
    GTCRN_STATE_CONV_DEC2,
    GTCRN_STATE_H_TRA,
    GTCRN_STATE_H_DPGRNN,
    GTCRN_STATE_COUNT     /* the graph's tensor count, not a capacity */
} GTCRNStateId;

typedef struct GTCRNGridSpec {   /* what an exported model fixes (config.ini) */
    int n_fft;            /* frame size = window length                                   */
    int erb_low_bins;     /* erb_subband_1: the lowest bins, passed through unbanded       */
    int erb_high_bands;   /* erb_subband_2: the ERB bands the remaining bins compress into */
} GTCRNGridSpec;

typedef struct GTCRNGrid {
    int n_fft;
    int hop;              /* n_fft / 2                                   */
    int n_bins;           /* n_fft / 2 + 1                               */
    int erb_low_bins;
    int erb_high_bands;
    int erb_bands;        /* erb_low_bins + erb_high_bands: the model's F */
    int erb_high_bins;    /* n_bins - erb_low_bins                       */
} GTCRNGrid;

enum {
    GTCRN_GRID_OK = 0,
    GTCRN_GRID_ERR_NULL = -1,
    GTCRN_GRID_ERR_N_FFT = -2,
    GTCRN_GRID_ERR_ERB_LOW_BINS = -3,
    GTCRN_GRID_ERR_ERB_HIGH_BANDS = -4,
    GTCRN_GRID_ERR_INCONSISTENT = -5
};

/* Derive a grid: GTCRN_GRID_OK (written) or the first failing check (grid untouched): n_fft
 * accepted by audiolib, erb_low_bins in [0, n_bins - 1], erb_high_bands in [1, erb_high_bins]. */
int gtcrn_grid_make(GTCRNGrid* grid, const GTCRNGridSpec* spec);

int gtcrn_grid_validate(const GTCRNGrid* grid);   /* make's checks, derived fields included */

const char* gtcrn_grid_status_text(int status);   /* for a log line; never NULL */

typedef struct GTCRNProcessConfig {
    GTCRNGrid    grid;          /* gtcrn_grid_make(); copied at init                         */
    /* Required, borrowed, never copied; the library never touches the filesystem. */
    const float* erb_fwd;       /* bin-major  [erb_high_bins][erb_high_bands] */
    const float* erb_inv;       /* band-major [erb_high_bands][erb_high_bins] */
    /* Optional, for a host working on float copies: borrowed (pointer, element count) pairs by
     * GTCRNStateId. Either all eight pairs are set or none; a pointer with a count of 0 and a
     * count with a NULL pointer are refused. Not part of the pool size or the hash. */
    float*       state[GTCRN_STATE_COUNT];
    size_t       state_elements[GTCRN_STATE_COUNT];   /* elements, not bytes */
} GTCRNProcessConfig;

typedef struct GTCRNProcessMemReq {
    uint32_t descriptor_version;  /* GTCRN_PROCESS_DESCRIPTOR_VERSION                         */
    uint32_t layout_version;      /* GTCRN_MODEL_LAYOUT_VERSION                               */
    uint32_t reserved0;           /* 0                                                        */
    uint32_t build_flags_hash;    /* FNV-1a-32 over the whole grid, layout and carve versions */
    uint32_t alignment;
    uint32_t reserved1;           /* 0                                                        */
    uint64_t bytes;
} GTCRNProcessMemReq;

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ >= 201112L
_Static_assert(sizeof(GTCRNProcessMemReq) == 32,
               "GTCRNProcessMemReq must stay 32 bytes");
#endif
#endif

/* Not reentrant: one instance per stream, and model_input / model_output of one instance share
 * its scratch. */
typedef struct GTCRNProcess GTCRNProcess;

/* Config for `grid` (copied), matrices and state NULL / 0. Returns 0, or -1 on NULL or bad grid. */
int gtcrn_process_config_defaults(GTCRNProcessConfig* cfg, const GTCRNGrid* grid);

/* Exact pool size; -1 (req untouched) on NULL, a bad grid, a missing matrix or bad registration. */
int gtcrn_process_get_mem_size(const GTCRNProcessConfig* cfg, GTCRNProcessMemReq* req);

/* Construct in `pool` (16-byte aligned, `bytes` >= req.bytes); NULL, nothing written, on a bad
 * config or pool. Zeroes the pool and registered state and starts the generation at 1; an
 * instance in use goes through gtcrn_process_reset(). The caller zeroes the accelerator's state
 * once. */
GTCRNProcess* gtcrn_process_init(void* pool, size_t bytes, const GTCRNProcessConfig* cfg);

/* As _init; NULL when a non-NULL `expected` differs from the config's own requirement. */
GTCRNProcess* gtcrn_process_init_ex(void* pool, size_t bytes, const GTCRNProcessConfig* cfg,
                                    const GTCRNProcessMemReq* expected);

/* Zero the registered state and advance the generation; config and pool untouched. NULL ignored. */
void gtcrn_process_reset(GTCRNProcess* p);

/* State generation: 1 after init, +1 per reset and per refused validation (mod 2^32); 0 for NULL.
 * The generation is kept whether or not a state is registered. */
unsigned int gtcrn_process_state_generation(const GTCRNProcess* p);

const GTCRNGrid* gtcrn_process_grid(const GTCRNProcess* p);   /* lives as long as the instance */

/* Replace the ERB matrices (same layouts as the config), borrowed. Returns 0, or -1 on NULL. */
int gtcrn_process_set_erb_matrices(GTCRNProcess* p, const float* erb_fwd, const float* erb_inv);

/* [sqrt(re^2 + im^2 + 1e-12), re, im] of one analysis frame, each banded through the forward
 * matrix (low bins pass through). `spectrum` is interleaved [n_bins][2]; mag, real_part and
 * imag_part are erb_bands floats each, not overlapping it. A NULL argument is a no-op. */
void gtcrn_model_input(GTCRNProcess* p, const float* spectrum, float* mag, float* real_part,
                       float* imag_part);

/* Expand the graph's ERB-domain complex mask (interleaved [erb_bands][2]) through the inverse
 * matrix and apply it to the frame the inputs came from. `enhanced` is interleaved [n_bins][2]
 * and may be the spectrum itself. The mask is checked by mask_validate, not here. A NULL
 * argument is a no-op. */
void gtcrn_model_output(GTCRNProcess* p, const float* mask_erb, const float* spectrum,
                        float* enhanced);

/* Finite-check all eight registered buffers in place after an invocation. Returns 0 when all
 * are finite or nothing is registered, -1 on NULL. A NaN or Inf anywhere zeroes all eight
 * buffers, advances the generation and returns -1; a runtime that holds the state itself
 * refills it when the generation changes. A finite partial write is not detectable. */
int gtcrn_process_state_validate(GTCRNProcess* p);

/* Finite-check the ERB-domain mask (interleaved [erb_bands][2]) after an invocation, state
 * registered or not. A NaN or Inf refuses as state_validate does (-1: pass the spectrum through
 * instead of gtcrn_model_output()); finite: 0; NULL: -1. Call it before
 * gtcrn_process_state_validate() (the pair then advances the generation at most once). To detect
 * a mask the accelerator did not write, fill it with NaN first. */
int gtcrn_process_mask_validate(GTCRNProcess* p, const float* mask_erb);

const char* gtcrn_simd_backend(void);   /* kernel backend name, for a log line */

#ifdef __cplusplus
}
#endif

#endif /* GTCRN_PROCESS_H */
