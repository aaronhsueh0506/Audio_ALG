/* CLI and lifecycle/hop-loop skeleton shared by the four 4-channel samples; pipeline configs stay in the typed sample adapters. Sample code, not part of a delivered archive. */
#ifndef ARRAY_SAMPLE_DRIVER_H
#define ARRAY_SAMPLE_DRIVER_H

#include "aec.h"
#include "array_sample_beam.h"
#include "utility/mem_align.h"
#include <stddef.h>
#include <stdint.h>

typedef struct ArraySampleCliOption { const char *name; int takes_value; } ArraySampleCliOption;
typedef int (*ArraySampleCliSet)(void *user, const char *name, const char *value);
typedef int (*ArraySampleCliUnknown)(void *user, const char *name);
/* Setters return 0 to continue, 1 for help, 2 for a rejected argument. A value option at the end
 * of the line, or followed by a token starting with "--", goes to `unknown` with the option name. */
int array_sample_scan_cli(int argc, char **argv, const ArraySampleCliOption *options,
                  size_t count, void *user, ArraySampleCliSet set,
                  ArraySampleCliUnknown unknown);

typedef struct ArraySampleSpatialOptions {
    const char *mic_path, *ref_path, *output_path, *vad_path;
    int sample_rate, fft_size;
    float uca_radius_m;
    int fixed_doa;
    float fixed_doa_rad;
    int print_mem_size;
} ArraySampleSpatialOptions;
/* Numbers are parsed whole (example_cli_parse_int / _float): an empty or padded string, trailing
 * characters, a value outside int, inf or nan, and an option without its value (the line ends, or
 * the next token starts with "--") return 2. raw=1
 * also checks the file options, the radius and the grid here; raw=0 defers those to the setup. */
int array_sample_parse_spatial_cli(int argc, char **argv, int raw,
                           ArraySampleSpatialOptions *out);

typedef struct ArraySampleHopView {
    const char *program;
    float *microphones, *reference, *output;
    int hop;
    int check_finite;
} ArraySampleHopView;
typedef struct ArraySampleOps {
    /* parse: 0=run, 1=help/exit success, 2=CLI error (the process exits 2). */
    int (*parse)(void *user, int argc, char **argv);
    /* setup: 0=run, 1=failure, 2=print-only success. */
    int (*setup)(void *user, ArraySampleHopView *hop);
    /* read: 1=complete hop, 0=end, -1=failure. No hot-path allocations. */
    int (*read)(void *user, ArraySampleHopView *hop, int frame);
    int (*process)(void *user, ArraySampleHopView *hop, int frame);
    int (*consume)(void *user, ArraySampleHopView *hop, int frame);
    void (*nonfinite)(void *user);
    /* Optional (NULL = success). */
    int (*finish)(void *user, ArraySampleHopView *hop, int frames);
    void (*cleanup)(void *user);
    /* Optional success report after resource destruction. */
    void (*complete)(void *user);
} ArraySampleOps;
int array_sample_main(int argc, char **argv, const ArraySampleOps *ops, void *user);

const char *array_sample_backend_name(uint32_t backend_id);
/* One pool the sample carves beside the unit (a post-beam chain, an NN stage). */
typedef struct ArraySampleMemLine { const char *label; unsigned long long bytes; } ArraySampleMemLine;
/* The unit's size: the array pool plus the beamformer helper (*total) and the array's descriptor
 * (*req). Passed in so this driver links without the algorithm libraries. */
typedef int (*ArraySampleMemoryQuery)(const AecConfig *cfg, const ArraySampleBeamParams *bp,
                                     AecMemReq *req, unsigned long long *total);
/* The query the samples pass: the array pool (aec_get_mem_requirements()) and the helper's pool
 * (array_sample_beam.h), each 16-byte aligned. */
static inline int array_sample_unit_memory(const AecConfig *cfg, const ArraySampleBeamParams *bp,
                                          AecMemReq *req, unsigned long long *total) {
    size_t beam_bytes;
    if (aec_get_mem_requirements(cfg, req) != 0 ||
        array_sample_beam_get_mem_size(cfg, bp, &beam_bytes) != 0) return -1;
    *total = (unsigned long long)ALIGN16((size_t)req->bytes) + beam_bytes;
    return 0;
}
/* Prints the unit's pool, the extra pools and their total, then the array's descriptor. Returns -1
 * (message on stderr) when the config is invalid. */
int array_sample_print_mem_budget(const char *title, const AecConfig *cfg,
                                 const ArraySampleBeamParams *bp, ArraySampleMemoryQuery query,
                                 const ArraySampleMemLine *extra, size_t n_extra);
#endif
