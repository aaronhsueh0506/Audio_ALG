/* CLI and lifecycle/hop-loop skeleton shared by the four 4-channel samples; pipeline configs stay in the typed sample adapters. Sample code, not part of a delivered archive. */
#ifndef MCAB_SAMPLE_DRIVER_H
#define MCAB_SAMPLE_DRIVER_H

#include "mcab.h"
#include "mcab_sample_beam.h"
#include "mcab_sample_flow.h"
#include <stddef.h>

typedef struct McabSampleCliOption { const char *name; int takes_value; } McabSampleCliOption;
typedef int (*McabSampleCliSet)(void *user, const char *name, const char *value);
typedef int (*McabSampleCliUnknown)(void *user, const char *name);
/* Setters return 0 to continue, 1 for help, 2 for a rejected argument. */
int mcab_sample_scan_cli(int argc, char **argv, const McabSampleCliOption *options,
                  size_t count, void *user, McabSampleCliSet set,
                  McabSampleCliUnknown unknown);

typedef struct McabSampleSpatialOptions {
    const char *mic_path, *ref_path, *output_path, *vad_path;
    int sample_rate, fft_size;
    float uca_radius_m;
    int fixed_doa;
    float fixed_doa_rad;
    int print_mem_size;
} McabSampleSpatialOptions;
/* Numbers are parsed whole (example_cli_parse_int / _float): an empty or padded string, trailing
 * characters, a value outside int, inf or nan, and an option without its value return 2. raw=1
 * also checks the file options, the radius and the grid here; raw=0 defers those to the setup. */
int mcab_sample_parse_spatial_cli(int argc, char **argv, int raw,
                           McabSampleSpatialOptions *out);

typedef struct McabSampleHopView {
    const char *program;
    float *microphones, *reference, *output;
    int hop;
    int check_finite;
} McabSampleHopView;
typedef struct McabSampleOps {
    /* parse: 0=run, 1=help/exit success, 2=CLI error (the process exits 2). */
    int (*parse)(void *user, int argc, char **argv);
    /* setup: 0=run, 1=failure, 2=print-only success. */
    int (*setup)(void *user, McabSampleHopView *hop);
    /* read: 1=complete hop, 0=end, -1=failure. No hot-path allocations. */
    int (*read)(void *user, McabSampleHopView *hop, int frame);
    int (*process)(void *user, McabSampleHopView *hop, int frame);
    int (*consume)(void *user, McabSampleHopView *hop, int frame);
    void (*nonfinite)(void *user);
    /* Optional (NULL = success). */
    int (*finish)(void *user, McabSampleHopView *hop, int frames);
    void (*cleanup)(void *user);
    /* Optional success report after resource destruction. */
    void (*complete)(void *user);
} McabSampleOps;
int mcab_sample_main(int argc, char **argv, const McabSampleOps *ops, void *user);

/* The grid a requested fft_size (0 = the rate's default) resolves to. */
static inline int mcab_sample_resolved_fft(int sample_rate, int fft_size) {
    return fft_size ? fft_size : (sample_rate == 16000 ? 256 : 1024);
}
const char *mcab_sample_backend_name(uint32_t backend_id);
/* One pool the sample carves beside the unit (a post-beam chain, an NN stage). */
typedef struct McabSampleMemLine { const char *label; unsigned long long bytes; } McabSampleMemLine;
/* The unit's size: every object of the flow plus the beamformer helper (*total) and mcab's own
 * descriptor (*req). Passed in so this driver links without the module. */
typedef int (*McabSampleMemoryQuery)(const McabConfig *cfg, const McabSampleBeamParams *bp,
                                     McabMemReq *req, unsigned long long *total);
/* The query the samples pass: the lanes, estimator and mcab pool (mcab_sample_flow.h) and the
 * helper's pool (mcab_sample_beam.h), each 16-byte aligned. */
static inline int mcab_sample_unit_memory(const McabConfig *cfg, const McabSampleBeamParams *bp,
                                          McabMemReq *req, unsigned long long *total) {
    size_t flow_bytes, beam_bytes;
    if (mcab_get_mem_requirements(cfg, req) != 0 ||
        mcab_sample_flow_get_mem_size(cfg, &flow_bytes) != 0 ||
        mcab_sample_beam_get_mem_size(cfg, bp, &beam_bytes) != 0) return -1;
    *total = (unsigned long long)flow_bytes + beam_bytes;
    return 0;
}
/* Prints the unit's pool, the extra pools and their total, then mcab's descriptor. Returns -1
 * (message on stderr) when the config is invalid. */
int mcab_sample_print_mem_budget(const char *title, const McabConfig *cfg,
                                 const McabSampleBeamParams *bp, McabSampleMemoryQuery query,
                                 const McabSampleMemLine *extra, size_t n_extra);
#endif
