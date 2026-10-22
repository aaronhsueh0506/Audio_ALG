/* CLI and lifecycle/hop-loop skeleton shared by the four 4-channel samples.
 * Pipeline configs and model resources stay in their typed sample adapters.
 * This host/sample code is not part of a delivered algorithm archive. */
#ifndef MCAB_SAMPLE_DRIVER_H
#define MCAB_SAMPLE_DRIVER_H

#include "audio_pipeline_4ch.h"
#include <stddef.h>

typedef struct McabCliOption { const char *name; int takes_value; } McabCliOption;
typedef int (*McabCliSet)(void *user, const char *name, const char *value);
typedef int (*McabCliUnknown)(void *user, const char *name);
/* Setters return 0 to continue, 1 for help, 2 for a rejected argument. */
int mcab_scan_cli(int argc, char **argv, const McabCliOption *options,
                  size_t count, void *user, McabCliSet set,
                  McabCliUnknown unknown);

typedef struct McabSpatialOptions {
    const char *mic_path, *ref_path, *output_path, *vad_path;
    int sample_rate, fft_size;
    float uca_radius_m;
    int fixed_doa;
    float fixed_doa_rad;
    int print_mem_size;
} McabSpatialOptions;
/* raw=1 preserves raw-runner numeric/grid validation; raw=0 preserves the
 * static sample's legacy atoi/atof conversion and deferred config checks. */
int mcab_parse_spatial_cli(int argc, char **argv, int raw,
                           McabSpatialOptions *out);

typedef struct McabDelayOptions {
    AecDelayMode mode;
    int num_filters, fixed_samples;
    int have_num_filters, have_fixed;
} McabDelayOptions;
int mcab_set_delay_option(McabDelayOptions *out, const char *name,
                           const char *value, int strict, const char *title);
int mcab_validate_delay_options(McabDelayOptions *out, const char *title);
const char *mcab_delay_mode_name(AecDelayMode mode);

typedef struct McabHopView {
    const char *program;
    float *microphones, *reference, *output;
    int hop;
    int check_finite;
} McabHopView;
typedef struct McabSampleOps {
    /* parse: 0=run, 1=help/exit success, 2=CLI error. */
    int (*parse)(void *user, int argc, char **argv);
    int cli_error_status;
    /* setup: 0=run, 1=failure, 2=print-only success. */
    int (*setup)(void *user, McabHopView *hop);
    /* read: 1=complete hop, 0=end, -1=failure. No hot-path allocations. */
    int (*read)(void *user, McabHopView *hop, int frame);
    int (*process)(void *user, McabHopView *hop, int frame);
    int (*consume)(void *user, McabHopView *hop, int frame);
    void (*nonfinite)(void *user);
    int (*finish)(void *user, McabHopView *hop, int frames);
    void (*cleanup)(void *user);
    /* Optional success report after resource destruction. */
    void (*complete)(void *user);
} McabSampleOps;
int mcab_sample_main(int argc, char **argv, const McabSampleOps *ops, void *user);

const char *mcab_backend_id_name(uint32_t backend_id);
typedef int (*McabMemoryQuery)(const AudioPipeline4ChConfig *, AudioPipeline4ChMemReq *);
int mcab_print_mem_budget(const char *title, const AudioPipeline4ChConfig *cfg,
                          McabMemoryQuery query);
#endif
