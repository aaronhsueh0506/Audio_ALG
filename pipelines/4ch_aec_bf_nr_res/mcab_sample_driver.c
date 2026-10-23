#include "mcab_sample_driver.h"

#include <stdio.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const char* mcab_backend_id_name(uint32_t backend_id) {
    switch (backend_id) {
        case FOUR_AEC_NR_RES_BACKEND_NE10: return "ne10";
        default:                           return "unknown";
    }
}

int mcab_print_mem_budget(const char* title, const AudioPipeline4ChConfig* cfg,
                          McabMemoryQuery query,
                          const McabMemLine* extra, size_t n_extra) {
    AudioPipeline4ChMemReq req;
    unsigned long long total;
    size_t i;
    if (query(cfg, &req) != 0) {
        fprintf(stderr, "Error: invalid 4ch spatial pipeline config\n");
        return -1;
    }
    total = (unsigned long long)req.bytes;
    for (i = 0; i < n_extra; ++i) total += extra[i].bytes;
    printf("Memory Budget (%s)\n", title);
    printf("=============================================================\n");
    /* cfg->core.fft_size may be the unresolved 0 sentinel; print the grid actually used. */
    printf("  sample_rate=%d fft=%d geometry=%d num_angles=%d\n",
           cfg->core.sample_rate,
           mcab_resolved_fft(cfg->core.sample_rate, cfg->core.fft_size),
           (int)cfg->geometry, cfg->num_angles);
    printf("  Unit:           %9llu bytes (%7.1f KB)\n",
           (unsigned long long)req.bytes, (float)req.bytes / 1024.0f);
    for (i = 0; i < n_extra; ++i)
        printf("  %-14s  %9llu bytes (%7.1f KB)\n", extra[i].label,
               extra[i].bytes, (float)extra[i].bytes / 1024.0f);
    printf("  Total:          %9llu bytes (%7.1f KB)\n",
           total, (float)total / 1024.0f);
    printf("  Descriptor:     descriptor_version=%u alignment=%u "
           "layout_version=%u backend_id=%u (%s) "
           "build_flags_hash=0x%08x\n\n",
           req.descriptor_version, req.alignment, req.layout_version,
           req.backend_id, mcab_backend_id_name(req.backend_id),
           req.build_flags_hash);
    return 0;
}

int mcab_scan_cli(int argc, char **argv, const McabCliOption *options,
                  size_t count, void *user, McabCliSet set,
                  McabCliUnknown unknown) {
    for (int i = 1; i < argc; ++i) {
        size_t j;
        for (j = 0; j < count; ++j)
            if (strcmp(argv[i], options[j].name) == 0) break;
        if (j == count || (options[j].takes_value && i + 1 >= argc))
            return unknown(user, argv[i]);
        const char *name = argv[i];
        const char *value = options[j].takes_value ? argv[++i] : NULL;
        int rc = set(user, name, value);
        if (rc != 0) return rc;
    }
    return 0;
}

static int mcab_raw_int(const char *text, int *value) {
    char *end = NULL;
    errno = 0;
    long parsed = strtol(text, &end, 10);
    if (errno || !end || *end != '\0' || parsed < -2147483647L ||
        parsed > 2147483647L) return 0;
    *value = (int)parsed;
    return 1;
}
static int mcab_raw_float(const char *text, float *value) {
    char *end = NULL;
    errno = 0;
    float parsed = strtof(text, &end);
    if (errno || !end || *end != '\0' || !isfinite(parsed)) return 0;
    *value = parsed;
    return 1;
}
typedef struct SpatialCli { McabSpatialOptions *out; int raw; } SpatialCli;
static int mcab_spatial_set(void *user, const char *name, const char *value) {
    SpatialCli *cli = user;
    McabSpatialOptions *o = cli->out;
    if (!strcmp(name, "--print-mem-size")) o->print_mem_size = 1;
    else if (!strcmp(name, "--mic-raw")) o->mic_path = value;
    else if (!strcmp(name, "--ref-raw")) o->ref_path = value;
    else if (!strcmp(name, "--output-raw")) o->output_path = value;
    else if (!strcmp(name, "--vad-u8")) o->vad_path = value;
    else if (!strcmp(name, "--sample-rate")) {
        if (cli->raw) { if (!mcab_raw_int(value, &o->sample_rate)) return 2; }
        else o->sample_rate = atoi(value);
    } else if (!strcmp(name, "--fft-size")) {
        if (cli->raw) { if (!mcab_raw_int(value, &o->fft_size)) return 2; }
        else o->fft_size = atoi(value);
    } else {
        float number;
        if (cli->raw) { if (!mcab_raw_float(value, &number)) return 2; }
        else number = (float)atof(value);
        if (!strcmp(name, "--uca-radius-m")) o->uca_radius_m = number;
        else {
            o->fixed_doa = 1;
            o->fixed_doa_rad = number * (float)(M_PI / 180.0);
        }
    }
    return 0;
}
static int mcab_spatial_unknown(void *user, const char *name) {
    (void)user; (void)name;
    return 2;
}
int mcab_parse_spatial_cli(int argc, char **argv, int raw, McabSpatialOptions *out) {
    static const McabCliOption spatial[] = {
        {"--print-mem-size",0}, {"--sample-rate",1}, {"--fft-size",1},
        {"--uca-radius-m",1}, {"--fixed-doa-deg",1},
        {"--mic-raw",1}, {"--ref-raw",1}, {"--output-raw",1}, {"--vad-u8",1}
    };
    memset(out, 0, sizeof(*out));
    out->sample_rate = raw ? 0 : 16000;
    out->uca_radius_m = 0.035f;
    SpatialCli cli = {out, raw};
    if (mcab_scan_cli(argc, argv, spatial, raw ? 9 : 5, &cli,
                      mcab_spatial_set, mcab_spatial_unknown) != 0) return 2;
    if (!raw) return 0;
    if (!out->print_mem_size && (!out->mic_path || !out->ref_path ||
                                 !out->output_path)) return 2;
    if (!isfinite(out->uca_radius_m) || out->uca_radius_m <= 0.0f) return 2;
    if (out->sample_rate == 16000)
        return out->fft_size == 0 || out->fft_size == 256 || out->fft_size == 512 ? 0 : 2;
    if (out->sample_rate == 48000)
        return out->fft_size == 0 || out->fft_size == 1024 ? 0 : 2;
    return 2;
}

const char *mcab_delay_mode_name(AecDelayMode mode) {
    switch (mode) {
        case AEC_DELAY_MATCHED: return "matched";
        case AEC_DELAY_FIXED: return "fixed";
        case AEC_DELAY_EXTERNAL_ALIGNED: return "external";
        default: return "?";
    }
}
int mcab_set_delay_option(McabDelayOptions *o, const char *name,
                           const char *value, int strict, const char *title) {
    if (!strcmp(name, "--delay-mode")) {
        if (!strcmp(value,"matched")) o->mode = AEC_DELAY_MATCHED;
        else if (!strcmp(value,"fixed")) o->mode = AEC_DELAY_FIXED;
        else if (!strcmp(value,"external")) o->mode = AEC_DELAY_EXTERNAL_ALIGNED;
        else {
            fprintf(stderr,"%s: --delay-mode '%s' is not a mode "
                    "(accepted: matched|fixed|external)\n",title,value);
            return 2;
        }
    } else {
        int number;
        if (strict) {
            char *end = NULL;
            errno = 0;
            long parsed = strtol(value,&end,10);
            if (errno || end == value || *end != '\0' ||
                parsed < INT_MIN || parsed > INT_MAX) {
                if (!strcmp(name,"--delay-num-filters"))
                    fprintf(stderr,"%s: --delay-num-filters '%s' is not an "
                            "integer (accepted: 1..%d)\n",title,value,DA_NUM_FILTERS);
                else fprintf(stderr,"%s: --fixed-delay '%s' is not an integer "
                            "(accepted: native-rate samples >= 0)\n",title,value);
                return 2;
            }
            number = (int)parsed;
        } else number = atoi(value);
        if (!strcmp(name,"--delay-num-filters")) {
            o->num_filters=number; o->have_num_filters=1;
        } else { o->fixed_samples=number; o->have_fixed=1; }
    }
    return 0;
}
int mcab_validate_delay_options(McabDelayOptions *o, const char *title) {
    if (o->have_fixed && o->mode != AEC_DELAY_FIXED) {
        fprintf(stderr,"%s: --fixed-delay %d is only valid with "
                "--delay-mode fixed (requested mode: %s)\n",title,
                o->fixed_samples,mcab_delay_mode_name(o->mode)); return 2;
    }
    if (o->mode == AEC_DELAY_FIXED && (!o->have_fixed || o->fixed_samples < 0)) {
        fprintf(stderr,"%s: --delay-mode fixed needs --fixed-delay "
                "<samples> >= 0 (requested: %d)\n",title,
                o->have_fixed ? o->fixed_samples : -1); return 2;
    }
    if (o->mode == AEC_DELAY_MATCHED) {
        if (o->num_filters < 1 || o->num_filters > DA_NUM_FILTERS) {
            fprintf(stderr,"%s: --delay-num-filters %d is out of range "
                    "(accepted: 1..%d; 0 does not mean 'off' -- use "
                    "--delay-mode fixed or external instead)\n",title,
                    o->num_filters,DA_NUM_FILTERS); return 2;
        }
    } else {
        if (o->have_num_filters && o->num_filters != DA_NUM_FILTERS) {
            fprintf(stderr,"%s: --delay-num-filters %d is only valid with "
                    "--delay-mode matched (requested mode: %s, which builds "
                    "no matched bank and requires n == %d)\n",title,
                    o->num_filters,mcab_delay_mode_name(o->mode),DA_NUM_FILTERS);
            return 2;
        }
        o->num_filters = DA_NUM_FILTERS;
    }
    return 0;
}

int mcab_sample_main(int argc, char **argv, const McabSampleOps *ops, void *user) {
    McabHopView hop = {0};
    hop.program = argv[0];
    int rc = ops->parse(user, argc, argv);
    if (rc != 0) return rc == 1 ? 0 : ops->cli_error_status;
    int result = 1, frames = 0;
    rc = ops->setup(user, &hop);
    if (rc == 2) { result = 0; goto cleanup; }
    if (rc != 0) goto cleanup;
    for (;;) {
        rc = ops->read(user, &hop, frames);
        if (rc < 0) goto cleanup;
        if (rc == 0) break;
        if (ops->process(user, &hop, frames) != 0) goto cleanup;
        if (hop.check_finite) {
            for (int i = 0; i < hop.hop; ++i) {
                if (!isfinite(hop.output[i])) {
                    if (ops->nonfinite) ops->nonfinite(user);
                    goto cleanup;
                }
            }
        }
        if (ops->consume && ops->consume(user, &hop, frames) != 0) goto cleanup;
        ++frames;
    }
    result = ops->finish ? ops->finish(user, &hop, frames) : 0;
cleanup:
    ops->cleanup(user);
    if (result == 0 && ops->complete) ops->complete(user);
    return result;
}
