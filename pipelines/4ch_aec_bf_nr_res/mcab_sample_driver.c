#include "mcab_sample_driver.h"
#include "common/example_cli.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const char* mcab_sample_backend_name(uint32_t backend_id) {
    switch (backend_id) {
        case MCAB_BACKEND_NE10: return "ne10";
        default:                return "unknown";
    }
}

int mcab_sample_print_mem_budget(const char* title, const McabConfig* cfg,
                                 const McabSampleBeamParams* bp, McabSampleMemoryQuery query,
                                 const McabSampleMemLine* extra, size_t n_extra) {
    McabMemReq req;
    unsigned long long unit_bytes, total;
    size_t i;
    if (query(cfg, bp, &req, &unit_bytes) != 0) {
        fprintf(stderr, "Error: invalid 4ch spatial pipeline config\n");
        return -1;
    }
    total = unit_bytes;
    for (i = 0; i < n_extra; ++i) total += extra[i].bytes;
    printf("Memory Budget (%s)\n", title);
    printf("=============================================================\n");
    /* cfg->fft_size may be the unresolved 0 sentinel; print the grid actually used. */
    printf("  sample_rate=%d fft=%d geometry=%d num_angles=%d\n",
           cfg->sample_rate,
           mcab_sample_resolved_fft(cfg->sample_rate, cfg->fft_size),
           (int)bp->geometry, bp->num_angles);
    printf("  Unit:           %9llu bytes (%7.1f KB)\n",
           unit_bytes, (float)unit_bytes / 1024.0f);
    for (i = 0; i < n_extra; ++i)
        printf("  %-14s  %9llu bytes (%7.1f KB)\n", extra[i].label,
               extra[i].bytes, (float)extra[i].bytes / 1024.0f);
    printf("  Total:          %9llu bytes (%7.1f KB)\n",
           total, (float)total / 1024.0f);
    printf("  Descriptor:     descriptor_version=%u alignment=%u "
           "layout_version=%u backend_id=%u (%s) "
           "build_flags_hash=0x%08x\n\n",
           req.descriptor_version, req.alignment, req.layout_version,
           req.backend_id, mcab_sample_backend_name(req.backend_id),
           req.build_flags_hash);
    return 0;
}

int mcab_sample_scan_cli(int argc, char **argv, const McabSampleCliOption *options,
                  size_t count, void *user, McabSampleCliSet set,
                  McabSampleCliUnknown unknown) {
    for (int i = 1; i < argc; ++i) {
        size_t j;
        for (j = 0; j < count; ++j)
            if (strcmp(argv[i], options[j].name) == 0) break;
        /* An option name where the value belongs counts as a missing value. */
        if (j == count || (options[j].takes_value &&
                           (i + 1 >= argc || example_cli_is_option(argv[i + 1]))))
            return unknown(user, argv[i]);
        const char *name = argv[i];
        const char *value = options[j].takes_value ? argv[++i] : NULL;
        int rc = set(user, name, value);
        if (rc != 0) return rc;
    }
    return 0;
}

static int mcab_sample_spatial_set(void *user, const char *name, const char *value) {
    McabSampleSpatialOptions *o = user;
    if (!strcmp(name, "--print-mem-size")) o->print_mem_size = 1;
    else if (!strcmp(name, "--mic-raw")) o->mic_path = value;
    else if (!strcmp(name, "--ref-raw")) o->ref_path = value;
    else if (!strcmp(name, "--output-raw")) o->output_path = value;
    else if (!strcmp(name, "--vad-u8")) o->vad_path = value;
    else if (!strcmp(name, "--sample-rate")) {
        if (example_cli_parse_int(value, &o->sample_rate) != 0) return 2;
    } else if (!strcmp(name, "--fft-size")) {
        if (example_cli_parse_int(value, &o->fft_size) != 0) return 2;
    } else {
        float number;
        if (example_cli_parse_float(value, &number) != 0) return 2;
        if (!strcmp(name, "--uca-radius-m")) o->uca_radius_m = number;
        else {
            o->fixed_doa = 1;
            o->fixed_doa_rad = number * (float)(M_PI / 180.0);
        }
    }
    return 0;
}
static int mcab_sample_spatial_unknown(void *user, const char *name) {
    (void)user; (void)name;
    return 2;
}
int mcab_sample_parse_spatial_cli(int argc, char **argv, int raw, McabSampleSpatialOptions *out) {
    static const McabSampleCliOption spatial[] = {
        {"--print-mem-size",0}, {"--sample-rate",1}, {"--fft-size",1},
        {"--uca-radius-m",1}, {"--fixed-doa-deg",1},
        {"--mic-raw",1}, {"--ref-raw",1}, {"--output-raw",1}, {"--vad-u8",1}
    };
    memset(out, 0, sizeof(*out));
    out->sample_rate = raw ? 0 : 16000;
    out->uca_radius_m = 0.035f;
    if (mcab_sample_scan_cli(argc, argv, spatial, raw ? 9 : 5, out,
                      mcab_sample_spatial_set, mcab_sample_spatial_unknown) != 0) return 2;
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

int mcab_sample_main(int argc, char **argv, const McabSampleOps *ops, void *user) {
    McabSampleHopView hop = {0};
    hop.program = argv[0];
    int rc = ops->parse(user, argc, argv);
    if (rc != 0) return rc == 1 ? 0 : 2;
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
