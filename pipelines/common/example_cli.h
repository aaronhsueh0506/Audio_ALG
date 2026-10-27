/* example_cli.h -- example: the delay-profile options of the example programs
 * (--delay-mode / --delay-num-filters / --fixed-delay, lib/aec's aec_wav spelling), parsed
 * strictly and checked together. Header-only, no ABI: copy and adapt.
 *
 *     ExampleCliDelayOptions d;
 *     example_cli_delay_options_default(&d, DA_NUM_FILTERS);
 *     for each option: example_cli_delay_set_option(&d, name, value, prog)  // 0, or 2 = rejected
 *     example_cli_delay_validate(&d, prog)                                  // 0, or 2 = rejected
 *
 * A rejection prints "<prog>: ..." with the requested value and what the build accepts. The
 * bank size `n` applies to MATCHED only; the other modes build no bank and leave
 * n == DA_NUM_FILTERS. */
#ifndef PIPELINES_EXAMPLE_CLI_H
#define PIPELINES_EXAMPLE_CLI_H

#include <errno.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ExampleCliDelayOptions {
    AecDelayMode mode;
    int num_filters;        /* MATCHED only; 1..DA_NUM_FILTERS */
    int fixed_samples;      /* FIXED only; -1 otherwise */
    int have_num_filters, have_fixed;
} ExampleCliDelayOptions;

static inline void example_cli_delay_options_default(ExampleCliDelayOptions* o, int num_filters) {
    memset(o, 0, sizeof(*o));
    o->mode = AEC_DELAY_MATCHED;
    o->num_filters = num_filters;
    o->fixed_samples = -1;
}

static inline const char* example_cli_delay_mode_name(AecDelayMode m) {
    switch (m) {
        case AEC_DELAY_MATCHED:          return "matched";
        case AEC_DELAY_FIXED:            return "fixed";
        case AEC_DELAY_EXTERNAL_ALIGNED: return "external";
        default:                         return "?";
    }
}

/* Whole-string decimal int, or -1: "", " 12", "12x" and values outside int are rejected. */
static inline int example_cli_parse_int(const char* text, int* value) {
    char* end = NULL;
    long parsed;
    if (!text || !*text || isspace((unsigned char)*text)) return -1;
    errno = 0;
    parsed = strtol(text, &end, 10);
    if (errno || end == text || *end != '\0' || parsed < INT_MIN || parsed > INT_MAX)
        return -1;
    *value = (int)parsed;
    return 0;
}

/* Whole-string finite float, or -1 under the same rules ("12dB", "", " 1", inf, nan). */
static inline int example_cli_parse_float(const char* text, float* value) {
    char* end = NULL;
    float parsed;
    if (!text || !*text || isspace((unsigned char)*text)) return -1;
    errno = 0;
    parsed = strtof(text, &end);
    if (errno || end == text || *end != '\0' || !isfinite(parsed)) return -1;
    *value = parsed;
    return 0;
}

/* One of --delay-mode / --delay-num-filters / --fixed-delay with its value. */
static inline int example_cli_delay_set_option(ExampleCliDelayOptions* o, const char* name,
                                               const char* value, const char* prog) {
    if (!strcmp(name, "--delay-mode")) {
        if (!strcmp(value, "matched"))       o->mode = AEC_DELAY_MATCHED;
        else if (!strcmp(value, "fixed"))    o->mode = AEC_DELAY_FIXED;
        else if (!strcmp(value, "external")) o->mode = AEC_DELAY_EXTERNAL_ALIGNED;
        else {
            fprintf(stderr, "%s: --delay-mode '%s' is not a mode "
                    "(accepted: matched|fixed|external)\n", prog, value);
            return 2;
        }
    } else if (!strcmp(name, "--delay-num-filters")) {
        if (example_cli_parse_int(value, &o->num_filters) != 0) {
            fprintf(stderr, "%s: --delay-num-filters '%s' is not an "
                    "integer (accepted: 1..%d)\n", prog, value, DA_NUM_FILTERS);
            return 2;
        }
        o->have_num_filters = 1;
    } else {
        if (example_cli_parse_int(value, &o->fixed_samples) != 0) {
            fprintf(stderr, "%s: --fixed-delay '%s' is not an integer "
                    "(accepted: native-rate samples >= 0)\n", prog, value);
            return 2;
        }
        o->have_fixed = 1;
    }
    return 0;
}

/* The options together; outside MATCHED, n is set to DA_NUM_FILTERS. */
static inline int example_cli_delay_validate(ExampleCliDelayOptions* o, const char* prog) {
    if (o->have_fixed && o->mode != AEC_DELAY_FIXED) {
        fprintf(stderr, "%s: --fixed-delay %d is only valid with "
                "--delay-mode fixed (requested mode: %s)\n", prog,
                o->fixed_samples, example_cli_delay_mode_name(o->mode));
        return 2;
    }
    if (o->mode == AEC_DELAY_FIXED && (!o->have_fixed || o->fixed_samples < 0)) {
        fprintf(stderr, "%s: --delay-mode fixed needs --fixed-delay "
                "<samples> >= 0 (requested: %d)\n", prog,
                o->have_fixed ? o->fixed_samples : -1);
        return 2;
    }
    if (o->mode == AEC_DELAY_MATCHED) {
        if (o->num_filters < 1 || o->num_filters > DA_NUM_FILTERS) {
            fprintf(stderr, "%s: --delay-num-filters %d is out of range "
                    "(accepted: 1..%d; 0 does not mean 'off' -- use "
                    "--delay-mode fixed or external instead)\n", prog,
                    o->num_filters, DA_NUM_FILTERS);
            return 2;
        }
    } else {
        if (o->have_num_filters && o->num_filters != DA_NUM_FILTERS) {
            fprintf(stderr, "%s: --delay-num-filters %d is only valid with "
                    "--delay-mode matched (requested mode: %s, which builds "
                    "no matched bank and requires n == %d)\n", prog,
                    o->num_filters, example_cli_delay_mode_name(o->mode), DA_NUM_FILTERS);
            return 2;
        }
        o->num_filters = DA_NUM_FILTERS;
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_EXAMPLE_CLI_H */
