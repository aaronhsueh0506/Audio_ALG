/* Command-line handling shared by the two DFN2 board skeletons
 * (mono_aec_dfn_res/main.c, 4ch_aec_bf_dfn_res/main.c; this header lives in
 * pipelines/ so both include it by name): the delay profile
 * in lib/aec's aec_wav spelling, the exported ERB matrices, the DFN2
 * attenuation limit and the comfort-noise switch (off by default). Header-only so each
 * skeleton stays a single translation unit; nothing here is linked into the
 * delivered archives. */
#ifndef DFN_EXAMPLE_ARGS_H
#define DFN_EXAMPLE_ARGS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "dfn2_process.h"
#include "dfn_unit_partition.h"

/* The delay profile is a product deployment decision, not a property of the
 * source: `n` (matched-filter bank size) sets how far the estimator can
 * search for the bulk far-to-mic delay, and it has to be chosen from the
 * measured delay distribution of the SKU/route the binary is deployed on.
 * Reliable bulk-delay search range per bank (lib/aec's contract value):
 * n=1 ~125 ms, 2 ~221 ms, 3 ~317 ms, 4 ~413 ms, 5 ~509 ms. Changing n means
 * re-querying the pool and re-init; there is no runtime setter. lib/aec's
 * DA_NUM_FILTERS is both the bank-size cap and the exact value the
 * non-MATCHED modes require. */
#ifndef DFN_EXAMPLE_DELAY_NUM_FILTERS
#define DFN_EXAMPLE_DELAY_NUM_FILTERS DA_NUM_FILTERS
#endif
_Static_assert(DFN_EXAMPLE_DELAY_NUM_FILTERS >= 1 &&
               DFN_EXAMPLE_DELAY_NUM_FILTERS <= DA_NUM_FILTERS,
               "DFN_EXAMPLE_DELAY_NUM_FILTERS outside lib/aec's bank range");

typedef struct DfnExampleArgs {
    int sample_rate;         /* host grid: 8000, 16000 (default) or 48000 */
    AecDelayMode delay_mode;
    int delay_num_filters;   /* MATCHED only; 1..DA_NUM_FILTERS             */
    int fixed_delay_samples; /* FIXED only; -1 otherwise                    */
    const char *erb_fwd_path;
    const char *erb_inv_path;
    float atten_lim_db;
    int enable_cng;
} DfnExampleArgs;

static void dfn_example_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --sample-rate {8000|16000|48000}\n"
        "                              host grid (default 16000); only the DFN2\n"
        "                              stage runs at 48 kHz, through the rate\n"
        "                              bridge below 48 kHz\n"
        "  --delay-mode {matched|fixed|external}\n"
        "                              far alignment policy (default: matched)\n"
        "  --delay-num-filters <1..5>  MATCHED matched-filter bank size\n"
        "                              (default %d). Reliable bulk-delay\n"
        "                              search: 1~125ms 2~221ms 3~317ms\n"
        "                              4~413ms 5~509ms. Not a runtime knob:\n"
        "                              changing it re-queries the pool.\n"
        "  --fixed-delay <samples>     FIXED delay in native-rate samples (>=0)\n"
        "  --erb-fwd <erb_fwd.bin>     exported DFN2 ERB matrices (both or\n"
        "  --erb-inv <erb_inv.bin>     neither; without them a unit partition\n"
        "                              serves the fail-open smoke run only)\n"
        "  --atten-lim <dB>            DFN2 attenuation limit (default 0 = off)\n"
        "  --cng                       enable the host's comfort-noise fill\n"
        "                              (off by default for the DFN2 variants)\n"
        "  -h, --help                  this message\n",
        prog, DFN_EXAMPLE_DELAY_NUM_FILTERS);
}

static int dfn_example_parse_delay_mode(const char *s, AecDelayMode *out) {
    if (!strcmp(s, "matched"))  { *out = AEC_DELAY_MATCHED;          return 0; }
    if (!strcmp(s, "fixed"))    { *out = AEC_DELAY_FIXED;            return 0; }
    if (!strcmp(s, "external")) { *out = AEC_DELAY_EXTERNAL_ALIGNED; return 0; }
    return -1;
}

static const char *dfn_example_delay_mode_name(AecDelayMode m) {
    switch (m) {
        case AEC_DELAY_MATCHED:          return "matched";
        case AEC_DELAY_FIXED:            return "fixed";
        case AEC_DELAY_EXTERNAL_ALIGNED: return "external";
        default:                         return "?";
    }
}

/* Returns 0 on success, 1 on --help, 2 on a rejected argument. Every
 * rejection names BOTH the requested value and what this build accepts,
 * because the pipeline TUs are stdio-free and can only answer with a NULL
 * handle. */
static int dfn_example_parse_args(int argc, char **argv, const char *name,
                                  DfnExampleArgs *out) {
    int have_num_filters = 0;
    int have_fixed = 0;
    int i;

    out->sample_rate = 16000;
    out->delay_mode = AEC_DELAY_MATCHED;
    out->delay_num_filters = DFN_EXAMPLE_DELAY_NUM_FILTERS;
    out->fixed_delay_samples = -1;
    out->erb_fwd_path = NULL;
    out->erb_inv_path = NULL;
    out->atten_lim_db = 0.0f;
    out->enable_cng = 0;

    for (i = 1; i < argc; ++i) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            dfn_example_usage(argv[0]);
            return 1;
        } else if (!strcmp(arg, "--sample-rate") && i + 1 < argc) {
            out->sample_rate = atoi(argv[++i]);
            if (out->sample_rate != 8000 && out->sample_rate != 16000 &&
                out->sample_rate != 48000) {
                fprintf(stderr, "%s: --sample-rate %d is not a host grid "
                        "(accepted: 8000|16000|48000)\n", name, out->sample_rate);
                return 2;
            }
        } else if (!strcmp(arg, "--delay-mode") && i + 1 < argc) {
            if (dfn_example_parse_delay_mode(argv[++i], &out->delay_mode) != 0) {
                fprintf(stderr, "%s: --delay-mode '%s' is not a mode "
                        "(accepted: matched|fixed|external)\n", name, argv[i]);
                return 2;
            }
        } else if (!strcmp(arg, "--delay-num-filters") && i + 1 < argc) {
            out->delay_num_filters = atoi(argv[++i]);
            have_num_filters = 1;
        } else if (!strcmp(arg, "--fixed-delay") && i + 1 < argc) {
            out->fixed_delay_samples = atoi(argv[++i]);
            have_fixed = 1;
        } else if (!strcmp(arg, "--erb-fwd") && i + 1 < argc) {
            out->erb_fwd_path = argv[++i];
        } else if (!strcmp(arg, "--erb-inv") && i + 1 < argc) {
            out->erb_inv_path = argv[++i];
        } else if (!strcmp(arg, "--atten-lim") && i + 1 < argc) {
            out->atten_lim_db = (float)atof(argv[++i]);
        } else if (!strcmp(arg, "--cng")) {
            out->enable_cng = 1;
        } else {
            fprintf(stderr, "%s: unknown argument '%s'\n", name, arg);
            dfn_example_usage(argv[0]);
            return 2;
        }
    }

    if ((out->erb_fwd_path == NULL) != (out->erb_inv_path == NULL)) {
        fprintf(stderr, "%s: --erb-fwd and --erb-inv go together\n", name);
        return 2;
    }
    /* n is meaningful only where a matched bank exists. 0 is NOT "disabled":
     * FIXED and EXTERNAL_ALIGNED are separate modes, and both require
     * n == DA_NUM_FILTERS because they build no bank at all. */
    if (have_fixed && out->delay_mode != AEC_DELAY_FIXED) {
        fprintf(stderr, "%s: --fixed-delay %d is only valid with "
                "--delay-mode fixed (requested mode: %s)\n", name,
                out->fixed_delay_samples,
                dfn_example_delay_mode_name(out->delay_mode));
        return 2;
    }
    if (out->delay_mode == AEC_DELAY_FIXED &&
        (!have_fixed || out->fixed_delay_samples < 0)) {
        fprintf(stderr, "%s: --delay-mode fixed needs --fixed-delay "
                "<samples> >= 0 (requested: %d)\n", name,
                have_fixed ? out->fixed_delay_samples : -1);
        return 2;
    }
    if (out->delay_mode == AEC_DELAY_MATCHED) {
        if (out->delay_num_filters < 1 ||
            out->delay_num_filters > DA_NUM_FILTERS) {
            fprintf(stderr, "%s: --delay-num-filters %d is out of range "
                    "(accepted: 1..%d; 0 does not mean 'off' -- use "
                    "--delay-mode fixed or external instead)\n", name,
                    out->delay_num_filters, DA_NUM_FILTERS);
            return 2;
        }
    } else {
        if (have_num_filters && out->delay_num_filters != DA_NUM_FILTERS) {
            fprintf(stderr, "%s: --delay-num-filters %d is only valid with "
                    "--delay-mode matched (requested mode: %s, which builds "
                    "no matched bank and requires n == %d)\n", name,
                    out->delay_num_filters,
                    dfn_example_delay_mode_name(out->delay_mode),
                    DA_NUM_FILTERS);
            return 2;
        }
        out->delay_num_filters = DA_NUM_FILTERS;
    }
    return 0;
}

/* Exactly `count` little-endian float32 values, or failure: a short or long
 * file is a matrix from another model geometry, not a partial success. */
static int dfn_example_read_matrix(const char *path, float *dst,
                                   size_t count, const char *name) {
    FILE *f = fopen(path, "rb");
    size_t got;
    int extra;
    if (!f) {
        fprintf(stderr, "%s: cannot open %s\n", name, path);
        return -1;
    }
    got = fread(dst, sizeof(float), count, f);
    extra = fgetc(f);
    fclose(f);
    if (got != count || extra != EOF) {
        fprintf(stderr, "%s: %s holds %s than the %zu floats this build "
                "expects\n", name, path, got != count ? "fewer" : "more",
                count);
        return -1;
    }
    return 0;
}

/* Fills the two matrices from the exported files, or with the unit
 * partition when none were named. */
static int dfn_example_load_matrices(const DfnExampleArgs *args,
                                     float *erb_fwd, float *erb_inv,
                                     const char *name) {
    if (!args->erb_fwd_path) {
        dfn_unit_partition(erb_fwd, erb_inv);
        printf("%s: no --erb-fwd/--erb-inv: unit ERB partition, fail-open "
               "smoke run only\n", name);
        return 0;
    }
    if (dfn_example_read_matrix(args->erb_fwd_path, erb_fwd,
                                (size_t)DFN2_N_BINS * DFN2_N_ERB, name) != 0 ||
        dfn_example_read_matrix(args->erb_inv_path, erb_inv,
                                (size_t)DFN2_N_ERB * DFN2_N_BINS, name) != 0)
        return -1;
    return 0;
}

#endif /* DFN_EXAMPLE_ARGS_H */
