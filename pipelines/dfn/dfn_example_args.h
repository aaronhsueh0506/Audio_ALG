/* Command-line handling shared by the DFN2 board skeletons
 * (mono/aec_dfn_res/main.c, 4ch/aec_bf_dfn_res/main.c): the host grid, the DFN2 model
 * grid (--ai-sample-rate / --ai-fft-size, a row of the table below built into a DFN2Grid with
 * dfn2_grid_make()), delay profile (common/example_cli.h), the exported ERB matrices, DFN2
 * attenuation limit and the comfort-noise switch (off by default). Header-only; not linked into
 * the delivered archives. */
#ifndef DFN_EXAMPLE_ARGS_H
#define DFN_EXAMPLE_ARGS_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "aec.h"
#include "common/example_cli.h"
#include "dfn2_grid.h"
#include "dfn_unit_partition.h"

/* The model grids the skeletons offer. Which (sample rate, fft size) pair maps to which n_erb,
 * df_bins, df_order and lookahead pair is a property of the trained model
 * (AINR/DeepFilterNet2/config.ini [signal], [model]), so it is this program's table; hop, bins,
 * the scales and the feature normaliser are all fixed by dfn2_grid.h / dfn2_grid_make(). There is
 * no 32 kHz row: the AEC in front of the stage has no 32 kHz grid, so no host could run that
 * model. */
static const DFN2GridSpec DFN_EXAMPLE_MODEL_ROWS[] = {
    { .sample_rate = 16000, .n_fft = 256, .n_erb = 24, .df_bins = 96, .df_order = 5,
      .mask_lookahead = 0, .df_lookahead = 0, .frames_per_call = 1 },
    { .sample_rate = 48000, .n_fft = 1024, .n_erb = 32, .df_bins = 96, .df_order = 5,
      .mask_lookahead = 1, .df_lookahead = 1, .frames_per_call = 1 },
};
#define DFN_EXAMPLE_MODEL_ROW_COUNT \
    ((int)(sizeof(DFN_EXAMPLE_MODEL_ROWS) / sizeof(DFN_EXAMPLE_MODEL_ROWS[0])))
#define DFN_EXAMPLE_DEFAULT_MODEL_RATE 48000
#define DFN_EXAMPLE_DEFAULT_MODEL_FFT  1024

/* Delay profile: `n` (matched-filter bank size) sets the bulk-delay search range
 * and is chosen from the delay distribution of the deployment route. Range per
 * bank: n=1 ~125 ms, 2 ~221 ms, 3 ~317 ms, 4 ~413 ms, 5 ~509 ms. No runtime
 * setter. DA_NUM_FILTERS is the bank-size cap and the value the non-MATCHED modes
 * require. */
#ifndef DFN_EXAMPLE_DELAY_NUM_FILTERS
#define DFN_EXAMPLE_DELAY_NUM_FILTERS DA_NUM_FILTERS
#endif
_Static_assert(DFN_EXAMPLE_DELAY_NUM_FILTERS >= 1 &&
               DFN_EXAMPLE_DELAY_NUM_FILTERS <= DA_NUM_FILTERS,
               "DFN_EXAMPLE_DELAY_NUM_FILTERS outside lib/aec's bank range");

typedef struct DfnExampleArgs {
    int sample_rate;         /* one of the skeleton's host grids (default 16000) */
    int ai_sample_rate;      /* --ai-sample-rate (default 48000) */
    int ai_fft_size;         /* --ai-fft-size (default 1024) */
    DFN2Grid model_grid;     /* the DFN2 model's grid of that pair, set by dfn_example_resolve_model_grid() */
    AecDelayMode delay_mode;
    int delay_num_filters;   /* MATCHED only; 1..DA_NUM_FILTERS             */
    int fixed_delay_samples; /* FIXED only; -1 otherwise                    */
    const char *erb_fwd_path;
    const char *erb_inv_path;
    float atten_lim_db;
    int enable_cng;
} DfnExampleArgs;

/* `rates` is the 0-terminated list of host grids the skeleton offers. */
static int dfn_example_rate_known(const int *rates, int value) {
    for (; *rates; ++rates)
        if (*rates == value) return 1;
    return 0;
}

static const char *dfn_example_rates_text(const int *rates, char *buf, size_t cap) {
    size_t used = 0;
    buf[0] = '\0';
    for (; *rates && used < cap; ++rates)
        used += (size_t)snprintf(buf + used, cap - used, "%s%d",
                                 used ? "|" : "", *rates);
    return buf;
}

/* "16000/256, 48000/1024". */
static const char *dfn_example_model_rows_text(char *buf, size_t cap) {
    size_t used = 0;
    int i;
    buf[0] = '\0';
    for (i = 0; i < DFN_EXAMPLE_MODEL_ROW_COUNT && used < cap; ++i)
        used += (size_t)snprintf(buf + used, cap - used, "%s%d/%d", i ? ", " : "",
                                 DFN_EXAMPLE_MODEL_ROWS[i].sample_rate,
                                 DFN_EXAMPLE_MODEL_ROWS[i].n_fft);
    return buf;
}

/* The row of a (sample rate, fft size) pair, or NULL. */
static const DFN2GridSpec *dfn_example_model_row(int sample_rate, int n_fft) {
    int i;
    for (i = 0; i < DFN_EXAMPLE_MODEL_ROW_COUNT; ++i)
        if (DFN_EXAMPLE_MODEL_ROWS[i].sample_rate == sample_rate &&
            DFN_EXAMPLE_MODEL_ROWS[i].n_fft == n_fft)
            return &DFN_EXAMPLE_MODEL_ROWS[i];
    return NULL;
}

static void dfn_example_usage(const char *prog, const char *rates_text) {
    char rows_text[96];
    dfn_example_model_rows_text(rows_text, sizeof(rows_text));
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --sample-rate {%s}\n"
        "                              host grid (default 16000); only the DFN2\n"
        "                              stage runs at the model grid, through the\n"
        "                              rate bridge when the host rate differs\n"
        "                              (the bridge needs the 48 kHz / 1024 model)\n"
        "  --ai-sample-rate <Hz>       DFN2 model grid: sample rate (default %d)\n"
        "  --ai-fft-size <n>           DFN2 model grid: FFT size (default %d);\n"
        "                              the pair must be one of %s\n"
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
        prog, rates_text, DFN_EXAMPLE_DEFAULT_MODEL_RATE, DFN_EXAMPLE_DEFAULT_MODEL_FFT,
        rows_text, DFN_EXAMPLE_DELAY_NUM_FILTERS);
}

/* Builds out->model_grid from out->ai_sample_rate / ai_fft_size: the row of the table through
 * dfn2_grid_make(). 0, or 2 (rejected argument) after a message that names the pair and the
 * accepted ones. */
static int dfn_example_resolve_model_grid(DfnExampleArgs *out, const char *name) {
    char rows_text[96];
    const DFN2GridSpec *row = dfn_example_model_row(out->ai_sample_rate, out->ai_fft_size);
    int status;
    if (!row) {
        fprintf(stderr, "%s: no DFN2 model grid for --ai-sample-rate %d --ai-fft-size %d "
                "(accepted pairs: %s; defaults %d and %d)\n", name, out->ai_sample_rate,
                out->ai_fft_size, dfn_example_model_rows_text(rows_text, sizeof(rows_text)),
                DFN_EXAMPLE_DEFAULT_MODEL_RATE, DFN_EXAMPLE_DEFAULT_MODEL_FFT);
        return 2;
    }
    status = dfn2_grid_make(&out->model_grid, row);
    if (status != DFN2_GRID_OK) {
        fprintf(stderr, "%s: the DFN2 model grid %d Hz / FFT %d is refused by the library: %s\n",
                name, row->sample_rate, row->n_fft, dfn2_grid_status_text(status));
        return 2;
    }
    return 0;
}

/* Returns 0 on success, 1 on --help, 2 on a rejected argument. `rates` is the
 * 0-terminated list of host grids the skeleton offers. A rejection names the
 * requested value and what the program accepts. out->model_grid is the DFN2 grid of the
 * (--ai-sample-rate, --ai-fft-size) pair, 48000 / 1024 when neither is given. */
static int dfn_example_parse_args(int argc, char **argv, const char *name,
                                  const int *rates, DfnExampleArgs *out) {
    char rates_text[64];
    ExampleCliDelayOptions delay;
    int i;

    dfn_example_rates_text(rates, rates_text, sizeof(rates_text));
    example_cli_delay_options_default(&delay, DFN_EXAMPLE_DELAY_NUM_FILTERS);

    out->sample_rate = 16000;
    out->ai_sample_rate = DFN_EXAMPLE_DEFAULT_MODEL_RATE;
    out->ai_fft_size = DFN_EXAMPLE_DEFAULT_MODEL_FFT;
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
            dfn_example_usage(argv[0], rates_text);
            return 1;
        } else if (!strcmp(arg, "--sample-rate")) {
            const char *value = example_cli_option_value(argc, argv, &i, name);
            if (!value) return 2;
            if (example_cli_parse_int(value, &out->sample_rate) != 0 ||
                !dfn_example_rate_known(rates, out->sample_rate)) {
                fprintf(stderr, "%s: --sample-rate '%s' is not a host grid of "
                        "this skeleton (accepted: %s)\n", name, value,
                        rates_text);
                return 2;
            }
        } else if (!strcmp(arg, "--ai-sample-rate") || !strcmp(arg, "--ai-fft-size")) {
            const char *value = example_cli_option_value(argc, argv, &i, name);
            if (!value) return 2;
            if (example_cli_parse_int(value, !strcmp(arg, "--ai-sample-rate")
                                                 ? &out->ai_sample_rate
                                                 : &out->ai_fft_size) != 0) {
                fprintf(stderr, "%s: %s '%s' is not an integer\n", name, arg, value);
                return 2;
            }
        } else if (!strcmp(arg, "--delay-mode") || !strcmp(arg, "--delay-num-filters") ||
                   !strcmp(arg, "--fixed-delay")) {
            const char *value = example_cli_option_value(argc, argv, &i, name);
            if (!value) return 2;
            if (example_cli_delay_set_option(&delay, arg, value, name) != 0) return 2;
        } else if (!strcmp(arg, "--erb-fwd") || !strcmp(arg, "--erb-inv")) {
            const char *value = example_cli_option_value(argc, argv, &i, name);
            if (!value) return 2;
            if (!strcmp(arg, "--erb-fwd")) out->erb_fwd_path = value;
            else out->erb_inv_path = value;
        } else if (!strcmp(arg, "--atten-lim")) {
            const char *value = example_cli_option_value(argc, argv, &i, name);
            if (!value) return 2;
            if (example_cli_parse_float(value, &out->atten_lim_db) != 0) {
                fprintf(stderr, "%s: --atten-lim '%s' is not a finite dB "
                        "value\n", name, value);
                return 2;
            }
        } else if (!strcmp(arg, "--cng")) {
            out->enable_cng = 1;
        } else {
            fprintf(stderr, "%s: unknown argument '%s'\n", name, arg);
            dfn_example_usage(argv[0], rates_text);
            return 2;
        }
    }

    if (dfn_example_resolve_model_grid(out, name) != 0) return 2;
    if ((out->erb_fwd_path == NULL) != (out->erb_inv_path == NULL)) {
        fprintf(stderr, "%s: --erb-fwd and --erb-inv go together\n", name);
        return 2;
    }
    if (example_cli_delay_validate(&delay, name) != 0) return 2;
    out->delay_mode = delay.mode;
    out->delay_num_filters = delay.num_filters;
    out->fixed_delay_samples = delay.fixed_samples;
    return 0;
}

/* The ERB matrices of the model grid: `fwd` bin-major [n_bins][n_erb], `inv` band-major
 * [n_erb][n_bins], both pointing into `block`, which the owner frees with dfn_example_erb_free()
 * after the objects that borrow them are destroyed. */
typedef struct DfnExampleErb {
    void *block;
    const float *fwd;
    const float *inv;
} DfnExampleErb;

static void dfn_example_erb_free(DfnExampleErb *erb) {
    free(erb->block);
    memset(erb, 0, sizeof(*erb));
}

/* Exactly `count` little-endian float32 values, or failure. */
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
        fprintf(stderr, "%s: %s holds %s than the %zu floats the model grid "
                "expects\n", name, path, got != count ? "fewer" : "more",
                count);
        return -1;
    }
    return 0;
}

/* Fills `erb` for args->model_grid from the two exported files, or with the unit partition when
 * none were named. 0, or -1 after a message. */
static int dfn_example_load_matrices(const DfnExampleArgs *args, DfnExampleErb *erb,
                                     const char *name) {
    const DFN2Grid *grid = &args->model_grid;
    const size_t floats = (size_t)grid->n_bins * (size_t)grid->n_erb;
    void *mem = NULL;
    float *fwd;
    memset(erb, 0, sizeof(*erb));
    if (posix_memalign(&mem, 16, 2u * floats * sizeof(float)) != 0) {
        fprintf(stderr, "%s: out of memory for the ERB matrices\n", name);
        return -1;
    }
    fwd = (float *)mem;
    erb->block = mem;
    erb->fwd = fwd;
    erb->inv = fwd + floats;
    if (!args->erb_fwd_path) {
        dfn_unit_partition(fwd, fwd + floats, grid);
        printf("%s: no --erb-fwd/--erb-inv: unit ERB partition, fail-open "
               "smoke run only\n", name);
        return 0;
    }
    if (dfn_example_read_matrix(args->erb_fwd_path, fwd, floats, name) != 0 ||
        dfn_example_read_matrix(args->erb_inv_path, fwd + floats, floats, name) != 0) {
        dfn_example_erb_free(erb);
        return -1;
    }
    return 0;
}

#endif /* DFN_EXAMPLE_ARGS_H */
