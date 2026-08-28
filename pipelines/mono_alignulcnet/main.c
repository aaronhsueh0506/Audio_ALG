#include "audio_pipeline_ulcnet.h"
#include "ulcnet_accelerator_adapter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Delay depth D for THIS standalone smoke example only.
 *
 * D is a runtime descriptor field, not a library constant: the C chain serves
 * whatever depth the attached model publishes. This macro is only the example
 * binary's compile-time default, overridable with
 * -DULCNET_EXAMPLE_DELAY_FRAMES=N at build time. It is NOT a switch the built
 * binary can flip -- changing D means rebuilding.
 *
 * It MUST equal the D the model in use was exported with. Nothing checks that
 * for you: the descriptor validator only bounds-checks the range, so a
 * mismatch between this value and the graph is silent and corrupting (see
 * ulcnet_process.h's I/O contract). The shipped checkpoints are not all D=8;
 * set this to match the model you are actually running. */
#ifndef ULCNET_EXAMPLE_DELAY_FRAMES
#define ULCNET_EXAMPLE_DELAY_FRAMES 8
#endif
_Static_assert(ULCNET_EXAMPLE_DELAY_FRAMES >= ULCNET_MODEL_IO_MIN_D &&
               ULCNET_EXAMPLE_DELAY_FRAMES <= ULCNET_MODEL_IO_MAX_D,
               "ULCNET_EXAMPLE_DELAY_FRAMES outside the descriptor's range");

/* The delay profile is a product deployment decision, not a property of this
 * source file: `n` (matched-filter bank size) sets how far the estimator can
 * search for the bulk far-to-mic delay, and it has to be chosen from the
 * measured delay distribution of the SKU/route this binary is deployed on.
 * It is therefore read from the command line here -- the same
 * --delay-mode / --delay-num-filters / --fixed-delay spelling lib/aec's
 * aec_wav example uses -- and the resolved profile is printed, so a board
 * bring-up run states which profile it actually ran instead of leaving it
 * buried in a literal. Changing n means re-querying the pool and re-init;
 * there is no runtime setter.
 *
 * Reliable bulk-delay search range per bank (lib/aec's contract value, not a
 * geometric span): n=1 ~125 ms, 2 ~221 ms, 3 ~317 ms, 4 ~413 ms, 5 ~509 ms.
 *
 * lib/aec's DA_NUM_FILTERS is both the bank-size cap and the exact value the
 * non-MATCHED modes require -- aec_validate_config() rejects any other n once
 * the mode is not MATCHED -- so it is used for both roles below. The MATCHED
 * default is a separate decision; see ULCNET_EXAMPLE_DELAY_NUM_FILTERS. */

/* MATCHED bank size n for THIS standalone smoke example only, overridable with
 * -DULCNET_EXAMPLE_DELAY_NUM_FILTERS=N so a per-SKU build can bake in its
 * measured delay profile instead of passing the flag on every run.
 *
 * It is only the DEFAULT: --delay-num-filters still wins. It applies to
 * MATCHED alone, because the other two modes build no bank and lib/aec
 * requires DA_NUM_FILTERS there. Like D, it is NOT a switch the built binary
 * can flip -- n sizes the matched bank, so changing it means a rebuild, a
 * fresh pool query and a re-init. */
#ifndef ULCNET_EXAMPLE_DELAY_NUM_FILTERS
#define ULCNET_EXAMPLE_DELAY_NUM_FILTERS DA_NUM_FILTERS
#endif
_Static_assert(ULCNET_EXAMPLE_DELAY_NUM_FILTERS >= 1 &&
               ULCNET_EXAMPLE_DELAY_NUM_FILTERS <= DA_NUM_FILTERS,
               "ULCNET_EXAMPLE_DELAY_NUM_FILTERS outside lib/aec's bank range");

typedef struct DelayProfile {
    AecDelayMode mode;
    int num_filters;      /* MATCHED only; 1..DA_NUM_FILTERS                 */
    int fixed_samples;    /* FIXED only; -1 otherwise                        */
} DelayProfile;

static void usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --delay-mode {matched|fixed|external}\n"
        "                              far alignment policy (default: matched)\n"
        "  --delay-num-filters <1..5>  MATCHED matched-filter bank size\n"
        "                              (default %d). Reliable bulk-delay\n"
        "                              search: 1~125ms 2~221ms 3~317ms\n"
        "                              4~413ms 5~509ms. Not a runtime knob:\n"
        "                              changing it re-queries the pool.\n"
        "  --fixed-delay <samples>     FIXED delay in native-rate samples (>=0)\n"
        "  -h, --help                  this message\n",
        prog, ULCNET_EXAMPLE_DELAY_NUM_FILTERS);
}

static int parse_delay_mode(const char* s, AecDelayMode* out) {
    if (!strcmp(s, "matched"))  { *out = AEC_DELAY_MATCHED;          return 0; }
    if (!strcmp(s, "fixed"))    { *out = AEC_DELAY_FIXED;            return 0; }
    if (!strcmp(s, "external")) { *out = AEC_DELAY_EXTERNAL_ALIGNED; return 0; }
    return -1;
}

static const char* delay_mode_name(AecDelayMode m) {
    switch (m) {
        case AEC_DELAY_MATCHED:          return "matched";
        case AEC_DELAY_FIXED:            return "fixed";
        case AEC_DELAY_EXTERNAL_ALIGNED: return "external";
        default:                         return "?";
    }
}

/* Returns 0 on success, 1 on --help, 2 on a rejected profile. Every rejection
 * names BOTH the requested value and what this build accepts, because the
 * pipeline TU itself is stdio-free and can only answer with a NULL handle. */
static int parse_delay_profile(int argc, char** argv, DelayProfile* out) {
    int have_num_filters = 0;
    int have_fixed = 0;
    int i;

    out->mode = AEC_DELAY_MATCHED;
    out->num_filters = ULCNET_EXAMPLE_DELAY_NUM_FILTERS;
    out->fixed_samples = -1;

    for (i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if ((!strcmp(arg, "-h") || !strcmp(arg, "--help"))) {
            usage(argv[0]);
            return 1;
        } else if (!strcmp(arg, "--delay-mode") && i + 1 < argc) {
            if (parse_delay_mode(argv[++i], &out->mode) != 0) {
                fprintf(stderr,
                        "mono_alignulcnet: --delay-mode '%s' is not a mode "
                        "(accepted: matched|fixed|external)\n", argv[i]);
                return 2;
            }
        } else if (!strcmp(arg, "--delay-num-filters") && i + 1 < argc) {
            out->num_filters = atoi(argv[++i]);
            have_num_filters = 1;
        } else if (!strcmp(arg, "--fixed-delay") && i + 1 < argc) {
            out->fixed_samples = atoi(argv[++i]);
            have_fixed = 1;
        } else {
            fprintf(stderr, "mono_alignulcnet: unknown argument '%s'\n", arg);
            usage(argv[0]);
            return 2;
        }
    }

    /* n is meaningful only where a matched bank exists. 0 is NOT "disabled" --
     * FIXED and EXTERNAL_ALIGNED are separate modes, and both require
     * n == DA_NUM_FILTERS because they build no bank at all. */
    if (have_fixed && out->mode != AEC_DELAY_FIXED) {
        fprintf(stderr,
                "mono_alignulcnet: --fixed-delay %d is only valid with "
                "--delay-mode fixed (requested mode: %s)\n",
                out->fixed_samples, delay_mode_name(out->mode));
        return 2;
    }
    if (out->mode == AEC_DELAY_FIXED &&
        (!have_fixed || out->fixed_samples < 0)) {
        fprintf(stderr,
                "mono_alignulcnet: --delay-mode fixed needs "
                "--fixed-delay <samples> >= 0 (requested: %d)\n",
                have_fixed ? out->fixed_samples : -1);
        return 2;
    }
    if (out->mode == AEC_DELAY_MATCHED) {
        if (out->num_filters < 1 || out->num_filters > DA_NUM_FILTERS) {
            fprintf(stderr,
                    "mono_alignulcnet: --delay-num-filters %d is out of range "
                    "(accepted: 1..%d; 0 does not mean 'off' -- use "
                    "--delay-mode fixed or external instead)\n",
                    out->num_filters, DA_NUM_FILTERS);
            return 2;
        }
    } else {
        if (have_num_filters && out->num_filters != DA_NUM_FILTERS) {
            fprintf(stderr,
                    "mono_alignulcnet: --delay-num-filters %d is only valid "
                    "with --delay-mode matched (requested mode: %s, which "
                    "builds no matched bank and requires n == %d)\n",
                    out->num_filters, delay_mode_name(out->mode),
                    DA_NUM_FILTERS);
            return 2;
        }
        /* The build-time default is a matched-bank knob, so it does not carry
         * into a mode that has no bank: the caller asked for the mode, not for
         * n, and lib/aec would reject the pair. */
        out->num_filters = DA_NUM_FILTERS;
    }
    return 0;
}

/* Replace this body with the board runtime call. `inputs` contains spectra
 * plus CPU-owned state; the runtime must fill every tensor in `outputs`. */
static int run_accelerator(void *user,
                           const UlcnetModelIoInputs *inputs,
                           UlcnetModelIoOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return -1; /* TODO(board): invoke the stateless ONNX accelerator. */
}

int main(int argc, char** argv) {
    AudioPipelineUlcnetConfig cfg;
    DelayProfile profile;
    UlcnetAcceleratorAdapter *adapter;
    UlcnetModelIoDescriptor model_descriptor;
    AudioPipelineUlcnet *pipeline;
    UlcnetModel model;
    void *adapter_pool = NULL;
    size_t adapter_bytes;
    size_t adapter_alignment;
    AudioPipelineUlcnetMemReq req;
    float mic[ULCNET_HOP] = {0};
    float far[ULCNET_HOP] = {0};
    float output[ULCNET_HOP];
    int hop;
    int index;
    int rc;
    rc = parse_delay_profile(argc, argv, &profile);
    if (rc != 0) return rc == 1 ? 0 : rc;

    cfg = audio_pipeline_ulcnet_default_config(ULCNET_SR);
    cfg.delay_mode = profile.mode;
    cfg.delay_num_filters = profile.num_filters;
    cfg.fixed_delay_samples = profile.fixed_samples;

    /* Print the RESOLVED profile with the pool the SAME config actually
     * costs -- query and init must agree on one (sample_rate, hop, n). The
     * query reads only the grid and delay fields, so it runs before anything
     * is allocated and the reject path has nothing to clean up. */
    if (audio_pipeline_ulcnet_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr,
                "mono_alignulcnet: pool query rejected the delay profile "
                "(mode=%s n=%d fixed_delay=%d)\n",
                delay_mode_name(profile.mode), profile.num_filters,
                profile.fixed_samples);
        return 1;
    }
    printf("mono_alignulcnet: delay profile mode=%s n=%d fixed_delay=%d "
           "-> pool %llu bytes (align %lu)\n",
           delay_mode_name(profile.mode), profile.num_filters,
           profile.fixed_samples, (unsigned long long)req.bytes,
           (unsigned long)req.alignment);

    /* The board loads this descriptor from the exported model metadata;
     * see ULCNET_EXAMPLE_DELAY_FRAMES at the top of this file. */
    if (ulcnet_model_io_descriptor_default(
            ULCNET_EXAMPLE_DELAY_FRAMES, &model_descriptor) != 0 ||
        ulcnet_accelerator_adapter_get_mem_size(
            &model_descriptor, &adapter_bytes, &adapter_alignment) != 0 ||
        posix_memalign(&adapter_pool, adapter_alignment, adapter_bytes) != 0) {
        return 1;
    }
    printf("%s: model I/O D=%d -> accelerator adapter pool %zu bytes (align %zu)\n",
           argv[0], ULCNET_EXAMPLE_DELAY_FRAMES, adapter_bytes,
           adapter_alignment);
    adapter = ulcnet_accelerator_adapter_init(
        adapter_pool, adapter_bytes, &model_descriptor,
        run_accelerator, NULL);
    if (!adapter) {
        free(adapter_pool);
        return 1;
    }

    model = ulcnet_accelerator_adapter_model(adapter);
    cfg.model = model;

    pipeline = audio_pipeline_ulcnet_create(&cfg);
    if (!pipeline) {
        fprintf(stderr, "mono_alignulcnet: pipeline init failed\n");
        free(adapter_pool);
        return 1;
    }

    /* Host smoke path. A product calls process() once per ULCNET_HOP-sample hop. */
    for (hop = 0; hop < 4; ++hop) {
        if (audio_pipeline_ulcnet_process(pipeline, mic, far, output) != 0) {
            audio_pipeline_ulcnet_destroy(pipeline);
            free(adapter_pool);
            return 1;
        }
        for (index = 0; index < ULCNET_HOP; ++index) {
            if (!isfinite(output[index])) {
                audio_pipeline_ulcnet_destroy(pipeline);
                free(adapter_pool);
                return 1;
            }
        }
    }

    audio_pipeline_ulcnet_destroy(pipeline);
    free(adapter_pool);
    puts("mono_alignulcnet: fail-open board skeleton PASS");
    return 0;
}
