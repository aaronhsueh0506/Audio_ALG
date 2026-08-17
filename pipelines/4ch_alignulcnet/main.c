#include "audio_pipeline_4ch_ulcnet.h"
#include "ulcnet_accelerator_adapter.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
 * This is a 4-channel pipeline but there is still exactly ONE matched bank:
 * the shared estimator in the core. The four lane AECs run
 * EXTERNAL_ALIGNED off its single aligned reference, so n does not multiply
 * by four and the lane pools do not move with it.
 *
 * Reliable bulk-delay search range per bank (lib/aec's contract value, not a
 * geometric span): n=1 ~125 ms, 2 ~221 ms, 3 ~317 ms, 4 ~413 ms, 5 ~509 ms.
 *
 * lib/aec's DA_NUM_FILTERS is both the bank-size cap and the default the
 * non-MATCHED modes require, so it is used for both roles below. */

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
        "  --delay-num-filters <1..5>  MATCHED matched-filter bank size for the\n"
        "                              ONE shared estimator (default %d).\n"
        "                              Reliable bulk-delay search: 1~125ms\n"
        "                              2~221ms 3~317ms 4~413ms 5~509ms. Not a\n"
        "                              runtime knob: changing it re-queries\n"
        "                              the pool.\n"
        "  --fixed-delay <samples>     FIXED delay in native-rate samples (>=0)\n"
        "  -h, --help                  this message\n",
        prog, DA_NUM_FILTERS);
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
    out->num_filters = DA_NUM_FILTERS;
    out->fixed_samples = -1;

    for (i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if ((!strcmp(arg, "-h") || !strcmp(arg, "--help"))) {
            usage(argv[0]);
            return 1;
        } else if (!strcmp(arg, "--delay-mode") && i + 1 < argc) {
            if (parse_delay_mode(argv[++i], &out->mode) != 0) {
                fprintf(stderr,
                        "4ch_alignulcnet: --delay-mode '%s' is not a mode "
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
            fprintf(stderr, "4ch_alignulcnet: unknown argument '%s'\n", arg);
            usage(argv[0]);
            return 2;
        }
    }

    /* n is meaningful only where a matched bank exists. 0 is NOT "disabled" --
     * FIXED and EXTERNAL_ALIGNED are separate modes, and both require n to
     * stay at the default because they build no bank at all. */
    if (have_fixed && out->mode != AEC_DELAY_FIXED) {
        fprintf(stderr,
                "4ch_alignulcnet: --fixed-delay %d is only valid with "
                "--delay-mode fixed (requested mode: %s)\n",
                out->fixed_samples, delay_mode_name(out->mode));
        return 2;
    }
    if (out->mode == AEC_DELAY_FIXED &&
        (!have_fixed || out->fixed_samples < 0)) {
        fprintf(stderr,
                "4ch_alignulcnet: --delay-mode fixed needs "
                "--fixed-delay <samples> >= 0 (requested: %d)\n",
                have_fixed ? out->fixed_samples : -1);
        return 2;
    }
    if (out->mode == AEC_DELAY_MATCHED) {
        if (out->num_filters < 1 || out->num_filters > DA_NUM_FILTERS) {
            fprintf(stderr,
                    "4ch_alignulcnet: --delay-num-filters %d is out of range "
                    "(accepted: 1..%d; 0 does not mean 'off' -- use "
                    "--delay-mode fixed or external instead)\n",
                    out->num_filters, DA_NUM_FILTERS);
            return 2;
        }
    } else if (have_num_filters && out->num_filters != DA_NUM_FILTERS) {
        fprintf(stderr,
                "4ch_alignulcnet: --delay-num-filters %d is only valid "
                "with --delay-mode matched (requested mode: %s, which "
                "builds no matched bank and requires n == %d)\n",
                out->num_filters, delay_mode_name(out->mode),
                DA_NUM_FILTERS);
        return 2;
    }
    return 0;
}

/* Replace with the product's stateless accelerator invocation. */
static int run_accelerator(void *user,
                           const UlcnetModelIoInputs *inputs,
                           UlcnetModelIoOutputs *outputs) {
    (void)user;
    (void)inputs;
    (void)outputs;
    return -1; /* TODO(board): write every output tensor, then return 0. */
}

int main(int argc, char** argv) {
    AudioPipeline4ChConfig cfg;
    DelayProfile profile;
    UlcnetAcceleratorAdapter *adapter;
    AudioPipeline4ChUlcnet *pipeline;
    UlcnetModel model;
    void *adapter_pool = NULL;
    size_t adapter_bytes;
    size_t adapter_alignment;
    AudioPipeline4ChUlcnetMemReq req;
    float microphones[256 * 4] = {0};
    float far[256] = {0};
    float output[256];
    int hop;
    int index;
    int rc;
    /* A product reads this from the deployed graph's ONNX metadata
     * ('far_input_mode', whose strings ulcnet_far_input_mode_name() mirrors)
     * instead of hard-coding it; every currently exported graph records
     * raw_far (explicit ALIGNED export override pending). */
    const int deployed_far_input_mode = ULCNET_FAR_RAW;

    rc = parse_delay_profile(argc, argv, &profile);
    if (rc != 0) return rc == 1 ? 0 : rc;

    cfg = audio_pipeline_4ch_ulcnet_default_config();
    cfg.core.delay_mode = profile.mode;
    cfg.core.delay_num_filters = profile.num_filters;
    cfg.core.fixed_delay_samples = profile.fixed_samples;

    /* Print the RESOLVED profile with the pool the SAME config actually
     * costs -- query and init must agree on one (sample_rate, hop, n). */
    if (audio_pipeline_4ch_ulcnet_get_mem_requirements(&cfg, &req) != 0) {
        fprintf(stderr,
                "4ch_alignulcnet: pool query rejected the delay profile "
                "(mode=%s n=%d fixed_delay=%d)\n",
                delay_mode_name(profile.mode), profile.num_filters,
                profile.fixed_samples);
        return 1;
    }
    printf("4ch_alignulcnet: delay profile mode=%s n=%d fixed_delay=%d "
           "(one shared estimator) -> pool %llu bytes (align %lu)\n",
           delay_mode_name(profile.mode), profile.num_filters,
           profile.fixed_samples, (unsigned long long)req.bytes,
           (unsigned long)req.alignment);

    if (ulcnet_accelerator_adapter_get_mem_size(
            8, &adapter_bytes, &adapter_alignment) != 0 ||
        posix_memalign(&adapter_pool, adapter_alignment, adapter_bytes) != 0) {
        return 1;
    }
    adapter = ulcnet_accelerator_adapter_init(
        adapter_pool, adapter_bytes, 8, deployed_far_input_mode,
        run_accelerator, NULL);
    pipeline = audio_pipeline_4ch_ulcnet_create(&cfg);
    if (!adapter || !pipeline) {
        audio_pipeline_4ch_ulcnet_destroy(pipeline);
        free(adapter_pool);
        return 1;
    }

    model = ulcnet_accelerator_adapter_model(adapter);
    if (audio_pipeline_4ch_ulcnet_set_model(pipeline, &model) != 0 ||
        audio_pipeline_4ch_ulcnet_set_far_input_mode(
            pipeline, ULCNET_FAR_RAW) != 0) {
        /* The pipeline TU has no stdio, so the far-contract disagreement it
         * rejects is named here, where both values are in hand. */
        fprintf(stderr,
                "4ch_alignulcnet: model/far-mode install failed "
                "(pipeline far_input_mode=%s, checkpoint far_input_mode=%s)\n",
                ulcnet_far_input_mode_name(
                    audio_pipeline_4ch_ulcnet_far_input_mode(pipeline)),
                ulcnet_far_input_mode_name(
                    model.io_descriptor ? model.io_descriptor->far_input_mode
                                        : -1));
        audio_pipeline_4ch_ulcnet_destroy(pipeline);
        free(adapter_pool);
        return 1;
    }

    for (hop = 0; hop < 4; ++hop) {
        if (audio_pipeline_4ch_ulcnet_process_with_activity(
                pipeline, microphones, far, 0, output) != 0) {
            audio_pipeline_4ch_ulcnet_destroy(pipeline);
            free(adapter_pool);
            return 1;
        }
        for (index = 0; index < 256; ++index) {
            if (!isfinite(output[index])) {
                audio_pipeline_4ch_ulcnet_destroy(pipeline);
                free(adapter_pool);
                return 1;
            }
        }
    }

    audio_pipeline_4ch_ulcnet_destroy(pipeline);
    free(adapter_pool);
    puts("4ch_alignulcnet: fail-open board skeleton PASS");
    return 0;
}
