/* Prints the AEC configuration the mono Align-ULCNet lane deploys, one key=value per line, for
 * tests/test_deployed_frontend_contract.py to compare against the training front-end contract
 * (dataset_gen/linear_aec.py LinearAecContract).
 *
 * The lane is not copied here: pipelines/mono/alignulcnet/main.c is compiled into this
 * driver (its main() renamed), so parse_options() and derive_lane() are the deployed functions
 * themselves. Values are read from the initialized instance, i.e. after lib/aec has resolved
 * the config. Grid: the example's own option, `--sample-rate <rate>` (default 16000), passed
 * to this driver.
 *
 * formed_hop_cng_invariant runs the deployed lane and the same lane with enable_cng = 0 on one
 * deterministic echo stimulus and reports whether formed_hop (the seam the stage feeds the
 * network) is byte-identical on every hop and comfort_noise is NULL; it is what allows the lane's
 * inherited enable_cng = 1 against the contract's False. */
#define main mono_alignulcnet_example_main
#include "mono/alignulcnet/main.c"
#undef main

#define LANE_DUMP_CONTEXT_LINEAR_ONLY 2   /* the gate's code for a lane that stops after the AecState (AEC_RESIDUAL_STATE) */
#define DUMP_HOPS 400
#define DUMP_ECHO_DELAY 37
#define DUMP_HOP_MAX 512   /* the largest hop of the example's grid table */

static Aec* dump_init(const AecConfig* cfg, void** pool) {
    size_t bytes = aec_get_mem_size(cfg);
    *pool = NULL;
    if (bytes == 0 || posix_memalign(pool, 16, bytes) != 0) return NULL;
    return aec_init(*pool, bytes, cfg);
}

static float dump_noise(uint32_t* state) {
    *state = *state * 1664525u + 1013904223u;
    return (float)((int32_t)*state) * (1.0f / 2147483648.0f);
}

/* 1 when formed_hop carries signal and matches byte for byte on every hop and comfort_noise is
 * NULL in both runs, 0 otherwise, -1 when an instance cannot be built or refuses a hop. */
static int formed_hop_cng_invariant(const AecConfig* lane, int hop_size) {
    AecConfig twin = *lane;
    void *pool_a = NULL, *pool_b = NULL;
    Aec *a, *b;
    float ref_line[DUMP_ECHO_DELAY + DUMP_HOP_MAX] = {0};
    float mic[DUMP_HOP_MAX], ref[DUMP_HOP_MAX];
    uint32_t far_seed = 1u, near_seed = 7u;
    int hop, k, same = 1;
    double energy = 0.0;

    if (hop_size < 1 || hop_size > DUMP_HOP_MAX) return -1;
    twin.enable_cng = 0;
    a = dump_init(lane, &pool_a);
    b = dump_init(&twin, &pool_b);
    if (!a || !b) {
        free(pool_a);
        free(pool_b);
        return -1;
    }
    for (hop = 0; hop < DUMP_HOPS && same; ++hop) {
        AecResidualView ra, rb;
        memmove(ref_line, ref_line + hop_size, DUMP_ECHO_DELAY * sizeof(float));
        for (k = 0; k < hop_size; ++k) {
            ref[k] = 0.3f * dump_noise(&far_seed);
            ref_line[DUMP_ECHO_DELAY + k] = ref[k];
            mic[k] = 0.5f * ref_line[k] + 0.003f * dump_noise(&near_seed);
        }
        ra.struct_size = rb.struct_size = sizeof(ra);
        if (aec_process_linear(a, mic, ref) != AEC_OK || aec_process_linear(b, mic, ref) != AEC_OK ||
            aec_process_residual(a, NULL) != AEC_OK || aec_process_residual(b, NULL) != AEC_OK ||
            aec_get_residual(a, &ra) != AEC_OK || aec_get_residual(b, &rb) != AEC_OK) {
            same = -1;
            break;
        }
        same = ra.formed_hop != NULL && rb.formed_hop != NULL &&
               ra.comfort_noise == NULL && rb.comfort_noise == NULL &&
               memcmp(ra.formed_hop, rb.formed_hop, (size_t)hop_size * sizeof(float)) == 0;
        for (k = 0; same && k < hop_size; ++k) energy += (double)ra.formed_hop[k] * ra.formed_hop[k];
    }
    aec_destroy(a);
    aec_destroy(b);
    free(pool_a);
    free(pool_b);
    if (same < 0) return -1;
    /* An all-zero seam would compare equal whatever the flag does. */
    return same && energy > 0.0;
}

int main(int argc, char** argv) {
    ExampleCliDelayOptions profile;
    UlcnetGrid grid;
    AecConfig lane;
    const AecConfig* c;
    void* pool = NULL;
    Aec* aec;
    int grid_index;
    int invariant;

    if (parse_options(argc, argv, &profile, &grid_index) != 0) return 1;
    if (ulcnet_grid_make(&grid, &k_grids[grid_index]) != ULCNET_GRID_OK) return 1;
    derive_lane(&profile, &grid, &lane);
    aec = dump_init(&lane, &pool);
    if (!aec) {
        fprintf(stderr, "deployed_lane_dump: the deployed lane config does not initialize\n");
        free(pool);
        return 1;
    }
    invariant = formed_hop_cng_invariant(&lane, grid.hop);
    if (invariant < 0) {
        aec_destroy(aec);
        free(pool);
        return 1;
    }
    c = &aec->cfg;
    /* The text keys of the contract gate (frontend_gate.py), written for the lane's AecConfig.residual:
     * the lane is stage-driven (aec_process_linear + aec_process_residual, never aec_process), so enable_res is
     * 0; return_res_context is 1 for any residual stage; spatial_linear_context is the stop-level code
     * (STATE = context_linear_only, ESTIMATE = 1, FULL = 0). */
    printf("frontend_contract=%s\n", ULCNET_MODEL_IO_FRONTEND_CONTRACT);
    printf("sample_rate=%d\n", c->sample_rate);
    printf("fft_size=%d\n", c->fft_size);
    printf("hop_size=%d\n", aec_hop_size(aec));
    printf("filter_length=%d\n", c->filter_length);
    printf("n_partitions=%d\n", aec->n_partitions);
    printf("enable_res=%d\n", 0);
    printf("enable_cng=%d\n", c->enable_cng);
    printf("enable_shadow=%d\n", c->enable_shadow);
    printf("enable_highpass=%d\n", c->enable_highpass);
    printf("enable_saturation=%d\n", c->enable_saturation);
    printf("saturation_softclip_ref=%d\n", c->saturation_softclip_ref);
    printf("enable_delay_est=%d\n", c->delay_mode == AEC_DELAY_MATCHED);
    printf("delay_mode=%d\n", (int)c->delay_mode);
    printf("delay_mode_matched=%d\n", (int)AEC_DELAY_MATCHED);
    printf("delay_num_filters=%d\n", c->delay_num_filters);
    printf("fixed_delay_samples=%d\n", c->fixed_delay_samples);
    printf("delay_analysis_every_hop=%d\n", aec->duty.period_hops == 1);
    printf("return_res_context=%d\n", c->residual != AEC_RESIDUAL_OFF);
    printf("spatial_linear_context=%d\n",
           c->residual == AEC_RESIDUAL_STATE ? LANE_DUMP_CONTEXT_LINEAR_ONLY :
           c->residual == AEC_RESIDUAL_ESTIMATE ? 1 : 0);
    printf("context_linear_only=%d\n", LANE_DUMP_CONTEXT_LINEAR_ONLY);
    printf("live_observer_enabled=%d\n", c->live_observer_enabled);
    printf("min_gain_floor_far_active_db=%.9g\n", c->min_gain_floor_far_active_db);
    printf("min_gain_floor_dt_db=%.9g\n", c->min_gain_floor_dt_db);
    printf("min_gain_split_floor_enabled=%d\n", c->min_gain_split_floor_enabled);
    printf("formed_hop_cng_invariant=%d\n", invariant);
    aec_destroy(aec);
    free(pool);
    return 0;
}
