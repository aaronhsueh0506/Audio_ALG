/* Prints the AEC configuration the mono Align-ULCNet lane deploys, one key=value per line, for
 * tests/test_deployed_frontend_contract.py to compare against the training front-end contract
 * (dataset_gen/linear_aec.py LinearAecContract).
 *
 * The lane is not copied here: pipelines/ulcnet/mono_alignulcnet/main.c is compiled into this
 * driver (its main() renamed), so parse_delay_profile() with no arguments and derive_lane() are
 * the deployed functions themselves. Values are read from the initialized instance, i.e. after
 * lib/aec has resolved the config. Grid: the ULCNET_MODEL_IO_SR / ULCNET_MODEL_IO_N_FFT the
 * driver is compiled with, as for the example.
 *
 * formed_hop_cng_invariant runs the deployed lane and the same lane with enable_cng = 0 on one
 * deterministic echo stimulus and reports whether formed_hop (the seam the stage feeds the
 * network) is byte-identical on every hop and comfort_noise is NULL; it is what allows the lane's
 * inherited enable_cng = 1 against the contract's False. */
#define main mono_alignulcnet_example_main
#include "mono_alignulcnet/main.c"
#undef main

#define DUMP_HOPS 400
#define DUMP_ECHO_DELAY 37

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
 * NULL in both runs, 0 otherwise, -1 when an instance cannot be built. */
static int formed_hop_cng_invariant(const AecConfig* lane) {
    AecConfig twin = *lane;
    void *pool_a = NULL, *pool_b = NULL;
    Aec *a, *b;
    float ref_line[DUMP_ECHO_DELAY + ULCNET_HOP] = {0};
    float mic[ULCNET_HOP], ref[ULCNET_HOP];
    uint32_t far_seed = 1u, near_seed = 7u;
    int hop, k, same = 1;
    double energy = 0.0;

    twin.enable_cng = 0;
    a = dump_init(lane, &pool_a);
    b = dump_init(&twin, &pool_b);
    if (!a || !b) {
        free(pool_a);
        free(pool_b);
        return -1;
    }
    for (hop = 0; hop < DUMP_HOPS && same; ++hop) {
        AecResContext ra, rb;
        memmove(ref_line, ref_line + ULCNET_HOP, DUMP_ECHO_DELAY * sizeof(float));
        for (k = 0; k < ULCNET_HOP; ++k) {
            ref[k] = 0.3f * dump_noise(&far_seed);
            ref_line[DUMP_ECHO_DELAY + k] = ref[k];
            mic[k] = 0.5f * ref_line[k] + 0.003f * dump_noise(&near_seed);
        }
        aec_process_context(a, mic, ref);
        aec_process_context(b, mic, ref);
        aec_get_res_context(a, &ra);
        aec_get_res_context(b, &rb);
        same = ra.formed_hop != NULL && rb.formed_hop != NULL &&
               ra.comfort_noise == NULL && rb.comfort_noise == NULL &&
               memcmp(ra.formed_hop, rb.formed_hop, ULCNET_HOP * sizeof(float)) == 0;
        for (k = 0; same && k < ULCNET_HOP; ++k) energy += (double)ra.formed_hop[k] * ra.formed_hop[k];
    }
    aec_destroy(a);
    aec_destroy(b);
    free(pool_a);
    free(pool_b);
    /* An all-zero seam would compare equal whatever the flag does. */
    return same && energy > 0.0;
}

int main(void) {
    char prog[] = "mono_alignulcnet";
    char* argv[] = {prog, NULL};
    ExampleCliDelayOptions profile;
    AecConfig lane;
    const AecConfig* c;
    void* pool = NULL;
    Aec* aec;
    int invariant;

    if (parse_delay_profile(1, argv, &profile) != 0) return 1;
    derive_lane(&profile, &lane);
    aec = dump_init(&lane, &pool);
    if (!aec) {
        fprintf(stderr, "deployed_lane_dump: the deployed lane config does not initialize\n");
        free(pool);
        return 1;
    }
    invariant = formed_hop_cng_invariant(&lane);
    if (invariant < 0) {
        aec_destroy(aec);
        free(pool);
        return 1;
    }
    c = &aec->cfg;
    printf("frontend_contract=%s\n", ULCNET_MODEL_IO_FRONTEND_CONTRACT);
    printf("sample_rate=%d\n", c->sample_rate);
    printf("fft_size=%d\n", c->fft_size);
    printf("hop_size=%d\n", aec_hop_size(aec));
    printf("filter_length=%d\n", c->filter_length);
    printf("n_partitions=%d\n", aec->n_partitions);
    printf("enable_res=%d\n", c->enable_res);
    printf("enable_cng=%d\n", c->enable_cng);
    printf("enable_shadow=%d\n", c->enable_shadow);
    printf("enable_highpass=%d\n", c->enable_highpass);
    printf("enable_saturation=%d\n", c->enable_saturation);
    printf("saturation_softclip_ref=%d\n", c->saturation_softclip_ref);
    printf("enable_delay_est=%d\n", c->enable_delay_est);
    printf("delay_mode=%d\n", (int)c->delay_mode);
    printf("delay_mode_matched=%d\n", (int)AEC_DELAY_MATCHED);
    printf("delay_num_filters=%d\n", c->delay_num_filters);
    printf("fixed_delay_samples=%d\n", c->fixed_delay_samples);
    printf("delay_analysis_every_hop=%d\n", aec->duty_period_hops == 1);
    printf("return_res_context=%d\n", c->return_res_context);
    printf("spatial_linear_context=%d\n", c->spatial_linear_context);
    printf("context_linear_only=%d\n", AEC_CONTEXT_LINEAR_ONLY);
    printf("live_observer_enabled=%d\n", c->live_observer_enabled);
    printf("min_gain_floor_far_active_db=%.9g\n", c->min_gain_floor_far_active_db);
    printf("min_gain_floor_dt_db=%.9g\n", c->min_gain_floor_dt_db);
    printf("min_gain_split_floor_enabled=%d\n", c->min_gain_split_floor_enabled);
    printf("formed_hop_cng_invariant=%d\n", invariant);
    aec_destroy(aec);
    free(pool);
    return 0;
}
