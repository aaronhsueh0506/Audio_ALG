/* pipelines/nr_overlay.h -- the NR configuration both AEC+NR pipelines build.
 *
 * Canonical strength preset plus the three overrides these pipelines have always
 * applied on top. It lives here rather than in either pipeline because BOTH
 * construct it and BOTH now recompose it at runtime (their strength setters
 * hand the result to mmse_lsa_reconfigure), so a tuning decision on this
 * overlay has to land in one place or the two products drift apart between
 * releases.
 *
 * Deliberately NOT in NR's public headers: it is this stack's product
 * decision, not something the denoiser owns, and putting it there would widen
 * NR's released API for a caller-side convention.
 */
#ifndef PIPELINES_NR_OVERLAY_H
#define PIPELINES_NR_OVERLAY_H

#include "mmse_lsa_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* This pipeline's NR configuration: the canonical strength preset plus the three
 * overrides it has always applied on top. Extracted so the runtime strength
 * setter can rebuild the SAME composition -- handing the bare preset to
 * mmse_lsa_set_mode() instead would either be refused (its L differs from the
 * override below) or silently revert these. Single source of truth for both
 * construction and reconfiguration. */
static inline MmseLsaConfig pipelines_compose_nr_config(int sample_rate, int fft_size,
                                       int hop_size, MmseLsaNrMode mode) {
    MmseLsaConfig nr_cfg =
        mmse_lsa_config_for_mode_grid(sample_rate, fft_size, mode);
    /* 2026-08-03: was an implicit side effect of the C standalone default
     * (mmse_lsa_default_config_for_grid) also happening to be 0.8f -- that
     * default is now fixed to match Python's own config/v3_2_config.yaml
     * (1.0f, disabled), so this pipeline must set 0.8f explicitly to keep its
     * actual runtime behaviour unchanged. Mirrors audio_pipeline.c (mono) and
     * the deliberate overlay aec_nr_pipeline.py:_build_denoiser documents. */
    nr_cfg.broadband_threshold = 0.8f;
    /* 2026-08-03 A/B decision (824-case VCTK+DEMAND + 90-case AEC blind
     * manifest, see NR/CHANGELOG.md): take mmse_lsa_config_for_mode_grid()'s
     * canonical alpha_d/alpha_attack as-is instead of overriding them back
     * to the old L=150/alpha_d=0.95/alpha_attack=0.3-old-retime tuning --
     * that legacy tuning measured worse on the AEC-residual/double-talk
     * angle that matters for this pipeline. L and alpha_decay are untouched:
     * they already coincide with Python's canonical composition (see
     * audio_pipeline.c's mono twin for the full rationale). */
    nr_cfg.L = mmse_lsa_retime_frames(150, sample_rate, hop_size);
    nr_cfg.alpha_decay = nr_cfg.alpha_g;
    return nr_cfg;
}

#ifdef __cplusplus
}
#endif

#endif /* PIPELINES_NR_OVERLAY_H */
