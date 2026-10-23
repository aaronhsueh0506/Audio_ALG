#include <stddef.h>

#include "se_config.h"

int se_grid_resolve(int sample_rate, int requested_fft, int allow_8k, SeGrid* out) {
    int fft = requested_fft;
    if (!out) return -1;
    if (fft == 0)
        fft = (sample_rate == 48000) ? 1024
            : (sample_rate == 16000) ? 256
            : (sample_rate == 8000 && allow_8k) ? 256 : 0;
    if (!((sample_rate == 8000 && allow_8k && fft == 256) ||
          (sample_rate == 16000 && (fft == 256 || fft == 512)) ||
          (sample_rate == 48000 && fft == 1024)))
        return -1;
    out->fft_size = fft;
    out->hop_size = fft / 2;
    out->n_freqs = fft / 2 + 1;
    return 0;
}

int se_grid_fixed(int sample_rate, int requested_fft, int rate, int fft, SeGrid* out) {
    if (!out || sample_rate != rate) return -1;
    if (requested_fft != 0 && requested_fft != fft) return -1;
    out->fft_size = fft;
    out->hop_size = fft / 2;
    out->n_freqs = fft / 2 + 1;
    return 0;
}

static int preset_valid(AecPreset p) {
    return p == AEC_PRESET_MILD || p == AEC_PRESET_BALANCED || p == AEC_PRESET_AGGRESSIVE ||
           p == AEC_PRESET_NO_FLOOR;
}

/* MATCHED sizes its bank, FIXED pins a bounded sample count (two minutes at the
 * request's rate), EXTERNAL_ALIGNED takes neither. */
static int delay_request_valid(const SeRequest* r) {
    switch (r->delay_mode) {
        case AEC_DELAY_MATCHED:
            return r->delay_num_filters >= 1 && r->delay_num_filters <= DA_NUM_FILTERS &&
                   r->fixed_delay_samples == -1;
        case AEC_DELAY_FIXED:
            return r->delay_num_filters == DA_NUM_FILTERS && r->fixed_delay_samples >= 0 &&
                   (long)r->fixed_delay_samples <= 120L * (long)r->sample_rate;
        case AEC_DELAY_EXTERNAL_ALIGNED:
            return r->delay_num_filters == DA_NUM_FILTERS && r->fixed_delay_samples == -1;
        default:
            return 0;
    }
}

int se_derive(const SeRequest* req, const SeRules* rules,
              SeGrid* grid, AecConfig* aec) {
    if (!req || !rules || !grid || !aec) return -1;
    if (rules->fixed_rate > 0) {
        if (se_grid_fixed(req->sample_rate, req->fft_size, rules->fixed_rate,
                          rules->fixed_fft, grid) != 0) return -1;
    } else if (se_grid_resolve(req->sample_rate, req->fft_size, rules->allow_8k, grid) != 0) {
        return -1;
    }
    if (!preset_valid(req->aec_preset)) return -1;
    if (rules->with_nr && !anr_nr_mode_is_valid(req->nr_mode)) return -1;
    if (req->filter_length < 0 || req->filter_length > SE_FILTER_LENGTH_MAX) return -1;
    if (rules->check_delay && !delay_request_valid(req)) return -1;

    aec_config_from_preset(aec, req->aec_preset, req->sample_rate);
    aec->fft_size = grid->fft_size;
    if (req->filter_length > 0) aec->filter_length = req->filter_length;
    if (rules->lane_aligned) {
        aec->delay_mode = AEC_DELAY_EXTERNAL_ALIGNED;
        aec->delay_num_filters = DA_NUM_FILTERS;
        aec->fixed_delay_samples = -1;
        aec->enable_delay_est = 0;
    } else {
        aec->delay_mode = req->delay_mode;
        aec->delay_num_filters = req->delay_num_filters;
        aec->fixed_delay_samples = req->fixed_delay_samples;
        aec->enable_delay_est = req->delay_mode == AEC_DELAY_MATCHED ? 1 : 0;
    }
    aec->enable_res = 0;
    aec->return_res_context = 1;
    if (rules->spatial_context != SE_CONTEXT_PRESET)
        aec->spatial_linear_context = rules->spatial_context;
    return 0;
}
