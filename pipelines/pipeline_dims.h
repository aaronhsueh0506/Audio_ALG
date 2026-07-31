/**
 * pipeline_dims.h -- one no-padding signal-grid resolver shared by the
 * traditional AEC -> NR -> RES C pipeline entry points.
 */

#ifndef PIPELINE_DIMS_H
#define PIPELINE_DIMS_H

/* requested_fft == 0 selects the rate default. Returns 0 on success. */
static inline int compute_frame_dims(int sr, int requested_fft,
                                     int* o_hop, int* o_frame_sz,
                                     int* o_fft_sz, int* o_n_freqs) {
    int fft_sz = requested_fft;
    if (fft_sz == 0) {
        fft_sz = (sr == 48000) ? 1024 : (sr == 16000) ? 512
               : (sr == 8000) ? 256 : 0;
    }

    if (!((sr == 8000 && fft_sz == 256) ||
          (sr == 16000 && (fft_sz == 256 || fft_sz == 512)) ||
          (sr == 48000 && fft_sz == 1024))) {
        return -1;
    }

    *o_hop = fft_sz / 2;
    *o_frame_sz = fft_sz;
    *o_fft_sz = fft_sz;
    *o_n_freqs = fft_sz / 2 + 1;
    return 0;
}

#endif /* PIPELINE_DIMS_H */
