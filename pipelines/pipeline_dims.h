/**
 * pipeline_dims.h — shared frame-geometry derivation for the AEC(linear) ->
 * NR -> RES pipeline pair (aec_nr_pipeline.c malloc / aec_nr_pipeline_static.c
 * static-memory). ONE definition of compute_frame_dims() used by BOTH TUs so
 * the two can never diverge again (M6, multi-rate campaign, review F01).
 *
 * Prior bug (the "8 kHz FFT mismatch"): the malloc pipeline seeded its
 * fft_sz doubling loop at a hardcoded 512 (`int fft_sz = 512; while (fft_sz <
 * frame_sz) fft_sz *= 2;`), which happens to equal next-pow2(frame_sz) for
 * every frame_sz > 256 (sr >= ~12.8 kHz) but OVERSHOOTS below that: at 8 kHz
 * frame_sz=160, so the loop never doubles and fft_sz stays 512 (257 bins)
 * while the AEC's own internal grid (aec.c next_pow2(block_size)) correctly
 * lands on 256 (129 bins) — every per-bin loop over the AEC's K=129-length
 * seam arrays (ctx.error_spec/res_gain/r2/comfort_noise) then reads/writes up
 * to 257, well out of bounds.
 *
 * compute_frame_dims() below seeds the doubling loop at 1 (true
 * next-pow2(frame_sz)), matching the AEC's own derivation exactly at EVERY
 * sample rate, and is IDENTICAL to the old hardcoded-512 result whenever
 * frame_sz > 256 (512 @ 16 kHz, 1024 @ 48 kHz) — so this is a strict fix with
 * zero risk to the byte-identical requirement at the rates already verified.
 */

#ifndef PIPELINE_DIMS_H
#define PIPELINE_DIMS_H

static inline void compute_frame_dims(int sr, int* o_hop, int* o_frame_sz,
                                       int* o_fft_sz, int* o_n_freqs) {
    int hop      = (int)(0.01f * sr);
    int frame_sz = 2 * hop;
    int fft_sz   = 1;
    while (fft_sz < frame_sz) fft_sz *= 2;
    *o_hop = hop; *o_frame_sz = frame_sz; *o_fft_sz = fft_sz;
    *o_n_freqs = fft_sz / 2 + 1;
}

#endif /* PIPELINE_DIMS_H */
