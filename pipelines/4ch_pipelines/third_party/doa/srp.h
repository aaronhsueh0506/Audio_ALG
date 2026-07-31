#ifndef SRP_H
#define SRP_H

#include "fft_wrapper.h"
#include "doa_smoother.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ===================== geometry ===================== */
typedef struct {
    int M;
    float* x;
    float* y;
} ArrayGeometry;

/* ===================== forward declaration ===================== */
typedef struct SRP SRP;

/* ===================== config ===================== */
typedef struct {
    int M;
    int F;
    int num_angles;

    float sr;
    float NFFT;
    float c;

    /* SRP frequency band */
    float low_freq;
    float high_freq;

    /* smoothing config */
    int enable_smoothing;
    int switch_consec;
    float angle_tol;

    int update_interval;

} SRP_Config;

/* ===================== SRP struct ===================== */
struct SRP {
    int M;
    int F;
    int num_angles;

    float* angles;
    Complex*** a_array;
    float* S_theta;

    int f_start;
    int f_end;

    int enable_smoothing;
    DOA_Smoother smoother;

    /* for post gain */
    int* bin_best_idx;   /* length = F, each bin's best angle index */

    /* DOA output state */
    float doa_raw;
    float doa_s;

    int update_interval;
    int frame_counter;

    float last_doa_raw;
    float last_doa_s;

    /* unique-pair SRP precompute
     * pair_steer[a][p][f] = conj(a_array[a][i][f]) * a_array[a][j][f]
     */
    int num_pairs;
    int* pair_i;
    int* pair_j;
    Complex*** pair_steer;   /* [num_angles][num_pairs][F] */
    Complex** pair_phat;     /* [num_pairs][F], one frame scratch */
    float* score_scratch;         /* [F], one candidate angle */
    float* best_score;            /* [F], best candidate seen so far */
};

/* ===================== angle / steering API ===================== */
float* srp_create_uniform_angles(int num_angles);
Complex*** srp_build_steering(
    const SRP_Config* cfg,
    const ArrayGeometry* geom,
    const float* angles
);
void srp_destroy_steering(
    Complex*** steering,
    int num_angles,
    int microphones
);

/*helper*/
int srp_angle_to_index(SRP* s, float doa_rad);

/* ===================== SRP create ===================== */
SRP* srp_create(
    const SRP_Config* cfg,
    float* angles,
    Complex*** a_array
);

SRP* srp_create_from_geometry(
    const SRP_Config* cfg,
    const ArrayGeometry* geom
);

/* ===================== SRP ===================== */
void srp(SRP* s, Complex** X,  const int* mask);

/* ===================== SRP to DOA ===================== */
float srp2doa(SRP* s);

/* ===================== DOA ===================== */
void doa_step(SRP* srp,
              Complex** X,
              const int* mask,
              int vad_raw,
              int vad_out);

float doa_get_raw(const SRP* srp);
float doa_get_smooth(const SRP* srp);
/* No new analysis frame: report raw=NAN while holding the smooth DOA. */
void srp_hold(SRP* s);
void srp_reset(SRP* s);
/* ===================== destroy ===================== */
void srp_destroy(SRP* s);

#ifdef __cplusplus
}
#endif

#endif
