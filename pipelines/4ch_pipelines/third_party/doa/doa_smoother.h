#ifndef DOA_SMOOTHER_H
#define DOA_SMOOTHER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int switch_consec;
    float angle_tol;
} DOA_Smoother_Config;

typedef struct {
    int switch_consec;
    float angle_tol;
    float null_value;

    float last;
    float pending;
    int cnt;
    int initialized;
} DOA_Smoother;

float doa_smoother_update(DOA_Smoother* s, float doa_raw, int vad);

#ifdef __cplusplus
}
#endif

#endif
