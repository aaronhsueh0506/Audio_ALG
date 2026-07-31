#ifndef DOA_SMOOTHER_H
#define DOA_SMOOTHER_H

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

DOA_Smoother* doa_smoother_create(const DOA_Smoother_Config* cfg);
float doa_smoother_update(DOA_Smoother* s, float doa_raw, int vad);

#endif
