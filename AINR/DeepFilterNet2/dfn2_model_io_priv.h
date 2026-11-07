/* Library-private: shared by dfn2_model_io.c and dfn2_prepost.c; not part of the API. */
#ifndef DFN2_MODEL_IO_PRIV_H
#define DFN2_MODEL_IO_PRIV_H

#include <stddef.h>

/* The registration rule of the recurrent state: the h_gru pair is set and, in the full
 * variant, the df_convp_history pair too, or all values are NULL/0; with DFN2_MASK_ONLY any
 * df_convp_history value is invalid. Returns 0 when valid, -1 otherwise. */
static inline int dfn2_registration_check(const float *gru_hidden, size_t gru_elements,
                                          const float *convp, size_t convp_elements)
{
    const int gru_set = gru_hidden != NULL || gru_elements != 0u;
    const int gru_ok = gru_hidden != NULL && gru_elements != 0u;
    const int convp_set = convp != NULL || convp_elements != 0u;
#ifdef DFN2_MASK_ONLY
    return !convp_set && (!gru_set || gru_ok) ? 0 : -1;
#else
    const int convp_ok = convp != NULL && convp_elements != 0u;
    return (!gru_set && !convp_set) || (gru_ok && convp_ok) ? 0 : -1;
#endif
}

#endif /* DFN2_MODEL_IO_PRIV_H */
