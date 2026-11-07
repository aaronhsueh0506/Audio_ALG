/* Library-private: shared by dfn3_model_io.c and dfn3_prepost.c; not part of the API. */
#ifndef DFN3_MODEL_IO_PRIV_H
#define DFN3_MODEL_IO_PRIV_H

#include <stddef.h>

/* The registration rule of the recurrent state: the h_gru pair and the df_convp_history pair
 * are both set, or all four values are NULL/0. Returns 0 when valid, -1 otherwise. */
static inline int dfn3_registration_check(const float *gru_hidden, size_t gru_elements,
                                          const float *convp, size_t convp_elements)
{
    const int gru_set = gru_hidden != NULL || gru_elements != 0u;
    const int convp_set = convp != NULL || convp_elements != 0u;
    const int gru_ok = gru_hidden != NULL && gru_elements != 0u;
    const int convp_ok = convp != NULL && convp_elements != 0u;
    return (!gru_set && !convp_set) || (gru_ok && convp_ok) ? 0 : -1;
}

#endif /* DFN3_MODEL_IO_PRIV_H */
