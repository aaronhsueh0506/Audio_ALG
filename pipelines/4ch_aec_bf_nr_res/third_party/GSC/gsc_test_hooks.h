#ifndef GSC_TEST_HOOKS_H
#define GSC_TEST_HOOKS_H

#include "gsc.h"

/* Internal oracle for scalar-vs-dispatch state-transition tests. It exists
 * only in the GSC_TESTING object and is absent from deployable libgsc.a;
 * production code must call gsc_process_with_weights(). */
void gsc_test_process_with_weights_scalar_rls(
    GSC* g,
    const Complex* const* X,
    float doa_s,
    int allow_adapt_in,
    const int* mask,
    Complex* gsc_out,
    Complex* effective_weights);

#endif
