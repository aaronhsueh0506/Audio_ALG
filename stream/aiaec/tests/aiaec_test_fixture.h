/* Host test fixture of the aiaec suites: the aligned pool allocator. Header-only; not part of the library. */

#ifndef AIAEC_TEST_FIXTURE_H
#define AIAEC_TEST_FIXTURE_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 16-byte aligned block filled with 0xA5, so a pool the library fails to initialise shows up; exits on failure. */
static inline void *alloc_aligned(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 16, bytes) != 0) { fprintf(stderr, "alloc fail\n"); exit(1); }
    memset(p, 0xA5, bytes);
    return p;
}

#endif /* AIAEC_TEST_FIXTURE_H */
