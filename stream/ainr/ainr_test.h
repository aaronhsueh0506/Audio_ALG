/* Test-only entry points of ainr.c: init with an explicit (e.g. synthetic) ERB file. No product code
 * includes this header; the customer interface is ainr.h. Not in a NO_STDIO archive (file-mode). */

#ifndef AINR_TEST_H
#define AINR_TEST_H

#include "ainr.h"

#ifdef __cplusplus
extern "C" {
#endif

/* As ainr_init(), with the ERB path given instead of the release-asset path of the grid. */
int ainr_init_from_erb_file(const AinrConfig *cfg, const char *erb_path, void *pool, size_t pool_bytes,
                            AinrCtx **ctx, AinrInfo *info);

/* As ainr_init_from_erb_file(), for the PCM -> spectrum family. */
int ainr_init_spec_from_erb_file(const AinrConfig *cfg, const char *erb_path, void *pool, size_t pool_bytes,
                                 AinrSpecCtx **ctx, AinrInfo *info);

/* As ainr_init_from_erb_file(), for the spectrum -> spectrum family. */
int ainr_init_freq_from_erb_file(const AinrConfig *cfg, const char *erb_path, void *pool, size_t pool_bytes,
                                 AinrFreqCtx **ctx, AinrInfo *info);

#ifdef __cplusplus
}
#endif

#endif /* AINR_TEST_H */
