#ifndef RNNOISE_TEST_ERB_H
#define RNNOISE_TEST_ERB_H
/* Loads the two ERB matrix files an application provides: erb_fwd.bin (bin-major
 * [n_bins][n_erb]) and erb_inv.bin (band-major [n_erb][n_bins]), headerless little-endian
 * float32, each exactly n_bins * n_erb values. Library-free.
 * Include after defining _POSIX_C_SOURCE 200809L. */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline void *rnn_test_aligned(size_t bytes)
{
    void *mem = NULL;
    if (bytes == 0 || posix_memalign(&mem, 16u, bytes) != 0) return NULL;
    memset(mem, 0, bytes);
    return mem;
}

enum {
    RNN_ERB_OK = 0,
    RNN_ERB_ERR_IO = -1,
    RNN_ERB_ERR_SIZE = -2
};

typedef struct RnnErb {
    const float *fwd;   /* 16-byte aligned, bin-major [n_bins][n_erb] */
    const float *inv;   /* 16-byte aligned, band-major [n_erb][n_bins] */
} RnnErb;

static inline void rnn_erb_free(RnnErb *erb)
{
    free((void *)erb->fwd);
    free((void *)erb->inv);
    memset(erb, 0, sizeof(*erb));
}

/* Exactly `count` floats from f: one more or fewer bytes is RNN_ERB_ERR_SIZE. */
static inline int rnn_erb_read_file(FILE *f, size_t count, const float **out)
{
    const size_t bytes = count * sizeof(float);
    float *block;
    size_t got;
    *out = NULL;
    if (!f || count == 0) return RNN_ERB_ERR_IO;
    block = (float *)rnn_test_aligned(bytes + sizeof(float));
    if (!block) return RNN_ERB_ERR_IO;
    got = fread(block, 1u, bytes + 1u, f);
    if (ferror(f)) { free(block); return RNN_ERB_ERR_IO; }
    if (got != bytes) { free(block); return RNN_ERB_ERR_SIZE; }
    *out = block;
    return RNN_ERB_OK;
}

static inline int rnn_erb_read(FILE *fwd_file, FILE *inv_file, int n_bins, int n_erb, RnnErb *out)
{
    int status;
    size_t count;
    memset(out, 0, sizeof(*out));
    if (n_bins <= 0 || n_erb <= 0) return RNN_ERB_ERR_IO;
    count = (size_t)n_bins * (size_t)n_erb;
    status = rnn_erb_read_file(fwd_file, count, &out->fwd);
    if (status == RNN_ERB_OK) status = rnn_erb_read_file(inv_file, count, &out->inv);
    if (status != RNN_ERB_OK) rnn_erb_free(out);
    return status;
}

static inline int rnn_erb_read_paths(const char *fwd_path, const char *inv_path,
                                     int n_bins, int n_erb, RnnErb *out)
{
    FILE *ff = fwd_path ? fopen(fwd_path, "rb") : NULL;
    FILE *fi = inv_path ? fopen(inv_path, "rb") : NULL;
    int status = rnn_erb_read(ff, fi, n_bins, n_erb, out);
    if (ff) fclose(ff);
    if (fi) fclose(fi);
    return status;
}

#endif
