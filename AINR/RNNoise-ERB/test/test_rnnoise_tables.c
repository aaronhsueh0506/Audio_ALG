#define _POSIX_C_SOURCE 200809L
/* ERB drift guard: the matrices of the two raw float32 files (tests/erb_fixture.py by default,
 * or the application's own) against an independent recompute of the triangular ERB filterbank,
 * byte for byte, plus the matrix contracts and the reader's refusals. Library-free.
 * Usage: test_rnnoise_tables ERB_FWD_FILE ERB_INV_FILE */

#include "rnnoise_test_erb.h"

#include <math.h>

/* Second source: these constants are deliberately not read from process.h or config.ini, so a
 * change of either shows up here. */
#define SR 16000
#define N_FFT 512
#define N_BANDS 22
#define MIN_BINS_PER_BAND 2
#define N_BINS (N_FFT / 2 + 1)

static int g_fail;
#define CHECK(name, cond) do { if (!(cond)) { ++g_fail; \
    printf("FAIL [%s] line %d: %s\n", (name), __LINE__, #cond); } } while (0)

static double freq2erb(double f) {
    return 9.265 * log(1.0 + f / (24.7 * 9.265));
}
static double erb2freq(double e) {
    return 24.7 * 9.265 * (exp(e / 9.265) - 1.0);
}

static int   ref_border[N_BANDS];
static float ref_fwd[N_BINS][N_BANDS];   /* bin-major */
static float ref_inv[N_BANDS][N_BINS];   /* band-major */

static void ref_compute_erb_matrices(void) {
    const int    N  = N_BANDS;
    const double high_lim = (double)SR / 2.0;
    const double bw = high_lim / ((double)N_FFT / 2.0);
    double e_lo = freq2erb(0.0), e_hi = freq2erb(high_lim);
    double ideal[N_BANDS];
    double nb0 = 0.0;

    for (int i = 0; i < N; i++) {
        double e = e_lo + (e_hi - e_lo) * i / (N - 1);
        ideal[i] = floor((erb2freq(e) + bw / 2.0) / bw + 0.5);
    }
    ref_border[0] = 0;
    for (int i = 1; i < N; i++) {
        double nxt = ideal[i];
        if (nxt < nb0 + MIN_BINS_PER_BAND) nxt = nb0 + MIN_BINS_PER_BAND;
        if (nxt > (double)N_BINS) nxt = (double)N_BINS;
        ref_border[i] = (int)nxt;
        nb0 = nxt;
    }
    ref_border[N - 1] = N_BINS;

    memset(ref_fwd, 0, sizeof(ref_fwd));
    memset(ref_inv, 0, sizeof(ref_inv));
    for (int i = 0; i < N - 1; i++) {
        int lo = ref_border[i];
        int bs = ref_border[i + 1] - lo;
        for (int j = 0; j < bs; j++) {
            float down = (float)(1.0 - (double)j / (double)bs);
            float up   = (float)((double)j / (double)bs);
            ref_inv[i][lo + j]     = down;
            ref_inv[i + 1][lo + j] = up;
            ref_fwd[lo + j][i]     = down;
            ref_fwd[lo + j][i + 1] = up;
        }
    }
    for (int k = 0; k < N_BINS; k++) {
        ref_fwd[k][0]     *= 2.0f;
        ref_fwd[k][N - 1] *= 2.0f;
    }
}

/* Mismatching elements of two float arrays compared by bit pattern. */
static size_t count_diff(const float *a, const float *b, size_t n, size_t *first) {
    size_t diff = 0;
    for (size_t i = 0; i < n; i++) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            if (diff == 0) *first = i;
            diff++;
        }
    }
    return diff;
}

static void check_equal(const char *name, const float *file, const float *ref, size_t n) {
    size_t first = 0;
    size_t diff = count_diff(file, ref, n, &first);
    if (diff != 0) {
        ++g_fail;
        printf("FAIL [%s] %zu of %zu elements differ, first at %zu: file=%.9g recompute=%.9g\n",
               name, diff, n, first, (double)file[first], (double)ref[first]);
    }
}

static void check_contracts(const float *fwd, const float *inv) {
    for (size_t i = 0; i < (size_t)N_BINS * N_BANDS; i++) {
        CHECK("finite.forward", isfinite(fwd[i]));
        CHECK("finite.inverse", isfinite(inv[i]));
    }
    /* Each bin is one crossfade whose weights add to 1: the partition of unity. */
    for (int k = 0; k < N_BINS; k++) {
        double sum = 0.0;
        for (int b = 0; b < N_BANDS; b++) sum += (double)inv[(size_t)b * N_BINS + k];
        if (fabs(sum - 1.0) > 1e-5) {
            ++g_fail;
            printf("FAIL [partition_of_unity] bin %d sums to %.9f\n", k, sum);
        }
    }
    /* Forward is the inverse with the two end bands doubled (exact: a power-of-two scale). */
    for (int k = 0; k < N_BINS; k++) {
        for (int b = 0; b < N_BANDS; b++) {
            float inv_kb = inv[(size_t)b * N_BINS + k];
            float want = (b == 0 || b == N_BANDS - 1) ? 2.0f * inv_kb : inv_kb;
            if (fwd[(size_t)k * N_BANDS + b] != want) {
                ++g_fail;
                printf("FAIL [forward_vs_inverse] bin %d band %d: %.9g, expected %.9g\n",
                       k, b, (double)fwd[(size_t)k * N_BANDS + b], (double)want);
                return;
            }
        }
    }
}

/* The reader accepts the pair and refuses a short, a long, an empty or a missing file. */
static int read_variant(const unsigned char *fwd_raw, size_t fwd_bytes,
                        const unsigned char *inv_raw, size_t inv_bytes, int n_bins, int n_erb) {
    FILE *ff = tmpfile();
    FILE *fi = tmpfile();
    RnnErb erb;
    int status = RNN_ERB_ERR_IO;
    if (ff && fi &&
        (fwd_bytes == 0 || fwrite(fwd_raw, 1u, fwd_bytes, ff) == fwd_bytes) &&
        (inv_bytes == 0 || fwrite(inv_raw, 1u, inv_bytes, fi) == inv_bytes)) {
        rewind(ff);
        rewind(fi);
        status = rnn_erb_read(ff, fi, n_bins, n_erb, &erb);
        if (status == RNN_ERB_OK) rnn_erb_free(&erb);
    }
    if (ff) fclose(ff);
    if (fi) fclose(fi);
    return status;
}

static unsigned char *slurp(const char *path, size_t bytes) {
    unsigned char *raw = (unsigned char *)malloc(bytes + 2u);
    FILE *f = fopen(path, "rb");
    size_t got = (f && raw) ? fread(raw, 1u, bytes + 1u, f) : 0u;
    if (f) fclose(f);
    if (!raw || got != bytes) { free(raw); return NULL; }
    return raw;
}

static void check_reader(const char *fwd_path, const char *inv_path) {
    const size_t bytes = (size_t)N_BINS * N_BANDS * sizeof(float);
    unsigned char *fwd = slurp(fwd_path, bytes);
    unsigned char *inv = slurp(inv_path, bytes);
    RnnErb erb;
    CHECK("reader.file_size", fwd && inv);
    if (!fwd || !inv) { free(fwd); free(inv); return; }
    fwd[bytes] = inv[bytes] = 0u;   /* the byte a long file would carry */

    CHECK("reader.ok", read_variant(fwd, bytes, inv, bytes, N_BINS, N_BANDS) == RNN_ERB_OK);
    CHECK("reader.path_missing", rnn_erb_read_paths("/nonexistent/erb_fwd.bin", inv_path,
          N_BINS, N_BANDS, &erb) == RNN_ERB_ERR_IO);
    CHECK("reader.inv_path_missing", rnn_erb_read_paths(fwd_path, "/nonexistent/erb_inv.bin",
          N_BINS, N_BANDS, &erb) == RNN_ERB_ERR_IO);
    CHECK("reader.fwd_empty", read_variant(fwd, 0u, inv, bytes, N_BINS, N_BANDS) == RNN_ERB_ERR_SIZE);
    CHECK("reader.inv_empty", read_variant(fwd, bytes, inv, 0u, N_BINS, N_BANDS) == RNN_ERB_ERR_SIZE);
    CHECK("reader.fwd_one_short", read_variant(fwd, bytes - 1u, inv, bytes, N_BINS, N_BANDS) ==
          RNN_ERB_ERR_SIZE);
    CHECK("reader.inv_one_short", read_variant(fwd, bytes, inv, bytes - 1u, N_BINS, N_BANDS) ==
          RNN_ERB_ERR_SIZE);
    CHECK("reader.fwd_one_float_short", read_variant(fwd, bytes - 4u, inv, bytes, N_BINS, N_BANDS) ==
          RNN_ERB_ERR_SIZE);
    CHECK("reader.fwd_one_long", read_variant(fwd, bytes + 1u, inv, bytes, N_BINS, N_BANDS) ==
          RNN_ERB_ERR_SIZE);
    CHECK("reader.inv_one_long", read_variant(fwd, bytes, inv, bytes + 1u, N_BINS, N_BANDS) ==
          RNN_ERB_ERR_SIZE);
    CHECK("reader.grid_larger", read_variant(fwd, bytes, inv, bytes, N_BINS, N_BANDS + 1) ==
          RNN_ERB_ERR_SIZE);
    CHECK("reader.grid_smaller", read_variant(fwd, bytes, inv, bytes, N_BINS, N_BANDS - 1) ==
          RNN_ERB_ERR_SIZE);
    free(fwd);
    free(inv);
}

int main(int argc, char **argv) {
    RnnErb erb;
    int status;

    if (argc != 3) {
        fprintf(stderr, "usage: %s ERB_FWD_FILE ERB_INV_FILE\n", argv[0]);
        return 2;
    }
    status = rnn_erb_read_paths(argv[1], argv[2], N_BINS, N_BANDS, &erb);
    if (status != RNN_ERB_OK) {
        printf("FAIL: %s and %s are not matching raw float32 files (status %d)\n",
               argv[1], argv[2], status);
        return 1;
    }
    ref_compute_erb_matrices();
    check_equal("forward", erb.fwd, &ref_fwd[0][0], (size_t)N_BINS * N_BANDS);
    check_equal("inverse", erb.inv, &ref_inv[0][0], (size_t)N_BANDS * N_BINS);
    check_contracts(erb.fwd, erb.inv);
    check_reader(argv[1], argv[2]);
    rnn_erb_free(&erb);

    if (g_fail != 0) return 1;
    printf("PASS: ERB matrix files match the independent recompute (forward [%d][%d] bin-major, "
           "inverse [%d][%d] band-major); matrix contracts and reader refusals hold\n",
           N_BINS, N_BANDS, N_BANDS, N_BINS);
    return 0;
}
