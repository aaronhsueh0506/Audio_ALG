/* ============================================================
 * Drift-guard: 確保 rnnoise_tables_gen.h 裡的編譯期常數表格，
 * 跟原始演算法 (process.c 舊版 ensure_erb()/ensure_window() 的算式,
 * F09 修正前) 逐 byte 相同。
 *
 * 這個測試「獨立複製」一份原始迴圈 (不是呼叫 gen_rnnoise_tables.c 裡的
 * 函式, 而是自己再寫一次), 執行期重新算出四張表格, 跟 header 裡的
 * `static const` 表格 memcmp。這樣就算日後有人手動改壞了
 * rnnoise_tables_gen.h、或者 gen 工具本身的算式跟這裡「本來該有」的
 * 演算法不小心分岔了, 這個測試都能抓到 (只要這份獨立複製沒有跟著錯)。
 *
 * 建置:
 *   cc -O2 -ffp-contract=off -Wall -Wextra test_rnnoise_tables.c -lm -o test_rnnoise_tables && ./test_rnnoise_tables
 * ============================================================ */

#include "process.h"
#include "rnnoise_tables_gen.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double freq2erb(double f) {
    return 9.265 * log(1.0 + f / (24.7 * 9.265));
}
static double erb2freq(double e) {
    return 24.7 * 9.265 * (exp(e / 9.265) - 1.0);
}

static int   ref_nfftborder[RNNOISE_N_BANDS];
static float ref_erb_fwd[RNNOISE_N_BINS][RNNOISE_N_BANDS];
static float ref_erb_inv[RNNOISE_N_BINS][RNNOISE_N_BANDS];
static float ref_hann_win[RNNOISE_WIN_LEN];

/* 獨立複製自原始 ensure_erb() 的計算本體 (F09 修正前的 process.c)。 */
static void ref_compute_erb_tables(void) {
    const int    N  = RNNOISE_N_BANDS;
    const double sr = (double)RNNOISE_SR;
    const double high_lim = sr / 2.0;
    const double bw = high_lim / ((double)RNNOISE_N_FFT / 2.0);

    {
        double e_lo = freq2erb(0.0), e_hi = freq2erb(high_lim);
        double nb[RNNOISE_N_BANDS];
        for (int i = 0; i < N; i++) {
            double e = e_lo + (e_hi - e_lo) * i / (N - 1);
            double cutoff = erb2freq(e);
            nb[i] = floor((cutoff + bw / 2.0) / bw + 0.5);
        }
        for (int i = 0; i < N - 2; i++) {
            if (nb[i + 2] - nb[i] < 2.0)
                nb[i + 2] += 2.0 - (nb[i + 2] - nb[i]);
        }
        nb[0] = 0.0;
        nb[N - 1] = (double)(RNNOISE_N_FFT / 2 + 1);
        for (int i = 0; i < N; i++) ref_nfftborder[i] = (int)nb[i];
    }

    memset(ref_erb_fwd, 0, sizeof(ref_erb_fwd));
    memset(ref_erb_inv, 0, sizeof(ref_erb_inv));
    for (int i = 0; i < N - 1; i++) {
        int lo = ref_nfftborder[i];
        int hi = ref_nfftborder[i + 1];
        int bs = hi - lo;
        for (int j = 0; j < bs; j++) {
            float down = (float)(1.0 - (double)j / (double)bs);
            float up   = (float)((double)j / (double)bs);
            ref_erb_inv[lo + j][i]     = down;
            ref_erb_inv[lo + j][i + 1] = up;
            ref_erb_fwd[lo + j][i]     = down;
            ref_erb_fwd[lo + j][i + 1] = up;
        }
    }
    for (int k = 0; k < RNNOISE_N_BINS; k++) {
        ref_erb_fwd[k][0]     *= 2.0f;
        ref_erb_fwd[k][N - 1] *= 2.0f;
    }
}

/* 獨立複製自原始 ensure_window() 的計算本體。 */
static void ref_compute_window_table(void) {
    for (int i = 0; i < RNNOISE_WIN_LEN; i++) {
        ref_hann_win[i] = sqrtf(0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / RNNOISE_WIN_LEN)));
    }
}

int main(void) {
    ref_compute_erb_tables();
    ref_compute_window_table();

    int ok = 1;

    if (memcmp(ref_nfftborder, rnn_nfftborder, sizeof(ref_nfftborder)) != 0) {
        printf("FAIL: rnn_nfftborder mismatch\n");
        ok = 0;
    }
    if (memcmp(ref_erb_fwd, rnn_erb_fwd, sizeof(ref_erb_fwd)) != 0) {
        printf("FAIL: rnn_erb_fwd mismatch\n");
        ok = 0;
    }
    if (memcmp(ref_erb_inv, rnn_erb_inv, sizeof(ref_erb_inv)) != 0) {
        printf("FAIL: rnn_erb_inv mismatch\n");
        ok = 0;
    }
    if (memcmp(ref_hann_win, rnn_hann_win, sizeof(ref_hann_win)) != 0) {
        printf("FAIL: rnn_hann_win mismatch\n");
        ok = 0;
    }

    if (ok) {
        printf("PASS: rnnoise_tables_gen.h byte-identical to reference algorithm "
               "(nfftborder[%d], erb_fwd[%d][%d], erb_inv[%d][%d], hann_win[%d])\n",
               RNNOISE_N_BANDS, RNNOISE_N_BINS, RNNOISE_N_BANDS,
               RNNOISE_N_BINS, RNNOISE_N_BANDS, RNNOISE_WIN_LEN);
    }

    return ok ? 0 : 1;
}
