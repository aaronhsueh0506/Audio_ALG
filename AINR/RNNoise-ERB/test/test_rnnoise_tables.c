/* ============================================================
 * Drift-guard (兩層 contract):
 *
 * 第一層 (預設, 未定義 RNN_TABLES_PORTABLE 時): 確保 rnnoise_tables_gen.h
 * 裡的編譯期常數表格，跟原始演算法 (process.c 舊版 ensure_erb()/
 * ensure_window() 的算式, F09 修正前) 逐 byte 相同 -- 這是「本機 + pinned
 * CI 工具鏈」的 canonical/bit-exact 檢查, 預設啟用 (即: canonical/
 * bit-exact 是預設值, portable mode 才是 opt-out)。
 *
 * 這個測試「獨立複製」一份原始迴圈 (不是呼叫 gen_rnnoise_tables.c 裡的
 * 函式, 而是自己再寫一次), 執行期重新算出四張表格, 跟 header 裡的
 * `static const` 表格 memcmp。這樣就算日後有人手動改壞了
 * rnnoise_tables_gen.h、或者 gen 工具本身的算式跟這裡「本來該有」的
 * 演算法不小心分岔了, 這個測試都能抓到 (只要這份獨立複製沒有跟著錯)。
 *
 * 第二層 (`-DRNN_TABLES_PORTABLE`, opt-out): 逐 byte memcmp 在不同 libm/
 * 工具鏈/交叉編譯目標下不保證成立 (user 明確拒絕跨工具鏈硬性規定 1 ULP)，
 * 所以這個 mode 改成檢查表格的「數學性質」而非精確位元:
 *   - 四張表都是有限值 (finite)
 *   - nfftborder 嚴格遞增, 兩端點釘死在 0 與 257
 *   - erb_inv 每一列 (每個 frequency bin) 對 band 求和是
 *     partition-of-unity (= 1.0, within a measured tolerance)
 *   - erb_fwd 除了頭尾兩個 band 欄位是 erb_inv 的兩倍以外, 其餘逐一相同
 *   - hann window 落在 [0,1], 且滿足公式本身要求的對稱關係
 *     (w[i]==w[WIN_LEN-i]), 端點 w[0]==0
 * 外加一個 recompute-vs-table 的 max-ULP 量測報告, 用一個「本機實測後
 * 留了大量餘裕」的門檻 (目前 256 ULP) 當作 garbage-detector -- 這不是
 * 位元契約, 只是抓「整個算式錯掉/表格對調/regen 損毀」這種明顯錯誤,
 * 門檻大小由實測值決定 (這台機器上實測是 0, 跟 Layer 1 的逐 byte memcmp
 * 互相印證) 並在程式碼裡註記為「量測校準過的 garbage-detector, 不是
 * 位元契約」。
 *
 * 建置 (兩個 mode 都要在這台機器上跑過, 都要 PASS):
 *   cc -O2 -ffp-contract=off -Wall -Wextra -I. test/test_rnnoise_tables.c -lm -o test_rnnoise_tables && ./test_rnnoise_tables
 *   cc -O2 -ffp-contract=off -Wall -Wextra -I. -DRNN_TABLES_PORTABLE test/test_rnnoise_tables.c -lm -o test_rnnoise_tables_portable && ./test_rnnoise_tables_portable
 * ============================================================ */

#include "process.h"
#include "rnnoise_tables_gen.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

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
        /* DELIBERATE SECOND SOURCE — do not replace with
         * RNNOISE_MIN_BINS_PER_BAND.  This file re-derives the tables
         * independently so that a change to the shared constant shows up as a
         * test failure rather than silently propagating.  Wiring it to the
         * header would make the check compare the generator against itself. */
#define MIN_BINS_PER_BAND 2
        double e_lo = freq2erb(0.0), e_hi = freq2erb(high_lim);
        double ideal[RNNOISE_N_BANDS];
        for (int i = 0; i < N; i++) {
            double e = e_lo + (e_hi - e_lo) * i / (N - 1);
            double cutoff = erb2freq(e);
            ideal[i] = floor((cutoff + bw / 2.0) / bw + 0.5);
        }
        double nb0 = 0.0;
        ref_nfftborder[0] = 0;
        for (int i = 1; i < N; i++) {
            double nxt = ideal[i];
            if (nxt < nb0 + MIN_BINS_PER_BAND) nxt = nb0 + MIN_BINS_PER_BAND;
            if (nxt > (double)(RNNOISE_N_FFT / 2 + 1)) nxt = (double)(RNNOISE_N_FFT / 2 + 1);
            ref_nfftborder[i] = (int)nxt;
            nb0 = nxt;
        }
        ref_nfftborder[N - 1] = RNNOISE_N_FFT / 2 + 1;
#undef MIN_BINS_PER_BAND
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

#ifdef RNN_TABLES_PORTABLE

/* ───────────────────────── Layer 2 helpers (portable mode) ───────────────────────── */

/* ULP distance via the standard "sign-and-magnitude to biased" total-order
 * mapping (same technique googletest's internal::FloatingPoint uses): maps
 * each float's raw bit pattern to a monotonically-ordered unsigned integer
 * so the ordinary unsigned difference between two such mapped values is
 * exactly their ULP distance, including across the +/-0 and sign boundary. */
static uint32_t float_bits_u32(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static uint32_t sam_to_biased(uint32_t sam) {
    if (sam & 0x80000000u) {
        return (uint32_t)(~sam + 1u);
    }
    return 0x80000000u | sam;
}

static uint32_t ulp_distance(float a, float b) {
    uint32_t ba = sam_to_biased(float_bits_u32(a));
    uint32_t bb = sam_to_biased(float_bits_u32(b));
    return (ba >= bb) ? (ba - bb) : (bb - ba);
}

static int check_all_finite(void) {
    int ok = 1;
    for (int k = 0; k < RNNOISE_N_BINS; k++) {
        for (int b = 0; b < RNNOISE_N_BANDS; b++) {
            if (!isfinite(rnn_erb_fwd[k][b])) {
                printf("FAIL: rnn_erb_fwd[%d][%d]=%.9g not finite\n", k, b, (double)rnn_erb_fwd[k][b]);
                ok = 0;
            }
            if (!isfinite(rnn_erb_inv[k][b])) {
                printf("FAIL: rnn_erb_inv[%d][%d]=%.9g not finite\n", k, b, (double)rnn_erb_inv[k][b]);
                ok = 0;
            }
        }
    }
    for (int i = 0; i < RNNOISE_WIN_LEN; i++) {
        if (!isfinite(rnn_hann_win[i])) {
            printf("FAIL: rnn_hann_win[%d]=%.9g not finite\n", i, (double)rnn_hann_win[i]);
            ok = 0;
        }
    }
    return ok;
}

static int check_nfftborder_monotone_pinned(void) {
    int ok = 1;
    const int expect_last = RNNOISE_N_FFT / 2 + 1;  /* 257 */

    if (rnn_nfftborder[0] != 0) {
        printf("FAIL: rnn_nfftborder[0]=%d, expected pinned endpoint 0\n", rnn_nfftborder[0]);
        ok = 0;
    }
    if (rnn_nfftborder[RNNOISE_N_BANDS - 1] != expect_last) {
        printf("FAIL: rnn_nfftborder[%d]=%d, expected pinned endpoint %d\n",
               RNNOISE_N_BANDS - 1, rnn_nfftborder[RNNOISE_N_BANDS - 1], expect_last);
        ok = 0;
    }
    for (int i = 0; i < RNNOISE_N_BANDS - 1; i++) {
        if (!(rnn_nfftborder[i] < rnn_nfftborder[i + 1])) {
            printf("FAIL: rnn_nfftborder not strictly monotone at i=%d (%d >= %d)\n",
                   i, rnn_nfftborder[i], rnn_nfftborder[i + 1]);
            ok = 0;
        }
    }
    return ok;
}

/* Every frequency bin's row across rnn_erb_inv must sum to 1.0 -- each bin
 * falls in exactly one (band, band+1) crossfade segment whose down/up
 * weights are complementary by construction ((1-j/bs)+(j/bs)==1), so this
 * is a real partition-of-unity property of the table, not a loose
 * approximation. `tol` absorbs float summation rounding only. */
static int check_erb_inv_partition_of_unity(double tol) {
    int ok = 1;
    for (int k = 0; k < RNNOISE_N_BINS; k++) {
        double sum = 0.0;
        for (int b = 0; b < RNNOISE_N_BANDS; b++) sum += (double)rnn_erb_inv[k][b];
        if (fabs(sum - 1.0) > tol) {
            printf("FAIL: rnn_erb_inv row (bin) %d sums to %.9f, expected 1.0 +/- %.3g\n", k, sum, tol);
            ok = 0;
        }
    }
    return ok;
}

/* rnn_erb_fwd is rnn_erb_inv with the two end bands (index 0 and N-1)
 * doubled -- an EXACT relation (multiplying a finite float already in
 * [0,1] by 2.0f is exact, no rounding), so this uses `!=`, not a tolerance. */
static int check_erb_fwd_matches_inv_except_ends(void) {
    int ok = 1;
    const int last = RNNOISE_N_BANDS - 1;
    for (int k = 0; k < RNNOISE_N_BINS; k++) {
        for (int b = 0; b < RNNOISE_N_BANDS; b++) {
            float inv = rnn_erb_inv[k][b];
            float fwd = rnn_erb_fwd[k][b];
            int is_end = (b == 0 || b == last);
            float expect = is_end ? 2.0f * inv : inv;
            if (fwd != expect) {
                printf("FAIL: rnn_erb_fwd[%d][%d]=%.9g, expected %.9g (%s rnn_erb_inv)\n",
                       k, b, (double)fwd, (double)expect, is_end ? "2x" : "same as");
                ok = 0;
            }
        }
    }
    return ok;
}

/* [0,1] range, w[0]==0 endpoint, and the symmetry the formula's
 * periodicity guarantees MATHEMATICALLY:
 * cos(2*pi*(WIN_LEN-i)/WIN_LEN) == cos(2*pi - 2*pi*i/WIN_LEN) == cos(2*pi*i/WIN_LEN)
 * by 2*pi-periodicity of cosine, so w[i] and w[WIN_LEN-i] should be the same
 * real number. But `2.0f*(float)M_PI*i/WIN_LEN` and
 * `2.0f*(float)M_PI*(WIN_LEN-i)/WIN_LEN` are DIFFERENT float expressions
 * (different multiply, different rounding) that only happen to evaluate to
 * the same real angle mod 2*pi -- so this is NOT bit-exact in practice
 * (measured on this host: up to 434 ULP / ~8.1e-7 absolute between a pair,
 * from cosf/sqrtf rounding a several-ULP-different argument). This checks
 * the formula-derived relationship (mirror index WIN_LEN-i, not e.g. the
 * wrong-but-plausible WIN_LEN-1-i) with a generous, measured-then-margined
 * ULP bound -- same "garbage-detector, not a bit contract" spirit as the
 * recompute-vs-table check below, just applied to this table's internal
 * self-consistency instead of table-vs-recompute. */
#define RNN_HANN_SYMMETRY_MAX_ULP 4096u
static int check_hann_window_contract(void) {
    int ok = 1;
    if (rnn_hann_win[0] != 0.0f) {
        printf("FAIL: rnn_hann_win[0]=%.9g, expected exactly 0.0f\n", (double)rnn_hann_win[0]);
        ok = 0;
    }
    for (int i = 0; i < RNNOISE_WIN_LEN; i++) {
        if (rnn_hann_win[i] < 0.0f || rnn_hann_win[i] > 1.0f) {
            printf("FAIL: rnn_hann_win[%d]=%.9g out of [0,1]\n", i, (double)rnn_hann_win[i]);
            ok = 0;
        }
    }
    uint32_t worst_sym_ulp = 0;
    for (int i = 1; i < RNNOISE_WIN_LEN; i++) {
        int mirror = RNNOISE_WIN_LEN - i;
        uint32_t d = ulp_distance(rnn_hann_win[i], rnn_hann_win[mirror]);
        if (d > worst_sym_ulp) worst_sym_ulp = d;
        if (d > RNN_HANN_SYMMETRY_MAX_ULP) {
            printf("FAIL: rnn_hann_win[%d]=%.9g vs rnn_hann_win[%d]=%.9g: %u ULP apart "
                   "(formula symmetry bound %u)\n",
                   i, (double)rnn_hann_win[i], mirror, (double)rnn_hann_win[mirror],
                   d, (unsigned)RNN_HANN_SYMMETRY_MAX_ULP);
            ok = 0;
        }
    }
    if (ok) {
        printf("INFO: measured max hann-window mirror-pair ULP distance = %u\n", worst_sym_ulp);
    }
    return ok;
}

/* ── recompute-vs-table max-ULP report (garbage-detector, NOT a bit contract) ── */
static uint32_t max_ulp_over_tables(void) {
    uint32_t worst = 0;
    for (int k = 0; k < RNNOISE_N_BINS; k++) {
        for (int b = 0; b < RNNOISE_N_BANDS; b++) {
            uint32_t d1 = ulp_distance(ref_erb_fwd[k][b], rnn_erb_fwd[k][b]);
            uint32_t d2 = ulp_distance(ref_erb_inv[k][b], rnn_erb_inv[k][b]);
            if (d1 > worst) worst = d1;
            if (d2 > worst) worst = d2;
        }
    }
    for (int i = 0; i < RNNOISE_WIN_LEN; i++) {
        uint32_t d = ulp_distance(ref_hann_win[i], rnn_hann_win[i]);
        if (d > worst) worst = d;
    }
    return worst;
}

/* Generous margin around what's actually MEASURED on this host/toolchain
 * (0, per the PASS below -- matching Layer 1's exact memcmp). This is
 * deliberately NOT tightened to a small bound like 1 ULP: the user
 * explicitly rejected hard-coding a 1-ULP cross-toolchain contract, since a
 * different libm/compiler/target is free to round transcendental functions
 * (log/exp/cos/sqrt, all used by ref_compute_*) differently while still
 * being a perfectly correct implementation. 256 ULP is loose enough to
 * absorb that legitimate variation while still catching a genuinely wrong
 * table (wrong formula, swapped/transposed table, corrupted regen) by many
 * orders of magnitude. */
#define RNN_TABLES_PORTABLE_MAX_ULP 256u

#endif /* RNN_TABLES_PORTABLE */

int main(void) {
    ref_compute_erb_tables();
    ref_compute_window_table();

    int ok = 1;

#ifndef RNN_TABLES_PORTABLE
    /* ── Layer 1 (DEFAULT): canonical-toolchain bit-exact memcmp ── */
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
#else
    /* ── Layer 2 (-DRNN_TABLES_PORTABLE): mathematical-contract checks ── */
    if (!check_all_finite())                        ok = 0;
    if (!check_nfftborder_monotone_pinned())         ok = 0;
    if (!check_erb_inv_partition_of_unity(1e-5))     ok = 0;
    if (!check_erb_fwd_matches_inv_except_ends())    ok = 0;
    if (!check_hann_window_contract())               ok = 0;

    uint32_t worst_ulp = max_ulp_over_tables();
    printf("INFO: measured max ULP (recompute vs table, this host) = %u\n", worst_ulp);
    if (worst_ulp > RNN_TABLES_PORTABLE_MAX_ULP) {
        printf("FAIL: measured max ULP %u exceeds garbage-detector bound %u\n",
               worst_ulp, (unsigned)RNN_TABLES_PORTABLE_MAX_ULP);
        ok = 0;
    }

    if (ok) {
        printf("PASS (portable): rnnoise_tables_gen.h satisfies the mathematical "
               "table contracts (finite; nfftborder monotone+pinned [0,%d]; "
               "erb_inv partition-of-unity; erb_fwd = 2x*erb_inv at end bands "
               "else equal; hann window in [0,1] with formula symmetry), "
               "max ULP vs recompute = %u\n",
               RNNOISE_N_FFT / 2 + 1, worst_ulp);
    }
#endif

    return ok ? 0 : 1;
}
