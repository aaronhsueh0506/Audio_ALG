/* ============================================================
 * RNNoise-ERB 編譯期查表產生器 (host-only 工具, 不進 production build)
 *
 * 這些表格的輸入 (RNNOISE_SR/N_BANDS/N_BINS/N_FFT/WIN_LEN) 全部是編譯期
 * 常數, 所以表格不在執行期計算。本檔算出四張表格 (ERB forward/inverse
 * filterbank、nfftborder、root-Hann window) 後印成一份 C header
 * (rnnoise_tables_gen.h), 內含 `static const` 陣列, 讓 process.c 直接
 * #include。表格在編譯期就固定, 沒有共享可變狀態, 執行期沒有 race。
 *
 * 重新產生 rnnoise_tables_gen.h:
 *   cc -O2 -ffp-contract=off -I../../../audiolib/common/include gen_rnnoise_tables.c ../../../audiolib/common/src/audio_window.c -lm -o gen_rnnoise_tables && ./gen_rnnoise_tables > rnnoise_tables_gen.h
 *
 * rnnoise_tables_gen.h 是「產生出來的檔案」— 不要手動編輯, 要改請改這裡
 * 再重新產生。test_rnnoise_tables.c 另外用一份獨立複製的原始演算法
 * 在執行期重算、逐 byte 比對這個 header 的內容, 當作 drift-guard。
 * ============================================================ */

#include "process.h"
#include "audio_window.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static double freq2erb(double f) {
    return 9.265 * log(1.0 + f / (24.7 * 9.265));
}
static double erb2freq(double e) {
    return 24.7 * 9.265 * (exp(e / 9.265) - 1.0);
}

static int   nfftborder[RNNOISE_N_BANDS];
static float erb_fwd[RNNOISE_N_BINS][RNNOISE_N_BANDS];
static float erb_inv[RNNOISE_N_BINS][RNNOISE_N_BANDS];
static float hann_win[RNNOISE_WIN_LEN];

/* ERB filterbank 與 nfftborder 的計算本體; test_rnnoise_tables.c 的
 * ref_compute_erb_tables() 是它的獨立複本。 */
static void compute_erb_tables(void) {
    const int    N  = RNNOISE_N_BANDS;
    const double sr = (double)RNNOISE_SR;
    const double high_lim = sr / 2.0;
    const double bw = high_lim / ((double)RNNOISE_N_FFT / 2.0);  /* = sr/n_fft */

    /* erb_bandborder: cutoffs = erb2freq(linspace(freq2erb(0), freq2erb(sr/2), N)),
     * ideal border = round((cutoff + bw/2) / bw), 再用嚴格 greedy-forward
     * 「每個 band 至少 MIN_BINS_PER_BAND bin」保證 (逐一往前推進, 從不後退),
     * 端點釘在 DC / Nyquist+1。逐一往前推進的規則對每一對相鄰 border 都
     * 成立 (只檢查 i, i+2 的規則在 sr=16000/n_fft=512/N=22 時仍會產生
     * 寬度僅 1 bin 的 band)。 */
    {
        /* Single source of truth: process.h mirrors config.ini's
         * [signal] min_bins_per_band, and tests/test_feature_contract.py
         * asserts the two agree.  (test_rnnoise_tables.c deliberately keeps
         * its own literal — it is the independent drift guard.) */
#define MIN_BINS_PER_BAND RNNOISE_MIN_BINS_PER_BAND
        double e_lo = freq2erb(0.0), e_hi = freq2erb(high_lim);
        double ideal[RNNOISE_N_BANDS];
        for (int i = 0; i < N; i++) {
            double e = e_lo + (e_hi - e_lo) * i / (N - 1);   /* linspace 含端點 */
            double cutoff = erb2freq(e);
            ideal[i] = floor((cutoff + bw / 2.0) / bw + 0.5); /* np.round (半數進位) */
        }
        double nb0 = 0.0;
        nfftborder[0] = 0;
        for (int i = 1; i < N; i++) {
            double nxt = ideal[i];
            if (nxt < nb0 + MIN_BINS_PER_BAND) nxt = nb0 + MIN_BINS_PER_BAND;
            if (nxt > (double)(RNNOISE_N_FFT / 2 + 1)) nxt = (double)(RNNOISE_N_FFT / 2 + 1);
            nfftborder[i] = (int)nxt;
            nb0 = nxt;
        }
        nfftborder[N - 1] = RNNOISE_N_FFT / 2 + 1;
#undef MIN_BINS_PER_BAND
    }

    /* compute_erb_matrix: block i 介於 border[i]..border[i+1], 欄 i 放下降斜坡、
     * 欄 i+1 放上升斜坡; forward 版兩端單邊欄 ×2。 (double 計算, float 儲存 —
     * 對應 numpy 以 double 算、float32 存的行為) */
    memset(erb_fwd, 0, sizeof(erb_fwd));
    memset(erb_inv, 0, sizeof(erb_inv));
    for (int i = 0; i < N - 1; i++) {
        int lo = nfftborder[i];
        int hi = nfftborder[i + 1];
        int bs = hi - lo;
        for (int j = 0; j < bs; j++) {
            float down = (float)(1.0 - (double)j / (double)bs);
            float up   = (float)((double)j / (double)bs);
            erb_inv[lo + j][i]     = down;
            erb_inv[lo + j][i + 1] = up;
            erb_fwd[lo + j][i]     = down;
            erb_fwd[lo + j][i + 1] = up;
        }
    }
    for (int k = 0; k < RNNOISE_N_BINS; k++) {
        erb_fwd[k][0]     *= 2.0f;
        erb_fwd[k][N - 1] *= 2.0f;
    }
}

/* root-Hann window 的計算本體; test_rnnoise_tables.c 有獨立複本。 */
static void compute_window_table(void) {
    ac_window_build(hann_win, RNNOISE_WIN_LEN);
}

/* %a hexfloat 往返精確; 印出時手動補 'f' 後綴, 讓讀回來的常數是 float
 * (而非先算成 double 常數再窄化, 兩者數值相同但這樣寫意圖明確)。
 * float 依 C 預設引數升格傳給 printf 時會變成 double, 但因為原始值本來
 * 就是 float 能精確表示的值, 升格是精確的, %a 印出的 hex 位數往返後仍
 * 精確落回同一個 float bit pattern。 */
static void print_float_hex(float v) {
    printf("%af", (double)v);
}

static void print_header(void) {
    printf("/* ============================================================\n");
    printf(" * AUTO-GENERATED FILE - DO NOT EDIT.\n");
    printf(" *\n");
    printf(" * Regenerate with:\n");
    printf(" *   cc -O2 -ffp-contract=off -I../../../audiolib/common/include gen_rnnoise_tables.c ../../../audiolib/common/src/audio_window.c -lm -o gen_rnnoise_tables && ./gen_rnnoise_tables > rnnoise_tables_gen.h\n");
    printf(" *\n");
    printf(" * 編譯期常數版 ERB filterbank (forward/inverse) + root-Hann window,\n");
    printf(" * 由 gen_rnnoise_tables.c 算出; test_rnnoise_tables.c 另外用獨立\n");
    printf(" * 複製的演算法在執行期重算並逐 byte 比對這份表格, 當作 drift-guard。\n");
    printf(" * ============================================================ */\n");
    printf("#ifndef RNNOISE_TABLES_GEN_H\n");
    printf("#define RNNOISE_TABLES_GEN_H\n\n");

    printf("#if defined(__GNUC__) || defined(__clang__)\n");
    printf("#define RNN_TABLE_MAYBE_UNUSED __attribute__((unused))\n");
    printf("#else\n");
    printf("#define RNN_TABLE_MAYBE_UNUSED\n");
    printf("#endif\n\n");

    /* rnn_nfftborder 只是算 erb_fwd/erb_inv 的中間結果, process.c 執行期
     * 不需要用到它 (但 test_rnnoise_tables.c 的 drift-guard 會用), 故標
     * __attribute__((unused)) 避免未使用的 -Wunused-const-variable。 */
    printf("static const int rnn_nfftborder[%d] RNN_TABLE_MAYBE_UNUSED = {\n", RNNOISE_N_BANDS);
    printf("    ");
    for (int i = 0; i < RNNOISE_N_BANDS; i++) {
        printf("%d%s", nfftborder[i], (i + 1 < RNNOISE_N_BANDS) ? ", " : "");
    }
    printf("\n};\n\n");

    printf("static const float rnn_erb_fwd[%d][%d] = {\n", RNNOISE_N_BINS, RNNOISE_N_BANDS);
    for (int k = 0; k < RNNOISE_N_BINS; k++) {
        printf("    { ");
        for (int b = 0; b < RNNOISE_N_BANDS; b++) {
            print_float_hex(erb_fwd[k][b]);
            if (b + 1 < RNNOISE_N_BANDS) printf(", ");
        }
        printf(" },\n");
    }
    printf("};\n\n");

    printf("static const float rnn_erb_inv[%d][%d] = {\n", RNNOISE_N_BINS, RNNOISE_N_BANDS);
    for (int k = 0; k < RNNOISE_N_BINS; k++) {
        printf("    { ");
        for (int b = 0; b < RNNOISE_N_BANDS; b++) {
            print_float_hex(erb_inv[k][b]);
            if (b + 1 < RNNOISE_N_BANDS) printf(", ");
        }
        printf(" },\n");
    }
    printf("};\n\n");

    printf("static const float rnn_hann_win[%d] = {\n", RNNOISE_WIN_LEN);
    printf("    ");
    for (int i = 0; i < RNNOISE_WIN_LEN; i++) {
        print_float_hex(hann_win[i]);
        if (i + 1 < RNNOISE_WIN_LEN) {
            printf(",");
            printf(((i + 1) % 4 == 0) ? "\n    " : " ");
        }
    }
    printf("\n};\n\n");

    printf("#endif /* RNNOISE_TABLES_GEN_H */\n");
}

int main(void) {
    compute_erb_tables();
    compute_window_table();
    print_header();
    return 0;
}
