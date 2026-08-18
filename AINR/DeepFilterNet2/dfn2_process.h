/* ============================================================
 * DeepFilterNet2 前後處理 (C 實現) — 對齊 train.py / model.py / inference.py
 *
 * 這個檔案是「網路以外的全部」。邊界與 model.py 的 heads()/compose() 完全一致:
 *
 *     C   : frame -> STFT -> ERB 特徵 + complex 特徵
 *     網路: (feat_erb, feat_spec) -> (erb_mask, coefs, alpha) <-- 只有這一段是學來的
 *     C   : erb_mask/coefs/alpha -> 全頻 mask · 低頻 deep filter · alpha blend
 *           · post-filter · atten_lim · ISTFT + OLA -> frame
 *
 * ⚠ 網路只出 mask、coefficient 與 alpha，合成全部在這裡。compose() 是 parameter-free
 *   的，正是為了讓這個檔案可以逐項對上它；任何學來的東西都該在 heads() 裡。
 *   上游把線畫在同一個位置: 它的 Python forward 會合成 (那是訓練圖)，而部署的
 *   runtime 只跑網路取這兩個張量，合成自己做。
 *
 * 與 Python 參考的對應 (inference.py 為準):
 *   - STFT: sqrt-Hann window, normalized=True (x N_FFT^-0.5)
 *   - 分析尺度: 額外乘 DFN2_ANALYSIS_SCALE 才進正規化器 (libDF wnorm, 見下)
 *   - ERB: caller-loaded exported matrices (erb_fwd.bin/erb_inv.bin)
 *          (mode=0 forward 邊緣欄加倍 / mode=1 inverse partition of unity)
 *   - ERB 特徵: band 能量取 dB 後，逐 band causal EMA mean norm，再 /40
 *   - Complex 特徵: 0..df_bins 的 real/imag，逐 bin magnitude EMA unit norm
 *   - band gain -> 全頻 bin gain: erb_mask @ ERB_inv^T (mode=1)
 *   - cascade: 全頻先套 mask，再對最低 DF_BINS 做 DF_ORDER-tap 複數 FIR
 *   - alpha blend: 低頻 = alpha*DF(masked) + (1-alpha)*masked；高頻保留 masked
 *   - ISTFT: normalized=True (x N_FFT^+0.5) + sqrt-Hann + 50% OLA (COLA)
 *
 * ============================================================
 * ⚠ 已知的 parity 限制 — 這兩項無法在 C 端修掉，不要當成 bug 追
 * ============================================================
 *
 * 1) 框對齊。train.py 用 torch.stft 的預設 center=True，也就是首框被置中且
 *    邊界用 reflect padding。這個 C 是串流實現 (等價 center=False，零填暖機)。
 *    後果只在 clip 起點: Python 的 frame 0 含有訊號的時間反轉拷貝，這裡是靜音。
 *    穩態下 center=True 只是固定的半窗時序偏移，串流可精確重現。
 *    ⚠ 實測: torch.istft 在 win=1024/hop=512 下直接拒絕 center=False
 *    ("window overlap add min: 1")，因為首尾框 OLA 覆蓋不足。所以要對齊得照
 *    libDF 做「零填暖機 + 丟掉暖機框」的 framing，不是翻一個 flag。
 *    ⚠ 這件事對 causal EMA 尤其要緊: 它的狀態從 frame 0 起累積，而 3 秒
 *    segment 全程都在暖機期 (3*tau = 273 frames vs 281 frames/segment)，所以
 *    frame 0 的差異會一路傳播。
 *
 * 2) spec_norm_eps。trainer 已經把它移除以對齊 libDF 的裸 sqrt，但這裡
 *    保留 (DFN2_SPEC_NORM_EPS)。這不是不一致，是刻意的:
 *    fp32 + 次正規數下 mu 永不為 0，所以 trainer 不需要它；但在 FTZ /
 *    denormals-are-zero 的目標上 mu 可以真的到 0，x/sqrt(0) = NaN。
 *    ⚠ 這正是「trainer 拿掉、C 保留」的理由，precedent 見 RNNoise-ERB 的
 *    process.c: sqrtf(state + RNNOISE_SPEC_NORM_EPS)。
 *    代價是與 Python 差 ~1e-9 相對，遠小於 float32 ULP 級的其他差異。
 *
 * → 與 torch 為 float32 ULP 級近似, 非 bit-exact。
 * ============================================================ */

#ifndef DFN2_PROCESS_H
#include "fft_wrapper.h"

#define DFN2_PROCESS_H

#include <stddef.h>

/* ---- 訊號格點: = config.ini [signal] ---- */
#define DFN2_SR             48000
#define DFN2_N_FFT          1024
#define DFN2_N_BINS         513   /* N_FFT/2 + 1 */
#define DFN2_WIN_LEN        1024  /* 分析窗長度 (= N_FFT) */
#define DFN2_HOP_LEN        512   /* 幀移 (= WIN_LEN/2, COLA) */
#define DFN2_OVL_LEN        (DFN2_WIN_LEN - DFN2_HOP_LEN)

/* ⚠ 不是上游的 960/480。960 不是 2 的次方，嵌入式 FFT 用不了。代價是 DF 的
 * 截止落在 4500 Hz 而非上游的 4800 Hz — 見 config.ini [model] df_bins。 */

/* ---- 模型幾何: = config.ini [model] ---- */
#define DFN2_N_ERB          32
#define DFN2_DF_BINS        96    /* deep filter 擁有的 bin 數 (0..95) */
#define DFN2_DF_ORDER       5     /* FIR tap 數 */
#define DFN2_DF_LOOKAHEAD   1     /* 未來 tap 數 */
#define DFN2_MASK_LOOKAHEAD 1

/* deep filter 的時間窗是 [t-(ORDER-LOOKAHEAD-1) .. t+LOOKAHEAD]。
 * 在出貨的 5/1 下是 [t-3 .. t+1]:
 *   - 需要 3 個歷史框 + 當前框 + 1 個未來框 = ring buffer 深度 5
 *   - 單看 DF FIR 需要 1 框輸出延 (要等 masked t+1 才能發 t)
 * ⚠ 上游出貨 2/2 (對稱 [t-2..t+2])。本 port 用 1/1 換延遲: hop 512 @ 48k
 *   一框 10.67 ms。但 DFN2 是 cascade: masked(t+1) 自己還要等
 *   mask head(t+1)，而 head 的 conv lookahead 又是 1。所以真正的
 *   streaming head-to-audio 總延遲是 MASK_LOOKAHEAD + DF_LOOKAHEAD = 2 框，
 *   不是 max(1,1)=1。見 dfn2_compose_stream()。 */
#define DFN2_DF_HISTORY     (DFN2_DF_ORDER - DFN2_DF_LOOKAHEAD - 1)  /* 3 */
#define DFN2_DF_RING        DFN2_DF_ORDER                            /* 5 */

/* ---- 特徵正規化: = config.ini [feature] ----
 * ⚠ FEATURE_VERSION 必須與 train.py 逐字相同，否則 checkpoint 與這個前處理
 * 不是同一個合約。 */
#define DFN2_FEATURE_VERSION  "dfn2_libdf_wnorm_upstream_init_no_eps_v5"

/* alpha 是導出值不是設定值: exp(-(hop/sr)/tau) 再取 3 位小數 (train.py 的
 * make_norm_alpha 重現上游 df/utils.py 的取整迴圈)。tau = 1.0 s，
 * hop 512 @ 48k -> 0.989。⚠ 這個數字在上游任何 config 都不存在，因為上游的
 * hop 480 會得到不同值。改 sr/hop 就要重新導出，不要照抄。
 * ⚠ 副作用: 3 位小數的取整讓有效 tau 是 0.9644 s 而非 1.0 s。 */
#define DFN2_ERB_NORM_ALPHA         0.989f
#define DFN2_ERB_NORM_INIT_LO_DB  (-60.0f)   /* = libDF MEAN_NORM_INIT */
#define DFN2_ERB_NORM_INIT_HI_DB  (-90.0f)
#define DFN2_ERB_NORM_SCALE_DB      40.0f
#define DFN2_ERB_LOG_FLOOR          1e-10f   /* = libDF lib.rs:209 的字面值 */

/* ⚠ scale_db 不是任意值: 它是 |log floor - init_lo|。
 *   10*log10(1e-10) = -100 dB，|-100 - (-60)| = 40，所以數值靜音的 band 會被
 *   硬界在精確的 -1.00，也就是上游自己的特徵範圍。上游這三個常數互相一致，
 *   動一個就破壞關係 (init_lo 移到 -15 就需要除數 85)。 */

#define DFN2_SPEC_NORM_ALPHA        0.989f
#define DFN2_SPEC_NORM_INIT_LO      0.001f   /* = libDF UNIT_NORM_INIT */
#define DFN2_SPEC_NORM_INIT_HI      0.0001f
#define DFN2_SPEC_NORM_EPS          1e-12f   /* ⚠ 見檔頭 parity 限制 (2) */

/* libDF 的分析尺度。上游對頻譜乘 wnorm = 2*hop/win^2 (= 1/1024)，而
 * torch.stft(normalized=True) 只乘 1/sqrt(n_fft) (= 1/32)，所以殘差因子是
 * wnorm * sqrt(n_fft) = 1/32。
 * ⚠ 這在 DF 路徑上不是裝飾: band_unit_norm 是 x/sqrt(EMA|x|)，輸入乘 c 則
 *   輸出乘 sqrt(c)，是永久因子不是暫態。漏掉它 feat_spec 會永遠以
 *   sqrt(32) = 5.66x 上游的量級進 DF 分支，而 feat_erb 是上游量級，兩個
 *   encoder 分支相對彼此錯配。
 * ⚠ 只作用在餵進正規化器的那一份。要送去 masking/ISTFT 的頻譜保持原尺度，
 *   否則 round trip 不再是 unity。 */
#define DFN2_ANALYSIS_SCALE   0.03125f       /* = 2*512/1024^2 * sqrt(1024) */

/* ---- 推論後處理: = config.ini [model] ---- */
#define DFN2_MASK_PF        0        /* Valin post-filter, 出貨 false */
#define DFN2_PF_BETA        0.02f    /* = 上游 PF_BETA */

/* ============================================================
 * 狀態 (呼叫端分配，跨 frame 保持)
 * ============================================================ */
typedef struct {
    /* Analysis overlap and instance-owned immutable-after-init tables. */
    float analysis_buf[DFN2_WIN_LEN];
    float window[DFN2_WIN_LEN];
    /* Caller-loaded exported ERB matrices (never derived here):
     * erb_fwd bin-major [DFN2_N_BINS][DFN2_N_ERB], erb_inv band-major
     * [DFN2_N_ERB][DFN2_N_BINS]; raw float32 from erb_fwd.bin/erb_inv.bin.
     * The loader owns the memory and may swap it between hops. */
    const float *erb_fwd;
    const float *erb_inv;

    /* OLA 緩衝 (長度 = WIN_LEN, 只用前 OVL_LEN) */
    float synthesis_buf[DFN2_WIN_LEN];

    /* 逐 band 的 causal log-ERB mean EMA */
    float erb_norm_state[DFN2_N_ERB];

    /* 逐 bin 的 complex-magnitude EMA (只有 DF 那段 bin) */
    float spec_norm_state[DFN2_DF_BINS];

    /* deep filter 的複數 ring buffer: 已套 ERB mask 的最低 DF_BINS 個 bin。
     * ⚠ DFN2 是串聯 cascade，不能把原始未遮罩頻譜直接推進此 buffer。 */
    float df_ring_re[DFN2_DF_RING][DFN2_DF_BINS];
    float df_ring_im[DFN2_DF_RING][DFN2_DF_BINS];
    float coef_ring[DFN2_DF_RING][DFN2_DF_BINS][DFN2_DF_ORDER][2];
    float alpha_ring[DFN2_DF_RING];
    float noisy_ring_re[DFN2_DF_RING][DFN2_N_BINS];
    float noisy_ring_im[DFN2_DF_RING][DFN2_N_BINS];
    int   df_ring_idx;     /* 下一個寫入位置 */
    int   df_ring_count;   /* 已累積框數，用來判斷暖機是否結束 */

    /* Streaming accelerator handoff.  dfn2_compose_stream() numbers each
     * pushed spectrum monotonically; returned heads are expected to describe
     * current_frame-MASK_LOOKAHEAD.  Do not mix the aligned and streaming
     * compose APIs without dfn2_state_init(). */
    long long stream_frame_index;

    /* 高頻段 (bin >= DF_BINS) 也要延遲同樣的框數，否則 band split 會把
     * 不同時刻的兩半拼在一起。⚠ 這是最容易漏的一步。 */
    float hi_delay_re[DFN2_DF_RING][DFN2_N_BINS - DFN2_DF_BINS];
    float hi_delay_im[DFN2_DF_RING][DFN2_N_BINS - DFN2_DF_BINS];

    /* --- scratch: 非跨 frame 狀態，每次使用前完整覆寫 --- */
    float scratch_time[DFN2_N_FFT];
    Complex scratch_freq[DFN2_N_BINS];
    /* Caller-owned audio_common FFT handle (fft_create/fft_init for
     * DFN2_N_FFT); borrowed like the .bin matrix pointers. */
    FftHandle* fft;
    float scratch_power[DFN2_N_BINS];
    float scratch_erb_db[DFN2_N_ERB];
    float scratch_bin_gain[DFN2_N_BINS];
} DFN2State;

/* 初始化 (歸零 + 兩個正規化器的 linspace 初值) */
void dfn2_state_init(DFN2State *st, FftHandle *fft);

/* Point the state at the caller-loaded ERB matrices (see the struct field
 * comment for layout). Must be called before feature extraction or mask
 * expansion; the library never reads files itself. */
void dfn2_set_erb_matrices(DFN2State *st, const float *erb_fwd,
                             const float *erb_inv);

/* frame (HOP_LEN 個新樣本，內部自己維護分析 overlap) -> 複數頻譜。
 * out_re/out_im 長度 N_BINS。⚠ 這是「原尺度」的頻譜，masking 與 ISTFT 用它；
 * 分析尺度只在特徵計算裡套。 */
void dfn2_analysis(DFN2State *st, const float *frame,
                   float *out_re, float *out_im);

/* 頻譜 -> 網路的兩組輸入。
 *   feat_erb : 長度 N_ERB
 *   feat_spec: 長度 2*DF_BINS，interleave 為 [re..., im...] 兩段 (非交錯)
 * ⚠ 內部會套 DFN2_ANALYSIS_SCALE，呼叫端不要自己先乘。 */
void dfn2_compute_features(DFN2State *st,
                           const float *spec_re, const float *spec_im,
                           float *feat_erb, float *feat_spec);

/* DeepFilterNet atten_lim 必須對最終複數頻譜混合：DF 輸出不是 noisy
 * spectrum 乘上實數 mask，因此不能像 RNNoise 那樣在 band gain 域做。 */
void dfn2_apply_atten_lim(const float *noisy_re, const float *noisy_im,
                          float *enh_re, float *enh_im,
                          float atten_lim_db);

/* 網路輸出 -> 增強頻譜。這是 model.py compose() 的 C 對應。
 *   erb_mask: 長度 N_ERB，sigmoid 後在 [0,1]
 *   coefs   : 長度 DF_BINS * DF_ORDER * 2，layout 與 model.py 的
 *             (df_bins, df_order, 2) 相同 (bin 最外、tap 次之、re/im 最內)
 *   alpha   : 單一 sigmoid blend weight；0 走 masked residual，1 走 DF
 *   spec_re/spec_im: 當前框的原尺度頻譜，會被推進 ring buffer
 *   out_re/out_im  : 延遲 DF_LOOKAHEAD 框後的增強頻譜
 * 回傳 1 表示 out 有效，0 表示還在暖機 (前 DF_LOOKAHEAD 框沒有輸出)。
 * ⚠ 呼叫端必須處理回傳 0: 那幾框不要送去 ISTFT。 */
int dfn2_compose(DFN2State *st,
                 const float *spec_re, const float *spec_im,
                 const float *erb_mask, const float *coefs, float alpha,
                 float *out_re, float *out_im);

/* 真正的串流/硬體 handoff。每次呼叫推進一框 CURRENT spectrum；
 * 當 heads_valid=1 時，erb_mask/coefs/alpha 必須是硬體在這一拍
 * 回傳的 current_frame-MASK_LOOKAHEAD 那框 head。heads_valid=0 只用於
 * 左邊暖機，此時三個 head pointer 可以是 NULL。
 *
 * DFN2 cascade 要先有每個 source frame 自己的 mask，所以有效輸出
 * 對應 current_frame-MASK_LOOKAHEAD-DF_LOOKAHEAD。出貨 1/1 因此是
 * 2 hops = 21.33 ms @48 kHz，再加 STFT framing 本身的算法時間。
 * output_frame_index 可為 NULL；非 NULL 時回報 out 所屬的 frame id。
 *
 * 右邊 flush 必須由中間硬體用與訓練相同的 zero padding 產生剩餘
 * heads，再以零 spectrum 繼續呼叫；C 不會臆測加速器的 flush。
 * atten_lim_db=0 停用 attenuation limit；非零時會對正確的延遲
 * noisy target 做 complex mix，呼叫端不需要另存一份 spectrum ring。
 * 回傳 1 = 有效 out，0 = 還在暖機，-1 = pointer/時序合約錯誤。 */
int dfn2_compose_stream(DFN2State *st,
                        const float *current_spec_re,
                        const float *current_spec_im,
                        int heads_valid,
                        const float *erb_mask,
                        const float *coefs,
                        float alpha,
                        float atten_lim_db,
                        float *out_re,
                        float *out_im,
                        long long *output_frame_index);

/* Valin post-filter (上游 deepfilternet3.py:388-394 的形式)。
 * ⚠ 作用在最終複數頻譜，mask 是從 |spec_e|/|spec| 推導出來的實際增益，不是
 *   網路輸出的那個 mask。分子沒有 mask 因子。上游不用 training 閘門。
 *   DFN2 舊版是在 ERB mask 上做、只在推論、beta 寫死 — 兩者不可互換。 */
void dfn2_post_filter(const float *spec_re, const float *spec_im,
                      float *enh_re, float *enh_im, float beta);

/* 增強頻譜 -> HOP_LEN 個輸出樣本 (sqrt-Hann 合成 + OLA)。
 * ⚠ 合成端不做 window-envelope 除法: sqrt-Hann 分析 x sqrt-Hann 合成 = Hann，
 *   在 50% overlap 下滿足 COLA。 */
void dfn2_synthesis(DFN2State *st,
                    const float *spec_re, const float *spec_im,
                    float *out_frame);

const char *dfn2_simd_backend(void);

#endif /* DFN2_PROCESS_H */
