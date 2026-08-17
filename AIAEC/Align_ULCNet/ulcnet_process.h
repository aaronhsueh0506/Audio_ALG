/* ============================================================
 * Align-ULCNet 前後處理 (C 實現) — 對齊 aiaec_streaming.StreamSTFT/StreamISTFT
 *
 * 這個檔案是「網路以外的訊號路徑」。邊界與部署 NPU graph 完全一致:
 *
 *     C   : hop(256 samples) -> centered sqrt-Hann STFT -> RI 頻譜 (兩路:
 *           linear_error / AEC aligned-far seam；seam 在 acquisition 前
 *           承載 raw far，production descriptor 固定 aligned_far)
 *     網路: (error_ri, far_ri, 顯式 states) -> enhanced_ri  <-- 只有這段是學來的
 *           (壓縮/attention/GRU/解壓縮全部在 graph 內；states 由 CPU/driver
 *           context 保存，加速器本身 stateless)
 *     C   : enhanced_ri -> centered WOLA (sqrt-Hann, 50% overlap, 包絡正規化,
 *           半窗 trim) -> hop(256 samples)
 *
 * ⚠ Framing contract = center=True 重現（專案既定決策, 見
 *   docs/align_ulcnet_embedded_streaming_design_zh_TW.md 第 7 節）:
 *   訓練/評測用 torch.stft(center=True, pad_mode='reflect')，本實作逐項重現
 *   它——串流開頭做 reflect 前綴、每幀中心對齊 hop 格點。這與
 *   AINR/DeepFilterNet2/dfn2_process.c 的 center=False 串流不同, 是刻意的:
 *   ULCNet checkpoint 的 feature-time contract 不能靜默改變。
 *
 * 時序（每 push 一個 256-sample hop）:
 *   analysis:  push#0 -> 0 幀;  push#1 -> 2 幀 (frame0+frame1);  之後每 push 1 幀
 *   synthesis: frame#0 -> 0 樣本(落在被 trim 的半窗);  之後每 frame 256 樣本
 *   端到端演算法延遲 = 256 samples (16 ms @16 kHz)，即 centered 分析的半窗
 *   lookahead 本身——hop#0 無輸出(呼叫端補零)，hop#p(p>=1) 輸出的內容對應
 *   輸入樣本 [(p-1)*256, p*256)。WOLA 收尾不再另加延遲(被半窗 trim 吸收)。
 *   與 Python reference 的 StreamSTFT/StreamISTFT 相同。
 *
 * Parity expectations: the Python reference (AIAEC/aiaec_streaming.py) is
 * bit-exact vs torch. This C side runs its transforms through audio_common's
 * fft_wrapper.h on a caller-owned FftHandle, so BACKEND=kiss/ne10 genuinely
 * selects the FFT backend, real signals pay a 512-point RFFT/IRFFT (not a
 * full complex FFT), and twiddles are precomputed once inside the handle,
 * never per call. Agreement vs Python is float-ULP class (~1e-5 absolute at
 * 512 points), not bit-exact. Regression gate:
 * AIAEC/tests/test_ulcnet_process_c.py (compiles this file with cc, links
 * libaudio_common.a BACKEND=kiss, compares frame by frame vs Python).
 *
 * Memory contract: all state is caller-owned plain structs -- zero heap,
 * zero global state, and NO big stack frames: the per-call FFT scratch is
 * embedded in the structs (the old stack-local scratch was ~6.2 KB in the
 * analysis push and ~4.2 KB in the synthesis push -- unsafe headroom for an
 * embedded RTOS task stack). The FftHandle and the sqrt-Hann window table
 * are caller-owned and SHARED: one 512-point handle may serve
 * err-analysis + far-analysis + synthesis, whose transforms are strictly
 * sequential within a hop (the handle is never used concurrently), and one
 * window table serves every struct (structs store const pointers only; the
 * caller keeps handle and table alive, and the table unchanged, for the
 * structs' whole lifetime).
 *
 * Standalone compile (the pytest harness uses these same lines):
 *   make -s -C ../audio_common BACKEND=kiss lib
 *   cc -O2 -std=c99 -ffp-contract=off -I AIAEC/Align_ULCNet \
 *      -I ../audio_common/include -c AIAEC/Align_ULCNet/ulcnet_process.c
 *   (link: $(make -s -C ../audio_common BACKEND=kiss print-lib-path) -lm)
 *
 * 不包含（porting 時的其他件）:
 *   - NPU graph 與其顯式 states（K/V ring / logit 史 / GRU h）—
 *     加速器視為 stateless；CPU 以 ulcnet_model_io.c/.h 的 caller-owned
 *     pool 保存並每幀傳入，graph 只回傳 delta-state outputs
 *   - PBFDKF/aligned-far seam — AEC C 的 aec_get_linear_context()
 *   - delay 狀態機（flush ring / fail-open / crossfade）— pipeline 層
 * ============================================================ */

#ifndef ULCNET_PROCESS_H
#define ULCNET_PROCESS_H

#include "fft_wrapper.h"     /* FftHandle, Complex (audio_common)          */
#include "ulcnet_model_io.h" /* UlcnetModelIoDescriptor + UlcnetFarInputMode
                              * -- a stddef/stdint-only leaf header; the
                              * UlcnetModel boundary below publishes the
                              * runtime's compiled model-I/O contract by
                              * pointer, so both pipeline wrappers get the
                              * far-input enum from ONE definition          */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 訊號格點: 16 kHz / 512 / 512 / 256 (ULCNet 部署 grid, 編譯期固定) ---- */
#define ULCNET_SR       16000
#define ULCNET_N_FFT    512
#define ULCNET_HOP      256          /* = N_FFT/2, 50% overlap (COLA) */
#define ULCNET_BINS     257          /* N_FFT/2 + 1 */

/* 模型的 modified power-law 壓縮指數 (compression_exponent)。 */
#define ULCNET_COMPRESSION_EXP 0.3f

/* ---- Shared sqrt-Hann window ----
 * Both analyses and the synthesis use the SAME table; build it once into
 * caller-owned storage and pass the pointer to every *_init below.
 * Ownership: the caller keeps the array alive AND unchanged for the
 * lifetime of every struct initialized with it -- the structs store only
 * the const pointer, never a copy (the old per-struct duplicate window
 * arrays are gone). Formula matches the Python reference's periodic
 * sqrt-Hann bit-for-bit at f32. */
void ulcnet_make_window(float window[ULCNET_N_FFT]);

/* ---- Analysis: centered sqrt-Hann STFT, 每 hop 進 256 樣本 ---- */
typedef struct UlcnetAnalysis {
    const float *window;           /* caller-owned shared sqrt-Hann table */
    FftHandle   *fft;              /* caller-owned 512-point handle; may be
                                    * shared with the other analysis and the
                                    * synthesis (strictly sequential use
                                    * within a hop -- never concurrent)   */
    float history[ULCNET_N_FFT];   /* 最近 N_FFT 個 raw 樣本 (rolling) */
    long  hops_seen;               /* SATURATES at 3 (guarded before the
                                    * increment). Consumers only distinguish
                                    * 1 (first hop, no frame yet), 2 (the
                                    * reflect-prefix double emission) and
                                    * >= 3 steady state (flush: < 2 vs
                                    * >= 2), so the clamp is semantics-
                                    * preserving; it exists to prevent
                                    * signed overflow (UB) on unbounded
                                    * streams. */
    /* Per-call FFT scratch: caller-owned via this struct, NOT the stack
     * (embedded task stacks cannot absorb multi-KB frames). Contents are
     * undefined between calls. */
    float   seg[ULCNET_N_FFT];     /* windowed segment; clobbered by FFT  */
    Complex spec[ULCNET_BINS];     /* RFFT output staging                 */
} UlcnetAnalysis;

/* fft must be a 512-point handle (fft_get_n_freqs(fft) == ULCNET_BINS);
 * window must be a ULCNET_N_FFT sqrt-Hann table from ulcnet_make_window().
 * Both stay caller-owned (see the window/handle sharing contract above).
 * Returns 0, or -1 on NULL args / a wrong-size handle. Re-init on the same
 * struct (same or different handle/window) is the reset. */
int ulcnet_analysis_init(UlcnetAnalysis *st, FftHandle *fft,
                         const float *window);

/* 推入一個 hop。回傳本次產出的幀數 (0 / 2 / 1)，幀依序寫入
 * out_re/out_im[frame][bin]（呼叫端保證至少容納 2 幀）。
 * 第一次 push 產出 0 幀（centered 前綴要等第 257 個樣本）；第二次 push 一次
 * 產出 frame0（含 reflect 前綴）與 frame1；之後每 push 產出 1 幀。 */
int ulcnet_analysis_push(UlcnetAnalysis *st, const float hop_in[ULCNET_HOP],
                         float out_re[2][ULCNET_BINS],
                         float out_im[2][ULCNET_BINS]);

/* 檔尾 flush：套 reflect 後綴，產出剩餘幀（總幀數 = L/HOP + 1，L 為總樣本
 * 數、hop 的整數倍時）。回傳幀數（0..2），寫入同樣的 out 佈局。連續串流
 * 部署用不到；離線 parity 對齊 torch 幀數時才需要。flush 後須 init 重來。 */
int ulcnet_analysis_flush(UlcnetAnalysis *st,
                          float out_re[2][ULCNET_BINS],
                          float out_im[2][ULCNET_BINS]);

/* ---- Synthesis: centered WOLA (sqrt-Hann, 包絡正規化, 半窗 trim) ---- */
typedef struct UlcnetSynthesis {
    const float *window;           /* caller-owned shared sqrt-Hann table */
    FftHandle   *fft;              /* caller-owned 512-point handle; same
                                    * sharing contract as UlcnetAnalysis  */
    float acc[ULCNET_N_FFT];       /* overlap-add 累加器 (局部原點 = 下一段輸出) */
    float env[ULCNET_N_FFT];       /* 窗平方包絡累加器 (torch.istft 語意) */
    long  frames_seen;             /* SATURATES at 2 (guarded before the
                                    * increment). Consumers only distinguish
                                    * 0 (nothing accumulated), 1 (frame #0,
                                    * inside the trimmed half window) and
                                    * >= 2 (steady one-hop emission), so the
                                    * clamp is semantics-preserving; it
                                    * exists to prevent signed overflow (UB)
                                    * on unbounded streams. */
    /* Per-call FFT scratch -- same off-stack rationale as UlcnetAnalysis. */
    Complex spec[ULCNET_BINS];     /* IRFFT input staging; clobbered by FFT */
    float   time[ULCNET_N_FFT];    /* IRFFT time-domain output              */
} UlcnetSynthesis;

/* Same argument/ownership/return contract as ulcnet_analysis_init. */
int ulcnet_synthesis_init(UlcnetSynthesis *st, FftHandle *fft,
                          const float *window);

/* 推入一幀 enhanced 頻譜。回傳寫入 out 的樣本數：frame#0 回 0（半窗 trim），
 * 之後每幀回 ULCNET_HOP。 */
int ulcnet_synthesis_push(UlcnetSynthesis *st,
                          const float re[ULCNET_BINS],
                          const float im[ULCNET_BINS],
                          float out[ULCNET_HOP]);

/* 檔尾 flush：輸出尚未 finalize 的尾巴（最多 N_FFT-HOP 樣本；呼叫端用
 * total_samples 截到原始長度）。連續串流部署用不到。 */
int ulcnet_synthesis_flush(UlcnetSynthesis *st, float out[ULCNET_N_FFT]);

/* ---- NPU model callback 邊界（兩個 pipeline 變體共用） ----
 * pipeline 不持有 NPU runtime；推論以每幀一次的 callback 進行。
 * 加速器本身不保存 state；NN 的 far K/V ring、logit 史、GRU h
 * 是 CPU/driver context 的 external state，建議使用 ulcnet_model_io.c/.h
 * 準備 ONNX inputs 並提交 delta-state outputs。reset 在 delay change /
 * pipeline reset 時被呼叫；目前 production contract 清空 K/V、logit 與
 * 兩組 GRU hidden，日後若只清 attention state 必須另做 A/B 並版本化。
 * reset callback 可為 NULL。infer 回傳 0 表成功；非 0 時 pipeline 以
 * fail-open 輸出線性誤差。
 *
 * FULL-WRITE CONTRACT: a return of 0 REQUIRES that infer wrote ALL
 * ULCNET_BINS entries of BOTH out_re and out_im. The pipelines enforce
 * this by pre-filling the output staging buffers with NaN before EVERY
 * infer call, so a partial write leaves non-finite bins behind and is
 * caught by the pipelines' finite-output guard, which falls open to the
 * identity (error passthrough) path -- a partially-written frame is never
 * applied and can never leak stale values from a previous frame.
 *
 * MODEL-CONTRACT PUBLICATION (io_descriptor): a runtime that owns a
 * ulcnet_model_io state publishes its descriptor here (the accelerator
 * adapter does this automatically). Production pipelines validate its grid,
 * D, state layout and fixed aligned-far contract before installation. NULL is
 * retained for the memset-zero identity/test boundary. The descriptor must
 * outlive every pipeline the model is installed into; pipelines copy the
 * UlcnetModel struct but never the descriptor. */
typedef struct UlcnetModel {
    void *user;
    int (*infer)(void *user,
                 const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
                 const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS],
                 float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]);
    void (*reset)(void *user);
    const UlcnetModelIoDescriptor *io_descriptor;  /* NULL = not published */
} UlcnetModel;

/* ---- 可選: 壓縮/解壓縮 helper ----
 * 模型 graph 內建 signed |x|^0.3 壓縮與 |x|^(1/0.3) 解壓縮（model.py
 * _signed_power）。若目標 runtime 不支援 pow 類算子而把這兩步移出 graph，
 * 用這兩個 helper——邊界移動要連 export graph 一起改，兩邊必須一致。 */
void ulcnet_compress_frame(const float re[ULCNET_BINS],
                           const float im[ULCNET_BINS],
                           float zr[ULCNET_BINS], float zi[ULCNET_BINS]);
void ulcnet_expand_frame(const float re[ULCNET_BINS],
                         const float im[ULCNET_BINS],
                         float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_PROCESS_H */
