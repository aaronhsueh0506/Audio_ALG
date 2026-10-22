/* ============================================================
 * Align-ULCNet 前後處理 (C 實現) — 對齊 aiaec_streaming.StreamSTFT/StreamISTFT
 *
 * 這個檔案是「網路以外的訊號路徑」。邊界與部署 NPU graph 完全一致:
 *
 *     C   : hop(ULCNET_HOP samples) -> centered sqrt-Hann STFT -> RI 頻譜 (兩路:
 *           linear_error / caller raw far；production descriptor 固定
 *           raw_far，AEC 對 far 的對齊只用來產生 linear_error)
 *     網路: (error_ri, far_ri, 顯式 states) -> enhanced_ri  <-- 只有這段是學來的
 *           (壓縮/attention/GRU/解壓縮全部在 graph 內；states 由 CPU/driver
 *           context 保存，加速器本身 stateless)
 *     C   : enhanced_ri -> centered WOLA (sqrt-Hann, 50% overlap, 包絡正規化,
 *           半窗 trim) -> hop(ULCNET_HOP samples)
 *
 * ⚠ Framing contract = center=True 重現（專案既定決策, 見
 *   docs/align_ulcnet_embedded_streaming_design_zh_TW.md 第 7 節）:
 *   訓練/評測用 torch.stft(center=True, pad_mode='reflect')，本實作逐項重現
 *   它——串流開頭做 reflect 前綴、每幀中心對齊 hop 格點。這與
 *   AINR/DeepFilterNet2/dfn2_process.c 的 center=False 串流不同, 是刻意的:
 *   ULCNet checkpoint 的 feature-time contract 不能靜默改變。
 *
 * 時序（每 push 一個 ULCNET_HOP-sample hop）:
 *   analysis:  push#0 -> 0 幀;  push#1 -> 2 幀 (frame0+frame1);  之後每 push 1 幀
 *   synthesis: frame#0 -> 0 樣本(落在被 trim 的半窗);  之後每 frame ULCNET_HOP 樣本
 *   端到端演算法延遲 = ULCNET_HOP samples，即 centered 分析的半窗
 *   (16 kHz build: 256 samples / 16 ms; 48 kHz build: 512 samples / 10.67 ms)
 *   lookahead 本身——hop#0 無輸出(呼叫端補零)，hop#p(p>=1) 輸出的內容對應
 *   輸入樣本 [(p-1)*ULCNET_HOP, p*ULCNET_HOP)。WOLA 收尾不另加延遲(被半窗 trim 吸收)。
 *   與 Python reference 的 StreamSTFT/StreamISTFT 相同。
 *
 * Parity expectations: the Python reference (AIAEC/aiaec_streaming.py) is
 * bit-exact vs torch. This C side runs its transforms through audiolib's
 * fft_wrapper.h on a caller-owned FftHandle, so real signals pay an
 * ULCNET_N_FFT-point RFFT/IRFFT (not a full complex FFT), and twiddles are
 * precomputed once inside the handle, never per call. Agreement vs Python is
 * float-ULP class (~1e-5 absolute at ULCNET_N_FFT points), not bit-exact.
 * Regression gate:
 * AIAEC/tests/test_ulcnet_process_c.py (drives AcStft / AcIstft in the
 * ulcnet configuration, links libaudio_common.a, compares frame by frame vs
 * Python).
 *
 * Memory contract: the framing state lives in AcStft / AcIstft objects built
 * on caller memory -- zero heap, zero global state, and no big stack frames
 * (the per-call FFT scratch is inside the objects). The FftHandle and the
 * sqrt-Hann window table are caller-owned and SHARED: one ULCNET_N_FFT-point
 * handle serves err-analysis + far-analysis + synthesis, whose transforms are
 * strictly sequential within a hop (the handle is never used concurrently),
 * and one window table serves every object (objects store const pointers
 * only; the caller keeps handle and table alive, and the table unchanged, for
 * the objects' whole lifetime).
 *
 * 不包含（porting 時的其他件）:
 *   - NPU graph 與其顯式 states（K/V ring / logit 史 / GRU h）—
 *     加速器視為 stateless；CPU 以 ulcnet_model_io.c/.h 的 caller-owned
 *     pool 保存並每幀傳入，graph 回傳每個 state 的完整下一個值（in-place）
 *   - PBFDKF linear-error seam — AEC C 的 aec_get_linear_context()
 *   - delay 狀態機（flush ring / fail-open / crossfade）— pipeline 層
 * ============================================================ */

#ifndef ULCNET_PROCESS_H
#define ULCNET_PROCESS_H

#include "audio_stft.h"      /* AcStft, AcIstft, FftHandle, Complex (audiolib) */
#include "ulcnet_model_io.h" /* UlcnetModelIoDescriptor + UlcnetFarInputMode
                              * -- a stddef/stdint-only leaf header; the
                              * UlcnetModel boundary below publishes the
                              * runtime's compiled model-I/O contract by
                              * pointer, so both pipeline wrappers get the
                              * far-input enum from ONE definition          */

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 訊號格點 (編譯期固定,一個 build 服務一個 grid) ----
 * 單一來源與改法都在 ulcnet_model_io.h;此處四個名稱是給既有呼叫端的
 * alias,與 ULCNET_COMPRESSION_EXP 同一個慣例。 */
#define ULCNET_SR       ULCNET_MODEL_IO_SR
#define ULCNET_N_FFT    ULCNET_MODEL_IO_N_FFT
#define ULCNET_HOP      ULCNET_MODEL_IO_HOP    /* = N_FFT/2, 50% overlap (COLA) */
#define ULCNET_BINS     ULCNET_MODEL_IO_BINS   /* N_FFT/2 + 1 */

/* "<SR>x<N_FFT>" as a string literal, for callers that fold the deployment
 * grid into a build-flags hash. Assembled once here so the two pipeline
 * wrappers cannot drift into two spellings of the same token. */
#define ULCNET_STRINGIFY_(x) #x
#define ULCNET_STRINGIFY(x)  ULCNET_STRINGIFY_(x)
#define ULCNET_GRID_TOKEN \
    ULCNET_STRINGIFY(ULCNET_SR) "x" ULCNET_STRINGIFY(ULCNET_N_FFT)

/* The other supported grid: a rate and an FFT size THIS build must reject.
 * Both are legal on that other grid, so a gate that accepts either is
 * reading something other than the compiled grid. */
#define ULCNET_OTHER_SR     (ULCNET_SR == 16000 ? 48000 : 16000)
#define ULCNET_OTHER_N_FFT  (ULCNET_N_FFT == 512 ? 1024 : 512)

/* 模型的 modified power-law 壓縮指數 (compression_exponent)。單一來源在
 * ulcnet_model_io.h 的部署契約常數;此名稱保留給既有呼叫端。 */
#define ULCNET_COMPRESSION_EXP ULCNET_MODEL_IO_COMPRESSION_EXP

/* ---- Framing: audiolib's AcStft / AcIstft ----
 * Analysis is a sqrt-Hann STFT with `center` 1 (centered, reflect-padded:
 * push#0 yields no frame, push#1 yields two, later pushes one; ac_stft_flush
 * adds the trailing frame) or 0 (rolling: exactly one frame per push from the
 * first push, the framing of the AEC/BF seam's own spectra). Synthesis is
 * AC_OLA_ENV_NORM: overlap-add with the squared-window envelope divided out
 * (torch.istft semantics); the first frame lies in the trimmed half window
 * and emits nothing.
 *
 * `fft` is a ULCNET_N_FFT-point handle (fft_get_n_freqs(fft) == ULCNET_BINS)
 * and `window` a ULCNET_N_FFT sqrt-Hann table (ac_window_build). Both may be
 * shared by every object of a pipeline -- the err and far analyses and the
 * synthesis run strictly sequentially within a hop -- and both stay
 * caller-owned, alive and unchanged as long as the objects. Pass them in the
 * config to every ac_stft_get_mem_size / ac_stft_init /
 * ac_istft_get_mem_size / ac_istft_init call; a NULL `fft` or `window` makes
 * the object carve its own. */
void ulcnet_framing_config(AcStftConfig *cfg, FftHandle *fft,
                           const float *window, int center);

/* ---- NPU model callback 邊界（兩個 pipeline 變體共用） ----
 * pipeline 不持有 NPU runtime；推論以每幀一次的 callback 進行。
 * 加速器本身不保存 state；NN 的 far K/V ring、logit 史、GRU h
 * 是 CPU/driver context 的 external state，建議使用 ulcnet_model_io.c/.h
 * 準備 ONNX inputs 並驗證 graph 回傳的 state。reset 在 delay change /
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
 * adapter does this automatically). A model with an `infer` callback MUST
 * publish one -- the pipelines refuse the combination -- because the
 * host-side attention/logit rings are carved from the descriptor's shapes and
 * nothing downstream can detect a disagreement with the graph: the finite
 * guard above catches an UNWRITTEN output, never a WRONG-SHAPED one, so a
 * graph whose D differs reads and writes past those rings silently. NULL is
 * retained only for the memset-zero identity boundary, where there are no
 * shapes to agree about.
 *
 * What the pipelines actually check is the grid, the state-layout version,
 * the fixed raw-far contract, and that D is inside
 * [ULCNET_MODEL_IO_MIN_D, ULCNET_MODEL_IO_MAX_D]. They do NOT and cannot
 * check D against the graph -- no C code reads the exported ONNX metadata or
 * its sidecar. Keeping the descriptor's delay_depth equal to the depth the
 * model was exported with is the integrator's obligation.
 *
 * The descriptor must outlive every pipeline the model is installed into;
 * pipelines copy the UlcnetModel struct but never the descriptor. */
typedef struct UlcnetModel {
    void *user;
    int (*infer)(void *user,
                 const float err_re[ULCNET_BINS], const float err_im[ULCNET_BINS],
                 const float far_re[ULCNET_BINS], const float far_im[ULCNET_BINS],
                 float out_re[ULCNET_BINS], float out_im[ULCNET_BINS]);
    void (*reset)(void *user);
    const UlcnetModelIoDescriptor *io_descriptor;  /* NULL = not published */
} UlcnetModel;

/* One model frame of a hop in flight, for a caller that runs the model itself
 * (the *_nn_begin / *_nn_end pair of the ULCNet pipelines). The error and far
 * spectra are the frame the pipeline would hand to UlcnetModel.infer(); the
 * output staging is pre-filled with NaN, and the model must write all
 * ULCNET_BINS entries of BOTH out_re and out_im (the FULL-WRITE CONTRACT above).
 * Valid until the matching *_nn_end(). */
typedef struct UlcnetNnJob {
    const float *err_re, *err_im;   /* [ULCNET_BINS] linear-error spectrum        */
    const float *far_re, *far_im;   /* [ULCNET_BINS] raw far spectrum, same frame */
    float *out_re, *out_im;         /* [ULCNET_BINS] enhanced spectrum, NaN-prefilled */
} UlcnetNnJob;

/* 1 when every bin of (re, im) is finite. */
int ulcnet_frame_is_finite(const float re[ULCNET_BINS], const float im[ULCNET_BINS]);

#ifdef __cplusplus
}
#endif

#endif /* ULCNET_PROCESS_H */
