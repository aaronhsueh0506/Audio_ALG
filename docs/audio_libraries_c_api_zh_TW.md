# AEC、NR、audiolib C API 整合速查

本文件是給 application／board integration 使用者的**短版入口**，涵蓋 AEC、NR、audiolib
三個 library，以及三個 AI 模型的 pre/post 類別（[§6](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md#6-ai-模型-prepost-類別)），回答三件事：

1. 應該 include 哪些 public header；
2. init、process、reset、destroy 的正確順序；
3. 哪些工作由 library 負責，哪些必須由呼叫端負責。

完整欄位、錯誤語意與演算法說明仍以各 library 的 canonical manual 為準：

- [AEC C 使用手冊](../lib/aec/docs/c_user_manual_zh_TW.md)
- [NR C 使用手冊](../lib/nr/docs/c_user_manual_zh_TW.md)
- [audiolib C 使用手冊](../../audiolib/common/docs/c_user_manual_zh_TW.md)
- [Audio_ALG mono／4ch pipeline 使用手冊](c_user_manual_zh_TW.md)
- 模型側的圖邊界、I/O 表與延遲：`docs/html/` 的各模型頁（`ainr_dfn2.html` 等）

若本文件與 header 不一致，以目前 pin 住的 public header 與測試為準。
## 目錄 <!-- split-toc -->
- [1. 先決定使用層級](#1-先決定使用層級) <!-- split-toc -->
- [2. 共通信號契約](#2-共通信號契約) <!-- split-toc -->
  - [2.1 產品 grid](#21-產品-grid) <!-- split-toc -->
  - [2.2 PCM、window 與 block](#22-pcmwindow-與-block) <!-- split-toc -->
- [3. AEC library](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#3-aec-library) <!-- split-toc -->
  - [3.1 Public header](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#31-public-header) <!-- split-toc -->
  - [3.2 Config 與建立](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#32-config-與建立) <!-- split-toc -->
  - [3.3 每-hop 處理入口](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#33-每-hop-處理入口) <!-- split-toc -->
  - [3.4 Reset、context 與診斷](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#34-resetcontext-與診斷) <!-- split-toc -->
  - [3.5 執行期改抑制強度](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#35-執行期改抑制強度) <!-- split-toc -->
  - [3.6 參數分級：外部與內部](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#36-參數分級外部與內部) <!-- split-toc -->
- [4. NR library](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#4-nr-library) <!-- split-toc -->
  - [4.1 Public header 與責任邊界](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#41-public-header-與責任邊界) <!-- split-toc -->
  - [4.2 Config 與建立](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#42-config-與建立) <!-- split-toc -->
  - [4.3 每-frame 頻域入口](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#43-每-frame-頻域入口) <!-- split-toc -->
  - [4.4 Reset 與查詢](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#44-reset-與查詢) <!-- split-toc -->
  - [4.5 執行期改降噪強度](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#45-執行期改降噪強度) <!-- split-toc -->
  - [4.6 參數分級：外部與內部](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#46-參數分級外部與內部) <!-- split-toc -->
- [5. audiolib library](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#5-audiolib-library) <!-- split-toc -->
  - [5.1 Header 與主要 API](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#51-header-與主要-api) <!-- split-toc -->
  - [5.2 FFT](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#52-fft) <!-- split-toc -->
  - [5.3 Resampler](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#53-resampler) <!-- split-toc -->
  - [5.4 HPF](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#54-hpf) <!-- split-toc -->
- [6. AI 模型 pre/post 類別](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md#6-ai-模型-prepost-類別) <!-- split-toc -->
  - [6.1 共通生命週期](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md#61-共通生命週期) <!-- split-toc -->
  - [6.2 每-hop 序列與回傳值合約](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md#62-每-hop-序列與回傳值合約) <!-- split-toc -->
  - [6.3 skip 政策：三個模型不一樣](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md#63-skip-政策三個模型不一樣) <!-- split-toc -->
  - [6.4 Align-ULCNet：`ulcnet_prepost.h`](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md#64-align-ulcnetulcnet_preposth) <!-- split-toc -->
  - [6.5 DeepVQE-S：`deepvqe_prepost.h`](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md#65-deepvqe-sdeepvqe_preposth) <!-- split-toc -->
  - [6.6 DeepFilterNet2：`dfn2_prepost.h`](audio_libraries_c_api_zh_TW_03_ai_prepost_pipeline.md#66-deepfilternet2dfn2_preposth) <!-- split-toc -->
  - [6.7 Pool 大小](audio_libraries_c_api_zh_TW_03_ai_prepost_pipeline.md#67-pool-大小) <!-- split-toc -->
  - [6.8 Framing：一 hop 一框的 analysis](audio_libraries_c_api_zh_TW_03_ai_prepost_pipeline.md#68-framing一-hop-一框的-analysis) <!-- split-toc -->
  - [6.9 Pipeline 層：頻域 stage 串接與 NN 的 begin / nn_begin / nn_end / end](audio_libraries_c_api_zh_TW_03_ai_prepost_pipeline.md#69-pipeline-層頻域-stage-串接與-nn-的-begin--nn_begin--nn_end--end) <!-- split-toc -->
  - [6.10 各函式庫各自初始化：記憶體加總與單元的公開面](audio_libraries_c_api_zh_TW_03_ai_prepost_pipeline.md#610-各函式庫各自初始化記憶體加總與單元的公開面) <!-- split-toc -->
- [7. Static-pool 共通規則](audio_libraries_c_api_zh_TW_04_pool_build_checklist.md#7-static-pool-共通規則) <!-- split-toc -->
- [8. 建置、SIMD 與連結](audio_libraries_c_api_zh_TW_04_pool_build_checklist.md#8-建置simd-與連結) <!-- split-toc -->
- [9. 最小整合檢查清單](audio_libraries_c_api_zh_TW_04_pool_build_checklist.md#9-最小整合檢查清單) <!-- split-toc -->

---

## 1. 先決定使用層級

| 需求 | 建議入口 |
|---|---|
| mono `AEC + NR + RES` | 沒有 mono 單元：`aec.h` 的 `Aec`（`aec_process_context()`）加 `common/se_stage.h` 的 `se_frame_from_aec()`，再串 `anr.h`、`audio_stft.h`（RES gain 就是 `Aec` 自己的 `ctx.res_gain`，經 `se_frame_from_aec()` 變成 `frame.g_res_ctx`）；範例 `pipelines/mono_aec_nr_res/main.c`、`static_main.c` |
| 4ch linear AEC + BF（唯一包起來的單元；不含 RES／NR／合成） | `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.h` |
| 4ch linear AEC 與外部 beamformer seam | `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.h` |
| 只做單路 AEC，或自行組合 NN／其他 post-filter | `aec.h` |
| 已有 STFT/WOLA，只需要頻域 NR | `anr.h` |
| 只需要 FFT、resample、HPF | `audiolib/common/include/*.h` |
| Align-ULCNet 加速器 handoff（hop 或頻譜） | `AIAEC/Align_ULCNet/ulcnet_prepost.h` |
| DeepVQE-S 加速器 handoff | `AIAEC/DeepVQE_S/deepvqe_prepost.h` |
| DeepFilterNet2 加速器 handoff | `AINR/DeepFilterNet2/dfn2_prepost.h` |
| DeepFilterNet3 加速器 handoff | `AINR/DeepFilterNet3/dfn3_prepost.h` |

只有「多聲道線性 AEC + BF」被包成一個單元（`AudioPipeline4Ch`；`fuse()` 之後 `frame->g_res_ctx` 為 NULL，單元不持有 `AecRes`）。
mono 是例外：它的 RES gain 就在 `Aec` 的呼叫裡（`aec_process_context()` 算出，`se_frame_from_aec()` 交成 `frame.g_res_ctx`）。
4ch 的 post-beam RES、NR、comfort noise、合成與 NN 後濾波都是獨立的 library，各自查大小、各自 init，由整合者串接。
各範例 `main` 自己初始化用到的函式庫並明寫每 hop 的串接（4ch 為 `se_res_stage_gain()` 的 RES gain → `se_anr_stage_gain()` → `sk_min_f32()` →
`sk_capply_gain_f32()` → `aec_cng_fill()` → `ac_istft_push()`）；`pipelines/samples/` 的共用範例碼
（`se_anr_stage.h`、`dfn_hop.h`、`ulcnet_nn_stage.{h,c}`）示範 NR 級與 NN stage 怎麼串；沒有 ABI，
可整份複製改寫，不要在 application 重新發明 AEC/NR gain fusion、window、OLA 或 CNG 的順序。

各條流程的**呼叫時序圖**（含回傳值、被拒時的處理與 hop 泳道）與**公開／內部函式、外部／內部參數**分級見各整合手冊的第 0 節：
[mono](integration_mono_zh_TW.md)、[4ch 核心](integration_4ch_core_zh_TW.md)、[4ch 單元](integration_4ch_spatial_zh_TW.md)；
板端的 query → allocate → init → teardown 順序見 [pipeline_board_integration.md](pipeline_board_integration.md)。本文 [§3.6](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#36-參數分級外部與內部)、[§4.6](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md#46-參數分級外部與內部) 是 AEC 與 NR 兩個函式庫自己的參數分級。

---

## 2. 共通信號契約

### 2.1 產品 grid

| sample rate | frame = FFT | hop | bins | 狀態 |
|---:|---:|---:|---:|---|
| 16 kHz | 256 | 128 | 129 | 預設、低延遲 |
| 16 kHz | 512 | 256 | 257 | 可選 |
| 48 kHz | 1024 | 512 | 513 | 48 kHz 產品 grid |

AEC 與 NR 另支援 8 kHz grid（非產品主 grid），4ch product pipeline 不接受它。不要只傳
`sample_rate` 後讓底層猜 FFT；16 kHz 有兩個合法值，建立實例前必須 resolve 成明確的
`fft_size`。

### 2.2 PCM、window 與 block

- PCM 使用 mono/interleaved `float32`，標稱範圍 `[-1, 1]`。
- AEC 每次固定吃一個 hop；函式沒有長度參數，傳錯長度不會替你報錯。
- NR C library 只吃 `Complex[n_freqs]`，不做 framing、window、FFT、IFFT 或 OLA。
- 時域 NR wrapper 使用 periodic sqrt-Hann analysis/synthesis window、50% overlap、
  `frame_size == fft_size`，不補零。
- AEC 不做最終 peak limiter。若產品需要 limiter／AGC／DRC，放在整條
  AEC/BF/NR/RES chain 最後。
- 每個 stateful instance 只服務一條 stream；換檔、換 session 或 discontinuity 時 reset。

---

