[索引](audio_libraries_c_api_zh_TW.md) ｜ [上一份](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md) ｜ [下一份](audio_libraries_c_api_zh_TW_04_pool_build_checklist.md) <!-- split-nav -->
### 6.6 DeepFilterNet2：`dfn2_prepost.h`

生命週期與每-hop 函式同樣對應（前綴 `dfn2_prepost_`，單路訊號 `in_hop`／`spec_re,spec_im`，
長度 `DFN2_HOP_LEN`／`DFN2_N_BINS`，`_config_defaults(cfg, io_mode)` **沒有** `delay_depth`）。
額外 API：

```c
int  dfn2_prepost_model_lookahead_frames(const DFN2Prepost *p);  /* = DFN2_MASK_LOOKAHEAD + DFN2_DF_LOOKAHEAD + DFN2_FRAMES_PER_CALL - 1（出貨 1/1、每次 1 個 frame 為 2） */
int  dfn2_prepost_layout_version(const DFN2Prepost *p);
unsigned int dfn2_prepost_state_generation(const DFN2Prepost *p);  /* init 後 1；每次 _reset 與被拒的 commit 加 1 */
int  dfn2_prepost_set_erb_matrices(DFN2Prepost *p, const float *erb_fwd, const float *erb_inv);
int  dfn2_prepost_set_atten_lim(DFN2Prepost *p, float atten_lim_db);
int  dfn2_prepost_post_process(DFN2Prepost *p, float out_hop[DFN2_HOP_LEN], int *written);
int  dfn2_prepost_post_process_freq(DFN2Prepost *p, float re[DFN2_N_BINS], float im[DFN2_N_BINS],
                                    int *valid);
int  dfn2_prepost_output_frame_index(const DFN2Prepost *p, long long *frame);
```

- `cfg` 欄位：`io_mode`、`fft`（`DFN2_IO_TIME` 必填）、`window`（可選，**會被複製**，
  NULL = 內建 sqrt-Hann，出貨路徑都用 NULL）、`istft`（`DFN2_IO_TIME` 必填，應用端持有的 `AcIstft`，borrowed；用 `dfn2_framing_config()` 建）、`erb_fwd`／`erb_inv`（**兩個模式都必填**，borrowed）、
  `atten_lim_db`（0 = 關閉）。
- **Lookahead pair。** mask lookahead `M`（`DFN2_MASK_LOOKAHEAD`，0..2）與 deep-filter lookahead `D`（`DFN2_DF_LOOKAHEAD`，0..4）
  是建置時決定的一對（共 15 個合法組合，兩者之間沒有大小限制），預設都是 1。匯出的 ONNX graph 不隨這一對改變，隨之改變的是 host 端
  的視窗（`heads[t]` 用 source frame `[t-(2-M) .. t+M]`）與就緒條件（看過 `M+1` 個 frame 才開始呼叫加速器）；checkpoint 與它訓練時的那一對綁定，
  目前只有出貨的 1/1 有 checkpoint，其他組合的音質沒有量測。C 端以 `-DDFN2_MASK_LOOKAHEAD=<M> -DDFN2_DF_LOOKAHEAD=<D>` 套到每個編譯單元
  （`pipelines/Makefile` 的同名變數會轉成這兩個旗標）；不同組合的 object 無法連結。細節見 `AINR/README.md` 的 "Lookahead pairs"。
- 兩個 setter **只能在 hop 之間**呼叫，frame 開著時回 `-1`。要精確理解它買到什麼：
  這是 **per-hop atomicity，不是 per-source-frame consistency**——hop *t* 展開的 mask 屬於 source frame *t*−*M*，
  而圖的特徵視窗橫跨三拍（出貨 1/1：`[t-1, t, t+1]`），
  所以一次被接受的 between-hop 換手仍會有幾拍同時踩到新舊矩陣。要 per-source-frame 一致，
  那就是一個 stream boundary：先 `_reset()`，再換。
- `_output_frame_index()` 給出最近一次輸出屬於哪一個 source frame（自 reset 起 0 起算），
  用來把輸出串流與其他同框時鐘的東西配對；還沒有輸出時回 `-1`。
- 暖機：`_pre_process()` 的前 `M` 拍回 0（不要跑加速器）；輸出前 `M + D` 拍（出貨 1/1 為兩拍）是暖機：`_post_process()` 的 `*written` = 0
  （`out_hop` 仍然寫滿零），`_post_process_freq()` 的 `*valid` = 0。收尾 flush 要用全零 hop 再跑 `M + D` 拍，**加速器那幾拍也要照跑**。
  DFN2 以 `-DDFN2_FRAMES_PER_CALL=N`（N > 1，需要 `export_onnx.py --frames-per-call N` 匯出的 graph）建置時，`_pre_process()` 每 N 拍回 1 一次、暖機與 flush 都多 `N - 1` 拍；pipelines 固定 N = 1（見 [ainr_dfn2_detail](html/ainr_dfn2_detail.html)）。
- **`DFN2_IO_FREQ` 的頻譜是 `torch.stft(normalized=True)` 尺度**（乘了 1/√`DFN2_N_FFT` = 1/32），
  AIAEC 兩個模型交出的則是**未正規化**的 rfft。直接串接會剛好差 32 倍（過驅動或欠驅動），
  而且兩邊都偵測不到——都是量綱合法的 float 頻譜，ERB 特徵又是 EMA 正規化的。
  兩者也不同 grid（48 kHz/1024/512 對 16 kHz/512），中間本來就有 resampler，換算就放在那一層。
  `DFN2_ANALYSIS_SCALE` **不是**這個因子（它只作用在特徵分支的拷貝上）。

### 6.7 Pool 大小

以現行 build 量得（NEON 開關不影響數字），供規劃用；產品一律以 resolved config 呼叫 `_get_mem_size()`，
**不要把這裡的數字寫死**（[§7](audio_libraries_c_api_zh_TW_04_pool_build_checklist.md#7-static-pool-共通規則) 規則同樣適用）。單位 bytes。

| 類別 / grid | D | `*_IO_TIME` | `*_IO_FREQ` |
|---|---:|---:|---:|
| Align-ULCNet 16 kHz | 4 | 59,568 | 46,064 |
| Align-ULCNet 16 kHz | 8 | 88,752 | 75,248 |
| Align-ULCNet 16 kHz | 16 | 147,120 | 133,616 |
| Align-ULCNet 16 kHz | 32 | 263,856 | 250,352 |
| Align-ULCNet 16 kHz | 64 | 497,328 | 483,824 |
| Align-ULCNet 48 kHz | 4 | 113,840 | 87,024 |
| Align-ULCNet 48 kHz | 8 | 169,648 | 142,832 |
| Align-ULCNet 48 kHz | 16 | 281,264 | 254,448 |
| Align-ULCNet 48 kHz | 32 | 504,496 | 477,680 |
| Align-ULCNet 48 kHz | 64 | 950,960 | 924,144 |
| DeepVQE-S 16 kHz | 32 | 628,784 | 615,280 |
| DeepFilterNet2 48 kHz | — | 233,936 | 219,536 |

DeepVQE-S 的量體由 16 個 state 張量（每個一份 buffer）主宰。DeepFilterNet2 的 FREQ 比 TIME 小 14,400 bytes：
`DFN2State` 不內嵌 analysis/window/synthesis 緩衝，TIME 另 carve 輸出 hop staging 與分析用的 `AcStft`
（16,480 B），FREQ 只多 carve 雙輸入入口的第二組 staging（`apply_re`/`apply_im`）。合成用的 `AcIstft`
（出貨 1/1 為 12,384 B，借用 FFT handle 與窗表）是應用端的物件，兩個模式的 pool 都不含它；它不隨 D 變，
所以各列的 TIME 欄都不含 `AcIstft`（Align-ULCNet／DeepVQE-S 的 ENV_NORM 合成：16 kHz grid 8,288 B、48 kHz grid 16,480 B）。

### 6.8 Framing：一 hop 一框的 analysis

兩個 AIAEC 類別在 TIME 模式下的 analysis 是 audiolib `audio_stft.h` 的 `AcStft`，`center` = 0：

```c
void ulcnet_framing_config(AcStftConfig *cfg, FftHandle *fft, const float *window, int center);
int  ac_stft_push_planar(AcStft *s, const float *hop_in, float *re, float *im);
```

center=False，**從第一次 push 起就是一 push 一框**（恆回 1），與 AEC/BF seam 的頻譜同一個慣例，
所以只有一路頻譜在手的呼叫端可以用它補另一路而兩路保持 hop 對齊。它與 `center` = 1（center=True，
第一拍 0 框、第二拍 2 框）對同一個 hop 產出的**最後一框 bit-identical**，差別只在 centered 版本多吐一個 reflect-prefix 開頭框。
DeepVQE-S 在 AIAEC 16 kHz／512 邊界上用 `deepvqe_framing_config()` 取得同組態（分析與 `AcIstft` 合成共用）。

> **同一個 `AcStft` 不可混用兩種 `center`**：`center` 在建立時固定，兩種排程不共用物件。


### 6.9 Pipeline 層：頻域 stage 串接與 NN 的 begin / nn_begin / nn_end / end

四個 pipeline 範例把三個 library 與這些類別串成一條鏈，範例碼在 `pipelines/mono_aec_nr_res/`、`pipelines/4ch_aec_bf_nr_res/`、`pipelines/dfn/`、`pipelines/ulcnet/`；接縫是借用視圖（mono 為 `aec.h` 的 `AecResContext`，4ch 為 `mcab.h` 的 `McabFrame`；有效至其產生者下一次執行；任何 stage 都不得寫入）。
增益類 stage 回傳 `const float*`（`aec_res_process()`、`anr_get_gain()`；mono 的 RES gain 也可直接讀
`ctx.res_gain`），各範例 `main` 在每個 hop 明寫 `sk_min_f32 → sk_capply_gain_f32 → aec_cng_fill`，再 `ac_istft_push`；
沒有函式指標表，也沒有把這幾步包起來的共用物件。

初始化也是各自為政：每個函式庫（`Aec`、`DelayAec`、`AecRes`（4ch 的範例自己建，`Mcab` 不含）、`AnrDenoiser`、`AcIstft`、`UlcnetNnStage`、`DfnResStage`／`DfnRateBridge`、
4ch 的 `Mcab`）有自己的大小查詢與 `init`，整合者加總大小並切塊；記憶體和（每個函式庫一行）見
[`pipelines/README_01_latency_call_flow.md`](../pipelines/README_01_latency_call_flow.md)「Memory Budget」與 [`integration_example.html`](html/integration_example.html)。

跑 NN 的應用把一個 hop 拆成四個呼叫，加速器夾在中間，模型 config 的 `infer` 回呼因此可以不設：

```c
/* DFN2：dfn/dfn_hop.h；ULCNet：ulcnet/ulcnet_nn_stage.h（begin 的引數不同） */
dfn_hop_begin(&hop, frame.e, P);                         /* P = B·G_res（B = frame.b）已由範例 main 先形成 */
while (dfn_hop_nn_begin(&hop, &job) > 0) {               /* 1 = 有一幀；0 = 沒有（warm-up／reprime） */
    dfn_hop_nn_end(&hop, run_npu(&job));                 /* rc != 0 或輸出非有限 ⇒ identity（fail-open） */
}
dfn_hop_end(&hop, out);                                  /* 合成剛好一個 hop */
```

`job` 是 `UlcnetNnJob`（ULCNet）或 `DfnResStageJob`（DFN2，內含本章 `DFN2PrepostInputs`／`Outputs`）；
輸出預填 NaN，必須寫滿。DFN 應用經 rate bridge 時一個 native hop 可能開 0..2 個 48 kHz 幀，所以 `nn_begin` 放在迴圈裡。
4 聲道 Align-ULCNet 若要用同一 hop 的 lane 頻譜算出的 mask 與 VAD（例如 masker 讀 lane 0），把 sample helper 的 hop 拆成
`mcab_sample_beam_process_front()`（回傳唯讀 lane 頻譜）與 `mcab_sample_beam_process_beam(b, &front, vad_raw, vad_out, mask, &beam, &info)`
（`mask` 可為 NULL）；`info` 回 DOA（rad，NaN = 無）、BF 適應旗標與 `info.delay`。`mcab_sample_beam_process_with_activity(b, mics, far, v, v, NULL, &beam, &info)` 等於
`process_front` 加 `process_beam(front, v, v, NULL)`；`ulcnet_nn_stage_begin_complex(stage, beam.spectrum, far, {info.delay.changed, info.delay.solid})` 接在後面。任何一步被拒，整合者把 helper（連同 flow）與 stage 一起 reset。
沒被取走或沒被關閉的幀走 identity，輸出時序不隨模型健康度改變。範例與記憶體規則見
`pipelines/README_01_latency_call_flow.md` 的 Call flow 與 [`integration_example.html`](html/integration_example.html)。

### 6.10 各函式庫各自初始化：記憶體加總與 flow 的公開面

沒有「整條 pipeline 的 `get_mem_requirements()`」：每個函式庫查自己的大小，整合者加總（每塊 16-byte 對齊）。
下表是本 checkout、`BACKEND=ne10`、`SIMD=1`、16 kHz／256（預設格點）的實測值；換格點、backend、編譯選項都會變，
一律用各自的查詢重量。

| 函式庫物件 | 大小查詢 | 16 kHz／256 | 備註 |
|---|---|---:|---|
| `Aec`（mono，MATCHED n=5） | `aec_get_mem_size()` | 366,000 | 48 kHz／1024：1,162,736 |
| `AnrDenoiser`（函式庫 level） | `anr_get_mem_size()` | 45,344 | 範例程式有 NR 時 |
| `AcIstft`（合成） | `ac_istft_get_mem_size()` | 11,344 | 範例程式有合成時；mono 範例的合成借用 AEC 的合成窗（`aec_synthesis_window()`），不另帶窗表，為 10,320 |
| `AecRes`（4ch 後波束 RES） | `aec_res_get_mem_size()` | 7,904 | `Mcab` 的 pool 不含；4ch 的範例自己建一個（48 kHz／1024：29,408） |
| 增益／頻譜暫存 | 範例程式自己算 | 2,624 | mono 與 `mcab_external_static` 為三個 `n_freqs` float 向量 + 一個複數向量，各自 `ALIGN16`；`mcab_static` 為兩個向量 + 一個複數向量（2,096） |
| 4ch lane（`Aec` × 4，`mcab_lane_config()`） | `aec_get_mem_size()` 每條 | 803,968（4 × 200,992） | 48 kHz／1024：2,936,832；由應用建立，不在 `Mcab` 的 pool 內 |
| 4ch flow `Mcab`（含共用 estimator 的 front end） | `mcab_get_mem_requirements()`；estimator 另用 `delay_aec_get_mem_size()` | `Mcab` pool 74,624；front end（estimator + `Mcab` pool）110,048 | `enable_post = 0` 時 `Mcab` pool 為 68,368；front end 48 kHz／1024：267,072（`mcab_external_static --print-mem-size`） |
| 4ch sample 的 `Unit`（lane + estimator + `Mcab` pool + SRP-PHAT／BF helper） | `mcab_sample_unit_memory()` | 1,706,240 | 48 kHz／1024：6,329,312（`mcab_static --print-mem-size`） |
| ULCNet stage | `ulcnet_nn_stage_mem_size(0)` | 36,400 | 一個建置一組格點（48 kHz：71,728） |
| DFN2 stage／rate bridge | `dfn_res_stage_get_mem_requirements()`／`dfn_rate_bridge_get_mem_requirements()` | 231,984／349,888 | bridge 內含 stage；16 kHz 走 bridge，48 kHz 直接用 stage 加呼叫端的 `AcIstft`；數字是出貨的 lookahead 1/1，其他組合的 pool 大小要現查 |

mono 的 16 kHz／256 合計 = `ALIGN16(366,000) + 45,344 + 10,320 + 2,624 = 424,288`（AEC + NR + 合成 + 暫存）；4ch sample 的 `Unit` 加 RES／NR／合成／暫存 = `1,706,240 + (7,904 + 45,344 + 11,344 + 2,096) = 1,772,928`（`mcab_static --sample-rate 16000` 印的 `Total:`）。

**flow 的公開面**（`mcab.h`）：`mcab_default_config`、`mcab_apply_lane_contract`／`mcab_lane_config`／`mcab_delay_estimator_config`、
`mcab_get_mem_requirements`、`mcab_init`／`mcab_init_ex`／`mcab_create`、`mcab_process_front`、`mcab_fuse`（可選）、`mcab_reset`／`mcab_destroy`、
形狀與生效值 accessor（`mcab_hop_size()`、`mcab_fft_size()`、`mcab_n_freqs()`、`mcab_sample_rate()`、`mcab_enable_post()`）、`mcab_get_last_timing`。
輸出是借用的 `McabFront`（lane 頻譜與對齊後的 far），`mcab_fuse()` 之後另有 `McabFrame`（`g_res_ctx` 為 NULL）；beamformer、RES、NR、comfort noise、合成是你的後級
（範例的 beamformer 是 sample 層 helper `mcab_sample_beam_*`，不在 archive 內）。

**串接順序**（4ch，一 hop）：`mcab_process_front()` → beamformer → `mcab_fuse()`（`frame.g_res_ctx` 為 NULL）→ `mcab_sample_res_input()` + `aec_res_process()` → `mcab_sample_anr_gain()` → `sk_min_f32()` →
`sk_capply_gain_f32()` → `aec_cng_fill()` → `ac_istft_push()`。**mono**：`aec_process_context()` → `aec_get_res_context()`（`ctx.res_gain` 是 `Aec` 自己的 RES gain）→ `anr_process_gain()`／`anr_get_gain()` → 同上。
失敗時重置 flow（`mcab_reset()` 連 lane 與 estimator 一起重置；或 mono 的 `Aec`）、4ch 的 `AecRes`、降噪器、合成與 comfort-noise RNG（以及 NN stage）。

---

[索引](audio_libraries_c_api_zh_TW.md) ｜ [上一份](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md) ｜ [下一份](audio_libraries_c_api_zh_TW_04_pool_build_checklist.md) <!-- split-nav -->
