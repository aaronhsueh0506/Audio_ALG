[索引](audio_libraries_c_api_zh_TW.md) ｜ [上一份](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md) ｜ [下一份](audio_libraries_c_api_zh_TW_04_pool_build_checklist.md) <!-- split-nav -->
### 6.6 DeepFilterNet2：`dfn2_prepost.h`

生命週期與每-hop 函式同樣對應（前綴 `dfn2_prepost_`，單路訊號 `in_hop`／`spec_re,spec_im`，
長度 `DFN2_HOP_LEN`／`DFN2_N_BINS`，`_config_defaults(cfg, io_mode)` **沒有** `delay_depth`）。
額外 API：

```c
int  dfn2_prepost_model_lookahead_frames(const DFN2Prepost *p);  /* = 2 */
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
  NULL = 內建 sqrt-Hann，出貨路徑都用 NULL）、`erb_fwd`／`erb_inv`（**兩個模式都必填**，borrowed）、
  `atten_lim_db`（0 = 關閉）。
- 兩個 setter **只能在 hop 之間**呼叫，frame 開著時回 `-1`。要精確理解它買到什麼：
  這是 **per-hop atomicity，不是 per-source-frame consistency**——`DFN2_MASK_LOOKAHEAD` = 1，
  hop *t* 展開的 mask 屬於 source frame *t*−1，而圖的特徵視窗 `[t-1, t, t+1]` 橫跨三拍，
  所以一次被接受的 between-hop 換手仍會有幾拍同時踩到新舊矩陣。要 per-source-frame 一致，
  那就是一個 stream boundary：先 `_reset()`，再換。
- `_output_frame_index()` 給出最近一次輸出屬於哪一個 source frame（自 reset 起 0 起算），
  用來把輸出串流與其他同框時鐘的東西配對；還沒有輸出時回 `-1`。
- 輸出前兩拍是暖機：`_post_process()` 的 `*written` = 0（`out_hop` 仍然寫滿零），
  `_post_process_freq()` 的 `*valid` = 0。收尾 flush 要用全零 hop 再跑 2 拍，**加速器那兩拍也要照跑**。
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
| Align-ULCNet 16 kHz | 4 | 67,856 | 46,064 |
| Align-ULCNet 16 kHz | 8 | 97,040 | 75,248 |
| Align-ULCNet 16 kHz | 16 | 155,408 | 133,616 |
| Align-ULCNet 16 kHz | 32 | 272,144 | 250,352 |
| Align-ULCNet 16 kHz | 64 | 505,616 | 483,824 |
| Align-ULCNet 48 kHz | 4 | 130,320 | 87,024 |
| Align-ULCNet 48 kHz | 8 | 186,128 | 142,832 |
| Align-ULCNet 48 kHz | 16 | 297,744 | 254,448 |
| Align-ULCNet 48 kHz | 32 | 520,976 | 477,680 |
| Align-ULCNet 48 kHz | 64 | 967,440 | 924,144 |
| DeepVQE-S 16 kHz | 32 | 637,072 | 615,280 |
| DeepFilterNet2 48 kHz | — | 246,320 | 219,536 |

DeepVQE-S 的量體由 16 個 state 張量（每個一份 buffer）主宰。DeepFilterNet2 的 FREQ 比 TIME 小 26,784 bytes：
`DFN2State` 不內嵌 analysis/window/synthesis 緩衝，TIME 另 carve 分析與合成兩個 audiolib 物件
（16,480 B 與 12,384 B），FREQ 只多 carve 雙輸入入口的第二組 staging（`apply_re`/`apply_im`）。

### 6.8 Framing：一 hop 一框的 analysis

兩個 AIAEC 類別在 TIME 模式下的 analysis 是 audiolib `audio_stft.h` 的 `AcStft`，`center` = 0：

```c
void ulcnet_framing_config(AcStftConfig *cfg, FftHandle *fft, const float *window, int center);
int  ac_stft_push_planar(AcStft *s, const float *hop_in, float *re, float *im);
```

center=False，**從第一次 push 起就是一 push 一框**（恆回 1），與 AEC/BF seam 的頻譜同一個慣例，
所以只有一路頻譜在手的呼叫端可以用它補另一路而兩路保持 hop 對齊。它與 `center` = 1（center=True，
第一拍 0 框、第二拍 2 框）對同一個 hop 產出的**最後一框 bit-identical**，差別只在 centered 版本多吐一個 reflect-prefix 開頭框。
DeepVQE-S 在 AIAEC 16 kHz／512 邊界上自建同組態的物件。

> **同一個 `AcStft` 不可混用兩種 `center`**：`center` 在建立時固定，兩種排程不共用物件。


### 6.9 Pipeline 層：頻域 stage 串接與 NN 的 begin / nn_begin / nn_end / end

四個 pipeline 範例把三個 library 與這些類別串成一條鏈，並在 `pipelines/samples/` 示範；接縫是
`pipelines/common/se_stage.h` 的 `SeSpecFrame`（借用視圖，有效至其產生者下一次執行；任何 stage 都不得寫入）。
增益類 stage 回傳 `const float*`（`se_res_stage_gain()`、`se_anr_stage_gain()`；RES gain 也可直接讀
`frame.g_res_ctx`），各範例 `main` 在每個 hop 明寫 `sk_min_f32 → sk_capply_gain_f32 → aec_cng_fill`，再 `ac_istft_push`；
沒有函式指標表，也沒有把這幾步包起來的共用物件。

初始化也是各自為政：每個函式庫（`Aec`、`AecRes`（4ch 的範例自己建，單元與核心都不含）、`AnrDenoiser`、`AcIstft`、`UlcnetNnStage`、`DfnResStage`／`DfnRateBridge`、
4ch 單元）有自己的大小查詢與 `init`，整合者加總大小並切塊；記憶體和（每個函式庫一行）見
[`pipelines/README_01_latency_call_flow.md`](../pipelines/README_01_latency_call_flow.md)「Memory Budget」與 [`integration_example.html`](html/integration_example.html)。

跑 NN 的應用把一個 hop 拆成四個呼叫，加速器夾在中間，模型 config 的 `infer` 回呼因此可以不設：

```c
/* DFN2：samples/dfn_hop.h；ULCNet：samples/ulcnet_nn_stage.h（begin 的引數不同） */
dfn_hop_begin(&hop, frame.e, P);                         /* P = E·G_res 已由範例 main 先形成 */
while (dfn_hop_nn_begin(&hop, &job) > 0) {               /* 1 = 有一幀；0 = 沒有（warm-up／reprime） */
    dfn_hop_nn_end(&hop, run_npu(&job));                 /* rc != 0 或輸出非有限 ⇒ identity（fail-open） */
}
dfn_hop_end(&hop, out);                                  /* 合成剛好一個 hop */
```

`job` 是 `UlcnetNnJob`（ULCNet）或 `DfnResStageJob`（DFN2，內含本章 `DFN2PrepostInputs`／`Outputs`）；
輸出預填 NaN，必須寫滿。DFN 應用經 rate bridge 時一個 native hop 可能開 0..2 個 48 kHz 幀，所以 `nn_begin` 放在迴圈裡。
4 聲道 Align-ULCNet 若要用同一 hop 的 lane 頻譜算出的 mask 與 VAD（例如 masker 讀 lane 0），把單元的 hop 拆成
`audio_pipeline_4ch_process_front()`（回傳唯讀 lane 頻譜）與 `audio_pipeline_4ch_process_beam(p, &front, vad_raw, vad_out, mask, &beam, &info)`
（`mask` 可為 NULL）；`info` 回 DOA（rad，NaN = 無）、BF 適應旗標與 `info.delay`。`audio_pipeline_4ch_process_with_activity(v)` 等於
`process_front` 加 `process_beam(front, v, v, NULL)`；`ulcnet_nn_stage_begin_complex(stage, beam.spectrum, far, {delay.changed, delay.solid})` 接在後面。任何一步被拒，整合者把單元與 stage 一起 reset。
沒被取走或沒被關閉的幀走 identity，輸出時序不隨模型健康度改變。範例與記憶體規則見
`pipelines/README_01_latency_call_flow.md` 的 Call flow 與 [`integration_example.html`](html/integration_example.html)。

### 6.10 各函式庫各自初始化：記憶體加總與單元的公開面

沒有「整條 pipeline 的 `get_mem_requirements()`」：每個函式庫查自己的大小，整合者加總（每塊 16-byte 對齊）。
下表是本 checkout、`BACKEND=ne10`、`SIMD=1`、16 kHz／256（預設格點）的實測值；換格點、backend、編譯選項都會變，
一律用各自的查詢重量。

| 函式庫物件 | 大小查詢 | 16 kHz／256 | 備註 |
|---|---|---:|---|
| `Aec`（mono，MATCHED n=5） | `aec_get_mem_size()` | 364,416 | 48 kHz／1024：1,156,544 |
| `AnrDenoiser`（函式庫 preset） | `anr_get_mem_size()` | 45,344 | 範例程式有 NR 時 |
| `AcIstft`（合成） | `ac_istft_get_mem_size()` | 11,344 | 範例程式有合成時 |
| `AecRes`（4ch 後波束 RES） | `aec_res_get_mem_size()` | 7,904 | 單元與核心的 pool 都不含；4ch 的範例自己建一個（48 kHz／1024：29,408） |
| 增益／頻譜暫存 | 範例程式自己算 | 2,624 | mono 為三個 `n_freqs` float 向量 + 一個複數向量，各自 `ALIGN16`；4ch 範例為兩個向量 + 一個複數向量（2,096） |
| 4ch 前端 `FourAecNrRes` | `four_aec_nr_res_get_mem_requirements()` | 906,560 | 共用 aligner + 四條線性 AEC + fuse 緩衝 |
| 4ch 單元 `AudioPipeline4Ch` | `audio_pipeline_4ch_get_mem_requirements()` | 1,698,864 | 前端 + SRP-PHAT + BF（不含 RES）；`core.enable_post = 0` 時 1,693,648 |
| ULCNet stage | `ulcnet_nn_stage_mem_size(0)` | 36,400 | 一個建置一組格點（48 kHz：71,728） |
| DFN2 stage／rate bridge | `dfn_res_stage_get_mem_requirements()`／`dfn_rate_bridge_get_mem_requirements()` | 231,984／349,888 | bridge 內含 stage；16 kHz 走 bridge，48 kHz 直接用 stage 加呼叫端的 `AcIstft` |

mono 的 16 kHz／256 合計 = `ALIGN16(364,416) + 45,344 + 11,344 + 2,624 = 423,728`（AEC + NR + 合成 + 暫存）；4ch 單元加 RES／NR／合成／暫存 = `1,698,864 + (7,904 + 45,344 + 11,344 + 2,096) = 1,765,552`（`audio_pipeline_4ch_static --print-mem-size` 的 `Total:`）。

**單元的公開面**（`audio_pipeline_4ch.h`）：`default_config`、`get_mem_requirements`、`init`／`init_ex`／`create`、
`process_front`／`process_beam`／`process_with_activity`／`process`、`fuse`（可選）、`reset`／`destroy`、
形狀與生效值 accessor、`get_core`、`last_core_timing`。輸出是借用的 `AudioPipeline4ChBeam`
（BF 後頻譜與權重），`fuse()` 之後另有 `SeSpecFrame`（`g_res_ctx` 為 NULL）；RES、NR、comfort noise、合成是你的後級。

**串接順序**（4ch，一 hop）：單元 `process_*` → `audio_pipeline_4ch_fuse()`（`frame.g_res_ctx` 為 NULL）→ `se_res_stage_gain()` → `se_anr_stage_gain()` → `sk_min_f32()` →
`sk_capply_gain_f32()` → `aec_cng_fill()` → `ac_istft_push()`。**mono**：`aec_process_context()` → `se_frame_from_aec()`（`g_res_ctx` 是 `Aec` 自己的 RES gain）→ 同上。
失敗時重置單元（或 `Aec`）、4ch 的 `AecRes`、降噪器、合成與 comfort-noise RNG（以及 NN stage）。

---

[索引](audio_libraries_c_api_zh_TW.md) ｜ [上一份](audio_libraries_c_api_zh_TW_02_ai_prepost_models.md) ｜ [下一份](audio_libraries_c_api_zh_TW_04_pool_build_checklist.md) <!-- split-nav -->
