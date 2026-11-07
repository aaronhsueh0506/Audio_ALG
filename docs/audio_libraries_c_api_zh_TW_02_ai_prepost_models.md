[索引](audio_libraries_c_api_zh_TW.md) ｜ [上一份](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md) ｜ [下一份](audio_libraries_c_api_zh_TW_03_ai_prepost_pipeline.md) <!-- split-nav -->
## 6. AI 模型 pre/post 類別

六個 AI 模型函式庫都**只在頻域工作**：收頻譜、交出頻譜，把「頻譜 ↔ 加速器張量」之間的特徵前端、遞迴 ring 與狀態交易收成一個物件。
framing、window、FFT／IFFT 與 overlap-add **不在函式庫裡**：應用端用 audiolib 的 `AcStft`／`AcIstft` 自己做（[§6.8](audio_libraries_c_api_zh_TW_03_ai_prepost_pipeline.md#68-framing應用端用-audiolib-分析與合成)），
`make -C pipelines audit-no-transform` 把關模型目錄。遞迴 state 也歸應用端（[§6.2](#62-每-hop-序列與回傳值合約)）。

| 模型 | Public header | 物件 | 訊號輸入 |
|---|---|---|---|
| Align-ULCNet | `lib/aiaec/ulcnet/ulcnet_prepost.h` | `UlcnetPrepost` | linear-AEC error + raw far |
| DeepVQE-S | `lib/aiaec/deepvqe_s/deepvqe_prepost.h` | `DeepVqePrepost` | **raw microphone** + far reference |
| DeepFilterNet2 | `lib/ainr/dfn2/dfn2_prepost.h` | `DFN2Prepost` | 單路 noisy |
| DeepFilterNet3 | `lib/ainr/dfn3/dfn3_prepost.h` | `DFN3Prepost` | 單路 noisy；raw 低頻 DF／高頻 ERB |
| GTCRN | `lib/ainr/gtcrn/gtcrn_process.h` | `GTCRNProcess` | 單路 noisy（interleaved 頻譜） |
| RNNoise-ERB | `lib/ainr/rnnoise_erb/process.h` | `RNNoiseState` | 單路 noisy |

前四個是 `*_prepost` 類別（`frame_open`／`frame_finish`／`frame_bypass` 三步，§6.1–6.6）；GTCRN 與 RNNoise-ERB 是同樣由應用端給 pool 的物件，但沒有 `frame_*`，呼叫序見 §6.6。
四個 prepost 類別各是獨立的翻譯單元（`*_prepost.c`），組合同目錄的 `*_process.c` 與（DFN2、DFN3、Align-ULCNet 的）`*_model_io.c`；這些檔案要能單獨連結給各自的 parity 測試，所以類別不放進其中任何一個。

### 6.1 共通生命週期

六個函式庫的生命週期形狀相同：**grid 在執行期給定 → 查 pool 大小 → 呼叫端配置 → init**。沒有編譯期容量，也沒有建置期的 grid 或 lookahead 旗標。
前綴為 `ulcnet_` / `deepvqe_` / `dfn2_` / `dfn3_`（`*_prepost_*`）；GTCRN 是 `gtcrn_process_*`，RNNoise-ERB 是 `rnnoise_*`。

```text
<Lib>GridSpec spec = { 匯出的模型固定什麼就填什麼 }
  -> <lib>_grid_make(&grid, &spec)        回 0（<LIB>_GRID_OK），否則第一個失敗的狀態碼，grid 不動
  -> _config_defaults(&cfg, &grid)        grid 按值複製；ERB 矩陣與 state 登錄為 NULL／0
  -> cfg.erb_fwd／cfg.erb_inv（DFN2、DFN3、GTCRN、RNNoise-ERB）；float host 另登錄 state（§6.2）
  -> _get_mem_size(&cfg, &req)
  -> _init(pool, req.bytes, &cfg)  或  _init_ex(pool, bytes, &cfg, &expected)
  -> 每-hop 序列（6.2）
  -> _reset(p)（stream boundary）
  -> _destroy(p)（只有 _create 的實例需要）
```

```c
DFN2GridSpec spec = {
    .sample_rate = 48000, .n_fft = 1024, .n_erb = 32, .df_bins = 96, .df_order = 5,
    .mask_lookahead = 1, .df_lookahead = 1, .frames_per_call = 1,
    .erb_norm_tau_sec = 1.0f, .erb_norm_init_lo_db = -60.0f, .erb_norm_init_hi_db = -90.0f,
    .erb_norm_scale_db = 40.0f, .spec_norm_tau_sec = 1.0f,
    .spec_norm_init_lo = 0.001f, .spec_norm_init_hi = 0.0001f,
};
DFN2Grid grid;
DFN2PrepostConfig cfg;
DFN2PrepostMemReq req;
if (dfn2_grid_make(&grid, &spec) != DFN2_GRID_OK) return -1;
dfn2_prepost_config_defaults(&cfg, &grid);
cfg.erb_fwd = erb.fwd;                        /* 應用端讀進來的 ERB 矩陣，借用（§6.6） */
cfg.erb_inv = erb.inv;
if (dfn2_prepost_get_mem_size(&cfg, &req) != 0) return -1;
void *pool = alloc16(req.bytes);              /* 至少 16-byte 對齊；alloc16 是應用端自己的配置函式 */
DFN2Prepost *p = dfn2_prepost_init_ex(pool, req.bytes, &cfg, &req);
```

| 函式庫 | `*GridSpec` 欄位 | 出貨值 |
|---|---|---|
| DFN2 | `sample_rate`、`n_fft`、`n_erb`、`df_bins`、`df_order`、`mask_lookahead`、`df_lookahead`、`frames_per_call`，加七個 normaliser 欄位 | 48000／1024／32／96／5／1／1／1 |
| DFN3 | 同 DFN2，沒有 `frames_per_call` | 48000／1024／32／96／5／1／1 |
| GTCRN | `n_fft`、`erb_low_bins`、`erb_high_bands` | 512／65／64（模型是 16 kHz，取樣率不是 grid 欄位） |
| RNNoise-ERB | `sample_rate`、`n_fft`、`n_bands`、`min_bins_per_band`、`lookahead_frames`、`spec_max_hz`，加 normaliser 欄位與 `spec_norm_eps` | 16000／512／22／2／1／4000 |
| Align-ULCNet | `sample_rate`、`n_fft`、`delay_depth` | 16000／512／D（匯出時的值） |
| DeepVQE-S | `sample_rate`、`n_fft`、`delay_depth` | 16000／512／32 |

DFN2、DFN3、GTCRN、RNNoise-ERB 的出貨值取自各自的 `config.ini`；Align-ULCNet 的 D 是匯出時的值，DeepVQE-S 的 D 見 §6.5。normaliser 欄位是 `erb_norm_tau_sec`、`erb_norm_init_lo_db`、`erb_norm_init_hi_db`、`erb_norm_scale_db`、`spec_norm_tau_sec`、`spec_norm_init_lo`、`spec_norm_init_hi`，須等於訓練值；
兩個 EMA alpha 由 `dfn2_norm_alpha()`／`dfn3_norm_alpha()`／`rnnoise_norm_alpha()` 推導，手改 hop、alpha 或 scale 的 grid 會被 `*_grid_validate()` 拒絕。

- **grid 的共通檢查。** `sample_rate > 0`；`n_fft` 是 audiolib 能建 framing 的大小（2 的冪、16..8192，即 `ac_stft_get_mem_size()` 非 0）；`hop = n_fft / 2`、`n_bins = n_fft / 2 + 1` 由公式推導。
  各庫的範圍檢查與狀態碼見 `dfn2_grid.h`、`dfn3_process.h`、`gtcrn_process.h`、`process.h`、`ulcnet_model_io.h`、`deepvqe_prepost.h`。
  函式庫沒有 preset：(sample rate, FFT size) 對應 `n_erb`、`df_bins`、lookahead pair 與 normaliser 的表是模型的性質，由應用端持有。
- `_get_mem_size()` 是 reject-first：NULL 引數、無效 grid、缺 ERB 矩陣（DFN2、DFN3、GTCRN、RNNoise-ERB）、非有限的 `atten_lim_db`（DFN2、DFN3）、不一致的 state 登錄（§6.2）
  （Align-ULCNet 另加：`model_layout_version` 不符），一律回 `-1` 且 `*req` 一個 byte 都不寫。
- `_init_ex()` 是 **stale-pool gate**：固定 32-byte 的 `*MemReq`（`descriptor_version`、`layout_version`、`reserved0`（必須為 0）、`build_flags_hash`（FNV-1a-32）、`alignment`、`reserved`、`bytes`；
  RNNoise-ERB 的 `RNNoiseMemReq` 是 24 bytes，沒有兩個 reserved 欄位）與本 build 不符就拒絕，而不是重新解讀一塊為別的組態配出來的 pool。hash 折入整個 grid 與版本；
  Align-ULCNet 與 DeepVQE-S 的 hash 含 D，但 pool 大小與 D 無關。`expected` 傳 NULL 時等同 `_init()`。
- `_create()` 是 get_mem_size + 對齊配置 + init，給 host 工具與測試，只有 DFN2、DFN3、Align-ULCNet、DeepVQE-S 有；板端走 `_init()`。
  `_destroy()` 只釋放 `_create()` 配置的東西：對 `_init()` 實例它是**可重複呼叫的 no-op**，對 `_create()` 實例呼叫第二次就是 double free；它從不釋放 ERB 矩陣或登錄的 state。
- 借用而不擁有：ERB 矩陣、登錄的 state 緩衝與 pool 都是 caller-owned，必須活得比實例久。類別裡沒有 `FftHandle`、window 表或 `AcStft`／`AcIstft`。
- `_reset()` 清空登錄的 state，並（各庫有的部分）清空 ring 與輸出 staging、丟掉未結束的 frame、讓 clock 與 normaliser 回 init 值；state generation 加一；config、borrowed 指標與 pool 都不動。
  **應用端自己的 `AcStft`／`AcIstft` 要一起 reset。**

### 6.2 每-hop 序列與回傳值合約

```c
/* 應用端：每路各一個 AcStft；合成用 AC_OLA_ENV_NORM 的 AcIstft（§6.8） */
ac_stft_push_planar(err_stft, err_hop, err_re, err_im);
ac_stft_push_planar(far_stft, far_hop, far_re, far_im);
if (ulcnet_prepost_pre_process_freq(p, err_re, err_im, far_re, far_im) == 1) {  /* 需要幾次推論 */
    UlcnetModelIoInputs  in;
    UlcnetModelIoOutputs out;
    ulcnet_prepost_frame_open(p, &in, &out);   /* 指向本實例 pool 與登錄緩衝的 view */
    /* 板端：先比對 state generation，變了就重填 state（見下） */
    /* 每個 out.*_out 綁到 in 的同名位址；先讀完所有 state 輸入、才寫輸出。 */
    if (accelerator_run(&in, &out) != 0 ||
        ulcnet_prepost_frame_finish(p) != 0)
        ulcnet_prepost_frame_bypass(p);            /* 保住 framing 節奏 */
}
ulcnet_prepost_post_process_freq(p, out_re, out_im);
if (ac_istft_push_planar(istft, out_re, out_im, out_hop) == 0)   /* ENV_NORM 的第一框不輸出 */
    memset(out_hop, 0, hop_size * sizeof(float));
```

- **`_pre_process_freq()` 回的是「這一拍需要的加速器呼叫次數」，不是布林。**
  Align-ULCNet 與 DeepVQE-S **從第 0 拍起恆為 1**（一 hop 進、一次推論、一 hop 出）；
  DeepFilterNet2／3（DFN2 `frames_per_call` N = 1）**也是每拍恆為 1**：圖吃 `[t-(2-M), …, t+M]`，前 M 拍（M = `mask_lookahead`；出貨 M = 1 即第 0 拍）右鄰居還沒到，
  那幾拍是**暖機幀**（warm-up frame）：加速器照跑，但 heads 描述的是串流開始之前的一幀，由 `_frame_finish()`／`_frame_bypass()` 丟棄、已登錄的 state 歸零、generation 加一（見下），
  所以第一個真幀仍從零 state 開始、音訊逐位元不變。M = 0 沒有暖機幀。DFN2 的 N > 1 維持舊流程：前面的拍回 0，之後每 N 拍回 1 一次（不可在回 0 的拍呼叫 `_frame_open()`）。已有 frame 開著時回 `-1`。
- `_frame_open()` 只把**非 state 的輸出**（heads：`output`／`taps`／`erb_mask`／`coefs`／`alpha`）**預填 NaN**，所以加速器只寫一半會在 `_frame_finish()` 被抓到，而不是把上一框的值當成本框結果。
  **state 不預填**（它同時是下一框的輸入）。回 `0`，沒有開啟中的 frame 回 `-1`。
- `_frame_finish()` 先驗證 heads（與有登錄時整個登錄緩衝）都是有限值，成功才把 heads 套到頻譜並 stage 增強頻譜（DFN2／DFN3 的 `*_enhance_spectrum()`、Align-ULCNet 的 mask、DeepVQE-S 的 CCM ring）；它不複製 state。
  失敗時登錄的 state **全部歸零**（冷啟動）、state generation 加一，framing／CCM ring 不前進、frame 維持開啟、回 `-1`。沒有登錄 state 的板端形態只檢查 heads，被拒時 generation 一樣加一。
  **DFN2／DFN3 的暖機幀例外**：`_frame_finish()` 不讀也不驗 heads（NaN 也不拒絕）、不驗 state、不輸出頻譜，只讓內部時鐘前進一拍、登錄的 state 歸零、generation 恰好加一（沒登錄也加）並回 `0`；`_frame_bypass()` 對暖機幀做同樣的事。`dfn2_model_run_frame()`／`dfn3_model_run_frame()` 在暖機幀 `infer()` 成功時回 1，不代表 heads 被採用。
  接著只有兩條路：`_frame_bypass()`，或重新 `_frame_open()` 再從冷 state 跑一次加速器。登錄的緩衝**整個**掃描（含較舊的 history 格）；非有限值以外的「寫一半／混寫」都偵測不到。
- **`_frame_finish()` 前面一定要有一次 `_frame_open()`**（`prepared` 閂鎖，四個類別皆然）：沒跑過的加速器不能把沒被碰過的 buffer 當成結果送出。
- **遞迴 state 歸應用端。** 函式庫不知道 state 的 shape：沒有 state shape 的巨集，也沒有持有 state 的 struct；它只看元素數。兩種形態：
  - **float host** 把緩衝登錄進 config（指標加元素數，borrowed、不複製、須活得比實例久；不計入 pool 大小與 hash）。函式庫在 init 與 reset 時歸零、每次 `_frame_finish()` 做 finite 檢查、被拒時歸零，並在 `Inputs`／`Outputs` view 裡給同一組位址。
  - **板端** 不登錄任何東西（view 裡是 NULL 與 0）；state 在加速器自己的緩衝，函式庫只數重啟次數（state generation）。
  - 兩種形態都是**就地綁定，沒有複製路徑**：runtime 把每個 `*_out`（C 端欄位名 `*_next`）綁到 `_frame_open()` 回傳的同名輸入位址，**先讀完**所有 state 輸入、**才寫**輸出；graph 回完整的下一個 window（位移在 graph 內）。
    被拒時無法回滾：遞迴 state 從零重來、generation 加一；回報失敗的 runtime 視為沒有寫入。
  - **登錄是 all-or-none**：指標與元素數須同時有或同時無，指標配 0 個元素、或元素數配 NULL 指標都被拒；已登錄的組數不齊也被拒。

| 函式庫 | 登錄欄位（config） | 成組規則 |
|---|---|---|
| DFN2 | `gru_hidden`、`df_convp_history`（各配 `*_elements`） | 兩組都填，或四個值全 NULL／0；`-DDFN2_MASK_ONLY` 時只有 `gru_hidden`，任何 `df_convp_history` 值都被拒 |
| DFN3 | 同 DFN2 | 兩組都填，或四個值全 NULL／0 |
| GTCRN | `state[8]`、`state_elements[8]`，依 `GTCRNStateId` | 八組全填或全無 |
| RNNoise-ERB | `h1`、`h2`、`h3`（各配 `*_elements`） | 三組全填或全無 |
| Align-ULCNet | `key_history`、`value_history`、`logit_history`、`h_gru`（`h_gru` 的元素數欄位叫 `gru_hidden_elements`） | 四組全填或全無 |
| DeepVQE-S | `state[16]`、`state_elements[16]`，依 `DeepVqeStateId` | 十六組全填或全無 |

- **state generation**：`ulcnet_model_io_state_generation()`、`ulcnet_prepost_state_generation()`、`ulcnet_accelerator_adapter_state_generation()`、
  `deepvqe_prepost_state_generation()`、`dfn2_prepost_state_generation()`、`dfn2_model_io_state_generation()`、`dfn3_prepost_state_generation()`、`dfn3_model_io_state_generation()`、
  `gtcrn_process_state_generation()`、`rnnoise_state_generation()`（pipelines 的 `dfn_res_stage_state_generation()`／`dfn_rate_bridge_state_generation()` 見 §6.9）
  回傳 C 端重啟 state 的次數：`_reset()` 與被拒絕的 `_frame_finish()`（GTCRN 為被拒的 `gtcrn_process_mask_validate()`／`gtcrn_process_state_validate()`，RNNoise-ERB 為被拒的 `rnnoise_frame_finish()`）各加一，
  有沒有登錄 state 都一樣；新建實例的起始值：Align-ULCNet、DeepVQE-S 為 0，DFN2、DFN3、GTCRN、RNNoise-ERB 為 1（**永遠比較、不要假設起始值**）；
  真幀成功的 `_frame_finish()`、真幀的 `_frame_bypass()`，以及被拒之後在重新 `_frame_open()` 之前再呼叫的 `_frame_finish()` 都不動它；DFN2／DFN3 另有一個來源：`_reset()` 之後每個暖機幀（N = 1、M ≥ 1）的 `_frame_finish()`／`_frame_bypass()` 各加一，板端在每個暖機幀之後都要重填一次量化零（M = 2 重填兩次）；`NULL` 回 0，數值回繞於 2^32。
  定點 accelerator 的 state 緩衝在 C 清零時不會跟著變，板端的規則是：

  ```c
  /* 啟動：先載入量化後的 0，再記下函式庫當下的值 */
  g_state_gen_seen = ulcnet_prepost_state_generation(p);
  /* 每次跑加速器之前 */
  unsigned gen = ulcnet_prepost_state_generation(p);
  if (gen != g_state_gen_seen) { reload_quantized_zero_state(); g_state_gen_seen = gen; }
  ```

- **Align-ULCNet 的四個 state**（`key_history`／`value_history`／`logit_history`／`h_gru` ↔ 對應的 `*_out`）：輸入是完整 window、graph 回每個 state 的**完整下一個 window**，shape 與輸入完全相同：
  K/V `[1,32,D,TA]`（newest-first）取輸入 slot `0..D-2` 再把新一幀接在前面，logit `[1,32,5,D]`（oldest-first）
  取輸入 frame `1..4` 再把新的一列接在後面（輸入端 Slice 加 Concat，沒有 state 輸出由 Slice 產生），`h_gru`
  `[1,4,1,128]` 是兩顆 subband GRU 的 hidden 疊在一起；沒有 CPU 端的 ring helper。這些 shape 是匯出模型的性質，由應用端算出元素數登錄。
  `output` 是學到的複數 mask（planar `[1,2,1,BINS]`），`_frame_finish()` 驗證後以壓縮域 error 乘上它並逆冪。
- `_frame_bypass()` 不動任何 state（維持它當下的樣子；回報失敗的 runtime 視為沒有寫入）；DFN2／DFN3 的暖機幀除外（state 歸零、generation 加一，見 `_frame_finish()`）。

拒絕情境整理：

| 呼叫 | 情境 | 回傳 |
|---|---|---|
| `_pre_process_freq` | 上一個 frame 還開著（既未 finish 也未 bypass） | `-1`（一拍被拒絕，不會默默疊上去） |
| `_frame_open` / `_frame_finish` / `_frame_bypass` | 沒有開啟中的 frame | `-1` |
| `_frame_finish` | 沒有 `_frame_open()` 在前面 | `-1`，狀態不動 |
| `_frame_finish` | heads 或登錄的 state 有非有限值 | `-1`；登錄 state 歸零、generation 加一、frame 維持開啟 |
| `dfn2_prepost_frame_finish` / `dfn3_…` | 暖機幀（N = 1，`_reset()` 或 init 後前 M 拍） | `0`；不讀 heads（NaN 也不拒絕），登錄 state 歸零、generation 加一 |
| `rnnoise_frame_finish` | 增益或登錄的 state 非有限、NULL | `-1`（用 unit 增益）；登錄 state 歸零、generation 加一（NULL 不動任何東西） |
| `gtcrn_process_mask_validate` / `gtcrn_process_state_validate` | mask 或登錄的 state 非有限 | `-1`（頻譜直通）；登錄 state 歸零、generation 加一 |
| `dfn2_prepost_set_erb_matrices` / `_set_atten_lim` | frame 開著 | `-1` |

### 6.3 bypass 政策：各模型不一樣

`_frame_bypass()` 是加速器跑失敗（或對齊邊界 reprime）時唯一正確的出口——它讓 framing
節奏繼續走，ring／CCM 不步進，遞迴 state **維持原樣**（回報失敗的 runtime 視為沒有寫入，所以 state 停在原處；寫了一半的 state 只有在非有限值時才偵測得到）。
但「這一框輸出什麼」每個模型不同：

| 模型 | bypass 輸出 | 理由 |
|---|---|---|
| Align-ULCNet | error 頻譜原樣穿過（identity，不經 AEC 輸出守衛） | 它的 stream 0 已經是 linear AEC 之後的殘差；守衛 hop 上可能大於 capture |
| DeepFilterNet2 | unit ERB mask + alpha 0（identity） | `erb_inv` 是 partition of unity（每個 bin 的 band 權重和為 1），unit band mask 展開就是 unit bin gain——**精確**而非近似 |
| DeepFilterNet3 | unit ERB 增益 + 原始低頻 bin 上一個 unit tap（identity） | 同上 |
| DeepVQE-S | **靜音（fail closed）** | 它的 stream 0 是**原始麥克風**，pass-through 會送出**完全未消除的回音**；`deepvqe_prepost_skip_policy_name()` 回 `"mute_fail_closed"` |
| GTCRN | 沒有 `frame_bypass`：`gtcrn_process_mask_validate()` 回 `-1` 時，應用端不呼叫 `gtcrn_model_output()`，直接送出輸入頻譜 | 由應用端執行 |
| RNNoise-ERB | 沒有 `frame_bypass`：`rnnoise_frame_finish()` 回 `-1` 時，應用端改用 unit 增益 | 由應用端執行 |

DeepVQE-S 的代價是 near-end 一框的缺口，上限就是一框；另一個選項的代價則是無上限的回音進遠端。
要 graceful degradation 的呼叫端得自己跑一個 canceller 並在 pipeline 層 cross-fade——
那不是這個類別的決定，它看不到有沒有那個 canceller。

### 6.4 Align-ULCNet：`ulcnet_prepost.h`

```c
int  ulcnet_prepost_config_defaults(UlcnetPrepostConfig *cfg, const UlcnetGrid *grid);
int  ulcnet_prepost_get_mem_size(const UlcnetPrepostConfig *cfg, UlcnetPrepostMemReq *req);
UlcnetPrepost *ulcnet_prepost_init(void *pool, size_t bytes, const UlcnetPrepostConfig *cfg);
UlcnetPrepost *ulcnet_prepost_init_ex(void *pool, size_t bytes, const UlcnetPrepostConfig *cfg,
                                      const UlcnetPrepostMemReq *expected);
UlcnetPrepost *ulcnet_prepost_create(const UlcnetPrepostConfig *cfg);
void ulcnet_prepost_destroy(UlcnetPrepost *p);
void ulcnet_prepost_reset(UlcnetPrepost *p);
uint32_t ulcnet_prepost_state_generation(const UlcnetPrepost *p);
int  ulcnet_prepost_hop_size(const UlcnetPrepost *p);
int  ulcnet_prepost_num_bins(const UlcnetPrepost *p);
const UlcnetModelIoDescriptor *ulcnet_prepost_descriptor(const UlcnetPrepost *p);

int  ulcnet_prepost_pre_process_freq(UlcnetPrepost *p,
                                     const float *err_re, const float *err_im,
                                     const float *far_re, const float *far_im);
int  ulcnet_prepost_frame_open(UlcnetPrepost *p, UlcnetModelIoInputs *inputs,
                                 UlcnetModelIoOutputs *outputs);
int  ulcnet_prepost_frame_finish(UlcnetPrepost *p);
int  ulcnet_prepost_frame_bypass(UlcnetPrepost *p);
int  ulcnet_prepost_post_process_freq(UlcnetPrepost *p, float *re, float *im);
```

- grid：`UlcnetGridSpec { sample_rate, n_fft, delay_depth }` 經 `ulcnet_grid_make()` 得 `UlcnetGrid`（另有 `hop`、`n_bins`）。D（`delay_depth`）的範圍 `ULCNET_MODEL_IO_MIN_D`=2 到 `ULCNET_MODEL_IO_MAX_D`=64 只是 sanity 檢查：D 必須等於匯出時的值，
  C 側抓不到不一致；pool 大小與 D 無關。
- `cfg` 欄位：`grid`、`model_layout_version`（0 = `ULCNET_MODEL_IO_LAYOUT_VERSION`，否則必須相等）、四組 state 登錄（`key_history`／`value_history`／`logit_history`／`h_gru`，各配元素數）。
- **error 與 far 必須是同一個輸入 hop**：類別內部不做任何 skew。上游 WOLA 造成的落差由呼叫端先對齊。
- `ulcnet_prepost_pre_process_freq()` 收的是未正規化 rfft（sqrt-Hann 的 n_fft 區塊，center 0，一 hop 一框；見 §6.8）——
  AEC 的 `AecResContext.error_spec` 已經正好是這個 framing 慣例（但它是交錯的 `Complex`，要拆成 re/im 兩個陣列再傳；error 一律取這個分析接縫，不取 `aec_get_output_base()` 的渲染底座）。
  far 是同一 hop 的 raw far 頻譜。
- 加速器邊界的 `UlcnetModelIoInputs`／`UlcnetModelIoOutputs` 定義在 `ulcnet_model_io.h`。
- 一個 binary 可以同時服務多個 grid（每個實例各帶一個 `UlcnetGrid`）。訓練過的只有 16000／512；48000／1024 只是 grid 通用性的測試，沒有 48 kHz 模型。

### 6.5 DeepVQE-S：`deepvqe_prepost.h`

生命週期與每-hop 函式和 6.4 逐一對應（前綴 `deepvqe_prepost_`，訊號引數為 `mic_re/mic_im/far_re/far_im`，長度 `grid.n_bins`），
`frame_open()` 用的是本檔自己的 `DeepVqePrepostInputs`／`DeepVqePrepostOutputs`。額外 API：

```c
const char *deepvqe_prepost_state_name(int state_id);       /* graph input 名 */
const char *deepvqe_prepost_state_name_out(int state_id);   /* graph output 名 */
const char *deepvqe_prepost_skip_policy_name(void);
int  deepvqe_prepost_descriptor_default(const DeepVqeGrid *grid, DeepVqePrepostDescriptor *descriptor);
int  deepvqe_prepost_descriptor_validate(const DeepVqeGrid *grid, const DeepVqePrepostDescriptor *descriptor);
const DeepVqePrepostDescriptor *deepvqe_prepost_descriptor(const DeepVqePrepost *p);
```

- grid：`DeepVqeGridSpec { sample_rate, n_fft, delay_depth }` 經 `deepvqe_grid_make()` 得 `DeepVqeGrid`。D 範圍 `DEEPVQE_PREPOST_MIN_D`=1 到 `DEEPVQE_PREPOST_MAX_D`=256，**沒有預設值**：
  應用端傳入模型匯出時的 D（出貨 32，16 kHz 上的半秒搜尋範圍）；D 是 export 期參數，圖 shape 與傳入的 `delay_depth` 必須相同，descriptor 只檢查範圍，抓不到不一致。pool 大小與 D 無關。
- 圖邊界：兩個 power-law 壓縮後的 planar 訊號 input（`mic`/`far`，`[1,2,1,BINS]`，BINS = `grid.n_bins`，先 real 平面、再 imaginary 平面；壓縮 `x·|x|^(DEEPVQE_COMPRESSION_EXP−1)` 由
  `frame_open()` 在 host 端做，圖自第一個 conv 起；`DeepVqePrepostInputs.mic`/`.far` 即這兩個 tensor，CCM 用的 raw mic 頻譜另存於類別內）＋
  **16 個顯式 state**，順序就是 `DeepVqeStateId` 列舉（= exporter 的 `input_names[2:]` 逐字），
  輸出是 packed CCM taps `[1,1,BINS,DEEPVQE_TIME_ORDER*DEEPVQE_FREQ_TAPS*2]`（16 kHz／512 為 `[1,1,257,18]`，末軸仍按 `[time][frequency][RI]` 排列；元素數 `deepvqe_prepost_taps_elements(grid)`）＋ 16 個 next state。
  按 index 綁的 adapter 必須用這個列舉，按名字綁的用 `_state_name()`。
- DeepVQE-S 回的是**每個 state 的完整下一個 window**（不是差量；shape 與輸入相同，全部 rank 4），runtime 把 `*_out` 就地綁定到輸入位址。state 的 shape 與元素數是匯出模型的性質：
  應用端自備一張以 `DeepVqeStateId` 為索引的表，登錄時用 `cfg.state[id]`／`cfg.state_elements[id]`。
  `_frame_finish()` 對每個 tap 與登錄的 state 整個緩衝做有限性檢查，遇非有限值時整組 state 歸零、`deepvqe_prepost_state_generation()` 加一。
- `descriptor_validate()` 拿 ONNX/JSON metadata 裡的 13 欄 `c_descriptor` 對 `grid` 的 ABI 逐欄比對（`sample_rate`／`fft_size` 必須等於 grid 的），
  只有 `delay_depth` 是 export-time 部署參數、僅做範圍檢查；`DEEPVQE_PREPOST_LAYOUT_VERSION` = 4。conv／score window 與 GRU 欄位是模型內部的，由函式庫給 0 且從不讀。
- grid 是執行期值：一個 binary 可同時服務多個 grid。訓練過的只有 16000／512；48000／1024 只是 grid 通用性的測試。

[索引](audio_libraries_c_api_zh_TW.md) ｜ [上一份](audio_libraries_c_api_zh_TW_01_aec_nr_audiolib.md) ｜ [下一份](audio_libraries_c_api_zh_TW_03_ai_prepost_pipeline.md) <!-- split-nav -->
