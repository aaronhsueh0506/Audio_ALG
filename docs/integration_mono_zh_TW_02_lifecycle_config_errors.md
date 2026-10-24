導覽：[索引](integration_mono_zh_TW.md) ｜ [上一部分](integration_mono_zh_TW_01_overview_link_quickstart.md) <!-- split-nav -->
## 3. Lifecycle

板端（caller-owned pool）的完整順序如下。每個函式庫各自查大小、各自 init，整合者負責加總與切塊。

| # | 步驟 | 呼叫 | 什麼時候呼叫 | 這一步可能怎麼失敗 |
|---|---|---|---|---|
| 1 | derive | `se_derive(&req, &rules, &grid, &aec_cfg)` | 每次 init 之前 | 回傳 `-1`：任何一條規則被違反（[§4.1](#41-serequest--se_derive)）；輸出內容未定義 |
| 2 | query | `aec_get_mem_size(&aec_cfg)`、`anr_get_mem_size(&nr_cfg)`、`ac_istft_get_mem_size(&synth_cfg)`，加上你自己的增益／頻譜暫存 | **每次 init 之前**都要重查，不可跨 build / backend / config 快取 | 回傳 `0`：config 無效（**注意：這幾個是 0 = 失敗，與 4ch 單元的 `-1` 慣例不同**） |
| 3 | allocate | 你自己的配置器，取得各塊 `ALIGN16(...)` 相加且 16-byte 對齊的記憶體 | 拿到大小之後 | 配置失敗由你自己處理。pool **不需要**預先清零（poison pattern 也可以）；但必須在 handle 生命週期內獨佔、不被其他人寫入 |
| 4 | init | `aec_init(pool, aec_bytes, &aec_cfg)`、`anr_init(pool + ALIGN16(aec_bytes), nr_bytes, &nr_cfg)`、`ac_istft_init(...)`；每一塊的起點都是前一塊加 `ALIGN16(前一塊大小)` | 拿到 pool 之後，處理第一個 hop 之前 | 回傳 `NULL`：pool 未 16-byte 對齊、太小、config 無效、子模組 init 失敗。已建好的那幾個要由你 destroy |
| 5 | process | `aec_process_context(aec, mic, ref)` → `se_frame_from_aec(aec, &f)` → 每 hop 串接（[§1.1](integration_mono_zh_TW_01_overview_link_quickstart.md#11-構成每個函式庫各自初始化每個-hop-的串接由你寫)：`g_res_ctx`、`se_anr_stage_gain()`、`sk_min_f32()`、`sk_capply_gain_f32()`、`aec_cng_fill()`）→ `ac_istft_push(istft, spec, out)` | 每收到一個 hop 呼叫一次。`mic`/`ref`/`out` 都必須剛好 `grid.hop_size` 個 float | `aec_process_context()` 回傳 `void`；`se_frame_from_aec()` 回 `-1`（NULL 引數或 lane 沒有 seam）；`g_res_ctx` 為 NULL、`se_anr_stage_gain()` 回 `NULL`（NULL 引數、或降噪器拒絕該 hop）；`ac_istft_push()` 的回傳值與輸出有限性（`se_hop_is_finite()`）要自己檢查。**不會**檢查 buffer 長度 —— 長度錯是你的責任 |
| 6 | recover | `aec_reset(aec)`、`anr_reset(nr)`、`ac_istft_reset(istft)`，並把你的 comfort-noise RNG 設回 `SE_CNG_RNG_SEED` | 任何一步被拒、輸出非有限、換回聲路徑（換喇叭、重新 seat）或切換到不相關的另一條串流時 | 不會失敗。必須全部一起 reset：漏掉其中一個就不等於新建的實例（`main.c`／`static_main.c` 的 `rig_reset()` 就是這四件事） |
| 7 | destroy | `anr_destroy(nr)`、`aec_destroy(aec)` | 不再處理任何 hop 之後、釋放 pool 之前 | 不會失敗。`aec_destroy` NULL 安全、可重複，只放開實例內部的參照，**不**釋放你的 pool；`anr_destroy` 對 pool 實例立即返回，對 `anr_create()` 的 heap 實例才釋放記憶體 |
| 8 | release | 你自己的釋放器 | **一定要在第 7 步之後** | 順序顛倒 = use-after-free |

補充規則：

* **大小不可快取**。firmware image 若拿舊 build 算出的大小去餵新 library，就會在錯誤大小/佈局的 pool 上切記憶體。
  mono 沒有 descriptor 閘（那是 4ch 單元的 `AudioPipeline4Ch` 才有的契約，見
  [`integration_4ch_spatial_zh_TW.md`](integration_4ch_spatial_zh_TW.md)），所以每次 build 都重查大小是你的責任；
  `aec_init()`、`anr_init()` 與 `ac_istft_init()` 只會擋下過小或未對齊的 pool。
* pool 生命週期內 sub-module 都是指進 pool 的裸指標，不是拷貝。pool 不能被搬動、
  不能被別人寫、不能和另一個 `Aec`/NR/FFT instance 共用。
* 單一 instance 不可被多個 thread 同時 `process`。多條串流請各自建立 instance。
* `SeSpecFrame` 的指標都借自 `Aec`，到下一次 `aec_process_context()` 或 reset 為止有效；`P`（輸出頻譜）必須在同一個 hop 內形成並合成。

### 3.1 執行期唯讀存取

| 呼叫 | 用途 |
|---|---|
| `grid.hop_size` / `grid.n_freqs` / `grid.fft_size`（`se_derive()` 輸出） | 每次 process 要餵幾個 float、頻域 bin 數 |
| `anr_debug_status(nr, &st)` | 診斷：降噪器的 init 旗標、增益、噪聲底（唯讀，不改狀態） |
| `aec_debug_status(aec, &st)` | 診斷：delay、ERLE、收斂旗標（mono CLI 的 `--debug` 就是這個加 `anr_debug_status()`） |

### 3.1b 逐階段耗時（診斷用）

`aec_get_last_timing(aec, &t)` 回報 **AEC 上一個 hop** 的各階段牆鐘耗時（微秒），純診斷：

```c
AecStageTiming t;
aec_process_context(aec, mic, ref);
aec_get_last_timing(aec, &t);
/* t.delay_us t.frontend_us t.linear_us t.res_us ← lib/aec 量 */
```

增益串接與合成的耗時是**範例程式自己**在呼叫前後讀時鐘量的（`mono_aec_nr_res/main.c` 的 `process_hop()`
與 `timing_now_us()`），不是任何函式庫的 API。

| 欄位 | 涵蓋範圍 |
|---|---|
| `delay_us` | **延遲估計與 ring 對齊**：matched-filter bank、其 duty-cycle 記帳、reference ring 寫入與延遲補償讀取。`EXTERNAL_ALIGNED` 下為 0，`FIXED` 下很小 |
| `frontend_us` | AEC 到主濾波器之前的**其餘**部分：mic HPF、飽和、render activity、`mu_scale`、mic-clip、RSA、shadow filter。**不含 `delay_us`**，兩者不重複計算 |
| `linear_us` | AEC 主自適應濾波器（`pbfdkf_process_with_far_cmag2`），含 far-end FFT |
| `res_us` | AEC 自己的 AEC3 post/RES 區塊 |

**本機實測：`delay_us` 是整個 hop 最大的一塊。** 主機（macOS）上以 `make PROFILE=1` 建置，
`aec_nr_pipeline --timing`、16 kHz/256、balanced、一段 20 s 的合成雜訊輸入、跑兩次：`delay_us` 約佔呼叫牆鐘的
54.6–54.8 %，`frontend_us`（mic HPF/RSA/shadow filter）約 10.1 %。比例隨輸入與平台改變，目標平台上的比例待量測。
這條線直接對應 `delay_num_filters`（1..5）這個可調旋鈕。

**各階段不會加總成整通呼叫。** 餘數包含引數檢查、`aec_get_res_context()`，以及
AEC 內部（因此從這層看不見）的 stationarity 更新、`e2_coarse`/ERL 發布、DT
analyzer、EPV、shadow_rise、失調估計、功率 EMA 與收斂偵測。要呈現完整分解的
呼叫端必須把餘數明確列出來，不能讓部分看起來像全部。

**預設關閉。** AEC 內部 7 次 clock 讀取，release build 一次都不做，欄位讀 0。
`make PROFILE=1` 會開啟（`-DAEC_STAGE_TIMING=1`）。呼叫端自己的顯示旗標（`--timing`）**不會**打開 AEC 的量測。

**記錄本身不是條件編譯的，只有取樣是。** 欄位無論旗標開關都在 struct 裡，所以
profile build 與 release build 的 pool 大小與每個 offset 都相同，開旗標不會移動任何 offset。

**目標平台沒有 `CLOCK_MONOTONIC` 時。** `clock_gettime` 是 POSIX 而非 C99。這種
平台用 `make PROFILE=1 EXTRA_CFLAGS='-DAEC_NOW_US=board_timer_us -include my_timer.h'`
換掉 AEC 的時鐘：巨集要是**純識別字**（Makefile 的 FP policy 不允許 `EXTRA_CFLAGS` 出現括號），
指向不收參數、回傳 `uint32_t` 微秒的函式。替代時鐘必須單調，否則無號減法會產生接近 32-bit 全距的
荒謬值。每個元件各有自己的覆寫點（`lib/aec`、四路核心），刻意不共用——一個元件的時鐘屬於該元件自己的建置契約。
開著旗標但指向常數函式是合法用法，所有欄位讀 0，`--timing` 的報表無法分辨這種情況與「根本沒開旗標」。

兩支 CLI 都有 `--timing`，在結束時印出整段 run 的每 hop 平均與佔比（含未歸屬餘數）。

### 3.2 執行期強度控制

兩條強度軸都可以在**運轉中**改，不需要重建、不動 pool：

```c
int aec_set_preset(Aec* a, AecPreset preset, float ramp_ms);
int anr_set_mode(AnrDenoiser* nr, AnrMode mode);   /* nr = 你自己 anr_init() 回來的那一個 */
```

```c
/* 使用者把回音抑制轉到 aggressive、降噪轉到 mild */
if (aec_set_preset(aec, AEC_PRESET_AGGRESSIVE, 100.0f) != 0)   { /* 引數不合法 */ }
if (anr_set_mode(nr, ANR_MODE_MILD) != 0)                      { /* 同上 */ }

aec_process_context(aec, mic, ref);   /* 下一個 hop 起生效 */
```

| 呼叫 | 改什麼 | 回 `-1` 的情況（`-1` 時**什麼都不寫**） |
|---|---|---|
| `aec_set_preset()` | `Aec` 的殘留回音地板（preset 改變 far-active 地板 -20/-28/-38 dB；aggressive 另把 DT 地板設成 -30 dB；NO_FLOOR 關閉 `min_gain_split_floor_enabled`；mono 的 RES 增益 `g_res_ctx` 就是這個 lane 自己算的，所以 mono 的 RES 強度就是這個 setter） | `a` 為 NULL；`preset` 超出 enum；`ramp_ms` 非有限值或超出 `[0, 60000]` |
| `anr_set_mode()` | 降噪器的強度 preset（`q`、`xi_min_db`、`g_min_db`、`noise_over_subtraction`） | `nr` 為 NULL（含你沒有建降噪器）；`mode` 超出 enum |

`ramp_ms == 0` 代表下一個 hop 就套用，**不是錯誤**，落點與「用該 preset 從頭建一個
新實例」完全相同；`> 0` 則以 dB 為單位線性走過去，上限 60 秒。mild ↔ aggressive 是
18 dB 落差、而地板是硬性 clamp，所以互動式旋鈕應該給一個 ramp（100 ms 是合理起點）。
ramp 進行中再呼叫一次，會從當前的 live 值重新起走。

兩個 setter 都**不是重啟**：AEC 的濾波器與延遲鎖定、NR 追蹤中的噪聲底與增益平滑歷史
全部繼續跑。要重啟請用 [§3](#3-lifecycle) 第 6 步的整組 reset。**在兩個 hop 之間呼叫、與
process 序列化；非 thread-safe。**

> 範例的降噪器以 `anr_config_for_mode_grid()` 建立、沒有覆寫任何欄位（最小值追蹤視窗 `L` 用函式庫預設），所以 `anr_set_mode()`
> 直接換強度。若你在 preset 之上改過欄位，`anr_set_mode()` 會把它們蓋回 preset 值：
> 那種實例請自己組好完整 target 再呼叫 `anr_reconfigure()`。

**A/B 量測時該預期什麼。** far-active 地板只在 **far-active 且非 double-talk** 的
hop 上生效：double-talk 期間套用的是 DT 地板（mild/balanced 共用 -20 dB，aggressive 為 -30 dB）；far-active latch 觸發之前套用的是 far-silent 地板。同一個 `ctx.res_gain`
還決定注入的 comfort noise 量（振幅正比於 `sqrt(1 − G_res²)`，再乘該 bin 的 NR gain（下限 −10 dB），見
`aec_cng_fill()`，範例的 `mono_form()` 在 CNG 步驟呼叫它——地板壓得越深、CNG 反而越多）。
因此**整段錄音的平均值移動
幅度會小於 dB 落差所暗示的量**，而且一個只量 echo／degradation 的 A/B 會把 CNG 的
變化錯記到別的機制頭上。請在 echo 對齊或 degradation 對齊的條件下比較，並實際試聽。

---

## 4. Config 完整參考

mono 的設定分成兩份：`SeRequest`（交給 `se_derive()`，決定格點與 `AecConfig`）與
範例程式自己的後級選項（RES gain 用不用、NR 開不開與強度、CNG、`use_r2`、合成），後者**不是**任何函式庫的
config struct，是 `main.c`／`static_main.c` 的 `PostOptions`／`PostPlan` 這類程式內部欄位，決定要不要呼叫、
要不要建對應的函式庫物件。驗證發生在 `se_derive()` 本身（回 `-1`）與
`aec_get_mem_size()`／`anr_get_mem_size()`／`ac_istft_get_mem_size()`（回 `0`）。

### 4.1 `SeRequest` → `se_derive()`

| 欄位 | 型別 | 範例的值 | 合法值（驗證器實際檢查的） | 你該怎麼設 |
|---|---|---|---|---|
| `sample_rate` | `int` | 你的取樣率 | `8000`（需 `rules.allow_8k`）/ `16000` / `48000`。其他一律拒絕（例如 `44100`） | 設成你音訊實際的取樣率。這個值同時決定預設 grid |
| `fft_size` | `int` | `0` | `0` = 依 rate 取預設。明確指定時：8 kHz 只接受 `256`；16 kHz 接受 `256` 或 `512`；48 kHz 只接受 `1024`。其他組合拒絕 | 除非你有明確理由要 16 kHz 的 512 grid，否則留 `0` |
| `filter_length` | `int` | `0` | `0` = 依 preset／rate 取預設；否則 `[1, SE_FILTER_LENGTH_MAX(4096)]` | PBFDKF taps，init 後不可變；沒有明確需求就留 `0` |
| `delay_mode` | `AecDelayMode` | `AEC_DELAY_MATCHED` | `AEC_DELAY_MATCHED` / `AEC_DELAY_FIXED` / `AEC_DELAY_EXTERNAL_ALIGNED` | 語意同 `lib/aec`；記憶體差異見 [§4.5a](#45a-delay_mode-對-16000預設-grid-的影響matched-n5-為-baseline) |
| `delay_num_filters` | `int` | `5` | 僅 `MATCHED` 有效：`[1, 5]`；其他模式要求維持 `5`（`rules.check_delay = 1` 時檢查） | matched-filter bank 大小，決定 bulk delay 搜尋範圍；init 參數，不是 runtime setter |
| `fixed_delay_samples` | `int` | `-1` | 僅 `FIXED` 有效（原生取樣點數，上限為該取樣率下的兩分鐘）；其他模式須為 `-1`（`rules.check_delay = 1` 時檢查） | 已知固定 delay 時使用 |
| `aec_preset` | `AecPreset` | `AEC_PRESET_BALANCED`（= 1） | `AEC_PRESET_MILD`(0) / `AEC_PRESET_BALANCED`(1) / `AEC_PRESET_AGGRESSIVE`(2) / `AEC_PRESET_NO_FLOOR`(3)。列舉以外的整數會被拒絕（**不會**默默 fallback 成 balanced） | 近端保留優先 → `MILD`；回聲抑制優先 → `AGGRESSIVE`；不確定就留預設 |

`SeRules` 是你的應用自己的規則：`allow_8k`、`check_delay`（是否拒絕與 `delay_mode` 矛盾的 delay 設定；
mono 範例沒開）、`spatial_context`（mono 用 `SE_CONTEXT_PRESET`：lane 保留 preset 自己的 context 層級，
所以 `g_res_ctx` 有值）。`se_derive()` 輸出的 `AecConfig` 已設好 `enable_res = 0`、`return_res_context = 1`：
lane 不自己輸出時域音訊，增益由你的每 hop 串接套用。

### 4.2 後級選項（範例程式的欄位與 CLI 旗標）

| 選項 | 範例的預設 | 它決定什麼 |
|---|---|---|
| RES gain | 用（`frame.g_res_ctx`，即 AEC 自己的殘留 gain） | `--no-res`：`G_res` 視為全 1 的 unity 向量，comfort noise 因為沒有被 RES 挖掉的 bin 而不填。強度用 `aec_set_preset()`（[§3.2](#32-執行期強度控制)），沒有獨立的 `AecRes`、也沒有 `res_preset` |
| NR | 用 | `--no-nr`：不建 `AnrDenoiser`（state 不進 pool），總增益只剩 `G_res`，`anr_set_mode()` 沒有對象 |
| NR 強度 | `ANR_MODE_BALANCED`（= 2） | `--nr-preset mild\|moderate\|balanced\|aggressive`：交給 `anr_config_for_mode_grid(sr, fft_size, mode)`，四級都可用；最小值追蹤視窗 `L` 為函式庫預設，不覆寫 |
| `use_r2` | 用 | `--legacy-amin`：`se_anr_stage_gain(..., use_r2 = 0)`，noise-only NR（noise floor 不摺入 R²）。只用於比對，新整合請保持用 R² |
| CNG | 開，與 `aec_cfg.enable_cng`（AEC preset 自己的值）取 AND | `--no-cng`：不呼叫 `aec_cng_fill()`。位準再乘該 bin 的 NR gain、下限 −10 dB（`SE_CNG_NR_GAIN_FLOOR`；NR 看不到 comfort noise，在此補套）；沒有 RES gain 時也不填 |
| 合成 | 用 | 範例是 `ac_istft_push()` 之後直接輸出；DFN2 之類後接的 stage 自己合成時就不建 `AcIstft`（見 `mono_aec_dfn_res`） |

`--aec-only` 不建任何後級、用 `aec_process()`（[§1.2](integration_mono_zh_TW_01_overview_link_quickstart.md#12---aec-only)）。

### 4.3 Grid（由 `sample_rate` + `fft_size` 唯一決定）

`hop = fft_size / 2`、`frame = fft_size`、`n_freqs = fft_size / 2 + 1`。
合法組合只有這四組：

| `sample_rate` | `fft_size` | `hop` | `frame` | `n_freqs` | 備註 |
|---:|---:|---:|---:|---:|---|
| 8000 | 256 | 128 | 256 | 129 | `fft_size = 0` 的 rate 預設 |
| 16000 | 256 | 128 | 256 | 129 | `fft_size = 0` 的 rate 預設 |
| 16000 | 512 | 256 | 512 | 257 | 需明確指定 |
| 48000 | 1024 | 512 | 1024 | 513 | `fft_size = 0` 的 rate 預設 |

**不要把 hop 寫死。** 16 kHz 有兩組合法 grid，寫死任一組對另一組就是錯的。
一律用 `se_derive()` 輸出的 `grid.hop_size` / `grid.n_freqs` 查。

### 4.4 哪些欄位會改變記憶體用量

| 欄位 | 影響 pool 大小嗎 |
|---|---|
| `sample_rate`、`fft_size` | 會（改 grid、改 AEC filter 長度） |
| `filter_length` | 會（非 0 時覆寫 AEC filter 長度／n_partitions） |
| `delay_mode` | 會（`MATCHED` 建 matched-filter estimator + 大 ring；`FIXED` 只建剛好夠用的小 ring；`EXTERNAL_ALIGNED` 兩者都不建，見 [§4.5a](#45a-delay_mode-對-16000預設-grid-的影響matched-n5-為-baseline)） |
| `delay_num_filters` | 僅 `MATCHED` 有效：每少一個固定省 5,728 B（四格點皆同，與 `lib/aec` 一致） |
| `fixed_delay_samples` | 僅 `FIXED` 有效：ring 大小 = `ALIGN16((fixed_delay_samples+hop)×4)` B |
| `aec_preset`、`nr_mode` | 實測不變 |
| 要不要建降噪器（`--no-nr`） | 會（不建就省下整份 MMSE-LSA state） |
| 要不要建合成 | 會（不建就省下 `AcIstft`） |
| RES gain、CNG、`use_r2` | 不變（RES 是 `Aec` 內的 gain，不另配記憶體） |

### 4.4b 診斷用的分項

沒有「後級總大小」的單一查詢：你的 pool 大小就是各個 `mem_size` 的回傳值各自 `ALIGN16` 後相加，再加上你自己的
增益／頻譜暫存。`static_main.c --print-mem-size` 印的表格（`AEC`、`Synthesis`、`NR (MMSE-LSA)`、
`Gain/spectrum buffers`、`Total`）就是這個加總，並且真的在那塊 pool 上跑一次每個物件的 init。
這只是給人看的表格；**真正要配置多少，永遠以各 `mem_size` 的回傳值為準**，加上各塊的對齊。

### 4.5 實測記憶體（僅供量級參考，務必自己重查）

以下是**本次 checkout、`BACKEND=ne10`、`pipelines/Makefile` 預設選項**下，以
`aec_nr_pipeline_static --print-mem-size` 量到的值（delay 列用 `aec_get_mem_size()` 加上同一組後級大小直接呼叫）。
換編譯選項、更新 submodule 都會變，請自己重查。

增益／頻譜暫存合計於「暫存」欄（三個 `n_freqs` float 向量與一個 `n_freqs` 複數向量，各自 `ALIGN16`）。

| Config | 總計 | AEC（`aec_get_mem_size`） | 合成 `AcIstft` | `nr_bytes` | 暫存 |
|---|---:|---:|---:|---:|---:|
| 8000，預設 | 310,160 | 267,408 | 11,344 | 28,784 | 2,624 |
| 16000，預設（256/128） | 423,728 | 364,416 | 11,344 | 45,344 | 2,624 |
| 16000，`fft_size=512` | 575,056 | 491,600 | 21,840 | 56,432 | 5,184 |
| 48000，預設（1024/512） | 1,354,256 | 1,156,544 | 42,832 | 144,576 | 10,304 |
| 16000，`--no-nr`（不建降噪器） | 378,384 | 364,416 | 11,344 | 0 | 2,624 |
| 16000，`--aec-only`（不建後級） | 364,416 | 364,416 | – | – | – |

總計 = `ALIGN16(AEC)` + `ALIGN16(nr_bytes)` + `ALIGN16(合成)` + 暫存（各塊對齊後相加）。

#### 4.5a delay_mode 對 16000/預設 grid 的影響（`MATCHED` n=5 為 baseline）

| `delay_mode` | AEC 加後級合計 | 相對 `MATCHED n=5` |
|---|---:|---:|
| `MATCHED` n=5（預設） | 423,728 | — |
| `MATCHED` n=1 | 400,816 | −22,912 |
| `FIXED`，`fixed_delay_samples=1600`（100 ms） | 265,632 | −158,096 |
| `EXTERNAL_ALIGNED` | 258,720 | −165,008 |

這四列的差額全部落在 AEC 那一塊（`delay_mode`/`delay_num_filters` 不影響
後級），數字與 `lib/aec` 自己的 `aec_get_mem_size()` 一致
（見 `lib/aec/docs/c_user_manual_zh_TW_01_lifecycle.md` §4）。**沒有 CLI 旗標**能測試
非預設 delay mode——上表用直接呼叫
`aec_get_mem_size()` 加上同一組後級大小（59,312 B）量得。

---

## 5. 錯誤處理

這個 codebase 混用了數種回傳慣例。下表是**每個 function 實際的**慣例。

### 5.1 錯誤語意總表

| Function | 成功 | 失敗 | 備註 |
|---|---|---|---|
| `se_derive()` | `0`，輸出填妥 | `-1` | 輸出未定義；無任何 log |
| `aec_get_mem_size()` | pool bytes | `0`（config 無效） | 與 4ch 的 `get_mem_requirements()`（0 = 成功）相反 |
| `anr_get_mem_size()` / `ac_istft_get_mem_size()` | pool bytes | `0`（config 無效） | 同上 |
| `aec_init()` / `anr_init()` / `ac_istft_init()` | 非 NULL handle | `NULL` | pool 未對齊、太小、config 無效；失敗後 pool 仍可重新用於正確的 init |
| `aec_process_context()` / `aec_process()` | `void` | – | 不回傳錯誤；異常由後面的 `se_frame_from_aec()`／`se_anr_stage_gain()` 回傳值看出 |
| `se_frame_from_aec()` | `0` | `-1` | NULL 引數，或 lane 沒有 seam（`enable_res=0`、`return_res_context=1`） |
| `se_anr_stage_gain()` | 降噪器擁有的 `n_freqs` 個增益 | `NULL` | NULL 引數、`frame->e` 為 NULL，或降噪器拒絕該 hop |
| `ac_istft_push()` | 寫出的樣本數（`hop`；`AC_OLA_ENV_NORM` 的第一個 frame 為 `0`） | – | 不檢查輸出是否有限，要另外查 |
| `se_hop_is_finite()` | `1` | `0` | 你在採用輸出之前要做的檢查 |
| `aec_reset()` / `anr_reset()` / `ac_istft_reset()` | `void` | 不會失敗 | 要一起呼叫，並把 comfort-noise RNG 設回 `SE_CNG_RNG_SEED` |
| `aec_destroy()` / `anr_destroy()` | `void` | 不會失敗 | NULL 安全 |
| `aec_set_preset()` / `anr_set_mode()` | `0` | `-1` | `-1` 時什麼都不寫（[§3.2](#32-執行期強度控制)） |

> **注意慣例不一致**：`aec_get_mem_size()` / `anr_get_mem_size()` / `ac_istft_get_mem_size()` /
> `fft_get_mem_size()` 是 **回傳 0 代表失敗**；4ch 單元的 `audio_pipeline_4ch_get_mem_requirements()` 是
> **0 = 成功 / −1 = 失敗**。判斷式別混用。

### 5.2 init 的拒絕清單

`aec_init()`：`mem == NULL`、`mem` 未 16-byte 對齊、`mem_size` 小於 `aec_get_mem_size()`、config 無效、子模組 init 失敗 → `NULL`。
`anr_init()`、`ac_istft_init()`：同樣的 NULL／未對齊／過小／config 無效規則（出界 enum、不合法格點）→ `NULL`。
這些 init 都**不會**動到 pool 以外的記憶體，被拒之後同一塊 pool 仍可用在正確的 init
（`example_board_adapter.c` 的 `run_negative_demonstrations()` 展示並檢查這一點）。

### 5.3 `destroy()` 的語意

`aec_destroy()` 與 `anr_destroy()` 都**不釋放**你的 pool；`aec_destroy()` NULL 安全、可重複呼叫（只放開實例內部的參照）；`anr_destroy()` 對 pool 實例立即返回，只有 `anr_create()` 的 heap 實例才釋放記憶體。`AcIstft` 沒有 destroy。
`posix_memalign` 之類由你配置的 pool，由你在 destroy 之後釋放。

### 5.4 診斷輸出去向

* mono 範例的每 hop 串接（`se_anr_stage.h` 與 `mono_form()`）不輸出任何東西。lib/aec／lib/nr／audiolib 的 init 期診斷寫到 **stderr**，
  以 `NO_STDIO=1` 建置時編成 no-op，**回傳值語意完全不變**。
* 每 hop 的 `aec_process_context()`、`se_anr_stage_gain()`、`ac_istft_push()` **在任何 build 下都不會輸出任何東西**。

---

導覽：[索引](integration_mono_zh_TW.md) ｜ [上一部分](integration_mono_zh_TW_01_overview_link_quickstart.md) <!-- split-nav -->
