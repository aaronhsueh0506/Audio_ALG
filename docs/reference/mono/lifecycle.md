# Mono 生命週期與設定

## 3. Lifecycle

板端（caller-owned pool）的完整順序如下。每個函式庫各自查大小、各自 init，整合者負責加總與切塊。

| # | 步驟 | 呼叫 | 什麼時候呼叫 | 這一步可能怎麼失敗 |
|---|---|---|---|---|
| 1 | derive | `aec_resolve_signal_grid(sr, fft_size, &grid)`、`aec_config_from_level(&aec_cfg, level, sr)`，再設接縫欄位（範例的 `derive_lane()`，[§4.1](#41-aecconfig-的接縫欄位範例的-derive_lane)） | 每次 init 之前 | `aec_resolve_signal_grid()` 回 `AEC_ERR_ARG`：沒有這組 `(sample_rate, fft_size)` 的格點，`grid` 不被寫入（`fft_size = 0` 也一樣）；`aec_config_from_level()` 是 `void` |
| 2 | query | `aec_get_mem_size(&aec_cfg)`、`anr_get_mem_size(&nr_cfg)`、`ac_istft_get_mem_size(&synth_cfg)`，加上你自己的增益／頻譜暫存 | **每次 init 之前**都要重查，不可跨 build / backend / config 快取 | 回傳 `0`：config 無效（**注意：這幾個是 0 = 失敗，與 `aec_get_mem_requirements()`（4ch 與 mono 共用）的 `0` = 成功、`-1` = 失敗慣例不同**） |
| 3 | allocate | 你自己的配置器，取得各塊 `ALIGN16(...)` 相加且 16-byte 對齊的記憶體 | 拿到大小之後 | 配置失敗由你自己處理。pool **不需要**預先清零（poison pattern 也可以）；但必須在 handle 生命週期內獨佔、不被其他人寫入 |
| 4 | init | `aec_init(pool, aec_bytes, &aec_cfg)`、`anr_init(pool + ALIGN16(aec_bytes), nr_bytes, &nr_cfg)`、`ac_istft_init(...)`（視窗用 `aec_init` 之後的 `aec_synthesis_window(aec, &synth_cfg.window, &len)` 借來的表；`ac_istft_get_mem_size` 先用任一非 NULL 的替身視窗量大小）；每一塊的起點都是前一塊加 `ALIGN16(前一塊大小)` | 拿到 pool 之後，處理第一個 hop 之前 | 回傳 `NULL`：pool 未 16-byte 對齊、太小、config 無效、子模組 init 失敗。已建好的那幾個要由你 destroy |
| 5 | process | `aec_process_linear(aec, mic, ref)` → `aec_process_residual(aec, NULL)` → `aec_get_residual(aec, &v)`（先設 `v.struct_size = sizeof(v)`）→ 每 hop 串接（[§1.1](integration.md#11-構成每個函式庫各自初始化每個-hop-的串接由你寫)：`v.res_gain` 留在 `Aec` 裡、`anr_process_gain_with_residual_psd()`，降噪與增益都作用在 `v.base_spec` 上、`aec_apply_gain()`、`aec_apply_cng()`，兩者傳同一個降噪增益）→ `ac_istft_push(istft, spec, out)`；`aec_post_reset_serial(aec)` 與已處理的值不同時，在 push 之前 `ac_istft_reset(istft)`（Aec 自己的 comfort noise 種子已重設；不重置 AEC 與降噪器） | 每收到一個 hop 呼叫一次。`mic`/`ref`/`out` 都必須剛好 `grid.hop_size` 個 float | `aec_process_linear()`、`aec_process_residual()`、`aec_get_residual()`、`aec_apply_gain()`、`aec_apply_cng()` 回 `AEC_OK` 或 `AEC_ERR_ARG`／`AEC_ERR_MODE`／`AEC_ERR_SEQUENCE`（實例不變）；`v.res_gain` 為 NULL（`residual` 低於 `AEC_RESIDUAL_FULL`）、`anr_process_gain_with_residual_psd()` 回 `NULL`（降噪器拒絕該 hop）都是你要處理的拒絕；`ac_istft_push()` 的回傳值與輸出有限性（選用 `skn_all_finite_f32()`）要自己檢查。**不會**檢查 buffer 長度 —— 長度錯是你的責任 |
| 6 | recover | `aec_reset(aec)`、`anr_reset(nr)`、`ac_istft_reset(istft)`（`aec_reset()` 同時把 `Aec` 自己的 comfort-noise 產生器設回 `AEC_CNG_SEED`，42），並更新你已處理的 `aec_post_reset_serial()` 值 | 任何一步被拒（含 `AEC_ERR_DSP`：AEC 已自己 reset，NR 與合成仍要你 reset）、輸出非有限（若你有檢查）、換回聲路徑（換喇叭、重新 seat）或切換到不相關的另一條串流時 | 不會失敗。必須全部一起 reset：漏掉其中一個就不等於新建的實例（`mono/aec_nr_res/main.c` 的 `rig_reset()` 就是這幾件事；`main.c` 在某個 stage 拒絕該 hop 時呼叫它，輸出有限性檢查是 `example_board_adapter.c` 的 `run_hops()` 做的） |
| 7 | destroy | `anr_destroy(nr)`、`aec_destroy(aec)` | 不再處理任何 hop 之後、釋放 pool 之前 | 不會失敗。`aec_destroy` NULL 安全；對 `aec_init()` 的 pool 實例可重複呼叫，只放開實例內部的參照，**不**釋放你的 pool；對 `aec_create()` 的實例釋放它自己的那一塊，**只能呼叫一次**，之後指標作廢（不要再 `free()`）；`anr_destroy` 對 pool 實例立即返回，對 `anr_create()` 的 heap 實例才釋放記憶體 |
| 8 | release | 你自己的釋放器 | **一定要在第 7 步之後** | 順序顛倒 = use-after-free |

補充規則：

* **大小不可快取**。firmware image 若拿舊 build 算出的大小去餵新 library，就會在錯誤大小/佈局的 pool 上切記憶體。
  mono 範例走 `aec_init()`，沒有 descriptor 閘；`Aec` 另提供 `aec_get_mem_requirements()`／`aec_init_ex()`（`AecMemReq`，與 4ch array 同一個型別，
  見 AEC 手冊 `AEC/docs/reference/lifecycle.md` §C 記憶池描述子），可擋下舊 build 的 pool。不用它時，每次 build 都重查大小是你的責任；
  `aec_init()`、`anr_init()` 與 `ac_istft_init()` 只會擋下過小或未對齊的 pool。
* pool 生命週期內 sub-module 都是指進 pool 的裸指標，不是拷貝。pool 不能被搬動、
  不能被別人寫、不能和另一個 `Aec`/NR/FFT instance 共用。
* 單一 instance 不可被多個 thread 同時 `process`。多條串流請各自建立 instance。
* `AecResidualView` 的指標都借自 `Aec`，到下一次 `aec_process_linear()` 或 reset 為止有效（見 `aec.h`）；`P`（輸出頻譜）必須在同一個 hop 內形成並合成。

### 3.1 執行期唯讀存取

| 呼叫 | 用途 |
|---|---|
| `grid.hop_size` / `grid.n_freqs` / `grid.fft_size`（`aec_resolve_signal_grid()` 輸出的 `AecSignalGrid`） | 每次 process 要餵幾個 float、頻域 bin 數 |
| `anr_debug_status(nr, &st)` | 診斷：降噪器的 init 旗標、增益、噪聲底（唯讀，不改狀態） |
| `aec_debug_status(aec, &st)` | 診斷：delay、ERLE、收斂旗標（mono CLI 的 `--debug` 就是這個加 `anr_debug_status()`） |

### 3.1b 逐階段耗時（診斷用）

`aec_get_last_timing(aec, &t)` 回報 **AEC 上一個 hop** 的各階段牆鐘耗時（微秒），純診斷：

```c
AecStageTiming t;
aec_process_linear(aec, mic, ref);
aec_process_residual(aec, NULL);
aec_get_last_timing(aec, &t);
/* t.delay_us t.frontend_us t.linear_us t.steering_us t.res_us ← lib/aec 量 */
```

增益串接與合成的耗時是**範例程式自己**在呼叫前後讀時鐘量的（`mono/aec_nr_res/main.c` 的 `process_hop()`
與 `timing_now_us()`），不是任何函式庫的 API。

| 欄位 | 涵蓋範圍 |
|---|---|
| `delay_us` | **延遲估計與 ring 對齊**：matched-filter bank、其 duty-cycle 記帳、reference ring 寫入與延遲補償讀取。`EXTERNAL_ALIGNED` 下為 0，`FIXED` 下很小 |
| `frontend_us` | AEC 到主濾波器之前的**其餘**部分：mic HPF、飽和、render activity、`mu_scale`、mic-clip、RSA、shadow filter。**不含 `delay_us`**，兩者不重複計算 |
| `linear_us` | AEC 主自適應濾波器（`pbfdkf_process_with_far_cmag2`），含 far-end FFT |
| `steering_us` | 主濾波器之後、RES 之前的 stationarity 更新、coarse-error 發布、DT analyzer、EPV、shadow_rise 與失調估計（`aec.h` 的註解） |
| `res_us` | AEC 自己的 AEC3 post/RES 區塊 |

**本機實測：`delay_us` 是整個 hop 最大的一塊。** 主機（桌面 aarch64、NE10）上以預設（`-DAEC_STAGE_TIMING=1`）建置，
`aec_nr_pipeline_static --timing`、16 kHz/256、balanced、一段 10 s 的合成回音輸入、連跑兩次：`delay_us` 約佔呼叫牆鐘的
53.2–54.0 %，`frontend_us`（mic HPF/RSA/shadow filter）約 10.2–10.3 %。比例隨輸入與平台改變，目標平台上的比例待量測。
這條線直接對應 `delay_num_filters`（1..5）這個可調旋鈕。

**各階段不會加總成整通呼叫。** 餘數包含引數檢查、`aec_get_residual()` 與每 hop 結尾的收尾步驟，它們都不在上面五欄內。範例的 `--timing` 表只列 `delay_us`、`frontend_us`、`linear_us`、`res_us` 四列，
`steering_us` 落在 unattributed。要呈現完整分解的呼叫端必須把餘數明確列出來，不能讓部分看起來像全部。

**預設開啟。** AEC 內部 7 次 clock 讀取（桌面主機約 0.13 µs／hop）。`make PROFILE=0`（`-DAEC_STAGE_TIMING=0`）關閉後欄位讀 0。
呼叫端自己的顯示旗標（`--timing`）只決定印不印，**不會**開關 AEC 的量測。

**記錄本身不是條件編譯的，只有取樣是。** 欄位無論旗標開關都在 struct 裡，所以
`PROFILE=0` 與預設建置的 pool 大小與每個 offset 都相同，切換旗標不會移動任何 offset。

**目標平台沒有 `CLOCK_MONOTONIC` 時。** `clock_gettime` 是 POSIX 而非 C99。這種
平台用 `make EXTRA_CFLAGS='-DAEC_NOW_US=board_timer_us -include my_timer.h'`
換掉 AEC 的時鐘（或 `make PROFILE=0` 關掉量測）：巨集要是**純識別字**（Makefile 的 FP policy 不允許 `EXTRA_CFLAGS` 出現括號），
指向不收參數、回傳 `uint32_t` 微秒的函式。替代時鐘必須單調，否則無號減法會產生接近 32-bit 全距的
荒謬值。每個元件各有自己的覆寫點（`lib/aec` 的 `AEC_NOW_US`，4ch 陣列的 `aec_array_get_timing()` 也用它），刻意不共用——一個元件的時鐘屬於該元件自己的建置契約。
指向常數函式是合法用法，所有欄位讀 0，`--timing` 的報表無法分辨這種情況與 `PROFILE=0`。

兩支 CLI 都有 `--timing`，在結束時印出整段 run 的每 hop 平均與佔比（含未歸屬餘數）。

### 3.2 執行期強度控制

兩條強度軸都可以在**運轉中**改，不需要重建、不動 pool：

```c
int aec_set_level(Aec* a, AecLevel level);
int anr_set_level(AnrDenoiser* nr, AnrLevel level);   /* nr = 你自己 anr_init() 回來的那一個 */
```

```c
/* 使用者把回音抑制轉到 aggressive、降噪轉到 mild */
if (aec_set_level(aec, AEC_LEVEL_AGGRESSIVE) != 0)   { /* 引數不合法 */ }
if (anr_set_level(nr, ANR_LEVEL_MILD) != 0)                  { /* 同上 */ }

aec_process_linear(aec, mic, ref);    /* 下一個 hop 起生效 */
```

| 呼叫 | 改什麼 | 回 `-1` 的情況（`-1` 時**什麼都不寫**） |
|---|---|---|
| `aec_set_level()` | `Aec` 的殘留回音地板（level 改變 far-active 地板 -20/-28/-38 dB；aggressive 另把 DT 地板設成 -30 dB；NO_FLOOR 關閉 `min_gain_split_floor_enabled`；mono 的 RES 增益 `v.res_gain` 就是這個 `Aec` 自己算的，所以 mono 的 RES 強度就是這個 setter；4ch array 的唯一增益級同樣由它設定） | `a` 為 NULL；`level` 超出 enum |
| `anr_set_level()` | 降噪器的強度 level（`q`、`xi_min_db`、`g_min_db`、`noise_over_subtraction`） | `nr` 為 NULL（含你沒有建降噪器）；`level` 超出 enum |

C 稱 level，Python 稱 preset（NR 為 strength）：**C level == Python preset**，值名與整數相同。

far-active 地板固定在 100 ms 內以 dB 為單位線性走過去（AEC 內部常數，呼叫端不能調），落點與「用該 level 從頭建一個
新實例」完全相同。mild ↔ aggressive 是 18 dB 落差、而地板是硬性 clamp，所以 ramp 內建。
ramp 進行中再呼叫一次，會從當前的 live 值重新起走。

兩個 setter 都**不是重啟**：AEC 的濾波器與延遲鎖定、NR 追蹤中的噪聲底與增益平滑歷史
全部繼續跑。要重啟請用 [§3](#3-lifecycle) 第 6 步的整組 reset。**在兩個 hop 之間呼叫、與
process 序列化；非 thread-safe。**

> 範例的降噪器以 `anr_config_for_level_grid()` 建立、沒有覆寫任何欄位（最小值追蹤視窗 `L` 用函式庫預設），所以 `anr_set_level()`
> 直接換強度。若你在 level 組態之上改過欄位，`anr_set_level()` 會把它們蓋回 level 值：
> 那種實例請自己組好完整 target 再呼叫 `anr_reconfigure()`。

**A/B 量測時該預期什麼。** far-active 地板只在 **far-active 且非 double-talk** 的
hop 上生效：double-talk 期間套用的是 DT 地板（mild/balanced 共用 -20 dB，aggressive 為 -30 dB）；far-active latch 觸發之前套用的是 far-silent 地板。同一個 `v.res_gain`
還決定注入的 comfort noise 量（振幅正比於 `sqrt(1 − G_res²)`，再乘該 bin 的 NR gain（下限 −10 dB），見
`aec_apply_cng()`，範例的 `mono_form()` 在 CNG 步驟呼叫它——地板壓得越深、CNG 反而越多）。
因此**整段錄音的平均值移動
幅度會小於 dB 落差所暗示的量**，而且一個只量 echo／degradation 的 A/B 會把 CNG 的
變化錯記到別的機制頭上。請在 echo 對齊或 degradation 對齊的條件下比較，並實際試聽。

---

## 4. Config 完整參考

mono 的設定分成兩份：lib/aec 的 `AecConfig`（格點與接縫欄位由範例的 `derive_lane()` 填，其餘欄位的預設來自
`aec_config_from_level()`）與範例程式自己的後級選項（RES gain 用不用、NR 開不開與強度、CNG、`use_r2`、合成），
後者**不是**任何函式庫的 config struct，是 `mono/aec_nr_res/main.c` 的 `PostPlan` 這類程式內部欄位，決定要不要呼叫、
要不要建對應的函式庫物件。驗證發生在 `aec_resolve_signal_grid()`（回 `AEC_ERR_ARG`）與
`aec_get_mem_size()`／`anr_get_mem_size()`／`ac_istft_get_mem_size()`（回 `0`）。

### 4.1 `AecConfig` 的接縫欄位（範例的 `derive_lane()`）

`derive_lane(sample_rate, fft_size, level, &grid, &aec_cfg)`（`mono/aec_nr_res/main.c`）依序做：
`aec_resolve_signal_grid(sample_rate, fft_size ? fft_size : aec_default_fft_size(sample_rate), &grid)`（回非 `AEC_OK` 即回報失敗）、
`aec_config_from_level(&aec_cfg, level, sample_rate)`，然後設下表的欄位。`aec_get_mem_size()` 與 `aec_init()` 會用 libaec 自己的驗證器
再檢查整份 `AecConfig`（回 `0`／`NULL`）。`aec_config_from_level()` 對 8000／16000／48000 以外的取樣率回傳 `fft_size` 為 `0` 的不可用設定，`aec_get_mem_size()` 回 `0`、`aec_init()` 與 `aec_create()` 回 `NULL`，函式本身不會溢位。`AecConfig` 只有 12 個公開欄位（`sample_rate`、`fft_size`、`n_mics`、`level`、`residual`、`delay_mode`、`fixed_delay_samples`、`delay_num_filters`、`enable_cng`、`capture_proxy_channel`、`delay_backward_quarantine_enabled`／`_s`）；其餘調校值（濾波器長度、shadow、HPF、飽和處理、各門檻）由函式庫依 `(sample_rate, fft_size, n_mics, level)` 在內部決定，呼叫端不能設。

| 欄位 | 型別 | 範例的值 | 合法值（驗證器實際檢查的） | 你該怎麼設 |
|---|---|---|---|---|
| `sample_rate` | `int` | 你的取樣率 | `8000` / `16000` / `48000`，其他一律拒絕（例如 `44100`；`aec_is_valid_sample_rate()` 可先查） | 設成你音訊實際的取樣率。這個值同時決定預設 grid |
| `fft_size` | `int` | `aec_default_fft_size(sample_rate)`（CLI `--fft-size 0` 時） | 8 kHz 只接受 `256`；16 kHz 接受 `256` 或 `512`；48 kHz 只接受 `1024`；`0` 與其他組合拒絕（`aec_resolve_signal_grid()` 回 `AEC_ERR_ARG`） | 除非你有明確理由要 16 kHz 的 512 grid，否則取 rate 預設 |
| `delay_mode` | `AecDelayMode` | `AEC_DELAY_MATCHED` | `AEC_DELAY_MATCHED` / `AEC_DELAY_FIXED` / `AEC_DELAY_EXTERNAL_ALIGNED` | 語意同 `lib/aec`；記憶體差異見 [§4.5a](#45a-delay_mode-對-16000預設-grid-的影響matched-n5-為-baseline) |
| `delay_num_filters` | `int` | `AEC_DELAY_NUM_FILTERS_MAX`（5） | 一律 `[1, AEC_DELAY_NUM_FILTERS_MAX]`；`FIXED` 與 `EXTERNAL_ALIGNED` 要求維持 `AEC_DELAY_NUM_FILTERS_MAX` | matched-filter bank 大小，決定 bulk delay 搜尋範圍；init 參數，不是 runtime setter |
| `fixed_delay_samples` | `int` | `-1` | 僅 `FIXED` 有效（原生取樣點數 ≥ 0，上限為該取樣率下的 120 秒）；其他模式須為 `-1` | 已知固定 delay 時使用 |
| `residual` | `AecResidualMode` | `AEC_RESIDUAL_FULL` | `AEC_RESIDUAL_OFF`／`STATE`／`ESTIMATE`／`FULL` | 接縫：`FULL` 時 lane 不自己輸出時域音訊（除非呼叫 `aec_process()`），`aec_get_residual()` 交出 `error_spec`／`base_spec`／`res_gain`／`r2`／`comfort_noise`，增益由你的每 hop 串接經 `aec_apply_gain()`／`aec_apply_cng()` 套用 |
| `level` | `AecLevel` | `AEC_LEVEL_BALANCED`（= 1，由 `aec_config_from_level(&cfg, level, sr)` 寫入） | `AEC_LEVEL_MILD`(0) / `AEC_LEVEL_BALANCED`(1) / `AEC_LEVEL_AGGRESSIVE`(2) / `AEC_LEVEL_NO_FLOOR`(3)；`aec_config_from_level()` 對列舉以外的整數存成 balanced；手填或全零的 `AecConfig` 是 level 0（`MILD`），手填的值不在 0..3 時 `aec_get_mem_size()`／`aec_init()`／`aec_create()` 拒絕（執行期換強度用 `aec_set_level()`，它對 enum 外回 `-1`，並把新 level 存回實例） | 近端保留優先 → `MILD`；回聲抑制優先 → `AGGRESSIVE`；不確定就留預設 |

`residual` 保持預設 `AEC_RESIDUAL_FULL`：`v.res_gain` 才有值；只要 `error_spec`／`formed_hop` 與狀態旗標的 lane（例如 Align-ULCNet 範例）用 `AEC_RESIDUAL_STATE`，
`ESTIMATE` 另給 `r2` 與 `comfort_noise`；4ch array 的 lane 用哪一級由陣列依它的 `residual` 衍生，不是 mono 範例的事。

### 4.2 後級選項（範例程式的欄位與 CLI 旗標）

| 選項 | 範例的預設 | 它決定什麼 |
|---|---|---|
| RES gain | 用（`v.res_gain`，即 AEC 自己的殘留 gain，由 `aec_apply_gain()` 取用） | `--no-res`：`G_res` 視為全 1 的 unity 向量（`aec_apply_gain_spec()`），comfort noise 因為沒有被 RES 挖掉的 bin 而不填。強度用 `aec_set_level()`（[§3.2](#32-執行期強度控制)） |
| NR | 用 | `--no-nr`：不建 `AnrDenoiser`（state 不進 pool），總增益只剩 `G_res`，`anr_set_level()` 沒有對象 |
| NR 強度 | `ANR_LEVEL_BALANCED`（= 2） | `--nr-level 0..3`（0 mild／1 moderate／2 balanced／3 aggressive，其他文字 exit 2；舊旗標 `--nr-preset` 為別名）：交給 `anr_config_for_level_grid(sr, fft_size, level)`，四級都可用；最小值追蹤視窗 `L` 為函式庫預設，不覆寫 |
| `use_r2` | 用 | `--legacy-amin`：`use_r2 = 0`，`mono_nr_gain()` 不把 R² 摺進 noise floor（noise-only NR）。只用於比對，新整合請保持用 R² |
| CNG | 開，與 `aec_cfg.enable_cng`（AEC level 自己的值）取 AND | `--no-cng`：不呼叫 `aec_apply_cng()`。位準再乘該 bin 的 NR gain、下限 −10 dB（`aec.h` 的 `AEC_CNG_NR_GAIN_FLOOR`，由 `aec_apply_cng()` 內部套用；NR 看不到 comfort noise，在此補套）；沒有 RES gain 時也不填 |
| 合成 | 用 | 範例是 `ac_istft_push()` 之後直接輸出；DFN2 經 rate bridge 時由 bridge 自己合成、不建 `AcIstft`；模型串流直接跑在主機格點時仍由應用建 `AcIstft`（見 `mono/aec_dfn_res`） |

`--aec-only` 不建任何後級，lane 以 `AEC_RESIDUAL_STATE` 建立（殘留階段照跑）、輸出取 `AecLinearView.linear_hop`（[§1.2](integration.md#12---aec-only)）。

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
一律用 `aec_resolve_signal_grid()` 輸出的 `grid.hop_size` / `grid.n_freqs` 查。

### 4.4 哪些欄位會改變記憶體用量

| 欄位 | 影響 pool 大小嗎 |
|---|---|
| `sample_rate`、`fft_size` | 會（改 grid、改 AEC filter 長度） |
| 濾波器長度 | 不是欄位：由取樣率決定（8 kHz 416、16 kHz 832、48 kHz 3072 samples），隨 `sample_rate`／`fft_size` 一起改 pool |
| `delay_mode` | 會（`MATCHED` 建 matched-filter estimator + 大 ring；`FIXED` 只建剛好夠用的小 ring；`EXTERNAL_ALIGNED` 兩者都不建，見 [§4.5a](#45a-delay_mode-對-16000預設-grid-的影響matched-n5-為-baseline)） |
| `delay_num_filters` | 僅 `MATCHED` 有效：每少一個省 11,264 B（8／16 kHz；48 kHz 為 23,552 B），因為單支的參考環隨 bank 大小成長 |
| `fixed_delay_samples` | 僅 `FIXED` 有效：ring 大小 = `ALIGN16((fixed_delay_samples+hop)×4)` B |
| `aec_level`、`nr_level` | 實測不變 |
| 要不要建降噪器（`--no-nr`） | 會（不建就省下整份 MMSE-LSA state） |
| 要不要建合成 | 會（不建就省下 `AcIstft`） |
| RES gain、CNG、`use_r2` | 不變（RES 是 `Aec` 內的 gain，不另配記憶體） |

### 4.4b 診斷用的分項

沒有「後級總大小」的單一查詢：你的 pool 大小就是各個 `mem_size` 的回傳值各自 `ALIGN16` 後相加，再加上你自己的
增益／頻譜暫存。`aec_nr_pipeline_static --print-mem-size` 印的表格（`AEC`、`Synthesis`、`NR (MMSE-LSA)`、
`Gain/spectrum buffers`、`Total`）就是這個加總，並且真的在那塊 pool 上跑一次每個物件的 init。
這只是給人看的表格；**真正要配置多少，永遠以各 `mem_size` 的回傳值為準**，加上各塊的對齊。

### 4.5 實測記憶體（僅供量級參考，務必自己重查）

以下是**本次 checkout、`BACKEND=ne10`、`pipelines/Makefile` 預設選項**下，以
`aec_nr_pipeline_static --print-mem-size` 量到的值（delay 列用 `aec_get_mem_size()` 加上同一組後級大小直接呼叫）。
換編譯選項、更新函式庫版本 都會變，請自己重查。

增益／頻譜暫存合計於「暫存」欄（兩個 `n_freqs` float 向量與一個 `n_freqs` 複數向量，各自 `ALIGN16`）。

| Config | 總計 | AEC（`aec_get_mem_size`） | 合成 `AcIstft` | `nr_bytes` | 暫存 |
|---|---:|---:|---:|---:|---:|
| 8000，預設 | 249,664 | 210,864 | 10,320 | 26,384 | 2,096 |
| 16000，預設（256/128） | 294,592 | 239,232 | 10,320 | 42,944 | 2,096 |
| 16000，`fft_size=512` | 426,752 | 351,088 | 19,792 | 51,728 | 4,144 |
| 48000，預設（1024/512） | 956,240 | 774,000 | 38,736 | 135,264 | 8,240 |
| 16000，`--no-nr`（不建降噪器） | 251,648 | 239,232 | 10,320 | 0 | 2,096 |
| 16000，`--aec-only`（STATE lane，不建後級） | 239,232 | 239,232 | – | – | – |

總計 = `ALIGN16(AEC)` + `ALIGN16(nr_bytes)` + `ALIGN16(合成)` + 暫存（各塊對齊後相加）。合成 `AcIstft` 借用 AEC 的合成窗（`aec_synthesis_window()`），所以不帶自己的窗表（比各自帶窗少一個 `fft_size` 長的 float 向量）。

#### 4.5a delay_mode 對 16000/預設 grid 的影響（`MATCHED` n=5 為 baseline）

| `delay_mode` | AEC 加後級合計 | 相對 `MATCHED n=5` |
|---|---:|---:|
| `MATCHED` n=5（預設） | 294,592 | — |
| `MATCHED` n=1 | 249,536 | −45,056 |
| `FIXED`，`fixed_delay_samples=1600`（100 ms） | 239,152 | −55,440 |
| `EXTERNAL_ALIGNED` | 232,240 | −62,352 |

這四列的差額全部落在 AEC 那一塊（`delay_mode`/`delay_num_filters` 不影響
後級），數字與 `lib/aec` 自己的 `aec_get_mem_size()` 一致
（見 `AEC/docs/reference/lifecycle.md`「記憶池大小」）。**沒有 CLI 旗標**能測試
非預設 delay mode——上表用直接呼叫
`aec_get_mem_size()` 加上同一組後級大小（55,360 B）量得。

---

## 5. 錯誤處理

這個 codebase 混用了數種回傳慣例。下表是**每個 function 實際的**慣例。

### 5.1 錯誤語意總表

| Function | 成功 | 失敗 | 備註 |
|---|---|---|---|
| `aec_resolve_signal_grid()` | `AEC_OK`（0），`grid` 填妥 | `AEC_ERR_ARG`（-1） | 沒有這組 `(sample_rate, fft_size)`（含 `fft_size = 0`）；`grid` 不被寫入。與 `aec_get_mem_size()` 的 `0` = 失敗相反方向，判斷式別混用 |
| `aec_get_mem_size()` | pool bytes | `0`（config 無效） | 與 `aec_get_mem_requirements()`（0 = 成功）相反 |
| `anr_get_mem_size()` / `ac_istft_get_mem_size()` | pool bytes | `0`（config 無效） | 同上 |
| `aec_init()` / `anr_init()` / `ac_istft_init()` | 非 NULL handle | `NULL` | pool 未對齊、太小、config 無效；失敗後 pool 仍可重新用於正確的 init |
| `aec_process_linear()` / `aec_process_residual()` / `aec_process()` | `AEC_OK`（0） | `AEC_ERR_ARG`／`AEC_ERR_MODE`／`AEC_ERR_SEQUENCE`；`AEC_ERR_DSP` | `AEC_ERR_DSP`：有限但極大的輸入使該 hop 的結果溢位（門檻隨場景而變；mic 的極大 sample 在帶它的那個 hop 被拒，ref 的在經對齊環到達濾波器時才被拒，晚數個 hop），AEC 已自己 reset（`aec.h` 的 “Refused hop”）；這個 hop 輸出零並照第 6 步 reset 其餘 stage（`rig_reset()` 再呼叫一次 `aec_reset()` 無害），下一個 hop 照常送。其餘三種實例不變；`aec_process_residual()` 必須緊接同一個 hop 的 `aec_process_linear()`，`residual == AEC_RESIDUAL_OFF` 回 `AEC_ERR_MODE`；其餘異常由 `v.res_gain` 為 NULL 與 `anr_process_gain_with_residual_psd()` 回傳值看出 |
| `aec_get_residual()` / `aec_get_linear()` / `aec_post_reset_serial()` | `AEC_OK`／計數值 | `AEC_ERR_*`／– | `aec_get_residual()`：`out` 為 NULL 或 `struct_size` 小於 `sizeof(AecResidualView)` 回 `AEC_ERR_ARG`、`residual == AEC_RESIDUAL_OFF` 回 `AEC_ERR_MODE`、殘留階段尚未跑完回 `AEC_ERR_SEQUENCE`（視為拒絕該 hop）；`aec_post_reset_serial()` 回 `uint32_t` 硬重置計數（NULL 回 0），值變了的處理見上方 process 列 |
| `aec_apply_gain()` / `aec_apply_cng()` | `AEC_OK` | `AEC_ERR_*` | 需要 `AEC_RESIDUAL_FULL` 與已處理完的 hop（`AEC_ERR_MODE`／`AEC_ERR_SEQUENCE`）；`aec_apply_gain()` 的 `out_spec` 不可與輸入重疊 |
| `anr_process_gain()` / `anr_process_gain_with_residual_psd()` | 降噪器擁有的 `n_freqs` 個增益（借用到下一次 process 呼叫前） | `NULL` | NULL 引數或降噪器拒絕該 hop |
| `ac_istft_push()` | 寫出的樣本數（`hop`；`AC_OLA_ENV_NORM` 的第一個 frame 為 `0`） | – | 不檢查輸出是否有限，要另外查 |
| `skn_all_finite_f32()` | `1` | `0` | 選用：你在採用輸出之前要做的檢查（`example_board_adapter.c` 的 `run_hops()` 就這樣檢查） |
| `aec_reset()` / `anr_reset()` / `ac_istft_reset()` | `void` | 不會失敗 | 要一起呼叫；`aec_reset()` 也把 `Aec` 的 comfort-noise 產生器設回 `AEC_CNG_SEED`（42）並使 `aec_post_reset_serial()` 加一 |
| `aec_destroy()` / `anr_destroy()` | `void` | 不會失敗 | NULL 安全 |
| `aec_set_level()` / `anr_set_level()` | `0` | `-1` | `-1` 時什麼都不寫（[§3.2](#32-執行期強度控制)） |

> **注意慣例不一致**：`aec_get_mem_size()` / `anr_get_mem_size()` / `ac_istft_get_mem_size()` /
> `fft_get_mem_size()` 是 **回傳 0 代表失敗**；`aec_get_mem_requirements()` 是
> **0 = 成功 / −1 = 失敗**。判斷式別混用。

### 5.2 init 的拒絕清單

`aec_init()`：`mem == NULL`、`mem` 未 16-byte 對齊、`mem_size` 小於 `aec_get_mem_size()`、config 無效、子模組 init 失敗 → `NULL`。
`anr_init()`、`ac_istft_init()`：同樣的 NULL／未對齊／過小／config 無效規則（出界 enum、不合法格點）→ `NULL`。
這些 init 都**不會**動到 pool 以外的記憶體，被拒之後同一塊 pool 仍可用在正確的 init
（`example_board_adapter.c` 的 `run_negative_demonstrations()` 展示並檢查這一點）。

### 5.3 `destroy()` 的語意

`aec_destroy()` 與 `anr_destroy()` 都**不釋放**你的 pool；`aec_destroy()` NULL 安全，對 pool 實例可重複呼叫（只放開實例內部的參照），對 `aec_create()` 的實例則釋放那一塊、只能呼叫一次、之後指標作廢；`anr_destroy()` 對 pool 實例立即返回，只有 `anr_create()` 的 heap 實例才釋放記憶體。`AcIstft` 沒有 destroy。
`posix_memalign` 之類由你配置的 pool，由你在 destroy 之後釋放。

### 5.4 診斷輸出去向

* mono 範例的每 hop 串接（`mono_form()` 與 `mono_nr_gain()`）不輸出任何東西。lib/aec／lib/nr／audiolib 的 init 期診斷寫到 **stderr**，
  以 `NO_STDIO=1` 建置時編成 no-op，**回傳值語意完全不變**。
* 每 hop 的 `aec_process_linear()`、`aec_process_residual()`、`anr_process_gain_with_residual_psd()`、`ac_istft_push()` **在任何 build 下都不會輸出任何東西**。

[返回本主題](README.md)
