導覽：[索引](integration_4ch_core_zh_TW.md) ｜ [上一部分](integration_4ch_core_zh_TW_02_lifecycle.md) ｜ [下一部分](integration_4ch_core_zh_TW_04_errors_checklist_versions.md) <!-- split-nav -->
## 4. Config 完整參考

`McabConfig` 只放 **flow 的參數**（格點、共用 delay 的對齊方式、是否要 fuse）。lane 的參數（level、`filter_length`…）
是應用自己的 `AecConfig`，不在 `McabConfig` 裡，flow 只用 `mcab_apply_lane_contract()` 蓋上契約欄位（見 [§4.1](#41-逐欄位)）。
取得預設值的唯一正確方式：

```c
McabConfig cfg = mcab_default_config(16000);
/* 之後只覆寫你真正要改的欄位 */
```

`mcab_default_config()` **不做驗證**，只是填值。
驗證發生在 `mcab_get_mem_requirements()` / `mcab_init()` / `mcab_init_ex()` / `mcab_get_mem_breakdown()` /
`mcab_lane_config()` / `mcab_apply_lane_contract()` / `mcab_delay_estimator_config()` 共用的同一個 reject-first 閘。

### 4.1 逐欄位

| 欄位 | 型別 | `mcab_default_config()` 給的值 | 合法值（驗證器實際檢查的） | 你該怎麼設 |
|---|---|---|---|---|
| `sample_rate` | `int` | 你傳給 `mcab_default_config()` 的值 | **只有 `16000` 或 `48000`**。`8000` 會被拒絕（這點和單聲道 pipeline 不同） | 設成你音訊實際的取樣率 |
| `fft_size` | `int` | `0` | `0` = 依 rate 取預設（16 kHz → 256，48 kHz → 1024）。明確指定時：16 kHz 接受 `256` 或 `512`；48 kHz 只接受 `1024` | 除非有明確理由要 16 kHz 的 512 grid，否則留 `0` |
| `delay_mode` | `AecDelayMode` | `AEC_DELAY_MATCHED` | `MATCHED` / `FIXED` / `EXTERNAL_ALIGNED` | 四條 lane 共用同一個決策。`MATCHED` 需要共用 estimator（`McabLibs.delay`）與對齊 ring；`FIXED` 只用固定延遲 ring，不給 estimator；`EXTERNAL_ALIGNED` 由呼叫端給已對齊的 far，不要 estimator 也不建 ring（記憶體差異見 4.6a） |
| `delay_num_filters` | `int` | `5`（`DA_NUM_FILTERS`） | 僅 `MATCHED`：`[1, 5]`；其餘模式須為預設 | matched-filter bank 大小，每少一個省 5,728 B（estimator 的 pool，不在 mcab 的 pool 內） |
| `fixed_delay_samples` | `int` | `-1` | 僅 `FIXED`：`[0, 120 × sample_rate]`，原生取樣點數；其餘模式須為 `-1` | 決定共用 delay ring 大小 |
| `capture_proxy_channel` | `int` | `0` | `[0, 3]`。其他值拒絕 | 共用 delay matcher 要用哪一支麥克風當輸入。挑訊噪比最穩定、離喇叭關係最典型的那一支 |
| `max_delay_ms` | `float` | `1024.0f` | 必須是有限值，且落在 `[0.0, 4096.0]` | 僅 `MATCHED` 使用：共用 reference delay line 的容量上限（毫秒）。實際的 ring 大小 = `ceil(max_delay_ms * sample_rate / 1000) + 2 * hop + 1` 個 sample。設得比你系統真實最大 mic↔ref 延遲**略大**即可；設太大只是白白吃記憶體 |
| `delay_backward_quarantine_enabled` | `int`（bool） | `0` | `0` 或 `1` | 僅 `MATCHED`。開啟後，比現行 accepted delay **更早**的共用估計會先被隔離（`capture_proxy_channel` 那條 lane 在目前對齊上仍明顯在消除回音時），到期才採用；往後跳與首次取得對齊不受影響。產品組態維持 `0` |
| `delay_backward_quarantine_s` | `float` | `1.0` | 有限值，`[0.0, 3600.0]` | 隔離窗長（秒），`init` 時換算成 hop 數（最小 1）；enable 為 0 時無作用但仍被驗證 |
| `delay_est_period_s` | `float` | `0.0` | 有限值，`[0.0, 3600.0]` | 共用 matched-filter 的 duty-cycle 週期：`0.0` = 每 hop 全速分析；正值時估計 solid 且不變 0.3 s（`DUTY_HOLD_S`，固定）後，改為每 round(`delay_est_period_s`/hop 時長) 除以 5（整數除法，最小 1）個 hop 分析一次，省算力但可能鎖到不同的 lag。同一個值也由 `mcab_apply_lane_contract()` 蓋進每條 lane 的 `delay_est_period_s` |
| `enable_post` | `int`（bool） | `1` | 只接受 `0` 或 `1` | `1` = mcab 的 pool carve fuse 緩衝，可呼叫 `mcab_fuse()`；`0` = front-only（只有 `mcab_process_front()`，`mcab_fuse()` 回 `MCAB_INVALID_ARGUMENT`），給從不 fuse 的應用（例如 ULCNet）。見 4.6b |

**lane 的 `AecConfig`（應用的決定）**：`mcab_apply_lane_contract(&cfg, &lane)` 只蓋下面這些契約欄位，其餘（level 的數值、`filter_length` …）照你給的，重複呼叫結果相同：

| `AecConfig` 欄位 | 契約值 |
|---|---|
| `fft_size` | flow 的格點（[§4.2](#42-grid由-sample_rate--fft_size-唯一決定)） |
| `delay_mode`、`delay_num_filters`、`fixed_delay_samples`、`enable_delay_est` | `AEC_DELAY_EXTERNAL_ALIGNED`、`DA_NUM_FILTERS`、`-1`、`0`（lane 不做自己的延遲估計，吃 flow 已對齊的 far） |
| `enable_res`、`return_res_context` | `0`、`1`（回傳 linear context，不算自己的 gain） |
| `spatial_linear_context` | `enable_post = 1` 為 `AEC_CONTEXT_NO_GAIN`，`0` 為 `AEC_CONTEXT_LINEAR_ONLY` |
| `delay_est_period_s` | `cfg.delay_est_period_s` |

回 `-1` 且 lane 不變：`cfg` 無效，或 lane 是為別的取樣率建的。`mcab_lane_config(&cfg, &out)` 是預設 lane：
`aec_config_from_level(AEC_LEVEL_BALANCED, cfg.sample_rate)` 再蓋契約。`mcab_init_ex()` 驗證不了這些欄位的來源
（`Aec` 沒有 config getter），所以 lane 的契約欄位只能靠這個函式蓋，不要手寫。

`McabConfig` **沒有** RES、NR、comfort noise 或 level 的設定欄位：`aec_level`、`nr_level`、`enable_res`、`enable_nr`、
`enable_cng`、`legacy_amin` 都不在其中（lane 一律只算線性殘留）。
這些屬於你的後級串接（例如 `mcab_external_static.c` 的 `--no-res`、`--nr-level`、`--no-cng`、`--legacy-amin` 旗標對應的選項；
它的位置參數 `[aec-level]` 選的是後級 `AecRes` 的 level，lane 本身固定 balanced）：
RES 用 `aec_config_from_level(&res_cfg, level, sample_rate)` 配上 `mcab_fft_size()` 解析出的 `fft_size` 建 `AecRes`，NR 用 `anr_config_for_level_grid()`。

`McabLibs`（`mcab_init_ex()` 的第四個參數）：`lanes[4]` 是四個不同的 `Aec*`，`delay` 是 MATCHED 時的 `DelayAec*`、其餘模式必須是 `NULL`；
struct 在 init 時複製，裡面的物件是**借用**，壽命必須涵蓋 `Mcab`。
`McabDelayEstimatorConfig`（`mcab_delay_estimator_config()` 填）：`required`（只在 MATCHED 為 1）、`sample_rate`、`hop_size`、`num_filters`，
就是 `delay_aec_get_mem_size()` 與 `delay_aec_init()` 要的三個參數。

### 4.2 Grid（由 `sample_rate` + `fft_size` 唯一決定）

`hop = fft_size / 2`、`n_freqs = fft_size / 2 + 1`。合法組合只有三組：

| `sample_rate` | `fft_size` | `hop` | `n_freqs` | 備註 |
|---:|---:|---:|---:|---|
| 16000 | 256 | 128 | 129 | `fft_size = 0` 的 rate 預設 |
| 16000 | 512 | 256 | 257 | 需明確指定 |
| 48000 | 1024 | 512 | 513 | `fft_size = 0` 的 rate 預設 |

**不要把 hop / n_freqs 寫死。** 用 `mcab_hop_size()` / `mcab_n_freqs()` 查。


### 4.3 Buffer 形狀（每次 `mcab_process_front` / `mcab_fuse`）

| Buffer | 誰擁有 | 形狀 | 說明 |
|---|---|---|---|
| `microphones_interleaved` | 你 | `float[hop * 4]`，`[sample * 4 + channel]` | 唯讀，只在該次呼叫期間被讀取。所有元素必須是有限值 |
| `ref` | 你 | `float[hop]` | 唯讀。所有元素必須是有限值 |
| `weights` | 你 | `Complex[4 * n_freqs]`，**channel-major**：`[channel * n_freqs + bin]` | 唯讀 |
| `beamformed_error` | 你（可為 NULL） | `Complex[n_freqs]` | 與 `weights` 同一次 beamformer 呼叫產生的 mono 誤差頻譜；NULL 則由權重重建。供應時 `frame.e` 別名這份陣列，它必須活過 frame |
| `out` | 你 | `float[hop]` | 由你的 `ac_istft_push()` 寫入 |
| `front.formed_hop[ch]` | lane（Aec） | `const float[hop]` | 見 §4.4 |
| `front.linear_spectra[ch]` | lane（Aec） | `const Complex[n_freqs]` | 見 §4.4 |

**權重約定**：`output[bin] = Σ_channel weights[channel][bin] * input[channel][bin]`，
**不做共軛**。多幀 / 時域型 GSC 請把它「這一幀的有效頻率響應」填進來。

**一條在每個 bin 權重都是 0 的 lane 不會投票。** `mcab_fuse()` 除了投影頻譜，還會把四條
lane 的 `filter_converged` / `usable_linear` / `dt_indicator` / `saturation_level` 縮成純量
（`frame.all_converged`／`usable_linear`／`max_dt`／`max_sat`）
去操縱那個共用的 post 級抑制器（其中 `dt_indicator` 決定它套哪一個地板）。一條在每個 bin 權重都是 0 的 lane 對
融合後的頻譜貢獻恰好為零，所以也不參與這些縮減——縮減從中性值起算（`all_converged`、`usable_linear` 為 1，`max_dt`、`max_sat` 為 0）。
四條 lane 的 far 功率一致性檢查不受影響，對每條 lane 都照做。整組
全零權重會先被 `-1` 擋掉（見 [§5.3](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#53-mcab_fuse-的完整拒絕清單依實際檢查順序)），所以至少有一條 lane 會投票；真正在 beamform 的
權重每條 lane 都有非零 bin，這條規則不會跳過任何東西。

外部 beamformer**不需要**再回傳一份 mono hop —— 合成是在你自己的 `ac_istft_push()` 內
以那組權重加權後的頻譜完成的。

### 4.4 `McabFront` 與 `McabFrame` 各欄位

`McabFront`（`mcab_process_front()` 填）：

| 欄位 | 型別 | 內容 |
|---|---|---|
| `delay` | `McabDelayState` | 共用 matcher 這一幀的狀態，見下 |
| `hop_size` | `int` | 這一幀的 hop |
| `n_channels` | `int` | 恆為 `MCAB_CHANNELS`（4） |
| `n_freqs` | `int` | 這一幀的 bin 數 |
| `linear_spectra[ch]` | `const Complex*` | 四路 linear AEC 的**頻譜**，就是 `mcab_fuse()` 稍後會用的同一份。讓外部 beamformer 直接沿用既有的分析轉換，不必重做一次 STFT。**這是別名**，直接指向各 lane 自己的 buffer |
| `formed_hop[ch]` | `const float*` | 各 lane 與 `linear_spectra[ch]` 同一次選擇的**時域** hop（`hop_size` 個 sample）。做自己的 sqrt-Hann 分析的時域 beamformer 要吃這個。**這也是借用的**，沒有 interleaved 拷貝 |
| `aligned_ref` | `const float*` | 這一 hop 實際餵給四條 lane 的對齊後 far（`hop_size` 個 sample）：ring 已存夠樣本、能供應 `delay.delay_samples` 這個偏移之後，是你的 far 位移該值的結果；在那之前是原樣的 far hop（整個 hop 一起切換）；給需要「與線性濾波器吃到同一份 far」的外部 NN 後級用 |

`McabFrame`（`mcab_fuse()` 填，定義在 `mcab.h`）：

| 欄位 | 內容 |
|---|---|
| `n_freqs`, `hop` | 這一幀的 bin 數與 hop |
| `e` | beamformed 誤差頻譜（pre-RES；RES 與 NR 據以估計，也是增益套用的對象） |
| `y` | 投影後的近端頻譜 |
| `x` | lane 0 的 far 頻譜（四條 lane 吃同一份 far；混合 warm/soft realign 的 hop 略過逐 bin 比對，`x` 仍取 lane 0） |
| `r2` | 投影後的殘餘回音 PSD（int16² 功率尺度；`mcab_sample_anr_gain()` 餵降噪器前乘以 `MCAB_SAMPLE_PSD_SCALE_INV` = 1/32768²） |
| `cn` | 投影後的 comfort-noise PSD（int16² 功率尺度） |
| `far_hop` | 對齊後的 far（同 `aligned_ref`） |
| `formed_hop` | 恆為 NULL（每條 lane 各有一份，見 `McabFront.formed_hop`） |
| `far_power` | lane 0 的 far 功率 |
| `max_dt`, `max_sat`, `all_converged`, `usable_linear` | 對權重有非零響應的 lane 取的 double-talk／飽和／收斂／linear 可用縮減 |
| `clock_drift`, `delay_changed`, `delay_locked` | 共用 aligner 的旗標（`clock_drift` 只在 MATCHED 讀 estimator，其餘為 0） |
| `g_res_ctx` | 恆為 NULL（四條 lane 都不算自己的增益；RES 增益由你的 `AecRes` 經 `mcab_sample_res_input()` 與 `aec_res_process()` 算出） |

`McabDelayState` 欄位：

| 欄位 | 型別 | 意義 |
|---|---|---|
| `delay_samples` | `int` | 目前採用的延遲（sample）；只隨 `changed` 移動 |
| `confidence` | `float` | 估計信心值 |
| `solid` | `int` | 已有可用的 accepted 對齊（MATCHED：直到 reset 都保持；FIXED：ring 填滿後；EXTERNAL_ALIGNED：恆為 1） |
| `changed` | `int` | 非 0 表示這一幀開始一個新的可用對齊世代（MATCHED：首次取得（`mcab_reset()` 之後也算）、鎖定中被 admission 接受的移動 —— 移動超過 32 個樣本，且在候選提出後的下 2 個 hop 內（候選壽命 3 個 hop，含提出的那一個）再被提出一次、與候選相差小於 16 個樣本）—— **你外部的 STFT / OLA 歷史必須跟著清掉**（flow 自己對四條 lane 做 realign，不重置也不清合成 OLA） |
| `estimator_calls` | `uint64_t` | 累計呼叫次數 |
| `estimator_updates` | `int` | 累計更新次數 |

**front 視圖與 frame 的生命週期（重要）**：
`McabFront` 的所有指標與 `McabFrame` 的所有指標，**只在下列任一事件發生之前有效**：
下一次 `mcab_process_front()`、`mcab_reset()`、`mcab_destroy()`。
所有 stage 對它們都是**唯讀**。同一個 front 可以 fuse 多次，但每次 fuse 都會覆寫上一次 frame 的內容。

這是 **API 約定，不是記憶體安全保證**。
`reset()` 不保證那些記憶體被清掉或覆寫 —— lane 的舊頻譜可能就原地留著。
也就是說：你違反這條規則時，程式**不會**壞給你看，只會安靜地讀到未定義內容。
不要用「讀到垃圾」當作偵測手段。

### 4.5 哪些欄位會改變記憶體用量

mcab 的 `req.bytes` 只涵蓋 mcab 自己的 pool；lane 與 estimator 各有自己的 `get_mem_size`，應用端總量是三者之和。

| 欄位 | 影響 mcab 的 `req.bytes` 嗎 | 影響應用端其他物件嗎 |
|---|---|---|
| `sample_rate` | 會（改 grid） | 會（lane 與 estimator 都隨格點與預設 filter 長度改變） |
| `fft_size` | 會（改 grid） | 會 |
| `delay_mode` | 會（`MATCHED` 與 `FIXED` 才有 ring；`EXTERNAL_ALIGNED` 沒有） | 僅 `MATCHED` 需要共用 estimator（三態差異見 4.6a） |
| `delay_num_filters` | 不變 | 僅 `MATCHED` 有效，每少一個固定省 5,728 B（estimator 的 pool） |
| `fixed_delay_samples` | 會（僅 `FIXED`：ring = `fixed_delay_samples + hop` 個 float） | 不變 |
| `max_delay_ms` | 會（僅 `MATCHED` 用於 sizing delay ring） | 不變 |
| `enable_post` | 會（`0` 時不 carve fuse 緩衝，見 4.6b；ULCNet 範例用這個） | lane 的 `spatial_linear_context` 隨之改變；總量差值就是 fuse 緩衝（見 4.6b） |
| `capture_proxy_channel`、quarantine、`delay_est_period_s` | 不變 | 不變 |
| lane 的 `filter_length` 等（應用的 `AecConfig`） | 不變 | 四條 lane 等比縮放 |

後級 stage 的記憶體（`AecRes`、`AnrDenoiser`、`AcIstft` 與暫存向量）不在 mcab 的 `req.bytes` 裡：
由應用用各自的 `mem_size` 查、自己加總。sample 層 beamformer helper 的 SRP-PHAT／BF 也不含在內
（`mcab_sample_beam_get_mem_size()`）。

### 4.6 實測記憶體（僅供量級參考，務必自己重查）

以下是本次 checkout（`layout_version=26`；表中「應用端合計」是四條 lane、共用 estimator 與 mcab 自己的 pool 之和）、`BACKEND=ne10`、`SIMD=1`、
`delay_mode=MATCHED`（預設,n=5）、`enable_post=1`（預設）、lane 為 `mcab_lane_config()` 的 balanced 預設下直接呼叫 API
量到的值（LP64 本機 build；mcab 的控制區塊在其中是 768 B）。換 backend、換編譯選項、更新 submodule 都會變。
mcab 的 `McabMemBreakdown` 的大小欄位只有 `control_bytes`、`buffer_bytes`（mic lane、aligned far、fuse 緩衝、ring）與 `total_bytes`（另帶 `hop_size`、`fft_size`、`n_freqs`），
不含 lane、estimator、RES／NR／iFFT。

| Config | mcab pool（`req.bytes`） | estimator（`DelayAec` 物件 1,488 B ＋ pool） | 四條 lane 合計 | 應用端合計 |
|---|---:|---:|---:|---:|
| 16000，預設（256/128，`max_delay_ms=1024`） | 74,624 | 35,424 | 803,968 | 914,016 |
| 16000，`fft_size=512` | 82,816 | 35,424 | 1,318,848 | 1,437,088 |
| 16000，`max_delay_ms=100` | 15,488 | 35,424 | 803,968 | 854,880 |
| 16000，lane `filter_length=512` | 74,624 | 35,424 | 680,128 | 790,176 |
| 48000，預設（1024/512） | 230,272 | 36,800 | 2,936,832 | 3,203,904 |

`mcab_get_mem_breakdown()` 的 `total_bytes` 與 `mcab_get_mem_requirements()` 的
`req.bytes` 在上述每一組都相等；`control_bytes` 已計入 `total_bytes`。四條 lane 合計除以 4 得單路 200,992 B（@16k/256）——
就是 `lib/aec` 的 `AEC_DELAY_EXTERNAL_ALIGNED` 單體大小,因為每路內部本來就是
`EXTERNAL_ALIGNED`（delay 由共用估計器提供,不建自己的 estimator/ring）。

`mcab_external_static` 的 `--print-mem-size` 把「estimator ＋ mcab pool」印成 `Front end:` 一行
（16k/256：110,048；16k/512：118,240；48k/1024：267,072），`AEC x4:` 是上表的四條 lane 合計。
帶 sample 層 SRP-PHAT／BF helper 的範例（`mcab_static --print-mem-size`）的 `Unit:` 行是 flow 全部物件加 helper 的 pool：
16k/256 為 1,706,240，16k/512 為 3,007,040，48k/1024 為 6,329,312 B。

#### 4.6a delay_mode 對 16000/預設 grid 的影響（`enable_post=1`，`MATCHED n=5` 為 baseline）

| `delay_mode` | mcab pool | estimator | 應用端合計 | 相對 `MATCHED n=5` |
|---|---:|---:|---:|---:|
| `MATCHED` n=5（預設） | 74,624 | 35,424 | 914,016 | — |
| `FIXED`，`fixed_delay_samples=1600`（100 ms） | 14,960 | 無 | 818,928 | −95,088 |
| `EXTERNAL_ALIGNED` | 8,048 | 無 | 812,016 | −102,000 |

省下的是**一份共用**的 estimator/ring（四路共用一個 aligner），不是四份各自的——與「單一共用 aligner」的結構一致。

#### 4.6b `enable_post=0`（ULCNet 範例用的 front-only flow）

| Config | mcab pool（`req.bytes`） | 應用端合計 |
|---|---:|---:|
| 16000，`enable_post=0` | 68,368 | 907,760 |
| 16000，`fft_size=512`，`enable_post=0` | 70,416 | 1,424,688 |
| 48000，`enable_post=0` | 205,584 | 3,179,216 |

這是 [`pipeline_ulcnet_4ch.html`](html/pipeline_ulcnet_4ch.html) 的 ULCNet 4ch 範例（`4ch_alignulcnet`，`cfg.enable_post = 0`）所用 flow 的大小；
比同格點 `enable_post=1` 少的是 fuse 緩衝（16 kHz/512：12,400 B，48 kHz：24,688 B）。
**實際配置一律以 `req.bytes` 與各 lib 物件自己的 `get_mem_size` 為準。**

---

導覽：[索引](integration_4ch_core_zh_TW.md) ｜ [上一部分](integration_4ch_core_zh_TW_02_lifecycle.md) ｜ [下一部分](integration_4ch_core_zh_TW_04_errors_checklist_versions.md) <!-- split-nav -->
