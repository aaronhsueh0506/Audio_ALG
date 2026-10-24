導覽：[索引](integration_4ch_core_zh_TW.md) ｜ [上一部分](integration_4ch_core_zh_TW_02_lifecycle.md) ｜ [下一部分](integration_4ch_core_zh_TW_04_errors_checklist_versions.md) <!-- split-nav -->
## 4. Config 完整參考

`FourAecNrResConfig` 的欄位如下表。取得預設值的唯一正確方式：

```c
FourAecNrResConfig cfg = four_aec_nr_res_default_config(16000);
/* 之後只覆寫你真正要改的欄位 */
```

`four_aec_nr_res_default_config()` **不做驗證**，只是填值。
驗證發生在 `get_mem_requirements()` / `init()` / `init_ex()` / `get_mem_breakdown()`
共用的同一個 reject-first 閘。

### 4.1 逐欄位

| 欄位 | 型別 | `default_config()` 給的值 | 合法值（驗證器實際檢查的） | 你該怎麼設 |
|---|---|---|---|---|
| `sample_rate` | `int` | 你傳給 `default_config()` 的值 | **只有 `16000` 或 `48000`**。`8000` 會被拒絕（這點和單聲道 pipeline 不同） | 設成你音訊實際的取樣率 |
| `fft_size` | `int` | `0` | `0` = 依 rate 取預設（16 kHz → 256，48 kHz → 1024）。明確指定時：16 kHz 接受 `256` 或 `512`；48 kHz 只接受 `1024` | 除非有明確理由要 16 kHz 的 512 grid，否則留 `0` |
| `filter_length` | `int` | `0` | `[0, 4096]`。負數或 > 4096 拒絕 | `0` = 沿用 AEC 依取樣率算出的預設（16 kHz → 832 taps，48 kHz → 3072 taps）。回聲尾巴短、記憶體吃緊時可調小；這是四路 AEC 記憶體最主要的旋鈕 |
| `delay_mode` | `AecDelayMode` | `AEC_DELAY_MATCHED` | `MATCHED` / `FIXED` / `EXTERNAL_ALIGNED` | 四條 lane 共用同一個決策。`MATCHED` 建共用 estimator 與對齊 ring；`FIXED` 只用固定延遲 ring；`EXTERNAL_ALIGNED` 由呼叫端給已對齊的 far，不建 matcher 也不建 ring（記憶體差異見 4.6a） |
| `delay_num_filters` | `int` | `5` | 僅 `MATCHED`：`[1, 5]`；其餘模式須為預設 | matched-filter bank 大小，每少一個省 5,728 B |
| `fixed_delay_samples` | `int` | `-1` | 僅 `FIXED`：原生取樣點數；其餘模式須為 `-1` | 決定共用 delay ring 大小 |
| `capture_proxy_channel` | `int` | `0` | `[0, 3]`。其他值拒絕 | 共用 delay matcher 要用哪一支麥克風當輸入。挑訊噪比最穩定、離喇叭關係最典型的那一支 |
| `max_delay_ms` | `float` | `1024.0f` | 必須是有限值，且落在 `[0.0, 4096.0]` | 共用 reference delay line 的容量上限（毫秒）。實際的 ring 大小 = `ceil(max_delay_ms * sample_rate / 1000) + 2 * hop + 1` 個 sample。設得比你系統真實最大 mic↔ref 延遲**略大**即可；設太大只是白白吃記憶體 |
| `delay_backward_quarantine_enabled` | `int`（bool） | `0` | `0` 或 `1` | 僅 `MATCHED`。開啟後，比現行 accepted delay **更早**的共用估計會先被隔離（`capture_proxy_channel` 那條 lane 在目前對齊上仍明顯在消除回音時），到期才採用；往後跳與首次取得對齊不受影響。產品組態維持 `0`；機制說明見 `integration_4ch_spatial_zh_TW_03_config.md` 的 `core.delay_backward_quarantine_enabled` 列 |
| `delay_backward_quarantine_s` | `float` | `1.0` | 有限值，`[0.0, 3600.0]` | 隔離窗長（秒），`init` 時換算成 hop 數（最小 1）；enable 為 0 時無作用但仍被驗證 |
| `delay_est_period_s` | `float` | `0.0` | 有限值，`[0.0, 3600.0]` | 共用 matched-filter 的 duty-cycle 週期：`0.0` = 每 hop 全速分析；正值時估計 solid 且不變 `delay_est_init_s`（0.3 s，固定）後，改為每 round(`delay_est_period_s`/hop)÷5 個 hop（最小 1）分析一次，省算力但可能鎖到不同的 lag |
| `enable_post` | `int`（bool） | `1` | 只接受 `0` 或 `1` | `1` = 前端 carve fuse 緩衝，可呼叫 `four_aec_fuse()`；`0` = front-only 前端（只有 `four_aec_front_process()`，`four_aec_fuse()` 回 `INVALID_ARGUMENT`），給從不 fuse 的應用（例如 ULCNet）。見 4.6b |

前端**沒有** RES、NR、comfort noise 或 preset 的設定欄位：`aec_preset`、`nr_mode`、`enable_res`、`enable_nr`、
`enable_cng`、`legacy_amin` 都不在 `FourAecNrResConfig` 裡（lane 一律以 `AEC_PRESET_BALANCED` 導出，且只算線性殘留）。
這些屬於你的後級串接（例如 `4aec_nr_res_static.c` 的 `--no-res`、`--nr-preset`、`--no-cng`、`--legacy-amin` 旗標對應的選項）：
RES 用 `aec_config_from_preset(&res_cfg, preset, sample_rate)` 配上核心解析出的 `fft_size` 建 `AecRes`，NR 用 `anr_config_for_mode_grid()`。

### 4.2 Grid（由 `sample_rate` + `fft_size` 唯一決定）

`hop = fft_size / 2`、`n_freqs = fft_size / 2 + 1`。合法組合只有三組：

| `sample_rate` | `fft_size` | `hop` | `n_freqs` | 備註 |
|---:|---:|---:|---:|---|
| 16000 | 256 | 128 | 129 | `fft_size = 0` 的 rate 預設 |
| 16000 | 512 | 256 | 257 | 需明確指定 |
| 48000 | 1024 | 512 | 513 | `fft_size = 0` 的 rate 預設 |

**不要把 hop / n_freqs 寫死。** 用 `four_aec_nr_res_hop_size()` / `four_aec_nr_res_n_freqs()` 查。


### 4.3 Buffer 形狀（每次 `front_process` / `fuse`）

| Buffer | 誰擁有 | 形狀 | 說明 |
|---|---|---|---|
| `microphones_interleaved` | 你 | `float[hop * 4]`，`[sample * 4 + channel]` | 唯讀，只在該次呼叫期間被讀取。所有元素必須是有限值 |
| `ref` | 你 | `float[hop]` | 唯讀。所有元素必須是有限值 |
| `weights` | 你 | `Complex[4 * n_freqs]`，**channel-major**：`[channel * n_freqs + bin]` | 唯讀 |
| `beamformed_error` | 你（可為 NULL） | `Complex[n_freqs]` | 與 `weights` 同一次 beamformer 呼叫產生的 mono 誤差頻譜；NULL 則由權重重建。供應時 `frame.e` 別名這份陣列，它必須活過 frame |
| `out` | 你 | `float[hop]` | 由你的 `ac_istft_push()` 寫入 |
| `front.formed_hop[ch]` | 函式庫 | `const float[hop]` | 見 §4.4 |
| `front.linear_spectra[ch]` | 函式庫 | `const Complex[n_freqs]` | 見 §4.4 |

**權重約定**：`output[bin] = Σ_channel weights[channel][bin] * input[channel][bin]`，
**不做共軛**。多幀 / 時域型 GSC 請把它「這一幀的有效頻率響應」填進來。

**一條在每個 bin 權重都是 0 的 lane 不會投票。** `four_aec_fuse()` 除了投影頻譜，還會把四條
lane 的 `filter_converged` / `dt_indicator` / `saturation_level` 縮成三個純量（`frame.all_converged`／`max_dt`／`max_sat`）
去操縱那個共用的 post 級抑制器（其中 `dt_indicator` 決定它套哪一個地板）。一條在每個 bin 權重都是 0 的 lane 對
融合後的頻譜貢獻恰好為零，所以也不參與這三個縮減——縮減從中性值起算，lane 0 一樣受檢。整組
全零權重會先被 `-1` 擋掉（見 [§5.3](integration_4ch_core_zh_TW_04_errors_checklist_versions.md#53-four_aec_fuse-的完整拒絕清單依實際檢查順序)），所以至少有一條 lane 會通過這個檢查；真正在 beamform 的
權重每條 lane 都有非零 bin，這條規則不會跳過任何東西。

外部 beamformer**不需要**再回傳一份 mono hop —— 合成是在你自己的 `ac_istft_push()` 內
以那組權重加權後的頻譜完成的。

### 4.4 `FourAecFront` 與 `SeSpecFrame` 各欄位

`FourAecFront`（`four_aec_front_process()` 填）：

| 欄位 | 型別 | 內容 |
|---|---|---|
| `delay` | `FourAecNrResDelayState` | 共用 matcher 這一幀的狀態，見下 |
| `hop_size` | `int` | 這一幀的 hop |
| `n_channels` | `int` | 恆為 `FOUR_AEC_NR_RES_CHANNELS`（4） |
| `n_freqs` | `int` | 這一幀的 bin 數 |
| `linear_spectra[ch]` | `const Complex*` | 四路 linear AEC 的**頻譜**，就是 `four_aec_fuse()` 稍後會用的同一份。讓外部 beamformer 直接沿用既有的分析轉換，不必重做一次 STFT。**這是別名**，直接指向各 lane 自己的 buffer |
| `formed_hop[ch]` | `const float*` | 各 lane 與 `linear_spectra[ch]` 同一次選擇的**時域** hop（`hop_size` 個 sample）。做自己的 sqrt-Hann 分析的時域 beamformer 要吃這個。**這也是借用的**，沒有 interleaved 拷貝 |
| `aligned_ref` | `const float*` | 這一 hop 實際餵給四條 lane 的對齊後 far（`hop_size` 個 sample）；給需要「與線性濾波器吃到同一份 far」的外部 NN 後級用 |

`SeSpecFrame`（`four_aec_fuse()` 填，定義在 `pipelines/common/se_stage.h`）：

| 欄位 | 內容 |
|---|---|
| `n_freqs`, `hop` | 這一幀的 bin 數與 hop |
| `e` | beamformed 誤差頻譜（pre-RES；RES 與 NR 據以估計，也是增益套用的對象） |
| `y` | 投影後的近端頻譜 |
| `x` | lane 0 的 far 頻譜（四條 lane 已驗證吃同一份） |
| `r2` | 投影後的殘餘回音 PSD（audio scale，餵 NR 前由 stage 除以 `32768²`） |
| `cn` | 投影後的 comfort-noise PSD |
| `far_hop` | 對齊後的 far（同 `aligned_ref`） |
| `max_dt`, `max_sat`, `all_converged` | 對權重有非零響應的 lane 取的 double-talk／飽和／收斂縮減 |
| `clock_drift`, `delay_changed`, `delay_locked` | 共用 aligner 的旗標 |
| `g_res_ctx` | 恆為 NULL（核心的四條 lane 都不算自己的增益；單元的 `audio_pipeline_4ch_fuse()` 也一樣，RES 增益由你的 `AecRes` 經 `se_res_stage_gain()` 算出） |

`FourAecNrResDelayState` 欄位：

| 欄位 | 型別 | 意義 |
|---|---|---|
| `delay_samples` | `int` | 目前採用的延遲（sample） |
| `confidence` | `float` | 估計信心值 |
| `solid` | `int` | 估計是否已穩定 |
| `changed` | `int` | 非 0 表示這一幀延遲值改變了 —— **你外部的 STFT / OLA 歷史必須跟著清掉**（前端自己對四條 lane 做 realign，不重置也不清合成 OLA） |
| `estimator_calls` | `uint64_t` | 累計呼叫次數 |
| `estimator_updates` | `int` | 累計更新次數 |

**front 視圖與 frame 的生命週期（重要）**：
`FourAecFront` 的所有指標與 `SeSpecFrame` 的所有指標，**只在下列任一事件發生之前有效**：
下一次 `four_aec_front_process()`、`four_aec_nr_res_reset()`、`four_aec_nr_res_destroy()`。
所有 stage 對它們都是**唯讀**。

這是 **API 約定，不是記憶體安全保證**。
`reset()` 不保證那些記憶體被清掉或覆寫 —— lane 的舊頻譜可能就原地留著。
也就是說：你違反這條規則時，程式**不會**壞給你看，只會安靜地讀到未定義內容。
不要用「讀到垃圾」當作偵測手段。

### 4.5 哪些欄位會改變記憶體用量

| 欄位 | 影響 `req.bytes` 嗎 |
|---|---|
| `sample_rate` | 會（改 grid、改 AEC 預設 filter 長度） |
| `fft_size` | 會（改 grid） |
| `filter_length` | 會（四路 AEC 等比縮放） |
| `delay_mode` | 會（僅 `MATCHED` 建共用 estimator；三態差異見 4.6a） |
| `delay_num_filters` | 僅 `MATCHED` 有效，每少一個固定省 5,728 B |
| `fixed_delay_samples` | 僅 `FIXED` 有效，決定共用 delay ring 大小 |
| `max_delay_ms` | 會（僅 `MATCHED` 用於 sizing delay ring） |
| `enable_post` | 會（`0` 時不 carve fuse 緩衝，見 4.6b；ULCNet 範例用這個） |
| `capture_proxy_channel` | 不變 |

後級 stage 的記憶體（`AecRes`、`AnrDenoiser`、`AcIstft` 與暫存向量）不在前端的 `req.bytes` 裡：
由應用用各自的 `mem_size` 查、自己加總。
線性 AEC + BF 單元的 `audio_pipeline_4ch_get_mem_requirements()` 同樣不含 `AecRes`、NR 與 `AcIstft`。

### 4.6 實測記憶體（僅供量級參考，務必自己重查）

以下是本次 checkout（`layout_version=24`）、`BACKEND=ne10`、`SIMD=1`、
`delay_mode=MATCHED`（預設,n=5）、`enable_post=1`（預設）下直接呼叫 API
量到的前端值。換 backend、換編譯選項、更新 submodule 都會變。前端不含 RES／NR／iFFT，
所以 `FourAecNrResMemBreakdown` 只有 `aec_bytes`、`wrapper_bytes`（控制區塊、共用 aligner 與 fuse 緩衝）與 `total_bytes`。

| Config | `req.bytes` | `aec_bytes`（四路合計） | `wrapper_bytes` |
|---|---:|---:|---:|
| 16000，預設（256/128，`max_delay_ms=1024`） | 906,560 | 797,632 | 108,928 |
| 16000，`fft_size=512` | 1,422,464 | 1,306,368 | 116,096 |
| 16000，`max_delay_ms=100` | 847,424 | 797,632 | 49,792 |
| 16000，`filter_length=512` | 782,720 | 673,792 | 108,928 |
| 48000，預設（1024/512） | 3,174,944 | 2,912,064 | 262,880 |

`four_aec_nr_res_get_mem_breakdown()` 的 `total_bytes` 與 `get_mem_requirements()` 的
`req.bytes` 在上述每一組都相等；`wrapper_bytes` 已包含控制區塊。四路合計的
`aec_bytes` 除以 4 得單路 199,408 B（@16k/256）——與 `lib/aec` 的
`AEC_DELAY_EXTERNAL_ALIGNED` 單體大小完全相同,因為每路內部本來就是
`EXTERNAL_ALIGNED`（delay 由本層共用估計器提供,不建自己的 estimator/ring）。

#### 4.6a delay_mode 對 16000/預設 grid 的影響（`enable_post=1`，`MATCHED n=5` 為 baseline）

| `delay_mode` | `req.bytes` | 相對 `MATCHED n=5` |
|---|---:|---:|
| `MATCHED` n=5（預設） | 906,560 | — |
| `FIXED`，`fixed_delay_samples=1600`（100 ms） | 812,960 | −93,600 |
| `EXTERNAL_ALIGNED` | 806,048 | −100,512 |

省下的量比單聲道版本小，因為這裡只省**一份共用**的 estimator/ring（四路
共用一個 aligner），不是四份各自的——與本頁「單一共用 aligner」的結構
一致。

#### 4.6b `enable_post=0`（ULCNet 範例用的 front-only 前端）

| Config | `req.bytes` |
|---|---:|
| 16000，`fft_size=512`，`enable_post=0` | 1,412,128 |
| 48000，`enable_post=0` | 3,154,368 |

這是 [`pipeline_ulcnet_4ch.html`](html/pipeline_ulcnet_4ch.html) 的 ULCNet 4ch 範例所用單元
（`AudioPipeline4Ch`，`core.enable_post = 0`）內的前端大小；比同格點 `enable_post=1` 少的是 fuse 緩衝（16 kHz/512：10,336 B，48 kHz：20,576 B）。
**實際配置一律以 `req.bytes` 為準。**

---

導覽：[索引](integration_4ch_core_zh_TW.md) ｜ [上一部分](integration_4ch_core_zh_TW_02_lifecycle.md) ｜ [下一部分](integration_4ch_core_zh_TW_04_errors_checklist_versions.md) <!-- split-nav -->
