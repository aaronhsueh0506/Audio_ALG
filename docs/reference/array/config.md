# 四麥克風 array 設定

## 4. Config 完整參考

array 的設定就是 `aec.h` 的 `AecConfig`，`n_mics = AEC_ARRAY_MICS`（4）。沒有獨立的 array config 型別，也沒有 lane config：陣列從這一份 config 衍生四條 lane
（`EXTERNAL_ALIGNED`、單支、`residual` 為 STATE 或 ESTIMATE，其餘欄位照抄）。`AecConfig` 只有 12 個公開欄位；濾波器長度、shadow、HPF、飽和處理、各門檻與 delay ring 的上限等調校值都在函式庫內部（`AecTuning`），由 `(sample_rate, fft_size, n_mics, level)` 決定。
取得預設值的方式：

```c
AecConfig cfg;
aec_config_from_level(&cfg, AEC_LEVEL_BALANCED, 16000);
cfg.n_mics = AEC_ARRAY_MICS;
/* 之後只覆寫你真正要改的欄位 */
```

sample 層另有 `array_sample_default_config(&cfg, sr)`（`array_sample_beam.h`）：balanced level、MATCHED、bank 5、`residual = AEC_RESIDUAL_FULL`、`n_mics = 4`、該取樣率的預設 `fft_size`。

兩個 factory **不做驗證**，只是填值。驗證發生在 `aec_get_mem_requirements()`／`aec_get_mem_size()`／`aec_get_mem_breakdown()`／`aec_init()`／`aec_init_ex()`／`aec_create()` 共用的同一個 reject-first 閘。

### 4.1 逐欄位

下表只列陣列新增或有不同限制的欄位；其餘欄位（`enable_cng`、`level`）的意義與單支相同，見 `aec.h`，並套用到四條 lane 與增益級。

| 欄位 | 型別 | 預設 | 合法值（驗證器實際檢查的） | 你該怎麼設 |
|---|---|---|---|---|
| `n_mics` | `int` | `1` | 陣列只接受 `AEC_ARRAY_MICS`（4）；`1` 是單支；其他值拒絕 | 設成 4 |
| `sample_rate` | `int` | 你傳給 factory 的值 | 陣列**只有 `16000` 或 `48000`**。`8000` 會被拒絕（單支可用 8 kHz，陣列不行） | 設成你音訊實際的取樣率 |
| `fft_size` | `int` | factory 填 rate 預設（16 kHz → 256，48 kHz → 1024） | `0` **被拒絕**（factory 不填的取樣率會得到 0）。16 kHz 接受 `256` 或 `512`；48 kHz 只接受 `1024` | 除非有明確理由要 16 kHz 的 512 grid，否則用 factory 的預設 |
| `residual` | `AecResidualMode` | `AEC_RESIDUAL_FULL` | `STATE`／`ESTIMATE`／`FULL`；`OFF` 拒絕 | STATE = 只有 lane，不融合（ULCNet 等吃 BF 頻譜的後濾波器）；ESTIMATE = 融合但無增益；FULL = 融合加唯一一個 G_res。見 §4.3 |
| `delay_mode` | `AecDelayMode` | `AEC_DELAY_MATCHED` | `MATCHED`／`FIXED`／`EXTERNAL_ALIGNED` | 描述**共用**的對齊（四條 lane 一律吃已對齊的 far）。`MATCHED` 建共用 estimator 與對齊 ring；`FIXED` 只建固定延遲 ring；`EXTERNAL_ALIGNED` 由呼叫端給已對齊的 far，不建 estimator 也不建 ring（記憶體差異見 [§4.4](#44-檢視欄位記憶體用量與實測-pool) 的連結） |
| `delay_num_filters` | `int` | `5`（`AEC_DELAY_NUM_FILTERS_MAX`） | 僅 `MATCHED`：`[1, 5]`；其餘模式須為預設 | matched-filter bank 大小，每少一個省 5,120 B（pool 內的 estimator 部分） |
| `fixed_delay_samples` | `int` | `-1` | 僅 `FIXED`：`[0, 120 × sample_rate]`，原生取樣點數；其餘模式須為 `-1`；`FIXED` 給 `-1` 拒絕 | 決定共用 delay ring 大小 |
| `capture_proxy_channel` | `int` | `0` | `[0, 3]`。其他值拒絕（單支必須為 0） | 共用 delay matcher 要用哪一支麥克風當輸入。挑訊噪比最穩定、離喇叭關係最典型的那一支 |
| `delay_backward_quarantine_enabled` | `int`（bool） | `0` | `0` 或 `1`（單支必須為 0） | 僅 `MATCHED`。開啟後，比現行 accepted delay **更早**的共用估計會先被隔離（`capture_proxy_channel` 那條 lane 在目前對齊上仍明顯在消除回音時），到期才採用；往後跳與首次取得對齊不受影響。產品組態維持 `0` |
| `delay_backward_quarantine_s` | `float` | `1.0` | 有限值，`[0.0, 3600.0]` | 隔離窗長（秒），init 時換算成 hop 數（最小 1）；enable 為 0 時無作用但仍被驗證 |

**level（強度）是 `AecConfig.level`**（`aec_config_from_level()` 寫入；`aec_set_level()` 在執行期更換並存回實例；不在 0..3 會被拒絕，全零的 `AecConfig` 是 `MILD`）：只作用在融合後的增益級（`FULL`），lane 自己的地板什麼都不塑形（[integration.md §3.2](integration.md#32-執行期強度控制)）。
RES 與 NR 的其他設定：增益級的調校值（地板、近端偵測、double-talk）是函式庫內部的，隨 `level` 成組決定；NR 是你的 `AnrConfig`（`anr_config_for_level_grid()`）。共用 delay ring 的容量由函式庫決定（`MATCHED` 為 estimator 可回報的最大延遲加 hop，與單聲道同式；`FIXED` 為 `fixed_delay_samples` 加 hop；隨 `delay_mode` 與 `delay_num_filters`），不是欄位。
`AecConfig.residual` 決定陣列做到哪裡：`AEC_RESIDUAL_STATE`（只有 lane）、`ESTIMATE`（融合、無增益）或 `FULL`。

### 4.2 Grid（由 `sample_rate` + `fft_size` 唯一決定）

`hop = fft_size / 2`、`n_freqs = fft_size / 2 + 1`。陣列合法組合只有三組：

| `sample_rate` | `fft_size` | `hop` | `n_freqs` | 備註 |
|---:|---:|---:|---:|---|
| 16000 | 256 | 128 | 129 | factory 的 rate 預設 |
| 16000 | 512 | 256 | 257 | 需明確指定 |
| 48000 | 1024 | 512 | 513 | factory 的 rate 預設 |

**不要把 hop / n_freqs 寫死。** 用 `aec_hop_size()`（n_freqs = hop + 1）或 `aec_resolve_signal_grid(sr, fft, &g)` 查。

### 4.3 Buffer 形狀（每次 `aec_process_linear` / `aec_process_residual`）

| Buffer | 誰擁有 | 形狀 | 說明 |
|---|---|---|---|
| `mics` | 你 | `float[hop * 4]`，`[sample * 4 + channel]` | 唯讀，只在該次呼叫期間被讀取。所有元素必須是有限值 |
| `ref` | 你 | `float[hop]` | 唯讀。所有元素必須是有限值 |
| `AecBeam.weights` | 你 | `Complex[4 * n_freqs]`，**channel-major**：`[channel * n_freqs + bin]` | 唯讀 |
| `AecBeam.beamformed_error` | 你（可為 NULL） | `Complex[n_freqs]` | 與 `weights` 同一次 beamformer 呼叫產生的 mono 誤差頻譜；NULL 則由權重重建。供應時 `AecResidualView.error_spec` 別名這份陣列，它必須活過這個 hop 的 view |
| `out` | 你 | `float[hop]` | 由你的 `ac_istft_push()` 寫入 |
| `AecArrayView.lane_formed_hop[ch]` | 陣列 | `const float[hop]` | 見 §4.4 |
| `AecArrayView.lane_error_spec[ch]` | 陣列 | `const Complex[n_freqs]` | 見 §4.4 |

**權重約定**：`output[bin] = Σ_channel weights[channel][bin] * input[channel][bin]`，**不做共軛**。多幀 / 時域型 GSC 請把它「這一幀的有效頻率響應」填進來。

**一條在每個 bin 權重都是 0 的 lane 不會投票。** `aec_process_residual()` 除了投影頻譜，還會把四條 lane 的 `filter_converged` / `usable_linear` / `dt_indicator` / `saturation_level`
縮成純量（`AecResidualView.all_converged`／`usable_linear`／`max_dt`／`max_saturation`）去操縱那個共用的增益級（其中 `dt_indicator` 決定它套哪一個地板）。
一條在每個 bin 權重都是 0 的 lane 對融合後的頻譜貢獻恰好為零，所以也不參與這些縮減——縮減從中性值起算（`all_converged`、`usable_linear` 為 1，`max_dt`、`max_saturation` 為 0）。
四條 lane 的 far 功率一致性檢查不受影響，對每條 lane 都照做。整組全零權重會先被 `AEC_ERR_ARG` 擋掉（見 [errors.md §5.3](errors.md#53-aec_process_residual-的完整拒絕清單依實際檢查順序)），所以至少有一條 lane 會投票；真正在 beamform 的權重每條 lane 都有非零 bin，這條規則不會跳過任何東西。

外部 beamformer**不需要**再回傳一份 mono hop —— 合成是在你自己的 `ac_istft_push()` 內以那組權重加權後的頻譜完成的。

### 4.4 檢視欄位、記憶體用量與實測 pool

這三項是函式庫的行為，不在此重複，以 AEC 文件與 `aec.h` 為準：

| 內容 | 在哪裡 |
|---|---|
| `AecArrayView`（`lane_error_spec`、`lane_base_spec`、`lane_formed_hop`、`lane_guard_active`、`aligned_ref`、`delay`）、`AecResidualView`（陣列上的欄位）、`AecDelayState`、視圖的有效期 | [integration.md §8.3、§8.8](../../../../AEC/docs/reference/integration.md#88-麥克風陣列實例n_mics--4)、`aec.h` 的 Array instance 一節 |
| 哪些 `AecConfig` 欄位改變 `req.bytes`（grid、`delay_mode`、`delay_num_filters`、`residual` …） | [lifecycle.md「陣列實例的記憶池」](../../../../AEC/docs/reference/lifecycle.md#陣列實例的記憶池)、[configuration.md](../../../../AEC/docs/reference/configuration.md) |
| 實測 pool 大小（`aec_get_mem_requirements()`／`aec_get_mem_breakdown()`；各 `delay_mode`、各 `residual` 的差異） | [lifecycle.md「陣列實例的記憶池」](../../../../AEC/docs/reference/lifecycle.md#陣列實例的記憶池)、AEC repo 的 `c_impl/STATIC_MEMORY.md` |

後級 stage 的記憶體（`AnrDenoiser`、`AcIstft` 與暫存向量）不在 `req.bytes` 裡：由應用用各自的 `mem_size` 查、自己加總。
sample 層 beamformer helper 的 SRP-PHAT／BF 也不含在內（`array_sample_beam_get_mem_size()`）；含 helper 的 Unit／Total 數字見 [beam-settings.md §4.7](beam-settings.md#47-哪些欄位會改變記憶體用量)。
**實際配置一律以 `req.bytes` 與各 lib 物件自己的 `get_mem_size` 為準。**

[返回本主題](README.md)
