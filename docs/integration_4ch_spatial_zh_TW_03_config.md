> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_02_lifecycle.md) ｜ [下一份](integration_4ch_spatial_zh_TW_04_errors.md) <!-- split-nav -->
## 4. Config 完整參考

`AudioPipeline4ChConfig` = 巢狀的 `FourAecNrResConfig core`（12 個欄位）
+ 本層自己的 23 個欄位（幾何 6、DOA 7、BF 7、內建 VAD 3）。RES、NR、comfort noise 的設定**不在**這裡（它們屬於你的後級串接）。取得預設值的唯一正確方式：

```c
AudioPipeline4ChConfig cfg = audio_pipeline_4ch_default_config(16000);
/* 之後只覆寫你真正要改的欄位 */
```

`audio_pipeline_4ch_default_config()` **不做驗證**，只是填值。

### 4.1 `cfg.core`（核心層欄位）

本層會把 `cfg.core` 原封不動交給核心層驗證與建構，規則完全一樣：

| 欄位 | 型別 | 預設 | 合法值 | 你該怎麼設 |
|---|---|---|---|---|
| `core.sample_rate` | `int` | 你傳給 `default_config()` 的值 | **只有 `16000` 或 `48000`** | 你音訊的取樣率 |
| `core.fft_size` | `int` | `0` | `0` = rate 預設（16 kHz → 256，48 kHz → 1024）；16 kHz 另可指定 `512`；48 kHz 只接受 `1024` | 一般留 `0` |
| `core.filter_length` | `int` | `0` | `[0, 4096]` | `0` = AEC 依取樣率算的預設（16 kHz → 832，48 kHz → 3072）。記憶體吃緊時最有效的旋鈕 |
| `core.capture_proxy_channel` | `int` | `0` | `[0, 3]` | 共用 delay matcher 用哪一支麥克風 |
| `core.delay_backward_quarantine_enabled` | `int`（bool） | `0` | `0` 或 `1` | 預設關閉。開啟後只隔離**一種**共用估計：比現行 accepted delay **更早**（backward／pre-echo 方向）、且 **`capture_proxy_channel` 那一條 lane**（不是任一條——那正是餵給共用 estimator 的那支 mic，也是唯一能對這個估計提供證據的 lane）在目前對齊上仍明顯在消除回音（`aec_linear_is_cancelling()`）。**往後跳（delay 變大）永不隔離**，**首次取得對齊不受影響**。隔離**有上限**（見下一列），到期就採用；消除能力崩掉時則立即採用，但這是 hard replacement 常見而非所有真實路徑變化必然具備的現象。**它不是 pre-echo 誤鎖的解藥**：那個誤鎖已在估計器端根治（aggregator 改為回報 dominant peak，見 `AEC/docs/aec_methods.md`），這個機制擋的是真實路徑往前移動時的候選變動，仍然只是延後一個窗。`FIXED` / `EXTERNAL_ALIGNED` 不重新決定對齊，此旗標對它們無作用 |
| `core.delay_backward_quarantine_s` | `float` | `1.0` | 有限值，`[0.0, 3600.0]` | 上一列的隔離窗長（秒），在 `init` 時按解析後的 hop／取樣率換算成 hop 數（最小 1；要關閉請用上一列的 enable）。enable 為 0 時無作用，但仍會被驗證——設定檔今天就要是「翻一個旗標也不會壞」的狀態。實測（multipath 場景，舊路徑不消失所以消除能力不崩、只有到期能釋放）：窗 1.0 s = 62 hops，採用點精確等於「未隔離 hop ＋ 窗長」。 |
| `core.max_delay_ms` | `float` | `1024.0f` | 有限值，`[0.0, 4096.0]` | 設成略大於系統真實最大 mic↔ref 延遲。設太大只是白吃記憶體 |
| `core.enable_post` | `int`（bool） | `1` | `0` 或 `1` | `1` = lane 保留 post context，`audio_pipeline_4ch_fuse()` 可用；`0` = 只留線性 context（`AEC_CONTEXT_LINEAR_ONLY`），`fuse()` 回 `INVALID_ARGUMENT`，給直接吃 BF 頻譜的 ULCNet。單元的驗證**不再**要求它為 1 |

另外還有 `core.delay_mode`、`core.delay_num_filters`、`core.fixed_delay_samples`、`core.delay_est_period_s`
（語意見核心手冊 §4.1）。核心沒有 `aec_preset`、`nr_mode`、`enable_nr`、`enable_cng`、`legacy_amin` 欄位：
lane 一律以 `AEC_PRESET_BALANCED` 導出並只算線性殘留。RES 不是本層欄位，由你自己的 `AecRes` 持有（[§1.1](integration_4ch_spatial_zh_TW.md#11-這是什麼)、[§2](integration_4ch_spatial_zh_TW_01_quick_start.md#2-quick-start)）。

> Quarantine 窗只對「連續符合條件的 backward episode」有界。候選轉為
> forward／無效、confidence 中斷，或 proxy lane 的 cancellation 證據消失時
> 會解除本次隔離；之後的新 backward episode 重新計時。Cancellation collapse
> 常見於路徑完全替換，但不是 multipath／新增路徑的必然特徵。
>
> **產品預設維持 `0`。** 打開之後只是把誤鎖往後推一個窗，不會修正誤鎖本身。

Grid 只有三組（`hop = fft_size / 2`、`n_freqs = fft_size / 2 + 1`）：

| `core.sample_rate` | `core.fft_size` | `hop` | `n_freqs` |
|---:|---:|---:|---:|
| 16000 | 256（rate 預設） | 128 | 129 |
| 16000 | 512 | 256 | 257 |
| 48000 | 1024（rate 預設） | 512 | 513 |

### 4.2 陣列幾何（4 + 2 欄）

| 欄位 | 型別 | 預設 | 合法值 | 你該怎麼設 |
|---|---|---|---|---|
| `geometry` | `AudioPipeline4ChGeometry` | `AUDIO_PIPELINE_4CH_GEOMETRY_UCA`（= 0） | `UCA`(0) / `ULA`(1) / `CUSTOM`(2)。其他值拒絕 | 依你的實體陣列選 |
| `uca_radius_m` | `float` | `0.035f` | **僅在 `geometry == UCA` 時檢查**：必須有限且 `> 0` | 圓陣半徑（公尺）。量你的板子 |
| `ula_spacing_m` | `float` | `0.035f` | **僅在 `geometry == ULA` 時檢查**：必須有限且 `> 0` | 線陣相鄰間距（公尺） |
| `microphone_x_m[4]` | `float[4]` | 依預設半徑 0.035 排出的 UCA 座標 | **僅在 `geometry == CUSTOM` 時檢查**：4 個都必須是有限值 | 每支麥克風的 x 座標（公尺） |
| `microphone_y_m[4]` | `float[4]` | 同上 | 同上 | 每支麥克風的 y 座標（公尺） |
| `speed_of_sound_m_s` | `float` | `343.0f` | 有限且 `> 0` | 一般不需要改 |

**重要陷阱**：`geometry == UCA` 時，實際幾何是**從 `uca_radius_m` 現算的**，
`microphone_x_m` / `microphone_y_m` 的內容**會被忽略**。
同理 `ULA` 只看 `ula_spacing_m`。只有 `CUSTOM` 才會讀那兩個陣列。
`default_config()` 之所以順手把兩個陣列填成預設 UCA 座標，是為了讓你直接切成
`CUSTOM` 時不會拿到未初始化的值 —— 但那組座標對應的是**預設半徑**，
不是你之後改過的 `uca_radius_m`。要用 `CUSTOM` 就自己把兩個陣列填滿。

### 4.3 Direction finding（DOA，7 欄）

| 欄位 | 型別 | 預設 | 合法值 | 你該怎麼設 |
|---|---|---|---|---|
| `num_angles` | `int` | `72` | `[4, 3600]` | 角度掃描解析度。**這是本層最大的記憶體旋鈕**（見 §4.7）。72 相當於每 5 度一格 |
| `doa_low_freq_hz` | `float` | `300.0f` | 有限、`>= 0`、且 **`<` Nyquist**（`sample_rate / 2`） | 定位用的頻帶下限 |
| `doa_high_freq_hz` | `float` | `7000.0f` | 有限、`>= 0`、`<=` Nyquist；若 `> 0` 則必須 `>= doa_low_freq_hz` | 定位用的頻帶上限。**`0` 代表自動**，實際取 `min(7000, Nyquist)` |
| `doa_enable_smoothing` | `int`（bool） | `1` | `0` 或 `1` | `1` = 啟用角度平滑 |
| `doa_switch_consecutive` | `int` | `3` | **必須 `> 0`** | 要連續幾幀指向新方向才真的切換。調大 = 更遲鈍但更穩 |
| `doa_angle_tolerance_rad` | `float` | `10° ≈ 0.174533` rad | 有限且 `>= 0` | 多大的角度差才算「換方向」 |
| `doa_update_interval` | `int` | `2` | **必須 `> 0`** | 每幾個 hop 更新一次 DOA |

### 4.4 Beamformer（BF，7 欄）

| 欄位 | 型別 | 預設 | 合法值 | 你該怎麼設 |
|---|---|---|---|---|
| `bf_enable` | `int`（bool） | `1` | `0` 或 `1` | `0` = 關閉 BF：直接輸出 channel 0，並在每個 bin 送出單位權重 `[1,0,0,0]`，因此 lane 1–3 權重全零。全零權重的 lane 不參與 post 級的三個縮減，見 [`integration_4ch_core_zh_TW.md` §4.3](integration_4ch_core_zh_TW_03_config.md#43-buffer-形狀每次-front_process--fuse) |
| `bf_lambda` | `float` | `0.995f` | 有限，且 `> 0.0` 並 `<= 1.0` | 自適應的遺忘因子。**你設的值不是實際生效的值，見 §4.6** |
| `bf_mu` | `float` | `0.1f` | 有限且 `>= 0` | 步長增益。**不會**被 retime，設多少就是多少 |
| `bf_fixed_mode` | `int`（bool） | `0` | `0` 或 `1` | `1` = 用固定方向，不吃 DOA 結果 |
| `bf_fixed_doa_rad` | `float` | `0.0f` | 有限值（不限範圍） | `bf_fixed_mode == 1` 時採用的固定方向（弧度） |
| `bf_fixed_align_notebook` | `int`（bool） | `0` | `0` 或 `1` | 固定對齊模式。**開啟時會強制改變 `bf_adapt_interval` 的生效值，見 §4.6** |
| `bf_adapt_interval` | `int` | `1` | **必須 `> 0`** | 每幾個 hop 做一次自適應更新。**你設的值不一定是實際生效的值，見 §4.6** |

### 4.5 內建 fallback VAD（3 欄）

**這三個欄位只在你呼叫 `audio_pipeline_4ch_process()` 時才會被用到。**
如果你走 `audio_pipeline_4ch_process_with_activity()`（產品整合建議用這條），
活動偵測完全由你提供，這三個欄位不影響任何行為 —— 但**仍然會被驗證**，
所以還是要保持合法值（照 `default_config()` 留著即可）。

| 欄位 | 型別 | 預設 | 合法值 | 你該怎麼設 |
|---|---|---|---|---|
| `auto_vad_threshold_dbfs` | `float` | `-55.0f` | 必須是有限值（不限範圍） | 判定為語音的最低能量門檻（dBFS）。調高 = 更不容易誤判成語音 |
| `auto_vad_snr_ratio` | `float` | `3.0f` | 有限且 **`>= 1.0`** | 相對背景噪聲要高出幾倍才算語音 |
| `auto_vad_hangover_frames` | `int` | `8` | 必須 `>= 0` | 語音結束後再保持多少幀。**你設的值不是實際生效的值，見 §4.6** |

### 4.6 你設定的值不一定是實際生效的值

**這一節是本手冊最重要的一節。** 有三個欄位在 init 時會被靜默轉換。
你在 `AudioPipeline4ChConfig` 裡讀到的值，和 pipeline 內部真正使用的值**不同**。
不知道這件事，你會在調參時得到完全無法解釋的結果。

#### (1) `bf_adapt_interval` —— 在 fixed-notebook 模式下被強制成 1

規則（來自 `bf_effective_adapt_interval()`）：

```
effective_interval = (bf_adapt_interval > 0) ? bf_adapt_interval : 1
if (bf_fixed_mode == 1 && bf_fixed_align_notebook == 1)
    effective_interval = 1        /* 不管你設了什麼 */
```

也就是說：只要 `bf_fixed_mode` 與 `bf_fixed_align_notebook` **同時**為 1，
你設的 `bf_adapt_interval = 4` 會被完全忽略，實際就是每一個 hop 更新一次。

**讀回生效值**：

```c
int effective = audio_pipeline_4ch_bf_effective_adapt_interval(p);
```

#### (2) `bf_lambda` —— 依 hop × 生效 interval 重新定時

`bf_lambda` 是以「每 10 ms 更新一次」為基準調出來的係數。
真正的遺忘時間常數會隨 `hop_size` / `sample_rate` 改變，所以 init 時會做換算：

```
effective_lambda = configured_lambda ^ ( (hop_size * effective_interval / sample_rate) / 0.010 )
```

注意指數裡用的是 **(1) 算出來的 `effective_interval`**，不是你設的原始值。
（`bf_mu` 是步長增益、不是時間常數，所以**不做**這個換算。）

**讀回生效值**：

```c
float effective = audio_pipeline_4ch_bf_lambda(p);
```

實測對照（`bf_lambda = 0.995`）：

| `sample_rate` | `hop` | 你設的 `bf_adapt_interval` | `bf_fixed_mode` / `bf_fixed_align_notebook` | 生效 interval | 生效 lambda |
|---:|---:|---:|---|---:|---:|
| 16000 | 128 | 1 | 0 / 0 | 1 | 0.995998025 |
| 16000 | 128 | 4 | 0 / 0 | 4 | 0.984087825 |
| 16000 | 128 | 4 | **1 / 1** | **1** | **0.995998025** |
| 16000 | 128 | 4 | 1 / 0 | 4 | 0.984087825 |
| 48000 | 512 | 1 | 0 / 0 | 1 | 0.994667590 |

看第 2 列與第 3 列：`bf_adapt_interval` 明明都是 4，
只因為多開了 `bf_fixed_align_notebook`，interval 與 lambda **兩個都變了**。

#### (3) `auto_vad_hangover_frames` —— 依 hop 重新定時

這個欄位的單位是「以 10 ms 為一幀」的幀數，init 時會換算成這個 grid 的實際幀數：

```
effective_frames = ceil( auto_vad_hangover_frames * sample_rate / (100 * hop_size) )
```

（`auto_vad_hangover_frames == 0` 時生效值也是 0，即關閉 hangover。）

實測對照（預設 `auto_vad_hangover_frames = 8`）：

| `sample_rate` | `hop` | 生效幀數 | 實際時間長度 |
|---:|---:|---:|---|
| 16000 | 128 | **10** | 80 ms |
| 16000 | 256 | **5** | 80 ms |
| 48000 | 512 | **8** | ≈ 85.3 ms |

**這個欄位沒有 read-back accessor。**
本層沒有提供 `audio_pipeline_4ch_..._vad_hangover_frames()` 之類的函式，
所以你只能用上面的公式自己算。（`bf_adapt_interval` 與 `bf_lambda` 兩個有，
`auto_vad_hangover_frames` 沒有 —— 這是目前 API 的不對稱之處。）

如果你走 `process_with_activity()`，這一項對你完全沒有影響。

#### 三個生效值的共同規則

* 都在 **init 時**算好並存起來。
* `reset()` **不會**重算它們（reset 只清 runtime 狀態）。
* 想改它們，唯一的方法是用新的 config 重新 init 一個 instance。

### 4.7 哪些欄位會改變記憶體用量

| 欄位 | 影響 `req.bytes` 嗎 |
|---|---|
| `core.sample_rate` / `core.fft_size` | 會（改 grid） |
| `core.filter_length` | 會（四路 AEC 等比縮放） |
| `core.delay_mode` / `core.delay_num_filters` / `core.fixed_delay_samples` | 會（見核心層手冊 §4.5/4.6a——僅省一份共用 estimator/ring，不是四份） |
| `core.max_delay_ms` | 會（僅 `MATCHED` 用於 delay ring 大小） |
| **`num_angles`** | **會，而且影響很大**（見下表） |
| 其他 DOA / BF / VAD 欄位 | 不變 |
| `geometry` 與座標 | 不變 |

ne10、`SIMD=1`、`delay_mode=MATCHED` 預設的單元 `req.bytes`
（`layout_version=21`；請以 `--print-mem-size` 重查；最後一欄標明每一列的來源）。單元**不含** RES、NR 與合成；
RES（`AecRes`）、NR、`AcIstft` 與增益／頻譜暫存（範例程式自己各自查大小後加總）另計，加起來是
`audio_pipeline_4ch_static --print-mem-size` 的 `Total:`：

| Config | 單元 `req.bytes` | RES + NR + 合成 + 暫存 | 合計 | 來源 |
|---|---:|---:|---:|---|
| 16000，全預設（256/128，`num_angles=72`） | 1,698,864 | 66,688 | 1,765,552 | 實測 |
| 16000，`core.fft_size = 512` | 2,992,496 | 96,464 | 3,088,960 | 實測 |
| 16000，`num_angles = 360` | 4,700,976 | 66,688 | 4,767,664 | 單元實測（直接呼叫 `get_mem_requirements()`）；合計為相加 |
| 48000，全預設（1024/512，`num_angles=72`） | 6,300,432 | 225,056 | 6,525,488 | 實測 |

範例程式在單元之外另切的 `AecRes`：16 kHz/256 為 7,904 B、16 kHz/512 為 14,048 B、48 kHz/1024 為 29,408 B
（`aec_res_get_mem_size()`，已含在上表第三欄）。`core.enable_post = 0` 時單元的 `req.bytes`：16 kHz/256 為 1,693,648、
16 kHz/512 為 2,982,160、48 kHz/1024 為 6,279,856。

⚠ 覆蓋差異：只有 `num_angles = 72` 的 16 kHz/256 與 48 kHz/1024 兩組會被
C 關卡自動驗證（static smoke 各印一次 `Total:` bytes）。`core.fft_size = 512`、
`num_angles = 360` 與 `core.enable_post = 0` 的各列是**手動查詢**得到的，沒有任何自動測試會在這些
config 上呼叫 `audio_pipeline_4ch_get_mem_requirements()`，所以它們不會隨程式
改動自動失效——引用前請自己現查一次。

`num_angles` 從 72 調到 360，單元的記憶體約從 1.70 MB 變成約 4.70 MB（量級可信，
精確值見上）。**先確認你的角度解析度真的需要那麼細，再調這個值。**

換 backend、換編譯選項、更新 submodule 都會讓上表失效。
**實際配置一律以 `audio_pipeline_4ch_get_mem_requirements()` 現查的 `req.bytes`，加上你各後級 stage（RES 的 `AecRes`、NR、合成）自己查的大小為準。**

---

> [索引](integration_4ch_spatial_zh_TW.md) ｜ [上一份](integration_4ch_spatial_zh_TW_02_lifecycle.md) ｜ [下一份](integration_4ch_spatial_zh_TW_04_errors.md) <!-- split-nav -->
