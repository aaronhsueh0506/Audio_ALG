# SRP／BF 與四路 AEC 設定

## 4. Config 完整參考

設定分兩組：array 的 `AecConfig`（`n_mics = 4`，`aec.h`；格點、共用 delay 的對齊方式、`residual`、level 與 lane 參數都在其中，欄位逐項見 [config.md §4.1](config.md#41-逐欄位)），
以及 helper 的 `ArraySampleBeamParams`（23 個欄位：幾何 6、DOA 7、BF 7、內建 VAD 3，`array_sample_beam.h`）。
NR 與 comfort noise 的設定**不在**這裡（NR 屬於你的後級串接；comfort noise 與 RES 增益由 array 的 `AecConfig` 決定）。取得預設值的方式：

```c
AecConfig cfg;
array_sample_default_config(&cfg, 16000);        /* 或 aec_config_from_level() + n_mics = AEC_ARRAY_MICS */
ArraySampleBeamParams bp = array_sample_beam_default_params();
/* 之後只覆寫你真正要改的欄位 */
```

`array_sample_default_config()` 與 `array_sample_beam_default_params()` **不做驗證**，只是填值。

### 4.1 array 的 `AecConfig`

`aec_get_mem_requirements()`、`aec_init_ex()` 與 `array_sample_beam_init()` 都會驗證 `cfg`，規則完全一樣，完整欄位表見 [config.md §4.1](config.md#41-逐欄位)。helper 用到的部分：

| 欄位 | 預設（`array_sample_default_config()`） | 合法值 | 與 helper 的關係 |
|---|---|---|---|
| `sample_rate`、`fft_size` | 傳入的 rate、該 rate 的預設 fft | 16000/256、16000/512、48000/1024 三組 | helper 由這個格點導出 SRP 與 BF 的設定（`array_sample_spatial_config()`） |
| `n_mics` | `AEC_ARRAY_MICS` | 4 | helper 只接四麥克風 array |
| `residual` | `AEC_RESIDUAL_FULL` | `STATE`／`ESTIMATE`／`FULL` | `STATE`：`array_sample_beam_fuse()` 回 `AEC_ERR_MODE`（給直接吃 BF 頻譜的 ULCNet）；其他：fuse 可用 |
| `delay_mode`、`delay_num_filters`、`fixed_delay_samples`、`capture_proxy_channel`、quarantine 兩欄 | MATCHED、5、-1、0、關 | 見 config.md | 決定共用 aligner；與 helper 無關 |
| `level`、`enable_cng` | balanced、開 | 見 `aec.h` | `level` 決定增益級的地板；lane 的濾波器長度由取樣率決定（不是欄位） |

> Quarantine 窗只對「連續符合條件的 backward episode」有界。候選轉為 forward／無效、confidence 中斷，或 proxy lane 的 cancellation 證據消失時會解除本次隔離；
> 之後的新 backward episode 重新計時。Cancellation collapse 常見於路徑完全替換，但不是 multipath／新增路徑的必然特徵。**產品預設維持 `0`。**
> 打開之後只是把誤鎖往後推一個窗，不會修正誤鎖本身（那個誤鎖已在估計器端根治，見 `AEC/docs/delay_estimator_design_zh_TW.md` §2）。

`AecConfig.residual = AEC_RESIDUAL_STATE` 時只有四條 lane（沒有融合與增益級）。
`AecConfig` 沒有 NR 欄位（`nr_level`、`enable_nr`、`legacy_amin` 屬於你的 `AnrConfig` 與後級串接）。
濾波器長度不是 `AecConfig` 欄位：由取樣率決定（52 ms；取樣率 ≥ 44.1 kHz 為 64 ms），即 16 kHz → 832、48 kHz → 3072 樣本。

Grid 只有三組（`hop = fft_size / 2`、`n_freqs = fft_size / 2 + 1`）：

| `sample_rate` | `fft_size` | `hop` | `n_freqs` |
|---:|---:|---:|---:|
| 16000 | 256（rate 預設） | 128 | 129 |
| 16000 | 512 | 256 | 257 |
| 48000 | 1024（rate 預設） | 512 | 513 |

### 4.2 陣列幾何（4 + 2 欄）

| 欄位 | 型別 | 預設 | 合法值 | 你該怎麼設 |
|---|---|---|---|---|
| `geometry` | `ArraySampleGeometry` | `ARRAY_SAMPLE_GEOMETRY_UCA`（= 0） | `UCA`(0) / `ULA`(1) / `CUSTOM`(2)。其他值拒絕 | 依你的實體陣列選 |
| `uca_radius_m` | `float` | `0.035f` | **僅在 `geometry == UCA` 時檢查**：必須有限且 `> 0` | 圓陣半徑（公尺）。量你的板子 |
| `ula_spacing_m` | `float` | `0.035f` | **僅在 `geometry == ULA` 時檢查**：必須有限且 `> 0` | 線陣相鄰間距（公尺） |
| `microphone_x_m[4]` | `float[4]` | 依預設半徑 0.035 排出的 UCA 座標 | **僅在 `geometry == CUSTOM` 時檢查**：4 個都必須是有限值 | 每支麥克風的 x 座標（公尺） |
| `microphone_y_m[4]` | `float[4]` | 同上 | 同上 | 每支麥克風的 y 座標（公尺） |
| `speed_of_sound_m_s` | `float` | `343.0f` | 有限且 `> 0` | 一般不需要改 |

**重要陷阱**：`geometry == UCA` 時，實際幾何是**從 `uca_radius_m` 現算的**，
`microphone_x_m` / `microphone_y_m` 的內容**會被忽略**。
同理 `ULA` 只看 `ula_spacing_m`。只有 `CUSTOM` 才會讀那兩個陣列。
`array_sample_beam_default_params()` 之所以順手把兩個陣列填成預設 UCA 座標，是為了讓你直接切成
`CUSTOM` 時不會拿到未初始化的值 —— 但那組座標對應的是**預設半徑**，
不是你之後改過的 `uca_radius_m`。要用 `CUSTOM` 就自己把兩個陣列填滿。

### 4.3 Direction finding（DOA，7 欄）

| 欄位 | 型別 | 預設 | 合法值 | 你該怎麼設 |
|---|---|---|---|---|
| `num_angles` | `int` | `72` | `[4, 3600]` | 角度掃描解析度。**這是 helper 最大的記憶體旋鈕**（見 §4.7）。72 相當於每 5 度一格 |
| `doa_low_freq_hz` | `float` | `300.0f` | 有限、`>= 0`、且 **`<` Nyquist**（`sample_rate / 2`） | 定位用的頻帶下限 |
| `doa_high_freq_hz` | `float` | `7000.0f` | 有限、`>= 0`、`<=` Nyquist；解析後的上限（`> 0` 取自身，`0` 取 `min(7000, Nyquist)`）必須 `> doa_low_freq_hz` | 定位用的頻帶上限。**`0` 代表自動**，實際取 `min(7000, Nyquist)` |
| `doa_enable_smoothing` | `int`（bool） | `1` | `0` 或 `1` | `1` = 啟用角度平滑 |
| `doa_switch_consecutive` | `int` | `3` | **必須 `> 0`** | 要連續幾幀指向新方向才真的切換。調大 = 更遲鈍但更穩 |
| `doa_angle_tolerance_rad` | `float` | `10° ≈ 0.174533` rad | 有限且 `>= 0` | 多大的角度差才算「換方向」 |
| `doa_update_interval` | `int` | `2` | **必須 `> 0`** | 每幾個 hop 更新一次 DOA |

### 4.4 Beamformer（BF，7 欄）

| 欄位 | 型別 | 預設 | 合法值 | 你該怎麼設 |
|---|---|---|---|---|
| `bf_enable` | `int`（bool） | `1` | `0` 或 `1` | `0` = 關閉 BF：直接輸出 channel 0，並在每個 bin 送出單位權重 `[1,0,0,0]`，因此 lane 1–3 權重全零。全零權重的 lane 不參與融合的 `all_converged`／`usable_linear`／`max_dt`／`max_saturation` 縮減 |
| `bf_lambda` | `float` | `0.995f` | 有限，且 `> 0.0` 並 `<= 1.0` | 自適應的遺忘因子。**你設的值不是實際生效的值，見 §4.6**。依生效 interval retime 後若下溢為 0，整組參數被拒 |
| `bf_mu` | `float` | `0.1f` | 有限且 `>= 0` | 步長增益。**不會**被 retime，設多少就是多少 |
| `bf_fixed_mode` | `int`（bool） | `0` | `0` 或 `1` | `1` = 用固定方向，不吃 DOA 結果 |
| `bf_fixed_doa_rad` | `float` | `0.0f` | 有限值（不限範圍） | `bf_fixed_mode == 1` 時採用的固定方向（弧度） |
| `bf_fixed_align_notebook` | `int`（bool） | `0` | `0` 或 `1` | 固定對齊模式。**開啟時會強制改變 `bf_adapt_interval` 的生效值，見 §4.6** |
| `bf_adapt_interval` | `int` | `1` | **`[1, ARRAY_SAMPLE_BEAM_MAX_ADAPT_INTERVAL]`**（上限 `1 << 20`，讓 `hop × interval` 在每個格點都不超出 `int`） | 每幾個 hop 做一次自適應更新。**你設的值不一定是實際生效的值，見 §4.6** |

### 4.5 內建 fallback VAD（3 欄）

**這三個欄位只在你呼叫 `array_sample_beam_process()` 時才會被用到。**
如果你走 `array_sample_beam_process_with_activity()`（產品整合建議用這條），
活動偵測完全由你提供，這三個欄位不影響任何行為 —— 但**仍然會被驗證**，
所以還是要保持合法值（照 `array_sample_beam_default_params()` 留著即可）。

| 欄位 | 型別 | 預設 | 合法值 | 你該怎麼設 |
|---|---|---|---|---|
| `auto_vad_threshold_dbfs` | `float` | `-55.0f` | 必須是有限值（不限範圍） | 判定為語音的最低能量門檻（dBFS）。調高 = 更不容易誤判成語音 |
| `auto_vad_snr_ratio` | `float` | `3.0f` | 有限且 **`>= 1.0`** | 相對背景噪聲要高出幾倍才算語音 |
| `auto_vad_hangover_frames` | `int` | `8` | 必須 `>= 0` | 語音結束後再保持多少幀。**你設的值不是實際生效的值，見 §4.6** |

### 4.6 你設定的值不一定是實際生效的值

**這一節是本手冊最重要的一節。** 有三個欄位在 init 時會被靜默轉換。
你在 `ArraySampleBeamParams` 裡讀到的值，和 helper 內部真正使用的值**不同**。
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
int effective = array_sample_beam_bf_effective_adapt_interval(b);
```

#### (2) `bf_lambda` —— 依 hop × 生效 interval 重新定時

`bf_lambda` 是以「每 10 ms 更新一次」為基準調出來的係數。
真正的遺忘時間常數會隨 `hop_size` / `sample_rate` 改變，所以 init 時會做換算（`anr_retime_alpha()`）：

```
effective_lambda = configured_lambda ^ ( (hop_size * effective_interval / sample_rate) / 0.010 )
```

注意指數裡用的是 **(1) 算出來的 `effective_interval`**，不是你設的原始值。
（`bf_mu` 是步長增益、不是時間常數，所以**不做**這個換算。）

**讀回生效值**：

```c
float effective = array_sample_beam_bf_lambda(b);
```

對照（`bf_lambda = 0.995`；以上式用 float32 重算）：

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

這個欄位的單位是「以 10 ms 為一幀」的幀數，init 時會換算成這個 grid 的實際幀數（`anr_retime_frames()`）：

```
effective_frames = ceil( auto_vad_hangover_frames * sample_rate / (100 * hop_size) )
```

（`auto_vad_hangover_frames == 0` 時生效值也是 0，即關閉 hangover。）

對照（預設 `auto_vad_hangover_frames = 8`）：

| `sample_rate` | `hop` | 生效幀數 | 實際時間長度 |
|---:|---:|---:|---|
| 16000 | 128 | **10** | 80 ms |
| 16000 | 256 | **5** | 80 ms |
| 48000 | 512 | **8** | ≈ 85.3 ms |

**這個欄位沒有 read-back accessor。**
helper 沒有提供 `array_sample_beam_..._vad_hangover_frames()` 之類的函式，
所以你只能用上面的公式自己算。（`bf_adapt_interval` 與 `bf_lambda` 兩個有，
`auto_vad_hangover_frames` 沒有 —— 這是目前 API 的不對稱之處。）
fallback VAD 的噪聲 EMA 係數（0.999／0.95）同樣依 hop retime。

如果你走 `process_with_activity()`，這一項對你完全沒有影響。

#### 三個生效值的共同規則

* 都在 **init 時**算好並存起來。
* `reset()` **不會**重算它們（reset 只清 runtime 狀態）。
* 想改它們，唯一的方法是用新的 config 重新 init 一個 helper。

### 4.7 哪些欄位會改變記憶體用量

| 欄位 | 影響哪一塊 |
|---|---|
| `sample_rate` / `fft_size` | 全部（改 grid） |
| `delay_mode` / `delay_num_filters` / `fixed_delay_samples` | array pool 內的共用 estimator（僅 MATCHED）與對齊環；不是四份 |
| `residual` | array pool 內的融合緩衝與增益級（STATE 最小） |
| **`num_angles`** | **helper pool，而且影響很大**（steering table） |
| 其他 DOA / BF / VAD 欄位 | 不變 |
| `geometry` 與座標 | 不變 |

ne10、`SIMD=1`、`delay_mode=MATCHED`（`delay_num_filters` = 5）預設的實測（`4ch_static --print-mem-size` 輸出，以及 `aec_get_mem_requirements()`、
`array_sample_beam_get_mem_size()` 的直接查詢；`AEC_LAYOUT_VERSION` 3）。
**Array** = `aec_get_mem_requirements()` 的 `req.bytes`（四條 lane、共用 estimator、對齊環、融合緩衝與增益級，加 `Aec` 結構）；**helper** = `array_sample_beam_get_mem_size()`；
**Unit** = Array + helper（`4ch_static` 的 `Unit:`）；**後級** = NR + 合成 + 暫存（範例程式自己各自查大小後加總）。array 與 helper 都**不含**後級，後級另計：

| Config | Array | helper | Unit | 後級 | 合計（`Total:`） |
|---|---:|---:|---:|---:|---:|
| 16000，全預設（256/128，`num_angles=72`） | 700,480（lane ×4 570,880） | 792,192 | 1,492,672 | 57,168 | 1,549,840 |
| 16000，`fft_size = 512` | 1,091,072（lane ×4 937,664） | 1,569,920 | 2,660,992 | 79,264 | 2,740,256 |
| 16000，`num_angles = 360` | 700,480 | 3,794,304 | 4,494,784 | 57,168 | 4,551,952 |
| 48000，全預設（1024/512，`num_angles=72`） | 2,410,720（lane ×4 2,074,176） | 3,125,376 | 5,536,096 | 189,424 | 5,725,520 |

後級在 16 kHz/256 為 NR 44,256 + 合成 11,344 + 暫存 1,568 B，16 kHz/512 為 54,320 + 21,840 + 3,104 B，48 kHz/1024 為 140,416 + 42,832 + 6,176 B
（`anr_get_mem_size()` 等的實測；增益級在 array 內，不再另列）。`4ch_static` 的 CLI 沒有 `num_angles` 選項，
`num_angles = 360` 那列是在程式裡設 `bp.num_angles` 後呼叫 `array_sample_beam_get_mem_size()` 的結果，後級與合計欄用同格點的後級。
`residual = STATE`（ULCNet 範例的設定，`delay_num_filters = 5`）時 array 較小（16 kHz/256 為 594,944、16 kHz/512 為 883,136、48 kHz/1024 為 1,994,912），
Unit 為 1,387,136（16 kHz/256）、2,453,056（16 kHz/512）、5,120,288（48 kHz/1024）。

⚠ 上表是某一次建置的實測輸出，不是契約：這些 byte 數不會隨程式改動自動更新——引用前請自己現查一次。

`num_angles` 越大，helper 的 SRP steering table 越大（16 kHz/256 下 72 → 360，helper 由 792,192 增為 3,794,304 B）。
**先確認你的角度解析度真的需要那麼細，再調這個值。**

換 backend、換編譯選項、更新函式庫版本 都會讓上表失效。
**實際配置一律以 `aec_get_mem_requirements()` 與 `array_sample_beam_get_mem_size()` 現查的大小，加上你各後級 stage（NR、合成）自己查的大小為準。**

[返回本主題](README.md)
