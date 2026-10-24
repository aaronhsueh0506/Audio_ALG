導覽：[索引](integration_4ch_core_zh_TW.md) ｜ [上一部分](integration_4ch_core_zh_TW_03_config.md) <!-- split-nav -->
## 5. 錯誤處理

### 5.1 front → fuse 的次序規則

前端沒有 token，只有一個「front frame 有效」旗標。次序規則很短：

| 情境 | 回傳 | 之後的狀態 |
|---|---|---|
| 正常 `front_process()` → `fuse()` | 兩者皆 `OK`(0) | front 視圖與 frame 有效至下一次 `front_process()`／`reset()`／`destroy()` |
| 連續兩次 `front_process()`（不 fuse） | 兩次皆 `OK` | 前一幀的視圖作廢，這是正常用法（從不 fuse 的 front-only 應用每 hop 都這樣） |
| 建構或 `reset()` 之後、尚未 `front_process()` 就 `fuse()` | `SEQUENCE_ERROR`(-2) | 不變 |
| `front_process()` 在接受引數後失敗（非有限輸入、`DSP_ERROR`），之後 `fuse()` | `SEQUENCE_ERROR`(-2) | 該次失敗清掉 front 有效旗標；沒有可 fuse 的幀 |
| `fuse()` 傳入全零 / 非有限的 weights | `INVALID_ARGUMENT`(-1) | **front 幀保留**，你可以修正 weights 後重試 |
| `enable_post = 0` 的前端呼叫 `fuse()` | `INVALID_ARGUMENT`(-1) | 不變（沒有 fuse 緩衝） |
| `front_process()` 或 `fuse()` 內部一致性檢查失敗 | `DSP_ERROR`(-3) | **instance 已被自動 `reset()`**，front 幀消失 |

**同一個 front frame 可以 fuse 多次**（例如以不同權重比較），每次都重新填 frame；但每次 fuse 都會覆寫上一次的 frame 內容。
想要 pipeline 化（beamformer 在另一顆核心上跑、排隊多幀）在這個 API 上是做不到的：視圖借用 lane 的 buffer，下一次 `front_process()` 就被覆寫。

### 5.2 `four_aec_front_process()` 的完整拒絕清單（依實際檢查順序）

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `p` 為 NULL、`p` 已 destroy、`microphones_interleaved` / `ref` / `out` 任一為 NULL | `INVALID_ARGUMENT`(-1) | 無 |
| 2 | `microphones_interleaved`（`hop*4` 個）或 `ref`（`hop` 個）含非有限值 | `INVALID_ARGUMENT`(-1) | 逐階段耗時的前半已清零；front 有效旗標已清 |
| 3 | 內部 render 對齊失敗 | `DSP_ERROR`(-3) | **instance 已被自動 reset** |
| 4 | 某條 lane 的輸出 context 不完整 | `DSP_ERROR`(-3) | **instance 已被自動 reset** |

`front_process()` **不會**檢查 buffer 長度 —— `hop*4` 與 `hop` 是你的責任。

### 5.3 `four_aec_fuse()` 的完整拒絕清單（依實際檢查順序）

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `p` 為 NULL、`p` 已 destroy、`weights` / `frame` 任一為 NULL，或 `cfg.enable_post == 0` | `INVALID_ARGUMENT`(-1) | 無 |
| 2 | 之前沒有有效的 front frame | `SEQUENCE_ERROR`(-2) | 無 |
| 3 | `weights` 含非有限值，或 `Σ(|re| + |im|)` ≤ `1e-12`（等同全零） | `INVALID_ARGUMENT`(-1) | **front 幀保留**，可修正後重試 |
| 4 | 內部融合/一致性檢查失敗（含 `beamformed_error` 非有限、四條 lane 的 far 頻譜不一致） | `DSP_ERROR`(-3) | **instance 已被自動 reset** |

**序列檢查排在 weights 檢查之前**：沒有 front frame + weights 也壞掉 → 你拿到的是 `-2`。

### 5.4 錯誤語意總表

| Function | 成功 | 失敗 | 備註 |
|---|---|---|---|
| `four_aec_nr_res_default_config()` | 回傳填好的 struct | 不會失敗，**也不驗證** | 傳入非法 `sample_rate` 也照填，之後才會被拒 |
| `four_aec_nr_res_get_mem_requirements()` | `0`，`*out` 填妥 | `-1` | — |
| `four_aec_nr_res_get_mem_breakdown()` | `0`，`*out` 填妥 | `-1` | — |
| `four_aec_nr_res_init()` | 非 NULL handle | `NULL` | 等同 `init_ex(..., NULL)` |
| `four_aec_nr_res_init_ex()` | 非 NULL handle | `NULL` | 8 項檢查任一不符即 NULL |
| `four_aec_nr_res_create()` | 非 NULL handle | `NULL` | config 無效或配置失敗 |
| `four_aec_front_process()` | `FOUR_AEC_NR_RES_OK`(0) | `-1` / `-3` | 見 §5.2 |
| `four_aec_fuse()` | `FOUR_AEC_NR_RES_OK`(0) | `-1` / `-2` / `-3` | 見 §5.3 |
| `four_aec_nr_res_reset()` | `void` | 不會失敗 | `p == NULL` 或已 destroy → 靜默 no-op |
| `four_aec_nr_res_destroy()` | `void` | 不會失敗 | 見 §5.6 |
| `*_hop_size` / `*_fft_size` / `*_n_freqs` / `*_sample_rate` | 正值 | `-1` | `p == NULL` 或已 destroy |
| `*_matched_filter_count` / `*_linear_aec_count` | `0`／`1` / `4` | `0` | `p == NULL` 或已 destroy |
| `four_aec_nr_res_far_fft_real_compute_count()` | 累計值（`long`） | `0` | `p == NULL` 或已 destroy |

> **注意慣例不一致**：本層的 `get_mem_requirements()` / `get_mem_breakdown()` 用
> **0 = 成功 / −1 = 失敗**；但下層模組的 `aec_get_mem_size()` /
> `anr_get_mem_size()` / `fft_get_mem_size()` 是**回傳 0 代表失敗**。
> 另外，形狀 accessor 用 `-1` 表示無效 handle，而結構稽核 accessor 用 `0`。

### 5.5 診斷輸出去向

**這個模組在任何拒絕路徑上都不輸出任何訊息，也不連結任何 stdio 符號。**
（`4aec_nr_res.c` 完全沒有 `printf` 家族的呼叫；已用 `nm` 確認 archive 內無相關符號。）

也就是說：`init_ex()` 回 `NULL` 時你**不會**知道是 8 項裡的哪一項不符。
板端 bring-up 時請自行在呼叫端逐欄位比對 `expected` 與剛查到的 `req`，
把差異印出來。

`NO_STDIO=1` 對這個模組沒有任何影響（本來就沒有 stdio）。

### 5.6 `destroy()` 的兩種語意

| Instance 來源 | `destroy()` 行為 |
|---|---|
| `four_aec_nr_res_init()` / `init_ex()`（caller-owned pool） | **不釋放**你的 pool。NULL 安全、**冪等**，重複呼叫安全（已實測）。之後由你自己交還或重用 pool |
| `four_aec_nr_res_create()`（heap） | 釋放 `create()` 配置的那一塊。遵循一般 `free()` 語意：**只能呼叫一次** |

destroy 之後不要再呼叫任何 accessor。

---

## 6. 出貨內容 vs 範例

`pipelines/4ch_aec_bf_nr_res/` 目錄不是自解釋的。

### 6.1 函式庫（會進產品）

| Source | 產出 | 說明 |
|---|---|---|
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c` | `libaudio_pipeline_4ch.a`（object 之一） | **本手冊的主體** |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.c` | `libaudio_pipeline_4ch.a`（object 之二） | 線性 AEC + BF 單元（SRP-PHAT + BF；不含 RES／NR／ISTFT）。只用本手冊 API 時不會被拉進來 |
| `pipelines/common/se_config.c` | `libaudio_pipeline_4ch.a`（object 之三） | 共用組態；`se_stage.h` / `se_res_stage_gain()` 為 header-only |
| `pipelines/samples/se_anr_stage.h` | header-only 範例碼 | NR 級的 gain 呼叫（`se_anr_stage_gain()`）：沒有 ABI、沒有 layout version、不進任何 archive，可整份複製改寫 |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res.h` | — | 本手冊的公開 API |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch.h` | — | 另一份手冊的公開 API |
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res_internal.h`、`4aec_projection_kernels.h` | — | 內部 header，**不是**公開 API，不要 include |

### 6.2 參考執行檔（全部都有 `main()`，**沒有任何一個會進產品**）

| Source | 是什麼 | 為什麼存在 |
|---|---|---|
| `pipelines/4ch_aec_bf_nr_res/4aec_nr_res_static.c` | caller-owned pool 路徑的主機參考程式 | 示範 query → allocate → init → front_process → fuse → 明寫的 stage 串接（`post_hop()`，它自己建一個 `AecRes`） → destroy → release。它在 front/fuse 之間塞的是**固定等權重**，那是決定性的 smoke adapter，**不是** production beamformer |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_static.c` | 線性 AEC + BF 單元加 RES（libaec `AecRes`）／NR／合成的 pool 路徑主機參考程式 | 屬於另一份手冊；列在這裡是為了讓目錄內容一目了然 |
| `pipelines/4ch_aec_bf_nr_res/audio_pipeline_4ch_raw.c` | raw-float 錄音驗證 runner（單元加 RES／NR／合成） | 主機一次性工具，走 heap `create()` 路徑 |

單聲道的 library 與參考程式（`pipelines/mono_aec_nr_res/main.c`、
`static_main.c`、`example_board_adapter.c`）見 `docs/integration_mono_zh_TW.md`。

### 6.3 其他（不出貨）

* `pipelines/4ch_aec_bf_nr_res/tests/` —— 測試；`tests/conv_chain_4ch.h` 是把後級串接寫在前端之上的參考鏈。
* `audiolib/{doa,bf,common/utility}` —— DOA / BF / spatial kernel 函式庫。
  **只用本手冊這套 API 的話完全不需要它們。**
* `.py` 檔 —— 主機端評估與參考實作，不是 C 產品的一部分。

---

## 7. 整合檢查清單

建置與連結：

- [ ] 四個 archive（`libaudio_pipeline_4ch.a` / `libaec.a` / `libanr.a` /
      `libaudio_common.a`）是用**同一組** backend 與編譯選項建出來的
- [ ] link 順序照 [§1.5](integration_4ch_core_zh_TW_01_library_quickstart.md#15-archive-link-order)；沒有多連 `libdoa.a` / `libbf.a` / `libspatial_common.a`
- [ ] `BACKEND=ne10` 時最終 link 用 C++ driver
- [ ] archive 路徑是用 `print-lib-path` 查的
- [ ] 沒有 include `4aec_nr_res_internal.h` 或 `4aec_projection_kernels.h`

初始化：

- [ ] 每次 init 之前都重新呼叫 `four_aec_nr_res_get_mem_requirements()`
- [ ] pool 對齊 16、大小 ≥ `req.bytes`
- [ ] 用 `init_ex()` 並把剛查到的 `req` 當 `expected` 傳回去
- [ ] `init_ex()` 回 `NULL` 時你的程式會停下來 —— 而且你**自己**在呼叫端印出了
      是哪一個欄位不符（函式庫不會告訴你）
- [ ] `sample_rate` 沒有設成 8000
- [ ] 後級 stage 的記憶體以各自的 `mem_size` 查大小，16-byte 對齊，與前端 pool 分開

front / fuse 協定：

- [ ] 每 hop 先 `front_process()` 再 `fuse()`；`fuse()` 之前沒有 front frame 會得到 `-2`
- [ ] `weights` 是 channel-major `[ch * n_freqs + bin]`，不是 bin-major
- [ ] `weights` 不會全零（會被 `-1` 拒絕；front 幀保留，可重試）
- [ ] 有處理 `-3`（instance 已被自動 reset，你的外部狀態與後級 stage 也要跟著 reset：`aec_res_reset()`／`anr_reset()`／`ac_istft_reset()` 與 CNG rng）
- [ ] `front.delay.changed` 非 0 時，你外部 beamformer 的 STFT / OLA 歷史有跟著清掉
- [ ] `FourAecFront` 與 `SeSpecFrame` 的指標只在下一次 `front_process()`／`reset()`／`destroy()` 之前讀取，沒有跨幀保存，也沒有寫入
- [ ] `reset()` 之後你的 beamformer 立刻停止讀那些指標

執行期：

- [ ] `microphones_interleaved` 是 `[sample * 4 + channel]` interleaved，長度 `hop * 4`
- [ ] mic / ref 內容都是有限值（NaN/Inf 會被 `-1` 拒絕）
- [ ] hop / n_freqs 是查來的，不是寫死的
- [ ] 同一個 instance 沒有被多 thread 同時處理
- [ ] 鏈的順序：`se_res_stage_gain` → `se_anr_stage_gain` → `sk_min_f32` → `sk_capply_gain_f32` → `aec_cng_fill` → `ac_istft_push`；缺席的增益來源用 unity 全 1 向量

收尾：

- [ ] `destroy()` 在釋放 pool **之前**
- [ ] `create()` 建的 instance 只 `destroy()` 一次
- [ ] destroy 之後沒有再呼叫任何 accessor

---

## 8. 版本與相容性

### 8.1 `FourAecNrResMemReq`

固定 32 bytes、每個欄位固定 byte offset（由 header 內的 `_Static_assert` 釘死），
全部是定寬整數。與 `AudioPipeline4ChMemReq` 同形狀、同語意，但**版本號各自獨立**（單元內嵌這個核心，其 `build_flags_hash` 折入核心的 hash）。

| Offset | 欄位 | 型別 | 目前值 |
|---:|---|---|---|
| 0 | `descriptor_version` | `uint32_t` | `1` |
| 4 | `layout_version` | `uint32_t` | `24` |
| 8 | `backend_id` | `uint32_t` | `2` = NE10（目前唯一的 FFT 後端）；永遠不會是 0 |
| 12 | `build_flags_hash` | `uint32_t` | FNV-1a-32，隨 build 變動 |
| 16 | `alignment` | `uint32_t` | `16` |
| 20 | `reserved` | `uint32_t` | `0`（必須為 0，`init_ex()` 會驗證） |
| 24 | `bytes` | `uint64_t` | pool 總需求 |

### 8.2 `init_ex()` 的 8 項檢查

`expected == NULL` 時完全等同 `four_aec_nr_res_init()`。
`expected != NULL` 時，下列 8 項全部成立才會開始 carve；任一不符即回 `NULL`
（**不輸出任何訊息**）：

1. `descriptor_version` 相符
2. `layout_version` 相符
3. `backend_id` 相符
4. `build_flags_hash` 相符
5. `alignment` 相符
6. `reserved == 0`
7. `expected->bytes >= 目前 build 的需求`
8. `bytes`（實際交進來的 pool 大小）`>= 目前 build 的需求`

### 8.3 序列化

可以逐 byte 拷貝到檔案 / flash / 訊息 buffer 再讀回來，**但僅限相同 endianness**。
不提供 byte-swap 輔助，跨 endianness 交換不在支援範圍。

### 8.4 相容性規則（給整合者的三條）

1. **descriptor 永遠現查現用**。不要存進 NVRAM、不要跨 firmware rebuild 沿用。
2. **`backend_id` 不是 2 的 descriptor 一律不相容**，即使 `bytes` 相同。
3. **升級 library 之後一定要重跑一次 `get_mem_requirements()`**，並確認 pool 預算仍足夠。

導覽：[索引](integration_4ch_core_zh_TW.md) ｜ [上一部分](integration_4ch_core_zh_TW_03_config.md) <!-- split-nav -->
