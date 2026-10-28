導覽：[索引](integration_4ch_core_zh_TW.md) ｜ [上一部分](integration_4ch_core_zh_TW_03_config.md) <!-- split-nav -->
## 5. 錯誤處理

### 5.1 front → fuse 的次序規則

flow 沒有 token，只有一個「front frame 有效」旗標。次序規則很短：

| 情境 | 回傳 | 之後的狀態 |
|---|---|---|
| 正常 `mcab_process_front()` → `mcab_fuse()` | 兩者皆 `MCAB_OK`(0) | front 視圖與 frame 有效至下一次 `mcab_process_front()`／`mcab_reset()`／`mcab_destroy()` |
| 連續兩次 `mcab_process_front()`（不 fuse） | 兩次皆 `MCAB_OK` | 前一幀的視圖作廢，這是正常用法（從不 fuse 的 front-only 應用每 hop 都這樣） |
| 建構或 `mcab_reset()` 之後、尚未 `mcab_process_front()` 就 `mcab_fuse()` | `MCAB_SEQUENCE_ERROR`(-2) | 不變 |
| `mcab_process_front()` 在接受引數後失敗（非有限輸入、`MCAB_DSP_ERROR`），之後 `mcab_fuse()` | `MCAB_SEQUENCE_ERROR`(-2) | 該次失敗清掉 front 有效旗標；沒有可 fuse 的幀 |
| `mcab_fuse()` 傳入全零 / 非有限的 weights | `MCAB_INVALID_ARGUMENT`(-1) | **front 幀保留**，你可以修正 weights 後重試 |
| `enable_post = 0` 的 flow 呼叫 `mcab_fuse()` | `MCAB_INVALID_ARGUMENT`(-1) | 不變（沒有 fuse 緩衝） |
| `mcab_process_front()` 或 `mcab_fuse()` 內部一致性檢查失敗 | `MCAB_DSP_ERROR`(-3) | **flow 已被自動 `mcab_reset()`**（連同四條 lane 與 estimator 自己的 reset），front 幀消失 |

**同一個 front frame 可以 fuse 多次**（例如以不同權重比較），每次都重新填 frame；但每次 fuse 都會覆寫上一次的 frame 內容。
想要 pipeline 化（beamformer 在另一顆核心上跑、排隊多幀）在這個 API 上是做不到的：視圖借用 lane 的 buffer，下一次 `mcab_process_front()` 就被覆寫。

### 5.2 `mcab_process_front()` 的完整拒絕清單（依實際檢查順序）

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `p` 為 NULL、`p` 已 destroy、`microphones_interleaved` / `far_reference` / `out` 任一為 NULL | `MCAB_INVALID_ARGUMENT`(-1) | 無 |
| 2 | `microphones_interleaved`（`hop*4` 個）或 `far_reference`（`hop` 個）含非有限值 | `MCAB_INVALID_ARGUMENT`(-1) | 逐階段耗時的前半四欄已清零；front 有效旗標已清；自適應狀態（lane、estimator、ring）沒動 |
| 3 | 內部 render 對齊失敗（例如目前採用的延遲超過 ring 能供應的範圍，如 `max_delay_ms = 0` 卻估到 608 samples（16 kHz 下約 38 ms）的回音延遲） | `MCAB_DSP_ERROR`(-3) | **flow 已被自動 reset** |
| 4 | 某條 lane 的輸出 context 不完整 | `MCAB_DSP_ERROR`(-3) | **flow 已被自動 reset** |

`mcab_process_front()` **不會**檢查 buffer 長度 —— `hop*4` 與 `hop` 是你的責任。

### 5.3 `mcab_fuse()` 的完整拒絕清單（依實際檢查順序）

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `p` 為 NULL、`p` 已 destroy、`weights` / `frame` 任一為 NULL，或 `cfg.enable_post == 0` | `MCAB_INVALID_ARGUMENT`(-1) | 無 |
| 2 | 之前沒有有效的 front frame | `MCAB_SEQUENCE_ERROR`(-2) | 無 |
| 3 | `weights` 含非有限值，或 `Σ(|re| + |im|)` ≤ `1e-12`（等同全零）或溢位成非有限值 | `MCAB_INVALID_ARGUMENT`(-1) | **front 幀保留**，可修正後重試 |
| 4 | 內部融合/一致性檢查失敗：`beamformed_error` 非有限、四條 lane 的 far 功率或 far 頻譜不一致，或投影溢位（權重有限但極大，投影結果非有限） | `MCAB_DSP_ERROR`(-3) | **flow 已被自動 reset** |

**序列檢查排在 weights 檢查之前**：沒有 front frame + weights 也壞掉 → 你拿到的是 `-2`。

### 5.4 錯誤語意總表

| Function | 成功 | 失敗 | 備註 |
|---|---|---|---|
| `mcab_default_config()` | 回傳填好的 struct | 不會失敗，**也不驗證** | 傳入非法 `sample_rate` 也照填，之後才會被拒 |
| `mcab_apply_lane_contract()`、`mcab_lane_config()`、`mcab_delay_estimator_config()` | `0` | `-1` | 無效 config（`mcab_apply_lane_contract()` 另含別的取樣率的 lane）；失敗時輸出參數不變 |
| `mcab_get_mem_requirements()` | `0`，`*out` 填妥 | `-1` | 另含編譯 `mcab.c` 時沒帶 `-DAUDIO_PIPELINE_BACKEND_STR="ne10"` |
| `mcab_get_mem_breakdown()` | `0`，`*out` 填妥 | `-1` | — |
| `mcab_init()` | 非 NULL handle | `NULL` | 等同 `mcab_init_ex(..., NULL)` |
| `mcab_init_ex()` | 非 NULL handle | `NULL` | 8 項 descriptor 檢查、config、lane／estimator 檢查任一不符即 NULL，pool 不動 |
| `mcab_create()` | 非 NULL handle | `NULL` | config 無效、lane／estimator 不符或配置失敗 |
| `mcab_process_front()` | `MCAB_OK`(0) | `-1` / `-3` | 見 §5.2 |
| `mcab_fuse()` | `MCAB_OK`(0) | `-1` / `-2` / `-3` | 見 §5.3 |
| `mcab_reset()` | `void` | 不會失敗 | `p == NULL` 或已 destroy → 靜默 no-op |
| `mcab_destroy()` | `void` | 不會失敗 | 見 §5.6 |
| `*_hop_size` / `*_fft_size` / `*_n_freqs` / `*_sample_rate` / `*_enable_post` | 正值（`enable_post` 為 `0`／`1`） | `-1` | `p == NULL` 或已 destroy |
| `*_matched_filter_count` / `*_linear_aec_count` | `0`／`1` / `4` | `0` | `p == NULL` 或已 destroy |
| `mcab_far_fft_real_compute_count()` | 累計值（`long`） | `0` | `p == NULL` 或已 destroy |

> **注意慣例不一致**：本層的 `mcab_get_mem_requirements()` / `mcab_get_mem_breakdown()` 用
> **0 = 成功 / −1 = 失敗**；但下層模組的 `aec_get_mem_size()` /
> `anr_get_mem_size()` / `fft_get_mem_size()` 是**回傳 0 代表失敗**。
> 另外，形狀 accessor 用 `-1` 表示無效 handle，而結構稽核 accessor 用 `0`。

### 5.5 診斷輸出去向

**這個模組在任何拒絕路徑上都不輸出任何訊息，也不連結任何 stdio 符號。**
（`mcab.c` 完全沒有 `printf` 家族的呼叫；在 `pipelines/4ch_aec_bf_nr_res/` 下執行的 `make NO_STDIO=1 audit-no-stdio` 以 `nm` 逐一檢查 `libaudio_pipeline_mcab.a`、
其他 archive 與兩個 sample object 都沒有 stdio 符號。）

也就是說：`mcab_init_ex()` 回 `NULL` 時你**不會**知道是哪一項不符。
板端 bring-up 時請自行在呼叫端逐欄位比對 `expected` 與剛查到的 `req`，
再逐條檢查 lane 與 estimator（hop／bin 數、是否為 `mcab_lane_config()`／`mcab_apply_lane_contract()` 建的、是否新建或已 reset、
estimator 的 bank 大小與取樣率），把差異印出來。

`NO_STDIO=1` 對這個模組沒有任何影響（本來就沒有 stdio）。

### 5.6 `mcab_destroy()` 的兩種語意

| Instance 來源 | `mcab_destroy()` 行為 |
|---|---|
| `mcab_init()` / `mcab_init_ex()`（caller-owned pool） | **不釋放**你的 pool。NULL 安全、**冪等**，重複呼叫安全（已實測）。之後由你自己交還或重用 pool |
| `mcab_create()`（heap） | 釋放 `mcab_create()` 配置的那一塊（mcab 自己的 pool）。遵循一般 `free()` 語意：**只能呼叫一次** |

兩種都**不碰** lane 與 estimator：`aec_destroy()` 與它們記憶體的釋放是你的事，順序是先 `mcab_destroy()`、再 lane。
destroy 之後不要再呼叫任何 accessor（caller-owned pool 的 instance 在 destroy 後 accessor 會回報無效值，但 `mcab_create()` 的 instance 已被釋放，不可再碰）。

---

## 6. 出貨內容 vs 範例

`pipelines/4ch_aec_bf_nr_res/` 目錄不是自解釋的。

### 6.1 函式庫（會進產品）

| Source | 產出 | 說明 |
|---|---|---|
| `pipelines/4ch_aec_bf_nr_res/mcab.c` | `libaudio_pipeline_mcab.a`（唯一的 object `mcab.o`） | **本手冊的主體**：flow 層，不含 lane、estimator、SRP／BF、RES／NR |
| `pipelines/4ch_aec_bf_nr_res/mcab.h` | — | 本手冊的公開 API |
| `pipelines/4ch_aec_bf_nr_res/mcab_internal.h` | — | 內部 header，**不是**公開 API，不要 include |

sample 層檔案（範例碼，**不在** archive 內，沒有 ABI、沒有 layout version，可整份複製改寫）：`mcab_sample_flow.{h,c}`（應用端 init 順序的實作：
四條 lane、estimator、mcab 的 pool 放進同一塊記憶體）、`mcab_sample_beam.{h,c}`（SRP-PHAT ＋ BF ＋ 備援 VAD 的 helper，另一份手冊）、
`mcab_sample_driver.{h,c}`（範例的命令列與 hop 迴圈骨架）、`mcab_sample_post.h`（header-only：`mcab_fuse()` 之後與後級函式庫之間的銜接，
`mcab_sample_res_input()` 填 `AecResInput`、`mcab_sample_anr_gain()` 呼叫降噪器並可折入 `r2`，以及 `MCAB_SAMPLE_CNG_NR_GAIN_FLOOR`、`MCAB_SAMPLE_CNG_RNG_SEED`、`MCAB_SAMPLE_PSD_SCALE_INV` 三個常數）。`mcab_sample_flow.o` 與 `mcab_sample_beam.o` 由 `make libs` 編出。

### 6.2 參考執行檔（全部都有 `main()`，**沒有任何一個會進產品**）

| Source | 是什麼 | 為什麼存在 |
|---|---|---|
| `pipelines/4ch_aec_bf_nr_res/mcab_external_static.c`（執行檔 `mcab_external_static`） | caller-owned pool 路徑的主機參考程式 | 示範 `aec_config_from_level`（balanced）＋ `mcab_apply_lane_contract` → `aec_init` ×4 → `delay_aec_init` → `mcab_init_ex`（`flow_plan()`／`flow_init()`）→ `mcab_process_front` → `mcab_fuse` → 明寫的 stage 串接（`post_hop()`，它自己建一個 `AecRes`） → 拆除（`flow_destroy()`）→ release。它在 front/fuse 之間塞的是**固定等權重**，那是決定性的 smoke adapter，**不是** production beamformer |
| `pipelines/4ch_aec_bf_nr_res/mcab_static.c`（`mcab_static`） | 同一個 flow 加 sample 層 SRP-PHAT／BF helper、RES（libaec `AecRes`）／NR／合成的 pool 路徑主機參考程式 | 屬於另一份手冊；列在這裡是為了讓目錄內容一目了然 |
| `pipelines/4ch_aec_bf_nr_res/mcab_raw.c`（`mcab_raw`） | raw-float 錄音驗證 runner（flow 加 helper 加 RES／NR／合成） | 主機一次性工具；四條 lane、estimator、mcab 的 pool 與 helper 放在一塊 `posix_memalign` 配置的記憶體，後級 stage 另一塊 |

另有兩支組合了這個 flow 與 helper 的範例，檔案在各自的 family 目錄、由本目錄的 Makefile 建置（`make 4ch_aec_bf_dfn_res`、`make 4ch_alignulcnet`）：
`pipelines/dfn/4ch_aec_bf_dfn_res/main.c`（DFN2 後級）與 `pipelines/ulcnet/4ch_alignulcnet/main.c`（Align-ULCNet 後級，`enable_post = 0`）。

單聲道的 library 與參考程式（`pipelines/mono_aec_nr_res/main.c`、`example_board_adapter.c`）見 `docs/integration_mono_zh_TW.md`。

### 6.3 其他（不出貨）

* `pipelines/4ch_aec_bf_nr_res/tests/` —— 測試：`test_mcab.c`（本手冊 API 的測試）、`test_mcab_sample_beam.c`（helper 與整條範例鏈）、
  `test_mcab_sample_driver.c`；`tests/conv_chain_mcab.h` 是把後級串接寫在 flow 之上的參考鏈；`tests/mcab_test_rig.h` 是測試用的建立／拆除輔助。
  `tests/test_mcab_sample_post.c`（`mcab_sample_post.h` 的測試）由上一層 `pipelines/Makefile` 建置（`make -C pipelines test_mcab_sample_post`）。
* `audiolib/{doa,bf}` 與 `libspatial_common.a` —— SRP-PHAT／BF／spatial kernel 函式庫，只有 sample 層 helper 需要。
  **只用本手冊這套 API 的話完全不需要它們。**
* `.py` 檔 —— 主機端評估與參考實作，不是 C 產品的一部分。

---

## 7. 整合檢查清單

建置與連結：

- [ ] 四個 archive（`libaudio_pipeline_mcab.a` / `libaec.a` / `libanr.a` /
      `libaudio_common.a`）是用**同一組** backend 與編譯選項建出來的
      （只用流程、不串 NR 時 flow 只連 `libaudio_pipeline_mcab.a` + `libaec.a` + `libaudio_common.a`；`libanr.a` 只給有串 NR 的程式）
- [ ] link 順序照 [§1.5](integration_4ch_core_zh_TW_01_library_quickstart.md#15-archive-link-order)；沒有多連 `libdoa.a` / `libbf.a` / `libspatial_common.a`
- [ ] `BACKEND=ne10` 時最終 link 用 C++ driver
- [ ] archive 路徑是用 `print-lib-path` 查的
- [ ] 沒有 include `mcab_internal.h`

初始化：

- [ ] 每次 init 之前都重新呼叫 `mcab_get_mem_requirements()`，lane 與 estimator 也各自重查 `get_mem_size`
- [ ] 四條 lane 是**四個不同**的 `Aec`，config 由 `mcab_lane_config()`（或 `aec_config_from_level()` ＋ `mcab_apply_lane_contract()`）產生（`aec_config_from_level()` 對 8000／16000／48000 以外的取樣率回傳 `fft_size`、`filter_length` 皆為 0 的設定，`aec_get_mem_size()` 回 0、`aec_init()` 回 NULL）；MATCHED 時給 `McabLibs.delay`，其他模式給 `NULL`
- [ ] lane 與 estimator 是剛建好的，或已經 reset 過（`aec_reset()`／`delay_aec_reset()`，或先前那個 flow 的 `mcab_reset()`）
- [ ] pool 對齊 16、大小 ≥ `req.bytes`
- [ ] 用 `mcab_init_ex()` 並把剛查到的 `req` 當 `expected` 傳回去
- [ ] `mcab_init_ex()` 回 `NULL` 時你的程式會停下來 —— 而且你**自己**在呼叫端印出了
      是哪一個欄位不符（函式庫不會告訴你）
- [ ] `sample_rate` 沒有設成 8000
- [ ] 後級 stage 的記憶體以各自的 `mem_size` 查大小，16-byte 對齊，與 mcab 的 pool 分開

front / fuse 協定：

- [ ] 每 hop 先 `mcab_process_front()` 再 `mcab_fuse()`；`mcab_fuse()` 之前沒有 front frame 會得到 `-2`
- [ ] `weights` 是 channel-major `[ch * n_freqs + bin]`，不是 bin-major
- [ ] `weights` 不會全零（會被 `-1` 拒絕；front 幀保留，可重試）
- [ ] 有處理 `-3`（flow 已被自動 reset，包含四條 lane 與 estimator；你的外部狀態與後級 stage 也要跟著 reset：`aec_res_reset()`／`anr_reset()`／`ac_istft_reset()` 與 CNG rng）
- [ ] flow 掛著的期間，沒有對 lane 或 estimator 自己呼叫處理或 reset 函式
- [ ] `front.delay.changed` 非 0 時，你外部 beamformer 的 STFT / OLA 歷史有跟著清掉
- [ ] `McabFront` 與 `McabFrame` 的指標只在下一次 `mcab_process_front()`／`mcab_reset()`／`mcab_destroy()` 之前讀取，沒有跨幀保存，也沒有寫入
- [ ] `mcab_reset()` 之後你的 beamformer 立刻停止讀那些指標

執行期：

- [ ] `microphones_interleaved` 是 `[sample * 4 + channel]` interleaved，長度 `hop * 4`
- [ ] mic / ref 內容都是有限值（NaN/Inf 會被 `-1` 拒絕）
- [ ] hop / n_freqs 是查來的，不是寫死的
- [ ] 同一個 instance 沒有被多 thread 同時處理
- [ ] 鏈的順序：`mcab_sample_res_input` → `aec_res_process` → `mcab_sample_anr_gain` → `sk_min_f32` → `sk_capply_gain_f32` → `aec_cng_fill` → `ac_istft_push`；缺席的增益來源用 unity 全 1 向量

收尾：

- [ ] 先 `mcab_destroy()`，再 `aec_destroy()` 四條 lane，最後才釋放記憶體
- [ ] `mcab_create()` 建的 instance 只 `mcab_destroy()` 一次
- [ ] destroy 之後沒有再呼叫任何 accessor

---

## 8. 版本與相容性

### 8.1 `McabMemReq`

固定 32 bytes、每個欄位固定 byte offset（由 header 內的 `_Static_assert` 釘死），
全部是定寬整數。它只描述 **mcab 自己的 pool**（`MCAB_LAYOUT_VERSION` 涵蓋的 carve：控制區塊、mic lane、aligned far、fuse 緩衝、alignment ring）；
lane、estimator 和 sample 層 helper 的 pool 不在其中；`aec.h`／`delay_aec.h` 沒有 descriptor 型別，大小只由各自的 `get_mem_size` 回報。

| Offset | 欄位 | 型別 | 目前值 |
|---:|---|---|---|
| 0 | `descriptor_version` | `uint32_t` | `1` |
| 4 | `layout_version` | `uint32_t` | `26` |
| 8 | `backend_id` | `uint32_t` | `2` = NE10（目前唯一的 FFT 後端）；永遠不會是 0（`1` 也不會回傳） |
| 12 | `build_flags_hash` | `uint32_t` | FNV-1a-32，隨 build 變動 |
| 16 | `alignment` | `uint32_t` | `16` |
| 20 | `reserved` | `uint32_t` | `0`（必須為 0，`mcab_init_ex()` 會驗證） |
| 24 | `bytes` | `uint64_t` | mcab pool 總需求 |

### 8.2 `mcab_init_ex()` 的 8 項檢查

`expected == NULL` 時完全等同 `mcab_init()`。
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

另外，不論有沒有給 `expected`，下列檢查也會讓 `mcab_init_ex()` 回 `NULL`，且 pool 一個 byte 都不寫：pool 未 16-byte 對齊、
config 無效、`libs` 為 NULL 或其中的 lane／estimator 不符（見 [§3 第 5 步](integration_4ch_core_zh_TW_02_lifecycle.md#3-lifecycle)）。
`layout_version` 只認 25：較早的版號（17～24；21～24 屬於 mcab 的前身層）一律被拒，即使 `bytes` 剛好夠
（`tests/test_mcab.c` 的 `run_static_parity` 逐一驗過 17～24）。

### 8.3 序列化

可以逐 byte 拷貝到檔案 / flash / 訊息 buffer 再讀回來，**但僅限相同 endianness**。
不提供 byte-swap 輔助，跨 endianness 交換不在支援範圍。

### 8.4 相容性規則（給整合者的三條）

1. **descriptor 永遠現查現用**。不要存進 NVRAM、不要跨 firmware rebuild 沿用。
2. **`backend_id` 不是 2 的 descriptor 一律不相容**，即使 `bytes` 相同。
3. **升級 library 之後一定要重跑一次 `mcab_get_mem_requirements()`**（lane 與 estimator 也重查各自的 `get_mem_size`），並確認 pool 預算仍足夠。

導覽：[索引](integration_4ch_core_zh_TW.md) ｜ [上一部分](integration_4ch_core_zh_TW_03_config.md) <!-- split-nav -->
