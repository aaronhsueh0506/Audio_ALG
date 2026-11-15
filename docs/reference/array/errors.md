# 四麥克風 array 錯誤與相容性

## 5. 錯誤處理

回傳碼是 `AecStatus`（`aec.h`）：`AEC_OK` = 0、`AEC_ERR_ARG` = −1、`AEC_ERR_MODE` = −2、`AEC_ERR_SEQUENCE` = −3、`AEC_ERR_DSP` = −4。
`AEC_ERR_MODE` 在陣列上是「`AecConfig.residual` 不到這個呼叫需要的層級」（STATE 陣列呼叫 `aec_process_residual()`／`aec_get_residual()`／增益函式，
或 `aec_process()`）；`AEC_ERR_DSP` 是陣列內部一個階段拒絕了這個 hop，**整個物件已被自動 `aec_reset()`**。

### 5.1 linear → residual 的次序規則

陣列只有一個「linear 幀有效」旗標與一個「已融合」旗標。次序規則很短：

| 情境 | 回傳 | 之後的狀態 |
|---|---|---|
| 正常 `aec_process_linear()` → `aec_process_residual()` | 兩者皆 `AEC_OK`(0) | 視圖有效至下一次 `aec_process_linear()`／`aec_reset()`／`aec_destroy()` |
| 連續兩次 `aec_process_linear()`（不 residual） | 兩次皆 `AEC_OK` | 後一次取代待處理的前一幀，前一幀的視圖作廢。這是正常用法（STATE 陣列每 hop 都這樣） |
| 建構或 `aec_reset()` 之後、尚未 `aec_process_linear()` 就 `aec_process_residual()` | `AEC_ERR_SEQUENCE`(−3) | 不變 |
| `aec_process_linear()` 在接受引數後失敗（`AEC_ERR_DSP`），之後 `aec_process_residual()` | `AEC_ERR_SEQUENCE`(−3) | 陣列已被自動 `aec_reset()`；沒有可融合的幀 |
| `aec_process_linear()` 因非有限輸入回 `AEC_ERR_ARG`（引數檢查階段被拒） | `AEC_ERR_ARG`(−1) | 什麼都不改：上一個 hop 的視圖仍可讀，之後的 `aec_process_residual()` 照常處理上一個待處理的幀 |
| `aec_process_residual()` 傳入全零 / 非有限的 weights | `AEC_ERR_ARG`(−1) | **linear 幀保留**，你可以修正 weights 後重試同一 hop |
| STATE 陣列呼叫 `aec_process_residual()` | `AEC_ERR_MODE`(−2) | 不變（沒有融合緩衝） |
| 同一 hop 第二次 `aec_process_residual()` | FULL：`AEC_ERR_SEQUENCE`；ESTIMATE：允許（無狀態的重新融合，例如另一組權重） | FULL 的增益級每 hop 只前進一次 |
| `aec_process_linear()` 或 `aec_process_residual()` 內部一致性檢查失敗 | `AEC_ERR_DSP`(−4) | **陣列已被自動 `aec_reset()`**（連同四條 lane、estimator 與增益級），linear 幀消失 |

想要 pipeline 化（beamformer 在另一顆核心上跑、排隊多幀）在這個 API 上是做不到的：視圖借用 lane 的 buffer，下一次 `aec_process_linear()` 就被覆寫。

### 5.2 `aec_process_linear()` 的完整拒絕清單（依實際檢查順序）

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `a` 為 NULL、`mics` 或 `ref` 為 NULL；陣列已 destroy | `AEC_ERR_ARG`(−1) | 無 |
| 2 | `mics`（`hop*4` 個）或 `ref`（`hop` 個）含非有限值 | `AEC_ERR_ARG`(−1) | 什麼都不改：耗時記錄、linear 有效旗標、上一個 hop 的視圖與自適應狀態（lane、estimator、ring）都保持原樣 |
| 3 | 內部 render 對齊失敗（防禦性檢查：ring 的容量是 estimator 可回報的最大延遲加 hop，採用的延遲不會超出） | `AEC_ERR_DSP`(−4) | **陣列已被自動 reset** |
| 4 | 某條 lane 的 linear 或殘餘階段拒絕（與單聲道相同的檢查：結果不是有限值，例如有限但極大的 mic／ref sample；`STATE` 也檢查），或其視圖綁定失敗 | `AEC_ERR_DSP`(−4) | **陣列已被自動 reset** |

`aec_process_linear()` **不會**檢查 buffer 長度 —— `hop*4` 與 `hop` 是你的責任。`aec_get_array()` 在沒有待處理的 linear 幀時回 `AEC_ERR_SEQUENCE`，`struct_size` 短於 `sizeof(AecArrayView)` 時回 `AEC_ERR_ARG`（`out` 保持原樣）。

### 5.3 `aec_process_residual()` 的完整拒絕清單（依實際檢查順序）

| # | 條件 | 回傳 | 副作用 |
|---|---|---|---|
| 1 | `a` 為 NULL；陣列已 destroy | `AEC_ERR_ARG`(−1) | 無 |
| 2 | `residual == AEC_RESIDUAL_STATE`（沒有融合緩衝） | `AEC_ERR_MODE`(−2) | 無 |
| 3 | `beam` 或 `beam->weights` 為 NULL | `AEC_ERR_ARG`(−1) | 無 |
| 4 | 之前沒有有效的 linear 幀 | `AEC_ERR_SEQUENCE`(−3) | 無 |
| 5 | FULL 且這一 hop 已經融合過 | `AEC_ERR_SEQUENCE`(−3) | 無 |
| 6 | `weights` 含非有限值，或 `Σ(|re| + |im|)` ≤ `1e-12`（等同全零）或溢位成非有限值（NEON 路徑另拒絕單一元素 `|re| + |im|` 超過 2^100 者）；`beamformed_error` 有任何非有限的 bin | `AEC_ERR_ARG`(−1) | **linear 幀保留**，可修正後重試；實例不 reset |
| 7 | 內部融合/一致性檢查失敗：四條 lane 的 far 功率或 far 頻譜不一致、投影溢位（權重有限但極大，投影結果非有限），或增益級拒絕 | `AEC_ERR_DSP`(−4) | **陣列已被自動 reset** |

**序列檢查排在 weights 檢查之前**：沒有 linear 幀 + weights 也壞掉 → 你拿到的是 `AEC_ERR_SEQUENCE`。

增益函式：`aec_apply_gain()`／`aec_apply_cng()` 在 `a` 已 destroy 或 `out`／`spec` 為 NULL 時回 `AEC_ERR_ARG`；陣列不是 FULL 時回 `AEC_ERR_MODE`；本 hop 還沒有融合時回 `AEC_ERR_SEQUENCE`。
`aec_get_residual()`：`struct_size` 短於 `sizeof(AecResidualView)` 時回 `AEC_ERR_ARG`（`out` 保持原樣）；STATE 回 `AEC_ERR_MODE`；沒有融合結果時回 `AEC_ERR_SEQUENCE`。

### 5.4 錯誤語意總表

各函式的成功／失敗值（`aec_get_mem_requirements()` 0／−1 與 `aec_get_mem_size()` 回 0 的慣例相反、`aec_init_ex()` 的 NULL、`aec_destroy()`、`aec_set_level()` …）
是函式庫的行為，見 AEC 的 [lifecycle.md §5 錯誤語意表](../../../../AEC/docs/reference/lifecycle.md#5-錯誤語意表)。陣列上與單支不同的只有：

* `aec_process()` 回 `AEC_ERR_MODE`（陣列要四路麥克風：用 `aec_process_linear()`）。
* `aec_process_linear()`／`aec_process_residual()` 的拒絕條件與回傳碼見上面的 §5.2、§5.3。
* `aec_apply_external_realign()` 回 `-1`：陣列在共用對齊改變時自己對四條 lane 做 realign。
* `aec_array_matched_filter_count()`／`aec_array_linear_aec_count()` 對 `NULL` 或已 destroy 的 `a` 回 `0`。

### 5.5 診斷輸出去向

**`libaec.a` 的陣列在任何拒絕路徑上都不輸出任何訊息。**（`pipelines/4ch/` 下執行的 `make NO_STDIO=1 audit-no-stdio` 以 `nm` 逐一檢查 sample object 與各 archive 都沒有 stdio 符號。）

也就是說：`aec_init_ex()` 回 `NULL` 時你**不會**知道是哪一項不符。板端 bring-up 時請自行在呼叫端逐欄位比對 `expected` 與剛查到的 `req`，
再逐欄檢查 config（取樣率與 fft 組合、delay 模式與欄位組合、`residual`、各範圍），把差異印出來。`NO_STDIO=1` 對陣列沒有任何影響。

### 5.6 `aec_destroy()` 的兩種語意

| Instance 來源 | `aec_destroy()` 行為 |
|---|---|
| `aec_init()` / `aec_init_ex()`（caller-owned pool） | **不釋放**你的 pool，只把 handle 標成已銷毀。NULL 安全、**冪等**，重複呼叫安全（已銷毀的 handle 之後被各入口拒絕）。之後由你自己交還或重用 pool |
| `aec_create()`（heap） | 先把 handle 標成不可用，再釋放 `aec_create()` 配置的那一塊（`Aec` 與 pool 同在這一塊）。**只能呼叫一次，指標之後作廢**（不要再 `aec_destroy()`、`free()` 或讀取；再次 `aec_create()` 回新的實例）|

destroy 之後，caller-owned pool 的 instance 的每個入口都拒絕該 handle（`AEC_ERR_ARG`／`-1`／`0`／NULL／清零的結構，`aec_hop_size()` 回 `0`），直到再次 init；`aec_create()` 的 instance 在 destroy 時已被釋放，之後不可再傳給任何入口。caller-owned pool 已被釋放或改寫時行為未定義，也不要保存 destroy 之前取得的視圖指標（包含 `aec_synthesis_window()` 的窗指標：窗在實例自己的記憶體內，借用它的 `AcIstft` 不可比 `Aec` 活得久）。
沒有 lane 或 estimator 要另外拆除：它們都在陣列的 pool 裡。

---

## 6. 出貨內容 vs 範例

`pipelines/4ch/` 目錄不是自解釋的。

### 6.1 函式庫（會進產品）

| Source | 產出 | 說明 |
|---|---|---|
| `AEC/c_impl/src/aec_array.c`（及 `aec.c` 的分派） | `libaec.a` | **本手冊的主體**：array instance，含 lane、共用 estimator、對齊環、融合與增益級；不含 SRP／BF、NR |
| `AEC/c_impl/include/aec.h` | — | 本手冊的公開 API（「Array instance」一節） |
| `AEC/c_impl/include/internal/aec_array_internal.h` | — | 內部 header，**不是**公開 API，不要 include |

`pipelines/4ch/` 本身沒有函式庫 archive。sample 層檔案（範例碼，**不在**任何 archive 內，沒有 ABI、沒有 layout version，可整份複製改寫）：
`array_sample_beam.{h,c}`（SRP-PHAT ＋ BF ＋ 備援 VAD 的 helper，另一份手冊；`array_sample_default_config()` 也在這裡）、
`array_sample_driver.{h,c}`（範例的命令列與 hop 迴圈骨架）、`array_sample_post.h`（header-only：`aec_get_residual()` 之後與降噪器之間的銜接，
`array_sample_anr_gain()` 呼叫降噪器並可折入 `r2`、`array_sample_post_reset_due()`，以及 `ARRAY_SAMPLE_PSD_SCALE_INV`）。`array_sample_beam.o` 由 `make libs` 編出。

### 6.2 參考執行檔（全部都有 `main()`，**沒有任何一個會進產品**）

| Source | 是什麼 | 為什麼存在 |
|---|---|---|
| `pipelines/4ch/4ch_external_static.c`（執行檔 `4ch_external_static`） | caller-owned pool 路徑的主機參考程式 | 示範 `aec_config_from_level`（balanced）＋ `n_mics = 4` → `aec_get_mem_requirements` → `aec_init_ex`（`flow_plan()`／`flow_init()`）→ `aec_process_linear` → `aec_process_residual` → 明寫的 stage 串接（`post_hop()`） → 拆除（`flow_destroy()`）→ release。它在 linear／residual 之間塞的是**固定等權重**，那是決定性的 smoke adapter，**不是** production beamformer |
| `pipelines/4ch/4ch_static.c`（`4ch_static`） | 同一個陣列加 sample 層 SRP-PHAT／BF helper、NR／合成的 pool 路徑主機參考程式 | 屬於另一份手冊；列在這裡是為了讓目錄內容一目了然 |
| `pipelines/4ch/4ch_raw.c`（`4ch_raw`） | raw-float 錄音驗證 runner（陣列加 helper 加 NR／合成） | 主機一次性工具；陣列與 helper 放在一塊 `posix_memalign` 配置的記憶體，後級 stage 另一塊 |

另有兩支組合了這個陣列與 helper 的範例，檔案在各自的 family 目錄、由本目錄的 Makefile 建置（`make 4ch_aec_bf_dfn_res`、`make 4ch_alignulcnet`）：
`pipelines/4ch/aec_bf_dfn_res/main.c`（DFN2 後級）與 `pipelines/4ch/alignulcnet/main.c`（Align-ULCNet 後級，`residual = AEC_RESIDUAL_STATE`）。

單聲道的 library 與參考程式（`pipelines/mono/aec_nr_res/main.c`、`example_board_adapter.c`）見 `docs/reference/mono/README.md`。

### 6.3 其他（不出貨）

* `pipelines/4ch/tests/` —— 測試：`test_4ch_chain.c`（鏈層級的 pool 位置不變性、reset 等價與 NR／RES bypass 契約）、`test_array_sample_beam.c`（helper 與整條範例鏈）、
  `test_array_sample_driver.c`；`tests/conv_chain_4ch.h` 是把後級串接寫在第二個陣列之上的參考鏈；`tests/array_test_rig.h` 是測試用的建立／拆除輔助。
  陣列本身的測試（配置、拒絕清單、融合、admission）在 AEC repo 的 `test/test_aec_array*.c`。
  `tests/test_array_sample_post.c`（`array_sample_post.h` 的測試）由上一層 `pipelines/Makefile` 建置（`make -C pipelines test_array_sample_post`）。
* `audiolib/{doa,bf}` 與 `libspatial_common.a` —— SRP-PHAT／BF／spatial kernel 函式庫，只有 sample 層 helper 需要。**只用本手冊這套 API 的話完全不需要它們。**
* `.py` 檔 —— 主機端評估與參考實作，不是 C 產品的一部分。

---

## 7. 整合檢查清單

建置與連結：

- [ ] 三個 archive（`libaec.a` / `libanr.a` / `libaudio_common.a`）是用**同一組** backend 與編譯選項建出來的
      （只用陣列、不串 NR 時連 `libaec.a` + `libaudio_common.a`；`libanr.a` 只給有串 NR 的程式）
- [ ] link 順序照 [integration.md §1.5](integration.md#15-archive-link-order)；沒有多連 `libdoa.a` / `libbf.a` / `libspatial_common.a`
- [ ] `BACKEND=ne10` 時最終 link 用 C++ driver
- [ ] archive 路徑是用 `print-lib-path` 查的
- [ ] 沒有 include `aec_array_internal.h`

初始化：

- [ ] 每次 init 之前都重新呼叫 `aec_get_mem_requirements()`；後級 stage 也各自重查 `get_mem_size`
- [ ] `n_mics = AEC_ARRAY_MICS`；`aec_config_from_level()` 對 8000／16000／48000 以外的取樣率回傳 `fft_size` 為 0 的設定，`aec_get_mem_requirements()` 回 `-1`、`aec_init()` 回 NULL
- [ ] `residual` 是 STATE／ESTIMATE／FULL 之一，且符合你的鏈（FULL 才有增益；STATE 不能呼叫 `aec_process_residual()`）
- [ ] pool 對齊 16、大小 ≥ `req.bytes`
- [ ] 用 `aec_init_ex()` 並把剛查到的 `req` 當 `expected` 傳回去
- [ ] `aec_init_ex()` 回 `NULL` 時你的程式會停下來 —— 而且你**自己**在呼叫端印出了是哪一個欄位不符（函式庫不會告訴你）
- [ ] `sample_rate` 沒有設成 8000
- [ ] 後級 stage 的記憶體以各自的 `mem_size` 查大小，16-byte 對齊，與陣列的 pool 分開

linear / residual 協定：

- [ ] 每 hop 先 `aec_process_linear()` 再 `aec_process_residual()`；residual 之前沒有 linear 幀會得到 `AEC_ERR_SEQUENCE`
- [ ] `AecArrayView`／`AecResidualView` 的 `struct_size` 在呼叫前設成 `sizeof(...)`
- [ ] `weights` 是 channel-major `[ch * n_freqs + bin]`，不是 bin-major
- [ ] `weights` 不會全零（會被 `AEC_ERR_ARG` 拒絕；linear 幀保留，可重試）
- [ ] 有處理 `AEC_ERR_DSP`（陣列已被自動 reset，包含四條 lane、estimator 與增益級；你的外部狀態與後級 stage 也要跟著 reset：`anr_reset()`／`ac_istft_reset()`，並更新你記的 `post_reset_serial`）
- [ ] 陣列存活期間，沒有對 lane 或 estimator 自己呼叫處理或 reset 函式（它們沒有公開 handle）
- [ ] `front.delay.changed` 非 0 時，你外部 beamformer 的 STFT / OLA 歷史有跟著清掉
- [ ] 視圖的指標只在下一次 `aec_process_linear()`／`aec_reset()`／`aec_destroy()` 之前讀取，沒有跨幀保存，也沒有寫入
- [ ] `aec_reset()` 之後你的 beamformer 立刻停止讀那些指標

執行期：

- [ ] `mics` 是 `[sample * 4 + channel]` interleaved，長度 `hop * 4`
- [ ] mic / ref 內容都是有限值（NaN/Inf 會被 `AEC_ERR_ARG` 拒絕）
- [ ] hop / n_freqs 是查來的，不是寫死的
- [ ] 同一個 instance 沒有被多 thread 同時處理
- [ ] 鏈的順序：`anr_process_gain`（4ch 鏈估計用 `error_spec`，即 beamformer 輸出 E；單聲道鏈用 `base_spec`）→ `aec_apply_gain` → `aec_apply_cng` → `ac_istft_push`；兩個 apply 都傳同一個 NR 增益；缺席的增益來源傳 NULL

收尾：

- [ ] 先拆你的 stage，再 `aec_destroy()`，最後才釋放記憶體
- [ ] `aec_create()` 建的 instance 只 `aec_destroy()` 一次，之後不 `free()`、不再讀取該指標
- [ ] destroy 之後沒有再呼叫任何 accessor

---

## 8. 版本與相容性

`AecMemReq`（32 bytes 的 descriptor）、`aec_init_ex()` 的八項 descriptor 檢查、序列化（同 endianness）與相容性規則
（descriptor 現查現用、`backend_id` 不符一律不相容、升級後重跑 `aec_get_mem_requirements()`）是函式庫的行為，
單支與 array 完全相同，見 AEC 的 [lifecycle.md §C](../../../../AEC/docs/reference/lifecycle.md#c-記憶池描述子與陣列實例aec_get_mem_requirements--aec_init_ex) 與 `aec.h`。
descriptor 只涵蓋 `libaec.a` 自己的 pool；後級 stage 與 sample 層 helper 的大小要各自重查。

[返回本主題](README.md)
