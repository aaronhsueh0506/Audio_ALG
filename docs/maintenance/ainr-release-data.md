# AINR 發布資料（DFN2 full／lite）

客戶介面只有 `AinrConfig { sr, fft, hop_in }`。下列資料屬於產品建置與發布，不放客戶頁面。來源是 `stream/ainr/ainr_product.h` 與 `stream/ainr/ainr.c` 的登錄表；本筆記只做索引，數值以原始碼為準。

## 產品綁定

- 一個產品只含一種 DFN2：full 或 lite（archive 以 `DFN2_MASK_ONLY` 區分，`make -C stream/ainr MASK_ONLY=1 lib` 為 lite）。
- 綁定的模型是編譯期常數 `AINR_PRODUCT_MODEL_ID`（`ainr_product.h`）。未指定時 full archive 取 `AINR_MODEL_ID_DFN2_FULL`，lite archive 取 `AINR_MODEL_ID_DFN2_LITE`。產品要綁量化替代版時，編譯 `ainr.c` 時加 `-DAINR_PRODUCT_MODEL_ID=<ID>`。
- 綁定的 ID 不屬於該 archive 時，每個 `ainr_get_mem_size()`／`ainr_init()` 都回 `AINR_ERR_MODEL_ID`。

| ID 常數 | 值 | 量化 | 登錄狀態 |
|---|---:|---|---|
| `AINR_MODEL_ID_DFN2_FULL` | 1 | 16x16（預設） | 已登錄 |
| `AINR_MODEL_ID_DFN2_FULL_ALT1` | 10 | 8x8 | 登錄列已註解；所有組態以 `AINR_ERR_GRID` 被拒 |
| `AINR_MODEL_ID_DFN2_FULL_ALT2` | 11 | 16x8 | 已登錄 |
| `AINR_MODEL_ID_DFN2_LITE` | 2 | 16x16（預設） | 已登錄 |
| `AINR_MODEL_ID_DFN2_LITE_ALT1` | 20 | 8x8 | 同 full 的 ALT1 |
| `AINR_MODEL_ID_DFN2_LITE_ALT2` | 21 | 16x8 | 已登錄 |

`ainr_product.h` 註明 ID 數值在發布紀錄指定正式值之前是暫定。量化名稱只出現在 `ainr.c`，不寫進公開標頭。`make -C stream/ainr test-ainr-product`（`MASK_ONLY=0` 與 `1` 各跑一次）涵蓋 ALT2 的 graph 路徑與預設的不同、ALT1 的每個組態都被 `ainr_get_mem_size()` 與 `ainr_init()` 以 `AINR_ERR_GRID` 拒絕（init 不寫輸出）、另一個 variant 的 ID 全部回 `AINR_ERR_MODEL_ID`。

## 登錄表的模型欄位

這些欄位由 checkpoint 決定，每個登錄列帶一組：

| 產品 | sr／fft | n_erb | df_bins | df_order | mask lookahead | df lookahead | 每包 N 的登錄 |
|---|---|---:|---:|---:|---:|---:|---|
| full | 16000／256 | 24 | 96 | 5 | 0 | 0 | 1、8 |
| full | 48000／1024 | 32 | 96 | 5 | 1 | 1 | 1 |
| lite | 48000／1024 | 32 | 96 | 0 | 1 | 0 | 1 |

`mask lookahead` 大於 0 時 N 只能是 1（`AINR_ERR_LOOKAHEAD_BLOCK_UNSUPPORTED`），所以 48 kHz 的 full 與 lite 沒有 N>1 的列。登錄表中 `atten_lim_db` 與 HPF 欄位在所有列都是 0／關閉。

## Graph 與 ERB 檔

- 登錄表的每個 N 帶一個 graph 路徑；路徑為 NULL 的 N 視為未交付，回 `AINR_ERR_FRAMES_PER_CALL`，不會改用別的 N 的圖。
- graph 路徑由三段組成：基底目錄（`AINR_MODEL_DIR_BASE`）、量化子目錄、產品子目錄（full 與 lite 各一），檔名依取樣率與 N 命名，形如 `<前綴>_model_<16k|48k>_n<N>.bin`。前綴與各列的完整路徑見 `ainr.c` 內的 `AinrDfn2FramesVariant` 陣列。`ainr.c` 的註解標明檔名為暫定，待正式命名慣例確認。
- ERB 矩陣檔依 `(sr, n_erb)` 命名，與量化無關，形如 `fb_<sr>_<n_erb>.bin`，放在 `AINR_ERB_MATRIX_DIR`（`ainr.c` 內有預設值，可在編譯時覆寫）。格式見 `docs/html/models/dfn2/reference/grid.html` 的 ERB 一節（`export_erb_matrix.py` 的 `erb_matrix.bin`）。
- 登錄的 graph 是否已交付、量化版本是否與該列的 grid 與 N 相符，本庫沒有核對紀錄。full 16 kHz／256／N=8 的路徑存在於登錄表；交付與是否相符待交付端確認。

## 驗收入口

```sh
make -C stream/ainr test                 # full：dfn2、spec-freq、work-shapes、tail、freq-nonfinite、state-generation、pool-gates、atten-lim、product、model-io、link-only
make -C stream/ainr MASK_ONLY=1 test     # lite：同上，去掉 dfn2 與 spec-freq
```

這些測項以主機與替身加速器執行，不涉及正式 graph、SDK 或目標板。
