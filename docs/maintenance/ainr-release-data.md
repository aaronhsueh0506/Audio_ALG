# AINR 發布資料（DFN2 full／lite）

客戶介面是 `AinrConfig { sr, fft, hop_in, erb_fwd, erb_inv, n_erb }`：三個格點值加上該 grid 的 ERB 表與其 band 數 n_erb（表借用，由 sample 載入；n_erb 不等於 grid 的以 `AINR_ERR_ERB_TABLES` 拒絕）。下列資料屬於產品建置與發布，不放客戶頁面。來源是 `stream/ainr/ainr_product.h`、`stream/ainr/dfn2_grids.h`（綁定模型的 grid 表）與 `stream/ainr/graph_io_dfn2.h`（sample 的 graph 表）；本筆記只做索引，數值以原始碼為準。

## 產品綁定

- 一個產品只含一種 DFN2：full 或 lite（archive 以 `DFN2_MASK_ONLY` 區分，`make -C stream/ainr MASK_ONLY=1 lib` 為 lite）。
- 綁定的模型是編譯期常數 `AINR_PRODUCT_MODEL_ID`（`ainr_product.h`）。未指定時 full archive 取 `AINR_MODEL_ID_DFN2_FULL`，lite archive 取 `AINR_MODEL_ID_DFN2_LITE`；mask_lookahead 0 的 full 模型用 `-DAINR_PRODUCT_MODEL_ID=3`。
- 綁定的 ID 不屬於該 archive 時，每個 `ainr_get_mem_size()`／`ainr_init()` 都回 `AINR_ERR_MODEL_ID`。

| ID 常數 | 值 | 模型 |
|---|---:|---|
| `AINR_MODEL_ID_DFN2_FULL` | 1 | full，48 kHz 的 mask_lookahead 1 |
| `AINR_MODEL_ID_DFN2_LITE` | 2 | lite（mask-only），48 kHz 的 mask_lookahead 1 |
| `AINR_MODEL_ID_DFN2_FULL_L0` | 3 | full，所有 grid 的 mask_lookahead 0 |

`ainr_product.h` 註明 ID 數值在發布紀錄指定正式值之前是暫定。量化資料夾不是函式庫的選擇，由 sample 的 `SAMPLE_MODEL_DIR` 決定。`make -C stream/ainr test-ainr-product`（`MASK_ONLY=0` 與 `1` 各跑一次）涵蓋產品綁定：另一個 variant 的 ID 全部回 `AINR_ERR_MODEL_ID`。

## Grid 表的模型欄位

這些欄位由 checkpoint 決定，`dfn2_grids.h` 每個 (模型, sr, fft) 帶一組：

| 產品 | sr／fft | n_erb | df_bins | df_order | mask lookahead | df lookahead | sample 表中的 N |
|---|---|---:|---:|---:|---:|---:|---|
| full | 16000／256 | 24 | 96 | 5 | 0 | 0 | 1、8 |
| full | 48000／1024 | 32 | 96 | 5 | 1 | 1 | 1、2 |
| lite | 48000／1024 | 32 | 96 | 0 | 1 | 0 | 1 |
| full，mask_lookahead 0 的模型（`AINR_MODEL_ID_DFN2_FULL_L0`） | 16000／256 | 24 | 96 | 5 | 0 | 0 | 1、8 |
| 同上 | 32000／512 | 28 | 96 | 5 | 0 | 0 | 1、4 |
| 同上 | 48000／1024 | 32 | 96 | 5 | 0 | 0 | 1、2 |

函式庫接受綁定模型的 grid 上任何 1..64 的 N（`AINR_ERR_FRAMES_PER_CALL` 只在 N 超過 64 時回）；哪個 N 有 graph 是 sample 的 `graph_io_dfn2.h` 表，沒有列的組態由 sample 回 `GRAPH_IO_ERR_NO_ROW` 並在任何 SDK 呼叫之前拒絕。

N>1 在任何 mask lookahead 都可用：full 的 48 kHz／1024 有 N = 1、2 兩列（同一個 checkpoint；第一包的前 M 個 head 屬於串流開始之前的 frame，輸出靜音，加速器的遞迴 state 比 N=1 多看過 M 個 frame），lite 沒有 N>1 的列。mask_lookahead 0 的模型另有一組列：板端以 1024 樣本一塊錄音時 N = 1024／模型 hop（16 kHz 為 8、32 kHz 為 4、48 kHz 為 2），一包是一次 job、沒有額外延遲。這個模型是自己的 id，因為它的 48 kHz／1024 與 mask_lookahead 1 的模型是 shape 相同的不同 graph：graph 表的鍵是 (模型, sr, fft, N, `mask_lookahead`)，`mask_lookahead` 取自 `AinrInfo.mask_lookahead`；其 df lookahead 暫取 0（交付的 graph 才是準）。

## Graph 與 ERB 檔

- `graph_io_dfn2.h` 的每一列帶 graph 檔在模型目錄下的相對路徑（`GraphIoConfig.file`）：`dfn2/nvt_model_16k_n1.bin`、`dfn2/nvt_model_16k_n8.bin`、`dfn2/nvt_model_48k_n1.bin`、`dfn2/nvt_model_48k_n2.bin`；mask_lookahead 0 的模型自己的 `dfn2_l0/nvt_model_48k_n1.bin`、`dfn2_l0/nvt_model_48k_n2.bin`、`dfn2_l0/nvt_model_32k_n1.bin`、`dfn2_l0/nvt_model_32k_n4.bin`（其 16 kHz 兩列與 full 同檔）；lite 的 `dfn2_lite/nvt_model_48k_n1.bin`。DeepVQE-S 與 Align-ULCNet 每個模型一個檔（`deepvqes/nvt_model.bin`、`alignulcnet/nvt_model.bin`，見 `graph_io_deepvqes.h`、`graph_io_alignulcnet.h`），開圖後的埠檢查拒絕其他 grid 或 N 的 graph。
- 檔案路徑與量化資料夾是 sample 的：sample 開 `SAMPLE_MODEL_DIR "/" cfg.file`（巨集，預設 `/mnt/sd/audio_model/16x16`）。不同量化放在不同資料夾，檔名相同。
- ERB 矩陣檔依 `(sr, n_erb)` 命名，與量化無關，形如 `fb_<sr>_<n_erb>.bin`（`graph_io_dfn2_erb_file(model, sr, fft, &n_erb)` 給名稱，`ainr_init()` 之前就要）。sample 以 `stream/ainr/dfn2_erb_file.h` 的 `dfn2_erb_file_load()` 從 `SAMPLE_ERB_DIR`（預設 `/mnt/sd/dfn2_blobs`）讀入，把兩個表指標放進 `AinrConfig`。格式見 `docs/html/models/dfn2/reference/grid.html` 的 ERB 一節（`export_erb_matrix.py --format bin` 的 `erb_matrix.bin` 改名）。
- 登錄的 graph 是否已交付、量化版本是否與該列的 grid 與 N 相符，本庫沒有核對紀錄。full 16 kHz／256／N=8 與 mask_lookahead 0 模型的 32 kHz、48 kHz 各列都只有檔名；交付與是否相符待交付端確認。

## 驗收入口

```sh
make -C stream/ainr test                 # full：dfn2、freq、latency、blocks、freq-nonfinite、state-generation、pool-gates、atten-lim、product、graph-io、link-only
make -C stream/ainr MASK_ONLY=1 test     # lite：同上，去掉 dfn2 與 freq
```

這些測項以主機與替身加速器執行，不涉及正式 graph、SDK 或目標板。
