# AINR 發布資料（DFN2 full／lite）

客戶介面只有 `AinrConfig { sr, fft, hop_in }`。下列資料屬於產品建置與發布，不放客戶頁面。來源是 `stream/ainr/ainr_product.h` 與 `stream/ainr/ainr.c` 的登錄表；本筆記只做索引，數值以原始碼為準。

## 產品綁定

- 一個產品只含一種 DFN2：full 或 lite（archive 以 `DFN2_MASK_ONLY` 區分，`make -C stream/ainr MASK_ONLY=1 lib` 為 lite）。
- 綁定的模型是編譯期常數 `AINR_PRODUCT_MODEL_ID`（`ainr_product.h`）。未指定時 full archive 取 `AINR_MODEL_ID_DFN2_FULL`，lite archive 取 `AINR_MODEL_ID_DFN2_LITE`。產品要綁量化替代版時，編譯 `ainr.c` 時加 `-DAINR_PRODUCT_MODEL_ID=<ID>`。
- 綁定的 ID 不屬於該 archive 時，每個 `ainr_get_mem_size()`／`ainr_init()` 都回 `AINR_ERR_MODEL_ID`。

| ID 常數 | 值 | 量化 | 登錄狀態 |
|---|---:|---|---|
| `AINR_MODEL_ID_DFN2_FULL` | 1 | 16x16（預設） | 已登錄 |
| `AINR_MODEL_ID_DFN2_FULL_ALT1` | 10 | 8x8 | 未登錄（列在下方「8x8（ALT1）登錄列」）；所有組態以 `AINR_ERR_GRID` 被拒 |
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
| full | 48000／1024 | 32 | 96 | 5 | 1 | 1 | 1、2 |
| lite | 48000／1024 | 32 | 96 | 0 | 1 | 0 | 1 |
| full，mask_lookahead 0 的模型（`AINR_MODEL_ID_DFN2_FULL_L0`） | 16000／256 | 24 | 96 | 5 | 0 | 0 | 1、8 |
| 同上 | 32000／512 | 28 | 96 | 5 | 0 | 0 | 1、4（graph 未交付，路徑 NULL） |
| 同上 | 48000／1024 | 32 | 96 | 5 | 0 | 0 | 1、2（graph 未交付，路徑 NULL） |

N>1 在任何 mask lookahead 都可用（`AINR_ERR_LOOKAHEAD_BLOCK_UNSUPPORTED` 不再回傳）：full 的 48 kHz／1024 登錄 N = 1、2（同一個 checkpoint，N = 2 的檔是 `nvt_model_48k_n2.bin`；`ALT2` 為 `16x8/dfn2/nvt_model_48k_n2.bin`；第一包的前 M 個 head 屬於串流開始之前的 frame，輸出靜音，加速器的遞迴 state 比 N=1 多看過 M 個 frame），lite 沒有 N>1 的列。mask_lookahead 0 的模型另有一列組：板端以 1024 樣本一塊錄音時 N = 1024／模型 hop（16 kHz 為 8、32 kHz 為 4、48 kHz 為 2），一包是一次 job、沒有額外延遲。這個模型是自己的 id，因為它的 48 kHz／1024／N=1 與 mask_lookahead 1 的模型的同一把鍵是兩張 shape 相同的不同 graph，而客戶組態沒有 lookahead 欄位；其 df lookahead 暫取 0（交付的 graph 才是準），沒有 ALT1／ALT2 量化替代；它的 48 kHz／1024／N=2 檔不得沿用 mask_lookahead 1 的 `nvt_model_48k_n2.bin` 檔名。32 kHz／512 的 ERB 檔是 `fb_32000_28.bin`。登錄表中 `atten_lim_db` 與 HPF 欄位在所有列都是 0／關閉。

## Graph 與 ERB 檔

- 登錄表的每個 N 帶一個 graph 路徑；路徑為 NULL 的 N 視為未交付，回 `AINR_ERR_FRAMES_PER_CALL`，不會改用別的 N 的圖。
- graph 路徑由三段組成：基底目錄（`AINR_MODEL_DIR_BASE`）、量化子目錄、產品子目錄（full 與 lite 各一），檔名依取樣率與 N 命名，形如 `<前綴>_model_<16k|48k>_n<N>.bin`。前綴與各列的完整路徑見 `ainr.c` 內的 `AinrDfn2FramesVariant` 陣列。`ainr.c` 的註解標明檔名為暫定，待正式命名慣例確認。
- ERB 矩陣檔依 `(sr, n_erb)` 命名，與量化無關，形如 `fb_<sr>_<n_erb>.bin`，放在 `AINR_ERB_MATRIX_DIR`（`ainr.c` 內有預設值，可在編譯時覆寫）。格式見 `docs/html/models/dfn2/reference/grid.html` 的 ERB 一節（`export_erb_matrix.py` 的 `erb_matrix.bin`）。
- 登錄的 graph 是否已交付、量化版本是否與該列的 grid 與 N 相符，本庫沒有核對紀錄。full 16 kHz／256／N=8 的路徑存在於登錄表；交付與是否相符待交付端確認。mask_lookahead 0 的模型的 32 kHz 與 48 kHz graph 路徑為 NULL（正式登錄拒絕），交付時把路徑填進 `ainr.c` 的 `AINR_DFN2_FULL_L0_32K_512_N`、`AINR_DFN2_FULL_L0_48K_1024_N` 兩個陣列；測試建置（`AINR_TEST_CATALOG`）用 `test://` 佔位路徑。

## 8x8（ALT1）登錄列

ALT1 的登錄列與 graph 路徑陣列不放在 `ainr.c`；要登錄時把下列片段放回對應的 `#ifndef DFN2_MASK_ONLY` 區塊（full）或 `#else` 區塊（lite），並把列加進 `AINR_DFN2_CATALOG`；`<前綴>` 與上一節相同（現行各列見 `ainr.c`），路徑為暫定檔名。full 的 graph 路徑陣列：

```c
static const AinrDfn2FramesVariant AINR_DFN2_FULL_16K_256_N_8X8[] = {
    { 1, AINR_MODEL_DIR_BASE "/8x8/dfn2/<前綴>_model_16k_n1.bin" },
    { 8, AINR_MODEL_DIR_BASE "/8x8/dfn2/<前綴>_model_16k_n8.bin" },
};
static const AinrDfn2FramesVariant AINR_DFN2_FULL_48K_1024_N_8X8[] = {
    { 1, AINR_MODEL_DIR_BASE "/8x8/dfn2/<前綴>_model_48k_n1.bin" },
};
```

full 的 catalog 列（接在 `AINR_MODEL_ID_DFN2_FULL` 的兩列之後）：

```c
{ AINR_MODEL_ID_DFN2_FULL_ALT1, 16000, 256,  24, 96, 5, 0, 0, 0.0f, 0, 0.0f,
  AINR_DFN2_FULL_16K_256_N_8X8, 2 },
{ AINR_MODEL_ID_DFN2_FULL_ALT1, 48000, 1024, 32, 96, 5, 1, 1, 0.0f, 0, 0.0f,
  AINR_DFN2_FULL_48K_1024_N_8X8, 1 },
```

lite 的 graph 路徑陣列與 catalog 列：

```c
static const AinrDfn2FramesVariant AINR_DFN2_LITE_48K_1024_N_8X8[] = {
    { 1, AINR_MODEL_DIR_BASE "/8x8/dfn2_lite/<前綴>_model_48k_n1.bin" },
};

{ AINR_MODEL_ID_DFN2_LITE_ALT1, 48000, 1024, 32, 96, 0, 1, 0, 0.0f, 0, 0.0f,
  AINR_DFN2_LITE_48K_1024_N_8X8, 1 },
```

登錄後 `make -C stream/ainr test-ainr-product` 中「ALT1 每個組態都被 `AINR_ERR_GRID` 拒絕」的檢查要改成 ALT1 的 graph 路徑檢查。

## 驗收入口

```sh
make -C stream/ainr test                 # full：dfn2、freq、latency、blocks、freq-nonfinite、state-generation、pool-gates、atten-lim、product、graph-io、link-only
make -C stream/ainr MASK_ONLY=1 test     # lite：同上，去掉 dfn2 與 freq
```

這些測項以主機與替身加速器執行，不涉及正式 graph、SDK 或目標板。
