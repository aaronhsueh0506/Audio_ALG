# 圖的埠順序與接線資料（stream/ainr、stream/aiaec、stream/common）

轉出的圖只有一條排序規則，所以埠的順序與 state 配對用位置表示，板端 sample 不需要依模型分支，也不比對名稱。stream 函式庫不描述圖的埠；接線資料是各函式庫旁的只有標頭的表，與共用接線程式 `stream/common/ai3_model_io.h`，函式庫不引用。

## 歸屬

| 項目 | 位置 |
|---|---|
| 產品、grid、`N = hop_in / (fft / 2)`、graph 與 ERB 的選擇，尾端長度 | `stream/ainr/ainr.c`、`stream/aiaec/aiaec.c` 的內部 catalog；`AinrInfo` / `AiaecInfo` 只回 `graph_path`、`trim_samples`、`queue_frames`、`tail_zero_blocks` |
| 這一包的訊號與 head（資料指標、元素數、`[b,c,h,w]`、generation、action、`frame_reset` 控制值） | `AinrWork` / `AiaecWork`，由真實的 pre 呼叫填 |
| 每個已交付 graph 的完整接線（訊號、head、state 的數量與 shape） | `stream/ainr/ai3_model_io_dfn2.h`、`stream/aiaec/ai3_model_io_deepvqes.h`、`stream/aiaec/ai3_model_io_alignulcnet.h`（只有標頭，`ainr.h`／`aiaec.h` 不引用）：每個登錄的 `(產品, sr, fft, N)` 一列，註明所對應的匯出器 |
| 共用接線程式：列的展開、state 配對、`stream_check_ports()` 呼叫、buffer 大小、pre 到 AI3 的邊界核對 | `stream/common/ai3_model_io.h`（SDK 無關、只有標頭；`make -C stream/common test-model-io`）。sample 加 `-I stream/common -I stream/ainr`（或 `stream/aiaec`）並編譯 `stream/common/stream_bind.c` |
| SDK 無關的 graph／裝置描述檢查器（`StreamWiring`、`StreamDeviceDesc`、`stream_check_ports()`、`STREAM_PORT_ERR_*`） | `stream/common/stream_bind.{h,c}`，只由 sample 連結；`libainr.a`、`libaiaec.a` 不含它 |
| 記憶體需求型別 `StreamMemReq` 與 pool 工具 | `stream/common/stream_mem.h`，與接線型別無關；`ainr.h`、`aiaec.h` 只引用它 |

`ainr.h` 與 `aiaec.h` 不引用 `stream_bind.h`；函式庫也沒有只為 graph 接線存在的 state 尺寸常數。缺少列或與所選 graph 不符的組態，由 sample 的啟動檢查在任何 SDK 呼叫之前拒絕。

## 規則

- 輸入 = 訊號輸入，再接 state 輸入；輸出 = head 輸出，再接 state 輸出。
- 第 k 組 state 是輸入 `signal_in_count + k` 與輸出 `head_out_count + k`，兩端形狀相同。
- 輸出的 state 名稱是輸入名稱加 `_out`（只有匯出器端讀名稱）。
- `stream_check_ports()` 的訊息以位置稱呼埠，例如 `input 3 (state 0)`。

| 模型 | 訊號輸入 | head | 輸入／輸出埠數 | state 組數 |
|---|---|---|---|---|
| DFN2 full | 2（`erb`、`spec`） | 3（`erb_mask`、`df_coefs`、`df_alpha`） | 4 ／ 5 | 2（`h_gru`、`df_convp_history`） |
| DFN2 lite | 2 | 1（`erb_mask`） | 3 ／ 2 | 1（`h_gru`） |
| DeepVQE-S | 2（`mic`、`far`） | 1（`output`） | 18 ／ 17 | 16 |
| Align-ULCNet N=1（layout 18） | 4（`error_mag`、`far_mag`、`error_cos`、`error_sin`） | 1（`output`） | 8 ／ 5 | 4 |
| Align-ULCNet N>1（layout 19） | 5（上列加最後一個 `frame_reset`，`[1, 1, 1, N]`） | 1 | 9 ／ 5 | 4 |

Align-ULCNet 只有一個描述產生器：N>1 在訊號輸入的最後多一個 `frame_reset`，由 stream 依逐 hop 事件填進 `AiaecWork.inputs`，sample 把它當一般的第五個訊號輸入搬運，沒有特例。N=1 維持 layout 18，圖裡沒有 `frame_reset`，事件 hop 是 BYPASS 區塊。包內第 3 個 hop 有事件時 `frame_reset` 為 `00100000`。

## 依據

- 四種匯出器（`AINR/DeepFilterNet2/export_onnx.py`、`AIAEC/_streaming_export.py` 的 DeepVQE-S、`AIAEC/Align_ULCNet/export_onnx.py` 的 N=1 與 N>1）實際寫出的 ONNX 名稱順序都符合上列規則，匯出器不必改。
- 規則成立時，名稱、角色與配對欄位不必在函式庫與 sample 兩側各維護一份；接線資料只在 `ai3_model_io_<model>.h` 的表裡存一份，函式庫與匯出器的一致性由 work view 的測試比對。
- 匯出器的順序或 shape 若改變，`ai3_model_io_<model>.h` 對應的列與本庫的 pin 測試要一起改，不在 sample 裡加特例。

## 驗證

```sh
python3 -m pytest -q AIAEC/tests/test_aiaec_bind_info_pin.py AINR/tests/test_dfn2_port_order.py
make -C stream/common test
make -C stream/ainr test-ainr-work-shapes test-ainr-model-io
make -C stream/ainr MASK_ONLY=1 test-ainr-work-shapes test-ainr-model-io
make -C stream/aiaec test-aiaec-model-io
```

- `test_aiaec_bind_info_pin.py`（一）用 `make -C stream/aiaec bind-dumpers` 建的傾印程式（真的 `aiaec_init` 與 `aiaec_preprocess*`，三個家族各一次），把 `AiaecWork` 的訊號與 head（數量、形狀、元素數）依位置比對隨機權重匯出圖的訊號輸入與 head 輸出，並檢查匯出圖的名稱順序與 state 兩端同形狀（DeepVQE-S 16k/256 N=1、N=8、16k/512 N=1；Align-ULCNet 同三組）。另固定 N=8 第 3 個 hop 事件的 `frame_reset` 為 `00100000`，與 N=1 事件 hop 為沒有任何 tensor 的 BYPASS。傾印程式不印 state，因為函式庫沒有 state 的資料。
  （二）另用 `make -C stream/aiaec model-io-dumper`（環境變數 `AIAEC_MODEL_IO_DUMPER` 可改指別的建置）傾印 `ai3_model_io_deepvqes.h`／`ai3_model_io_alignulcnet.h` 的每一列，逐列比對全新匯出的圖：layout 版本、訊號／head／輸入／輸出的數量、每個輸入輸出的 shape、state 名稱與順序、輸出名稱 `<輸入>_out`、state 兩端同 shape，以及 Align-ULCNet N>1 的 `frame_reset` 為最後一個訊號；列集合必須恰為匯出的六個案例。
- `test_dfn2_port_order.py` 對 DFN2 full 與 lite、N=1 與 8 的匯出圖做名稱順序與 state 形狀檢查，並用 `make -C stream/ainr [MASK_ONLY=1] work-dumper` 建的傾印程式，把四個登錄列（full 16 kHz/256 的 N=1 與 8、full 與 lite 48 kHz/1024 的 N=1）的 `AinrWork` 對匯出圖的訊號輸入與 head 輸出，另驗拒絕的組態回傳碼。它也用 `make -C stream/ainr model-io-dumper`（`AINR_MODEL_IO_DUMPER` 可改指別的建置）傾印 `ai3_model_io_dfn2.h` 的每一列，逐列比對全新匯出的圖（`test_sample_row_equals_a_fresh_export`，項目同上），列集合必須恰為登錄的四列。
- `test-ainr-work-shapes` 以匯出器寫下的形狀表檢查三個家族的 `AinrWork`（登錄的每個組態，full 與 lite）。
- `stream/common` 的 `test_stream_bind` 涵蓋 `stream_check_ports()` 的每條拒絕規則（含定點 state 埠兩端 `zero_point` 必須為 0）；`test_stream_mem` 涵蓋 `StreamMemReq` 與 pool 工具。
- state 的 shape 只在 `ai3_model_io_<model>.h` 的表裡（`stream/ainr/ai3_model_io_dfn2.h`、`stream/aiaec/ai3_model_io_{deepvqes,alignulcnet}.h`），函式庫沒有。表各列與匯出器的對照有兩層：上面兩個 pytest 對全新匯出逐列比對（埠序、每個登錄 grid 與 N 的形狀、state 順序）；C 端的 `test-ainr-model-io`、`test-aiaec-model-io` 把每一列展開後對真實 pre 呼叫給出的 `AinrWork`／`AiaecWork` 核對。共用的展開、配對與檢查程式 `stream/common/ai3_model_io.h` 由 `make -C stream/common test-model-io` 測試。
