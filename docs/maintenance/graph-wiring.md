# 圖的埠順序與接線資料（stream/ainr、stream/aiaec、stream/common）

轉出的圖只有一條排序規則，所以埠的順序與 state 配對用位置表示，板端 sample 不需要依模型分支，也不比對名稱。stream 函式庫不描述圖的埠；接線資料是各函式庫旁的只有標頭的表，與共用接線程式 `stream/common/graph_io.h`，函式庫不引用。

## 歸屬

| 項目 | 位置 |
|---|---|
| 產品、grid、`N = hop_in / (fft / 2)`、graph 與 ERB 的選擇，尾端長度 | `stream/ainr/ainr.c`、`stream/aiaec/aiaec.c` 的內部 catalog；`AinrInfo` / `AiaecInfo` 只回 `graph_path`、`trim_samples`、`queue_frames`、`tail_zero_blocks` |
| 這一包的訊號與 head（資料指標、元素數、`[b,c,h,w]`、generation、action、`frame_reset` 控制值） | `AinrWork` / `AiaecWork`，由真實的 pre 呼叫填 |
| 每個登錄的組態（含尚未交付 graph 的）的完整接線（訊號、head、state 的數量與 shape） | `stream/ainr/graph_io_dfn2.h`、`stream/aiaec/graph_io_deepvqes.h`、`stream/aiaec/graph_io_alignulcnet.h`（只有標頭，`ainr.h`／`aiaec.h` 不引用）：每個登錄的 `(產品, sr, fft, N)` 一列，註明所對應的匯出器 |
| 共用接線程式：列的展開、state 配對、`stream_check_ports()` 呼叫、buffer 大小、pre 到 AI3 的邊界核對 | `stream/common/graph_io.h`（SDK 無關、只有標頭；`make -C stream/common test-graph-io`）。sample 加 `-I stream/common -I stream/ainr`（或 `stream/aiaec`）並編譯 `stream/common/stream_bind.c` |
| SDK 無關的 graph／裝置描述檢查器（`StreamWiring`、`StreamDeviceDesc`、`stream_check_ports()`、`STREAM_PORT_ERR_*`） | `stream/common/stream_bind.{h,c}`，只由 sample 連結；`libainr.a`、`libaiaec.a` 不含它 |
| 記憶體需求型別 `StreamMemReq` 與 pool 工具 | `stream/common/stream_mem.h`，與接線型別無關；`ainr.h`、`aiaec.h` 只引用它 |

`ainr.h` 與 `aiaec.h` 不引用 `stream_bind.h`；函式庫也沒有只為 graph 接線存在的 state 尺寸常數。缺少列或與所選 graph 不符的組態，由 sample 的啟動檢查在任何 SDK 呼叫之前拒絕。

接線表登錄的組態比函式庫目前交付的 graph 多：AIAEC 的正式目錄（`aiaec.c`）現在只有 DeepVQE-S 16 kHz／fft 512／N=1 帶 graph 路徑，DeepVQE-S 其餘兩列與 Align-ULCNet 的三列有接線、沒有交付的 graph，正式建置在 init 就拒絕它們（測試目錄 `-DAIAEC_TEST_CATALOG` 才接受，見 `stream/aiaec/tests/test_aiaec_graph_io.c`）。

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

## Float I/O 模式（FLOAT_IN／FLOAT_OUT）

函式庫與 stream 層的所有 tensor 本來就是 float32（`AinrWork`／`AiaecWork` 的資料指標都是 `float *`），沒有任何量化程式。定點與 float 之間的軟體轉換只在板端 sample：輸入端把 float 轉成定點再送進 NPU，head 讀回時再轉回 float。Float I/O 的目的只有一個：讓 NPU 在開圖時做這兩步（量化與反量化），CPU 不再做。

- 開關是每支 AI3 sample 開頭的編譯期 `#define FLOAT_IN 0`、`#define FLOAT_OUT 0`，預設關閉；沒有執行期旗標，也沒有全域模式結構。`FLOAT_IN` 管訊號輸入，`FLOAT_OUT` 管 head 輸出。
- 這些 model 的 state 就地繼承 input／output 的型別（recurrent state 的輸入與輸出共用同一塊 buffer），所以**要求 `FLOAT_IN == FLOAT_OUT`**；混合設定只適用於不繼承 state 的模型，這裡的 AI3 sample 都不是。混合設定不會被編譯擋下，只有一個編譯警告（`#warning`，內容為 `FLOAT_IN != FLOAT_OUT: these models inherit the state port types, so FLOAT_IN must equal FLOAT_OUT; ...`）和啟動時 `[AI3] float_in=%d float_out=%d` 那行 log 後面的 `WARNING: mixed setting`。用 `-Werror` 建置時這個 `#warning` 會變成錯誤。
- State 埠不受這兩個 define 管：它們維持 SDK 回報的型別。state 的清零一律是對 buffer 填位元組 0 並 flush cache，不分 float 與定點：float 的 `0.0f` 全為零位元組；定點則靠 `stream_check_ports()` 保證兩端 `zero_point` 為 0（`STREAM_PORT_ERR_STATE_ZERO_POINT`，-110），所以通過埠檢查的 graph 不會有需要非零量化零的 state 埠。
- 開 float I/O 只省轉換，不省搬運：函式庫的資料在 host heap，NPU buffer 在 NPU 可見的記憶體，輸入的 `memcpy` 與 cache flush 仍在，head 讀回變成 flush 加 `memcpy(elem * 4 bytes)`。Host 端的 DFN2 `df_alpha` 夾到 [0, 1] 仍保留，NaN 預填與函式庫的有限值檢查仍在：float head 不會像定點那樣飽和，NPU 輸出的 NaN／Inf 由函式庫拒絕該拍（直通加 generation 前進，見上節 State 的回復）。

### 啟動 log

模型永遠是定點；`FLOAT_IN`／`FLOAT_OUT` 只讓硬體在邊界做轉換。SDK 回報的埠型別因此不能當作請求是否生效的依據，sample 不拿請求去比對回報的型別，也沒有對應的錯誤碼。埠檢查（`ports_check()` 或等價函式）仍只做 `stream_check_ports()`，它通過後、任何 buffer 覆寫之前，sample 印出（`float_io_compare.py` 讀這幾行）：

```
[AI3] float_in=0 float_out=0
[AI3] in[0] float=0 bits=16 scale_ratio=... zero_point=0
[AI3] out[0] float=0 bits=16 scale_ratio=... zero_point=0
```

第一行在 `FLOAT_IN != FLOAT_OUT` 時尾端加 ` WARNING: mixed setting`；之後每個輸入埠與輸出埠各一行，內容是 SDK 回報的 `is_float`、`elem_bits`、`scale_ratio`、`zero_point`。埠檢查不會因 float 與定點的差異而失敗，是否生效由板端 A/B（下節）從這些行與輸出比對判斷。

### 板端尚未驗證的項目

以下只能上板確認，本庫沒有 SDK 標頭也沒有板子，所以沒有猜：

1. CTRL 位元（`CTRL_BUF_FLOATIN`、`CTRL_BUF_FLOATOUT`）是否也作用在 state 埠。
2. NPU 內建轉換與 host 呼叫的數值是否相等：host 對輸入傳 scale 1.0f，對輸出傳埠的 `scale_ratio`。
3. 現有 `model.bin` 是否直接接受 float I/O，還是工具鏈需要另一個選項並重編。若需要另一個 bin，在 `stream/ainr/ainr.c`／`stream/aiaec/aiaec.c` 的 catalog 加第二個 `graph_path`，不改 layout 版號。

上板 A/B 的步驟與比對腳本見 `docs/maintenance/float-io-board-ab.md` 與 `docs/maintenance/float_io_compare.py`。

## 依據

- 四種匯出器（`AINR/DeepFilterNet2/export_onnx.py`、`AIAEC/_streaming_export.py` 的 DeepVQE-S、`AIAEC/Align_ULCNet/export_onnx.py` 的 N=1 與 N>1）實際寫出的 ONNX 名稱順序都符合上列規則，匯出器不必改。
- 規則成立時，名稱、角色與配對欄位不必在函式庫與 sample 兩側各維護一份；接線資料只在 `graph_io_<model>.h` 的表裡存一份，函式庫與匯出器的一致性由 work view 的測試比對。
- 匯出器的順序或 shape 若改變，`graph_io_<model>.h` 對應的列與本庫的 pin 測試要一起改，不在 sample 裡加特例。

## 驗證

```sh
python3 -m pytest -q AIAEC/tests/test_aiaec_bind_info_pin.py AINR/tests/test_dfn2_port_order.py
make -C stream/common test
make -C stream/ainr test-ainr-graph-io
make -C stream/ainr MASK_ONLY=1 test-ainr-graph-io
make -C stream/aiaec test-aiaec-graph-io
```

- `test_aiaec_bind_info_pin.py`（一）用 `make -C stream/aiaec bind-dumpers` 建的傾印程式（真的 `aiaec_init` 與 `aiaec_preprocess*`，三個家族各一次），把 `AiaecWork` 的訊號與 head（數量、形狀、元素數）依位置比對隨機權重匯出圖的訊號輸入與 head 輸出，並檢查匯出圖的名稱順序與 state 兩端同形狀（DeepVQE-S 16k/256 N=1、N=8、16k/512 N=1；Align-ULCNet 同三組）。另固定 N=8 第 3 個 hop 事件的 `frame_reset` 為 `00100000`，與 N=1 事件 hop 為沒有任何 tensor 的 BYPASS。傾印程式不印 state，因為函式庫沒有 state 的資料。
  （二）另用 `make -C stream/aiaec graph-io-dumper`（環境變數 `AIAEC_GRAPH_IO_DUMPER` 可改指別的建置）傾印 `graph_io_deepvqes.h`／`graph_io_alignulcnet.h` 的每一列，逐列比對全新匯出的圖：layout 版本、訊號／head／輸入／輸出的數量、每個輸入輸出的 shape、state 名稱與順序、輸出名稱 `<輸入>_out`、state 兩端同 shape，以及 Align-ULCNet N>1 的 `frame_reset` 為最後一個訊號；列集合必須恰為匯出的六個案例。
- `test_dfn2_port_order.py` 對 DFN2 full 與 lite、N=1 與 8 的匯出圖做名稱順序與 state 形狀檢查，並用 `make -C stream/ainr [MASK_ONLY=1] work-dumper` 建的傾印程式，把四個登錄列（full 16 kHz/256 的 N=1 與 8、full 與 lite 48 kHz/1024 的 N=1）的 `AinrWork` 對匯出圖的訊號輸入與 head 輸出，另驗拒絕的組態回傳碼。它也用 `make -C stream/ainr graph-io-dumper`（`AINR_GRAPH_IO_DUMPER` 可改指別的建置）傾印 `graph_io_dfn2.h` 的每一列，逐列比對全新匯出的圖（`test_sample_row_equals_a_fresh_export`，項目同上），列集合必須恰為登錄的四列。
- `test-ainr-graph-io` 另驗每個家族的第二個 block 與第一個同形狀，以及 `ainr_get_mem_size()` 的每個拒絕碼由三個 init 家族原樣回傳（`*ctx` 為 NULL、`*info` 不動）。
- `stream/common` 的 `test_stream_bind` 涵蓋 `stream_check_ports()` 的每條拒絕規則（含定點 state 埠兩端 `zero_point` 必須為 0）；`test_stream_mem` 涵蓋 `StreamMemReq` 與 pool 工具。
- 函式庫端的 float head 非有限值回歸：`make -C stream/ainr test-ainr-state-generation`（DFN2 三個 head 任一端的 NaN／+Inf／-Inf 都是被拒的 commit，generation 前進）、`make -C stream/aiaec test-aiaec-block test-aiaec-packet`（DeepVQE-S taps、Align-ULCNet mask 首尾元素的 Inf，以及有限但使增強後頻譜溢位的 mask）。
- state 的 shape 只在 `graph_io_<model>.h` 的表裡（`stream/ainr/graph_io_dfn2.h`、`stream/aiaec/graph_io_{deepvqes,alignulcnet}.h`），函式庫沒有。表各列與匯出器的對照有兩層：上面兩個 pytest 對全新匯出逐列比對（埠序、每個登錄 grid 與 N 的形狀、state 順序）；C 端的 `test-ainr-graph-io`、`test-aiaec-graph-io` 把每一列展開後對真實 pre 呼叫給出的 `AinrWork`／`AiaecWork` 核對。共用的展開、配對與檢查程式 `stream/common/graph_io.h` 由 `make -C stream/common test-graph-io` 測試。
