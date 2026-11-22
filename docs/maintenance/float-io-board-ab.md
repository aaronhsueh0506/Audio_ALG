# Float I/O 上板 A/B 清單

對象：八支 AI3 sample（`alg_ai3_dfn.c`、`alg_ai3_dfn_lite.c`、`alg_ai3_aec_dfn.c`、`alg_ai3_alignulcnet.c`、`alg_ai3_deepvqes.c`、`uvc_ai_nr_stream.c`、`uvc_audio_live2a.c`、`uvc_audio_live3.c`）的編譯期 `FLOAT_IN`／`FLOAT_OUT`。機制與規則見 [圖的埠順序與接線資料](graph-wiring.md) 的 Float I/O 一節。本庫沒有 SDK 標頭也沒有板子，所以下面是使用者上板要做的事；容許誤差不事先猜，由第一次量測決定。

## 要確認的三件事

| 問題 | 怎麼看 |
|---|---|
| CTRL 位元（`CTRL_BUF_FLOATIN`／`CTRL_BUF_FLOATOUT`）是否也作用在 state 埠 | 開 `FLOAT_IN=FLOAT_OUT=1` 後，啟動 log 的 `[AI3] in[i]`／`out[i]` 行裡 state 埠的 `float`／`bits`。state 回報 float 即作用在 state；回報定點即不作用（兩種結果 sample 都能接受，只要兩端一致） |
| NPU 內建轉換與 host 呼叫是否數值相等 | 步驟 4 的 head 比對。host 對輸入傳 scale 1.0f，對輸出傳埠的 `scale_ratio`，所以差異可能來自這個不對稱，不一定是 NPU 的問題 |
| 現有 `model.bin` 是否直接接受 float I/O | 步驟 2 的啟動結果：開圖失敗、`[AI3] in[i]`／`out[i]` 行仍回報定點，或啟動後 head 全為 NaN／全 0，都表示這個 bin 需要工具鏈的 float I/O 選項並重編；此時為它另放一個資料夾（`SAMPLE_MODEL_DIR`，檔名不變）或在 `graph_io_<model>.h` 加一列另一個檔名，不改 layout 版號 |

## 步驟

每支 sample 各做一輪，同一個板子、同一個輸入 wav、同一個 `model.bin`、同樣的 grid。

1. 備份。`FLOAT_IN`／`FLOAT_OUT` 都留 0 建置，跑完整輸入，存 A 的輸出 wav 與 stdout log（`a.wav`、`a.log`）。這是現況，必須與改動前的輸出逐位元相同；不同就停下來，問題不在 float I/O。參考錄音必須用同一份原始碼重錄：介面整理之後，`alg_ai3_alignulcnet.c` 在對齊事件的 hop 不再輸出 16 ms 的零（PCM 版也只重啟模型），`alg_ai3_aec_dfn.c` 在 AEC 硬重置附近的輸出也不同；舊錄音與這兩支新 sample 會在這些位置不同，不是 float I/O 的問題。
2. 兩個都改 1 建置（只改這兩個 define，其他不動），啟動。預期 log 有 `[AI3] float_in=1 float_out=1`，沒有 `WARNING: mixed setting`，埠檢查通過（埠檢查不比對 float 與定點）。記下：每個訊號輸入與 head 的 `scale_ratio`／`zero_point`、state 埠回報的型別，都在 sample 開圖時印的 `[AI3] in[i]`／`out[i]` 行裡（`compare` 腳本只讀含 `scale_ratio`／`zero_point` 的行）。
3. 跑同一個輸入，存 B 的輸出 wav 與 log（`b.wav`、`b.log`）。要比 head 時，兩邊都讓 sample 把每拍的 head 以 float32 依圖的順序寫到檔案（`a.f32`、`b.f32`）；沒有 dump 功能就略過 head 比對，只比音訊與時間。
4. 比對：

```sh
python3 -I docs/maintenance/float_io_compare.py \
    --audio-a a.wav --audio-b b.wav \
    --heads-a a.f32 --heads-b b.f32 --head-elems 24,960,1 \
    --log-a a.log --log-b b.log
```

`--head-elems` 是每拍各個 head 的元素數（DFN2 full 16 kHz/256 N=1 為 `24,960,1`；Align-ULCNet 為 `2*BINS`，DeepVQE-S 為 `BINS*18`）。腳本印出：兩邊的模式行、時間 bucket 的 B/A 比值（`io_in`、`io_out`、`ai3_*`、`conv`、`wall` 等名稱以 `us`／`ms` 結尾或前綴相符者）、`scale_ratio`／`zero_point` 的原文行、每個 head 的最大絕對差與相對 RMS、非有限值個數、最終音訊的最大差（LSB）、差異的 RMS 與 SNR。選定容許誤差後可用 `--fail-max-abs`、`--fail-audio-lsb` 變成退出碼；B 比 A 多出的 NaN／Inf 一律不算在限內；長度或聲道數不同、空的 WAV、短於一個 hop 的 head 傾印（B 跑到一半停掉的典型樣子）在設了限值時同樣視為失敗。

5. 判斷：
   - 100 拍以上的運行中 B 的 head 沒有非有限值，`state_check` 通過，沒有該拍被函式庫拒絕（generation 沒有無故前進）。
   - 最終音訊的差異與 head 的差異在同一個量級；若 head 差明顯而音訊差很小，記錄但不一定要處理。容許誤差在第一次量測後才定。
   - 時間：看 `io_in`、`io_out` 與（若有）轉換 bucket 是否下降，`ai3_net` 是否上升。若轉換只是搬進 SDK 的 CPU 外掛（在 `net_proc` 裡），節省會很小，只剩 `net_get`、暫存配置與回寫；這時開 float I/O 不值得，如實記錄。節省與代價（`ai3_net` 變長、記憶體）一起報。
6. 混合設定（`FLOAT_IN != FLOAT_OUT`）不在這份清單內，沒有驗證過；若要試，只看是否有編譯警告、啟動 log 的 `WARNING: mixed setting` 與 state 配對是否通過，不要拿來量效能。

## ULCNet N > 1 的 `frame_reset`

`frame_reset`（值 0.0／1.0）也是訊號輸入，開 `FLOAT_IN` 後經過 NPU 用該圖校準的輸入 scale 量化。0 和 1 在該 scale 下是否精確沒有驗證：在有事件的 hop（包內第 3 拍）比對 A、B 的 state 與輸出，確認 reset 真的發生。

## 結果回報格式

每支 sample 一行：sample 名稱、grid、`float_in`／`float_out`、state 埠型別、最大 head 差、最終音訊最大差（LSB）與 SNR、`io_in`／`io_out`／`ai3_net`／`wall` 的 B/A、是否有非有限值。
