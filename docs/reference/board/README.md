# 板端 C 整合

已經有 PCM 的單模型呼叫端可使用 DFN2、DeepVQE-S 或 Align-ULCNet 的 stream SDK。已有頻譜、AEC 或 BF 的 pipeline 可直接串頻域介面，保留既有 framing、增益與時序。

| 工作 | 參考 |
|---|---|
| 選模型、接入 AI3、處理 reset 和推論失敗 | [各模型部署頁](../../html/models/index.html) |
| 接既有 mono 或四路前端 | [呼叫順序與資料來源](sequence.md) |
| 配置 caller-owned pool、檢查 descriptor、拆除物件 | [記憶體與生命週期](memory.md) |
| 指定 target compiler、ABI 與數值選項 | [交叉建置](build.md) |
| 查訊號、耗時及拒收狀態 | [現有診斷介面](diagnostics.md) |
