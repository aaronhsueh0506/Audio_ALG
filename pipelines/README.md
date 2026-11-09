# Pipelines

選擇麥克風數量與後級。下列六個入口各自說明輸入、建置和每 hop 的呼叫方式。

| 後級 | Mono | 4ch |
|---|---|---|
| 傳統 AEC、NR、RES | [Mono](mono/aec_nr_res/README.md) | [四路 AEC 與 post-beam NR／RES](4ch/README.md) |
| DeepFilterNet2 | [Mono DFN2](mono/aec_dfn_res/README.md) | [4ch DFN2](4ch/aec_bf_dfn_res/README.md) |
| Align-ULCNet | [Mono ULCNet](mono/alignulcnet/README.md) | [4ch ULCNet](4ch/alignulcnet/README.md) |

從 repo 根目錄執行 `make -C pipelines` 建置 mono，`make -C pipelines/4ch` 建置 4ch。透過 `make -s -C pipelines print-bin-dir` 取得相同設定下的 mono 執行檔目錄。

**單模型 PCM 呼叫端**可直接接 [stream SDK](../docs/html/models/index.html)。已持有頻譜的管線可使用 [DFN2 的 RES stage 與 rate bridge](dfn/README.md) 或 [ULCNet 的頻域 stage](ulcnet/README.md)。四麥克風共用的 MCAB 前端、beam 和後級的接線見 [四麥克風整合](../docs/reference/mcab/README.md)。

- [逐 hop 呼叫、延遲與新增頻域 stage](reference/call-flow.md)
- [建置、板端和 debug](reference/build.md)
- [調參與故障排查](reference/tuning.md)
