# Audio_ALG 文件

從 **[Audio_ALG 使用手冊](html/index.html)** 選擇建置、六條 pipeline、八個模型或底層函式庫。PCM stream 的板端接法收在各模型的部署頁。

## 原始碼整合參考

| 要做的事 | Markdown 入口 |
|---|---|
| 組 mono AEC、NR、RES 與 OLA | [mono](reference/mono/README.md) |
| 整合四路 AEC、自備 BF 或使用 sample beam helper | [四麥克風 MCAB](reference/mcab/README.md) |
| 交叉編譯、配置 pool、診斷裝置 | [板端整合](reference/board/README.md) |
| 核對 C 與 Python 函式、維護發布基準 | [維護手冊](maintenance/README.md) |
