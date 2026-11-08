# Audio_ALG

Audio_ALG 提供 mono／四麥克風 AEC 管線、八個 AI 模型的頻域 C 核心，以及 DFN2、DeepVQE-S、Align-ULCNet 的 PCM 串流 SDK。板端持有 AI3 graph 與裝置上的 recurrent state。

**[開啟使用手冊](docs/html/index.html)**，可按任務選擇建置、pipeline、模型或底層函式庫。下載後可直接在瀏覽器開啟。

## 第一次執行 mono AEC + NR

從 Audio_ALG 根目錄執行，準備同取樣率的 mono 麥克風與遠端參考 WAV。兩路長度不同時，程式處理較短一路的完整 hop。

```sh
make -f lib/deps.mk bootstrap-deps
make -C pipelines check-pins
make -C pipelines
bin="$(make -s -C pipelines print-bin-dir)"
"$bin/aec_nr_pipeline" mic.wav far.wav out.wav balanced --nr-level balanced
```

C 建置預設從同層的 AEC、NR、audiolib 讀取依賴，版本在 `lib/*.pin`。執行 Python pipeline 或 C／Python parity 前，另執行 `git submodule update --init --recursive`，以取得 Python 套件及錄音。
