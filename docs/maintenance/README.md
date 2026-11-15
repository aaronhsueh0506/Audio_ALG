# 建置維護與發布

C 的預設來源是 Audio_ALG 旁的 AEC、NR、audiolib。先在元件庫驗證版本，再更新相應的 `lib/*.pin`。Python pipeline 和使用錄音的 parity 同時依賴 `lib/aec`、`lib/nr` submodule。要用 submodule 的 C 實作，可執行 `make -C pipelines AEC_DIR=../lib/aec/c_impl NR_DIR=../lib/nr/c_impl`。

從 Audio_ALG 根目錄檢查依賴及對外文件。

```sh
make -C pipelines check-pins
python3 -m pytest -q pipelines/tests/test_doc_layout_versions.py pipelines/tests/test_c_python_map.py
```

讀 AEC 標頭的 Python 守門測試（`test_doc_layout_versions.py`、`pipelines/mono/tests/test_cng_*.py`、`pipelines/tests/test_pyc_parity_sources.py`、`AIAEC/tests/test_deployed_frontend_contract.py`）的 AEC 來源與 `lib/deps.mk` 同一套解析：環境變數 `AEC_DIR`（要給絕對路徑），否則是並列的 `../AEC/c_impl`，而不是 `lib/aec` submodule。要對 submodule 跑，就把 `AEC_DIR` 設成它的 `c_impl` 絕對路徑。

| 驗收項目 | 入口 |
|---|---|
| 鎖定版本、量測條件及 AEC／NR／audiolib 發布條件 | [依賴與品質](dependencies.md) |
| Mono、4ch、記憶體隔離與 parity | [管線驗收](pipeline.md) |
| AINR／AIAEC 模型與匯出 | [模型驗收](models.md) |
| Manifest、artifact 與跨庫發布順序 | [發布](publish.md) |
| AINR 產品綁定、graph 與 ERB 檔的登錄資料 | [AINR 發布資料](ainr-release-data.md) |
| 轉出圖的埠順序、sample 端接線資料與 `stream_check_ports()` 的比對依據 | [圖的埠順序與接線資料](graph-wiring.md) |
| Float I/O（`FLOAT_IN`／`FLOAT_OUT`）上板 A/B 的步驟與比對腳本 | [Float I/O 上板 A/B](float-io-board-ab.md)、`float_io_compare.py` |
| 比對 C 與 Python 的函式和覆蓋 | [函式對照表](../c_python_function_map.md) |

## 交付相容離線文件

每庫的 `docs/html/index.html` 是現行入口。若使用者保留舊書籤，可從 Audio_ALG 根目錄用 Python 3.9 以上建立含舊網址的文件副本。輸出請指定新的目錄。

```sh
python3 docs/maintenance/offline/build_legacy_docs.py --repo all --source .. --output ../SE-manuals
```

單獨發送 AEC 時改用 `--repo AEC --source ../AEC --output ../AEC-manual`。四庫副本由 `Audio_ALG/docs/html/index.html` 開啟，單庫副本由 `docs/html/index.html` 開啟。只有 checkout 具備的來源檔在交付副本中顯示為檔案名稱。`aliases.tsv` 只收 `docs/html/` 下的 HTML 網址，Markdown 目錄的搬移（例如 `docs/reference/mcab/` 改為 `docs/reference/array/`）無法登錄。
