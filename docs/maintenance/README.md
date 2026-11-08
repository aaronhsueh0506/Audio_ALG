# 建置維護與發布

C 的預設來源是 Audio_ALG 旁的 AEC、NR、audiolib。先在元件庫驗證版本，再更新相應的 `lib/*.pin`。Python pipeline 和使用錄音的 parity 同時依賴 `lib/aec`、`lib/nr` submodule。要用 submodule 的 C 實作，可執行 `make -C pipelines AEC_DIR=../lib/aec/c_impl NR_DIR=../lib/nr/c_impl`。

從 Audio_ALG 根目錄檢查依賴及對外文件。

```sh
make -C pipelines check-pins
python3 -m pytest -q pipelines/tests/test_doc_layout_versions.py pipelines/tests/test_c_python_map.py
```

| 驗收項目 | 入口 |
|---|---|
| 鎖定版本、量測條件及 AEC／NR／audiolib 發布條件 | [依賴與品質](dependencies.md) |
| Mono、4ch、記憶體隔離與 parity | [管線驗收](pipeline.md) |
| AINR／AIAEC 模型與匯出 | [模型驗收](models.md) |
| Manifest、artifact 與跨庫發布順序 | [發布](publish.md) |
| 比對 C 與 Python 的函式和覆蓋 | [函式對照表](../c_python_function_map.md) |

## 交付相容離線文件

每庫的 `docs/html/index.html` 是現行入口。若使用者保留舊書籤，可從 Audio_ALG 根目錄用 Python 3.9 以上建立含舊網址的文件副本。輸出請指定新的目錄。

```sh
python3 docs/maintenance/offline/build_legacy_docs.py --repo all --source .. --output ../SE-manuals
```

單獨發送 AEC 時改用 `--repo AEC --source ../AEC --output ../AEC-manual`。四庫副本由 `Audio_ALG/docs/html/index.html` 開啟，單庫副本由 `docs/html/index.html` 開啟。只有 checkout 具備的來源檔在交付副本中顯示為檔案名稱。
