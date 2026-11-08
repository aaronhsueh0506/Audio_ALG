# ONNX 匯出

請在模型目錄執行本頁的 Python 指令。

將訓練好的模型匯出為 ONNX 格式（用於部署）：

```bash
# 匯出
python3 export_onnx.py --config config.ini --model output/rnnoise_best.pth \
                       --output output/rnnoise.onnx

# 匯出 + 驗證 (比較 PyTorch 與 ONNX 輸出)
python3 export_onnx.py --config config.ini --model output/rnnoise_best.pth \
                       --output output/rnnoise.onnx --verify
```

Streaming ONNX 輸入為 `erb_input[1,3,22]`、`spec_input[1,3,2,129]`
與三組 GRU hidden state（`h1_in`／`h2_in`／`h3_in`，各 `[1,1,128]`）；輸出中心 frame 的 gains 與更新後 hidden state。
加速器本身不保存 state，三個 hidden 張量屬於應用端：板端把圖的 `h*_in` 輸入與 `h*_out` 輸出綁到同一個 buffer，CPU 不複製；
runtime 須先讀完所有 state 輸入、才可寫同位址的 state 輸出。三個 feature frames
是輸入 temporal kernel 的完整可視範圍，不代表一次輸出三個 frames。

float host 在 `RNNoiseConfig` 註冊 `h1`／`h2`／`h3`（指標加元素數，各 128；三組全給或全不給，指標配元素數 0 或
元素數配 NULL 指標都會被拒絕）：`rnnoise_init()` 與 `rnnoise_reset()` 歸零，每次 `rnnoise_frame_finish(st, band_gains)`
檢查 `band_gains` 與整塊註冊的 state 是否 finite；任一為 NaN/Inf 時，註冊的張量歸零、state generation 加一、回傳 `-1`
（呼叫端改用單位增益）。仍為 finite 的部分寫入或混合寫入無法偵測。板端不註冊任何 state：加速器自己的定點副本不會被
reset 或歸零動到，所以 `rnnoise_state_generation()` 在 init 後為 1，每次 `rnnoise_reset()` 與每次被拒的
`rnnoise_frame_finish()` 加一（modulo 2^32，註冊與否都計），runtime 看到與上一 frame 不同的值就把自己的 state 張量重填
quantized zero。匯出的 metadata 帶 `state_layout_version`，數值與 `process.h` 的
`RNNOISE_MODEL_IO_LAYOUT_VERSION` 相同，整合端可據此拒絕 state layout
已經對不上自己配置的圖。

PTQ calibration 可直接使用：

```bash
python3 inference.py --config config.ini --model output/rnnoise_best.pth \
  --input noisy.wav --output enhanced.wav \
  --dump-calib calib/rnnoise_erb --format bin --max-frames 8192
```

BIN 會依 ONNX input 分資料夾，每個 streaming frame 各寫一個檔案；例如
`h1_in/h1_in_0000.bin`。使用 `--format npz --dump-calib
calib/rnnoise_erb.npz` 可改輸出 NumPy archive。兩種格式都保存實際串流中
的三-frame feature window 與非零 GRU state，而不是重複 zero-state 樣本。

## ERB 矩陣匯出

匯出 ERB 轉換矩陣（C 部署用）：

```bash
# 匯出所有格式 (bin + npy + C header)
python3 export_erb_matrix.py --config config.ini --format all
```

`--format bin` 的 `erb_matrix.bin`（little-endian）依序是 `int32 n_bins`、`int32 n_bands`、`float32` forward `[n_bins][n_bands]`、
`float32` inverse `[n_bins][n_bands]`（forward 與 inverse 都是 bin-major），尾端接 `int32 nfftborder[n_bands]`；`--format npy`
的 `erb_fwd.npy`／`erb_inv.npy` 同為 `(n_bins, n_bands)`。函式庫的 `erb_inv` 是 band-major `[n_bands][n_bins]`（見上面
「C 串流整合」）：樹內沒有 exporter 輸出這個 layout，應用端須把 inverse 轉置後再交入。

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `train.py` | 模型定義與 packed raw-WAV tensor 訓練迴圈 |
| `inference.py` | 推論與 calibration 入口 |
| `export_onnx.py` | ONNX 匯出（streaming 推論格式） |
| `export_erb_matrix.py` | ERB 矩陣匯出（bin / npy / C header） |
| `config.ini` | 所有超參數設定 |
| `process.c` / `process.h` | C 前後處理實作（頻譜域，嵌入式部署用） |
| `tests/` | Python feature、loss、checkpoint 與 Python/C parity tests |
| `../dataset_gen/` | 共用 augmentation、WAV generation、resample、packing 與 packed loader |
| `requirements.txt` | Python 依賴套件 |

## ERB 矩陣 drift-guard

C 端沒有編譯期 ERB 表；矩陣是應用端提供的資料。`make test-tables` 把測試讀入的矩陣與獨立重算的三角 ERB filterbank 比對，
並檢查矩陣契約（例如 partition of unity）。測試讀的是兩個無標頭 float32 檔 `$(BUILD_DIR)/erb_fwd.bin`（bin-major）與
`$(BUILD_DIR)/erb_inv.bin`（band-major），由 `tests/erb_fixture.py`（只用標準函式庫）依 `train.py` 的公式寫出；以
`ERB_FWD=FILE ERB_INV=FILE` 可改讀其他檔。`make test-erb-python`（訓練環境）再把這個 fixture 釘到 `train.py` 與
`export_erb_matrix.py`。

```bash
make test-tables
```

重算用的常數（SR、N_FFT、N_BANDS、MIN_BINS_PER_BAND）刻意不讀 `process.h` 或 `config.ini`，兩者之一改動時這裡會失敗。

## Feature tests

```bash
make test                 # tables + feature contract + feature state + framing + SIMD/scalar digest
make test-feature-python  # 需 torch training environment
```

C test 會以獨立 recurrence 對照 `rnnoise_compute_features()`，並讓固定非平坦
spectrum 運行 4096 frames，確認 ERB mean norm 收斂為 0，complex preprocessing
仍非零。Python test 另驗證兩路 state/chunk equivalence、ERB 穩態收斂、complex
穩態可觀測性，以及純 ERB／可選 complex 兩種 model forward contract；
`tests/test_python_c_features.py` 以 golden vectors 對照 Python/C 的 STFT、
兩路特徵與最終 normalization state。即使出貨模型未使用 complex 分支，保留的
可選路徑仍須通過這些測試，避免日後重新啟用時出現 Python/C 漂移。

[返回本主題](../README.md)
