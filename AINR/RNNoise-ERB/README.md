# RNNoise-ERB

以 RNNoise v0.2 的 Conv+GRU 方向為基礎的本地噪音抑制模型。出貨設定只讓網路
使用 mean-normalized log-ERB；程式仍保留可選的低頻 complex spectrum 分支，
但 `config.ini` 預設 `use_complex_input = false`。本模型並非官方 RNNoise v0.2
或 DeepFilterNet 的逐層相容移植。

## 架構

- **ERB 輸入**: 22 個 log-energy bands，做每-band causal EMA mean norm
- **Complex 輸入（可選、預設關閉）**: 0–4 kHz 的 129 個 real/imag bins，做每-bin magnitude EMA unit norm
- **模型**: ERB temporal conv → 3 層 GRU(128)；開啟 complex 分支時才加入 frequency encoder/temporal conv 與 fusion
- **輸出**: 22 個 gain masks [0, 1]
- **參數量**: 329,302（預設純 ERB、無 VAD head）；開啟 complex 分支為 376,254
- **Latency**: 16ms lookahead（`lookahead_frames=1`，gain 對應 3-frame window 中間幀）

### Feature preprocessing

目前 feature contract 為 `log_erb_dfn_mean_cplx_unit_0_4k_v8`：

```text
normalized complex STFT
├─ power → triangular ERB → 10*log10(E+1e-10)
│          → per-band EMA mean → (dB - mean) / 40 → model input
└─ bins 0..4 kHz → per-bin |X[k]| EMA
                    → real/imag / sqrt(EMA+eps) → optional branch (off by default)
```

ERB 路徑對穩態訊號會逐步趨近 0，這是 DeepFilterNet 公式的預期行為；但
DeepFilterNet 同時使用 complex feature，而本模型的出貨設定已關閉該分支。
因此純 ERB 模式不保留穩態頻譜的 absolute level、帶內細節或相位；這是刻意的
低算量 bake-off 設定，也是與 complex/hybrid 模型比較時必須納入解讀的限制。
`X[k] / sqrt(EMA(|X[k]|))` 不是完全 gain-invariant；穩態下輸入振幅放大 `a` 倍，
complex feature 約放大 `sqrt(a)` 倍。只有啟用 complex 分支時，模型才會使用這項
資訊。同一 stream 分 chunk 處理時必須傳遞 ERB EMA 與 GRU state；若 complex
分支啟用，還要傳遞 complex EMA。不同 WAV 之間必須重置。
兩路公式、state 初始值與先更新再輸出的順序精確對齊原作 `libDF`——
v3 曾額外保留 ERB `±5` 與 complex `±10` clip 作為部署數值安全界線，v4 拿掉了
這個 clip：對照 upstream `libDF::band_mean_norm_erb`/`band_unit_norm` 與本
repo 自己的 `AINR/DeepFilterNet2` port 皆確認兩者都不 clip，v4 起兩路皆是
byte-for-byte 忠實移植，沒有額外的安全界線。
公式來源：[Rikorose/DeepFilterNet `libDF/src/lib.rs`](https://github.com/Rikorose/DeepFilterNet/blob/main/libDF/src/lib.rs)。

Feature 常數在 `config.ini [feature]`，C 部署常數固定於 `process.h`。
修改後需同步兩邊並重訓。

## 環境安裝

```bash
# 建議使用 Python 3.9+
pip install -r requirements.txt
```

依賴套件：
- `torch` >= 1.13, < 2.9
- `torchaudio` >= 0.13, < 2.9（2.9 會忽略 `PCM_F`/32-bit WAV 參數）
- `soundfile` >= 0.12
- `numpy`
- `tqdm`

匯出 ONNX 額外需要（訓練不需要）：
```bash
pip install onnx onnxoptimizer onnxruntime
```

## 資料集準備 (DNS4)

本專案使用 [DNS Challenge 4](https://github.com/microsoft/DNS-Challenge) 資料集。下載後目錄結構如下：

```
datasets_fullband/
├── clean_fullband/              ← config.ini 的 speech_dir
│   ├── emotional_speech/
│   ├── french_speech/
│   ├── german_speech/
│   ├── italian_speech/
│   ├── read_speech/
│   ├── russian_speech/
│   ├── spanish_speech/
│   ├── vctk_wav48_silence_trimmed/
│   └── ...
├── noise_fullband/              ← config.ini 的 noise_dir
│   ├── audioset/
│   ├── freesound/
│   └── ...
└── impulse_responses/           ← config.ini 的 rir_dir (optional)
    ├── SLR26/
    ├── SLR28/
    └── ...
```

### 注意事項

- 所有音檔必須是 **`.wav` 格式**（程式只掃描 `*.wav`）
- 音檔可以是任意 sample rate，程式會自動 resample 到 16kHz
- **RIR 為 optional**：若不使用，可將 `rir_dir` 設為空字串或刪除該行。建議使用以提升 dereverberation 效果
- RIR 會自動用 Schroeder 積分法估算 RT60，只保留 `rt60_min` ~ `rt60_max` (預設 0.1s ~ 1.3s) 範圍內的 RIR
- **RIR RT60 快取**：首次掃描後會自動存成 `.rir_cache_*.json`，後續相同設定直接讀取，大幅加速初始化
- 不一定要用 DNS4，**任何符合上述結構的語音/噪音資料集都可以**（只要是 wav 檔放在對應目錄下）

## 設定 config.ini

訓練前需要修改 `config.ini` 中的資料路徑：

```ini
[paths]
speech_dir = /your/path/to/datasets_fullband/clean_fullband
noise_dir = /your/path/to/datasets_fullband/noise_fullband
rir_dir = /your/path/to/datasets_fullband/impulse_responses
output_dir = ./output
```

其他常用參數：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `[training] epochs` | 100 | 訓練 epoch 數 |
| `[training] batch_size` | 128 | Batch size |
| `[training] lr` | 3e-4 | Learning rate |
| `[training] device` | cuda | 訓練裝置 (`cuda` 或 `cpu`) |
| `[training] epoch_size` | 0 | 每 epoch sample 上限；0 表示使用全部資料 |
| `[audio] segment_sec` | 3.0 | 每筆訓練音檔長度 (秒) |
| `[rir] p_rir` | 0.5 | 套用 RIR 的機率（共用 dataset config） |
| `[feature] erb_norm_tau_sec` | 1.0 | ERB per-band mean EMA 時間常數 |
| `[feature] erb_norm_init_lo_db` | -20 | 最低 ERB band EMA 初值 |
| `[feature] erb_norm_init_hi_db` | -45 | 最高 ERB band EMA 初值 |
| `[feature] erb_norm_scale_db` | 40.0 | mean-subtracted ERB 縮放尺度 |
| `[feature] spec_max_hz` | 4000 | complex branch 頻率上限 |
| `[feature] spec_norm_tau_sec` | 1.0 | complex per-bin magnitude EMA 時間常數 |

改動任何 feature 常數都需要同步修改 `process.h` 並重新訓練。Checkpoint 會保存
`feature_version` 與所有 normalization 常數；train resume、denoise 和 ONNX export
都會拒絕缺少版本的舊 checkpoint，或拒絕與 runtime config 不一致的 checkpoint。
匯出的 ONNX model metadata 也會帶上相同 feature contract，供部署端檢查。

> v1/legacy ERB-only 與 v2 absolute-ERB checkpoint/ONNX 都無法沿用。v3、v4、v5
> 雖然 input shape 不變，normalization semantics 已改變，必須重訓後重新匯出
> (v4 拿掉了 v3 的 erb_norm_clip/spec_clip；v5 修正 erb_bandborder() 的最小
> band 寬度保證，改變 erb_fwd/erb_inv 矩陣本身，同樣是 semantics 改變)。

## 訓練

目前訓練只接受共用 `../dataset_gen/pack_dataset.py` 產生的 raw
noisy/clean WAV tensor；STFT、log-ERB 與 runtime normalization 會在訓練時
即時計算。資料增強、指定 sample-rate 的 WAV generation、resample 與
packing 全部由 `../dataset_gen/` 維護，RNNoise-ERB 不再保存重複版本。

訓練 objective 已改回直接 ERB IRM：對 22 個 band gain 與 ideal ratio
mask 在 `gamma=0.5` 的壓縮域做 MSE，對齊原始 RNNoise 論文採用的 perceptual
compromise。目前 MRSL 的兩個 factor 都是 0；
沒有假 VAD/activity 加權。低於 `energy_floor` 的 mixture band 無法定義
ratio，會被 mask 掉而不產生梯度。若要重開 MRSL，必須先依實測 gradient
norm 重新縮放 IRM；直接恢復 factor 500 會讓 MRSL 壓過 IRM。

```bash
# Step 0: 建立 dataset_gen 專用設定並填入 speech/noise/RIR paths
cp ../dataset_gen/config.example.ini ../dataset_gen/config.ini

# Step 1: 產生 RNNoise-ERB 專用的 16 kHz dataset
python3 ../dataset_gen/gen_dataset.py \
    --config ../dataset_gen/config.ini --output data_16k --hours 25 \
    --sample-rate 16000 --workers 4

# Step 2: 打包
python3 ../dataset_gen/pack_dataset.py \
    --input data_16k/pairs --output data_16k/packed.pt --dtype float16

# Step 3: 訓練
python3 train.py --config config.ini --packed-data data_16k/packed.pt

# 可選：指定 GPU、降低 RAM、或載入同一目錄下多個 packed files
python3 train.py --config config.ini \
    --packed-dir data_16k/packed_shards --gpu 0 --mmap
```

`--mmap` 模式不會在 `__getitem__` 逐筆展開成 float32；FP16 batch 送到 GPU
後才轉 float32。訓練會隨機化資料區塊與區塊內順序，避免大型 shard 的全域
random access page fault。`config.ini` 的 `mmap_block_size`、
`mmap_num_workers`、`prefetch_factor` 可依磁碟速度與共用 RAM 調整；預設
256 / 2 / 2 是偏向低 RAM 的起點。

`../dataset_gen/gen_dataset.py` 常用參數：

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `--hours` | 8.3 | 目標音檔總時數；最多只向上補一個完整 segment |
| `--output` | `data` | 輸出目錄 |
| `--sample-rate` | config | 此次 generation rate；RNNoise 使用 16000 |
| `--workers` | 4 | DataLoader workers；0 表示單程序 |
| `--resume` | false | 從 `pairs/` 最大編號後繼續寫入 |
| `--seed` | config | 負數為每次 OS-random；非負數可重現 |

### 從斷點續訓

訓練中途中斷後，可以從最後的 checkpoint 繼續：

```bash
python3 train.py --config config.ini --packed-data data_16k/packed.pt \
    --resume output/rnnoise_epoch5.pth
```

續訓會恢復：model weights、optimizer 狀態、scheduler 狀態、epoch 計數、best_val_loss。
Checkpoint 同時檢查 loss contract；舊的 MRSL+IRM checkpoint 不能直接 resume，
必須從新初始化的訓練開始。

### 訓練輸出

訓練結果儲存在 `output/` 目錄：

```
output/
├── rnnoise_epoch1.pth      # 每 epoch 的 checkpoint
├── rnnoise_epoch2.pth
├── ...
└── rnnoise_best.pth        # 最佳 validation loss 的模型
```

## 推論 (Denoise)

對單一音檔進行降噪：

```bash
python3 inference.py --config config.ini --model output/rnnoise_best.pth \
                   --input noisy.wav --output clean.wav
```

`--pf-beta` 預設為 0（關閉），避免在還沒驗證 raw gain 前再加劇抑制。

`--atten-lim <dB>` 限制最大抑制量，做法對齊 Rikorose/DeepFilterNet
`enhance.py` 的 `--atten-lim`/`-a`：把 gain 往 1.0（不抑制）方向線性混合，
`lim = 10^(-|atten_lim_db|/20)`，`gain' = lim + gain*(1-lim)`——例如
`--atten-lim 12` 保證輸出最多只壓 12dB，其餘 noise floor 會保留。這在 ERB
band 層級套用（`rnnoise_apply_atten_lim`，C 端同名函式對應),
數學上等價於直接對 spectrum 做同樣的線性混合(因為 mode=1 反向矩陣是
partition of unity),但成本只需 22 次而非 257 次。預設不啟用(`None`,
維持目前最大抑制行為);純推論後處理,不影響 checkpoint/feature contract。
若要定位「全壓成 0」是 feature、model 或 post-filter 造成：

```bash
python3 inference.py --config config.ini --model output/rnnoise_best.pth \
  --input failing.wav --output debug.wav --dump-debug debug/failing.npz
```

dump 含 `erb_db`、兩路 features、`raw_gains`、`post_gains`
與輸入/輸出 waveform。這是 debug 用；checkpoint version gate 則是部署防呆。

## ONNX 匯出

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
與三組 GRU hidden state；輸出中心 frame 的 gains 與更新後 hidden state。
加速器本身不保存 state：host 以 `RNNoiseModelState` 保存三組 hidden，並將
每次的 `h1_out/h2_out/h3_out` 回送為下一次的 input。三個 feature frames
是輸入 temporal kernel 的完整可視範圍，不代表一次輸出三個 frames。
`rnnoise_model_state_commit()` 會先檢查三組輸出皆為 finite；失敗時回傳
`-1` 並保留前一個 state，呼叫端應 fail-open 該 frame，不得回寫壞狀態。
匯出的 metadata 帶 `state_layout_version`，數值與 `process.h` 的
`RNNOISE_MODEL_IO_LAYOUT_VERSION` 相同，整合端可據此拒絕 state layout
已經對不上自己配置的 struct 的圖。

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
# 匯出所有格式 (npy + C header)
python3 export_erb_matrix.py --config config.ini --format all
```

## 檔案說明

| 檔案 | 說明 |
|------|------|
| `train.py` | 模型定義與 packed raw-WAV tensor 訓練迴圈 |
| `inference.py` | 推論與 calibration 入口 |
| `export_onnx.py` | ONNX 匯出（streaming 推論格式） |
| `export_erb_matrix.py` | ERB 矩陣匯出（npy / C header） |
| `config.ini` | 所有超參數設定 |
| `process.c` / `process.h` | C 前後處理實作（嵌入式部署用） |
| `tests/` | Python feature、loss、checkpoint 與 Python/C parity tests |
| `../dataset_gen/` | 共用 augmentation、WAV generation、resample、packing 與 packed loader |
| `requirements.txt` | Python 依賴套件 |

## 表格 drift-guard(C 部署表格的兩層契約)

`rnnoise_tables_gen.h` 的編譯期常數表(ERB fwd/inv、nfftborder、Hann window)
有兩層 drift 契約(已接進 `make test-tables`):

```bash
make test-tables   # 兩層都建置+執行,任一 FAIL 即非零退出
```

- **Layer 1(canonical,預設)**:獨立重算 vs header 逐 byte `memcmp`——
  本機/pinned CI 工具鏈的 bit-exact 契約。
- **Layer 2(portable,`-DRNN_TABLES_PORTABLE`)**:數學性質檢查(有限值、
  nfftborder 單調且端點釘死、erb_inv partition-of-unity、erb_fwd 端帶 2x
  關係、Hann 落在 [0,1] 且滿足公式對稱)+ 實測定界的 ULP 門檻——
  **recompute-vs-table 為 256 ULP、Hann 鏡像對稱為 4096 ULP(實測 434)**,
  兩者是不同檢查的不同門檻,皆為 garbage-detector 而非位元契約。

## Feature tests

```bash
make test                 # tables + config/Python/C constants + C feature-state contract
make test-feature-python  # 需 torch training environment
```

C test 會以獨立 recurrence 對照 `rnnoise_compute_features()`，並讓固定非平坦
spectrum 運行 4096 frames，確認 ERB mean norm 收斂為 0，complex preprocessing
仍非零。Python test 另驗證兩路 state/chunk equivalence、ERB 穩態收斂、complex
穩態可觀測性，以及純 ERB／可選 complex 兩種 model forward contract；
`tests/test_python_c_features.py` 以 golden vectors 對照 Python/C 的 STFT、
兩路特徵與最終 normalization state。即使出貨模型未使用 complex 分支，保留的
可選路徑仍須通過這些測試，避免日後重新啟用時出現 Python/C 漂移。
