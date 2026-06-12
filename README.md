# Audio_ALG - Audio Processing Algorithms Integration

整合音頻處理算法，包含降噪 (NR)、AI 降噪 (AINR) 和聲學回聲消除 (AEC) 模組。

## 項目結構

- **lib/** — 獨立算法模組（Git Submodules）
  - **[aec/](https://github.com/aaronhsueh0506/AEC)**: 聲學回聲消除
  - **[nr/](https://github.com/aaronhsueh0506/CVNR)**: 傳統降噪算法
- **ainr/**: AI 降噪模型
  - **RNNoise-ERB/**: RNNoise v0.2 架構 + ERB bands（16kHz, DNS4 dataset, DeepFilterNet-style augmentation）
- **pipelines/**: AEC + NR 串接處理鏈
- **shared/**: 共享工具和代碼
- **docs/**: 統一文檔
- **scripts/**: 管理腳本

## 快速開始

### 克隆項目（包含 submodules）

```bash
# HTTPS（推薦，不需要 SSH key）
git clone --recursive https://github.com/aaronhsueh0506/Audio_ALG.git

# SSH（需要設定 GitHub SSH key）
git clone --recursive git@github.com:aaronhsueh0506/Audio_ALG.git
```

如果已經 clone 但還沒初始化 submodules：
```bash
cd Audio_ALG
git submodule update --init --recursive
```

### 更新 Submodules 到最新版本

```bash
# 更新所有 submodules
./scripts/update_submodules.sh

# 或手動更新
git submodule update --remote lib/nr
git submodule update --remote lib/aec
```

## 開發工作流

### 獨立開發 NR
```bash
cd lib/nr/
git checkout -b feature/xxx
# ... 開發 ...
git push origin feature/xxx
```

### 獨立開發 AEC
```bash
cd lib/aec/
git checkout -b feature/xxx
# ... 開發 ...
git push origin feature/xxx
```

### 更新整合倉庫的 Submodule 引用
```bash
# 在 Audio_ALG 根目錄
git submodule update --remote lib/nr   # 拉取 NR 最新
git add lib/nr
git commit -m "update: NR submodule to latest"
git push
```

## 處理鏈 (Pipeline)

AEC 輸出接 NR 輸入的串接處理：

```bash
# malloc 版（標準用法）
make -C pipelines libs
make -C pipelines aec_nr_pipeline
./pipelines/aec_nr_pipeline mic.wav ref.wav out.wav [preset]

# 僅 AEC（不跑 NR）
./pipelines/aec_nr_pipeline mic.wav ref.wav out.wav balanced --aec-only
```

架構圖：[Overview](docs/pipeline_overview.png) · [Detailed](docs/pipeline_detailed.png)

---

## AEC Preset 設定

三個 preset 的差異**只有一個參數**：`min_gain_floor_far_active_db`（遠端活躍時的最低增益下限）。
其餘參數（filter length、Kalman Q、delay buffer、CNG 等）全部相同。

| Preset | `min_gain_floor_far_active_db` | 特性 |
|--------|-------------------------------|------|
| `gentle` | **−20 dB** | 近端優先：保留更多近端語音，允許較多殘留回聲；DT deg 接近 AEC2 |
| `balanced` | **−28 dB** | 預設值（生產用）：四項 ship bar 全過；FS echo >3.5、DT echo >4、DT deg >2、NE deg ≥4 |
| `aggressive` | **−38 dB** | 回聲優先：更深的回聲消除，近端損失較多；FS echo 超越 AEC3，deg 仍 >2 |

**共用參數（balanced 基準）：**

| 參數 | 值 | 說明 |
|------|----|------|
| `sample_rate` | 16000 Hz | |
| `filter_length` | 832 samples | 52 ms |
| `n_partitions` | 5 | 832/160 ≈ 5 |
| `mu` (Kalman init) | 0.3 | PBFDKF 初始步長 |
| `max_delay_ms` | 1024 ms | 最大延遲搜尋範圍 |
| `enable_highpass` | 1 | mic 路徑 80 Hz HPF（內建，不需外接） |
| `enable_saturation` | 1 | ref 路徑 soft-clip |
| `enable_delay_est` | 1 | GCC-PHAT 自動對齊 |
| `enable_shadow` | 1 | PBFDAF shadow filter（DT 偵測） |
| `enable_res` | 1 | AEC3 post-filter（REE + SuppressionGain + CNG） |
| `enable_cng` | 1 | Comfort Noise Generator |
| `shadow_mu_min` | 0.5 | shadow filter 最小步長 |
| `warmup_frames` | 100 | 10 ms × 100 = 1 s 暖機 |

---

## NR Preset（Mode）設定

NR 有三個 mode，透過 `MmseLsaNrMode` 或 `--nr-gain` 控制最低增益：

| Mode | `g_min_db` | `q` (SPP) | `xi_min_db` | `alpha_d` | `alpha_g` | 特性 |
|------|-----------|-----------|-------------|-----------|-----------|------|
| `MILD` | **−10 dB** | 0.60 | −15 dB | 0.85 | 0.92 | 近端優先：最低壓制；適合近距離說話場景 |
| `BALANCED` | **−15 dB** | 0.50 | −20 dB | 0.70 | 0.88 | 預設值：噪聲抑制與語音保留的平衡點 |
| `AGGRESSIVE` | **−20 dB** | 0.35 | −25 dB | 0.50 | 0.75 | 噪聲優先：強力壓制；語音可能有些許洩漏感 |

**參數說明：**

| 參數 | 說明 |
|------|------|
| `g_min_db` | 最低增益下限（power domain：實際幅度下限 = 10^(g_min/20)） |
| `q` | SPP 先驗語音存在機率（高 q → 偏向語音，壓制保守） |
| `xi_min_db` | 先驗 SNR 最小值（低值 → 可在更低 SNR 下壓制） |
| `alpha_d` | 噪聲追蹤 IIR 係數（高 α → 慢追蹤，穩態噪聲好；低 α → 快追蹤，突發噪聲好） |
| `alpha_g` | 增益平滑係數（高 α → 平滑，少 musical noise；低 α → 快響應） |

**共用參數（16 kHz）：**

| 參數 | 值 | 說明 |
|------|----|------|
| `frame_size` | 320 samples | 20 ms |
| `hop_size` | 160 samples | 10 ms |
| `fft_size` | 512 | next pow2 ≥ frame_size |
| `n_freqs` | 257 | fft_size/2 + 1 |
| `alpha_xi` | 0.88 | 先驗 SNR decision-directed 平滑 |
| `alpha_s` | 0.95 | MCRA 能量平滑 |
| `alpha_p` | 0.20 | MCRA SPP 指示平滑 |
| `L` | 32 frames | 320 ms min-stat 窗（MCRA） |
| `delta_db` | 10 dB | MCRA 語音存在門限 |
| `num_init_frames` | 20 | 噪聲估計初始化幀數（200 ms） |
