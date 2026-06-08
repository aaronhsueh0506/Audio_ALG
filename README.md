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

## 系統架構 (System Architecture)

整路共享一個 16 kHz / fft=512 / hop=160 (10 ms) / sqrt-Hann (COLA) 的頻域格點。
AEC 做完線性消除後，把頻域 seam（`AecResContext`）直接交給 NR 與 RES，**不再各自
ifft/fft 來回**。關鍵設計是 **echo-aware joint gain**：把 AEC 估的殘響回音功率 R²
折進 NR 的雜訊地板，用**一個** MMSE-LSA gain 同時壓雜訊＋殘響回音，再用 per-bin
近端保留 floor 在「沒有回音」的頻段把 gain 抬回 1.0。

```
mic ─►HPF─┐                                          (HPF 只在 mic path)
          ├─► Linear AEC (PBFDKF + Shadow) ─► E(f) + AecResContext{ R², res_gain, CNG }
ref ──────┘   (ref 不經 HPF · 時域進，頻域出)         │
                                                    ▼
                              NR  (MMSE-LSA / MCRA noise est.)
                              ξ = S² / (N² + R²)   ◄── R² 折進雜訊地板
                              → 一個 per-bin joint gain  G(f)
                                                    │
                                                    ▼
                              per-bin 近端保留 floor  (ne_floor=0.4)
                              gate = res_gain · (1 − R²/|E|²)
                              無回音處把 G(f) → 1.0，保住近端語音
                                                    │
                                                    ▼
                              S(f) = E(f) · G(f)  (+ CNG) ─► iFFT + OLA ─► output[n]
```

| 階段 | 角色 | 頻域 seam |
|------|------|-----------|
| **Linear AEC** | PBFDKF + Shadow + EPC，只做線性消除 | 輸出 `E(f)` + `AecResContext`（R²、res_gain、CNG、ERLE、DT 指標） |
| **NR (joint)** | MMSE-LSA，把 R² 當額外雜訊一起壓 | 吃 `E(f)`，輸出 per-bin joint gain `G(f)`（無 re-FFT） |
| **RES floor** | 近端保留 + CNG | `S(f)=E(f)·G(f)`，無回音處抬回 1.0；單一 sqrt-Hann OLA 收尾 |

> 每個階段都是可替換的 freq-in/freq-out block：NN 之後可單獨換掉其中一塊（NN-residual /
> NN-NR / NN-joint），介面就是同一份 `AecResContext`。詳見
> [docs/architecture.html](docs/architecture.html) 與
> [docs/freq_domain_pipeline_design.md](docs/freq_domain_pipeline_design.md)。

## 處理鏈 (Pipeline)

AEC(linear) → echo-aware NR → 近端保留 RES（單次 FFT/OLA，joint gain）：
```bash
cd Audio_ALG
python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav --preset balanced
# joint echo-aware 為預設；可調 --ne-floor 0.4 / --ne-gate both
```

### Blind Test 成績（AEC Challenge Interspeech 2021, 800 cases, balanced）

整路 **AECMOS**（本地 ONNX，整合 pipeline harness，含 250 ms 全域 pre-align，echo↑ deg↑）。
比較**舊串接（classic：AEC 內建 RES → 時域 NR）**與**現行 joint echo-aware**：

| 指標 | classic 串接 | **joint echo-aware** | 變化 |
|------|-------------|----------------------|------|
| FS_static echo↑   | 2.739 | 3.084 | **+0.345** |
| FS_movement echo↑ | 2.686 | 3.108 | **+0.422** |
| NE deg↑           | 3.466 | **4.015** | **+0.549** |
| DT_static deg↑    | 2.420 | 2.686 | **+0.266** |
| DT_movement deg↑  | 2.448 | 2.745 | **+0.297** |
| DT_static echo↑   | 4.029 | 3.799 | −0.230 |
| DT_movement echo↑ | 3.933 | 3.763 | −0.170 |

> **joint echo-aware 修好了舊串接的 NE deg 掉分**：串接版把 NR 排在 AEC 後，NR 在
> 「沒有回音」的純近端場景仍套 gain 壓近端語音 → NE deg 只剩 3.466。joint 版把 R²
> 折進雜訊地板＋per-bin floor，無回音處不壓近端 → **NE deg 救回 4.015（+0.549）**，
> 同時 FS echo（一個 joint gain 壓得更乾淨）與 DT deg 一起提升；代價是 DT echo 微降
> 0.17–0.23（Pareto，仍 >3.76）。
>
> 這是**整合 harness** 的相對 A/B。**AEC 模組單獨的 ship 數字**（FS_movement 3.512 > 3.5
> 硬門檻、NE deg 4.047、DT echo 4.08–4.20）用的是 AEC repo 自己的 bench，見
> [lib/aec/README.md](lib/aec/README.md)；兩套 harness 的 pre-align 不同，FS 絕對值不可直接互比。
