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

整路 **AECMOS**（本地 ONNX，echo↑ deg↑），**無 harness pre-align** —— 讓 AEC 自己的線上
`EchoPathDelayEstimator` 對齊（真實情境；offline 全訊號 GCC-PHAT pre-align 是已退役的 crutch，
會讓 movement 分數偏樂觀）。比較 **有無 NR**（AEC 單獨 vs AEC+NR 完整 pipeline）：

| 指標 | AEC（無 NR） | AEC+NR | 變化 |
|------|-------------|--------|------|
| FS_static echo↑   | 3.504 | 3.416 | −0.089 |
| FS_movement echo↑ | 3.471 | 3.390 | −0.081 |
| NE deg↑           | 4.052 | 4.014 | −0.039 |
| DT_static echo↑   | 4.203 | 3.970 | −0.233 |
| DT_static deg↑    | 2.080 | **2.381** | **+0.301** |
| DT_movement echo↑ | 4.107 | 3.918 | −0.189 |
| DT_movement deg↑  | 2.193 | **2.469** | **+0.277** |

> **NR 在 AECMOS 上的貢獻 = 雙講近端保護（DT deg +0.28～0.30）**，代價是 **DT echo −0.19～0.23**。
> 這是**結構性的 Pareto trade,不是 bug**:在 DT,回音與近端落在同一個 bin —— 保住近端就會漏一點
> 回音(root cause:joint 用 MMSE-LSA 的 `ξ=Ŝ²/(N²+R²)` gain,DT 回音 bin 的 g≈0.22 ≫ AEC 專用
> echo gain 0.10,近端把 ξ 灌大壓不下去)。把專用 echo gain 加回去能救 DT echo,但 DT deg 立刻掉回
> AEC-alone 水準 —— 同一條 Pareto 線。NE deg 幾乎持平(−0.04):AEC 單獨本來就有 4.05,NR 沒有
> 「救回」什麼。
>
> ⚠️ **AECMOS 不量背景噪音抑制** —— 那才是 NR 的本職。本表只呈現 echo/近端的 trade,**看不到 NR
> 真正的降噪價值**(背景品質 / musical-noise 等需 DNSMOS BAK 或聽感評估)。
>
> AEC 模組單獨的完整 ours / AEC3 / Speex 三方對照見 [lib/aec/README.md](lib/aec/README.md)。
> 渲染器:`pipelines/rebench_aec_only.py`(AEC)、`pipelines/rebench_joint.py`(AEC+NR),皆 `NO_PREALIGN=1`。
