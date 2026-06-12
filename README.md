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
ifft/fft 來回**。關鍵設計是 **`min(G_nr, G_res)` 合併**：NR 只管雜訊抑制（`G_nr`，
純 OM-LSA），AEC 自己那顆**近端感知**的 AEC3 殘響回音 gain（`G_res = GainToNoAudibleEcho`）
負責回音。逐 bin 取兩者較小值 —— 近端 bin `G_res≈1`→ 留住 `G_nr`；回音 bin `G_res<G_nr`
→ 撿回回音抑制，且**沒有 product 的雙重壓抑**（雙講近端不被雙殺）。再用 per-bin 近端保留
floor 在「沒有回音」的頻段把 gain 抬回 1.0。

```
mic ─►HPF─┐                                          (HPF 只在 mic path)
          ├─► Linear AEC (PBFDKF + Shadow) ─► E(f) + AecResContext{ R², res_gain, CNG }
ref ──────┘   (ref 不經 HPF · 時域進，頻域出)         │
                                                    ▼
                              NR  (OM-LSA / MCRA noise est.)  →  G_nr(f)  純雜訊抑制
                                                    │
                                                    ▼
                              g_total = min( G_nr , G_res )
                              (撿回 AEC3 近端感知回音 gain；近端 bin G_res≈1 → 保留近端)
                                                    │
                                                    ▼
                              per-bin 近端保留 floor  (ne_floor=0.4)
                              gate = res_gain · (1 − R²/|E|²)
                              無回音處把 g_total → 1.0，保住近端語音
                                                    │
                                                    ▼
                              S(f) = E(f) · g_total  (+ CNG) ─► iFFT + OLA ─► output[n]
```

| 階段 | 角色 | 頻域 seam |
|------|------|-----------|
| **Linear AEC** | PBFDKF + Shadow + EPC，只做線性消除 | 輸出 `E(f)` + `AecResContext`（R²、res_gain、CNG、ERLE、DT 指標） |
| **NR** | OM-LSA，純雜訊抑制 | 吃 `E(f)`，輸出 `G_nr(f)`（無 re-FFT） |
| **RES (合併+floor)** | `min(G_nr, G_res)` + 近端保留 + CNG | `S(f)=E(f)·min(G_nr,G_res)`，無回音處抬回 1.0；單一 sqrt-Hann OLA |

> 每個階段都是可替換的 freq-in/freq-out block：NN 之後可單獨換掉其中一塊（NN-residual /
> NN-NR / NN-joint），介面就是同一份 `AecResContext`。詳見
> [docs/architecture.html](docs/architecture.html) 與
> [docs/freq_domain_pipeline_design.md](docs/freq_domain_pipeline_design.md)。

## 處理鏈 (Pipeline)

AEC(linear) → NR(OM-LSA) → RES `min(G_nr, G_res)` + 近端保留（單次 FFT/OLA）：
```bash
cd Audio_ALG
python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav --preset balanced
# 預設 --combine min (A_min_pl)；--legacy-v0 還原 v0 echo-aware joint
```

### Blind Test 成績（AEC Challenge Interspeech 2021, 800 cases · no-prealign）

**AEC(balanced) alone vs AEC(balanced) + NR + RES**，NR 兩個 strength preset：
**balanced**（`--nr-gain -15`）與 **mild**（`--nr-gain -10`，近端優先）。RES 用 `min(G_nr, G_res)`。
本地 ONNX AECMOS/DNSMOS，**無 harness pre-align** —— 讓 AEC 自己的線上 `EchoPathDelayEstimator`
對齊（真實情境；offline 全訊號 GCC-PHAT pre-align 是已退役的 crutch，會讓 movement 分數偏樂觀）。

**AECMOS**（echo↑ / deg↑）:

| 指標 | AEC-alone | +NR(balanced) | +NR(mild) |
|------|----------|---------------|-----------|
| FS_static echo   | 3.504 | **3.559** | 3.552 |
| FS_movement echo | 3.471 | **3.500** | 3.495 |
| DT_static echo   | 4.203 | 4.201 | 4.199 |
| DT_static deg    | 2.080 | **2.156** | 2.150 |
| DT_movement deg  | 2.193 | **2.233** | **2.233** |
| NE deg           | 4.052 | 4.007 | **4.029** |

**DNSMOS**（SIG/BAK/OVRL↑；BAK = 背景品質，NR 本職，AECMOS 看不到）:

| 指標 | AEC-alone | +NR(balanced) | +NR(mild) |
|------|----------|---------------|-----------|
| FS_static BAK   | 2.632 | **2.817** | 2.778 |
| FS_movement BAK | 2.366 | **2.578** | 2.535 |
| DT_static BAK   | 2.823 | **2.991** | 2.966 |
| DT_movement BAK | 2.725 | **2.903** | 2.871 |
| NE BAK          | 3.848 | 3.867 | **3.869** |
| NE SIG          | 3.472 | 3.411 | **3.431** |
| DT_static SIG   | 2.332 | **2.377** | **2.377** |
| FS_static SIG   | 1.730 | **1.806** | 1.792 |
| FS_movement SIG | 1.573 | **1.663** | 1.652 |
| DT_movement SIG | 2.292 | **2.334** | 2.333 |

> **NR pipeline 對 AEC-alone**：背景 **BAK +0.17~0.21**（降噪本職，AECMOS 量不到）、DT/FS echo & deg
> 全贏，只賠純 NE SIG（−0.06）。`min(G_nr, G_res)` 撿回 AEC3 的近端感知回音 gain，修好 v0 的
> DT echo <4.0（v0：DT echo 3.97 / DT deg +0.30，落在被支配的 Pareto 角落，`--legacy-v0` 可還原）。
> **balanced vs mild**：mild（g_min −10）把 NE 救回（deg **+0.022** / SIG **+0.020**），代價是 FS/DT
> 背景降噪 **−0.03~0.04 BAK**。**balanced = 降噪優先，mild = 近端優先**（Pareto trade，無全贏）。
>
> AEC 模組單獨的完整 ours / AEC3 / Speex 三方對照見 [lib/aec/README.md](lib/aec/README.md)。
> 渲染器:`pipelines/rebench_aec_only.py`(AEC)、`pipelines/rebench_joint.py`(A_min_pl)，皆 `NO_PREALIGN=1`。
