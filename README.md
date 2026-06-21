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

三階段串接，mic 路徑共享一個 16 kHz / FFT-512 / hop=160 (10 ms) / sqrt-Hann 格點：

```
mic ─►HPF─┐
          ├─► Linear AEC (PBFDKF + Shadow + EPC) ─► e[n]  +  AecResContext
ref ──────┘   (ref 不經 HPF；時域進，頻域 context 出)          │
                                                             ▼
                              NR  (MMSE-LSA + MCRA / OM-LSA SPP)
                              g_nr = MMSE-LSA gain per bin
                                                             │
                                                             ▼
                              RES  (Residual Echo Suppression)
                              g_total = min(g_nr, g_res)     ◄── g_res from AecResContext
                              → S(f) = e[n] · g_total  +  CNG ─► iFFT + OLA ─► output[n]
```

| 階段 | 角色 | 輸出 |
|------|------|------|
| **Linear AEC** | PBFDKF + Shadow + EPC，純線性消除，不套 post-filter | `e[n]` 時域 + `AecResContext`（echo PSD、far power、dt_indicator、ERLE） |
| **NR** | MMSE-LSA（MCRA noise est + OM-LSA SPP），壓背景噪聲 | 增強語音 + per-bin gain `g_nr` |
| **RES** | `g_total = min(g_nr, g_res)`，逐 bin 取小值；無回音處 g_res≈1 → NR 正常發揮 | 最終輸出 + CNG |

```bash
# malloc 版（標準用法）
make -C pipelines libs
make -C pipelines aec_nr_pipeline

# AEC preset 為 positional（gentle/balanced/aggressive）；NR 用 --nr-preset（mild/balanced/aggressive）
./pipelines/aec_nr_pipeline mic.wav ref.wav out.wav balanced --nr-preset balanced
./pipelines/aec_nr_pipeline mic.wav ref.wav out.wav aggressive --nr-preset aggressive

# 僅 AEC（不跑 NR/RES）
./pipelines/aec_nr_pipeline mic.wav ref.wav out.wav balanced --aec-only
```

### Python 用法（演算法開發 / 驗證）

Python 參考實作與上面的 C 版**同演算法**。先初始化 submodule（`git submodule update --init --recursive`），相依 `numpy` / `soundfile`（macOS 用 `python3`）。從 **Audio_ALG 根目錄**以 module 形式跑（腳本會自動把 `lib/aec/python` 與 `lib/nr` 加進 path）：

```bash
cd Audio_ALG

# 單檔（freq A_min_pl：AEC 線性 → NR(E) → RES，單一 FFT；g_total=min(g_nr,g_res)）
python3 -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav

# 各自指定 AEC / NR preset
python3 -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav --aec-preset balanced --nr-preset aggressive

# 僅 AEC（跳過 NR/RES）
python3 -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav --aec-only
```

**主要參數（CLI 只開放 preset + 開關）：**

| 參數 | 預設 | 說明 |
|------|------|------|
| `--mic` `--ref` `--output` | (必填) | mic 輸入 / 參考(喇叭) / 輸出 WAV |
| `--aec-preset` | `balanced` | `gentle` / `balanced` / `aggressive`（見下方 AEC Preset 表；差異只在 min-gain floor）|
| `--nr-preset` | `balanced` | `mild` / `balanced` / `aggressive`（見下方 NR Preset 表）|
| `--aec-only` | off | 只跑 AEC，跳過 NR/RES |

> Pipeline 固定走生產的 **freq A_min_pl** 路徑（PBFDKF、`g_total=min(g_nr,g_res)`、per-bin 近端 floor）。低階旋鈕（mu / ne-floor / combine / pipeline-mode / aec-mode）與 legacy v0 已移除，改由 preset 決定。

**800-case Blind Test（render + 本地 AECMOS/DNSMOS 評分）：**

```bash
cd Audio_ALG

# A) 直接 render 三個 scenario（farend / nearend / doubletalk）
python3 -m pipelines.eval_pipeline_blind <aec_challenge_blind/> --preset balanced --nr-preset balanced -o out_pipeline/

# B) ship 基準的 A_min_pl 並行 renderer（上方成績表就是用這支）
NE_FLOOR=0.4 NE_GATE=both REBENCH_WORKERS=8 python3 -m pipelines.rebench_joint <out_dir> [ne_floor] [ne_gate] [limit]
# 評分：見 rebench_joint.py docstring（用 AEC repo 的 bench_aecmos.py 對照 echo/deg）
```
> ✓ 所有 renderer 皆 **真 no-prealign**（offline pre-align 已移除，靠 AEC 線上 matched-filter 對齊）。
> ⚠ `rebench_joint` 預設 **skip-if-exists**：`<out_dir>` 有舊版 render 會被當 ok 跳過 → 換版本前先刪掉重 render。AECMOS/DNSMOS 評分需 `speechmos` + `onnxruntime≤1.16.3` + `numpy<2` 的 venv。

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

NR 有三個 preset（pipeline `--nr-preset`；C 端 `MmseLsaNrMode` / `mmse_lsa_config_for_mode`）：

| Mode | `g_min_db` | `q` (SPP) | `xi_min_db` | `alpha_d` | `alpha_g` | 特性 |
|------|-----------|-----------|-------------|-----------|-----------|------|
| `MILD` | **−10 dB** | 0.60 | −15 dB | 0.85 | 0.92 | 近端優先：最低壓制；適合近距離說話場景 |
| `BALANCED` | **−15 dB** | 0.50 | −20 dB | 0.70 | 0.88 | 預設值：噪聲抑制與語音保留的平衡點 |
| `AGGRESSIVE` | **−20 dB** | 0.35 | −25 dB | 0.50 | 0.75 | 噪聲優先：強力壓制；語音可能有些許洩漏感 |

> **Python pipeline vs C**：Python `--nr-preset` 套用 strength 四元組 **{g_min, q, xi_min, alpha_g}**；`alpha_d` / `L` 維持 pipeline 自己的結構性 tuning（`alpha_d≈0.95` / `L=150`，為 AEC 殘餘信號調過，見 `_build_denoiser`），不隨 preset 變。C 的 `config_for_mode` 另含 `alpha_attack/alpha_decay/alpha_d`。`balanced` = 既有 shipped 值（pipeline byte-equal）。

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

---

### Blind Test 成績（AEC Challenge Interspeech 2021, 800 cases · **真 no-prealign**）

**AEC(balanced) alone vs AEC(balanced) + NR + RES**（RES 用 `min(G_nr, G_res)`）。四欄 =
AEC 單獨、+NR 三個 preset（`--nr-preset balanced|mild|aggressive`，皆完整 strength 參數組）。

> **本表 2026-06-20 以 AEC v3.23.0（no-PA matched-filter pre-echo fix + 預設 ON 的 DT-deg recovery stack）全部重跑、真 no-prealign 產生。**
> AEC-alone = `AEC(balanced)` 含自身 AEC3 post-filter 的完整輸出（= AEC repo 標準產品，**非**純線性）；
> 已與 AEC/ repo 交叉驗證一致（DT_static deg 2.075 ≈ AEC/ 2.074、DT echo 4.217 ≈ 4.218）。

本地 ONNX AECMOS/DNSMOS，**完全無 pre-align** —— 純靠 AEC 線上的 matched-filter `EchoPathDelayEstimator`
自對齊（與 AEC3 內部對齊機制同類，也是生產真實情境）。offline 全訊號 GCC-PHAT pre-align 是已退役的
crutch：它不只灌高 echo 分數，跟線上 matched filter 併用還會 double-alignment、在部分 movement case
反而鎖到 phantom peak 害 ERLE 崩掉（見 AEC `CHANGELOG [3.22.1]`）。

**AECMOS**（echo↑ / deg↑；echo 由 AEC+RES 決定，NR 近乎中性，故不標記）:

| 指標 | AEC-alone | +NR(balanced) | +NR(mild) | +NR(aggressive) |
|------|----------|---------------|-----------|-----------------|
| FS_static echo   | 3.525 | 3.494 | 3.498 | 3.465 |
| FS_movement echo | 3.508 | 3.469 | 3.477 | 3.424 |
| DT_static echo   | 4.217 | 4.181 | 4.186 | 4.196 |
| DT_movement echo | 4.112 | 4.095 | 4.098 | 4.107 |
| DT_static deg    | 2.075 | **2.277** | 2.223 | 2.261 |
| DT_movement deg  | 2.141 | **2.281** | 2.246 | 2.255 |
| NE deg           | 4.024 | 4.033 | **4.053** | 3.952 |

**DNSMOS**（SIG/BAK↑；BAK = 背景品質 = NR 本職，AECMOS 量不到；每列粗體為最佳）:

| 指標 | AEC-alone | +NR(balanced) | +NR(mild) | +NR(aggressive) |
|------|----------|---------------|-----------|-----------------|
| FS_static BAK   | 2.653 | 2.861 | 2.784 | **2.982** |
| FS_movement BAK | 2.376 | 2.624 | 2.521 | **2.771** |
| DT_static BAK   | 2.871 | 3.070 | 3.003 | **3.143** |
| DT_movement BAK | 2.741 | 2.955 | 2.883 | **3.033** |
| NE BAK          | 3.843 | **3.898** | 3.887 | **3.898** |
| NE SIG          | **3.464** | 3.410 | 3.441 | 3.370 |
| DT_static SIG   | 2.381 | **2.429** | 2.425 | 2.419 |
| DT_movement SIG | 2.301 | **2.355** | 2.348 | 2.335 |
| FS_static SIG   | 1.791 | 1.898 | 1.865 | **1.900** |
| FS_movement SIG | 1.584 | 1.667 | 1.640 | **1.688** |

> **NR vs AEC-alone**：背景 **BAK +0.20~0.25（FS/DT）**（降噪本職，AECMOS 量不到）、DT 近端 deg 也明顯
> 改善（DT_static **+0.20** / DT_movement **+0.14**）；echo 由 AEC+RES 決定，NR 近乎中性（FS echo ±0.04 內）。
> 唯一代價是純 NE SIG（balanced −0.05）。`min(G_nr, G_res)` 撿回 AEC3 的近端感知回音 gain。
> **NR preset = 純噪聲抑制強度的 Pareto 軸**（不影響 echo/對齊）：
> - **aggressive** — 背景最乾淨（BAK 全列最高），近端最受損（NE SIG 3.370 最低、NE deg 3.952）。
> - **mild** — 近端最保留（NE SIG 3.441 / NE deg 4.053 最高），背景降噪最少。
> - **balanced** — 折衷（預設）。
>
> ✓ **v3.23.0 no-PA matched-filter 修正後，真 no-prealign 的 echo 也達到 ship bar**（AEC-alone FS echo
> 3.525 >3.5、DT echo 4.217 >4；+NR 近乎同值）—— 先前 no-prealign echo 偏低的落差已由 no-PA 延遲修正補上。
> AEC 模組 ours / AEC3 / Speex 三方對照（目前仍 pre-align，待重生）見 [lib/aec/README.md](lib/aec/README.md)。
> 渲染器:`pipelines/rebench_aec_only.py`(AEC-alone = AEC+RES)、`pipelines/rebench_joint.py`
> (A_min_pl;`NR_PRESET=mild|aggressive` 切 preset)。**offline pre-align 已從所有 pipeline 腳本移除**
> (含 `eval_pipeline_blind.py`),一律走 AEC 線上 matched-filter 對齊(=生產真實情境),不再有
> `NO_PREALIGN` / `PREALIGN` 開關。
