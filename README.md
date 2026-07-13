# Audio_ALG - Audio Processing Algorithms Integration

> **C 使用與整合**：[Audio_ALG C User Manual（繁體中文）](docs/c_user_manual_zh_TW.md)

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

AEC（線性）→ NR → 增益合併（min），mic 路徑共享一個 16 kHz / FFT-512 / hop=160 (10 ms) / sqrt-Hann 格點。
注意 RES **不是** NR 之後另一個 filter：`g_res` 是線性 AEC **內部**算出的 AEC3 SuppressionGain，跟 `E(f)` 一起從
seam 輸出，最後在頻域跟 `g_nr` 取 `min` 合併、**一次**套用到 `E(f)`：

```
mic ─►HPF─┐
ref ──────┴─► Linear AEC (PBFDKF + Shadow + EPC，純線性、不套 post-filter)
                 │
                 ├─► E(f)            線性殘差（頻域 seam）
                 └─► AecResContext ─► g_res（＝RES＝AEC3 SuppressionGain）、R²、CNG(N²)、far_power
                        │
   E(f), R² ─► NR (MMSE-LSA + MCRA/OM-LSA SPP，echo-aware ξ=S²/(N²+R²)) ─► g_nr
                        │
        g_nr ─┐
        g_res ┴─► min ─► g_total ─(＋ far/near 雙閘 ne_floor)─► S(f)=E(f)·g_total ＋ CNG ─► iFFT+OLA ─► output[n]
```

| 階段 | 角色 | 輸出 |
|------|------|------|
| **Linear AEC** | PBFDKF + Shadow + EPC，純線性消除，不套 post-filter | `E(f)` 頻域殘差 + `AecResContext`（`g_res`=AEC3 SuppressionGain、R²、CNG N²、far_power） |
| **NR** | MMSE-LSA（MCRA + OM-LSA SPP），壓背景噪聲；**echo-aware**（R² 折入 ξ=S²/(N²+R²)；`--legacy-amin`=純噪聲） | per-bin gain `g_nr` |
| **RES** | `g_total = min(g_nr, g_res)`（`g_res`＝AEC 內 AEC3 SuppressionGain，非獨立 filter）；無回音處 g_res≈1 → NR 正常發揮 | 最終輸出 + CNG |

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
| `--legacy-amin` | off | 還原 2026-06-23 前的 min-only A_min_pl（不注入 R²、scalar 近端 floor）|

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

## 增益合併設計（echo-aware 統一增益 + far/near 雙閘 ne_floor + CNG）

這節說明 NR 與 AEC「**怎麼結合**」——不是把 NR 當 AEC 後面另一個獨立 filter，而是讓
NR 的 gain `G_nr` 和 AEC3 內部的 `G_res` 在頻域以 **`min`** 合併成一條 `g_total`，**一次**
套用到線性殘差 `E(f)`。三個生產設計（2026-06-23 `far02_near` re-tune）依序如下。

### 1. Echo-aware 統一增益（`inject_echo_psd`，生產預設 ON）

普通 NR 的 a priori SNR 是 `ξ = S²/N²`（S=語音估計、N=噪聲底）。echo-aware 把線性 AEC
算出的**殘留回音 PSD `R²(f)`** 折進噪聲底：

```
ξ = S² / (N² + R²)        ← Speex/Habets「回音視為額外噪聲」的統一增益
```

- `R²` 來自 `ctx.r2`（AEC seam 輸出的殘留回音 PSD），已在跟 NR 噪聲底**相同的 |E|² 尺度**上
  （β_r=1），可直接相加。
- 效果：**同一條 `G_nr` 現在連殘留回音也壓**。回音 bin 的 `R²` 大 → ξ 掉 → `G_nr` 掉 →
  回音被 NR 一起壓掉；近端/穩態噪聲 bin 的 `R²≈0` → 退化回 `ξ=S²/N²`，行為不變。
- **為什麼 `min(G_nr, G_res)` 不直接丟掉 `G_res`**：`G_res` 是 AEC3 用**收斂後 filter** 算出
  的近端感知回音 gain，帶有 `G_nr` 看不到的資訊（線性 filter 收斂品質 / ERLE）；丟掉它
  echo 會崩。echo-aware 是「讓 NR **也**幫忙壓回音」，不是「取代 RES」。
- `--legacy-amin`（或 `LEGACY_AMIN=1`）還原成純噪聲 NR（不注入 R²）；default-OFF 時 byte-equal。
- 成績（見上表）：balanced FS echo **+0.07~0.10**、DT BAK **+0.06~0.07**，打破先前「echo 中性」。

### 2. far/near 雙閘 near-end floor（**0.4 / 0.2** 是什麼）

`min(G_nr, G_res)` 在純近端（無回音）會把近端語音也壓掉（NR 的 `g_min` 下限）。**ne_floor**
在「低回音 bin」把 `g_total` 拉回 1.0 保護近端：

```
echo_frac = R²/|E|²            ≈0 乾淨近端、≈1 回音主導
lift      = nf_eff · no_echo   （no_echo 由 ne_gate 決定，預設 both = g_res·(1−echo_frac)）
g_total   = (1 − lift)·g_total + lift·1.0
```

關鍵：`nf_eff` **不是固定值**，而是**依遠端是否活躍切換**——這就是 0.4 與 0.2：

| 場景 | `ctx.far_power` | `nf_eff` | 為什麼 |
|------|-----------------|----------|--------|
| **NE**（純近端：遠端靜音 + 近端有語音）| ≈0（< `far_gate_thresh`）| **0.4** | 沒有回音要打，降 floor 只會傷近端語音 → 用**高 floor 全力保護近端** |
| **FS / DT**（遠端活躍）| 有 burst | **0.2** | floor 原本在**過度保護背景噪聲**；v3.24.0 收斂更深後回音變少，**降 floor** 讓壓制把背景/殘響清乾淨 |

- gate：`ctx.far_power > far_gate_thresh`（NE≈0、FS/DT 有 burst → 乾淨分離兩種場景）。
- 再疊 near-VAD gate（`near_gate_thresh` + hangover）：只有「遠端靜音 **且** 近端真的有語音」
  才保 0.4；遠端靜音但近端也靜音（FS/DT 的噪聲空檔）→ 沒語音可傷 → 也降到 0.2 清背景。
- 0.4 / 0.2 是 800-case 調出的 operating point（`far02_near`）：NE 近端被保護、FS/DT 背景更乾淨、
  ship bar 全過。常數在 `pipelines/aec_nr_pipeline.py`（`PROD_NE_FLOOR=0.4` /
  `PROD_NE_FLOOR_FAR_ACTIVE=0.2`）；`--legacy-amin` 還原成單一 scalar floor（不分 far/near）。

### 3. CNG（comfort noise）只填回音消掉的 bin

最後 `S(f) = E(f)·g_total` 之後注入舒適噪聲，遮住「回音被消除留下的洞」：

```
noise_gain = sqrt(1 − g_res²)     ← 看 AEC 回音 gain g_res，不是 g_total
n_amp      = sqrt(comfort_noise)  ← AEC 估的背景底噪水平
S         += noise_gain · n_amp · gaussian
```

- 關鍵：`noise_gain` 用 **`g_res`（回音 gain）而非 `g_total`**。所以 CNG **只填回音被消掉的
  bin**（`g_res<1`），**完全不碰 NR 壓噪音的 bin**（`g_res≈1` → `noise_gain≈0`）。若改用
  `g_total` 會把 NR 剛清掉的噪聲又灌回去 → 傷 BAK。**串 NR 不改變 CNG**（它只看 `g_res`），
  故 NR 串接後 CNG 標準不需調整。
- 量測（`pipelines/compare_res_vs_nr.py --cng-ab`，A/B 開關 CNG）：FS 案 CNG 能量約在訊號
  **−37 dB**（回音 bin 有填）、NE 案 **−72 dB**（近端無回音 → 幾乎不填）。

```bash
# A/B 串 NR 的開不開差異 + CNG 開不開差異（會多輸出 _aec_nr_res_nocng.wav）
python3 pipelines/compare_res_vs_nr.py input.wav --cng-ab --dnsmos
```

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
| `n_partitions` | 6 | ceil(832/160) = 6（832/160 = 5.2，無條件進位）|
| `mu` (step size) | 0.3 | PBFDKF 自適應步長（Kalman 初值是另一參數 `kalman_q_high`=1e-3）|
| `max_delay_ms` | 1024 ms | 最大延遲搜尋範圍 |
| `enable_highpass` | 1 | mic 路徑 80 Hz HPF（內建，不需外接） |
| `enable_saturation_detect` | 1 | ref 路徑 soft-clip（`saturation_softclip_ref`）|
| `enable_delay_est` | 1 | matched-filter (+ ring buffer) 自動對齊 |
| `enable_shadow` | 1 | PBFDAF shadow filter（DT 偵測） |
| `enable_res` | 1 | AEC3 post-filter（preset 預設；**+NR pipeline 覆寫為 0** + `return_res_context`=1，改外部 min 合併）|
| `enable_cng` | 1 | Comfort Noise Generator |
| `shadow_mu_min` | 0.5 | shadow filter 最小步長 |
| `warmup_frames` | 100 | 10 ms × 100 = 1 s 暖機 |

---

## NR Preset（Mode）設定

底層 NR library（C `MmseLsaNrMode` / `mmse_lsa_config_for_mode`，Python `core/nr_strength.py`）有四個 mode：

| Mode | `g_min_db` | `q` (SPP) | `xi_min_db` | `alpha_d` | `alpha_g` | 特性 |
|------|-----------|-----------|-------------|-----------|-----------|------|
| `MILD` | **−20 dB** | 0.60 | −15 dB | 0.85 | 0.92 | 近端優先：最低壓制；適合近距離說話場景 |
| `MODERATE` | **−25 dB** | 0.55 | −18 dB | 0.85 | 0.92 | 介於 mild 與 balanced 之間 |
| `BALANCED` | **−30 dB** | 0.50 | −20 dB | 0.70 | 0.88 | 預設值：噪聲抑制與語音保留的平衡點 |
| `AGGRESSIVE` | **−40 dB** | 0.35 | −25 dB | 0.50 | 0.85 | 噪聲優先：強力壓制；語音可能有些許洩漏感 |

> **`--nr-preset` 只暴露三個**：`pipelines/aec_nr_pipeline.py` 的 `argparse` 用 `choices=['mild', 'balanced', 'aggressive']`；C 的 `pipelines/aec_nr_pipeline.c` `parse_nr_mode()` 只認得 `"mild"`／`"aggressive"` 字串，其餘（含 `"moderate"`）一律 silently 落回 balanced，不報錯。兩邊 pipeline CLI 都還沒接上 `MODERATE`；要用它必須直接呼叫 library API（C `mmse_lsa_config_for_mode(sr, MMSE_LSA_NR_MODERATE)`，Python 對應函式）。
>
> **Python pipeline vs C**：Python `--nr-preset` 套用 strength 四元組 **{g_min, q, xi_min, alpha_g}**；`alpha_d` / `L` 維持 pipeline 自己的結構性 tuning（`alpha_d≈0.95` / `L=150`，為 AEC 殘餘信號調過，見 `_build_denoiser`），不隨 preset 變。C 的 `config_for_mode` 另含 `alpha_attack/alpha_decay/alpha_d`。`balanced` = 既有 shipped 值（pipeline byte-equal）。

**參數說明：**

| 參數 | 說明 |
|------|------|
| `g_min_db` | 最低增益下限（amplitude domain，/20 convention：g_min = 10^(g_min_db/20)，例 −30 dB→0.0316（balanced）、−40 dB→0.01（aggressive））|
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
| `L` | 32（standalone）→ **pipeline 150** | min-stat 窗：C/NR 預設 320 ms；production pipeline 覆寫為 1.5 s（`_build_denoiser`，與 `alpha_d→0.95` 同，不隨 preset 變）|
| `delta_db` | 10 dB | MCRA 語音存在門限 |
| `num_init_frames` | 20 | 噪聲估計初始化幀數（200 ms） |

---

### Blind Test 成績（AEC Challenge Interspeech 2021, 800 cases · **真 no-prealign**）

**AEC(balanced) alone vs AEC(balanced) + NR + RES**。完整鏈路（freq A_min_pl，production）:

> `AEC 線性殘差 E(f)` → `echo-aware NR → G_nr`（ξ=S²/(N²+R²)） → `g_total = min(G_nr, G_res)`
> ＋ far/near 雙閘 ne_floor → `S = E·g_total` ＋ CNG（comfort noise，只填 echo-suppressed bin）

**RES 在 `G_res` 這一項**：它就是線性 AEC **自己** per-frame 的 **AEC3 SuppressionGain**（由殘留回音
PSD R² 算出，≈1 無回音、回音 bin <1）。在 `enable_res=0` 下這個 gain **不套用**到 AEC 輸出，只透過
`return_res_context` seam 把 `G_res` / R² / comfort_noise 輸出，改在上面用 `min(G_nr, G_res)` **外部**合併
回來。所以「RES」不是 NR 之後另一個獨立 filter，而是 AEC3 內部殘響抑制器的 gain 被 `min` 進這條鏈路。
四欄 = AEC 單獨、+NR 三個 preset（`--nr-preset balanced|mild|aggressive`，皆完整 strength 參數組）。

> **本表 2026-06-23 以 AEC v3.24.0（round-robin TD constraint，每 hop 只約束 1 個 partition → 收斂更深、各 far-active
> bucket echo↑）＋ `far02_near` 統一 NR gain re-tune（R² 折入 ξ=S²/(N²+R²)、far-activity＋near-VAD 雙閘 ne_floor）
> 全部重跑、真 no-prealign 產生。**
> AEC-alone = `AEC(balanced)` 含自身 AEC3 post-filter 的完整輸出（= AEC repo 標準產品，**非**純線性；DT_static deg 2.077、DT echo 4.231）。

本地 ONNX AECMOS/DNSMOS，**完全無 pre-align** —— 純靠 AEC 線上的 matched-filter `EchoPathDelayEstimator`
自對齊（與 AEC3 內部對齊機制同類，也是生產真實情境）。offline 全訊號 GCC-PHAT pre-align 是已退役的
crutch：它不只灌高 echo 分數，跟線上 matched filter 併用還會 double-alignment、在部分 movement case
反而鎖到 phantom peak 害 ERLE 崩掉（見 AEC `CHANGELOG [3.22.1]`）。

**AECMOS**（echo↑ / deg↑；deg 每列粗體為最佳。`far02_near` 統一 gain 後 NR 也**幫到 echo**，故 echo 不再標「中性」）:

| 指標 | AEC-alone | +NR(balanced) | +NR(mild) | +NR(aggressive) |
|------|----------|---------------|-----------|-----------------|
| FS_static echo   | 3.557 | 3.626 | 3.578 | 3.705 |
| FS_movement echo | 3.527 | 3.626 | 3.561 | 3.706 |
| DT_static echo   | 4.231 | 4.220 | 4.206 | 4.291 |
| DT_movement echo | 4.140 | 4.169 | 4.142 | 4.253 |
| DT_static deg    | 2.077 | **2.259** | 2.220 | 2.193 |
| DT_movement deg  | 2.138 | **2.234** | 2.215 | 2.164 |
| NE deg           | 4.015 | 4.013 | **4.052** | 3.906 |

**DNSMOS**（SIG/BAK↑；BAK = 背景品質 = NR 本職，AECMOS 量不到；每列粗體為最佳）:

| 指標 | AEC-alone | +NR(balanced) | +NR(mild) | +NR(aggressive) |
|------|----------|---------------|-----------|-----------------|
| FS_static BAK   | 2.640 | 2.889 | 2.763 | **3.073** |
| FS_movement BAK | 2.346 | 2.583 | 2.458 | **2.816** |
| DT_static BAK   | 2.830 | 3.097 | 2.994 | **3.217** |
| DT_movement BAK | 2.710 | 2.978 | 2.865 | **3.100** |
| NE BAK          | 3.841 | **3.910** | 3.897 | 3.909 |
| NE SIG          | **3.462** | 3.397 | 3.436 | 3.348 |
| DT_static SIG   | 2.375 | **2.461** | 2.448 | 2.423 |
| DT_movement SIG | 2.309 | 2.353 | **2.360** | 2.309 |
| FS_static SIG   | 1.810 | **1.936** | 1.911 | 1.913 |
| FS_movement SIG | 1.602 | 1.687 | 1.671 | **1.690** |

> **NR vs AEC-alone**：背景 **BAK +0.24~0.27（balanced，FS/DT）**（降噪本職，AECMOS 量不到；aggressive 更達 FS +0.43~0.47、DT +0.39）、
> DT 近端 deg 也明顯改善（DT_static **+0.18** / DT_movement **+0.10**）。**`far02_near` 統一 gain 把 R² 折入 ξ=S²/(N²+R²)
> 後，NR 連 echo 也幫到了**（balanced FS echo **+0.07~0.10**、aggressive **+0.15~0.18**；DT echo aggressive 亦 +0.06~0.11），
> 不再是先前的「echo 中性」。唯一代價是純 NE SIG（balanced −0.065）。`min(G_nr, G_res)` 撿回 AEC3 的近端感知回音 gain。
> **NR preset = 噪聲抑制強度的 Pareto 軸**：
> - **aggressive** — 背景最乾淨（BAK 全列最高）、echo 撿最多，近端最受損（NE SIG 3.348 最低、NE deg 3.906）。
> - **mild** — 近端最保留（NE SIG 3.436 / NE deg 4.052 最高），背景降噪與 echo 撿回最少。
> - **balanced** — 折衷（預設），DT/FS SIG 反而最高。
>
> ✓ **真 no-prealign 的 echo 全達 ship bar**（AEC-alone FS echo 3.557 >3.5、DT echo 4.231 >4；+NR 同值或更高，
> 因 v3.24.0 round-robin 收斂更深、`far02_near` 統一 gain 再加碼）。
> AEC 模組 ours / AEC3 / Speex 三方對照（目前仍 pre-align，待重生）見 [lib/aec/README.md](lib/aec/README.md)。
> 渲染器:`pipelines/rebench_aec_only.py`(AEC-alone = AEC+RES)、`pipelines/rebench_joint.py`
> (A_min_pl;`NR_PRESET=mild|aggressive` 切 preset)。**offline pre-align 已從所有 pipeline 腳本移除**
> (含 `eval_pipeline_blind.py`),一律走 AEC 線上 matched-filter 對齊(=生產真實情境),不再有
> `NO_PREALIGN` / `PREALIGN` 開關。
