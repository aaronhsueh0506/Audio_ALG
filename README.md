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

AEC(linear) → NR(MMSE-LSA) → RES 串接處理：
```bash
cd Audio_ALG
python -m pipelines.aec_nr_pipeline --mic mic.wav --ref ref.wav --output out.wav --preset balanced
```

### Blind Test 成績（AEC Challenge Interspeech 2021, 800 cases, balanced）

| 指標 | AEC-only | AEC+NR+RES | 變化 |
|------|---------|------------|------|
| **AECMOS** | | | |
| FS echo↑ | 3.615 | 3.568 | -0.047 |
| DT echo↑ | 4.232 | 4.339 | **+0.107** |
| DT deg↑ | 2.095 | 2.314 | **+0.219** |
| NE deg↑ | 3.975 | 3.488 | -0.487 |
| **DNSMOS** | | | |
| FS SIG↑ | 2.007 | 2.118 | +0.111 |
| FS BAK↑ | 3.217 | 3.730 | **+0.513** |
| DT SIG↑ | 2.461 | 2.510 | +0.049 |
| DT BAK↑ | 3.380 | 3.738 | **+0.358** |
| NE SIG↑ | 3.430 | 3.228 | -0.202 |
| NE BAK↑ | 3.892 | 3.926 | +0.034 |

> NR 主要貢獻：BAK +0.5（背景噪音品質）、DT deg +0.2（語音保護）。
