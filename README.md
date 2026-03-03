# Audio_ALG - Audio Processing Algorithms Integration

整合音頻處理算法，包含降噪 (NR) 和聲學回聲消除 (AEC) 模組。

## 項目結構

此項目使用 **Git Submodules** 管理：

- **[nr/](https://github.com/aaronhsueh0506/CVNR)**: 降噪算法（獨立倉庫，Git Submodule）
- **[aec/](https://github.com/aaronhsueh0506/AEC)**: 聲學回聲消除（獨立倉庫，Git Submodule）
- **shared/**: 共享工具和代碼
- **pipelines/**: AEC + NR 串接處理鏈
- **docs/**: 統一文檔
- **scripts/**: 管理腳本

## 快速開始

### 克隆項目（包含 submodules）

```bash
# 方法 1: 克隆時同時初始化 submodules
git clone --recursive git@github.com:aaronhsueh0506/Audio_ALG.git

# 方法 2: 先克隆，再初始化 submodules
git clone git@github.com:aaronhsueh0506/Audio_ALG.git
cd Audio_ALG
git submodule update --init --recursive
```

### 更新 Submodules 到最新版本

```bash
# 更新所有 submodules
./scripts/update_submodules.sh

# 或手動更新
git submodule update --remote nr
git submodule update --remote aec
```

## 開發工作流

### 獨立開發 NR
```bash
cd nr/
git checkout -b feature/xxx
# ... 開發 ...
git push origin feature/xxx
```

### 獨立開發 AEC
```bash
cd aec/
git checkout -b feature/xxx
# ... 開發 ...
git push origin feature/xxx
```

### 更新整合倉庫的 Submodule 引用
```bash
# 在 Audio_ALG 根目錄
git submodule update --remote nr   # 拉取 NR 最新
git add nr
git commit -m "update: NR submodule to latest"
git push
```

## 處理鏈 (Pipeline)

AEC 輸出接 NR 輸入的串接處理：
```bash
python pipelines/aec_nr_pipeline.py --input input.wav --ref ref.wav --output output.wav
```
