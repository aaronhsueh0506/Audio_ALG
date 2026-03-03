# 開發指南

## Submodule 管理

### 日常開發流程

1. **只開發 NR**: 直接在 `/SE/NR/` 目錄開發，與 Audio_ALG 無關
2. **只開發 AEC**: 直接在 `/SE/AEC/` 目錄開發，與 Audio_ALG 無關
3. **整合測試**: 在 Audio_ALG 中更新 submodule 後測試

### 更新 Submodule

```bash
# 更新 NR 到最新
git submodule update --remote nr
git add nr
git commit -m "update: NR submodule"

# 更新 AEC 到最新
git submodule update --remote aec
git add aec
git commit -m "update: AEC submodule"
```

### 鎖定特定版本

```bash
cd nr
git checkout <specific-commit-hash>
cd ..
git add nr
git commit -m "pin: NR to version xxx"
```

## 目錄說明

| 目錄 | 說明 |
|------|------|
| `nr/` | NR submodule (CVNR) |
| `aec/` | AEC submodule |
| `shared/` | 共享工具代碼 |
| `pipelines/` | 串接處理鏈 |
| `docs/` | 文檔 |
| `scripts/` | 管理腳本 |
