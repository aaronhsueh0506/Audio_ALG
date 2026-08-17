# 開發指南

## Submodule 管理

### 日常開發流程

1. **只開發 NR**：在 `lib/nr/` 對應的獨立 repository 開 branch、測試、提交。
2. **只開發 AEC**：在 `lib/aec/` 對應的獨立 repository 開 branch、測試、提交。
3. **整合測試**：回到 Audio_ALG，明確更新 gitlink 後測試。

不要把 submodule 內的 dirty working tree 當成 Audio_ALG 的一般檔案一起
commit；父 repository 只能記錄 submodule commit，不會收進其內部 diff。

### 更新 Submodule

```bash
# 更新 NR 到最新
git submodule update --remote lib/nr
git add lib/nr
git commit -m "update: NR submodule"

# 更新 AEC 到最新
git submodule update --remote lib/aec
git add lib/aec
git commit -m "update: AEC submodule"
```

### 鎖定特定版本

```bash
cd lib/nr
git checkout <specific-commit-hash>
cd ../..
git add lib/nr
git commit -m "pin: NR to version xxx"
```

## 目錄說明

| 目錄 | 說明 |
|------|------|
| `lib/nr/` | NR submodule (CVNR) |
| `lib/aec/` | AEC submodule |
| `AINR/` | standalone AI noise reduction models |
| `AIAEC/` | neural AEC candidate models and dataset |
| `shared/` | 共享工具代碼 |
| `pipelines/` | conventional mono pipeline and 4-channel AEC shell |
| `docs/` | current docs, dated audits, and archived design records |
| `scripts/` | 管理腳本 |

## 文件維護

- component behavior 改動時先更新最近的 README；
- model output graph 或 checkpoint contract 改動時，README 與版本 gate 必須
  同一個 commit；
- signal grid 改動時同步更新 root、component README 與測試；
- 已被實作取代的設計稿加上 archived notice，避免與現行 API 混用；
- 文件入口與 current/archive 分流見 [`README.md`](README.md)。
