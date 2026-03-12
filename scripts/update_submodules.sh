#!/bin/bash
# 更新所有 submodules 到最新版本

set -e

echo "=== Updating Audio_ALG Submodules ==="
echo ""

echo "Updating NR submodule..."
git submodule update --remote lib/nr
echo "NR updated to: $(cd lib/nr && git log --oneline -1)"
echo ""

echo "Updating AEC submodule..."
git submodule update --remote lib/aec
echo "AEC updated to: $(cd lib/aec && git log --oneline -1)"
echo ""

echo "=== All submodules updated ==="
echo ""
echo "Run 'git add lib/nr lib/aec && git commit' to save the update."
