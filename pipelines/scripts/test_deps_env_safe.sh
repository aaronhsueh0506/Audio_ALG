#!/bin/bash
# The directories lib/deps.mk resolves reach the shell scripts as data: print-deps-sh output is read one key at a time and never
# eval'd, so a directory name with a space or a shell metacharacter neither breaks the script nor runs anything.
# Usage: bash scripts/test_deps_env_safe.sh   (run from pipelines/)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
fail=0
ck() { if [ "$1" = ok ]; then echo "  PASS  $2"; else echo "  FAIL  $2"; fail=$((fail + 1)); fi; }

# 1. No script evaluates the print-deps-sh output.
bad="$(grep -n 'eval .*\(print-deps-sh\|DEPS_SH\)' "$ROOT"/pipelines/scripts/*.sh "$ROOT"/pipelines/tests/*.sh 2>/dev/null | grep -v 'test_deps_env_safe.sh' | grep -v '^[^:]*:[0-9]*:[[:space:]]*#' || true)"
if [ -z "$bad" ]; then ck ok "no script evals print-deps-sh output"; else ck no "a script evals print-deps-sh output: $bad"; fi

# 2. The per-key read the scripts use returns a hostile directory name verbatim and runs nothing.
T="$(mktemp -d)" || exit 2
trap 'rm -rf "$T"' EXIT
hostile="$T/a b;touch $T/MARK"
deps="$(AEC_DIR="$hostile" make -s --no-print-directory -f "$ROOT/lib/deps.mk" print-deps-sh)" || { echo "FAIL: print-deps-sh"; exit 1; }
for key in AEC_DIR NR_DIR AUDIOLIB_DIR COMMON_DIR; do
    printf -v "$key" '%s' "$(printf '%s\n' "$deps" | sed -n "s/^$key=//p")"
done
if [ "$AEC_DIR" = "$hostile" ]; then ck ok "hostile AEC_DIR read back verbatim"; else ck no "hostile AEC_DIR read back as '$AEC_DIR'"; fi
if [ ! -e "$T/MARK" ]; then ck ok "nothing ran while reading it"; else ck no "the directory name was executed"; fi
if [ -n "$NR_DIR" ] && [ -n "$COMMON_DIR" ] && [ -n "$AUDIOLIB_DIR" ]; then ck ok "the other three keys resolved"; else ck no "a key is empty"; fi

if [ "$fail" -eq 0 ]; then echo "test_deps_env_safe: PASS"; exit 0; fi
echo "test_deps_env_safe: FAIL ($fail)"; exit 1
