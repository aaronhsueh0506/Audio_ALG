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

# 3. A name with an apostrophe, a backtick and parentheses is data too, in print-deps-sh and in check-deps-pins.
quoted="$T/q'x;touch $T/MARK2;'\`touch $T/MARK3\`(y)"
deps="$(AEC_DIR="$quoted" make -s --no-print-directory -f "$ROOT/lib/deps.mk" print-deps-sh 2>/dev/null)"
got="$(printf '%s\n' "$deps" | sed -n 's/^AEC_DIR=//p')"
if [ "$got" = "$quoted" ]; then ck ok "AEC_DIR with ' \` ( ) read back verbatim"; else ck no "AEC_DIR with ' \` ( ) read back as '$got'"; fi
others="$(printf '%s\n' "$deps" | grep -Ec '^(NR_DIR|AUDIOLIB_DIR|COMMON_DIR)=.')"
if [ "$others" = 3 ]; then ck ok "the other three keys still resolve beside it"; else ck no "only $others of the other three keys resolved"; fi
pins="$(AEC_DIR="$quoted" make -s --no-print-directory -f "$ROOT/lib/deps.mk" check-deps-pins 2>&1 || true)"
if [ ! -e "$T/MARK2" ] && [ ! -e "$T/MARK3" ]; then ck ok "nothing ran in print-deps-sh or check-deps-pins"; else ck no "a directory name was executed"; fi
if printf '%s\n' "$pins" | grep -qF "AEC ($quoted)"; then ck ok "check-deps-pins reports the name verbatim"; else ck no "check-deps-pins altered the name"; fi

# 4. bootstrap-deps reads each directory, URL and commit from the environment: a name with a space or an apostrophe is cloned into
# verbatim and runs nothing. The clone source is the sibling AEC checkout at the pinned commit; skipped when it lacks that commit.
aec_pin="$(sed -n 's/^AEC_COMMIT=//p' "$ROOT/lib/aec.pin")"
aec_src="$(cd "$ROOT/.." && pwd)/AEC"
if git -C "$aec_src" cat-file -e "$aec_pin^{commit}" 2>/dev/null; then
    mkdir -p "$T/nr" "$T/al" "$T/b d" "$T/q'z;touch $T/MARK4;'"
    for parent in "$T/b d" "$T/q'z;touch $T/MARK4;'"; do
        make -s --no-print-directory -f "$ROOT/lib/deps.mk" bootstrap-deps AEC_REPO_DIR="$parent/AEC" NR_REPO_DIR="$T/nr" \
            AUDIOLIB_DIR="$T/al" AEC_URL="$aec_src" >/dev/null 2>&1
        if [ "$(git -C "$parent/AEC" rev-parse HEAD 2>/dev/null)" = "$aec_pin" ]; then ck ok "bootstrap-deps cloned into '$parent/AEC'"
        else ck no "bootstrap-deps did not clone into '$parent/AEC'"; fi
    done
    if [ ! -e "$T/MARK4" ]; then ck ok "nothing ran while bootstrapping into the hostile names"; else ck no "a bootstrap directory name was executed"; fi
else
    echo "  SKIP  bootstrap-deps: $aec_src does not hold the pinned commit"
fi

# 5. The toolchain values pasted into shell text (TOOLCHAIN_FWD): a single quote or a line break in CC, CXX, AR, RANLIB or
# EXTRA_CFLAGS, or a WERROR that is not exactly one word, 0 or 1, is a parse-time error; ordinary values expand.
cat > "$T/probe.mk" <<EOF
include $ROOT/lib/deps.mk
ifdef SHOW_FWD
\$(info \$(TOOLCHAIN_FWD))
endif
ifdef SHOW_SIG
\$(info \$(TARGET_ABI_SIGNATURE))
endif
show: ; @:
EOF
okfwd="$(make -s -f "$T/probe.mk" show SHOW_FWD=1 CC=cc EXTRA_CFLAGS='-DA=1 -DB=2' WERROR=1 2>&1)"
if [ "$okfwd" = "CC='cc' CXX='c++' AR='ar' RANLIB='ranlib' EXTRA_CFLAGS='-DA=1 -DB=2' WERROR=1" ]; then ck ok "ordinary toolchain values expand"
else ck no "ordinary toolchain values expanded as '$okfwd'"; fi
n=0
for bad in "CC=cc'; touch $T/MARK5; echo '" "EXTRA_CFLAGS=-O2' ; touch $T/MARK5; echo '" "WERROR=1; touch $T/MARK5" "WERROR=2"; do
    n=$((n + 1))
    if make -s -f "$T/probe.mk" show SHOW_FWD=1 "$bad" >/dev/null 2>&1; then ck no "toolchain value $n was accepted: $bad"; else ck ok "toolchain value $n was refused"; fi
done
for bad in "WERROR=1 1" "WERROR=0 1" "WERROR="; do
    out="$(make -s -f "$T/probe.mk" show SHOW_FWD=1 "$bad" 2>&1)" && rc=0 || rc=$?
    if [ "$rc" -ne 0 ] && printf '%s\n' "$out" | grep -qF "WERROR must be exactly 0 or 1"; then ck ok "'$bad' was refused with the WERROR text"
    else ck no "'$bad' gave rc=$rc: $out"; fi
done
if [ ! -e "$T/MARK5" ]; then ck ok "nothing ran while the toolchain values were refused"; else ck no "a toolchain value was executed"; fi

# 6. The compiler-target probe finds target_abi_signature.sh under a COMMON_DIR with a space and one with an apostrophe.
real_common="$(cd "$ROOT/.." && pwd)/audiolib/common"
if [ -f "$real_common/scripts/target_abi_signature.sh" ]; then
    want="$(make -s -f "$T/probe.mk" show SHOW_SIG=1 CFLAGS=-O2 COMMON_DIR="$real_common")"
    ln -s "$real_common" "$T/c d"
    ln -s "$real_common" "$T/c'e"
    for link in "$T/c d" "$T/c'e"; do
        got="$(make -s -f "$T/probe.mk" show SHOW_SIG=1 CFLAGS=-O2 COMMON_DIR="$link" 2>&1)"
        if [ -n "$want" ] && [ "$got" = "$want" ]; then ck ok "target probe through '$link'"; else ck no "target probe through '$link' gave '$got', want '$want'"; fi
    done
else
    echo "  SKIP  target probe: no audiolib/common/scripts/target_abi_signature.sh beside the repo"
fi

if [ "$fail" -eq 0 ]; then echo "test_deps_env_safe: PASS"; exit 0; fi
echo "test_deps_env_safe: FAIL ($fail)"; exit 1
