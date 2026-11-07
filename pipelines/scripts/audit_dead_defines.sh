#!/bin/sh
# audit_dead_defines.sh -- build-flag gate: every -DNAME a build file passes to the compiler is read by a
# C source or header. cc accepts an unread -D without a word, so a knob whose reader is gone keeps building
# the same objects while claiming to select something.
#
# Build files: every Makefile under AINR/, AIAEC/ and pipelines/ (obj, bin, build, output, dist and
# gtcrn_github pruned), or the files named on the command line (a relative name is looked up from the
# working directory, then from ROOT). --py adds the Python files under test/ and
# tests/ directories, whose -D strings are flags they pass to a C compiler.
#
# A flag is read when its name is an identifier in the code of a source, with comments and string and
# character literals blanked first. Sources: the .c/.h files of AINR/, AIAEC/, pipelines/, shared/ and lib/,
# and audiolib's common/ (include, src, utility). A flag set on a line that assigns a variable whose name
# contains TEST (DFN_TEST_CFLAGS) is a test flag and may be read by a test source; every other flag needs a
# reader outside the test and tests directories.
#
# Usage:  scripts/audit_dead_defines.sh [--root DIR] [--common DIR] [--py] [--verbose] [FILE...]
#         scripts/audit_dead_defines.sh --self-test
#   ROOT   the Audio_ALG tree (default: this one); COMMON  audiolib's common/ (default: ROOT/../audiolib/common)
# Exit 0 if every flag is read; 1 with the dead flags listed; 2 on a setup error.

set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT=""
COMMON=""
PY=0
VERBOSE=0
SELF=0

while [ $# -gt 0 ]; do
    case "$1" in
        --root)      [ $# -ge 2 ] || { echo "FAIL: --root needs a directory" >&2; exit 2; }; ROOT="$2"; shift 2 ;;
        --common)    [ $# -ge 2 ] || { echo "FAIL: --common needs a directory" >&2; exit 2; }; COMMON="$2"; shift 2 ;;
        --py)        PY=1; shift ;;
        --verbose)   VERBOSE=1; shift ;;
        --self-test) SELF=1; shift ;;
        --)          shift; break ;;
        -*)          echo "FAIL: unknown option $1" >&2; exit 2 ;;
        *)           break ;;
    esac
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT INT TERM

# ---- self-test: the gate must report a dead flag and must pass once the flag is gone ----
if [ "$SELF" -eq 1 ]; then
    T="$TMP/tree"
    mkdir -p "$T/AINR/tests" "$T/AIAEC" "$T/pipelines" "$T/common/include" "$T/common/src" "$T/common/utility"
    cat > "$T/AINR/Makefile" <<'EOF'
CFLAGS += -DLIVE_FLAG -DDEAD_FLAG -DCOMMENT_FLAG -DSTRING_FLAG -DTESTONLY_FLAG
# a mention in a comment line: -DIN_COMMENT_LINE
DFN_TEST_CFLAGS += -DTEST_SCOPED
EOF
    cat > "$T/AINR/lib.c" <<'EOF'
/* COMMENT_FLAG */ static const char *s = "STRING_FLAG"; // DEAD_FLAG
#ifdef LIVE_FLAG
int live;
#endif
EOF
    cat > "$T/AINR/tests/t.c" <<'EOF'
#ifdef TESTONLY_FLAG
int a;
#endif
#ifdef TEST_SCOPED
int b;
#endif
EOF
    cat > "$T/AINR/tests/p.py" <<'EOF'
cmd = ['cc', '-DPY_LIVE', '-DPY_DEAD=1']
EOF
    printf '#ifdef PY_LIVE\n#endif\n' > "$T/AINR/py_reader.c"
    run() { sh "$HERE/audit_dead_defines.sh" --root "$T" --common "$T/common" "$@" 2>&1 || true; }
    first="$(run)"
    for must in DEAD_FLAG COMMENT_FLAG STRING_FLAG TESTONLY_FLAG; do
        printf '%s\n' "$first" | grep -q "$must" || { echo "FAIL: self-test: $must missing from the report"; printf '%s\n' "$first"; exit 1; }
    done
    for must_not in LIVE_FLAG TEST_SCOPED IN_COMMENT_LINE PY_LIVE PY_DEAD; do
        ! printf '%s\n' "$first" | grep -q "$must_not" || { echo "FAIL: self-test: $must_not listed as dead"; printf '%s\n' "$first"; exit 1; }
    done
    printf '%s\n' "$first" | grep -q '^FAIL' || { echo "FAIL: self-test: no FAIL line"; exit 1; }
    sh "$HERE/audit_dead_defines.sh" --root "$T" --common "$T/common" >/dev/null 2>&1 \
        && { echo "FAIL: self-test: exit status 0 with dead flags"; exit 1; }
    py="$(run --py)"
    printf '%s\n' "$py" | grep -q 'PY_DEAD' || { echo "FAIL: self-test: --py did not report PY_DEAD"; printf '%s\n' "$py"; exit 1; }
    ! printf '%s\n' "$py" | grep -q 'PY_LIVE' || { echo "FAIL: self-test: --py reported PY_LIVE"; exit 1; }
    sed -i.bak 's/ -DDEAD_FLAG -DCOMMENT_FLAG -DSTRING_FLAG -DTESTONLY_FLAG//' "$T/AINR/Makefile"
    sh "$HERE/audit_dead_defines.sh" --root "$T" --common "$T/common" >/dev/null 2>&1 \
        || { echo "FAIL: self-test: a tree without dead flags is refused"; exit 1; }
    echo "PASS: self-test: dead, comment-only, string-only and test-only readers reported; live and test-scoped flags pass"
    exit 0
fi

ROOT="${ROOT:-$(cd "$HERE/../.." && pwd)}"
COMMON="${COMMON:-$ROOT/../audiolib/common}"
for d in "$ROOT/AINR" "$ROOT/AIAEC" "$ROOT/pipelines" "$COMMON/include"; do
    [ -d "$d" ] || { echo "FAIL: no $d (ROOT=$ROOT, COMMON=$COMMON)" >&2; exit 2; }
done

dirs_of() { for d in "$@"; do [ -d "$d" ] && printf '%s\n' "$d"; done; }
PRUNE='-name obj -o -name bin -o -name build -o -name output -o -name dist -o -name gtcrn_github'

# ---- build files ----
if [ $# -gt 0 ]; then
    for f in "$@"; do
        if [ -f "$f" ]; then case "$f" in /*) printf '%s\n' "$f" ;; *) printf '%s\n' "$PWD/$f" ;; esac
        else printf '%s\n' "$ROOT/$f"; fi
    done > "$TMP/build_files"
else
    # shellcheck disable=SC2086
    find $(dirs_of "$ROOT/AINR" "$ROOT/AIAEC" "$ROOT/pipelines") \( $PRUNE \) -prune -o -type f -name Makefile -print \
        | sort > "$TMP/build_files"
fi
if [ "$PY" -eq 1 ]; then
    # shellcheck disable=SC2086
    find $(dirs_of "$ROOT/AINR" "$ROOT/AIAEC" "$ROOT/pipelines") \( $PRUNE \) -prune -o -type f -name '*.py' \
        \( -path '*/tests/*' -o -path '*/test/*' \) -print | sort >> "$TMP/build_files"
fi
[ -s "$TMP/build_files" ] || { echo "FAIL: no build files found under $ROOT" >&2; exit 2; }
while IFS= read -r f; do [ -f "$f" ] || { echo "FAIL: no such build file: $f" >&2; exit 2; }; done < "$TMP/build_files"

# ---- the -D flags: file TAB line TAB name TAB scope ----
: > "$TMP/flags"
while IFS= read -r f; do
    case "$f" in *.py) kind=py ;; *) kind=mk ;; esac
    awk -v py="$kind" '
    {
        line = $0
        gsub(/\t/, " ", line)
        if (line ~ /^ *#/) next
        if (py != "py" && match(line, / #/)) line = substr(line, 1, RSTART - 1)
        scope = (py != "py" && line ~ /^ *(override +)?[A-Za-z0-9_]*TEST[A-Za-z0-9_]* *[:+?]*=/) ? "test" : "build"
        if (py == "py") scope = "test"
        rest = line
        while (match(rest, /-D[A-Za-z_][A-Za-z0-9_]*/)) {
            tok = substr(rest, RSTART, RLENGTH)
            before = (RSTART > 1) ? substr(rest, RSTART - 1, 1) : " "
            rest = substr(rest, RSTART + RLENGTH)
            if (before ~ /[A-Za-z0-9_-]/) continue
            print FILENAME "\t" FNR "\t" substr(tok, 3) "\t" scope
        }
    }' "$f" >> "$TMP/flags"
done < "$TMP/build_files"
nfiles=$(wc -l < "$TMP/build_files" | tr -d ' ')
if [ ! -s "$TMP/flags" ]; then
    echo "PASS: no -D flag in $nfiles build files"
    exit 0
fi
cut -f3 "$TMP/flags" | sort -u > "$TMP/names"

# ---- readers: name TAB path, once per (name, file) ----
# shellcheck disable=SC2086
find $(dirs_of "$ROOT/AINR" "$ROOT/AIAEC" "$ROOT/pipelines" "$ROOT/shared" "$ROOT/lib" \
               "$COMMON/include" "$COMMON/src" "$COMMON/utility") \
     \( $PRUNE \) -prune -o -type f \( -name '*.c' -o -name '*.h' \) -print | sort > "$TMP/sources"
[ -s "$TMP/sources" ] || { echo "FAIL: no C sources found under $ROOT" >&2; exit 2; }
xargs awk -v sq="'" -v names="$TMP/names" '
BEGIN { while ((getline n < names) > 0) want[n] = 1; close(names) }
FNR == 1 { inblock = 0 }
{
    line = $0
    if (!inblock && index(line, "/") == 0 && index(line, "\"") == 0 && index(line, sq) == 0) { out = line }
    else {
        out = ""; n = length(line); i = 1
        while (i <= n) {
            c = substr(line, i, 1); c2 = substr(line, i, 2)
            if (inblock) { if (c2 == "*/") { inblock = 0; i += 2 } else i++; continue }
            if (c2 == "/*") { inblock = 1; i += 2; out = out " "; continue }
            if (c2 == "//") break
            if (c == "\"" || c == sq) {
                q = c; i++
                while (i <= n) { d = substr(line, i, 1); if (d == "\\") { i += 2; continue } i++; if (d == q) break }
                out = out " "; continue
            }
            out = out c; i++
        }
    }
    rest = out
    while (match(rest, /[A-Za-z_][A-Za-z0-9_]*/)) {
        id = substr(rest, RSTART, RLENGTH); rest = substr(rest, RSTART + RLENGTH)
        if ((id in want) && !((id SUBSEP FILENAME) in done)) { done[id SUBSEP FILENAME] = 1; print id "\t" FILENAME }
    }
}' < "$TMP/sources" > "$TMP/readers"

# ---- verdict ----
rc=0
awk -F '\t' -v root="$ROOT/" -v verbose="$VERBOSE" '
function rel(p) { return index(p, root) == 1 ? substr(p, length(root) + 1) : p }
FILENAME == ARGV[1] {
    r = rel($2); any[$1] = r
    if (r !~ /(^|\/)tests?\//) nontest[$1] = r
    next
}
{
    total++
    name = $3; file = rel($1)
    if ($4 == "test") ok = (name in any); else ok = (name in nontest)
    if (ok) {
        if (verbose) printf "  ok    %s:%s  -D%s  (read by %s)\n", file, $2, name, ($4 == "test" ? any[name] : nontest[name])
        next
    }
    dead++
    why = (name in any) ? "read only by tests (" any[name] ")" : "read by no source"
    msg[dead] = sprintf("  %s:%s  -D%s  %s", file, $2, name, why)
}
END {
    if (dead) {
        printf "FAIL: %d of %d -D flags are read by no source:\n", dead, total
        for (i = 1; i <= dead; i++) print msg[i]
        exit 1
    }
    printf "PASS: %d -D flags, each read by a source\n", total
}' "$TMP/readers" "$TMP/flags" || rc=$?
exit "$rc"
