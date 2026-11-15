#!/bin/sh
# Customer header set (docs/maintenance/models.md, "Customer header set"): one C99 translation unit with -Wpedantic
# -Werror and the same unit as C++11, given only the documented include roots, both DFN2 products. Every
# ainr_ / aiaec_ / stream_ / graph_io_ function the unit declares is referenced; the C++ object must reference the
# same external symbols as the C object, none mangled (header-only static functions have no linkage to keep).
# A positive control proves the mangling scan would fire.
# Usage: check_public_headers.sh CC CXX COMMON_DIR BUILD_DIR   (run from stream/common)
set -eu
CC=$1; CXX=$2; COMMON_DIR=$3; OUT=$4
ROOTS="-I../ainr -I../ainr/bridge -I../aiaec -I../aiaec/front -I. -I$COMMON_DIR -I$COMMON_DIR/include"
mkdir -p "$OUT"
HDR="$OUT/public_headers_probe.h"
cat > "$HDR" <<'EOF'
#include "ainr.h"
#include "ainr_bridge.h"
#include "aiaec.h"
#include "aiaec_front.h"
#include "stream_mem.h"
#include "stream_run.h"
#include "stream_bind.h"
#include "graph_io.h"
#include "graph_io_dfn2.h"
#include "graph_io_deepvqes.h"
#include "graph_io_alignulcnet.h"
EOF

mangled() { # $1 = nm -u output: 0 when it holds a C++-mangled symbol
    grep -Eq '(^|[ _])_Z[A-Za-z0-9_]+$' "$1"
}

for variant in full mask_only; do
    def=""; [ "$variant" = mask_only ] && def="-DDFN2_MASK_ONLY"
    base="$OUT/public_headers_$variant"
    # shellcheck disable=SC2086
    "$CC" -std=c99 $def $ROOTS -E -P "$HDR" > "$base.i"
    names=$(grep -oE '(ainr|aiaec|stream|graph_io)_[A-Za-z0-9_]+[[:space:]]*\(' "$base.i" | sed 's/[[:space:]]*($//' | sort -u)
    [ -n "$names" ] || { echo "check-public-headers: no function found in the header set" >&2; exit 1; }
    {
        printf '#include "%s"\n' "$HDR"
        printf 'typedef void (*probe_fn)(void);\nprobe_fn probe_refs[] = {\n'
        for n in $names; do printf '    (probe_fn)&%s,\n' "$n"; done
        printf '    0\n};\n'
    } > "$base.c"
    cp "$base.c" "$base.cpp"
    # shellcheck disable=SC2086
    "$CC" -std=c99 -Wall -Wextra -Wpedantic -Werror $def $ROOTS -c "$base.c" -o "$base.c.o"
    # shellcheck disable=SC2086
    "$CXX" -std=c++11 -Wall -Wextra -Wpedantic -Werror $def $ROOTS -c "$base.cpp" -o "$base.cpp.o"
    nm -u "$base.cpp.o" > "$base.cpp.all"
    if mangled "$base.cpp.all"; then
        echo "check-public-headers: $variant: a mangled external reference (a declaration outside extern \"C\"):" >&2
        grep -E '_Z[A-Za-z0-9_]+$' "$base.cpp.all" >&2; exit 1
    fi
    # Only the header set's own names: the compilers may lower memset and the like differently.
    nm -u "$base.c.o" | grep -E '(^|[ _])(ainr|aiaec|stream|graph_io)_' | sort > "$base.c.u" || true
    grep -E '(^|[ _])(ainr|aiaec|stream|graph_io)_' "$base.cpp.all" | sort > "$base.cpp.u" || true
    if [ ! -s "$base.c.u" ] || ! cmp -s "$base.c.u" "$base.cpp.u"; then
        echo "check-public-headers: $variant: the C and C++ objects reference different symbols" >&2
        diff "$base.c.u" "$base.cpp.u" >&2 || true; exit 1
    fi
    echo "check-public-headers: $variant: $(echo "$names" | wc -l | tr -d ' ') functions, C99 and C++11, $(wc -l < "$base.c.u" | tr -d ' ') external references, C linkage"
done

# Positive control: a declaration outside extern "C" is referenced mangled and the scan sees it.
printf 'int probe_outside_extern_c(void);\nint (*probe_ref)(void) = &probe_outside_extern_c;\n' > "$OUT/mangle_probe.cpp"
"$CXX" -std=c++11 -c "$OUT/mangle_probe.cpp" -o "$OUT/mangle_probe.o"
nm -u "$OUT/mangle_probe.o" > "$OUT/mangle_probe.u"
mangled "$OUT/mangle_probe.u" || { echo "check-public-headers: positive control not detected" >&2; exit 1; }
echo "check-public-headers: PASS"
