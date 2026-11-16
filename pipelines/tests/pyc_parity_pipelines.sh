#!/bin/bash
# pyc_parity_pipelines.sh -- Python <-> C parity of the pipelines glue (mono, 4-channel).
#
# What it measures (nothing here modifies an algorithm, a C numeric behaviour or a tolerance):
#   (a) 4-channel NR gain: the C chain's fused frame fed to Python run_nr_spectrum, against the C G_nr,
#       with usable_linear taken from the C frame flag (the array's fusion AND-folds it over the
#       contributing lanes) and, as an independent cross-check, from the Python verdict of the same hop, plus
#       "legacy_amin 0 and 1 must differ" inside C.
#   (b) 4-channel post-beam RES gain: PostBeamResidualSuppressor (+ set_preset) fed with the C fused
#       frame, against the array's G_res (AecResidualView.res_gain), for balanced / mild / aggressive / no_floor.
#   (c) 4-channel shared-delay realign: the hop on which the lanes take a mixed warm/soft path at 48 kHz:
#       stock Python 4ch against the C core on the same input.
#   (d) mono pipeline glue (NR gain, min, apply, comfort noise, OLA): Python functions of
#       mono/aec_nr_res/pipeline.py fed with the C AEC seam, against the C output.
#
# Layers:
#   Layer 1  std-math build (SIMD=0, -DUSE_STANDARD_MATH, NR's USE_OPTIMIZED_E1 off, i.e. NR's
#            DEBUG=1) of the C code, compared with Python. Catches sync bugs. The fused frame / AEC seam that Python is
#            fed with is always the one the C code produced, so the comparison isolates the stage under test
#            from the (separately audited) divergence of the AEC library itself.
#   Layer 2  shipped build (Makefile defaults: SIMD=1, fast-math, USE_OPTIMIZED_E1) compared with the std-math
#            C: stage replays over identical frames (open loop, max |d|) and the closed-loop chains (the
#            adaptive filters diverge, so those rows are judged on the hops before the first divergence, by
#            p99.9 and rms dB; the whole-run numbers are reported next to them).
#
# Scenario ids (see --list): a.nr4.<grid>.*  4ch NR gain      b.res4.<grid>.<preset>  4ch post-beam RES
#   c.realign4.<grid>  shared-delay realign   d.mono.*  mono glue     l2.*  Layer 2      <grid> = 16k256 | 16k512 | 48k1024
#   Rows a.nr4.*.glue_cflag and a.nr4.*.glue_fold both compare the Python NR stage with C over the same
#   frame; the first takes usable_linear from the C frame flag (the array fusion's AND-over-contributing-lanes
#   fold), the second forces the Python verdict into the C stage as an independent check of the fold.
#
# Tolerances are frozen (pyc_parity.py's TOL1/TOL2 carry the measured max error each constant was set
# from); a frozen tolerance still prints next to the number it judged and changes only through
# measure-then-sign-off. A row for a known gap is tagged EXPECTED_RED(<audit id>): it may exceed its
# tolerance without failing the script, and if it turns green the script FAILS (UNEXPECTED_GREEN) so the
# tag is removed with the fix. A scenario that cannot be exercised on its input (for example no mixed
# warm/soft realign hop, no double-talk hop) is NOT_RUN, never a vacuous pass.
#
# Exit status: 0 = every scenario PASS / EXPECTED_RED / SKIP / INFO;  1 = a FAIL or an UNEXPECTED_GREEN;
#              2 = infrastructure error (build, input checksum, usage);  3 = some scenario NOT_RUN.
#
# usage: pyc_parity_pipelines.sh [options]
#   --work DIR       scratch directory for build trees, dumps and logs (default $PYC_WORK, else
#                    ${TMPDIR:-/tmp}/pyc_parity_pipelines.<uid>, created 0700 and refused when it belongs to someone
#                    else or is writable by others); never inside the source tree
#   --json FILE      machine-readable results (default <work>/results.json)
#   --layers L       1, 2 or 1,2 (default 1,2)
#   --only P[,P...]  run only scenarios whose id starts with one of the prefixes (see --list)
#   --list           print the scenario ids and exit
#   --reuse-build    keep the build trees of an earlier run (skips the build-flag assertion)
# environment: PYTHON (python3 with numpy, scipy, soundfile, pyyaml), PYC_TREE (a directory holding
#   Audio_ALG/, audiolib/common/ and the sibling AEC/ and NR/ -- or AEC_DIR / NR_DIR / COMMON_DIR set -- to
#   build and test instead of this checkout),
#   PYC_JOBS (make -j, default 4), CC / CXX (default cc / c++).
# Python runs serially (OMP_NUM_THREADS=1); the C builds use PYC_JOBS.

set -u
set -o pipefail

die() { echo "pyc_parity: $*" >&2; exit 2; }

HERE="$(cd "$(dirname "$0")" && pwd)"
PARITY="$HERE/pyc_parity"
SELF_ALG="$(cd "$HERE/../.." && pwd)"
TREE="${PYC_TREE:-$(cd "$SELF_ALG/.." && pwd)}"
ALG="$TREE/Audio_ALG"

PYTHON="${PYTHON:-python3}"
JOBS="${PYC_JOBS:-4}"
CC="${CC:-cc}"
CXX="${CXX:-c++}"

WORK="${PYC_WORK:-}"
WORK_IS_DEFAULT=0
if [ -z "$WORK" ]; then WORK="${TMPDIR:-/tmp}/pyc_parity_pipelines.$(id -u)"; WORK_IS_DEFAULT=1; fi
JSON=""
LAYERS="1,2"
ONLY=""
LIST=0
REUSE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --work) WORK="$2"; WORK_IS_DEFAULT=0; shift 2 ;;
    --json) JSON="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --list) LIST=1; shift ;;
    --reuse-build) REUSE=1; shift ;;
    -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option $1" ;;
  esac
done

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
PYRUN=("$PYTHON" -I -B)

# --list only reads pyc_parity.py's own scenario table; it needs neither a tree to build nor
# AEC_DIR/NR_DIR/COMMON_DIR resolved, so it runs before (and regardless of whether) either exists
# -- in particular a PYC_TREE checkout from before lib/deps.mk existed can still be listed.
if [ "$LIST" = 1 ]; then
  exec "${PYRUN[@]}" "$PARITY/pyc_parity.py" --list
fi

# AEC_DIR/NR_DIR/COMMON_DIR: resolved the same way pipelines/Makefile resolves them for the build
# below (lib/deps.mk's default -- the sibling AEC/c_impl, NR/c_impl and audiolib/common dev
# checkouts, pinned in lib/aec.pin/lib/nr.pin/lib/audiolib.pin), via deps.mk's own print-deps-sh
# instead of a second, independently hardcoded copy of that resolution here -- an override (e.g.
# COMMON_DIR=) must change what this script builds, links and hashes identically, not just what
# the Makefile sees. $ALG is this run's Audio_ALG checkout (PYC_TREE), so a $PYC_TREE run resolves
# against that tree's own siblings, not this one's.
PYC_DEPS_SH="$(make -s -f "$ALG/lib/deps.mk" print-deps-sh)" || die "could not resolve $ALG/lib/deps.mk"
for key in AEC_DIR NR_DIR AUDIOLIB_DIR COMMON_DIR; do    # one key at a time, never eval'd: a directory name is data
  printf -v "$key" '%s' "$(printf '%s\n' "$PYC_DEPS_SH" | sed -n "s/^$key=//p")"
done
[ -n "${AEC_DIR:-}" ] && [ -n "${NR_DIR:-}" ] && [ -n "${COMMON_DIR:-}" ] \
  || die "lib/deps.mk print-deps-sh did not set AEC_DIR/NR_DIR/COMMON_DIR"
# Absolute now: this script cd's into several build directories below (mono, then 4ch), and a
# relative override (e.g. the README's AEC_DIR=../lib/aec/c_impl, still resolved here against
# $(pwd) at this point) would otherwise resolve against whichever directory happens to be current
# at each later use instead of the one place it was meant to name.
AEC_DIR="$(cd "$AEC_DIR" && pwd)" || die "AEC_DIR=$AEC_DIR does not exist"
NR_DIR="$(cd "$NR_DIR" && pwd)" || die "NR_DIR=$NR_DIR does not exist"
COMMON_DIR="$(cd "$COMMON_DIR" && pwd)" || die "COMMON_DIR=$COMMON_DIR does not exist"
AC="$COMMON_DIR"
PL="$ALG/pipelines"
P4="$PL/4ch"

[ -d "$ALG" ] && [ -d "$AC" ] || die "tree $TREE must hold Audio_ALG/ and audiolib/common/ (and AEC/ and NR/, or AEC_DIR / NR_DIR set)"

case "$(cd "$(dirname "$WORK")" 2>/dev/null && pwd)/$(basename "$WORK")" in
  "$TREE"/*) die "work directory $WORK is inside the source tree $TREE" ;;
esac
[ -n "$JSON" ] || JSON="$WORK/results.json"

# The default directory has a predictable name: it is created private (0700) and used only when it is a plain directory
# of this user that no one else can write to, since the drivers built in it are run afterwards.
if [ "$WORK_IS_DEFAULT" = 1 ]; then
  [ -e "$WORK" ] || mkdir -m 700 "$WORK" 2>/dev/null || true
  if [ -L "$WORK" ] || [ ! -d "$WORK" ] || [ ! -O "$WORK" ] || [ -n "$(find "$WORK" -maxdepth 0 \( -perm -020 -o -perm -002 \))" ]; then
    die "default work directory $WORK is not a directory of this user that is closed to group and others; remove it or pass --work DIR"
  fi
fi
mkdir -p "$WORK" || die "cannot create $WORK"
mkdir -p "$WORK/logs" "$WORK/inputs" "$WORK/run"

# --------------------------------------------------------------------------------------------
# Inputs: generated from the pinned recordings with a fixed seed; a checksum mismatch is an error.
# --------------------------------------------------------------------------------------------
"${PYRUN[@]}" "$PARITY/gen_inputs.py" generate "$WORK/inputs" "$PARITY/inputs.sha256" \
  > "$WORK/logs/inputs.log" 2>&1 || { cat "$WORK/logs/inputs.log" >&2; die "input generation failed (checksum mismatch, or a missing python package: numpy scipy soundfile)"; }

# --------------------------------------------------------------------------------------------
# Builds. Each layer gets its own OBJ_ROOT / BIN_ROOT under $WORK, so nothing is written into the tree.
#   Layer 1: SIMD=0 DEBUG=1 EXTRA_CFLAGS=-DUSE_STANDARD_MATH.
#            DEBUG=1 is the only knob that drops NR's USE_OPTIMIZED_E1.
#   Layer 2: Makefile defaults.
# --------------------------------------------------------------------------------------------
BUILD_INFO="$WORK/build_info.txt"
: > "$BUILD_INFO.new"
# So pyc_parity.py can hash the AEC sources it actually linked, not the lib/aec submodule checkout
# a tree-relative path would otherwise name.
echo "AEC_DIR=$AEC_DIR" >> "$BUILD_INFO.new"

make_quiet() { # logfile dir args...
  local log="$1" dir="$2"; shift 2
  ( cd "$dir" && make -j"$JOBS" "$@" ) >> "$log" 2>&1
}

build_layer() { # name simd debug extra
  local name="$1" simd="$2" debug="$3" extra="$4"
  local root="$WORK/build/$name" log="$WORK/logs/build_$name.log"
  # AEC_DIR/NR_DIR/COMMON_DIR passed explicitly (not left to inherited environment/MAKEFLAGS): a
  # caller's command-line override (e.g. the README's AEC_DIR=../lib/aec/c_impl) reaches this
  # script's own `make -C pipelines ... pyc-parity` as a command-line variable, which GNU Make
  # re-propagates to every nested $(MAKE) invocation via MAKEFLAGS using that ORIGINAL (relative)
  # string, at a precedence that wins over both this script's own absolutized shell variable and
  # whatever COMMON_DIR/AEC_DIR/NR_DIR the Makefile would otherwise resolve -- silently undoing
  # the absolutization above for every build this function invokes unless restated here.
  local knobs=("OBJ_ROOT=$root/obj" "BIN_ROOT=$root/bin" "SIMD=$simd" "DEBUG=$debug" "CC=$CC" "CXX=$CXX" \
               "AEC_DIR=$AEC_DIR" "NR_DIR=$NR_DIR" "COMMON_DIR=$COMMON_DIR")
  [ -n "$extra" ] && knobs+=("EXTRA_CFLAGS=$extra")
  if [ "$REUSE" = 0 ]; then rm -rf "$root"; : > "$log"; fi
  mkdir -p "$root"
  make_quiet "$log" "$PL" "${knobs[@]}" aec_nr_pipeline || { tail -20 "$log" >&2; die "build $name (pipelines) failed"; }
  local ac aec aect nr
  ac="$(make -s -C "$AC" "${knobs[@]}" print-lib-path | tail -1)"
  aec="$(make -s -C "$AEC_DIR" "${knobs[@]}" COMMON_DIR="$AC" COMMON_LIB="$ac" print-lib-path | tail -1)"
  nr="$(make -s -C "$NR_DIR" "${knobs[@]}" COMMON_DIR="$AC" COMMON_LIB="$ac" print-lib-path | tail -1)"
  # The AEC_TESTING archive keys its own build tree; only the 4-channel replay driver links it (it runs the array's
  # gain stage on recorded frames through aec_testing_res_*), so no test hook reaches the other two drivers.
  make_quiet "$log" "$AEC_DIR" "${knobs[@]}" AEC_TESTING=1 COMMON_DIR="$AC" COMMON_LIB="$ac" lib || { tail -20 "$log" >&2; die "build $name (AEC_TESTING archive) failed"; }
  aect="$(make -s -C "$AEC_DIR" "${knobs[@]}" AEC_TESTING=1 COMMON_DIR="$AC" COMMON_LIB="$ac" print-lib-path | tail -1)"
  [ "$aect" != "$aec" ] || die "the AEC_TESTING archive resolved to the production archive $aec"
  for f in "$ac" "$aec" "$aect" "$nr"; do [ -f "$f" ] || die "archive or object $f missing after build $name"; done

  # Compile line of the C drivers: the same math/SIMD macros as the libraries, and the FP policy of the
  # Makefiles (-ffp-contract=off -fno-math-errno).
  local macros="" inc
  if [ "$simd" = 0 ]; then macros="-DSIMD_KERNELS_FORCE_SCALAR"; fi
  case "$extra" in *USE_STANDARD_MATH*) macros="$macros -DUSE_STANDARD_MATH" ;; esac
  inc="-I$PL -I$P4 -I$AEC_DIR/include -I$AEC_DIR/include/internal -I$AEC_DIR/example -I$NR_DIR/include -I$AC/include -I$AC"
  mkdir -p "$root/drv"
  # shellcheck disable=SC2086
  "$CC" -Wall -Wextra -O2 -std=gnu99 -ffp-contract=off -fno-math-errno $macros $inc -DPYC_WITH_4CH \
      -c "$PARITY/pyc_drv.c" -o "$root/drv/pyc_drv_4ch.o" >> "$log" 2>&1 || { tail -20 "$log" >&2; die "driver 4ch ($name) failed"; }
  # shellcheck disable=SC2086
  "$CC" -Wall -Wextra -O2 -std=gnu99 -ffp-contract=off -fno-math-errno $macros $inc -DPYC_WITH_MONO \
      -c "$PARITY/pyc_drv.c" -o "$root/drv/pyc_drv_mono.o" >> "$log" 2>&1 || { tail -20 "$log" >&2; die "driver mono ($name) failed"; }
  # shellcheck disable=SC2086
  "$CC" -Wall -Wextra -O2 -std=gnu99 -ffp-contract=off -fno-math-errno $macros $inc -I"$AEC_DIR/test" -DAEC_TESTING \
      -DPYC_WITH_4CH -DPYC_REPLAY_RES -c "$PARITY/pyc_drv.c" -o "$root/drv/pyc_drv_4ch_replay.o" >> "$log" 2>&1 || { tail -20 "$log" >&2; die "driver 4ch replay ($name) failed"; }
  "$CXX" -o "$root/drv/pyc_drv_4ch" "$root/drv/pyc_drv_4ch.o" "$aec" "$nr" "$ac" -lm >> "$log" 2>&1 || die "link 4ch ($name) failed"
  "$CXX" -o "$root/drv/pyc_drv_4ch_replay" "$root/drv/pyc_drv_4ch_replay.o" "$aect" "$nr" "$ac" -lm >> "$log" 2>&1 || die "link 4ch replay ($name) failed"
  "$CXX" -o "$root/drv/pyc_drv_mono" "$root/drv/pyc_drv_mono.o" "$aec" "$nr" "$ac" -lm >> "$log" 2>&1 || die "link mono ($name) failed"

  # A replay seam whose header is not self-consistent (a hop smaller than fft / 2, or more records than the file holds) is
  # refused with exit 1; the replay drivers must not size a buffer from it.
  local bad_seam drv_name
  for bad_seam in "16000 256 129 4 2" "16000 256 129 128 2000000000"; do
    "${PYRUN[@]}" -c 'import struct, sys
sr, fft, nf, hop, nh = (int(x) for x in sys.argv[2:7])
rec = 10 * nf + 2 * hop + 16   # floats per seam record; the file holds every record the header announces, up to 1 Mi floats
open(sys.argv[1], "wb").write(struct.pack("<10i", 0x53435950, 1, 0, sr, fft, nf, hop, nh, 0, 0) + bytes(4 * min(rec * nh, 1 << 20)))' \
      "$root/drv/bad.seam" $bad_seam
    for drv_name in pyc_drv_mono pyc_drv_4ch_replay; do
      rc=0
      "$root/drv/$drv_name" replay seam="$root/drv/bad.seam" out="$root/drv/bad.out" >> "$log" 2>&1 || rc=$?
      [ "$rc" = 1 ] || die "$drv_name ($name) did not refuse the inconsistent seam header '$bad_seam' (exit $rc)"
    done
  done

  # Build-flag assertion on the compile lines of this run's log (not possible on a reused tree).
  local nr_flags="USE_OPTIMIZED_E1"
  local verdict="not_asserted_reused_tree" d
  if [ "$REUSE" = 0 ]; then
    verdict="ok"
    if [ "$name" = L1 ]; then
      for d in $nr_flags; do grep -q -- "-D$d" "$log" && verdict="FAIL_nr_flag_$d"; done
      grep -q -- "-DUSE_STANDARD_MATH" "$log" || verdict="FAIL_no_standard_math"
      grep -q -- "-DSIMD_KERNELS_FORCE_SCALAR" "$log" || verdict="FAIL_simd_not_forced_scalar"
      grep -q -- " -O0 " "$log" && verdict="FAIL_O0"
    else
      for d in $nr_flags; do grep -q -- "-D$d" "$log" || verdict="FAIL_missing_$d"; done
      grep -q -- "-DUSE_STANDARD_MATH" "$log" && verdict="FAIL_standard_math_in_shipped"
      grep -q -- "-DSIMD_KERNELS_FORCE_SCALAR" "$log" && verdict="FAIL_scalar_in_shipped"
    fi
  fi
  echo "$name.flags=$verdict" >> "$BUILD_INFO.new"
  case "$verdict" in FAIL*) die "build flag assertion for $name: $verdict" ;; esac
  echo "$name.drv_4ch=$root/drv/pyc_drv_4ch" >> "$BUILD_INFO.new"
  echo "$name.drv_4ch_replay=$root/drv/pyc_drv_4ch_replay" >> "$BUILD_INFO.new"
  echo "$name.drv_mono=$root/drv/pyc_drv_mono" >> "$BUILD_INFO.new"
}

want_layer() { case ",$LAYERS," in *",$1,"*) return 0 ;; esac; return 1; }

want_layer 1 || want_layer 2 || die "--layers must be 1, 2 or 1,2"
# Layer 2 compares against Layer 1, so Layer 1 is always built.
build_layer L1 0 1 "-DUSE_STANDARD_MATH"
want_layer 2 && build_layer L2 1 0 ""
mv -f "$BUILD_INFO.new" "$BUILD_INFO"

# --------------------------------------------------------------------------------------------
# The comparisons.
# --------------------------------------------------------------------------------------------
ARGS=(--tree "$TREE" --work "$WORK" --layers "$LAYERS" --json "$JSON" --build-info "$BUILD_INFO")
[ -n "$ONLY" ] && ARGS+=(--only "$ONLY")
"${PYRUN[@]}" "$PARITY/pyc_parity.py" "${ARGS[@]}"
exit $?
