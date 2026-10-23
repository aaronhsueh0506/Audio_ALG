#!/bin/bash
# pyc_parity_pipelines.sh -- Python <-> C parity of the pipelines glue (mono, 4-channel).
#
# What it measures (nothing here modifies an algorithm, a C numeric behaviour or a tolerance):
#   (a) 4-channel NR gain: the C chain's fused frame fed to Python run_nr_spectrum, against the C G_nr,
#       with usable_linear taken from the C frame flag (four_aec_fuse AND-folds it over contributing
#       lanes) and, as an independent cross-check, from the Python verdict of the same hop, plus
#       "legacy_amin 0 and 1 must differ" inside C.
#   (b) 4-channel post-beam RES gain: PostBeamResidualSuppressor (+ set_preset) fed with the C fused
#       frame, against the C G_res, for balanced / mild / aggressive / no_floor.
#   (c) 4-channel shared-delay realign: the hop on which the lanes take a mixed warm/soft path at 48 kHz:
#       stock Python 4ch against the C core on the same input.
#   (d) mono pipeline glue (NR gain, min, apply, comfort noise, OLA): Python functions of
#       aec_nr_pipeline.py fed with the C AEC seam, against the C output.
#
# Layers:
#   Layer 1  std-math build (SIMD=0, -DUSE_STANDARD_MATH, NR's six USE_* optimisations off, i.e. NR's
#            DEBUG=1; the 4ch Makefile additionally gets OPT_CFLAGS=-O2 so DEBUG=1 does not turn on -O0) of
#            the C code, compared with Python. Catches sync bugs. The fused frame / AEC seam that Python is
#            fed with is always the one the C code produced, so the comparison isolates the stage under test
#            from the (separately audited) divergence of the AEC library itself.
#   Layer 2  shipped build (Makefile defaults: SIMD=1, fast-math, the six USE_*) compared with the std-math
#            C: stage replays over identical frames (open loop, max |d|) and the closed-loop chains (the
#            adaptive filters diverge, so those rows are judged on the hops before the first divergence, by
#            p99.9 and rms dB; the whole-run numbers are reported next to them).
#
# Scenario ids (see --list): a.nr4.<grid>.*  4ch NR gain      b.res4.<grid>.<preset>  4ch post-beam RES
#   c.realign4.<grid>  shared-delay realign   d.mono.*  mono glue     l2.*  Layer 2      <grid> = 16k256 | 16k512 | 48k1024
#   Rows a.nr4.*.glue_cflag and a.nr4.*.glue_fold both compare the Python NR stage with C over the same
#   frame; the first takes usable_linear from the C frame flag (four_aec_fuse's AND-over-contributing-lanes
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
#                    ${TMPDIR:-/tmp}/pyc_parity_pipelines.<uid>); never inside the source tree
#   --json FILE      machine-readable results (default <work>/results.json)
#   --layers L       1, 2 or 1,2 (default 1,2)
#   --only P[,P...]  run only scenarios whose id starts with one of the prefixes (see --list)
#   --list           print the scenario ids and exit
#   --reuse-build    keep the build trees of an earlier run (skips the build-flag assertion)
#   --mutation-proof ["name[:K,K,..] ..."] [--make-test]
#                    plant small deviations, one at a time, in a scratch copy of the sources and show that the
#                    script FAILS on each (details at the top of pyc_parity/mutation_proof.sh); --make-test also
#                    runs the repository's own `make SIMD=0 test` on the mutated copy
# environment: PYTHON (python3 with numpy, scipy, soundfile, pyyaml), PYC_TREE (a directory holding
#   Audio_ALG/ and audiolib/common/ to build and test instead of this checkout; used by --mutation-proof),
#   PYC_JOBS (make -j, default 4), CC / CXX (default cc / c++).
# Python runs serially (OMP_NUM_THREADS=1); the C builds use PYC_JOBS.

set -u
set -o pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PARITY="$HERE/pyc_parity"
SELF="$HERE/$(basename "$0")"
SELF_ALG="$(cd "$HERE/../.." && pwd)"
TREE="${PYC_TREE:-$(cd "$SELF_ALG/.." && pwd)}"
ALG="$TREE/Audio_ALG"
AC="$TREE/audiolib/common"
PL="$ALG/pipelines"
P4="$PL/4ch_aec_bf_nr_res"
PYTHON="${PYTHON:-python3}"
JOBS="${PYC_JOBS:-4}"
CC="${CC:-cc}"
CXX="${CXX:-c++}"

WORK="${PYC_WORK:-${TMPDIR:-/tmp}/pyc_parity_pipelines.$(id -u)}"
JSON=""
LAYERS="1,2"
ONLY=""
LIST=0
REUSE=0
MUTATION=""
MUTATION_ARGS_SET=0
MAKE_TEST=0

die() { echo "pyc_parity: $*" >&2; exit 2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --work) WORK="$2"; shift 2 ;;
    --json) JSON="$2"; shift 2 ;;
    --layers) LAYERS="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --list) LIST=1; shift ;;
    --reuse-build) REUSE=1; shift ;;
    --mutation-proof) MUTATION_ARGS_SET=1; shift
                      if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then MUTATION="$1"; shift; fi ;;
    --make-test) MAKE_TEST=1; shift ;;
    -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "unknown option $1" ;;
  esac
done

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
PYRUN=("$PYTHON" -I -B)

if [ "$LIST" = 1 ]; then
  exec "${PYRUN[@]}" "$PARITY/pyc_parity.py" --list
fi

[ -d "$ALG" ] && [ -d "$AC" ] || die "tree $TREE must hold Audio_ALG/ and audiolib/common/"
case "$(cd "$(dirname "$WORK")" 2>/dev/null && pwd)/$(basename "$WORK")" in
  "$TREE"/*) die "work directory $WORK is inside the source tree $TREE" ;;
esac
[ -n "$JSON" ] || JSON="$WORK/results.json"

# --------------------------------------------------------------------------------------------
# Mutation proof: see the end of the file.
# --------------------------------------------------------------------------------------------
if [ "$MUTATION_ARGS_SET" = 1 ]; then
  # shellcheck disable=SC1090
  . "$PARITY/mutation_proof.sh"
  mutation_proof_main
  exit $?
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
#   Layer 1: SIMD=0 DEBUG=1 EXTRA_CFLAGS=-DUSE_STANDARD_MATH (+ OPT_CFLAGS=-O2 for the 4ch Makefile,
#            where DEBUG=1 alone would also select -O0 -g; the override isolates the math axis).
#            DEBUG=1 is the only knob that drops NR's six USE_* optimisations.
#   Layer 2: Makefile defaults.
# --------------------------------------------------------------------------------------------
BUILD_INFO="$WORK/build_info.txt"
: > "$BUILD_INFO.new"

make_quiet() { # logfile dir args...
  local log="$1" dir="$2"; shift 2
  ( cd "$dir" && make -j"$JOBS" "$@" ) >> "$log" 2>&1
}

build_layer() { # name simd debug extra opt
  local name="$1" simd="$2" debug="$3" extra="$4" opt="$5"
  local root="$WORK/build/$name" log="$WORK/logs/build_$name.log"
  local knobs=("OBJ_ROOT=$root/obj" "BIN_ROOT=$root/bin" "SIMD=$simd" "DEBUG=$debug" "CC=$CC" "CXX=$CXX")
  local knobs4=("${knobs[@]}")
  [ -n "$extra" ] && { knobs+=("EXTRA_CFLAGS=$extra"); knobs4+=("EXTRA_CFLAGS=$extra"); }
  [ -n "$opt" ] && knobs4+=("OPT_CFLAGS=$opt")
  if [ "$REUSE" = 0 ]; then rm -rf "$root"; : > "$log"; fi
  mkdir -p "$root"
  make_quiet "$log" "$PL" "${knobs[@]}" aec_nr_pipeline || { tail -20 "$log" >&2; die "build $name (pipelines) failed"; }
  make_quiet "$log" "$P4" "${knobs4[@]}" libaudio_pipeline_4ch.a || { tail -20 "$log" >&2; die "build $name (4ch) failed"; }
  local ac aec nr mono four
  ac="$(make -s -C "$AC" "${knobs[@]}" print-lib-path | tail -1)"
  aec="$(make -s -C "$ALG/lib/aec/c_impl" "${knobs[@]}" COMMON_DIR="$AC" COMMON_LIB="$ac" print-lib-path | tail -1)"
  nr="$(make -s -C "$ALG/lib/nr/c_impl" "${knobs[@]}" COMMON_DIR="$AC" COMMON_LIB="$ac" print-lib-path | tail -1)"
  mono="$(make -s -C "$PL" "${knobs[@]}" print-obj-dir | tail -1)/se_config.o"
  four="$(make -s -C "$P4" "${knobs4[@]}" print-lib-path | tail -1)"
  for f in "$ac" "$aec" "$nr" "$mono" "$four"; do [ -f "$f" ] || die "archive or object $f missing after build $name"; done

  # Compile line of the C drivers: the same math/SIMD macros as the libraries, and the FP policy of the
  # Makefiles (-ffp-contract=off -fno-math-errno).
  local macros="" inc
  if [ "$simd" = 0 ]; then macros="-DSIMD_KERNELS_FORCE_SCALAR"; fi
  case "$extra" in *USE_STANDARD_MATH*) macros="$macros -DUSE_STANDARD_MATH" ;; esac
  inc="-I$PL -I$PL/samples -I$P4 -I$ALG/lib/aec/c_impl/include -I$ALG/lib/aec/c_impl/example -I$ALG/lib/nr/c_impl/include -I$AC/include -I$AC"
  mkdir -p "$root/drv"
  # shellcheck disable=SC2086
  "$CC" -Wall -Wextra -O2 -std=gnu99 -ffp-contract=off -fno-math-errno $macros $inc -DPYC_WITH_4CH \
      -c "$PARITY/pyc_drv.c" -o "$root/drv/pyc_drv_4ch.o" >> "$log" 2>&1 || { tail -20 "$log" >&2; die "driver 4ch ($name) failed"; }
  # shellcheck disable=SC2086
  "$CC" -Wall -Wextra -O2 -std=gnu99 -ffp-contract=off -fno-math-errno $macros $inc -DPYC_WITH_MONO \
      -c "$PARITY/pyc_drv.c" -o "$root/drv/pyc_drv_mono.o" >> "$log" 2>&1 || { tail -20 "$log" >&2; die "driver mono ($name) failed"; }
  "$CXX" -o "$root/drv/pyc_drv_4ch" "$root/drv/pyc_drv_4ch.o" "$four" "$aec" "$nr" "$ac" -lm >> "$log" 2>&1 || die "link 4ch ($name) failed"
  "$CXX" -o "$root/drv/pyc_drv_mono" "$root/drv/pyc_drv_mono.o" "$mono" "$aec" "$nr" "$ac" -lm >> "$log" 2>&1 || die "link mono ($name) failed"

  # Build-flag assertion on the compile lines of this run's log (not possible on a reused tree).
  local six="USE_FAST_GAIN_SMOOTHING USE_SHARED_XI_RATIO USE_OPTIMIZED_MIN_BUFFER USE_OPTIMIZED_E1 USE_SINGLE_CLAMP USE_INCREMENTAL_MIN"
  local verdict="not_asserted_reused_tree" d
  if [ "$REUSE" = 0 ]; then
    verdict="ok"
    if [ "$name" = L1 ]; then
      for d in $six; do grep -q -- "-D$d" "$log" && verdict="FAIL_six_flag_$d"; done
      grep -q -- "-DUSE_STANDARD_MATH" "$log" || verdict="FAIL_no_standard_math"
      grep -q -- "-DSIMD_KERNELS_FORCE_SCALAR" "$log" || verdict="FAIL_simd_not_forced_scalar"
      grep -q -- " -O0 " "$log" && verdict="FAIL_O0"
    else
      for d in $six; do grep -q -- "-D$d" "$log" || verdict="FAIL_missing_$d"; done
      grep -q -- "-DUSE_STANDARD_MATH" "$log" && verdict="FAIL_standard_math_in_shipped"
      grep -q -- "-DSIMD_KERNELS_FORCE_SCALAR" "$log" && verdict="FAIL_scalar_in_shipped"
    fi
  fi
  echo "$name.flags=$verdict" >> "$BUILD_INFO.new"
  case "$verdict" in FAIL*) die "build flag assertion for $name: $verdict" ;; esac
  echo "$name.drv_4ch=$root/drv/pyc_drv_4ch" >> "$BUILD_INFO.new"
  echo "$name.drv_mono=$root/drv/pyc_drv_mono" >> "$BUILD_INFO.new"
}

want_layer() { case ",$LAYERS," in *",$1,"*) return 0 ;; esac; return 1; }

want_layer 1 || want_layer 2 || die "--layers must be 1, 2 or 1,2"
# Layer 2 compares against Layer 1, so Layer 1 is always built.
build_layer L1 0 1 "-DUSE_STANDARD_MATH" "-O2"
want_layer 2 && build_layer L2 1 0 "" ""
mv -f "$BUILD_INFO.new" "$BUILD_INFO"

# --------------------------------------------------------------------------------------------
# The comparisons.
# --------------------------------------------------------------------------------------------
ARGS=(--tree "$TREE" --work "$WORK" --layers "$LAYERS" --json "$JSON" --build-info "$BUILD_INFO")
[ -n "$ONLY" ] && ARGS+=(--only "$ONLY")
"${PYRUN[@]}" "$PARITY/pyc_parity.py" "${ARGS[@]}"
exit $?
