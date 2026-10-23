# mutation_proof.sh -- "can it fail?" proof of pyc_parity_pipelines.sh. Sourced by `pyc_parity_pipelines.sh
# --mutation-proof [name[:K,K..] ...] [--make-test]`; defines functions only.
#
# A scratch copy of Audio_ALG + audiolib (under <work>/mutation_proof/tree; the checkout is never touched)
# gets one small, real deviation at a time in a C constant of the code the script judges; the script is run on the
# copy over the Layer-1 rows that should notice it; the proof records which rows turn FAIL. Sweeps (name:K,K,..)
# scale the deviation by K units of the last place of the constant, so the smallest DETECTED deviation -- the
# detection threshold of the script -- is measured, not assumed. One unit in the last place of a float32 constant is
# ~1.2e-7 relative: far below the ~1e-5 Python/C agreement the Layer-1 tolerances are built on, so a one-ulp
# deviation is expected to go undetected by a Python-parity check and is reported as such, not hidden.
#
# --make-test additionally runs the repository's own `make SIMD=0 test` (pipelines and 4ch) on the mutated copy at the
# largest K of each mutation, to show whether the existing gates notice it.
#
# Mutations (name: file / change / rows that must notice it at the largest K):
#   fold_coef_ulp          pipelines/common/se_anr_stage.c   r2 fold coefficient SE_PSD_SCALE_INV * (1 + K * 2^-23)
#   xi_min_db_div10_5      lib/nr/c_impl/src/spp_estimator.c xi_min_db / 10.0f -> / 10.5f
#   aec_floor_balanced_ulp lib/aec/c_impl/src/aec.c          balanced far-active floor -28.0f - K * 2^-19 dB
#   cng_seed_1bit          pipelines/common/se_stage.h       SE_CNG_RNG_SEED 0x9e3779b9u -> 0x9e3779b8u
#   usable_gate_dropped    pipelines/common/se_anr_stage.c   "&& frame->usable_linear" removed from the r2 fold condition
#
# Flip tests (name flip_*): a known past defect is planted back into the copy; the row(s) guarding it must turn FAIL:
#   flip_s25_c_flag_forced  4aec_nr_res.c        four_aec_fuse forces frame->usable_linear = 1   (a.nr4.*.verdict_flag)
#   flip_s26_python_dt_floor 4ch pipeline.py     PostBeamResidualSuppressor.set_preset stops retargeting the DT floor (b.res4.*.aggressive)
#   flip_s27_python_borrow  4ch pipeline.py      _fuse_contexts raises on lane far-spectrum disagreement (c.realign4.48k1024)

MP_DEFAULT="fold_coef_ulp:1,16,256,4096,65536 xi_min_db_div10_5 aec_floor_balanced_ulp:1,64,4096,65536 cng_seed_1bit usable_gate_dropped flip_s25_c_flag_forced flip_s26_python_dt_floor flip_s27_python_borrow"

# mp_def NAME K  -> M_FILE M_PERL M_ONLY M_MUST (rows that must be FAIL when detected)
mp_def() {
  local k="${2:-1}"
  M_ONLY=""; M_MUST=""
  case "$1" in
    fold_coef_ulp)
      M_FILE="Audio_ALG/pipelines/common/se_anr_stage.c"
      M_PERL='s|(#include "utility/simd_kernels.h"\n)|$1#undef SE_PSD_SCALE_INV\n#define SE_PSD_SCALE_INV (0x1p-30f * (1.0f + '"$k"'.0f * 0x1p-23f))\n|'
      M_ONLY="a.nr4.16k256.glue_cflag,a.nr4.16k256.glue_fold,d.mono.16k256.nr_glue"
      M_MUST="a.nr4.16k256.glue_fold d.mono.16k256.nr_glue" ;;
    xi_min_db_div10_5)
      M_FILE="Audio_ALG/lib/nr/c_impl/src/spp_estimator.c"
      M_PERL='s|config->xi_min_db / 10\.0f|config->xi_min_db / 10.5f|'
      M_ONLY="a.nr4.16k256.glue_cflag,a.nr4.16k256.glue_fold,d.mono.16k256.nr_glue"
      M_MUST="a.nr4.16k256.glue_cflag a.nr4.16k256.glue_fold d.mono.16k256.nr_glue" ;;
    aec_floor_balanced_ulp)
      M_FILE="Audio_ALG/lib/aec/c_impl/src/aec.c"
      M_PERL='s|\[AEC_PRESET_BALANCED\]   = \{ -28\.0f, -20\.0f, 1 \}|[AEC_PRESET_BALANCED]   = { (-28.0f - '"$k"'.0f * 0x1p-19f), -20.0f, 1 }|'
      M_ONLY="b.res4.16k256.balanced,b.res4.16k256.mild"
      M_MUST="b.res4.16k256.balanced" ;;
    cng_seed_1bit)
      M_FILE="Audio_ALG/pipelines/common/se_stage.h"
      M_PERL='s|SE_CNG_RNG_SEED 0x9e3779b9u|SE_CNG_RNG_SEED 0x9e3779b8u|'
      M_ONLY="d.mono.16k256.cng_glue,d.mono.16k256.min_apply_ola"
      M_MUST="d.mono.16k256.cng_glue" ;;
    usable_gate_dropped)
      M_FILE="Audio_ALG/pipelines/common/se_anr_stage.c"
      M_PERL='s| \&\& frame->usable_linear\)|)|'
      M_ONLY="a.nr4.16k256.glue_cflag,a.nr4.16k256.glue_fold,d.mono.16k256.nr_glue"
      M_MUST="a.nr4.16k256.glue_cflag d.mono.16k256.nr_glue" ;;
    flip_s25_c_flag_forced)
      M_FILE="Audio_ALG/pipelines/4ch_aec_bf_nr_res/4aec_nr_res.c"
      M_PERL='s|(    frame->delay_locked = p->last_delay\.solid;\n)|$1    frame->usable_linear = 1;\n|'
      M_ONLY="a.nr4.16k256.verdict_flag,a.nr4.16k256.glue_cflag"
      M_MUST="a.nr4.16k256.verdict_flag" ;;
    flip_s26_python_dt_floor)
      M_FILE="Audio_ALG/pipelines/4ch_aec_bf_nr_res/pipeline.py"
      # Delete the dt_db retarget call in set_preset(): without it only the far-active floor moves and the
      # DT floor stays at the construction value. Inserting a duplicate of the call would be a no-op that
      # no row can detect.
      M_PERL='s|        self\._gain\.set_split_floor_dt_db\(dt_db\)\n||'
      M_ONLY="b.res4.16k256.balanced,b.res4.16k256.aggressive"
      M_MUST="b.res4.16k256.aggressive" ;;
    flip_s27_python_borrow)
      M_FILE="Audio_ALG/pipelines/4ch_aec_bf_nr_res/pipeline.py"
      # Insert a guard that raises on lane far-spectrum disagreement, in the shape of the far_power check a
      # few lines down, before `far_spec = far_specs[0]`.
      M_PERL='s|(    far_spec = far_specs\[0\]\n)|    if not np.allclose(far_specs, far_specs[0], rtol=1e-6, atol=1e-9):\n        raise ValueError("linear AEC lanes do not share one far-end spectrum")\n$1|'
      M_ONLY="c.realign4.48k1024"
      M_MUST="c.realign4.48k1024" ;;
    *) return 1 ;;
  esac
  return 0
}

mp_failed_rows() { # json -> space separated ids with status FAIL / UNEXPECTED_GREEN / a crashed NOT_RUN
  # A planted regression can also surface as NOT_RUN rather than FAIL: pyc_parity.py's main() wraps each
  # scenario-group function (sec_a/b/c/d) in one try/except, so an uncaught exception in a group turns the
  # whole group's rows into a single "scenario group could not run" NOT_RUN row (flip_s27_python_borrow's
  # raise is such a case). Counted only when the note names a raised exception ("<ExceptionType>: <message>"),
  # never for a legitimately vacuous row (e.g. "no mixed warm/soft realign hop occurred"), so a benign skip
  # is never mistaken for detection.
  "${PYRUN[@]}" -c '
import json, sys
d = json.load(open(sys.argv[1]))
def crashed(r):
    return r["status"] == "NOT_RUN" and "Error" in (r.get("note") or "")
print(" ".join(r["id"] for r in d["scenarios"] if r["status"] == "FAIL" or r["status"].startswith("UNEXPECTED_GREEN") or crashed(r)))
' "$1" 2>/dev/null
}

mutation_proof_main() {
  local mpw="$WORK/mutation_proof" copy="$WORK/mutation_proof/tree"
  local make_test="$MAKE_TEST" specs="${MUTATION:-$MP_DEFAULT}"
  mkdir -p "$mpw"
  if [ ! -d "$copy/Audio_ALG" ]; then
    echo "mutation proof: copying the sources to $copy (excluding .git, obj, bin, caches)"
    mkdir -p "$copy"
    ( cd "$TREE" && tar -cf - --exclude .git --exclude obj --exclude bin --exclude __pycache__ --exclude .pytest_cache \
        --exclude dist --exclude '*.o' --exclude '*.a' Audio_ALG audiolib ) | ( cd "$copy" && tar -xf - ) || return 2
  fi
  local common=(--layers 1) rc rows
  # Control: the unmutated copy over the union of the rows must pass, otherwise a FAIL below proves nothing.
  local all_only="a.nr4.16k256.glue_cflag,a.nr4.16k256.glue_fold,a.nr4.16k256.verdict_flag,d.mono.16k256.nr_glue,b.res4.16k256.balanced,b.res4.16k256.mild,b.res4.16k256.aggressive,d.mono.16k256.cng_glue,d.mono.16k256.min_apply_ola,c.realign4.48k1024"
  PYC_TREE="$copy" bash "$SELF" --work "$mpw/run_control" "${common[@]}" --only "$all_only" > "$mpw/control.log" 2>&1
  rc=$?
  echo
  printf '%-26s %-8s %-6s %-9s %s\n' "mutation" "K (ulp)" "exit" "detected" "rows that FAIL"
  printf '%s\n' "----------------------------------------------------------------------------------------------------"
  printf '%-26s %-8s %-6s %-9s %s\n' "(control, no mutation)" "-" "$rc" "-" "$(mp_failed_rows "$mpw/run_control/results.json")"
  [ "$rc" = 0 ] || { echo "control run failed: see $mpw/control.log"; return 2; }
  local overall=0 spec name ks k tag backup detected_any last_k first_k missing r kk det
  for spec in $specs; do
    name="${spec%%:*}"; ks="1"
    case "$spec" in *:*) ks="$(echo "${spec#*:}" | tr ',' ' ')" ;; esac
    detected_any=0; last_k=""; first_k=""; missing=""
    for k in $ks; do
      mp_def "$name" "$k" || { echo "unknown mutation $name"; return 2; }
      backup="$mpw/backup_$name.orig"
      cp -p "$copy/$M_FILE" "$backup" || return 2
      perl -0pi -e "$M_PERL" "$copy/$M_FILE" || return 2
      if cmp -s "$backup" "$copy/$M_FILE"; then
        echo "mutation $name did not change $M_FILE (pattern drifted?)"; return 2
      fi
      tag="${name}_K${k}"
      PYC_TREE="$copy" bash "$SELF" --work "$mpw/run_$tag" "${common[@]}" --only "$M_ONLY" > "$mpw/$tag.log" 2>&1
      rc=$?
      rows="$(mp_failed_rows "$mpw/run_$tag/results.json")"
      cp -p "$backup" "$copy/$M_FILE"
      det="no"
      # Detection is "did the row(s) we're watching turn up", not "was the sub-run's exit code exactly 1": the
      # sub-run also exits 3 for INCOMPLETE (its NOT_RUN-only outcome, e.g. a whole scenario group crashing, see
      # mp_failed_rows above), which is just as real a detection.
      [ -n "$rows" ] && det="YES" && detected_any=1
      [ "$rc" = 2 ] && [ -z "$rows" ] && det="build error"
      printf '%-26s %-8s %-6s %-9s %s\n' "$name" "$k" "$rc" "$det" "$rows"
      [ -n "$first_k" ] || first_k="$k"
      last_k="$k"
      # at the largest K every row named in M_MUST has to be among the failing rows
      missing=""
      for r in $M_MUST; do case " $rows " in *" $r "*) ;; *) missing="$missing $r" ;; esac; done
    done
    [ "$detected_any" = 1 ] || { overall=1; echo "  -> $name: NOT detected at any K"; }
    [ -z "$missing" ] || { overall=1; echo "  -> $name: rows expected to fail at K=$last_k but did not:$missing"; }
    if [ "$make_test" = 1 ] && [ "${name#flip_}" = "$name" ]; then
      for kk in $first_k $last_k; do
        mp_def "$name" "$kk"
        cp -p "$copy/$M_FILE" "$backup"
        perl -0pi -e "$M_PERL" "$copy/$M_FILE"
        mp_make_test "$name" "$kk" "$copy"
        cp -p "$backup" "$copy/$M_FILE"
        [ "$first_k" = "$last_k" ] && break
      done
    fi
  done
  echo
  echo "mutation proof: $([ "$overall" = 0 ] && echo 'every mutation was detected at some K' || echo 'AT LEAST ONE MUTATION WAS NOT DETECTED')"
  return $overall
}

# The repository's own gates on the mutated copy: make SIMD=0 test for pipelines and for the 4-channel directory.
mp_make_test() { # name K copy
  local copy="$3" root="$WORK/mutation_proof/make_test_$1_K$2" log rc1 rc4 nf1 nf4
  mkdir -p "$root"
  log="$root/pipelines.log"
  ( cd "$copy/Audio_ALG/pipelines" && make -j"$JOBS" OBJ_ROOT="$root/obj" BIN_ROOT="$root/bin" SIMD=0 test ) > "$log" 2>&1; rc1=$?
  nf1="$(grep -c -E '^FAIL|: FAIL| FAILED' "$log")"
  log="$root/4ch.log"
  ( cd "$copy/Audio_ALG/pipelines/4ch_aec_bf_nr_res" && make -j"$JOBS" OBJ_ROOT="$root/obj4" BIN_ROOT="$root/bin4" SIMD=0 test ) > "$log" 2>&1; rc4=$?
  nf4="$(grep -c -E '^FAIL|: FAIL| FAILED' "$log")"
  printf '  make SIMD=0 test on the mutated copy (%s, K=%s): pipelines exit %s (%s FAIL lines), 4ch exit %s (%s FAIL lines)\n' "$1" "$2" "$rc1" "$nf1" "$rc4" "$nf4"
}
