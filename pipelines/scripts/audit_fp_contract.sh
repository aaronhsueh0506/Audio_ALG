#!/bin/sh
# audit_fp_contract.sh -- disassembly check for the pipeline objects that run
# windowed overlap-add.
#
# Why this exists here and not in audio_common: the KERNEL those objects call
# (sk_wola_accumulate_f32) is audited by audio_common's own script, against a
# non-inline instantiation in that repo. The CALL SITES are a separate
# question -- each of these translation units also carries its own per-bin
# floating-point work, and none of them was ever disassembled. Being compiled
# under -ffp-contract=off is a Makefile property; being fma-free is a property
# of the emitted code, and only a disassembly says so. audio_common must not
# reach up into Audio_ALG, so the audit for these objects lives with them.
#
# The object under test is BUILT BY THIS SCRIPT and its directory is then
# queried from the very same make configuration. Picking "the newest object
# lying around" would not survive scrutiny: the obj/ tree accumulates many
# config signatures (backend, SIMD, and the producer signatures of
# audio_common/AEC/NR, which move whenever those repos are edited), so an
# audit of an unrelated stale object would pass while saying nothing.
#
# Usage:  scripts/audit_fp_contract.sh [BACKEND ...]      (default: ne10 kiss)
#         SIMD=0|1 (default 1) and WERROR are honoured and forwarded.
# Exit 0 if every audited object disassembles free of fma-class instructions.

set -eu

cd "$(dirname "$0")/.."
BACKENDS="${*:-ne10 kiss}"
SIMD="${SIMD:-1}"
WERROR="${WERROR:-1}"

# The forbidden mnemonic set and the disassembler wrapper come from
# audio_common, so extending the set is one edit rather than two copies that
# can silently disagree. Depending on audio_common is the allowed direction;
# the reverse is not, which is why the audit LIST below stays here.
AC_DIR="${AC_DIR:-../../audio_common}"
. "$AC_DIR/scripts/fp_contract_lib.sh"
FMA_RE="$FP_CONTRACT_FMA_RE"
disas() { fp_contract_disas "$1"; }

# One row per TU: "make-dir:build-goal:object:note". Every one of these runs
# the WOLA accumulate and must stay fma-free, so the shared kernel's
# multiply-then-add rounding is what the whole path performs.
ENTRIES='
.:libaudio_pipeline.a:audio_pipeline.o:mono AEC+NR pipeline -- post-NR/RES synthesis WOLA
4ch_aec_bf_nr_res:libs:4aec_nr_res.o:4-channel core -- post-beam synthesis WOLA
4ch_aec_bf_nr_res:4ch_alignulcnet:audio_pipeline_4ch_ulcnet.o:4-channel Align-ULCNet -- beamformed-spectrum WOLA
'

fail=0
audited=0
rows=$(printf '%s\n' "$ENTRIES" | grep -cv '^[[:space:]]*$')
printf '%-6s %-6s %-34s %-5s %s\n' BACKEND SIMD OBJECT FMA RESULT
for be in $BACKENDS; do
    printf '%s\n' "$ENTRIES" | grep -v '^[[:space:]]*$' | while IFS=: read -r dir goal obj note; do
        MK="BACKEND=$be SIMD=$SIMD WERROR=$WERROR"
        # Build first, then ask THAT configuration where its objects went.
        # shellcheck disable=SC2086
        make -s --no-print-directory -C "$dir" $MK "$goal" >/dev/null
        # shellcheck disable=SC2086
        objdir=$(make -s --no-print-directory -C "$dir" $MK print-obj-dir | tail -1)
        path="$objdir/$obj"
        if [ ! -f "$path" ]; then
            echo "FATAL: $obj not found at $path after building '$goal' in $dir" >&2
            echo "       (print-obj-dir must resolve the SAME config signature the" >&2
            echo "        build used -- see this script's header)" >&2
            exit 1
        fi
        n=$(disas "$path" | grep -icE "$FMA_RE" || true)
        if [ "$n" -eq 0 ]; then
            printf '%-6s %-6s %-34s %-5s %s\n' "$be" "$SIMD" "$obj" "$n" PASS
        else
            printf '%-6s %-6s %-34s %-5s %s\n' "$be" "$SIMD" "$obj" "$n" FAIL
            echo "     -- $note" >&2
            echo "     -- $path" >&2
            exit 1
        fi
    done || fail=1
    audited=$((audited + rows))
done

echo
if [ "$fail" -ne 0 ]; then
    echo "FAIL: the audit did not complete cleanly -- see the reason above" >&2
    exit 1
fi
echo "objects audited: $audited (backends: $BACKENDS, SIMD=$SIMD)"
echo "PASS: every audited pipeline object disassembled fma-class-instruction-free"
