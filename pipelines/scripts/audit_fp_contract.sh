#!/bin/sh
# audit_fp_contract.sh -- disassembly check for pipeline objects whose multiply/add ordering is part of
# the numerical contract. The kernel they call (sk_wola_accumulate_f32) is audited by audiolib's own
# script; this audits the call sites, which carry WOLA or DFN per-bin floating-point work. Compiling
# under -ffp-contract=off is a Makefile property, but being fma-free is a property of the emitted code
# that only a disassembly shows. audiolib must not reach up into Audio_ALG, so the audit for these
# objects lives here.
#
# The object under test is built by this script and its directory is queried from the same make
# configuration: the obj/ tree accumulates many config signatures, so auditing the newest object lying
# around could pass on an unrelated stale one while saying nothing.
#
# Usage:  scripts/audit_fp_contract.sh [BACKEND ...]      (default: ne10, the only backend)
#         SIMD=0|1 (default 1) and WERROR are honoured and forwarded.
# Exit 0 if every audited object disassembles free of fma-class instructions.

set -eu

cd "$(dirname "$0")/.."
BACKENDS="${*:-ne10}"
SIMD="${SIMD:-1}"
WERROR="${WERROR:-1}"

# The forbidden mnemonic set and the disassembler wrapper come from
# audiolib, so extending the set is one edit rather than two copies that
# can silently disagree. Depending on audiolib is the allowed direction;
# the reverse is not, which is why the audit LIST below stays here.
COMMON_DIR="${COMMON_DIR:-../../audiolib/common}"
. "$COMMON_DIR/scripts/fp_contract_lib.sh"
FMA_RE="$FP_CONTRACT_FMA_RE"
disas() { fp_contract_disas "$1"; }

# One row per TU: "make-dir:build-goal:object:note". These either run WOLA, implement the
# DFN spectrum/compose path or do per-bin spectral arithmetic that the C/Python parity gate
# compares, and must stay fma-free so the scalar/NEON and C/Python multiply-then-add
# contracts remain comparable. The gain chain of samples/ is header-only and compiled into the
# example programs; it only calls kernels audited elsewhere.
ENTRIES='
.:libaudio_pipeline_dfn.a:dfn_res_stage.o:shared DFN post-RES stage -- 2^-5/2^5 boundary scaling
.:libaudio_pipeline_dfn.a:dfn_rate_bridge.o:DFN rate bridge -- native and 48 kHz WOLA, analysis windows
.:libaudio_pipeline_dfn.a:dfn2_process.o:DFN feature and streaming compose kernels
.:mono_alignulcnet:ulcnet_nn_stage.o:Align-ULCNet stage (mono build) -- far-end analysis and enhanced-spectrum synthesis
4ch_aec_bf_nr_res:libs:4aec_nr_res.o:4-channel core -- lane-spectrum projection and the fused PSD views
4ch_aec_bf_nr_res:libs:audio_pipeline_4ch.o:4-channel unit -- SRP-PHAT and beamformer spectral math
4ch_aec_bf_nr_res:4ch_alignulcnet:ulcnet_nn_stage.o:Align-ULCNet stage (4-channel build) -- far-end analysis and enhanced-spectrum synthesis
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
