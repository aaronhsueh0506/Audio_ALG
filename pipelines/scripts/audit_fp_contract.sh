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
# Exit 0 if every audited object disassembles free of fma-class instructions, apart from the
# functions named in EXEMPT_FNS.

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
# DFN spectrum path or do per-bin spectral arithmetic that the C/Python parity gate
# compares, and must stay fma-free so the scalar/NEON and C/Python multiply-then-add
# contracts remain comparable. The per-hop gain chain the example programs write out only calls
# kernels audited elsewhere.
ENTRIES='
.:libaudio_pipeline_dfn.a:dfn_res_stage.o:shared DFN post-RES stage -- boundary scaling by dfn2_stft_scale()/dfn2_stft_unscale()
.:libaudio_pipeline_dfn.a:dfn_rate_bridge.o:DFN rate bridge -- native and 48 kHz WOLA, analysis windows
.:libaudio_pipeline_dfn.a:dfn2_process.o:DFN feature and streaming enhance-spectrum kernels
.:mono_alignulcnet:ulcnet_nn_stage.o:Align-ULCNet stage (mono build) -- far-end analysis and enhanced-spectrum synthesis
4ch:libs:mcab.o:4-channel flow -- lane-spectrum projection and the fused PSD views
4ch:libs:mcab_sample_beam.o:4-channel sample beamformer helper -- SRP-PHAT and beamformer glue
4ch:4ch_alignulcnet:ulcnet_nn_stage.o:Align-ULCNet stage (4-channel build) -- far-end analysis and enhanced-spectrum synthesis
'

# An explicit fma() call is deterministic by definition, not a contraction of a*b+c, so a function
# that rounds a value exactly on purpose is not audited for fma-class instructions: dfn2_norm_alpha
# (and its helper round_decimals, which a compiler may inline) rounds the EMA coefficient to decimals
# once at init. One row per object, "object:function ...". The match is on the function's own
# disassembly block, never on the object: every other function of the object stays fma-free.
EXEMPT_FNS='
dfn2_process.o:dfn2_norm_alpha round_decimals
'

# fma_split <object> <function names>: prints "<fma-class lines outside the named functions>
# <fma-class lines inside them>". The disassembly is cut at its function labels (objdump's
# "ADDR <name>:", otool's "name:"; a leading underscore and a compiler suffix such as .constprop.0
# are ignored). A Mach-O objdump labels the first function of the text section with the assembler
# temporary "ltmp0" instead of its name, so for a label of that form only, the start addresses of
# the named symbols (nm) are matched too. A named label is never matched by address: with
# -ffunction-sections every function starts at section offset 0 and an address match would exempt
# them all.
fma_split() {
    addrs=$(nm "$1" 2>/dev/null | awk -v names=" $2 " '
        $2 ~ /^[TtWw]$/ { n = $3; sub(/^_/, "", n); sub(/\..*$/, "", n)
                          if (index(names, " " n " ")) { a = tolower($1); sub(/^0+/, "", a); print (a == "" ? "0" : a) } }')
    disas "$1" | awk -v re="$FMA_RE" -v names=" $2 " -v addrs=" $(echo $addrs) " '
        function exempt_block(label, addr,   n, hit) {
            n = label; sub(/^_/, "", n); sub(/\..*$/, "", n)
            hit = n != "" && index(names, " " n " ") > 0
            if (addr != "" && label ~ /^ltmp[0-9]+$/) {
                sub(/^0+/, "", addr); if (addr == "") addr = "0"
                hit = hit || index(addrs, " " tolower(addr) " ") > 0
            }
            return hit
        }
        /^[0-9a-fA-F]+ <[^>]+>:$/ { l = $2; sub(/^</, "", l); sub(/>:$/, "", l); in_ex = exempt_block(l, $1); next }
        /^[^ \t]+:$/ { l = $1; sub(/:$/, "", l); in_ex = exempt_block(l, ""); next }
        tolower($0) ~ re { if (in_ex) inside++; else outside++ }
        END { printf "%d %d\n", outside, inside }'
}

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
        exempt=$(printf '%s\n' "$EXEMPT_FNS" | awk -F: -v o="$obj" '$1 == o { print $2 }')
        set -- $(fma_split "$path" "$exempt")
        n=$1
        if [ "$n" -eq 0 ]; then
            printf '%-6s %-6s %-34s %-5s %s\n' "$be" "$SIMD" "$obj" "$n" "PASS${exempt:+ ($2 explicit fma() in $exempt)}"
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
