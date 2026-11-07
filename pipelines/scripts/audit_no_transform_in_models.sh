#!/bin/sh
# audit_no_transform_in_models.sh -- source gate: no framing code inside a model directory. Framing (the
# AcStft analysis, the AcIstft inverse FFT and overlap-add) is audiolib's (audio_stft.h), built and run by
# the application; a model directory holds neither side of the transform and builds, sizes or runs no
# AcStft or AcIstft. Scans the non-test C sources and headers of AINR/ and AIAEC/.
#
# The one audiolib framing use a model directory may have is the n_fft check: ac_stft_config_default()
# filling an AcStftConfig and ac_stft_get_mem_size() of it (non-zero = a size the framing builds).
#
# Forbidden identifiers in code (comments are blanked first, so headers may show the application-side
# usage; string literals are scanned):
#   synthesis   fft_inverse, fft_inverse_scratch, sk_wola_*, skn_div_floor*, ifft, irfft, overlap_add,
#               ola_buf*, ola_state
#   analysis    fft_forward, fft_forward_scratch, spec_scale, ispec_scale
#   framing     AcStft, AcIstft, AcOlaMode, AC_OLA_*, every ac_istft_* and every ac_stft_* except
#               ac_stft_config_default and ac_stft_get_mem_size
#
# Usage:  scripts/audit_no_transform_in_models.sh [ROOT]     (ROOT = the Audio_ALG tree, default: this one)
# Exit 0 if no hit; 1 with the hits listed (file:line:identifier) otherwise.

set -eu

ROOT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"

[ -d "$ROOT/AINR" ] && [ -d "$ROOT/AIAEC" ] || { echo "FAIL: $ROOT has no AINR/ and AIAEC/" >&2; exit 2; }

# Non-test model sources: tests, third-party checkouts and build trees are not model code.
files=$(find "$ROOT/AINR" "$ROOT/AIAEC" \
    \( -name test -o -name tests -o -name gtcrn_github -o -name build -o -name obj -o -name bin -o -name output \) -prune -o \
    -type f \( -name '*.c' -o -name '*.h' \) -print | sort)
[ -n "$files" ] || { echo "FAIL: no model sources found under $ROOT" >&2; exit 2; }
count=$(printf '%s\n' "$files" | wc -l | tr -d ' ')

# Blank the comments (comment markers inside a string or character literal are text, the literal itself
# stays), then test every identifier.
hits=$(printf '%s\n' "$files" | tr '\n' '\0' | xargs -0 awk -v q="'" '
function forbidden(t) {
    if (t ~ /^(fft_inverse|fft_inverse_scratch|fft_forward|fft_forward_scratch|ifft|irfft|overlap_add|ola_state|spec_scale|ispec_scale|AcStft|AcIstft|AcOlaMode)$/) return 1
    if (t ~ /^(sk_wola_|skn_div_floor|ola_buf|ac_stft_|ac_istft_|AC_OLA_)/)
        return t !~ /^(ac_stft_config_default|ac_stft_get_mem_size)$/
    return 0
}
FNR == 1 { inblock = 0 }
{
    line = $0; out = ""; i = 1; n = length(line)
    while (i <= n) {
        c = substr(line, i, 1); two = substr(line, i, 2)
        if (inblock) {
            if (two == "*/") { inblock = 0; i += 2 } else i++
        } else if (two == "/*") { inblock = 1; i += 2; out = out " " }
        else if (two == "//") break
        else if (c == "\"" || c == q) {
            lit = c; i++
            while (i <= n) {
                d = substr(line, i, 1); lit = lit d; i++
                if (d == "\\") { lit = lit substr(line, i, 1); i++; continue }
                if (d == c) break
            }
            out = out lit
        } else { out = out c; i++ }
    }
    rest = out
    while (match(rest, /[A-Za-z_][A-Za-z0-9_]*/)) {
        t = substr(rest, RSTART, RLENGTH); rest = substr(rest, RSTART + RLENGTH)
        if (forbidden(t)) print FILENAME ":" FNR ":" t
    }
}')
if [ -n "$hits" ]; then
    echo "FAIL: framing code in model directories -- analysis and synthesis belong to the application (audiolib AcStft/AcIstft); only the n_fft check pair is allowed:"
    printf '%s\n' "$hits"
    exit 1
fi
echo "PASS: no framing code in $count model sources under AINR/ and AIAEC/ (the n_fft check pair only)"
