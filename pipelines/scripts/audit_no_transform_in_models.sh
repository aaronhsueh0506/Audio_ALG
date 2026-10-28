#!/bin/sh
# audit_no_transform_in_models.sh -- source gate: no transform code (inverse FFT, overlap-add) inside a
# model directory. The inverse FFT and overlap-add are audiolib's AcIstft (audio_stft.h), owned by the
# application; a model directory may configure one (AcStftConfig), borrow one (ac_istft_push*, _reset,
# _get_config) and own an AcStft analysis, but must not contain the transform itself or size/construct a
# synthesis object. Scans the non-test C sources and headers of AINR/ and AIAEC/.
#
# Forbidden identifiers in code (comments are blanked first, so headers may show the application-side
# usage; string literals are scanned):
#   fft_inverse, fft_inverse_scratch      the inverse transform
#   sk_wola_*, skn_div_floor*             the overlap-add and envelope kernels AcIstft is built on
#   ac_istft_init, ac_istft_get_mem_size  constructing / sizing a synthesis object
#   ifft, irfft, overlap_add, ola_buf*, ola_state   hand-rolled synthesis
#
# Usage:  scripts/audit_no_transform_in_models.sh [ROOT]     (ROOT = the Audio_ALG tree, default: this one)
# Exit 0 if no hit; 1 with the hits listed otherwise.

set -eu

ROOT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
PATTERN='(^|[^A-Za-z0-9_])(fft_inverse(_scratch)?|sk_wola_[a-z0-9_]*|skn_div_floor[a-z0-9_]*|ac_istft_init|ac_istft_get_mem_size|ifft|irfft|overlap_add|ola_buf[a-z0-9_]*|ola_state)($|[^A-Za-z0-9_])'

[ -d "$ROOT/AINR" ] && [ -d "$ROOT/AIAEC" ] || { echo "FAIL: $ROOT has no AINR/ and AIAEC/" >&2; exit 2; }

# Non-test model sources: tests, third-party checkouts and build trees are not model code.
files=$(find "$ROOT/AINR" "$ROOT/AIAEC" \
    \( -name test -o -name tests -o -name gtcrn_github -o -name build -o -name obj -o -name bin -o -name output \) -prune -o \
    -type f \( -name '*.c' -o -name '*.h' \) -print | sort)
[ -n "$files" ] || { echo "FAIL: no model sources found under $ROOT" >&2; exit 2; }
count=$(printf '%s\n' "$files" | wc -l | tr -d ' ')

# Blank the comments (keeping line numbers), then look for the identifiers.
strip_comments() {
    awk '
    BEGIN { inblock = 0 }
    {
        line = $0; out = ""; i = 1; n = length(line)
        while (i <= n) {
            two = substr(line, i, 2)
            if (inblock) {
                if (two == "*/") { inblock = 0; i += 2 } else { i += 1 }
            } else if (two == "/*") { inblock = 1; i += 2 }
            else if (two == "//") { break }
            else { out = out substr(line, i, 1); i += 1 }
        }
        print out
    }' "$1"
}
hits=$(printf '%s\n' "$files" | while IFS= read -r f; do
    strip_comments "$f" | grep -nE "$PATTERN" | sed "s|^|$f:|" || true
done)
if [ -n "$hits" ]; then
    echo "FAIL: transform code (inverse FFT / overlap-add) in model directories -- it belongs to the application's AcIstft:"
    printf '%s\n' "$hits"
    exit 1
fi
echo "PASS: no inverse-FFT / overlap-add code in $count model sources under AINR/ and AIAEC/"
