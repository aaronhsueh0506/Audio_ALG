#!/bin/sh
# test_delay_depth_refusal.sh -- an Align-ULCNet example built with an out-of-range -DULCNET_EXAMPLE_DELAY_FRAMES
# refuses at start-up with exit 1 and one stderr line naming the depth and the accepted range.
#
# Usage: sh scripts/test_delay_depth_refusal.sh PROGRAM BINARY DEPTH MIN MAX
#   PROGRAM  the name the message starts with (mono_alignulcnet | 4ch_alignulcnet)
#   BINARY   the example built with -DULCNET_EXAMPLE_DELAY_FRAMES=DEPTH
# Exit 0 when exit code, stdout and stderr match; 1 on a mismatch; 2 on a usage error.

set -u

[ $# -eq 5 ] || { echo "usage: sh $0 PROGRAM BINARY DEPTH MIN MAX" >&2; exit 2; }
prog="$1"; bin="$2"; depth="$3"; min="$4"; max="$5"
[ -x "$bin" ] || { echo "FAIL: no executable $bin" >&2; exit 2; }

out="$("$bin" 2>/dev/null)"
err="$("$bin" 2>&1 >/dev/null)"
rc=$?
want="$prog: grid refused (-4): delay depth $depth, accepted range [$min, $max] (-DULCNET_EXAMPLE_DELAY_FRAMES)"

[ "$rc" -eq 1 ] || { echo "FAIL: $prog at D=$depth: exit $rc, want 1" >&2; exit 1; }
[ -z "$out" ] || { echo "FAIL: $prog at D=$depth: stdout not empty: $out" >&2; exit 1; }
[ "$err" = "$want" ] || { printf 'FAIL: %s at D=%s: stderr\n  got:  %s\n  want: %s\n' "$prog" "$depth" "$err" "$want" >&2; exit 1; }
echo "$prog: D=$depth refused with the range message (exit 1)"
