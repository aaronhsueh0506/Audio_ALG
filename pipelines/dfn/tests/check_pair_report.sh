#!/bin/sh
# Gate of `make test-pairs` (pipelines/Makefile): read a DFN test program's output on stdin and check
# the "lookahead pair" line it prints (dfn_test_grid.h, dfn_test_report_pair()).
#
#   sh check_pair_report.sh <program> <mask> <df> <need_measured>
#
# The line says which lookahead pair the program was BUILT with, so a pair flag that never reached
# the test build is caught here: it prints 1/1 and make asked for another pair. When the program also
# measured the latency (need_measured = 1; the stage test from the hops until its first emitted
# spectrum, the mono composition from the shift against the hand-written chain) the stage latency L
# must equal mask + df and the algorithmic latency (L + 1) * hop must equal (mask + df + 1) * hop.
# Exit 0 and the accepted line on success; exit 1 and the reason otherwise.
prog=$1
mask=$2
df=$3
need=$4
awk -v prog="$prog" -v m="$mask" -v d="$df" -v need="$need" '
function fail(msg) { printf "FAIL: %s: %s\n  line: %s\n", prog, msg, $0; failed = 1; exit 1 }
/^[A-Za-z0-9_]+: lookahead pair / {
    seen++
    line = $0
    if (!match(line, /lookahead pair [0-9]+\/[0-9]+/)) fail("malformed lookahead pair line")
    pair = substr(line, RSTART + 15, RLENGTH - 15)
    if (pair != m "/" d) fail("built for lookahead pair " pair " but make asked for " m "/" d)
    if (need == 1 && line !~ /measured stage latency/) fail("no measured latency on the pair line")
    if (line ~ /measured stage latency/) {
        if (!match(line, /measured stage latency [0-9]+ hops/)) fail("malformed measured latency")
        stage = substr(line, RSTART + 23, RLENGTH - 23); sub(/ hops.*/, "", stage)
        if (!match(line, /\* [0-9]+ = [0-9]+ samples/)) fail("malformed algorithmic latency")
        split(substr(line, RSTART, RLENGTH), f, " ")
        hop = f[2] + 0; samples = f[4] + 0
        if (stage + 0 != m + d) fail("measured stage latency " stage " hops, expected mask + df = " m + d)
        if (samples != (m + d + 1) * hop)
            fail("algorithmic latency " samples " samples, expected (" m " + " d " + 1) * " hop " = " (m + d + 1) * hop)
    }
    printf "  %s\n", line
}
END { if (!failed && !seen) { printf "FAIL: %s printed no lookahead pair line\n", prog; exit 1 } }
'
