#!/bin/sh
# test_cli_exit_codes.sh -- the command-line exit codes of the example programs: a malformed or unsupported argument
# is refused at parse time with the documented exit code 2, a refusal by the library (a pool query, a missing
# file) is exit 1. A crash or a later rejection is nonzero too, so each case names its exact code.
#
# Usage: sh scripts/test_cli_exit_codes.sh PROGRAM BINARY
#   PROGRAM  mono_aec_dfn_res | mono_alignulcnet | 4ch_aec_bf_dfn_res | 4ch_alignulcnet
#   BINARY   the built program
# Exit 0 if every case returns its code; 1 with the first mismatch; 2 on a usage error.

set -u

[ $# -eq 2 ] || { echo "usage: sh $0 PROGRAM BINARY" >&2; exit 2; }
prog="$1"
bin="$2"
[ -x "$bin" ] || { echo "FAIL: no executable $bin" >&2; exit 2; }

# expect CODE ARG...: run the program with the arguments, output discarded, and compare the exit code.
expect() {
    want="$1"; shift
    "$bin" "$@" >/dev/null 2>&1
    rc=$?
    [ "$rc" -eq "$want" ] || { echo "FAIL: $prog $*: exit $rc, want $want" >&2; exit 1; }
}

# expect_each CODE ARGS...: one expect per argument string; an argument string is split at spaces.
expect_each() {
    want="$1"; shift
    for a in "$@"; do
        # shellcheck disable=SC2086
        expect "$want" $a
    done
}

# Empty, leading or trailing garbage and out-of-int-range numbers.
delay_num_filters_garbage() {
    for v in "" " 3" "3 " x3 +-1 99999999999 999999999999999999999999; do
        expect 2 --delay-num-filters "$v"
    done
}

case "$prog" in
mono_aec_dfn_res)
    # The DFN2 model grid is a run-time choice (--ai-sample-rate / --ai-fft-size, one of the rows of
    # dfn/dfn_example_args.h): a host on the model's own rate runs the model stream directly at that grid, any other host
    # rate needs the bridge, which accepts the 48 kHz / 1024 model only (pool query rejected, exit 1); a pair outside
    # the table is a rejected argument (2).
    expect 2 --ai-sample-rate 32000 --ai-fft-size 512
    expect 1 --ai-sample-rate 16000 --ai-fft-size 256 --sample-rate 8000
    expect 1 --ai-sample-rate 16000 --ai-fft-size 256 --sample-rate 48000
    expect_each 2 "--ai-sample-rate 44100" "--ai-sample-rate 16000" "--ai-fft-size 512" "--ai-fft-size 256x" \
        "--ai-sample-rate --cng" "--ai-fft-size --cng"
    # Malformed numbers are refused at parse time, not merely "some failure".
    expect 2 --sample-rate 16000Hz
    expect 2 --delay-mode fixed --fixed-delay 12x
    expect 2 --atten-lim 12dB
    # An option followed by another option has no value (exit 2); "-x" is a value, here a file that is missing (exit 1).
    expect_each 2 "--erb-fwd --cng --erb-inv x" "--erb-inv --cng --erb-fwd x" "--erb-fwd" "--atten-lim --cng" \
        "--sample-rate --cng" "--delay-num-filters --cng" "--delay-mode --" "--fixed-delay --"
    expect 1 --erb-fwd -x --erb-inv -y
    ;;
mono_alignulcnet|4ch_alignulcnet)
    expect 2 --sample-rate 44100
    expect 2 --sample-rate 16000x
    expect 2 --delay-num-filters 3x
    expect 2 --delay-mode fixed --fixed-delay 12x
    delay_num_filters_garbage
    ;;
4ch_aec_bf_dfn_res)
    # The DFN2 model grid is a run-time choice (--ai-sample-rate / --ai-fft-size): the host on the model's own rate
    # runs the model stream directly at that grid, any other host rate needs the bridge (48 kHz / 1024 model only: pool query
    # rejected, exit 1), a pair outside the table is exit 2.
    expect 2 --ai-sample-rate 32000 --ai-fft-size 512
    expect 1 --ai-sample-rate 16000 --ai-fft-size 256 --sample-rate 48000
    expect_each 2 "--ai-sample-rate 44100" "--ai-sample-rate 16000" "--ai-fft-size 512" "--ai-fft-size 256x" \
        "--ai-sample-rate --cng"
    # Malformed numbers, and the 8 kHz grid the core does not offer, are refused at parse time.
    expect 2 --sample-rate 16000Hz
    expect 2 --sample-rate 8000
    expect 2 --atten-lim 12dB
    ;;
*)
    echo "usage: $0 PROGRAM BINARY (unknown PROGRAM '$prog')" >&2
    exit 2
    ;;
esac

echo "PASS: $prog command-line exit codes"
