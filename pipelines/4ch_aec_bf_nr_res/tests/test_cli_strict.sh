#!/bin/bash
# test_cli_strict.sh <bin_dir> -- the command lines of mcab_static and mcab_external_static.
#
# Every numeric option takes a whole decimal number or the program exits 2 (a command-line
# error): an empty or padded string, trailing characters, a value outside int, and for the
# float options inf and nan. An option without its value and an unknown option exit 2 as well.
# A refused config (a number that parses but names no grid) is a different failure and stays 1.
# Valid command lines run to completion (exit 0).
set -u
BIN=${1:?usage: test_cli_strict.sh <bin_dir>}
STATIC=$BIN/mcab_static
EXTERNAL=$BIN/mcab_external_static
BAD_INT=("" " 16000" "16000 " "16000Hz" x "1e3" "0x10" "1.5" "+-1" 99999999999 -99999999999
         2147483648 -2147483649 999999999999999999999999)
BAD_FLOAT=("" " 0.035" "0.035 " "0.035m" inf -inf nan NaN 1e99 -1e99 0x "--")
failures=0
total=0

expect() {      # expect <exit code> <program> <args...>
    local want=$1 prog=$2 got
    shift 2
    "$prog" "$@" >/dev/null 2>&1
    got=$?
    total=$((total + 1))
    if [ "$got" -ne "$want" ]; then
        printf 'FAIL: %s' "$(basename "$prog")"
        printf " '%s'" "$@"
        printf ' -> exit %s, expected %s\n' "$got" "$want"
        failures=$((failures + 1))
    fi
}

for v in "${BAD_INT[@]}"; do
    for opt in --sample-rate --fft-size; do
        expect 2 "$STATIC" "$opt" "$v"
        expect 2 "$STATIC" --print-mem-size "$opt" "$v"
    done
    for opt in --sample-rate --fft-size --capture-proxy; do
        expect 2 "$EXTERNAL" "$opt" "$v"
        expect 2 "$EXTERNAL" --print-mem-size "$opt" "$v"
    done
done
for v in "${BAD_FLOAT[@]}"; do
    for opt in --uca-radius-m --fixed-doa-deg; do
        expect 2 "$STATIC" --print-mem-size "$opt" "$v"
    done
done

for opt in --sample-rate --fft-size --uca-radius-m --fixed-doa-deg; do
    expect 2 "$STATIC" "$opt"
    expect 2 "$STATIC" --print-mem-size "$opt"
done
for opt in --sample-rate --fft-size --capture-proxy --nr-preset; do
    expect 2 "$EXTERNAL" "$opt"
    expect 2 "$EXTERNAL" --no-cng "$opt"
done

expect 2 "$STATIC" --bogus
expect 2 "$STATIC" 16000
expect 2 "$EXTERNAL" --bogus
expect 2 "$EXTERNAL" -

# A bad number is refused before anything is built, wherever it stands on the line.
expect 2 "$STATIC" --sample-rate 16000 --fft-size 256 --fixed-doa-deg 3x
expect 2 "$EXTERNAL" 1 --nr-preset mild --sample-rate 16000 --capture-proxy 1x

# Numbers that parse but name no grid or channel: the config is refused, not the command line.
expect 1 "$STATIC" --sample-rate 44100
expect 1 "$STATIC" --sample-rate 16000 --fft-size 1024
expect 1 "$STATIC" --uca-radius-m -1
expect 1 "$EXTERNAL" --sample-rate 44100
expect 1 "$EXTERNAL" --sample-rate 16000 --fft-size 1024
expect 1 "$EXTERNAL" --capture-proxy 4

# Valid command lines.
expect 0 "$STATIC" --print-mem-size
expect 0 "$STATIC" --sample-rate 16000 --fft-size 256 --uca-radius-m 0.0325 --fixed-doa-deg 30
expect 0 "$STATIC" --sample-rate +48000 --fft-size 1024
expect 0 "$EXTERNAL" --print-mem-size --no-res
expect 0 "$EXTERNAL" 2 --nr-preset aggressive --capture-proxy 3 --sample-rate 16000 --fft-size 512
expect 0 "$EXTERNAL" --sample-rate +48000 --fft-size 1024 --capture-proxy +1

if [ "$failures" -ne 0 ]; then
    echo "test_cli_strict: $failures of $total command lines FAILED" >&2
    exit 1
fi
echo "test_cli_strict: $total command lines PASS"
