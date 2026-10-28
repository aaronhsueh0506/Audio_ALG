#!/bin/bash
# test_cli_strict.sh <bin_dir> -- the command lines of mcab_static and mcab_external_static (and,
# for the option scan they share with it, the file options of mcab_raw and --erb-fwd/--erb-inv of
# 4ch_aec_bf_dfn_res).
#
# Every numeric option takes a whole decimal number or the program exits 2 (a command-line
# error): an empty or padded string, trailing characters, a value outside int, and for the
# float options inf and nan. An option without its value and an unknown option exit 2 as well.
# A refused config (a number that parses but names no grid) is a different failure and stays 1.
# Valid command lines run to completion (exit 0).
#
# A token that starts with "--" where an option's value belongs is the next option, so the value
# is missing (exit 2): it is never taken as the value, whatever the option (number, preset name,
# file). A lone "-", "-1" and "-x" are values. mcab_external_static also refuses an AEC preset
# or --nr-preset name it does not know and a second AEC preset.
set -u
BIN=${1:?usage: test_cli_strict.sh <bin_dir>}
STATIC=$BIN/mcab_static
EXTERNAL=$BIN/mcab_external_static
RAW=$BIN/mcab_raw
DFN4=$BIN/4ch_aec_bf_dfn_res
# What can stand where a value belongs: flags of the program, an unknown one, and the bare "--".
FLAGS=(--print-mem-size --no-res --no-cng --legacy-amin --sample-rate --bogus --)
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

# An option name in place of the value is a missing value, with and without --print-mem-size.
for flag in "${FLAGS[@]}"; do
    for opt in --sample-rate --fft-size --uca-radius-m --fixed-doa-deg; do
        expect 2 "$STATIC" "$opt" "$flag"
        expect 2 "$STATIC" --print-mem-size "$opt" "$flag"
        expect 2 "$STATIC" "$opt" "$flag" --print-mem-size
    done
    for opt in --sample-rate --fft-size --capture-proxy --nr-preset; do
        expect 2 "$EXTERNAL" "$opt" "$flag"
        expect 2 "$EXTERNAL" --print-mem-size "$opt" "$flag"
        expect 2 "$EXTERNAL" "$opt" "$flag" --print-mem-size
    done
    for opt in --mic-raw --ref-raw --output-raw --vad-u8; do   # the file options of mcab_raw
        expect 2 "$RAW" "$opt" "$flag" --mic-raw a --ref-raw b --output-raw /dev/null --sample-rate 16000
        expect 2 "$RAW" --print-mem-size "$opt" "$flag"
    done
    expect 2 "$DFN4" --erb-fwd "$flag" --erb-inv x
    expect 2 "$DFN4" --erb-inv "$flag" --erb-fwd x
done
# The reproducer: --nr-preset must not take --no-res as its preset.
expect 2 "$EXTERNAL" --print-mem-size --nr-preset --no-res
expect 2 "$EXTERNAL" --nr-preset --no-res
expect 2 "$EXTERNAL" 1 --nr-preset --no-res --print-mem-size
# A flag-looking file name that is a value for the file option is a command-line error too, even
# when the other options are all there and the file would otherwise be opened.
expect 2 "$RAW" --vad-u8 --print-mem-size --mic-raw a --ref-raw b --output-raw /dev/null --sample-rate 16000
expect 2 "$RAW" --mic-raw --print-mem-size --ref-raw b --output-raw /dev/null --sample-rate 16000
expect 2 "$DFN4" --erb-fwd --cng --erb-inv x

expect 2 "$STATIC" --bogus
expect 2 "$STATIC" 16000
# Names the program does not know are refused, not replaced by balanced; one AEC preset only.
for v in "" bogus Mild " mild" "mild " 4 -1 - -x 00; do
    expect 2 "$EXTERNAL" --print-mem-size --nr-preset "$v"
done
for v in "" bogus Mild " mild" 4 00; do
    expect 2 "$EXTERNAL" --print-mem-size "$v"
done
expect 2 "$EXTERNAL" --print-mem-size 1 2
expect 2 "$EXTERNAL" --print-mem-size mild mild
expect 2 "$EXTERNAL" --print-mem-size 1 --no-res bogus
expect 2 "$EXTERNAL" --bogus
expect 2 "$EXTERNAL" -

# A bad number is refused before anything is built, wherever it stands on the line.
expect 2 "$STATIC" --sample-rate 16000 --fft-size 256 --fixed-doa-deg 3x
expect 2 "$EXTERNAL" 1 --nr-preset mild --sample-rate 16000 --capture-proxy 1x

# A value that does not start with "--" is a value: a negative number is one, and so is a file
# name such as "-x" (the file is then missing: the run fails, the command line does not).
expect 1 "$EXTERNAL" --capture-proxy -1
expect 1 "$EXTERNAL" --sample-rate -16000
expect 2 "$EXTERNAL" --capture-proxy -
expect 2 "$EXTERNAL" --capture-proxy -x
expect 1 "$RAW" --mic-raw -x --ref-raw b --output-raw /dev/null --sample-rate 16000
expect 1 "$RAW" --mic-raw - --ref-raw b --output-raw /dev/null --sample-rate 16000

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
expect 0 "$STATIC" --fixed-doa-deg -30
expect 0 "$STATIC" --print-mem-size --fixed-doa-deg -0.5 --uca-radius-m 0.04
expect 0 "$EXTERNAL" --print-mem-size --no-res
expect 0 "$EXTERNAL" 2 --nr-preset aggressive --capture-proxy 3 --sample-rate 16000 --fft-size 512
expect 0 "$EXTERNAL" --sample-rate +48000 --fft-size 1024 --capture-proxy +1
for v in mild moderate balanced aggressive 0 1 2 3; do
    expect 0 "$EXTERNAL" --print-mem-size --nr-preset "$v"
done
for v in mild balanced aggressive no_floor 0 1 2 3; do
    expect 0 "$EXTERNAL" --print-mem-size "$v"
done
expect 0 "$EXTERNAL" --print-mem-size --no-res --nr-preset mild --no-cng 3 --legacy-amin

if [ "$failures" -ne 0 ]; then
    echo "test_cli_strict: $failures of $total command lines FAILED" >&2
    exit 1
fi
echo "test_cli_strict: $total command lines PASS"
