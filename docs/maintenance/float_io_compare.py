#!/usr/bin/env python3
"""Compare two board runs of one AI3 sample: A = FLOAT_IN/FLOAT_OUT 0 (CPU converts), B = 1 (NPU converts).

Inputs are whatever the run left behind; every part is optional and the report only covers what is given:
  --audio-a/--audio-b   final output WAVs (16-bit PCM, same input, same length)
  --heads-a/--heads-b   raw little-endian float32 dumps of the head tensors, hop after hop, heads in graph order
                        within a hop; --head-elems 24,960,1 gives the element count of each head per hop
  --log-a/--log-b       the sample's stdout; numeric "name: value" / "name=value" pairs, the timing buckets
                        (names ending in us/ms or starting with io_, ai3_, conv, wall) and the lines that print
                        scale_ratio / zero_point of the ports are compared
A non-finite value must be the same on both sides at the same position (NaN against NaN, +Inf against +Inf); one
that is finite on only one side, or of another kind, fails a --fail-max-abs limit. No tolerance is built in: derive it from the first measurement. --fail-max-abs / --fail-audio-lsb turn the
corresponding number into an exit status when a limit has been chosen. Runs that cannot be compared (a different
length or channel count, an empty WAV, a head dump shorter than one hop) fail such a limit; without a limit they
are reported only.

Usage:
  python3 -I float_io_compare.py --audio-a a.wav --audio-b b.wav --heads-a a.f32 --heads-b b.f32 \
      --head-elems 24,960,1 --log-a a.log --log-b b.log
"""
import argparse
import array
import math
import re
import struct
import sys
import wave

TIMING = re.compile(r"(_us|_ms|us|ms)$|^(io_|ai3_|conv|wall|cpu_|npu_)", re.I)
# A name is at most eight words of 64 characters: an unbounded name made a long line without ':' or '=' quadratic.
PAIR = re.compile(r"([A-Za-z_][\w.\-/]{0,63}(?: [\w.\-/]{1,63}){0,7})\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
MAX_LINE = 16384   # longer lines (binary noise, a capture without newlines) are scanned only up to here
PORT = re.compile(r"scale_ratio|zero_point", re.I)
MODE = re.compile(r"\[AI3\]\s*float_in=(\d+)\s+float_out=(\d+)(.*)")


def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            sys.exit("%s: only 16-bit PCM is supported" % path)
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    a = array.array("h")
    a.frombytes(raw)
    if sys.byteorder == "big":
        a.byteswap()
    return a, ch


def audio_report(pa, pb):
    (a, ca), (b, cb) = read_wav(pa), read_wav(pb)
    print("== final audio ==")
    incomparable = False
    if ca != cb or len(a) != len(b):
        print("  length/channels differ: A %d samples x%d, B %d samples x%d (compared over the common part)"
              % (len(a) // ca, ca, len(b) // cb, cb))
        incomparable = True
    n = min(len(a), len(b))
    if n == 0:
        print("  empty")
        return float("inf")
    mx = 0
    se = sa = 0.0
    for i in range(n):
        d = a[i] - b[i]
        mx = max(mx, abs(d))
        se += d * d
        sa += a[i] * a[i]
    rms_d = math.sqrt(se / n)
    snr = 10.0 * math.log10(sa / se) if se > 0 and sa > 0 else float("inf")
    print("  samples %d  max |A-B| %d LSB  rms(A-B) %.3f LSB  SNR(A vs A-B) %.1f dB" % (n, mx, rms_d, snr))
    return float("inf") if incomparable else float(mx)


def read_f32(path):
    a = array.array("f")
    with open(path, "rb") as f:
        data = f.read()
    a.frombytes(data[: len(data) // 4 * 4])
    if sys.byteorder == "big":
        a.byteswap()
    if len(data) % 4:
        print("  warning: %s has %d trailing bytes" % (path, len(data) % 4))
    return a


def head_report(pa, pb, elems):
    a, b = read_f32(pa), read_f32(pb)
    print("== head tensors ==")
    n = min(len(a), len(b))
    incomparable = n == 0
    if len(a) != len(b):
        print("  size differs: A %d floats, B %d floats (compared over the common part)" % (len(a), len(b)))
        incomparable = True
    per_hop = sum(elems) if elems else n
    hops = n // per_hop if per_hop else 0
    if hops == 0:
        print("  fewer than one whole hop in the common part")
        incomparable = True
    ranges = []
    if elems:
        off = 0
        for k, e in enumerate(elems):
            ranges.append(("head %d (%d elems)" % (k, e), off, e))
            off += e
    else:
        ranges.append(("all", 0, per_hop))
    worst = 0.0
    for name, off, e in ranges:
        mx = se = sa = 0.0
        bad_a = bad_b = cnt = moved = 0
        for h in range(max(hops, 1)):
            base = h * per_hop + off
            for i in range(base, min(base + e, n)):
                x, y = a[i], b[i]
                fx, fy = math.isfinite(x), math.isfinite(y)
                bad_a += not fx
                bad_b += not fy
                if fx != fy or (not fx and (math.isnan(x) != math.isnan(y) or (not math.isnan(x) and x != y))):
                    moved += 1   # finite on one side only, or NaN / +Inf / -Inf of a different kind
                if fx and fy:
                    d = x - y
                    mx = max(mx, abs(d))
                    se += d * d
                    sa += x * x
                    cnt += 1
        rel = math.sqrt(se / sa) if sa > 0 else (0.0 if se == 0 else float("inf"))
        print("  %-22s hops %d  max|A-B| %.6g  rel RMS %.6g  non-finite A %d B %d  non-finite mismatches %d"
              % (name, hops, mx, rel, bad_a, bad_b, moved))
        worst = max(worst, mx, float("inf") if moved else 0.0)   # a non-finite value that is not the same on both sides is never within a limit
    return float("inf") if incomparable else worst


def parse_log(path):
    pairs, ports, mode = {}, [], None
    with open(path, errors="replace") as f:
        for line in f:
            m = MODE.search(line)
            if m:
                mode = (int(m.group(1)), int(m.group(2)), m.group(3).strip())
            if PORT.search(line):
                ports.append(line.strip())
            for k, v in PAIR.findall(line[:MAX_LINE]):
                pairs[k.strip()] = float(v)
    return pairs, ports, mode


def log_report(pa, pb):
    (ka, pta, ma), (kb, ptb, mb) = parse_log(pa), parse_log(pb)
    print("== modes ==")
    for tag, m in (("A", ma), ("B", mb)):
        print("  %s: %s" % (tag, "float_in=%d float_out=%d %s" % m if m else "no '[AI3] float_in=.. float_out=..' line"))
    print("== timing buckets (B/A) ==")
    keys = [k for k in ka if k in kb and TIMING.search(k)]
    if not keys:
        print("  no common timing buckets found")
    for k in keys:
        r = kb[k] / ka[k] if ka[k] else float("nan")
        print("  %-32s A %-12.6g B %-12.6g B/A %.3f" % (k, ka[k], kb[k], r))
    print("== port scale_ratio / zero_point (as printed) ==")
    for tag, lines in (("A", pta), ("B", ptb)):
        print("  -- %s: %d line(s)" % (tag, len(lines)))
        for ln in lines[:64]:
            print("     " + ln)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-a")
    ap.add_argument("--audio-b")
    ap.add_argument("--heads-a")
    ap.add_argument("--heads-b")
    ap.add_argument("--head-elems", help="comma-separated element counts of the heads per hop")
    ap.add_argument("--log-a")
    ap.add_argument("--log-b")
    ap.add_argument("--fail-max-abs", type=float, help="exit 1 when the largest head |A-B| exceeds this")
    ap.add_argument("--fail-audio-lsb", type=float, help="exit 1 when the largest audio |A-B| exceeds this")
    args = ap.parse_args()

    if bool(args.audio_a) != bool(args.audio_b) or bool(args.heads_a) != bool(args.heads_b) \
            or bool(args.log_a) != bool(args.log_b):
        ap.error("each comparison needs both its A and its B file")
    if not (args.audio_a or args.heads_a or args.log_a):
        ap.error("nothing to compare")
    elems = None
    if args.head_elems:
        try:
            elems = [int(x) for x in args.head_elems.split(",")]
        except ValueError:
            ap.error("--head-elems: comma-separated integers")
        if any(e <= 0 for e in elems):
            ap.error("--head-elems: counts must be positive")

    rc = 0
    if args.log_a:
        log_report(args.log_a, args.log_b)
    if args.heads_a:
        worst = head_report(args.heads_a, args.heads_b, elems)
        if args.fail_max_abs is not None and worst > args.fail_max_abs:
            print("FAIL: head max|A-B| %.6g > %.6g" % (worst, args.fail_max_abs))
            rc = 1
    if args.audio_a:
        worst = audio_report(args.audio_a, args.audio_b)
        if args.fail_audio_lsb is not None and worst > args.fail_audio_lsb:
            print("FAIL: audio max|A-B| %g LSB > %g" % (worst, args.fail_audio_lsb))
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
