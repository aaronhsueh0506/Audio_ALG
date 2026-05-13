#!/usr/bin/env python3
"""
顯示 encoded.bin 的 RLE pair，可與 raw.bin 或 C model 輸出做對比。

Header 類型：
  split   : z=zmax, nz=0  — 溢位 split header（必定後接有 nz>0 的 final）
  z_start : z=0,   nz>0   — 只有第一個 header 才允許
  trail   : z>0,   nz=0   — trailing zeros（非 zmax，為 trailing final）
  final   : z>0,   nz>0   — 一般 run 的最後一個 header

用法：
  python3 show_encoded.py --enc encoded.bin --bit 16
  python3 show_encoded.py --enc encoded.bin --raw raw.bin --bit 16
  python3 show_encoded.py --enc encoded.bin --raw raw.bin --cmodel cmodel_enc.bin --bit 16
"""

import argparse
import struct
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--enc",    required=True, help="我們產生的 encoded.bin")
    p.add_argument("--raw",    default=None,  help="raw.bin（選填，用於解碼驗證）")
    p.add_argument("--cmodel", default=None,  help="C model 壓縮輸出（選填，與我們的做對比）")
    p.add_argument("--bit",    type=int, choices=[16, 32], default=16)
    return p.parse_args()


def read_words(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 2 != 0:
        print(f"[警告] {path} 大小 {len(data)} bytes 不是 2 的倍數", file=sys.stderr)
    return list(struct.unpack_from(f"<{len(data)//2}H", data))


def decode_16bit(words, zn_order="z_hi"):
    """zn_order: 'z_hi' (我們的格式: z 在高 byte) 或 'z_lo' (z 在低 byte)"""
    runs = []
    i = 0
    while i < len(words):
        hdr = words[i]; i += 1
        if zn_order == "z_hi":
            z  = (hdr >> 8) & 0xFF
            nz =  hdr       & 0xFF
        else:
            z  =  hdr       & 0xFF
            nz = (hdr >> 8) & 0xFF
        payload = words[i:i+nz]
        i += nz
        runs.append((z, nz, payload))
    return runs


def decode_32bit(words, zn_order="z_hi", word_order="lo_first"):
    """zn_order: 'z_hi' (z 在高 16-bit) 或 'z_lo' (z 在低 16-bit)
    word_order: 'lo_first' (我們的格式: lo word 先) 或 'hi_first'"""
    runs = []
    i = 0
    while i + 1 < len(words):
        w0 = words[i]; w1 = words[i+1]; i += 2
        if word_order == "lo_first":
            w = (w1 << 16) | w0
        else:
            w = (w0 << 16) | w1
        if zn_order == "z_hi":
            z  = (w >> 16) & 0xFFFF
            nz =  w        & 0xFFFF
        else:
            z  =  w        & 0xFFFF
            nz = (w >> 16) & 0xFFFF
        payload = words[i:i+nz]
        i += nz
        runs.append((z, nz, payload))
    return runs


def try_decode(words, bit, raw_elems):
    """嘗試所有 byte order 組合，回傳能正確 decode 出 raw_elems 的格式。
    回傳 list of (zn_order, word_order, runs_count)"""
    results = []
    if bit == 16:
        configs = [("z_hi", None), ("z_lo", None)]
    else:
        configs = [
            ("z_hi", "lo_first"), ("z_hi", "hi_first"),
            ("z_lo", "lo_first"), ("z_lo", "hi_first"),
        ]
    for cfg in configs:
        zn, wo = cfg
        try:
            if bit == 16:
                runs = decode_16bit(words, zn_order=zn)
            else:
                runs = decode_32bit(words, zn_order=zn, word_order=wo)
            decoded = to_element_stream(runs)
            if decoded == raw_elems:
                results.append((zn, wo, len(runs)))
        except (IndexError, ValueError):
            pass
    return results


def to_element_stream(runs):
    elems = []
    for z, nz, payload in runs:
        elems.extend([0] * z)
        elems.extend(payload)
    return elems


def merge_splits(runs, zmax, hw):
    """合併 split header 成邏輯 run，回傳 (z_total, nz, payload, enc_words)"""
    merged = []
    acc_z = 0
    acc_hdrs = 0
    for z, nz, payload in runs:
        acc_hdrs += 1
        if nz == 0 and z == zmax:   # split header
            acc_z += z
        else:                        # final header (包含 z_start, trail, normal final)
            acc_z += z
            words = hw * acc_hdrs + nz
            merged.append((acc_z, nz, payload, words))
            acc_z = 0
            acc_hdrs = 0
    if acc_z > 0:
        print(f"[警告] 串流結尾有未配對的 split header（累積 z={acc_z}）", file=sys.stderr)
    return merged


def header_kind(z, nz, zmax):
    if   nz == 0 and z == zmax: return "split"
    elif nz == 0:               return "trail"
    elif z  == 0:               return "z_start"
    else:                       return "final"


def fmt_payload(payload, limit=8):
    s = " ".join(f"0x{v:04X}" for v in payload[:limit])
    if len(payload) > limit:
        s += f" ... (+{len(payload)-limit})"
    return s


def compare_with_raw(decoded_elems, raw_elems, label):
    n_dec = len(decoded_elems)
    n_raw = len(raw_elems)
    if n_dec != n_raw:
        print(f"[FAIL] {label}: element 數量不符 decoded={n_dec}, raw={n_raw}")
        return False
    mismatches = [(i, raw_elems[i], decoded_elems[i])
                  for i in range(n_raw) if raw_elems[i] != decoded_elems[i]]
    if not mismatches:
        print(f"[PASS] {label}: 全部 {n_raw} elements bit-exact 相符")
        return True
    print(f"[FAIL] {label}: {len(mismatches)} 個 element 不符（共 {n_raw}）")
    print(f"{'index':>8}  {'raw':>8}  {'decoded':>8}")
    print("-" * 30)
    for i, rv, dv in mismatches[:20]:
        print(f"{i:>8}  0x{rv:04X}  0x{dv:04X}")
    if len(mismatches) > 20:
        print(f"  ... 另有 {len(mismatches)-20} 個不符")
    return False


def show_encoded(enc_words, bit, zmax, hw, label="encoded"):
    if bit == 16:
        runs = decode_16bit(enc_words)
    else:
        runs = decode_32bit(enc_words)

    decoded = to_element_stream(runs)

    # ── Raw headers（含 split）──
    print(f"=== {label} — Raw headers（{len(runs)} 個，含 split）===")
    print(f"{'#':>5}  {'z':>8}  {'nz':>6}  {'enc_off':>8}  {'words':>6}  {'type':<8}  payload（前8個）")
    print("-" * 80)
    enc_off = 0
    for idx, (z, nz, payload) in enumerate(runs):
        kind  = header_kind(z, nz, zmax)
        words = hw + nz
        print(f"{idx:>5}  {z:>8}  {nz:>6}  {enc_off:>8}  {words:>6}  {kind:<8}  {fmt_payload(payload)}")
        enc_off += words
    print()

    # ── 邏輯 runs（split 合併後）──
    logical = merge_splits(runs, zmax, hw)
    print(f"=== {label} — 邏輯 runs（{len(logical)} 個，split 已合併）===")
    print(f"{'#':>5}  {'zeros':>8}  {'nonzero':>8}  {'enc_off':>8}  {'enc_words':>10}  {'type':<8}  payload（前8個）")
    print("-" * 85)
    enc_off = 0
    for idx, (z, nz, payload, words) in enumerate(logical):
        kind = "trail"   if nz == 0 else \
               "z_start" if z  == 0 else "normal"
        print(f"{idx:>5}  {z:>8}  {nz:>8}  {enc_off:>8}  {words:>10}  {kind:<8}  {fmt_payload(payload)}")
        enc_off += words
    print()

    # ── 統計 ──
    total_z  = sum(z  for z, nz, _ in runs)
    total_nz = sum(nz for z, nz, _ in runs)
    print(f"解碼後總 elements = {len(decoded)}  (zero={total_z}, nonzero={total_nz})")
    print(f"encoded words     = {len(enc_words)}")
    print()

    return decoded


def main():
    args = parse_args()
    zmax = 255 if args.bit == 16 else 65535
    hw   = 1   if args.bit == 16 else 2

    enc_words = read_words(args.enc)
    print(f"encoded : {args.enc}  ({len(enc_words)} words = {len(enc_words)*2} bytes)  bit={args.bit}")

    raw_elems = None
    if args.raw:
        raw_words = read_words(args.raw)
        raw_elems = raw_words
        print(f"raw     : {args.raw}  ({len(raw_words)} elements = {len(raw_words)*2} bytes)")

    if args.cmodel:
        cmodel_words = read_words(args.cmodel)
        print(f"cmodel  : {args.cmodel}  ({len(cmodel_words)} words = {len(cmodel_words)*2} bytes)")
    print()

    # ── 顯示我們的 encoded ──
    our_decoded = show_encoded(enc_words, args.bit, zmax, hw, label="我們的 encoded")

    # ── 與 raw.bin 對比 ──
    if raw_elems is not None:
        print("=== 驗證：我們的 encoded vs raw.bin ===")
        compare_with_raw(our_decoded, raw_elems, "我們的 encoded")
        print()

    # ── C model 對比 ──
    if args.cmodel:
        # ── Byte order 自動偵測（必須有 raw.bin 才能比對）──
        if raw_elems is not None:
            print("=== Byte order 自動偵測：嘗試所有解讀方式找出 C model 用的格式 ===")
            matches = try_decode(cmodel_words, args.bit, raw_elems)
            if not matches:
                print("[FAIL] 沒有任何 byte order 組合能讓 C model 的 encoded decode 出 raw.bin")
                print("      可能是 encoding 規則不同 (例如 trail/split semantics)，不只是 byte order")
            else:
                print(f"[PASS] 找到 {len(matches)} 個能正確 decode 的格式：")
                for zn, wo, n_runs in matches:
                    our_fmt = (args.bit == 16 and zn == "z_hi") or \
                              (args.bit == 32 and zn == "z_hi" and wo == "lo_first")
                    flag = "  ← 我們的格式" if our_fmt else ""
                    if args.bit == 16:
                        print(f"  zn_order={zn}{flag} ({n_runs} headers)")
                    else:
                        print(f"  zn_order={zn} word_order={wo}{flag} ({n_runs} headers)")
            print()

            # ── Hex dump 並排 ──
            print("=== 前 32 bytes hex dump 對比 ===")
            our_bytes = b"".join(w.to_bytes(2, "little") for w in enc_words[:16])
            cmd_bytes = b"".join(w.to_bytes(2, "little") for w in cmodel_words[:16])
            print(f"  ours  : {our_bytes.hex(' ')}")
            print(f"  cmodel: {cmd_bytes.hex(' ')}")
            print()

        cmodel_decoded = show_encoded(cmodel_words, args.bit, zmax, hw, label="C model encoded")

        print("=== 對比：C model decoded vs raw.bin (用我們的格式 z_hi) ===")
        if raw_elems is not None:
            compare_with_raw(cmodel_decoded, raw_elems, "C model encoded")
        else:
            print("[跳過] 未提供 --raw，無法對比 raw.bin")
        print()

        print("=== 對比：C model decoded vs 我們的 decoded ===")
        compare_with_raw(cmodel_decoded, our_decoded, "C model vs ours")
        print()


if __name__ == "__main__":
    main()
