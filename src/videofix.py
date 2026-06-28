#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import mmap

def hevc_aud_bytes(au_index: int, aus) -> bytes:
    """
    生成 HEVC AUD bytes，逻辑和 AVC 一样：
      - I 帧 AU = 0x10
      - 其它 AU 按 0x30,0x50 循环
      - 空 AU 不强制 I
    """
    au = aus[au_index]
    if au["slices"]:
        first_slice_type = au["slices"][0]["type"]
    else:
        first_slice_type = None

    if first_slice_type in (2, 19, 20):  # I slice / IDR_W_RADL / IDR_N_LP
        primary_pic_type = 0x10
    else:
        cycle = [0x30, 0x50, 0x50]
        last_i_index = -1
        for j in range(au_index - 1, -1, -1):
            s = aus[j]["slices"]
            if s and s[0]["type"] in (2, 19, 20):
                last_i_index = j
                break
        offset = au_index - last_i_index - 1
        primary_pic_type = cycle[offset % len(cycle)]

    # 组装 HEVC AUD
    return bytes.fromhex(f"00 00 00 01 46 01 {primary_pic_type:02X} 00")

def avc_aud_bytes(au_index: int, aus) -> bytes:
    """
    生成 AVC AUD bytes，严格逻辑：
      - I 帧 AU 前 AUD = 0x10
      - P/B AU 循环 0x30,0x50,0x50
      - 空 AU 不强制 0x10
    """
    au = aus[au_index]
    if au["slices"]:
        first_slice_type = au["slices"][0]["type"]
    else:
        # 空 AU，没有 slice
        first_slice_type = None

    if first_slice_type == 5:  # IDR Slice
        primary_pic_type = 0x10
    else:
        # 非 I AU 循环 AUD
        cycle = [0x30, 0x50, 0x50]
        last_i_index = -1
        # 找最近上一个 I AU
        for j in range(au_index - 1, -1, -1):
            s = aus[j]["slices"]
            if s and s[0]["type"] == 5:
                last_i_index = j
                break
        offset = au_index - last_i_index - 1
        primary_pic_type = cycle[offset % len(cycle)]

    return bytes.fromhex(f"00 00 00 01 09 {primary_pic_type:02X}")

def detect_codec_by_ext(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".h264", ".avc"):
        return "avc"
    if ext in (".hevc", ".h265"):
        return "hevc"
    return "avc"

def parse_nal_type_avc(b: int) -> int:
    return b & 0x1F

def parse_nal_type_hevc(b0: int) -> int:
    return (b0 & 0x7E) >> 1

def is_vcl_avc(nal_type: int) -> bool:
    return nal_type in (1, 5)

def is_vcl_hevc(nal_type: int) -> bool:
    return 0 <= nal_type <= 31

def is_aud_avc(nal_type: int) -> bool:
    return nal_type == 9

def is_aud_hevc(nal_type: int) -> bool:
    return nal_type == 35

def scan_all_nalus(input_path: str, codec: str):
    nalus = []
    with open(input_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        size = mm.size()
        find_pat = b"\x00\x00\x01"
        pos = 0
        while True:
            i = mm.find(find_pat, pos)
            if i == -1:
                break
            nal_start = i
            if i > 0 and mm[i - 1] == 0x00:
                nal_start = i - 1
            header_idx = i + 3
            if header_idx >= size:
                break
            if codec == "avc":
                nal_type = parse_nal_type_avc(mm[header_idx])
                if is_aud_avc(nal_type):
                    kind = "AUD"
                elif is_vcl_avc(nal_type):
                    kind = "VCL"
                else:
                    kind = "NONVCL"
            else:
                nal_type = parse_nal_type_hevc(mm[header_idx])
                if is_aud_hevc(nal_type):
                    kind = "AUD"
                elif is_vcl_hevc(nal_type):
                    kind = "VCL"
                else:
                    kind = "NONVCL"
            nalus.append({"pos": nal_start, "type": nal_type, "kind": kind})
            pos = i + 3
        mm.close()
    return nalus

def group_into_aus(nalus):
    aus = []
    current = None
    def new_au():
        return {"aud": [], "prefix": [], "slices": []}
    for nal in nalus:
        kind = nal["kind"]
        if kind == "AUD":
            if current and (current["aud"] or current["prefix"] or current["slices"]):
                aus.append(current)
            current = new_au()
            current["aud"].append(nal)
        elif kind == "VCL":
            if current is None:
                current = new_au()
            current["slices"].append(nal)
        else:
            if current is None:
                current = new_au()
                current["prefix"].append(nal)
            else:
                if current["slices"]:
                    aus.append(current)
                    current = new_au()
                    current["prefix"].append(nal)
                else:
                    current["prefix"].append(nal)
    if current and (current["aud"] or current["prefix"] or current["slices"]):
        aus.append(current)

    insertion_offsets = []
    for au in aus:
        # 找 AU 内最小位置（替换原 AUD 或插入）
        candidates = []
        for key in ("aud", "prefix", "slices"):
            for n in au[key]:
                candidates.append(n["pos"])
        if candidates:
            insertion_offsets.append(min(candidates))
    insertion_offsets = sorted(set(insertion_offsets))
    return insertion_offsets, aus

def insert_bytes_at_offsets(input_path: str, output_path: str, offsets, codec: str, aus):
    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        current = 0
        for i, off in enumerate(offsets):
            if off < current:
                continue
            fin.seek(current)
            chunk = fin.read(off - current)
            fout.write(chunk)
            # 插入 AUD
            if codec == "avc":
                fout.write(avc_aud_bytes(i, aus))
            else:
                fout.write(hevc_aud_bytes(i, aus))
            current = off
        fin.seek(current)
        fout.write(fin.read())

def main():
    if len(sys.argv) < 3:
        print("用法: python nalu_scanner_insert.py <input.(avc|h264|hevc|h265)> <output>")
        sys.exit(1)
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    if not os.path.exists(input_path):
        print(f"输入文件不存在: {input_path}")
        sys.exit(1)
    codec = detect_codec_by_ext(input_path)
    print(f"检测编码类型: {codec}")
    print("扫描所有 NALU ...")
    nalus = scan_all_nalus(input_path, codec)
    print(f"找到 {len(nalus)} 个 NAL 单元。")
    print("按 AU 分组并计算插入偏移 ...")
    offsets, _aus = group_into_aus(nalus)
    print(f"共 {len(offsets)} 个 AU，将在这些位置插入 AUD：前 10 个偏移示例：{offsets[:10]}")
    print("插入 AUD 并写出文件 ...")
    insert_bytes_at_offsets(input_path, output_path, offsets, codec, _aus)
    print(f"完成。输出文件：{output_path}")

if __name__ == "__main__":
    main()
