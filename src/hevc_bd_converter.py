#!/usr/bin/env python3
"""
HEVC WEB-DL to UHD Blu-ray Structure Converter
Converts HEVC WEB-DL streams into a structure compatible with UHD 4K Blu-ray, *without* re-encoding the video.
Preserves pixel data and chroma information; only modifies NAL unit structure and metadata.

Usage: python hevc_bd_converter.py input.hevc output.hevc [options]

Key Changes:
1. All NAL units utilize 4-byte start codes (00 00 00 01).
2. `tier_flag` in VPS/SPS is set to High (1); `level_idc` is set to 5.1 (153).
3. VUI parameters (timing info, HRD, colour description, etc.) are added/corrected within the SPS.
4. An AUD (Access Unit Delimiter) is inserted before every Access Unit (AU).
5. VPS/SPS/PPS units are repeated prior to each IDR/CRA frame.
6. SEI messages are properly arranged (retaining HDR metadata while removing x265 user data).
7. An EOS (End of Sequence) NAL unit is appended to the end of the stream.
"""

import sys
import os
import struct
import argparse
import io
import re
import multiprocessing
import subprocess
import shutil
import tempfile
import time
from typing import List, Tuple, Optional, Dict

# ============================================================
# HEVC NAL Unit Types
# ============================================================
NAL_TRAIL_N    = 0
NAL_TRAIL_R    = 1
NAL_TSA_N      = 2
NAL_TSA_R      = 3
NAL_STSA_N     = 4
NAL_STSA_R     = 5
NAL_RADL_N     = 6
NAL_RADL_R     = 7
NAL_RASL_N     = 8
NAL_RASL_R     = 9
NAL_BLA_W_LP   = 16
NAL_BLA_W_RADL = 17
NAL_BLA_N_LP   = 18
NAL_IDR_W_RADL = 19
NAL_IDR_N_LP   = 20
NAL_CRA_NUT    = 21
NAL_VPS        = 32
NAL_SPS        = 33
NAL_PPS        = 34
NAL_AUD        = 35
NAL_EOS_NUT    = 36
NAL_EOB_NUT    = 37
NAL_FD_NUT     = 38
NAL_SEI_PREFIX = 39
NAL_SEI_SUFFIX = 40

NAL_NAMES = {
    0: 'TRAIL_N', 1: 'TRAIL_R', 2: 'TSA_N', 3: 'TSA_R',
    4: 'STSA_N', 5: 'STSA_R', 6: 'RADL_N', 7: 'RADL_R',
    8: 'RASL_N', 9: 'RASL_R',
    16: 'BLA_W_LP', 17: 'BLA_W_RADL', 18: 'BLA_N_LP',
    19: 'IDR_W_RADL', 20: 'IDR_N_LP', 21: 'CRA_NUT',
    32: 'VPS', 33: 'SPS', 34: 'PPS', 35: 'AUD',
    36: 'EOS', 37: 'EOB', 38: 'FD',
    39: 'SEI_PREFIX', 40: 'SEI_SUFFIX'
}

# SEI payload types
SEI_BUFFERING_PERIOD = 0
SEI_PIC_TIMING = 1
SEI_USER_DATA_REG = 4
SEI_USER_DATA_UNREG = 5
SEI_RECOVERY_POINT = 6
SEI_ACTIVE_PARAM_SETS = 129
SEI_MASTERING_DISPLAY = 137
SEI_CONTENT_LIGHT_LEVEL = 144
SEI_ALT_TRANSFER_CHAR = 147

FOUR_BYTE_SC = b'\x00\x00\x00\x01'

# ============================================================
# Bitstream Reader/Writer
# ============================================================
class BitstreamReader:
    def __init__(self, data: bytes):
        self.data = data
        self.bit_pos = 0

    def get_bits(self, n: int) -> int:
        val = 0
        for _ in range(n):
            byte_idx = self.bit_pos >> 3
            bit_idx = 7 - (self.bit_pos & 7)
            if byte_idx < len(self.data):
                val = (val << 1) | ((self.data[byte_idx] >> bit_idx) & 1)
            else:
                val = val << 1
            self.bit_pos += 1
        return val

    def get_ue(self) -> int:
        lz = 0
        while self.get_bits(1) == 0:
            lz += 1
            if lz > 32:
                return 0
        return (1 << lz) - 1 + self.get_bits(lz)

    def get_se(self) -> int:
        v = self.get_ue()
        return -(v >> 1) if (v & 1) == 0 else (v + 1) >> 1

    @property
    def bits_left(self) -> int:
        return len(self.data) * 8 - self.bit_pos


class BitstreamWriter:
    def __init__(self):
        self.data = bytearray()
        self.current_byte = 0
        self.bit_count = 0

    def put_bits(self, n: int, val: int):
        for i in range(n - 1, -1, -1):
            self.current_byte = (self.current_byte << 1) | ((val >> i) & 1)
            self.bit_count += 1
            if self.bit_count == 8:
                self.data.append(self.current_byte)
                self.current_byte = 0
                self.bit_count = 0

    def put_ue(self, val: int):
        if val == 0:
            self.put_bits(1, 1)
            return
        val_plus1 = val + 1
        nbits = val_plus1.bit_length()
        self.put_bits(nbits - 1, 0)  # leading zeros
        self.put_bits(nbits, val_plus1)

    def put_se(self, val: int):
        if val > 0:
            self.put_ue(2 * val - 1)
        elif val < 0:
            self.put_ue(-2 * val)
        else:
            self.put_ue(0)

    def flush(self) -> bytes:
        if self.bit_count > 0:
            # RBSP trailing bits: 1 followed by 0s
            self.current_byte <<= (8 - self.bit_count)
            self.current_byte |= (1 << (7 - self.bit_count))
            self.data.append(self.current_byte)
            self.current_byte = 0
            self.bit_count = 0
        return bytes(self.data)

    def rbsp_trailing_bits(self):
        self.put_bits(1, 1)
        while self.bit_count != 0:
            self.put_bits(1, 0)


# ============================================================
# Emulation Prevention
# ============================================================
def remove_emulation_prevention(data: bytes) -> bytes:
    result = bytearray()
    i = 0
    while i < len(data):
        if (i + 2 < len(data) and data[i] == 0 and data[i + 1] == 0
                and data[i + 2] == 3):
            if i + 3 < len(data) and data[i + 3] in (0, 1, 2, 3):
                result.append(0)
                result.append(0)
                i += 3  # skip 0x03
                continue
        result.append(data[i])
        i += 1
    return bytes(result)


def add_emulation_prevention(data: bytes) -> bytes:
    result = bytearray()
    i = 0
    count_zeros = 0
    while i < len(data):
        if count_zeros == 2 and data[i] in (0, 1, 2, 3):
            result.append(0x03)
            count_zeros = 0
        if data[i] == 0:
            count_zeros += 1
        else:
            count_zeros = 0
        result.append(data[i])
        i += 1
    return bytes(result)


# ============================================================
# NAL Unit Parsing (Streaming)
# ============================================================
class NALUnit:
    def __init__(self, nal_type: int, layer_id: int, temporal_id_plus1: int,
                 rbsp: bytes, start_code_len: int = 4):
        self.nal_type = nal_type
        self.layer_id = layer_id
        self.temporal_id_plus1 = temporal_id_plus1
        self.rbsp = rbsp  # raw data after NAL header (with emulation prevention)
        self.start_code_len = start_code_len

    @property
    def type_name(self) -> str:
        return NAL_NAMES.get(self.nal_type, f'UNK({self.nal_type})')

    def is_vcl(self) -> bool:
        return self.nal_type <= 31

    def is_irap(self) -> bool:
        return 16 <= self.nal_type <= 23

    def is_idr(self) -> bool:
        return self.nal_type in (NAL_IDR_W_RADL, NAL_IDR_N_LP)

    def full_data_with_header(self) -> bytes:
        """Return NAL header + RBSP (with emulation prevention bytes)"""
        hdr_byte0 = (self.nal_type << 1) | (self.layer_id >> 5)
        hdr_byte1 = ((self.layer_id & 0x1F) << 3) | (self.temporal_id_plus1 & 0x7)
        return bytes([hdr_byte0, hdr_byte1]) + self.rbsp


def parse_nal_header(data: bytes, offset: int) -> Tuple[int, int, int]:
    """Parse 2-byte NAL unit header, return (type, layer_id, temporal_id_plus1)"""
    byte0 = data[offset]
    byte1 = data[offset + 1]
    nal_type = (byte0 >> 1) & 0x3F
    layer_id = ((byte0 & 1) << 5) | (byte1 >> 3)
    temporal_id_plus1 = byte1 & 0x07
    return nal_type, layer_id, temporal_id_plus1


# Pre-compiled regex for start code scanning (C-accelerated)
_SC_PATTERN = re.compile(b'\x00\x00\x01')


def _scan_chunk(args):
    """Worker function for parallel start code scanning.
    Scans a chunk of a file for start codes using C-accelerated regex.
    Returns list of (file_offset, start_code_length)."""
    filepath, chunk_start, chunk_size, overlap_before = args
    positions = []
    with open(filepath, 'rb') as f:
        # Read with extra overlap to detect boundary start codes
        read_start = max(0, chunk_start - overlap_before)
        read_size = chunk_size + (chunk_start - read_start)
        f.seek(read_start)
        data = f.read(read_size)

    base_offset = read_start
    # Use C-accelerated regex to find all 00 00 01 sequences
    for m in _SC_PATTERN.finditer(data):
        pos = m.start()
        abs_pos = base_offset + pos
        # Only include positions that belong to this chunk
        if abs_pos < chunk_start and chunk_start > 0:
            continue
        # Check for 4-byte start code (00 00 00 01)
        if pos > 0 and data[pos - 1] == 0:
            positions.append((abs_pos - 1, 4))
        else:
            positions.append((abs_pos, 3))
    return positions


class StreamingNALParser:
    """Parse NAL units from an HEVC file using file-offset-based approach.
    Phase 1: Parallel scan using all CPU cores to find start code offsets.
    Phase 2: Read each NAL unit's data directly from the file by offset.
    """

    def __init__(self, filepath: str, buf_size: int = 32 * 1024 * 1024):
        self.filepath = filepath
        self.file_size = os.path.getsize(filepath)
        self.f = open(filepath, 'rb')
        self.buf_size = buf_size
        self._sc_positions = None  # list of (file_offset, sc_length)

    def _scan_start_codes(self):
        """Scan the entire file for start codes using parallel workers."""
        import time
        t0 = time.perf_counter()
        num_workers = max(1, multiprocessing.cpu_count())
        file_size = self.file_size
        print(f"  Scanning {file_size / 1024 / 1024:.1f} MB with {num_workers} CPU cores...", end='', flush=True)

        if file_size < 4 * 1024 * 1024 or num_workers == 1:
            # Small file: single-threaded C-accelerated scan
            self._scan_start_codes_single()
            elapsed = time.perf_counter() - t0
            speed = file_size / 1024 / 1024 / elapsed if elapsed > 0 else 0
            print(f" {len(self._sc_positions)} NALs found in {elapsed:.2f}s ({speed:.0f} MB/s)")
            return

        # Split file into chunks for parallel processing
        chunk_size = max(1024 * 1024, file_size // num_workers)
        overlap = 4  # Overlap to catch boundary start codes
        tasks = []
        offset = 0
        while offset < file_size:
            sz = min(chunk_size, file_size - offset)
            tasks.append((self.filepath, offset, sz, overlap))
            offset += sz

        # Parallel scan
        with multiprocessing.Pool(num_workers) as pool:
            results = pool.map(_scan_chunk, tasks)

        # Merge and deduplicate (boundary start codes may appear in multiple chunks)
        all_positions = []
        for chunk_positions in results:
            all_positions.extend(chunk_positions)

        # Sort by file offset and deduplicate
        all_positions.sort(key=lambda x: x[0])
        if all_positions:
            deduped = [all_positions[0]]
            for i in range(1, len(all_positions)):
                if all_positions[i][0] != deduped[-1][0]:
                    deduped.append(all_positions[i])
            self._sc_positions = deduped
        else:
            self._sc_positions = []
        elapsed = time.perf_counter() - t0
        speed = file_size / 1024 / 1024 / elapsed if elapsed > 0 else 0
        print(f" {len(self._sc_positions)} NALs found in {elapsed:.2f}s ({speed:.0f} MB/s)")

    def _scan_start_codes_single(self):
        """Single-threaded C-accelerated start code scan."""
        self.f.seek(0)
        positions = []
        file_offset = 0
        overlap = b''

        while True:
            chunk = self.f.read(self.buf_size)
            if not chunk:
                break
            data = overlap + chunk
            base_offset = file_offset - len(overlap)

            for m in _SC_PATTERN.finditer(data):
                pos = m.start()
                abs_pos = base_offset + pos
                if pos > 0 and data[pos - 1] == 0:
                    positions.append((abs_pos - 1, 4))
                else:
                    positions.append((abs_pos, 3))

            overlap = data[-3:] if len(data) >= 3 else data
            file_offset += len(chunk)

        # Deduplicate (4-byte SC at pos X might also match 3-byte at pos X+1)
        if positions:
            deduped = [positions[0]]
            for i in range(1, len(positions)):
                prev_off, prev_len = deduped[-1]
                cur_off, cur_len = positions[i]
                # Skip if current is inside previous 4-byte start code
                if cur_off <= prev_off + prev_len - 1 and cur_off > prev_off:
                    continue
                deduped.append(positions[i])
            positions = deduped

        self._sc_positions = positions

    def iter_nal_units(self):
        """Yield NALUnit objects one by one"""
        if self._sc_positions is None:
            self._scan_start_codes()

        positions = self._sc_positions
        for idx in range(len(positions)):
            sc_offset, sc_len = positions[idx]
            # NAL header starts right after start code
            hdr_offset = sc_offset + sc_len

            # NAL data ends at next start code or EOF
            if idx + 1 < len(positions):
                end_offset = positions[idx + 1][0]
            else:
                end_offset = self.file_size

            # Read NAL header (2 bytes) + RBSP data
            data_len = end_offset - hdr_offset
            if data_len < 2:
                continue

            self.f.seek(hdr_offset)
            raw = self.f.read(data_len)

            nal_type = (raw[0] >> 1) & 0x3F
            layer_id = ((raw[0] & 1) << 5) | (raw[1] >> 3)
            tid_plus1 = raw[1] & 0x07

            rbsp = raw[2:]  # data after 2-byte NAL header

            yield NALUnit(nal_type, layer_id, tid_plus1, bytes(rbsp), sc_len)

    def close(self):
        self.f.close()


# ============================================================
# SPS Parser & Rebuilder
# ============================================================
class SPSInfo:
    """Parsed SPS parameters needed for reconstruction"""
    def __init__(self):
        self.vps_id = 0
        self.max_sub_layers_minus1 = 0
        self.temporal_nesting = 0
        # profile_tier_level
        self.profile_space = 0
        self.tier_flag = 0
        self.profile_idc = 0
        self.profile_compat_flags = 0  # 32 bits
        self.progressive_source = 0
        self.interlaced_source = 0
        self.non_packed = 0
        self.frame_only = 0
        self.constraint_reserved_44 = 0  # 44 bits
        self.level_idc = 0
        # coding params
        self.sps_id = 0
        self.chroma_format_idc = 0
        self.separate_colour_plane = 0
        self.width = 0
        self.height = 0
        self.conformance_window = False
        self.conf_offsets = (0, 0, 0, 0)
        self.bit_depth_luma_minus8 = 0
        self.bit_depth_chroma_minus8 = 0
        self.log2_max_poc_lsb_minus4 = 0
        self.sub_layer_ordering_present = 0
        self.dpb_params = []  # list of (max_dec_pic_buf, max_reorder, max_latency)
        self.log2_min_cb_minus3 = 0
        self.log2_diff_max_min_cb = 0
        self.log2_min_tb_minus2 = 0
        self.log2_diff_max_min_tb = 0
        self.max_transform_depth_inter = 0
        self.max_transform_depth_intra = 0
        self.scaling_list_enabled = 0
        self.sps_scaling_list_data_present = 0
        self.scaling_list_raw_bits = None  # raw bit data if present
        self.amp_enabled = 0
        self.sao_enabled = 0
        self.pcm_enabled = 0
        self.pcm_raw_bits = None  # raw bit data if present
        self.num_short_term_rps = 0
        self.short_term_rps_raw_bits = None  # raw bit data for all ST-RPS
        self.long_term_present = 0
        self.long_term_raw_bits = None  # raw bit data
        self.temporal_mvp_enabled = 0
        self.strong_intra_smoothing = 0
        self.vui_present = 0
        self.vui_raw_bits = None  # raw bit data for VUI
        self.sps_extension_flag = 0
        # full raw RBSP for fallback
        self.raw_rbsp = None


def parse_sps_info(rbsp_with_ep: bytes) -> SPSInfo:
    """Parse SPS RBSP (with emulation prevention already in place from NAL)
    We work on clean RBSP (ep removed)."""
    clean = remove_emulation_prevention(rbsp_with_ep)
    bs = BitstreamReader(clean)
    sps = SPSInfo()
    sps.raw_rbsp = rbsp_with_ep

    sps.vps_id = bs.get_bits(4)
    sps.max_sub_layers_minus1 = msl = bs.get_bits(3)
    sps.temporal_nesting = bs.get_bits(1)
    # PTL
    sps.profile_space = bs.get_bits(2)
    sps.tier_flag = bs.get_bits(1)
    sps.profile_idc = bs.get_bits(5)
    sps.profile_compat_flags = bs.get_bits(32)
    sps.progressive_source = bs.get_bits(1)
    sps.interlaced_source = bs.get_bits(1)
    sps.non_packed = bs.get_bits(1)
    sps.frame_only = bs.get_bits(1)
    sps.constraint_reserved_44 = bs.get_bits(44)
    sps.level_idc = bs.get_bits(8)

    # sub-layer ptl flags
    sub_layer_profile_present = []
    sub_layer_level_present = []
    for i in range(msl):
        sub_layer_profile_present.append(bs.get_bits(1))
        sub_layer_level_present.append(bs.get_bits(1))
    if msl > 0:
        for i in range(msl, 8):
            bs.get_bits(2)
    # sub-layer PTL data
    for i in range(msl):
        if sub_layer_profile_present[i]:
            bs.get_bits(2 + 1 + 5 + 32 + 4 + 44)
        if sub_layer_level_present[i]:
            bs.get_bits(8)

    sps.sps_id = bs.get_ue()
    sps.chroma_format_idc = bs.get_ue()
    if sps.chroma_format_idc == 3:
        sps.separate_colour_plane = bs.get_bits(1)
    sps.width = bs.get_ue()
    sps.height = bs.get_ue()
    sps.conformance_window = bs.get_bits(1) == 1
    if sps.conformance_window:
        l = bs.get_ue(); r = bs.get_ue(); t = bs.get_ue(); b_ = bs.get_ue()
        sps.conf_offsets = (l, r, t, b_)
    sps.bit_depth_luma_minus8 = bs.get_ue()
    sps.bit_depth_chroma_minus8 = bs.get_ue()
    sps.log2_max_poc_lsb_minus4 = bs.get_ue()

    # Record bit position for the rest (we'll copy raw from here)
    sps._remaining_bit_pos = bs.bit_pos
    sps._clean_data = clean

    return sps


def _set_bit(data: bytearray, bit_pos: int, val: int):
    """Set a single bit in bytearray at the given bit position (MSB-first)."""
    byte_idx = bit_pos >> 3
    bit_idx = 7 - (bit_pos & 7)
    if val:
        data[byte_idx] |= (1 << bit_idx)
    else:
        data[byte_idx] &= ~(1 << bit_idx)


def _set_bits(data: bytearray, bit_pos: int, n: int, val: int):
    """Set n bits in bytearray starting at bit_pos (MSB-first)."""
    for i in range(n):
        _set_bit(data, bit_pos + i, (val >> (n - 1 - i)) & 1)


def _find_sps_vui_bit_pos(clean: bytes, msl: int) -> dict:
    """Parse SPS clean RBSP to find key bit positions and VUI state.
    Returns dict with bit positions and VUI info."""
    bs = BitstreamReader(clean)
    info = {}

    # Fixed header
    bs.get_bits(4)  # vps_id
    bs.get_bits(3)  # max_sub_layers_minus1
    bs.get_bits(1)  # temporal_nesting

    # PTL: tier at bit 10, level at bits 96-103 (always fixed for general PTL)
    info['tier_bit'] = bs.bit_pos + 2  # profile_space(2) then tier
    bs.get_bits(2 + 1 + 5 + 32 + 4 + 44 + 8)  # general PTL

    # Sub-layer PTL
    if msl > 0:
        for _ in range(msl): bs.get_bits(2)
        for _ in range(msl, 8): bs.get_bits(2)
    for _ in range(msl):
        # Can't easily know sub_layer flags without reading them
        pass
    # Actually need to parse sub-layer flags properly
    # Re-parse from scratch for accuracy
    bs2 = BitstreamReader(clean)
    bs2.get_bits(4 + 3 + 1)  # header
    bs2.get_bits(2 + 1 + 5 + 32 + 4 + 44 + 8)  # general PTL
    sub_prof = []
    sub_lev = []
    for i in range(msl):
        sub_prof.append(bs2.get_bits(1))
        sub_lev.append(bs2.get_bits(1))
    if msl > 0:
        for i in range(msl, 8):
            bs2.get_bits(2)
    for i in range(msl):
        if sub_prof[i]:
            bs2.get_bits(2 + 1 + 5 + 32 + 4 + 44)
        if sub_lev[i]:
            bs2.get_bits(8)

    bs2.get_ue()  # sps_id
    cf = bs2.get_ue()  # chroma_format_idc
    if cf == 3: bs2.get_bits(1)
    bs2.get_ue(); bs2.get_ue()  # width, height
    cw = bs2.get_bits(1)
    if cw:
        for _ in range(4): bs2.get_ue()
    bs2.get_ue(); bs2.get_ue()  # bit depths
    log2_max_poc_m4 = bs2.get_ue()
    info['log2_max_poc_lsb'] = log2_max_poc_m4 + 4

    slo = bs2.get_bits(1)
    start = 0 if slo else msl
    for _ in range(start, msl + 1):
        bs2.get_ue(); bs2.get_ue(); bs2.get_ue()

    for _ in range(6): bs2.get_ue()  # coding params

    sle = bs2.get_bits(1)
    if sle:
        slp = bs2.get_bits(1)
        if slp:
            # Parse scaling list data
            for size_id in range(4):
                step = 3 if size_id == 3 else 1
                for matrix_id in range(0, 6, step):
                    pred = bs2.get_bits(1)
                    if not pred:
                        bs2.get_ue()
                    else:
                        nc = min(64, 1 << (4 + (size_id << 1)))
                        if size_id > 1:
                            bs2.get_se()
                        for _ in range(nc):
                            bs2.get_se()

    bs2.get_bits(1)  # amp
    bs2.get_bits(1)  # sao
    pcm = bs2.get_bits(1)
    if pcm:
        bs2.get_bits(4 + 4)
        bs2.get_ue(); bs2.get_ue()
        bs2.get_bits(1)

    nstrps = bs2.get_ue()
    # Parse all short-term RPS (need to track num_delta_pocs)
    rps_ndp = []
    for idx in range(nstrps):
        inter = 0
        if idx > 0: inter = bs2.get_bits(1)
        if inter:
            if idx == nstrps:  # slice only
                bs2.get_ue()
            ds = bs2.get_bits(1)
            adrm1 = bs2.get_ue()
            ref_idx = idx - 1
            ref_num = rps_ndp[ref_idx]
            ndp = 0
            for j in range(ref_num + 1):
                uf = bs2.get_bits(1)
                ud = 0
                if not uf: ud = bs2.get_bits(1)
                if uf or ud: ndp += 1
            rps_ndp.append(ndp)
        else:
            nn = bs2.get_ue(); np_ = bs2.get_ue()
            for _ in range(nn): bs2.get_ue(); bs2.get_bits(1)
            for _ in range(np_): bs2.get_ue(); bs2.get_bits(1)
            rps_ndp.append(nn + np_)

    lt_present = bs2.get_bits(1)
    if lt_present:
        nlt = bs2.get_ue()
        for _ in range(nlt):
            bs2.get_bits(info['log2_max_poc_lsb'])
            bs2.get_bits(1)

    bs2.get_bits(1)  # temporal_mvp
    bs2.get_bits(1)  # strong_intra_smoothing

    info['vui_flag_bit_pos'] = bs2.bit_pos
    vui_present = bs2.get_bits(1)
    info['vui_present'] = vui_present

    if vui_present:
        # Parse VUI to check what's there
        ari = bs2.get_bits(1)
        if ari:
            idc = bs2.get_bits(8)
            if idc == 255: bs2.get_bits(32)
        oi = bs2.get_bits(1)
        if oi: bs2.get_bits(1)
        vst = bs2.get_bits(1)
        if vst:
            bs2.get_bits(3 + 1)
            cdp = bs2.get_bits(1)
            if cdp: bs2.get_bits(24)
            info['chroma_loc_info_bit_pos'] = bs2.bit_pos
            cli = bs2.get_bits(1)
            info['chroma_loc_info_present'] = cli
            if cli: bs2.get_ue(); bs2.get_ue()
        else:
            info['chroma_loc_info_present'] = 0
            info['chroma_loc_info_bit_pos'] = -1
        bs2.get_bits(1)  # neutral_chroma
        bs2.get_bits(1)  # field_seq
        info['ffipf_bit_pos'] = bs2.bit_pos
        ffipf = bs2.get_bits(1)
        info['frame_field_info_present'] = ffipf
        ddw = bs2.get_bits(1)
        if ddw:
            for _ in range(4): bs2.get_ue()
        timing = bs2.get_bits(1)
        info['timing_present'] = timing
        if timing:
            info['nuit'] = bs2.get_bits(32)
            info['time_scale'] = bs2.get_bits(32)
            pp = bs2.get_bits(1)
            if pp: bs2.get_ue()
            hrd = bs2.get_bits(1)
            info['hrd_present'] = hrd
        else:
            info['hrd_present'] = False
    else:
        info['timing_present'] = False
        info['hrd_present'] = False
        info['frame_field_info_present'] = False

    return info


def rebuild_sps(sps: SPSInfo, new_tier: int, new_level: int,
                vui_timing: Optional[dict] = None) -> bytes:
    """Modify SPS: patch tier/level, optionally splice VUI.
    Strategy: binary-patch where possible, splice only if VUI must be added.
    Preserves all original coding parameters exactly."""
    clean = bytearray(remove_emulation_prevention(sps.raw_rbsp))
    msl = sps.max_sub_layers_minus1

    # Find key bit positions
    info = _find_sps_vui_bit_pos(bytes(clean), msl)

    # Patch tier_flag (always at bit 10 in SPS RBSP)
    _set_bit(clean, 10, new_tier)

    # Patch level_idc (always at bits 96-103 in SPS RBSP)
    _set_bits(clean, 96, 8, new_level)

    has_good_vui = (info['vui_present'] and info['timing_present']
                    and info['hrd_present'])

    if has_good_vui:
        # Keep original VUI unchanged to avoid parsing errors
        return add_emulation_prevention(bytes(clean))


def _write_fresh_vui(w, vui_timing, max_sub_layers_minus1):
    """Write a complete VUI from scratch for Blu-ray compliance"""
    w.put_bits(1, 1)  # aspect_ratio_info_present
    w.put_bits(8, 1)  # aspect_ratio_idc = 1 (1:1)

    w.put_bits(1, 0)  # overscan_info_present = 0

    w.put_bits(1, 1)  # video_signal_type_present
    w.put_bits(3, 5)  # video_format = 5 (unspecified)
    w.put_bits(1, 0)  # video_full_range = 0 (limited)
    w.put_bits(1, 1)  # colour_description_present
    w.put_bits(8, vui_timing.get('colour_primaries', 9))
    w.put_bits(8, vui_timing.get('transfer_characteristics', 16))
    w.put_bits(8, vui_timing.get('matrix_coeffs', 9))

    # For BT.2020 (HDR), BD requires chroma_loc_info_present_flag = 1 with chroma_sample_loc_type = 2
    # For BT.709 (SDR), it can be 0
    colour_primaries = vui_timing.get('colour_primaries', 9)
    if colour_primaries == 9:  # BT.2020
        w.put_bits(1, 1)  # chroma_loc_info_present = 1 (required for BT.2020 HDR)
        w.put_ue(2)  # chroma_sample_loc_type_top_field = 2
        w.put_ue(2)  # chroma_sample_loc_type_bottom_field = 2
    else:
        w.put_bits(1, 0)  # chroma_loc_info_present = 0 (for BT.709 SDR)

    w.put_bits(1, 0)  # neutral_chroma_indication = 0
    w.put_bits(1, 0)  # field_seq_flag = 0
    w.put_bits(1, 1)  # frame_field_info_present_flag = 1

    w.put_bits(1, 0)  # default_display_window = 0

    w.put_bits(1, 1)  # vui_timing_info_present
    w.put_bits(32, vui_timing.get('num_units_in_tick', 1001))
    w.put_bits(32, vui_timing.get('time_scale', 24000))
    w.put_bits(1, 0)  # poc_proportional_to_timing_flag = 0

    w.put_bits(1, 1)  # hrd_parameters_present
    _write_hrd_parameters(w, vui_timing, max_sub_layers_minus1)

    w.put_bits(1, 1)  # bitstream_restriction
    w.put_bits(1, 0)  # tiles_fixed_structure = 0
    w.put_bits(1, 1)  # motion_vectors_over_pic_boundaries = 1
    w.put_bits(1, 0)  # restricted_ref_pic_lists = 0
    w.put_ue(0)       # min_spatial_segmentation_idc
    w.put_ue(2)       # max_bytes_per_pic_denom
    w.put_ue(1)       # max_bits_per_min_cu_denom
    w.put_ue(15)      # log2_max_mv_length_horizontal
    w.put_ue(15)      # log2_max_mv_length_vertical


def _write_hrd_parameters(w, vui_timing, max_sub_layers_minus1):
    """Write HRD parameters for Blu-ray compliance"""
    w.put_bits(1, 1)  # nal_hrd_parameters_present_flag = 1
    w.put_bits(1, 0)  # vcl_hrd_parameters_present_flag = 0
    w.put_bits(1, 0)  # sub_pic_hrd_params_present_flag = 0
    w.put_bits(4, 2)  # bit_rate_scale = 2
    w.put_bits(4, 2)  # cpb_size_scale = 2
    w.put_bits(5, 23)  # initial_cpb_removal_delay_length_minus1 = 23
    w.put_bits(5, 23)  # au_cpb_removal_delay_length_minus1 = 23
    w.put_bits(5, 23)  # dpb_output_delay_length_minus1 = 23

    for i in range(max_sub_layers_minus1 + 1):
        w.put_bits(1, 0)  # fixed_pic_rate_general_flag = 0
        w.put_bits(1, 1)  # fixed_pic_rate_within_cvs_flag = 1
        w.put_ue(0)  # elemental_duration_tc_minus1 = 0
        w.put_ue(0)  # cpb_cnt_minus1 = 0 (1 CPB entry)
        w.put_ue(78125)  # bit_rate_value_minus1
        w.put_ue(97656)  # cpb_size_value_minus1
        w.put_bits(1, 0)  # cbr_flag = 0


# ============================================================
# VPS Modification
# ============================================================
def modify_vps(rbsp_with_ep: bytes, new_tier: int, new_level: int,
               vps_timing: Optional[dict] = None) -> bytes:
    """Modify VPS tier_flag and level_idc directly in binary.
    Keeps original VPS structure intact."""
    clean = bytearray(remove_emulation_prevention(rbsp_with_ep))

    # VPS PTL layout:
    # PTL starts at bit 32:
    #   tier_flag at bit 34
    #   level_idc at bits 120-127 (byte 15)
    _set_bit(clean, 34, new_tier)
    _set_bits(clean, 120, 8, new_level)

    return add_emulation_prevention(bytes(clean))


# ============================================================
# SEI Message Builder
# ============================================================
def parse_sei_payloads(rbsp: bytes) -> List[Tuple[int, bytes]]:
    """Parse SEI NAL RBSP into list of (payload_type, payload_data)"""
    clean = remove_emulation_prevention(rbsp)
    payloads = []
    i = 0
    while i < len(clean):
        if clean[i] == 0x80:
            break
        # Read payload type
        pt = 0
        while i < len(clean) and clean[i] == 0xFF:
            pt += 255
            i += 1
        if i < len(clean):
            pt += clean[i]
            i += 1
        # Read payload size
        ps = 0
        while i < len(clean) and clean[i] == 0xFF:
            ps += 255
            i += 1
        if i < len(clean):
            ps += clean[i]
            i += 1
        # Read payload data
        if i + ps <= len(clean):
            payloads.append((pt, clean[i:i + ps]))
        i += ps
    return payloads


def build_sei_nal(payloads: List[Tuple[int, bytes]], prefix: bool = True) -> bytes:
    """Build a SEI NAL unit RBSP from list of (type, data) payloads"""
    result = bytearray()
    for pt, data in payloads:
        # Encode payload type
        t = pt
        while t >= 255:
            result.append(0xFF)
            t -= 255
        result.append(t)
        # Encode payload size
        s = len(data)
        while s >= 255:
            result.append(0xFF)
            s -= 255
        result.append(s)
        result.extend(data)
    result.append(0x80)  # trailing bits
    return add_emulation_prevention(bytes(result))


def build_active_parameter_sets_payload() -> bytes:
    """Build ActiveParameterSets SEI payload"""
    w = BitstreamWriter()
    w.put_bits(4, 0)  # active_video_parameter_set_id = 0
    w.put_bits(1, 0)  # self_contained_cvs_flag = 0
    w.put_bits(1, 0)  # no_parameter_set_update_flag = 0
    w.put_ue(0)       # num_sps_ids_minus1 = 0
    w.put_ue(0)       # active_seq_parameter_set_id = 0
    w.rbsp_trailing_bits()
    return w.flush()


# ============================================================
# Main Converter
# ============================================================
class HevcBlurayConverter:
    def __init__(self, args):
        self.input_path = args.input
        self.output_path = args.output
        self.target_tier = args.tier
        self.target_level = args.level
        self.target_fps_num = args.fps_num
        self.target_fps_den = args.fps_den
        self.verbose = args.verbose

        # State
        self.original_vps = None
        self.original_sps = None
        self.original_pps = None
        self.modified_vps = None
        self.modified_sps = None
        self.modified_pps = None
        self.sps_info = None
        self.sei_mastering = None
        self.sei_content_light = None
        self.sei_payloads_first_au = []  # Collected from first AU
        self.au_count = 0
        self.nal_count = 0

    def _make_vui_timing(self) -> dict:
        """Build VUI timing parameters based on target fps"""
        # For 23.976fps: num_units_in_tick=1001, time_scale=24000
        # For 24fps: num_units_in_tick=1, time_scale=24
        # For 25fps: num_units_in_tick=1, time_scale=25
        # For 29.97fps: num_units_in_tick=1001, time_scale=30000
        # For 59.94fps: num_units_in_tick=1001, time_scale=60000
        return {
            'num_units_in_tick': self.target_fps_den,
            'time_scale': self.target_fps_num,
            'bit_rate_scale': 2,
            'cpb_size_scale': 2,
            'initial_cpb_removal_delay_length_minus1': 23,
            'au_cpb_removal_delay_length_minus1': 23,
            'dpb_output_delay_length_minus1': 23,
            'elemental_duration_tc_minus1': 0,
            'bit_rate_value_minus1': 78125 - 1,  # ~80Mbps with scale=2
            'cpb_size_value_minus1': 97656 - 1,  # ~100MB with scale=2
            'colour_primaries': 9,     # BT.2020
            'transfer_characteristics': 16,  # SMPTE ST 2084 (PQ)
            'matrix_coeffs': 9,        # BT.2020 non-constant
            'chroma_sample_loc_top': 2,
            'chroma_sample_loc_bottom': 2,
        }

    def _detect_framerate(self):
        """Try to detect framerate from original VUI if present"""
        if self.sps_info:
            clean = remove_emulation_prevention(self.sps_info.raw_rbsp)
            # If SPS has VUI with timing info, we could extract it
            # For now, use defaults or user-specified values
            pass

    def _collect_hdr_sei(self, nal: NALUnit):
        """Extract HDR-related SEI payloads from SEI NAL"""
        payloads = parse_sei_payloads(nal.rbsp)
        for pt, data in payloads:
            if pt == SEI_MASTERING_DISPLAY:
                self.sei_mastering = data
                if self.verbose:
                    print(f"  Found Mastering Display SEI ({len(data)} bytes)")
            elif pt == SEI_CONTENT_LIGHT_LEVEL:
                self.sei_content_light = data
                if self.verbose:
                    print(f"  Found Content Light Level SEI ({len(data)} bytes)")

    def _build_prefix_sei_payloads(self, is_first_au: bool) -> List[Tuple[int, bytes]]:
        """Build SEI payloads for prefix SEI NAL"""
        payloads = []

        if is_first_au:
            # Active Parameter Sets
            payloads.append((SEI_ACTIVE_PARAM_SETS, build_active_parameter_sets_payload()))

        # Mastering Display (if available)
        if self.sei_mastering and is_first_au:
            payloads.append((SEI_MASTERING_DISPLAY, self.sei_mastering))

        # Content Light Level (if available)
        if self.sei_content_light and is_first_au:
            payloads.append((SEI_CONTENT_LIGHT_LEVEL, self.sei_content_light))

        return payloads

    def _make_aud_nal(self, pic_type: int = 2) -> bytes:
        """Build AUD NAL unit data (RBSP). pic_type: 0=I, 1=P/I, 2=B/P/I"""
        w = BitstreamWriter()
        w.put_bits(3, pic_type)
        w.rbsp_trailing_bits()
        return add_emulation_prevention(w.flush())

    def _write_nal(self, f, nal_type: int, layer_id: int, tid_plus1: int, rbsp: bytes):
        """Write a NAL unit with 4-byte start code"""
        hdr0 = (nal_type << 1) | (layer_id >> 5)
        hdr1 = ((layer_id & 0x1F) << 3) | (tid_plus1 & 0x7)
        f.write(FOUR_BYTE_SC)
        f.write(bytes([hdr0, hdr1]))
        f.write(rbsp)

    def convert(self):
        """Main conversion process"""
        print(f"Input:  {self.input_path}")
        print(f"Output: {self.output_path}")
        print(f"Target: Tier={'High' if self.target_tier else 'Main'}, "
              f"Level={self.target_level / 30:.1f}, "
              f"FPS={self.target_fps_num}/{self.target_fps_den}")

        parser = StreamingNALParser(self.input_path)
        vui_timing = self._make_vui_timing()

        # First pass: collect VPS/SPS/PPS and HDR SEI from first AU
        print("\nPhase 1: Scanning header NALs...")
        first_au_nals = []
        found_vcl = False
        second_au_start = False

        for nal in parser.iter_nal_units():
            if nal.nal_type == NAL_AUD and found_vcl:
                second_au_start = True
                break

            if nal.nal_type == NAL_VPS and self.original_vps is None:
                self.original_vps = nal
                if self.verbose:
                    print(f"  Found VPS ({len(nal.rbsp)} bytes RBSP)")
            elif nal.nal_type == NAL_SPS and self.original_sps is None:
                self.original_sps = nal
                self.sps_info = parse_sps_info(nal.rbsp)
                if self.verbose:
                    print(f"  Found SPS ({len(nal.rbsp)} bytes RBSP)")
                    print(f"    Profile={self.sps_info.profile_idc} Tier={self.sps_info.tier_flag} "
                          f"Level={self.sps_info.level_idc}")
                    print(f"    {self.sps_info.width}x{self.sps_info.height} "
                          f"chroma={self.sps_info.chroma_format_idc} "
                          f"bitdepth={self.sps_info.bit_depth_luma_minus8 + 8}/"
                          f"{self.sps_info.bit_depth_chroma_minus8 + 8}")
            elif nal.nal_type == NAL_PPS and self.original_pps is None:
                self.original_pps = nal
            elif nal.nal_type == NAL_SEI_PREFIX:
                self._collect_hdr_sei(nal)

            if nal.is_vcl():
                found_vcl = True

            first_au_nals.append(nal)

        parser.close()

        if not self.original_vps or not self.original_sps or not self.original_pps:
            print("ERROR: Could not find VPS/SPS/PPS in input file!")
            sys.exit(1)

        # Build modified VPS/SPS/PPS
        print("\nPhase 2: Building modified parameter sets...")
        self.modified_vps = modify_vps(
            self.original_vps.rbsp, self.target_tier, self.target_level, vui_timing)
        self.modified_sps = rebuild_sps(
            self.sps_info, self.target_tier, self.target_level, vui_timing)
        # PPS: keep original (just re-add emulation prevention for safety)
        self.modified_pps = self.original_pps.rbsp

        print(f"  VPS: {len(self.original_vps.rbsp)}B -> {len(self.modified_vps)}B")
        print(f"  SPS: {len(self.original_sps.rbsp)}B -> {len(self.modified_sps)}B")
        print(f"  PPS: {len(self.original_pps.rbsp)}B (unchanged)")

        # Second pass: write output
        print("\nPhase 3: Writing output...")
        parser2 = StreamingNALParser(self.input_path)

        with open(self.output_path, 'wb') as out:
            current_au_nals = []
            au_count = 0
            first_vcl_in_au = False
            total_nals = 0

            for nal in parser2.iter_nal_units():
                total_nals += 1

                # Detect AU boundary: AUD or first VCL after non-VCL
                is_au_boundary = False
                if nal.nal_type == NAL_AUD:
                    if current_au_nals:
                        self._flush_au(out, current_au_nals, au_count)
                        au_count += 1
                        current_au_nals = []
                    is_au_boundary = True
                    first_vcl_in_au = False
                elif nal.is_vcl() and not first_vcl_in_au:
                    # First VCL NAL - check if we need to start a new AU
                    if not current_au_nals and au_count == 0:
                        pass  # Will be handled
                    first_vcl_in_au = True

                current_au_nals.append(nal)

                if total_nals % 10000 == 0:
                    print(f"\r  Processed {total_nals} NALs, {au_count} AUs...", end='')

            # Flush last AU
            if current_au_nals:
                self._flush_au(out, current_au_nals, au_count)
                au_count += 1

            # Write EOS NAL
            self._write_nal(out, NAL_EOS_NUT, 0, 1, b'')

            self.au_count = au_count

        parser2.close()
        print(f"\r  Total: {total_nals} NALs, {au_count} AUs                    ")
        print(f"\nDone! Output: {self.output_path}")
        out_size = os.path.getsize(self.output_path)
        in_size = os.path.getsize(self.input_path)
        print(f"  Input size:  {in_size:,} bytes")
        print(f"  Output size: {out_size:,} bytes")
        print(f"  Difference:  {out_size - in_size:+,} bytes")

    def _flush_au(self, out, au_nals: List[NALUnit], au_index: int):
        """Write one Access Unit to output with proper Blu-ray structure"""
        # Classify NALs
        has_aud = False
        has_irap = False
        vcl_nals = []
        sei_prefix_nals = []
        sei_suffix_nals = []

        for nal in au_nals:
            if nal.nal_type == NAL_AUD:
                has_aud = True
            elif nal.nal_type in (NAL_VPS, NAL_SPS, NAL_PPS):
                pass  # We'll write our own
            elif nal.nal_type == NAL_SEI_PREFIX:
                sei_prefix_nals.append(nal)
            elif nal.nal_type == NAL_SEI_SUFFIX:
                sei_suffix_nals.append(nal)
            elif nal.nal_type in (NAL_EOS_NUT, NAL_EOB_NUT, NAL_FD_NUT):
                pass  # Skip
            elif nal.is_vcl():
                vcl_nals.append(nal)
                if nal.is_irap():
                    has_irap = True

        if not vcl_nals:
            return  # Skip empty AUs

        # Determine pic_type for AUD
        has_b = any(n.nal_type in (NAL_TRAIL_N, NAL_RADL_N, NAL_RASL_N, NAL_STSA_N,
                                    NAL_TSA_N) for n in vcl_nals)
        has_p = any(n.nal_type in (NAL_TRAIL_R, NAL_RADL_R, NAL_RASL_R, NAL_STSA_R,
                                    NAL_TSA_R) for n in vcl_nals)
        has_idr_cra = has_irap

        if has_b or has_p:
            pic_type = 2  # B (or mixed)
        elif has_p:
            pic_type = 1  # P
        else:
            pic_type = 0  # I only

        # Write AU structure:
        # 1. AUD
        aud_rbsp = self._make_aud_nal(pic_type)
        self._write_nal(out, NAL_AUD, 0, 1, aud_rbsp)

        # 2. VPS/SPS/PPS (always on IRAP, or first AU)
        if has_irap or au_index == 0:
            self._write_nal(out, NAL_VPS, 0, 1, self.modified_vps)
            self._write_nal(out, NAL_SPS, 0, 1, self.modified_sps)
            self._write_nal(out, NAL_PPS, 0, 1, self.modified_pps)

        # 3. SEI PREFIX - filter and rebuild
        sei_payloads = []
        for sei_nal in sei_prefix_nals:
            payloads = parse_sei_payloads(sei_nal.rbsp)
            for pt, data in payloads:
                # Keep: Mastering Display, Content Light Level, Buffering Period,
                #        Pic Timing, Active Param Sets, Recovery Point, Time Code
                # Remove: User Data Unregistered (x265 info), User Data Registered
                if pt in (SEI_MASTERING_DISPLAY, SEI_CONTENT_LIGHT_LEVEL,
                          SEI_BUFFERING_PERIOD, SEI_PIC_TIMING,
                          SEI_ACTIVE_PARAM_SETS, SEI_RECOVERY_POINT,
                          SEI_ALT_TRANSFER_CHAR, 136):  # 136 = time_code
                    sei_payloads.append((pt, data))

        if sei_payloads:
            sei_rbsp = build_sei_nal(sei_payloads)
            self._write_nal(out, NAL_SEI_PREFIX, 0, 1, sei_rbsp)

        # 4. VCL NALs (write with original data, just ensure 4-byte start code)
        for vcl in vcl_nals:
            self._write_nal(out, vcl.nal_type, vcl.layer_id,
                           vcl.temporal_id_plus1, vcl.rbsp)

        # 5. SEI SUFFIX (if any)
        for sei_nal in sei_suffix_nals:
            self._write_nal(out, sei_nal.nal_type, sei_nal.layer_id,
                           sei_nal.temporal_id_plus1, sei_nal.rbsp)


# ============================================================
# FFmpeg Re-encoder (HEVC → RAW YUV → HEVC, same bitrate)
# ============================================================
def _get_stream_info(input_path: str) -> dict:
    """Get video stream duration and bitrate info via ffprobe."""
    ffprobe_path = shutil.which('ffprobe')
    if not ffprobe_path:
        return {}
    try:
        proc = subprocess.run([
            ffprobe_path, '-v', 'quiet',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=duration,bit_rate,nb_frames,r_frame_rate',
            '-show_entries', 'format=duration,bit_rate',
            '-print_format', 'json',
            input_path
        ], capture_output=True, text=True, timeout=30)
        if proc.returncode == 0:
            import json
            data = json.loads(proc.stdout)
            info = {}
            streams = data.get('streams', [])
            fmt = data.get('format', {})
            if streams:
                s = streams[0]
                if s.get('duration'):
                    info['duration'] = float(s['duration'])
                if s.get('bit_rate'):
                    info['bitrate'] = int(s['bit_rate'])
                if s.get('nb_frames'):
                    info['nb_frames'] = int(s['nb_frames'])
                if s.get('r_frame_rate'):
                    info['r_frame_rate'] = s['r_frame_rate']
            if not info.get('duration') and fmt.get('duration'):
                info['duration'] = float(fmt['duration'])
            if not info.get('bitrate') and fmt.get('bit_rate'):
                info['bitrate'] = int(fmt['bit_rate'])
            return info
    except Exception:
        pass
    return {}


def reencode_with_ffmpeg(input_path: str, output_path: str, args) -> bool:
    """Re-encode HEVC → RAW YUV → HEVC with same bitrate and proper GOP.
    2-pass ABR encoding to match original file size.
    Returns True on success."""

    ffmpeg_path = shutil.which('ffmpeg')
    if not ffmpeg_path:
        print("ERROR: ffmpeg not found in PATH!")
        return False

    # Probe input for HDR metadata and stream info
    print("Probing input for stream info and HDR metadata...")
    hdr_meta = _probe_hdr_metadata(input_path)
    stream_info = _get_stream_info(input_path)

    preset = getattr(args, 'preset', 'slow')
    fps_num = args.fps_num
    fps_den = args.fps_den

    # Calculate target bitrate from original file
    in_size = os.path.getsize(input_path)
    duration = stream_info.get('duration', 0)
    if not duration and stream_info.get('nb_frames'):
        duration = stream_info['nb_frames'] * fps_den / fps_num
    if not duration:
        # Estimate from file: assume ~24fps, count AUs from our scanner
        print("  Warning: Could not determine duration, using file-based estimate")
        parser_tmp = StreamingNALParser(input_path)
        # Quick estimate: count NALs / rough ratio
        duration = in_size / (in_size / 63.5)  # fallback

    target_bitrate_kbps = int(in_size * 8 / duration / 1000)
    print(f"  Duration: {duration:.2f}s")
    print(f"  Original size: {in_size:,} bytes")
    print(f"  Target bitrate: {target_bitrate_kbps} kbps")

    # Calculate keyint from fps (1 second interval) or use custom
    keyint = getattr(args, 'keyint', 0)
    if not keyint:
        keyint = round(fps_num / fps_den)

    # Build x265-params (common for both passes)
    x265_base_params = [
        f"keyint={keyint}",
        "min-keyint=1",
        "scenecut=0",
        "open-gop=0",
        "repeat-headers=1",
        "hrd-opt=1",
        "aud=1",
        "annexb=1",
        "colorprim=bt2020",
        "transfer=smpte2084",
        "colormatrix=bt2020nc",
        "level-idc=5.1",
        "high-tier=1",
        f"vbv-maxrate={min(target_bitrate_kbps * 2, 160000)}",
        f"vbv-bufsize={min(target_bitrate_kbps * 2, 160000)}",
        f"bitrate={target_bitrate_kbps}",
    ]

    # Add HDR metadata if found
    if hdr_meta.get('master_display'):
        x265_base_params.append(f"master-display={hdr_meta['master_display']}")
    if hdr_meta.get('max_cll'):
        x265_base_params.append(f"max-cll={hdr_meta['max_cll']}")

    # Pass 1: Analysis
    pass1_params = x265_base_params + ["pass=1"]
    pass1_str = ":".join(pass1_params)

    # Stats file for 2-pass
    stats_file = output_path + ".x265_2pass.log"

    cmd_pass1 = [
        ffmpeg_path,
        '-i', input_path,
        '-map', '0:v:0',
        '-c:v', 'libx265',
        '-preset', preset,
        '-pix_fmt', 'yuv420p10le',
        '-x265-params', pass1_str,
        '-passlogfile', stats_file,
        '-f', 'hevc',
        '-y', 'NUL' if os.name == 'nt' else '/dev/null',
    ]

    print(f"\n{'='*60}")
    print(f"Pass 1/2: Analysis (preset={preset}, keyint={keyint})")
    print(f"{'='*60}")
    print(f"  Target: {target_bitrate_kbps} kbps (matching original)")
    if args.verbose:
        print(f"  Command: {' '.join(cmd_pass1)}")

    t0 = time.perf_counter()
    proc = subprocess.run(cmd_pass1, capture_output=False)
    elapsed1 = time.perf_counter() - t0

    if proc.returncode != 0:
        print(f"\nERROR: Pass 1 failed (exit code {proc.returncode})")
        return False
    print(f"  Pass 1 complete in {elapsed1:.1f}s")

    # Pass 2: Encode
    pass2_params = x265_base_params + ["pass=2"]
    pass2_str = ":".join(pass2_params)

    cmd_pass2 = [
        ffmpeg_path,
        '-i', input_path,
        '-map', '0:v:0',
        '-c:v', 'libx265',
        '-preset', preset,
        '-pix_fmt', 'yuv420p10le',
        '-x265-params', pass2_str,
        '-passlogfile', stats_file,
        '-f', 'hevc',
        '-y', output_path,
    ]

    print(f"\n{'='*60}")
    print(f"Pass 2/2: Encoding (target={target_bitrate_kbps} kbps)")
    print(f"{'='*60}")
    if args.verbose:
        print(f"  Command: {' '.join(cmd_pass2)}")

    t1 = time.perf_counter()
    proc = subprocess.run(cmd_pass2, capture_output=False)
    elapsed2 = time.perf_counter() - t1

    # Cleanup stats files
    for suffix in ['', '.cutree']:
        sf = stats_file + suffix
        if os.path.exists(sf):
            os.remove(sf)

    if proc.returncode != 0:
        print(f"\nERROR: Pass 2 failed (exit code {proc.returncode})")
        return False

    out_size = os.path.getsize(output_path)
    print(f"\n{'='*60}")
    print(f"Re-encode complete ({elapsed1 + elapsed2:.1f}s total)")
    print(f"{'='*60}")
    print(f"  Input:  {in_size:,} bytes")
    print(f"  Output: {out_size:,} bytes ({out_size/in_size*100:.1f}%)")
    print(f"  Bitrate match: {out_size*8/duration/1000:.0f} kbps "
          f"(target: {target_bitrate_kbps} kbps)")
    return True


def _probe_hdr_metadata(input_path: str) -> dict:
    """Use ffprobe to extract HDR metadata from input file."""
    result = {}
    ffprobe_path = shutil.which('ffprobe')
    if not ffprobe_path:
        return result

    try:
        proc = subprocess.run([
            ffprobe_path,
            '-v', 'quiet',
            '-select_streams', 'v:0',
            '-show_frames', '-read_intervals', '%+#1',
            '-print_format', 'json',
            input_path
        ], capture_output=True, text=True, timeout=30)

        if proc.returncode == 0:
            import json
            data = json.loads(proc.stdout)
            frames = data.get('frames', [])
            if frames:
                frame = frames[0]
                side_data_list = frame.get('side_data_list', [])
                for sd in side_data_list:
                    sd_type = sd.get('side_data_type', '')
                    if 'Mastering' in sd_type:
                        # Extract mastering display color volume
                        try:
                            r = sd.get('red_x', ''), sd.get('red_y', '')
                            g = sd.get('green_x', ''), sd.get('green_y', '')
                            b = sd.get('blue_x', ''), sd.get('blue_y', '')
                            wp = sd.get('white_point_x', ''), sd.get('white_point_y', '')
                            lmax = sd.get('max_luminance', '')
                            lmin = sd.get('min_luminance', '')
                            # Format: G(gx,gy)B(bx,by)R(rx,ry)WP(wpx,wpy)L(max,min)
                            # Values from ffprobe are in rational format "N/M"
                            def rat_to_int(s, scale=50000):
                                if '/' in str(s):
                                    n, d = str(s).split('/')
                                    return round(int(n) / int(d) * scale)
                                return int(s)
                            gx = rat_to_int(g[0]); gy = rat_to_int(g[1])
                            bx = rat_to_int(b[0]); by = rat_to_int(b[1])
                            rx = rat_to_int(r[0]); ry = rat_to_int(r[1])
                            wpx = rat_to_int(wp[0]); wpy = rat_to_int(wp[1])
                            max_l = rat_to_int(lmax, 10000)
                            min_l = rat_to_int(lmin, 10000)
                            result['master_display'] = (
                                f"G({gx},{gy})B({bx},{by})R({rx},{ry})"
                                f"WP({wpx},{wpy})L({max_l},{min_l})"
                            )
                            print(f"  Mastering Display: {result['master_display']}")
                        except Exception as e:
                            print(f"  Warning: Could not parse mastering display: {e}")

                    elif 'Content light' in sd_type:
                        max_cll = sd.get('max_content', 0)
                        max_fall = sd.get('max_average', 0)
                        result['max_cll'] = f"{max_cll},{max_fall}"
                        print(f"  Content Light Level: {result['max_cll']}")

    except Exception as e:
        print(f"  Warning: ffprobe failed: {e}")

    return result


# ============================================================
# Entry Point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='HEVC WEB-DL to UHD Blu-ray Structure Converter')
    parser.add_argument('input', help='Input HEVC file path')
    parser.add_argument('output', help='Output HEVC file path')
    parser.add_argument('--tier', type=int, default=1,
                        help='Target tier: 0=Main, 1=High (default: 1)')
    parser.add_argument('--level', type=int, default=153,
                        help='Target level_idc (default: 153 = Level 5.1)')
    parser.add_argument('--fps-num', type=int, default=24000,
                        help='Frame rate numerator (default: 24000)')
    parser.add_argument('--fps-den', type=int, default=1001,
                        help='Frame rate denominator (default: 1001)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--reencode', action='store_true',
                        help='Re-encode: HEVC→RAW→HEVC with same bitrate, proper GOP (fixes EPMAP)')
    parser.add_argument('--preset', default='slow',
                        help='x265 preset (ultrafast/fast/medium/slow/veryslow, default: slow)')
    parser.add_argument('--keyint', type=int, default=0,
                        help='Keyframe interval in frames (default: 1 second = fps_num/fps_den)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    if args.reencode:
        # Step 1: Re-encode with ffmpeg to get proper GOP structure
        print("=" * 60)
        print("Step 1: Re-encoding with ffmpeg/x265 (proper GOP)")
        print("=" * 60)

        # Use temp file for the intermediate re-encoded output
        temp_dir = os.path.dirname(os.path.abspath(args.output))
        temp_fd, temp_path = tempfile.mkstemp(suffix='.hevc', dir=temp_dir)
        os.close(temp_fd)

        try:
            if not reencode_with_ffmpeg(args.input, temp_path, args):
                sys.exit(1)

            # Step 2: Apply BD structure conversion to re-encoded file
            print("\n" + "=" * 60)
            print("Step 2: Applying Blu-ray structure conversion")
            print("=" * 60)
            args_copy = argparse.Namespace(**vars(args))
            args_copy.input = temp_path
            converter = HevcBlurayConverter(args_copy)
            converter.convert()
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                if args.verbose:
                    print(f"  Cleaned up temp file: {temp_path}")
    else:
        converter = HevcBlurayConverter(args)
        converter.convert()


if __name__ == '__main__':
    main()
