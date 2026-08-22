"""
Extract Opus audio frames from a WebM/EBML container and repackage as OGG/Opus.

This is a diagnostic utility to test whether Sarvam's "Okay" response
is caused by the WebM container format vs the Opus audio content.

Chrome's MediaRecorder produces WebM/Opus that may use container-level
features (Clustering, TimecodeScale, SimpleBlock headers) that Sarvam's
WebM parser doesn't fully handle. By extracting the raw Opus packets and
wrapping them in an OGG/Opus container (which Opus natively supports),
we can isolate whether the problem is the container or the audio content.

No external dependencies — pure struct-based parsing.
"""
from __future__ import annotations

import io
import struct
from dataclasses import dataclass


@dataclass
class OpusPacket:
    """A single Opus packet extracted from a WebM SimpleBlock."""
    data: bytes
    timecode: int  # milliseconds relative to cluster


def _read_ebml_id(data: bytes, offset: int) -> tuple[int, int] | None:
    """Read an EBML Element ID, preserving the width-marker bits.

    EBML Element IDs use the VINT encoding where the leading zeros
    indicate the byte width, but unlike VINT *sizes*, the full on-disk
    value (including the width marker) is the canonical element ID.

    For example, the EBML Header ID ``0x1A45DFA3`` is stored as the
    4 bytes ``1A 45 DF A3``.  This function returns the full 4-byte
    value ``0x1A45DFA3``, NOT the stripped ``0x0A45DFA3``.

    Returns (element_id, bytes_consumed) or None.
    """
    if offset >= len(data):
        return None
    b = data[offset]
    for width in range(1, 9):
        mask = 1 << (8 - width)
        if b & mask:
            val = 0
            for i in range(width):
                if offset + i >= len(data):
                    return None
                val = (val << 8) | data[offset + i]
            return val, width
    return None


def _read_vint(data: bytes, offset: int) -> tuple[int, int] | None:
    """Read an EBML variable-length integer (sizes).

    Strips the width-marker bits, returning the numeric value.
    Used for element *sizes*, not element IDs.

    Returns (value, bytes_consumed) or None.
    """
    if offset >= len(data):
        return None
    b = data[offset]
    for width in range(1, 9):
        mask = 1 << (8 - width)
        if b & mask:
            val = b & (mask - 1)
            for i in range(1, width):
                if offset + i >= len(data):
                    return None
                val = (val << 8) | data[offset + i]
            return val, width
    return None


def _read_ebml_element(data: bytes, offset: int) -> tuple[int, int, int] | None:
    """Read an EBML element: (element_id, value_size, value_start_offset)."""
    eid_result = _read_ebml_id(data, offset)
    if eid_result is None:
        return None
    eid, eid_len = eid_result
    size_offset = offset + eid_len
    size_result = _read_vint(data, size_offset)
    if size_result is None:
        return None
    esize, esize_len = size_result
    val_start = size_offset + esize_len
    return eid, esize, val_start


def _parse_track_entry(data: bytes, offset: int, size: int) -> dict | None:
    """Parse a TrackEntry element, returning track info dict."""
    info: dict = {}
    pos = offset
    end = offset + size
    while pos < end:
        result = _read_ebml_element(data, pos)
        if result is None:
            break
        eid, esize, val_start = result
        if eid == 0xD7:  # TrackNumber
            r = _read_vint(data, val_start)
            if r:
                info["track_number"] = r[0]
        elif eid == 0x83:  # TrackType (1=video, 2=audio)
            r = _read_vint(data, val_start)
            if r:
                info["track_type"] = r[0]
        elif eid == 0x86:  # CodecID
            end_id = min(val_start + esize, len(data))
            info["codec_id"] = data[val_start:end_id].decode("utf-8", errors="replace")
        elif eid == 0xE1:  # Audio sub-element
            apos = val_start
            aend = val_start + esize
            while apos < aend:
                asub = _read_ebml_element(data, apos)
                if asub is None:
                    break
                asid, asz, asval = asub
                if asid == 0xB5:  # SamplingFrequency
                    end_f = min(asval + asz, len(data))
                    if asz == 4:
                        info["sample_rate"] = struct.unpack(">f", data[asval:end_f])[0]
                    elif asz == 8:
                        info["sample_rate"] = struct.unpack(">d", data[asval:end_f])[0]
                elif asid == 0x9F:  # Channels
                    r = _read_vint(data, asval)
                    if r:
                        info["channels"] = r[0]
                elif asid == 0x638A87:  # BitDepth
                    r = _read_vint(data, asval)
                    if r:
                        info["bit_depth"] = r[0]
                apos = asval + asz
        pos = val_start + esize
    return info if info else None


def extract_opus_packets(webm_bytes: bytes) -> list[OpusPacket]:
    """Extract all Opus packets from a WebM file by parsing EBML structure.

    Walks through Segment → Cluster → SimpleBlock, extracting Opus audio
    data from each block that belongs to the first audio track.

    Returns a list of OpusPacket objects with raw data and timecodes.
    """
    if len(webm_bytes) < 12:
        return []

    # Verify EBML magic number.
    if webm_bytes[:4] != b"\x1a\x45\xdf\xa3":
        return []

    packets: list[OpusPacket] = []
    audio_track_number: int | None = None

    # Parse top-level EBML elements to find Segment.
    pos = 0
    while pos < len(webm_bytes) - 4:
        result = _read_ebml_element(webm_bytes, pos)
        if result is None:
            break
        eid, esize, val_start = result

        if eid == 0x18538067:  # Segment
            seg_end = val_start + esize
            spos = val_start

            while spos < seg_end:
                sub = _read_ebml_element(webm_bytes, spos)
                if sub is None:
                    break
                sid, ssz, sval = sub

                if sid == 0x1654AE6B:  # Tracks
                    # First pass: find the audio track number.
                    tpos = sval
                    tend = sval + ssz
                    while tpos < tend:
                        tsub = _read_ebml_element(webm_bytes, tpos)
                        if tsub is None:
                            break
                        ttid, ttsz, ttval = tsub
                        if ttid == 0xAE:  # TrackEntry
                            tinfo = _parse_track_entry(webm_bytes, ttval, ttsz)
                            if tinfo and tinfo.get("track_type") == 2:  # Audio
                                audio_track_number = tinfo.get("track_number", 1)
                        tpos = ttval + ttsz

                elif sid == 0x1F43B675:  # Cluster
                    cpos = sval
                    cend = sval + ssz
                    cluster_timecode = 0

                    while cpos < cend:
                        csub = _read_ebml_element(webm_bytes, cpos)
                        if csub is None:
                            break
                        cid, csz, cval = csub

                        if cid == 0x67:  # ClusterTimecode
                            r = _read_vint(webm_bytes, cval)
                            if r:
                                cluster_timecode = r[0]
                        elif cid == 0xA3:  # SimpleBlock
                            # SimpleBlock: track_number + timecode(2B) + flags(1B) + data
                            if csz < 4:
                                cpos = cval + csz
                                continue

                            # Read track number from the block header.
                            tn_result = _read_vint(webm_bytes, cval)
                            if tn_result is None:
                                cpos = cval + csz
                                continue
                            tn, tn_len = tn_result

                            # If we haven't found audio track yet, assume track 1.
                            target_track = audio_track_number or 1

                            if tn == target_track:
                                # Read timecode (2 bytes big-endian, signed).
                                tc_offset = cval + tn_len
                                if tc_offset + 3 <= cval + csz:
                                    tc_slice = webm_bytes[tc_offset:tc_offset + 2]
                                    timecode_rel = struct.unpack(">h", tc_slice)[0]
                                    absolute_timecode = cluster_timecode + timecode_rel

                                    # Payload starts after flags byte.
                                    payload_start = tc_offset + 3
                                    payload_end = cval + csz
                                    if payload_start < payload_end:
                                        opus_data = webm_bytes[payload_start:payload_end]
                                        packets.append(OpusPacket(
                                            data=opus_data,
                                            timecode=absolute_timecode,
                                        ))

                        cpos = cval + csz

                spos = sval + ssz

        pos = val_start + esize

    return packets


def packets_to_ogg(opus_packets: list[OpusPacket], sample_rate: int = 48000) -> bytes:
    """Convert a list of Opus packets into an OGG/Opus file.

    This creates a minimal valid OGG/Opus stream:
    - OGG page 1: OpusHead header (channel count, sample rate)
    - OGG page 2: OpusTags (comment header)
    - OGG pages: one per Opus packet

    Args:
        opus_packets: List of OpusPacket objects.
        sample_rate: Sample rate in Hz (default 48000 for Opus).

    Returns:
        Complete OGG/Opus file as bytes.
    """
    if not opus_packets:
        return b""

    output = io.BytesIO()

    # --- OGG Page utilities ---
    def _make_page(
        header_type: int,
        granule_pos: int,
        serial: int,
        page_seq: int,
        segment_table: list[int],
        payload: bytes,
    ) -> bytes:
        """Build a single OGG page."""
        header = bytearray()
        # Capture pattern
        header.extend(b"OggS")
        # Header type
        header.append(header_type)
        # Granule position (8 bytes little-endian)
        header.extend(struct.pack("<Q", granule_pos))
        # Serial number
        header.extend(struct.pack("<I", serial))
        # Page sequence number
        header.extend(struct.pack("<I", page_seq))
        # Checksum placeholder
        header.extend(b"\x00\x00\x00\x00")
        # Number of segments
        header.append(len(segment_table))
        # Segment table
        header.extend(bytes(segment_table))

        page_data = bytes(header) + payload

        # Compute OGG checksum (Xiph CRC-32).
        checksum = _ogg_checksum(page_data)
        page_data = bytearray(page_data)
        page_data[22:26] = struct.pack("<I", checksum)
        return bytes(page_data)

    def _ogg_checksum(data: bytes) -> int:
        """Compute OGG Xiph CRC-32."""
        crc = 0
        lookup = _CRC_TABLE
        for byte in data:
            crc = (crc << 8) ^ lookup[((crc >> 24) & 0xFF) ^ byte]
            crc &= 0xFFFFFFFF
        return crc

    # Pre-compute CRC lookup table (Xiph/Ogg polynomial).
    if not hasattr(packets_to_ogg, "_CRC_TABLE"):
        crc_table = [0] * 256
        poly = 0x04C11DB7
        for i in range(256):
            c = i << 24
            for _ in range(8):
                if c & 0x80000000:  # noqa: SIM108
                    c = ((c << 1) ^ poly) & 0xFFFFFFFF
                else:
                    c = (c << 1) & 0xFFFFFFFF
            crc_table[i] = c
        packets_to_ogg._CRC_TABLE = crc_table  # type: ignore[attr-defined]

    _CRC_TABLE = packets_to_ogg._CRC_TABLE  # type: ignore[attr-defined]

    serial = 0x12345678  # Arbitrary serial number.
    page_seq = 0

    # --- OpusHead (channel count from first packet's TOC byte) ---
    # Opus TOC byte: config(5 bits) + stereo(1 bit) + frame_count_code(2 bits)
    toc = opus_packets[0].data[0] if opus_packets[0].data else 0
    channels = 2 if (toc & 0x04) else 1

    opus_head = bytearray()
    opus_head.extend(b"OpusHead")
    opus_head.append(1)  # Version
    opus_head.append(channels)
    opus_head.extend(struct.pack("<H", 0))  # Pre-skip
    opus_head.extend(struct.pack("<I", sample_rate))
    opus_head.extend(struct.pack("<h", 0))  # Output gain
    opus_head.append(0)  # Channel mapping family

    segment_table = [len(opus_head)]
    page1 = _make_page(0x02, 0, serial, page_seq, segment_table, bytes(opus_head))
    output.write(page1)
    page_seq += 1

    # --- OpusTags (comment header) ---
    opus_tags = bytearray()
    opus_tags.extend(b"OpusTags")
    # Vendor string
    vendor = b"extract-opus"
    opus_tags.extend(struct.pack("<I", len(vendor)))
    opus_tags.extend(vendor)
    opus_tags.extend(struct.pack("<I", 0))  # User comment count

    segment_table = [len(opus_tags)]
    page2 = _make_page(0x00, 0, serial, page_seq, segment_table, bytes(opus_tags))
    output.write(page2)
    page_seq += 1

    # --- Audio packets ---
    # Opus uses 48kHz clock. Each packet at48kHz represents 20ms of audio
    # by default (960 samples per frame).
    samples_per_frame = 960  # Default for 20ms frames
    granule = 0

    for pkt in opus_packets:
        # Build OGG segments for this packet.
        # OGG segments are max 255 bytes each.
        data = pkt.data
        seg_table = []
        while len(data) > 255:
            seg_table.append(255)
            data = data[255:]
        seg_table.append(len(data))

        granule += samples_per_frame
        page = _make_page(0x00, granule, serial, page_seq, seg_table, pkt.data)
        output.write(page)
        page_seq += 1

    # --- EOS page ---
    eos_page = _make_page(0x04, granule, serial, page_seq, [0], b"")
    output.write(eos_page)

    return bytes(output)


def webm_to_ogg_opus(webm_bytes: bytes) -> tuple[bytes, dict]:
    """Extract Opus from WebM and rewrap as OGG/Opus.

    Returns:
        (ogg_bytes, diagnostics_dict)
    """
    packets = extract_opus_packets(webm_bytes)

    diag = {
        "webm_bytes": len(webm_bytes),
        "opus_packets_found": len(packets),
        "packet_sizes": [len(p.data) for p in packets[:20]],  # First 20
        "total_opus_bytes": sum(len(p.data) for p in packets),
        "first_packet_hex": packets[0].data[:16].hex() if packets else "",
        "first_packet_toc": packets[0].data[0] if packets and packets[0].data else None,
        "timecodes_ms": [p.timecode for p in packets[:20]],
    }

    if not packets:
        diag["error"] = "No Opus packets extracted from WebM"
        return b"", diag

    ogg_bytes = packets_to_ogg(packets)
    diag["ogg_bytes"] = len(ogg_bytes)

    # Signal-level analysis.
    diag["signal_analysis"] = analyze_opus_packets(packets)

    return ogg_bytes, diag


# ---------------------------------------------------------------------------
# Opus TOC byte analysis and signal diagnostics
# ---------------------------------------------------------------------------

# Opus TOC byte layout:
#   Bits 7-3: configuration number
#   Bit 2: stereo (0=mono, 1=stereo)
#   Bits 1-0: frame count code (0=1 frame, 1=2 frames, 2=2 frames, 3=arbitrary)
#
# Configuration 0-3:   NB (narrowband, 8kHz)
# Configuration 4-7:   MB (medium-band, 12kHz)
# Configuration 8-11:  WB (wideband, 16kHz)
# Configuration 12-15: SWB (super-wideband, 24kHz)
# Configuration 16-19: FB (fullband, 48kHz)
# Configuration 20+:  special (CELT, hybrid, etc.)

_OPUS_BANDWIDTH_NAMES = {
    range(0, 4): "NB",
    range(4, 8): "MB",
    range(8, 12): "WB",
    range(12, 16): "SWB",
    range(16, 20): "FB",
    range(20, 32): "Special",
}


def _get_bandwidth_name(config: int) -> str:
    for rng, name in _OPUS_BANDWIDTH_NAMES.items():
        if config in rng:
            return name
    return "Unknown"


def _estimate_packet_duration_ms(toc: int) -> float:
    """Estimate the duration of a single Opus packet from its TOC byte."""
    frame_count_code = toc & 0x03
    # Default frame sizes for common configs.
    # Config 0-19: frame size is 2.5, 5, 10, 20, 40, or 60 ms depending on config.
    # For simplicity, use 20ms as the most common MediaRecorder frame size.
    frame_ms = 20.0
    if frame_count_code == 0:
        return frame_ms
    elif frame_count_code == 1 or frame_count_code == 2:
        return frame_ms * 2
    else:
        # Arbitrary frame count — cannot determine without parsing.
        return frame_ms


def analyze_opus_packets(packets: list[OpusPacket]) -> dict:
    """Analyze Opus packets for signal diagnostics.

    Reports TOC byte distribution, estimated duration, bandwidth,
    stereo/mono ratio, and packet size statistics.
    """
    if not packets:
        return {"error": "No packets to analyze"}

    toc_values: list[int] = []
    config_counts: dict[str, int] = {}
    stereo_count = 0
    mono_count = 0
    frame_count_codes: dict[int, int] = {}
    sizes = [len(p.data) for p in packets]
    total_duration_ms = 0.0
    timecode_span_ms: float | None = None

    for pkt in packets:
        if not pkt.data:
            continue
        toc = pkt.data[0]
        toc_values.append(toc)
        config = (toc >> 3) & 0x1F
        stereo = bool(toc & 0x04)
        frame_code = toc & 0x03

        bw = _get_bandwidth_name(config)
        config_counts[bw] = config_counts.get(bw, 0) + 1
        if stereo:
            stereo_count += 1
        else:
            mono_count += 1
        frame_count_codes[frame_code] = frame_count_codes.get(frame_code, 0) + 1
        total_duration_ms += _estimate_packet_duration_ms(toc)

    if len(packets) >= 2:
        first_tc = packets[0].timecode
        last_tc = packets[-1].timecode
        timecode_span_ms = float(last_tc - first_tc)

    first_sizes = sizes[:10]
    last_sizes = sizes[-5:]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    # Estimate speech energy from packet sizes.
    # Very small packets (< 10 bytes) typically indicate silence/DTX.
    silence_packets = sum(1 for s in sizes if s < 10)
    tiny_packets = sum(1 for s in sizes if s < 5)

    return {
        "packet_count": len(packets),
        "total_opus_bytes": sum(sizes),
        "avg_packet_bytes": round(avg_size, 1),
        "first_packet_sizes": first_sizes,
        "last_packet_sizes": last_sizes,
        "first_toc_hex": hex(toc_values[0]) if toc_values else None,
        "last_toc_hex": hex(toc_values[-1]) if toc_values else None,
        "stereo_packets": stereo_count,
        "mono_packets": mono_count,
        "bandwidth_distribution": config_counts,
        "frame_count_codes": frame_count_codes,
        "estimated_duration_ms": round(total_duration_ms, 1),
        "timecode_span_ms": round(timecode_span_ms, 1) if timecode_span_ms is not None else None,
        "silence_packets_lt10bytes": silence_packets,
        "tiny_packets_lt5bytes": tiny_packets,
        "silence_ratio": round(silence_packets / len(sizes), 3) if sizes else 0,
        "timecodes_first_10": [p.timecode for p in packets[:10]],
        "timecodes_last_5": [p.timecode for p in packets[-5:]],
        "first_packet_hex_32": (
            packets[0].data[:32].hex()
            if packets and len(packets[0].data) >= 32
            else (packets[0].data.hex() if packets else "")
        ),
    }
