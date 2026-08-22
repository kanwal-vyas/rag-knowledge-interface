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


def _read_vint(data: bytes, offset: int) -> tuple[int, int] | None:
    """Read an EBML variable-length integer. Returns (value, bytes_consumed)."""
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
    eid_result = _read_vint(data, offset)
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

    return ogg_bytes, diag
