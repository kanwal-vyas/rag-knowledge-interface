"""
Lightweight WebM/EBML header inspector.

Parses WebM files to extract:
- DocType
- Duration (if present in SegmentInfo)
- Track codec IDs, sample rates, channels

Handles Chrome MediaRecorder's WebM structure where SegmentInfo and Tracks
may be nested directly inside the Segment element.

No external dependencies — pure struct-based parsing.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field


@dataclass
class WebMTrack:
    track_number: int = 0
    track_type: int = 0  # 1=video, 2=audio
    codec_id: str = ""
    sample_rate: float = 0.0
    channels: int = 0


@dataclass
class WebMInfo:
    valid: bool = False
    doctype: str = ""
    duration_ms: float | None = None
    timecode_scale: int | None = None
    total_bytes: int = 0
    first_bytes_hex: str = ""
    last_bytes_hex: str = ""
    tracks: list[WebMTrack] = field(default_factory=list)
    error: str | None = None


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


def _read_element(data: bytes, offset: int) -> tuple[int, int, int] | None:
    """Read an EBML element: (element_id, value_size, value_start_offset).

    Element ID preserves the on-disk value (with width marker).
    Size is the stripped numeric value.
    """
    id_result = _read_ebml_id(data, offset)
    if id_result is None:
        return None
    eid, eid_len = id_result
    size_offset = offset + eid_len
    size_result = _read_vint(data, size_offset)
    if size_result is None:
        return None
    esize, esize_len = size_result
    val_start = size_offset + esize_len
    return eid, esize, val_start


def _read_uint(data: bytes, offset: int, size: int) -> int | None:
    """Read an unsigned big-endian integer."""
    if offset + size > len(data):
        return None
    val = 0
    for i in range(size):
        val = (val << 8) | data[offset + i]
    return val


def _read_float(data: bytes, offset: int, size: int) -> float | None:
    """Read a big-endian float."""
    if offset + size > len(data):
        return None
    if size == 4:
        return struct.unpack(">f", data[offset:offset + 4])[0]
    if size == 8:
        return struct.unpack(">d", data[offset:offset + 8])[0]
    return None


def _read_string(data: bytes, offset: int, size: int) -> str:
    """Read a UTF-8 string."""
    if offset + size > len(data):
        return ""
    return data[offset:offset + size].decode("utf-8", errors="replace")


def _parse_segment_info(
    data: bytes, val_start: int, val_end: int,
) -> tuple[int | None, float | None]:
    """Parse SegmentInfo children. Returns (timecode_scale, duration_ms)."""
    timecode_scale: int | None = None
    duration_ns: float | None = None
    spos = val_start
    while spos < val_end:
        sub = _read_element(data, spos)
        if sub is None:
            break
        sid, ssz, sval = sub
        if sid == 0x2AD7B1:  # TimecodeScale
            timecode_scale = _read_uint(data, sval, ssz)
        elif sid == 0x4489:  # Duration
            duration_ns = _read_float(data, sval, ssz)
        spos = sval + ssz
    duration_ms: float | None = None
    if duration_ns is not None and timecode_scale and timecode_scale > 0:
        duration_ms = (duration_ns * timecode_scale) / 1_000_000
    return timecode_scale, duration_ms


def _parse_track_entry(data: bytes, tval: int, tsz: int) -> WebMTrack | None:
    """Parse a TrackEntry element."""
    track = WebMTrack()
    epos = tval
    while epos < tval + tsz:
        esub = _read_element(data, epos)
        if esub is None:
            break
        esid, essz, esval = esub
        if esid == 0xD7:  # TrackNumber
            val = _read_uint(data, esval, essz)
            track.track_number = val or 0
        elif esid == 0x83:  # TrackType
            track.track_type = _read_uint(data, esval, essz) or 0
        elif esid == 0x86:  # CodecID
            track.codec_id = _read_string(data, esval, essz)
        elif esid == 0xE1:  # Audio
            apos = esval
            while apos < esval + essz:
                asub = _read_element(data, apos)
                if asub is None:
                    break
                asid, asz, asval = asub
                if asid == 0xB5:  # SamplingFrequency
                    rate = _read_float(data, asval, asz)
                    track.sample_rate = rate or 0.0
                elif asid == 0x9F:  # Channels
                    ch = _read_uint(data, asval, asz)
                    track.channels = ch or 0
                apos = asval + asz
        epos = esval + essz
    return track if track.track_number else None


def _parse_tracks_element(data: bytes, val_start: int, val_end: int) -> list[WebMTrack]:
    """Parse a Tracks element, returning list of tracks."""
    tracks: list[WebMTrack] = []
    tpos = val_start
    while tpos < val_end:
        sub = _read_element(data, tpos)
        if sub is None:
            break
        tid, tsz, tval = sub
        if tid == 0xAE:  # TrackEntry
            track = _parse_track_entry(data, tval, tsz)
            if track:
                tracks.append(track)
        tpos = tval + tsz
    return tracks


def inspect_webm(audio_bytes: bytes) -> WebMInfo:
    """Inspect WebM header metadata without external dependencies.

    Handles two WebM structures:
    1. Standard: Segment → SegmentInfo + Tracks + Clusters
    2. Chrome MediaRecorder: Segment → Clusters (with SegmentInfo/Tracks
       nested inside or after Clusters, or absent entirely).
    """
    info = WebMInfo(total_bytes=len(audio_bytes))

    if len(audio_bytes) < 12:
        info.error = "File too small to be valid WebM"
        return info

    info.first_bytes_hex = audio_bytes[:16].hex()
    info.last_bytes_hex = audio_bytes[-16:].hex()

    # Check EBML magic number: 0x1A 0x45 0xDF 0xA3
    if audio_bytes[:4] != b"\x1a\x45\xdf\xa3":
        info.error = f"Not a valid EBML/WebM file (magic: {audio_bytes[:4].hex()})"
        return info

    try:
        pos = 0
        doc_found = False

        while pos < len(audio_bytes) - 4:
            result = _read_element(audio_bytes, pos)
            if result is None:
                break
            eid, esize, val_start = result
            val_end = val_start + esize

            if eid == 0x4282:  # DocType
                info.doctype = _read_string(audio_bytes, val_start, esize)
                doc_found = True

            elif eid == 0x18538067:  # Segment
                # Parse Segment children — find SegmentInfo and Tracks.
                # Chrome may put them directly inside Segment.
                spos = val_start
                while spos < val_end:
                    sub = _read_element(audio_bytes, spos)
                    if sub is None:
                        break
                    sid, ssz, sval = sub

                    if sid == 0x1549A966:  # SegmentInfo
                        info.timecode_scale, info.duration_ms = (
                            _parse_segment_info(audio_bytes, sval, sval + ssz)
                        )
                    elif sid == 0x1654AE6B:  # Tracks
                        info.tracks = _parse_tracks_element(
                            audio_bytes, sval, sval + ssz,
                        )
                    # Note: We do NOT recurse into Clusters — they can be very large.

                    spos = sval + ssz

            pos = val_end

        info.valid = doc_found

    except Exception as e:
        info.error = f"Parse error: {e}"

    return info
