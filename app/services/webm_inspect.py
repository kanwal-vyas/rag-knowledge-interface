"""
Lightweight WebM/EBML header inspector.

Parses only the first few hundred bytes of a WebM file to extract:
- DocType
- Duration (if present in SegmentInfo)
- Track codec IDs, sample rates, channels

No external dependencies — pure struct-based parsing.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field


@dataclass
class WebMTrack:
    track_number: int = 0
    track_type: int = 0  # 1=audio, 2=video
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


def _read_element_id(data: bytes, offset: int) -> tuple[int, int] | None:
    """Read an EBML element ID and size at offset. Returns (id, size) or None."""
    if offset >= len(data):
        return None
    b = data[offset]
    # Determine ID length from leading bits.
    if b & 0x80:
        eid = b & 0x7F
        size_start = offset + 1
    elif b & 0x40:
        eid = (b & 0x3F) << 8 | (data[offset + 1] if offset + 1 < len(data) else 0)
        size_start = offset + 2
    elif b & 0x20:
        eid = ((b & 0x1F) << 16
               | (data[offset + 1] << 8 if offset + 1 < len(data) else 0)
               | (data[offset + 2] if offset + 2 < len(data) else 0))
        size_start = offset + 3
    elif b & 0x10:
        eid = ((b & 0x0F) << 24
               | (data[offset + 1] << 16 if offset + 1 < len(data) else 0)
               | (data[offset + 2] << 8 if offset + 2 < len(data) else 0)
               | (data[offset + 3] if offset + 3 < len(data) else 0))
        size_start = offset + 4
    else:
        return None

    # Read variable-length size.
    if size_start >= len(data):
        return None
    sb = data[size_start]
    if sb & 0x80:
        esize = sb & 0x7F
        val_start = size_start + 1
    elif sb & 0x40:
        if size_start + 1 >= len(data):
            return None
        esize = ((sb & 0x3F) << 8) | data[size_start + 1]
        val_start = size_start + 2
    elif sb & 0x20:
        if size_start + 2 >= len(data):
            return None
        esize = ((sb & 0x1F) << 16) | (data[size_start + 1] << 8) | data[size_start + 2]
        val_start = size_start + 3
    elif sb & 0x10:
        if size_start + 3 >= len(data):
            return None
        esize = ((sb & 0x0F) << 24 | data[size_start + 1] << 16
                 | data[size_start + 2] << 8 | data[size_start + 3])
        val_start = size_start + 4
    else:
        return None

    return (eid, esize, val_start)  # type: ignore[return-value]


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


def inspect_webm(audio_bytes: bytes) -> WebMInfo:
    """Inspect WebM header metadata without external dependencies."""
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
        current_track: WebMTrack | None = None
        limit = min(len(audio_bytes), 4096)  # Only inspect first 4KB

        while pos < limit:
            result = _read_element_id(audio_bytes, pos)
            if result is None:
                break
            eid, esize, val_start = result
            val_end = val_start + esize

            if eid == 0x4282:  # DocType
                info.doctype = _read_string(audio_bytes, val_start, esize)
                doc_found = True

            elif eid == 0x1549A966:  # SegmentInfo
                # Parse children of SegmentInfo
                spos = val_start
                while spos < val_end:
                    sub = _read_element_id(audio_bytes, spos)
                    if sub is None:
                        break
                    sid, ssz, sval = sub
                    if sid == 0x2AD7B1:  # TimecodeScale
                        info.timecode_scale = _read_uint(audio_bytes, sval, ssz)
                    elif sid == 0x4489:  # Duration
                        info.duration_ms = _read_float(audio_bytes, sval, ssz)
                        if info.timecode_scale and info.timecode_scale > 0:
                            # Duration in ns → ms
                            info.duration_ms = (info.duration_ms * info.timecode_scale) / 1_000_000
                    spos = sval + ssz

            elif eid == 0x1654AE6B:  # Tracks
                # Parse track entries
                tpos = val_start
                while tpos < val_end:
                    sub = _read_element_id(audio_bytes, tpos)
                    if sub is None:
                        break
                    tid, tsz, tval = sub
                    if tid == 0xAE:  # TrackEntry
                        current_track = WebMTrack()
                        # Parse track entry children
                        epos = tval
                        while epos < tval + tsz:
                            esub = _read_element_id(audio_bytes, epos)
                            if esub is None:
                                break
                            esid, essz, esval = esub
                            if esid == 0xD7:  # TrackNumber
                                val = _read_uint(audio_bytes, esval, essz)
                                current_track.track_number = val or 0
                            elif esid == 0x83:  # TrackType
                                current_track.track_type = _read_uint(audio_bytes, esval, essz) or 0
                            elif esid == 0x86:  # CodecID
                                current_track.codec_id = _read_string(audio_bytes, esval, essz)
                            elif esid == 0xE1:  # Audio
                                # Parse audio sub-elements
                                apos = esval
                                while apos < esval + essz:
                                    asub = _read_element_id(audio_bytes, apos)
                                    if asub is None:
                                        break
                                    asid, asz, asval = asub
                                    if asid == 0xB5:  # SamplingFrequency
                                        rate = _read_float(audio_bytes, asval, asz)
                                        current_track.sample_rate = rate or 0.0
                                    elif asid == 0x9F:  # Channels
                                        ch = _read_uint(audio_bytes, asval, asz)
                                        current_track.channels = ch or 0
                                    apos = asval + asz
                            epos = esval + essz
                        if current_track.track_number:
                            info.tracks.append(current_track)
                    tpos = tval + tsz

            pos = val_end

        info.valid = doc_found

    except Exception as e:
        info.error = f"Parse error: {e}"

    return info
