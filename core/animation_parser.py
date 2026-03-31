"""PAA animation parser for Crimson Desert.

Parses .paa animation files to extract keyframe data.
Animation data is stored as int16 quaternion components per bone per frame.

PAA format (PAR v3.2):
  Header: 32 bytes
    [0x00] 4B: 'PAR ' magic
    [0x04] 4B: version (0x01000302)
    [0x08] 8B: hash
    [0x10] 4B: data_flags / frame info
    [0x14] 4B: duration (float, sometimes garbage)
    [0x18] 4B: additional flags
    [0x1C] 4B: padding

  Data (after header):
    Packed int16 values representing bone transforms.
    Each bone transform = 4 int16 (quaternion XYZW) = 8 bytes.
    Quaternion components: value / 32767.0 gives -1.0 to 1.0 range.

Note: PAA is a simplified format. Full skeletal animation with
proper timing/interpolation requires matching with a PAB skeleton
and the .paa_metabin companion file.
"""

from __future__ import annotations

import os
import struct
import math
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger

logger = get_logger("core.animation_parser")

PAR_MAGIC = b"PAR "


@dataclass
class AnimationKeyframe:
    """A single keyframe with per-bone quaternion rotations."""
    frame_index: int = 0
    bone_rotations: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class ParsedAnimation:
    """Parsed animation data."""
    path: str = ""
    duration: float = 0.0
    frame_count: int = 0
    bone_count: int = 0
    keyframes: list[AnimationKeyframe] = field(default_factory=list)
    raw_quaternions: list[tuple[float, float, float, float]] = field(default_factory=list)


def parse_paa(data: bytes, filename: str = "", expected_bone_count: int = 0) -> ParsedAnimation:
    """Parse a .paa animation file.

    Args:
        data: Raw file bytes.
        filename: File path for logging.
        expected_bone_count: If known (from PAB), helps determine frame count.

    Returns:
        ParsedAnimation with quaternion keyframe data.
    """
    if len(data) < 32 or data[:4] != PAR_MAGIC:
        raise ValueError(f"Not a valid PAA file: {data[:4]!r}")

    result = ParsedAnimation(path=filename)

    # Header
    version = struct.unpack_from("<I", data, 4)[0]
    flags_raw = struct.unpack_from("<I", data, 0x10)[0]

    # [0x10]: high bits are flags (0xC0000000), low bits are bone count
    header_bone_count = flags_raw & 0x0FFF  # mask out flag bits
    header_flags = (flags_raw >> 28) & 0xF

    # Duration: try different offsets — format varies
    duration_raw = struct.unpack_from("<f", data, 0x14)[0]
    if not (0.001 < duration_raw < 10000.0) or math.isnan(duration_raw):
        # Try 0x18 as float (sometimes duration is here)
        duration_raw = struct.unpack_from("<f", data, 0x18)[0]
    if 0.001 < duration_raw < 10000.0 and not math.isnan(duration_raw):
        result.duration = duration_raw

    # Animation data starts at offset 0x20
    data_start = 0x20
    data_size = len(data) - data_start

    if data_size < 8:
        return result

    # Each quaternion = 4 × int16 = 8 bytes
    total_quats = data_size // 8

    # Extract all quaternions
    quats = []
    for i in range(total_quats):
        off = data_start + i * 8
        if off + 8 > len(data):
            break
        x, y, z, w = struct.unpack_from("<hhhh", data, off)
        qx = x / 32767.0
        qy = y / 32767.0
        qz = z / 32767.0
        qw = w / 32767.0
        length = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if length > 0.001:
            qx /= length
            qy /= length
            qz /= length
            qw /= length
        quats.append((qx, qy, qz, qw))

    result.raw_quaternions = quats

    # Determine bone count: use header value, expected_bone_count, or heuristic
    if expected_bone_count > 0 and total_quats % expected_bone_count == 0:
        result.bone_count = expected_bone_count
    elif header_bone_count > 0 and total_quats % header_bone_count == 0:
        result.bone_count = header_bone_count
    elif header_bone_count > 0:
        # Header has bone count but doesn't divide evenly — still use it
        result.bone_count = header_bone_count
    else:
        # Heuristic fallback
        for bc in [1, 2, 4, 8, 15, 16, 32, 47, 64, 111, 128, 160, 169, 192, 218, 256]:
            if total_quats % bc == 0:
                frames = total_quats // bc
                if 1 <= frames <= 10000:
                    result.bone_count = bc
                    break

    if result.bone_count > 0:
        result.frame_count = max(1, total_quats // result.bone_count)

    # Build keyframes
    if result.bone_count > 0 and result.frame_count > 0:
        for f in range(result.frame_count):
            kf = AnimationKeyframe(frame_index=f)
            for b in range(result.bone_count):
                qi = f * result.bone_count + b
                if qi < len(quats):
                    kf.bone_rotations.append(quats[qi])
            result.keyframes.append(kf)

    logger.info("Parsed PAA %s: %d quats, %d bones, %d frames, %.2fs",
                filename, len(quats), result.bone_count, result.frame_count, result.duration)
    return result


def is_animation_file(path: str) -> bool:
    """Check if a file is an animation file."""
    return os.path.splitext(path.lower())[1] == ".paa"
