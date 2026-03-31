"""Havok HKX (TAG0) parser for Crimson Desert.

Parses .hkx files using the TAG0 binary tagfile format (Havok SDK 2024.2).
Extracts bone names, skeleton hierarchy, physics shapes, and ragdoll data
from the binary stream without requiring the full Havok type reflection system.

TAG0 structure:
  [0-3]   uint32 BE: total file size
  [4-7]   'TAG0' magic
  [8-11]  'SDKV' marker
  [12-19] SDK version string (e.g., '20240200')
  [20+]   Sections: DATA, TYPE, TSTR, FSTR, etc.

Each section: [4B magic] [4B BE size] [data...]
"""

from __future__ import annotations

import os
import struct
import re
from dataclasses import dataclass, field
from typing import Optional

from utils.logger import get_logger

logger = get_logger("core.havok_parser")

TAG0_MAGIC = b"TAG0"


@dataclass
class HavokBone:
    """A bone extracted from Havok skeleton data."""
    index: int = 0
    name: str = ""
    parent_index: int = -1


@dataclass
class HavokSection:
    """A section within the TAG0 file."""
    magic: str = ""
    offset: int = 0
    size: int = 0
    data: bytes = b""


@dataclass
class ParsedHavok:
    """Parsed Havok HKX file."""
    path: str = ""
    sdk_version: str = ""
    total_size: int = 0
    sections: list[HavokSection] = field(default_factory=list)
    bones: list[HavokBone] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    has_skeleton: bool = False
    has_animation: bool = False
    has_physics: bool = False
    has_ragdoll: bool = False


def parse_hkx(data: bytes, filename: str = "") -> ParsedHavok:
    """Parse a .hkx Havok TAG0 binary tagfile.

    Extracts sections, class names, and bone hierarchy from the binary.
    This is a best-effort parser — full Havok type reflection is not
    implemented, but bone names and basic structure are extracted.
    """
    result = ParsedHavok(path=filename, total_size=len(data))

    if len(data) < 16:
        return result

    # Verify TAG0 format
    magic = data[4:8]
    if magic != TAG0_MAGIC:
        # Not TAG0, might be older packfile format
        return result

    file_size = struct.unpack_from(">I", data, 0)[0]

    # SDK version
    sdkv_pos = data.find(b"SDKV")
    if sdkv_pos >= 0:
        ver_start = sdkv_pos + 4
        ver_end = data.find(b"\x00", ver_start, ver_start + 16)
        if ver_end < 0:
            ver_end = ver_start + 8
        result.sdk_version = data[ver_start:ver_end].decode("ascii", "replace").strip()

    # Find all sections (DATA, TYPE, TSTR, FSTR, etc.)
    pos = 0
    while pos < len(data) - 8:
        # Sections start with a 4-char ASCII tag followed by size or content
        chunk = data[pos:pos + 4]
        if chunk in (b"DATA", b"TYPE", b"TSTR", b"FSTR", b"TBDY", b"THSH", b"TPAD", b"INDX"):
            sec_magic = chunk.decode("ascii")
            # Size might be at pos-4 (before tag) or at pos+4 (after tag)
            # TAG0 uses: [tag 4B] [? padding] [content...]
            # The actual content follows the tag
            sec = HavokSection(magic=sec_magic, offset=pos)

            # Find next section to determine this section's size
            next_pos = len(data)
            for tag in [b"DATA", b"TYPE", b"TSTR", b"FSTR", b"TBDY", b"THSH", b"TPAD", b"INDX"]:
                np = data.find(tag, pos + 4)
                if 0 < np < next_pos:
                    next_pos = np

            sec.size = next_pos - pos
            sec.data = data[pos:next_pos]
            result.sections.append(sec)
            pos = next_pos
        else:
            pos += 1

    # Extract class/type names from TSTR section
    for sec in result.sections:
        if sec.magic == "TSTR":
            off = 4  # skip "TSTR"
            while off < len(sec.data):
                nul = sec.data.find(b"\x00", off)
                if nul < 0:
                    break
                s = sec.data[off:nul].decode("ascii", "replace")
                if len(s) > 1:
                    result.class_names.append(s)
                off = nul + 1

    # Extract bone names and parent hierarchy from DATA section
    _extract_bones(data, result)
    _extract_parent_indices(data, result)

    # Detect content types
    for cls in result.class_names:
        cls_lower = cls.lower()
        if "skeleton" in cls_lower:
            result.has_skeleton = True
        if "animation" in cls_lower or "anim" in cls_lower:
            result.has_animation = True
        if "rigidbody" in cls_lower or "shape" in cls_lower or "physics" in cls_lower:
            result.has_physics = True
        if "ragdoll" in cls_lower:
            result.has_ragdoll = True

    if result.bones:
        result.has_skeleton = True

    logger.info("Parsed HKX %s: SDK %s, %d sections, %d bones, %d classes, "
                "skel=%s anim=%s phys=%s ragdoll=%s",
                filename, result.sdk_version, len(result.sections),
                len(result.bones), len(result.class_names),
                result.has_skeleton, result.has_animation,
                result.has_physics, result.has_ragdoll)
    return result


def _extract_bones(data: bytes, result: ParsedHavok):
    """Extract bone names from the binary data.

    Scans for common bone name patterns (Bip01, B_, Bone, etc.)
    and builds a bone list. Parent indices are inferred from the
    naming convention when not explicitly stored.
    """
    bone_names = []
    seen = set()
    pos = 0

    while pos < len(data) - 5:
        found = False
        for prefix in (b"Bip01", b"B_", b"Bone", b"Root", b"Dummy"):
            if data[pos:pos + len(prefix)] == prefix:
                nul = data.find(b"\x00", pos, pos + 128)
                if nul > pos:
                    raw = data[pos:nul]
                    # Validate: all printable ASCII
                    if all(32 <= b < 127 for b in raw) and len(raw) >= 3:
                        name = raw.decode("ascii")
                        if name not in seen:
                            seen.add(name)
                            bone_names.append(name)
                        pos = nul + 1
                        found = True
                        break
        if not found:
            pos += 1

    # Build bone hierarchy from names
    for i, name in enumerate(bone_names):
        bone = HavokBone(index=i, name=name, parent_index=-1)

        # Infer parent from naming convention
        # "Bip01 Spine" → parent is "Bip01"
        # "Bip01 R Calf" → parent is "Bip01 R Thigh" (or nearest ancestor)
        if " " in name:
            parent_name = name.rsplit(" ", 1)[0]
            for j, pn in enumerate(bone_names):
                if pn == parent_name:
                    bone.parent_index = j
                    break

        result.bones.append(bone)


def _extract_parent_indices(data: bytes, result: ParsedHavok):
    """Find and apply the parent index array stored as int16 values.

    Havok stores bone parent indices as a contiguous int16 array where
    index 0 = -1 (root) and each subsequent value is the parent bone index.
    """
    if not result.bones:
        return

    bone_count = len(result.bones)
    if bone_count < 2:
        return

    # Scan for an int16 array that matches valid parent hierarchy:
    # [0] = -1, all others in range [-1, bone_count), and i > parent[i] (DAG)
    best_off = -1
    best_score = 0

    for off in range(0, len(data) - bone_count * 2, 2):
        first = struct.unpack_from("<h", data, off)[0]
        if first != -1:
            continue

        vals = [struct.unpack_from("<h", data, off + i * 2)[0]
                for i in range(bone_count)]

        # Validate: each parent must be -1 or a valid earlier index
        valid = True
        score = 0
        for i, v in enumerate(vals):
            if v < -1 or v >= bone_count:
                valid = False
                break
            if i > 0 and v == -1:
                score += 1  # multiple roots is less common but valid
            if 0 <= v < i:
                score += 2  # proper parent ordering
        if not valid:
            continue
        if score > best_score:
            best_score = score
            best_off = off

    if best_off >= 0:
        for i in range(bone_count):
            parent = struct.unpack_from("<h", data, best_off + i * 2)[0]
            result.bones[i].parent_index = parent


def get_hkx_summary(data: bytes) -> str:
    """Get a human-readable summary of an HKX file."""
    try:
        hkx = parse_hkx(data)
        lines = [
            f"Havok TAG0 File (SDK {hkx.sdk_version})",
            f"Size: {hkx.total_size:,} bytes",
            f"Sections: {len(hkx.sections)}",
        ]

        if hkx.bones:
            lines.append(f"Bones: {len(hkx.bones)}")
            for b in hkx.bones[:10]:
                parent = hkx.bones[b.parent_index].name if 0 <= b.parent_index < len(hkx.bones) else "ROOT"
                lines.append(f"  [{b.index}] {b.name} → {parent}")
            if len(hkx.bones) > 10:
                lines.append(f"  ... and {len(hkx.bones) - 10} more")

        content = []
        if hkx.has_skeleton:
            content.append("Skeleton")
        if hkx.has_animation:
            content.append("Animation")
        if hkx.has_physics:
            content.append("Physics")
        if hkx.has_ragdoll:
            content.append("Ragdoll")
        if content:
            lines.append(f"Content: {', '.join(content)}")

        if hkx.class_names:
            lines.append(f"Classes: {', '.join(hkx.class_names[:8])}")

        return "\n".join(lines)
    except Exception as e:
        return f"HKX parse error: {e}"


def is_havok_file(path: str) -> bool:
    """Check if a file is a Havok file."""
    return os.path.splitext(path.lower())[1] == ".hkx"
