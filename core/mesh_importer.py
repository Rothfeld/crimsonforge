"""OBJ importer and PAC/PAM binary builder for round-trip mesh modding.

Pipeline: Export .pac → edit in Blender → save .obj → import_obj() → build_pac() → repack

The OBJ file must have been exported by CrimsonForge (contains source_path
and source_format comments). The original PAC/PAM binary is needed to
preserve metadata (names, materials, bones, flags) that OBJ cannot store.
"""

from __future__ import annotations

import os
import struct
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from core.mesh_parser import ParsedMesh, SubMesh, parse_pac, parse_pam, parse_pamlod
from utils.logger import get_logger

logger = get_logger("core.mesh_importer")


# ═══════════════════════════════════════════════════════════════════════
#  OBJ IMPORTER
# ═══════════════════════════════════════════════════════════════════════

def import_obj(obj_path: str) -> ParsedMesh:
    """Import an OBJ file back into a ParsedMesh.

    Reads CrimsonForge metadata comments (source_path, source_format)
    to identify the original game file.

    Returns:
        ParsedMesh with vertices, UVs, normals, faces per submesh.
    """
    source_path = ""
    source_format = ""
    submeshes: list[SubMesh] = []

    # Current submesh being built
    current_name = ""
    verts: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    normals: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    # Global vertex/uv/normal arrays (OBJ uses global indices)
    all_verts: list[tuple[float, float, float]] = []
    all_uvs: list[tuple[float, float]] = []
    all_normals: list[tuple[float, float, float]] = []

    # Per-submesh: track which global indices belong to each submesh
    submesh_list: list[dict] = []
    current_faces_global: list[tuple] = []
    current_material = ""

    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Parse metadata comments
            if line.startswith("# source_path:"):
                source_path = line.split(":", 1)[1].strip()
                continue
            if line.startswith("# source_format:"):
                source_format = line.split(":", 1)[1].strip()
                continue
            if line.startswith("#") or not line:
                continue

            parts = line.split()
            if not parts:
                continue

            if parts[0] == "v" and len(parts) >= 4:
                all_verts.append((float(parts[1]), float(parts[2]), float(parts[3])))

            elif parts[0] == "vt" and len(parts) >= 3:
                u = float(parts[1])
                v = 1.0 - float(parts[2])  # flip V back (OBJ export flipped it)
                all_uvs.append((u, v))

            elif parts[0] == "vn" and len(parts) >= 4:
                all_normals.append((float(parts[1]), float(parts[2]), float(parts[3])))

            elif parts[0] == "o":
                # New object/submesh — save previous
                if current_name and current_faces_global:
                    submesh_list.append({
                        "name": current_name,
                        "material": current_material,
                        "faces_global": current_faces_global,
                    })
                current_name = parts[1] if len(parts) > 1 else f"submesh_{len(submesh_list)}"
                current_faces_global = []
                current_material = ""

            elif parts[0] == "usemtl":
                current_material = parts[1] if len(parts) > 1 else ""

            elif parts[0] == "f" and len(parts) >= 4:
                # Parse face indices (supports v, v/vt, v/vt/vn, v//vn)
                face_verts = []
                for fp in parts[1:4]:  # triangles only
                    indices = fp.split("/")
                    vi = int(indices[0]) - 1  # OBJ is 1-based
                    ti = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else -1
                    ni = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else -1
                    face_verts.append((vi, ti, ni))
                current_faces_global.append(tuple(face_verts))

    # Save last submesh
    if current_name and current_faces_global:
        submesh_list.append({
            "name": current_name,
            "material": current_material,
            "faces_global": current_faces_global,
        })

    # Convert global indices to per-submesh local indices.
    # Key: keep ALL vertices in each submesh's range (not just face-referenced ones).
    # Some meshes have unused vertices that must be preserved for correct rebuild.

    # First, determine vertex ownership: each submesh "owns" a contiguous range
    # based on the order vertices appear in the OBJ (submesh 0 first, etc.)
    vert_offset = 0
    for sm_data in submesh_list:
        # Count vertices that belong to this submesh in the OBJ
        # (vertices appear between 'o' markers, counted during parse above)
        # We stored them in all_verts in order — need to find this submesh's range
        pass

    # Build vertex ranges from the OBJ structure:
    # Vertices between successive 'o' markers belong to that submesh
    # Re-parse to find vertex counts per submesh
    sm_vert_counts = []
    sm_uv_counts = []
    sm_normal_counts = []
    current_v = current_vt = current_vn = 0

    with open(obj_path, "r", encoding="utf-8") as f:
        in_submesh = False
        for line in f:
            line = line.strip()
            if line.startswith("o "):
                if in_submesh:
                    sm_vert_counts.append(current_v)
                    sm_uv_counts.append(current_vt)
                    sm_normal_counts.append(current_vn)
                current_v = current_vt = current_vn = 0
                in_submesh = True
            elif line.startswith("v ") and not line.startswith("vt") and not line.startswith("vn"):
                current_v += 1
            elif line.startswith("vt "):
                current_vt += 1
            elif line.startswith("vn "):
                current_vn += 1
        if in_submesh:
            sm_vert_counts.append(current_v)
            sm_uv_counts.append(current_vt)
            sm_normal_counts.append(current_vn)

    # Now build each submesh using the FULL vertex range (not just face-referenced)
    v_offset = 0
    vt_offset = 0
    vn_offset = 0

    for si, sm_data in enumerate(submesh_list):
        nv = sm_vert_counts[si] if si < len(sm_vert_counts) else 0
        nvt = sm_uv_counts[si] if si < len(sm_uv_counts) else 0
        nvn = sm_normal_counts[si] if si < len(sm_normal_counts) else 0

        # ALL vertices in this submesh's range
        local_verts = [all_verts[v_offset + i] if (v_offset + i) < len(all_verts) else (0, 0, 0)
                       for i in range(nv)]
        local_uvs = [all_uvs[vt_offset + i] if (vt_offset + i) < len(all_uvs) else (0, 0)
                     for i in range(nvt)]
        local_normals = [all_normals[vn_offset + i] if (vn_offset + i) < len(all_normals) else (0, 1, 0)
                         for i in range(nvn)]

        # Remap face indices from global to local (subtract offset)
        local_faces = []
        for face in sm_data["faces_global"]:
            local_face = []
            for vi, ti, ni in face:
                local_vi = vi - v_offset
                if 0 <= local_vi < nv:
                    local_face.append(local_vi)
                else:
                    local_face.append(0)  # safety fallback
            if len(local_face) == 3:
                local_faces.append(tuple(local_face))

        sm = SubMesh(
            name=sm_data["name"],
            material=sm_data["material"],
            vertices=local_verts,
            uvs=local_uvs if len(local_uvs) == len(local_verts) else [],
            normals=local_normals if len(local_normals) == len(local_verts) else [],
            faces=local_faces,
            vertex_count=len(local_verts),
            face_count=len(local_faces),
        )
        submeshes.append(sm)

        v_offset += nv
        vt_offset += nvt
        vn_offset += nvn

    result = ParsedMesh(
        path=source_path,
        format=source_format,
        submeshes=submeshes,
        total_vertices=sum(len(s.vertices) for s in submeshes),
        total_faces=sum(len(s.faces) for s in submeshes),
        has_uvs=any(s.uvs for s in submeshes),
    )

    if result.submeshes:
        all_v = [v for s in submeshes for v in s.vertices]
        if all_v:
            xs, ys, zs = zip(*all_v)
            result.bbox_min = (min(xs), min(ys), min(zs))
            result.bbox_max = (max(xs), max(ys), max(zs))

    logger.info("Imported OBJ %s: %d submeshes, %d verts, %d faces, source=%s (%s)",
                obj_path, len(submeshes), result.total_vertices,
                result.total_faces, source_path, source_format)
    return result


# ═══════════════════════════════════════════════════════════════════════
#  QUANTIZATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _quantize_u16(value: float, vmin: float, vmax: float) -> int:
    """Float → uint16 quantized: inverse of dequantize."""
    if abs(vmax - vmin) < 1e-10:
        return 32768
    t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    return min(65535, max(0, round(t * 65535)))


def _compute_bbox(vertices: list[tuple[float, float, float]]):
    """Compute tight bounding box from vertex list."""
    if not vertices:
        return (0, 0, 0), (1, 1, 1)
    xs, ys, zs = zip(*vertices)
    # Add tiny epsilon to avoid zero-size bbox
    eps = 1e-6
    bmin = (min(xs) - eps, min(ys) - eps, min(zs) - eps)
    bmax = (max(xs) + eps, max(ys) + eps, max(zs) + eps)
    return bmin, bmax


# ═══════════════════════════════════════════════════════════════════════
#  PAC BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_pac(mesh: ParsedMesh, original_data: bytes) -> bytes:
    """Rebuild a PAC binary from modified mesh + original file data.

    The original PAC is needed to preserve:
    - Header (magic, version, timestamp)
    - Section 0 structure (flags, bone data, Havok data)
    - Non-geometry metadata

    Only vertex positions, UVs, and face indices are replaced.
    """
    if not original_data or original_data[:4] != b"PAR ":
        raise ValueError("Original PAC data required for rebuild")

    # Parse original to get metadata
    header_size = 80
    s0_start = header_size
    flags = struct.unpack_from("<I", original_data, s0_start)[0]
    n_lods = original_data[s0_start + 4]

    if n_lods == 0 or n_lods > 10:
        raise ValueError(f"Invalid n_lods: {n_lods}")

    # Read original section offsets
    off = s0_start + 5
    orig_lod_offsets = [struct.unpack_from("<I", original_data, off + i * 4)[0] for i in range(n_lods)]
    off += n_lods * 4
    orig_split_offsets = [struct.unpack_from("<I", original_data, off + i * 4)[0] for i in range(n_lods)]
    off += n_lods * 4

    # Compute original section boundaries
    sorted_offsets = sorted(orig_lod_offsets)
    orig_boundaries = [header_size] + sorted_offsets + [len(original_data)]
    orig_sections = [(orig_boundaries[i], orig_boundaries[i + 1])
                     for i in range(len(orig_boundaries) - 1)]

    # Extract original section 0 content (everything from s0 start to first LOD)
    orig_s0 = bytearray(original_data[orig_sections[0][0]:orig_sections[0][1]])

    # Parse original submesh descriptors to get metadata we need to preserve
    orig_mesh = parse_pac(original_data, mesh.path)

    # ── Build LOD data sections ──
    # We only modify LOD0 (highest quality). Lower LODs get the same data
    # (simplified — proper LOD generation would decimate the mesh).

    lod0_verts_buf = bytearray()
    lod0_idx_buf = bytearray()

    for sm_idx, sm in enumerate(mesh.submeshes):
        bmin, bmax = _compute_bbox(sm.vertices)

        # Build vertex records (stride auto-matched to original)
        # Detect original stride from original LOD0 section
        orig_lod0 = orig_sections[-1]
        orig_lod0_size = orig_lod0[1] - orig_lod0[0]
        orig_total_verts = sum(
            s.get("vert_counts", [0])[0] if isinstance(s, dict) else s.vertex_count
            for s in (orig_mesh.submeshes if orig_mesh.submeshes else [{"vert_counts": [0]}])
        )
        orig_total_idx = sum(
            s.get("idx_counts", [0])[0] if isinstance(s, dict) else len(s.faces) * 3
            for s in (orig_mesh.submeshes if orig_mesh.submeshes else [{"idx_counts": [0]}])
        )

        if orig_total_verts > 0:
            stride = (orig_lod0_size - orig_total_idx * 2) // orig_total_verts
        else:
            stride = 40  # default

        stride = max(36, min(64, stride))  # clamp to reasonable range

        for vi in range(len(sm.vertices)):
            vx, vy, vz = sm.vertices[vi]
            xu = _quantize_u16(vx, bmin[0], bmax[0])
            yu = _quantize_u16(vy, bmin[1], bmax[1])
            zu = _quantize_u16(vz, bmin[2], bmax[2])

            rec = bytearray(stride)
            # Position: bytes 0-5
            struct.pack_into("<HHH", rec, 0, xu, yu, zu)
            # UV: bytes 8-11 as float16
            if vi < len(sm.uvs):
                u, v = sm.uvs[vi]
                try:
                    struct.pack_into("<e", rec, 8, u)
                    struct.pack_into("<e", rec, 10, v)
                except (OverflowError, ValueError):
                    pass
            # Constant at bytes 12-15
            struct.pack_into("<I", rec, 12, 0x3C000000)
            # Bone: bytes 28-31 = 0xFF000000 (no bone / default)
            if stride >= 32:
                struct.pack_into("<I", rec, 28, 0x000000FF)
            # Terminator at last 4 bytes
            struct.pack_into("<I", rec, stride - 4, 0xFFFFFFFF)

            lod0_verts_buf.extend(rec)

        # Index buffer: triangle list
        for a, b, c in sm.faces:
            lod0_idx_buf.extend(struct.pack("<HHH", a, b, c))

    # For lower LODs, copy LOD0 data (simplified)
    lod_data = [bytes(lod0_verts_buf) + bytes(lod0_idx_buf)] * n_lods

    # ── Rebuild section 0 ──
    # Update submesh descriptors in section 0 with new bbox and counts
    new_s0 = _rebuild_pac_section0(
        orig_s0, original_data, n_lods, mesh.submeshes, stride,
        flags, orig_lod_offsets, orig_split_offsets
    )

    # ── Assemble final PAC ──
    # Header (80 bytes) + section 0 + LOD sections (lowest to highest)
    # LOD sections are stored in ascending quality order: LOD(n-1), ..., LOD1, LOD0

    # Compute new section positions
    s0_size = len(new_s0)
    lod_sizes = [len(d) for d in lod_data]

    # Sections are ordered: sec0, LOD_lowest, ..., LOD_highest
    # LOD offsets (stored LOD0-first in section 0) are absolute file positions
    sec_positions = [header_size]  # sec0 start
    pos = header_size + s0_size
    for sz in reversed(lod_sizes):  # lowest LOD first in file
        sec_positions.append(pos)
        pos += sz

    # LOD offsets in descending order (LOD0 first)
    new_lod_offsets = list(reversed(sec_positions[1:]))

    # Split offsets: vertex data ends, index data begins
    new_split_offsets = []
    for i, sm_list_data in enumerate(lod_data):
        total_v = sum(len(s.vertices) for s in mesh.submeshes)
        split = sec_positions[n_lods - i] + total_v * stride  # absolute
        new_split_offsets.append(split)

    # Update offsets in section 0
    off = 5  # after flags(4) + n_lods(1)
    for i in range(n_lods):
        struct.pack_into("<I", new_s0, off + i * 4, new_lod_offsets[i])
    off += n_lods * 4
    for i in range(n_lods):
        struct.pack_into("<I", new_s0, off + i * 4, new_split_offsets[i])

    # Build header
    header = bytearray(original_data[:header_size])

    # Update section sizes in header (try u64 format first)
    all_sec_sizes = [s0_size] + list(reversed(lod_sizes))
    # Write as u64 at 0x14 (fits in 5 slots for up to 5 sections)
    for i, sz in enumerate(all_sec_sizes):
        if 0x14 + i * 8 + 8 <= header_size:
            struct.pack_into("<Q", header, 0x14 + i * 8, sz)

    # Assemble
    result = bytearray(header)
    result.extend(new_s0)
    for d in reversed(lod_data):  # lowest LOD first in file
        result.extend(d)

    logger.info("Built PAC %s: %d bytes (%d submeshes, %d verts, %d faces)",
                mesh.path, len(result), len(mesh.submeshes),
                mesh.total_vertices, mesh.total_faces)
    return bytes(result)


def _rebuild_pac_section0(orig_s0: bytearray, original_data: bytes,
                          n_lods: int, submeshes: list[SubMesh],
                          stride: int, flags: int,
                          orig_lod_offsets: list, orig_split_offsets: list) -> bytearray:
    """Rebuild section 0 with updated submesh bbox and counts.

    Preserves all original data (names, materials, bones, Havok data),
    only updates the bounding box floats and vertex/index counts.
    """
    s0 = bytearray(orig_s0)

    # Find submesh descriptors by scanning for strings (same as parser)
    off = 5 + n_lods * 4 * 2  # after flags + offset tables

    # Scan for first string
    scan = off
    while scan < len(s0) - 10:
        b = s0[scan]
        if 4 < b < 100:
            test = s0[scan + 1:scan + 1 + b]
            if len(test) == b and all(32 <= c < 127 for c in test):
                break
        scan += 1
    off = scan

    sm_idx = 0
    while off < len(s0) - 20 and sm_idx < len(submeshes):
        name_len = s0[off]
        if name_len == 0 or name_len > 200:
            break
        off += 1 + name_len  # skip name

        mat_len = s0[off]
        off += 1 + mat_len  # skip material

        # flag + pad
        off += 3

        # Update 8 bbox floats: [pivot_x, pivot_y, bmin_x, bmin_y, bmin_z, bmax_x, bmax_y, bmax_z]
        sm = submeshes[sm_idx]
        bmin, bmax = _compute_bbox(sm.vertices)

        # Preserve original pivot (floats[0:2])
        # Update bbox (floats[2:8])
        struct.pack_into("<f", s0, off + 2 * 4, bmin[0])
        struct.pack_into("<f", s0, off + 3 * 4, bmin[1])
        struct.pack_into("<f", s0, off + 4 * 4, bmin[2])
        struct.pack_into("<f", s0, off + 5 * 4, bmax[0])
        struct.pack_into("<f", s0, off + 6 * 4, bmax[1])
        struct.pack_into("<f", s0, off + 7 * 4, bmax[2])
        off += 32

        # Skip bone data
        bone_count = s0[off]
        off += 1
        bones_size = bone_count + (bone_count % 2)
        off += bones_size

        # Update vertex counts (n_lods × u16) — set all LODs to LOD0 value
        nv = len(sm.vertices)
        for i in range(n_lods):
            struct.pack_into("<H", s0, off + i * 2, nv)
        off += n_lods * 2

        # Update index counts (read until garbage, then update valid ones)
        ni = len(sm.faces) * 3
        for i in range(n_lods):
            if off + 4 > len(s0):
                break
            val = struct.unpack_from("<I", s0, off)[0]
            if val > 10_000_000:
                break
            struct.pack_into("<I", s0, off, ni)
            off += 4

        sm_idx += 1

        # Check next submesh
        if off >= len(s0) - 4:
            break
        next_b = s0[off]
        if next_b == 0 or next_b > 200:
            break
        peek = s0[off + 1:off + 1 + min(next_b, 6)]
        if not all(32 <= c < 127 for c in peek):
            break

    return s0


# ═══════════════════════════════════════════════════════════════════════
#  PAM BUILDER
# ═══════════════════════════════════════════════════════════════════════

def build_pam(mesh: ParsedMesh, original_data: bytes) -> bytes:
    """Rebuild a PAM binary by patching vertex positions in-place.

    Instead of rebuilding geometry from scratch (which breaks scan-fallback
    parsed files), we parse the original to find where each vertex lives
    in the binary, then overwrite just the position uint16 values.
    This preserves all other data (indices, UVs, normals, metadata).
    """
    if not original_data or original_data[:4] != b"PAR ":
        raise ValueError("Original PAM data required for rebuild")

    HDR_BBOX_MIN = 0x14
    HDR_BBOX_MAX = 0x20
    HDR_GEOM_OFF = 0x3C

    result = bytearray(original_data)

    # Read original bbox — use for quantization, expand only if needed
    orig_bmin = struct.unpack_from("<fff", original_data, HDR_BBOX_MIN)
    orig_bmax = struct.unpack_from("<fff", original_data, HDR_BBOX_MAX)

    bmin, bmax = orig_bmin, orig_bmax
    if mesh.submeshes:
        all_v = [v for s in mesh.submeshes for v in s.vertices]
        if all_v:
            xs, ys, zs = zip(*all_v)
            bmin = (min(orig_bmin[0], min(xs)), min(orig_bmin[1], min(ys)),
                    min(orig_bmin[2], min(zs)))
            bmax = (max(orig_bmax[0], max(xs)), max(orig_bmax[1], max(ys)),
                    max(orig_bmax[2], max(zs)))
            struct.pack_into("<fff", result, HDR_BBOX_MIN, *bmin)
            struct.pack_into("<fff", result, HDR_BBOX_MAX, *bmax)

    # Parse the original to find exact vertex byte positions.
    # Instead of computing offsets ourselves, we find each vertex by
    # reverse-searching its quantized uint16 position in the binary.
    orig_mesh = parse_pam(original_data, mesh.path)
    if not orig_mesh.submeshes:
        return bytes(result)

    geom_off = struct.unpack_from("<I", original_data, HDR_GEOM_OFF)[0]

    # For each original vertex, find its byte offset by matching its
    # quantized uint16 values. Build a map: (submesh_idx, vert_idx) → byte_offset
    vert_offsets = []
    search_start = geom_off

    for sm_idx, (orig_sm, new_sm) in enumerate(zip(orig_mesh.submeshes, mesh.submeshes)):
        n = min(len(orig_sm.vertices), len(new_sm.vertices))
        sm_offsets = []

        for vi in range(len(orig_sm.vertices)):
            vx, vy, vz = orig_sm.vertices[vi]
            # Quantize original vertex to find its uint16 pattern
            xu = _quantize_u16(vx, orig_bmin[0], orig_bmax[0])
            yu = _quantize_u16(vy, orig_bmin[1], orig_bmax[1])
            zu = _quantize_u16(vz, orig_bmin[2], orig_bmax[2])
            target = struct.pack("<HHH", xu, yu, zu)

            # Search forward from last found position
            found = -1
            for scan in range(search_start, len(original_data) - 6):
                if original_data[scan:scan + 6] == target:
                    found = scan
                    search_start = scan + 6  # next search starts after this
                    break

            sm_offsets.append(found)

        # Patch vertices that have valid offsets
        patched = 0
        for vi in range(n):
            if vi >= len(sm_offsets) or sm_offsets[vi] < 0:
                continue
            byte_off = sm_offsets[vi]

            vx, vy, vz = new_sm.vertices[vi]
            xu = _quantize_u16(vx, bmin[0], bmax[0])
            yu = _quantize_u16(vy, bmin[1], bmax[1])
            zu = _quantize_u16(vz, bmin[2], bmax[2])

            if byte_off + 6 <= len(result):
                struct.pack_into("<HHH", result, byte_off, xu, yu, zu)
                patched += 1

        vert_offsets.append(patched)

    total_patched = sum(vert_offsets)
    logger.info("Built PAM %s: %d bytes (patched %d verts in-place)",
                mesh.path, len(result), total_patched)
    return bytes(result)


# ═══════════════════════════════════════════════════════════════════════
#  AUTO-DETECT AND BUILD
# ═══════════════════════════════════════════════════════════════════════

def build_mesh(mesh: ParsedMesh, original_data: bytes) -> bytes:
    """Auto-detect format and rebuild binary from modified mesh.

    Args:
        mesh: Modified ParsedMesh (from import_obj or manual modification).
        original_data: Original binary data (needed for metadata preservation).

    Returns:
        New binary data ready for repack.
    """
    fmt = mesh.format.lower()
    if fmt == "pac":
        return build_pac(mesh, original_data)
    elif fmt in ("pam", "pamlod"):
        return build_pam(mesh, original_data)
    else:
        raise ValueError(f"Unsupported mesh format for rebuild: {fmt}")
