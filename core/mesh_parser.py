"""PAM / PAMLOD / PAC mesh parser for Crimson Desert.

Parses Pearl Abyss 3D mesh files from PAZ archives into an intermediate
representation (vertices, UVs, normals, faces, materials, bones, weights)
that can be exported to OBJ, FBX, or rendered in the 3D preview.

Format overview (all share the 'PAR ' magic):
  PAM     — static meshes (objects, props, world geometry)
  PAMLOD  — LOD variants (5 quality levels per mesh)
  PAC     — skinned character meshes (with bone indices + weights)

Vertex positions are uint16-quantized and dequantized using the per-file
bounding box.  UVs are stored as float16 at vertex offset +8/+10.  Bone
weights (PAC only) follow the UV data.
"""

from __future__ import annotations

import os
import re
import struct
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger("core.mesh_parser")

# ── Constants ────────────────────────────────────────────────────────

PAR_MAGIC = b"PAR "

# PAM header offsets
HDR_MESH_COUNT = 0x10
HDR_BBOX_MIN = 0x14
HDR_BBOX_MAX = 0x20
HDR_GEOM_OFF = 0x3C

# Submesh table
SUBMESH_TABLE = 0x410
SUBMESH_STRIDE = 0x218
SUBMESH_TEX_OFF = 0x10
SUBMESH_MAT_OFF = 0x110

# Global-buffer prefab constants
GLOBAL_VERT_BASE = 3068
PAM_IDX_OFF = 0x19840

# PAMLOD header offsets
PAMLOD_LOD_COUNT = 0x00
PAMLOD_GEOM_OFF = 0x04
PAMLOD_BBOX_MIN = 0x10
PAMLOD_BBOX_MAX = 0x1C
PAMLOD_ENTRY_TABLE = 0x50

# Stride candidates for auto-detection
STRIDE_CANDIDATES = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64]


# ── Data structures ──────────────────────────────────────────────────

@dataclass
class MeshVertex:
    """Single vertex with position, UV, and optional bone data."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    u: float = 0.0
    v: float = 0.0
    nx: float = 0.0
    ny: float = 1.0
    nz: float = 0.0
    bone_indices: tuple[int, ...] = ()
    bone_weights: tuple[float, ...] = ()


@dataclass
class SubMesh:
    """A submesh within a PAM/PAC file."""
    name: str = ""
    material: str = ""
    texture: str = ""
    vertices: list[tuple[float, float, float]] = field(default_factory=list)
    uvs: list[tuple[float, float]] = field(default_factory=list)
    normals: list[tuple[float, float, float]] = field(default_factory=list)
    faces: list[tuple[int, int, int]] = field(default_factory=list)
    bone_indices: list[tuple[int, ...]] = field(default_factory=list)
    bone_weights: list[tuple[float, ...]] = field(default_factory=list)
    vertex_count: int = 0
    face_count: int = 0


@dataclass
class ParsedMesh:
    """Complete parsed mesh file."""
    path: str = ""
    format: str = ""  # "pam", "pamlod", "pac"
    bbox_min: tuple[float, float, float] = (0, 0, 0)
    bbox_max: tuple[float, float, float] = (0, 0, 0)
    submeshes: list[SubMesh] = field(default_factory=list)
    lod_levels: list[list[SubMesh]] = field(default_factory=list)  # PAMLOD only
    total_vertices: int = 0
    total_faces: int = 0
    has_uvs: bool = False
    has_bones: bool = False


# ── Utility ──────────────────────────────────────────────────────────

def _dequant_u16(v: int, mn: float, mx: float) -> float:
    """uint16 → float: bbox_min + (v / 65535) * (bbox_max - bbox_min)."""
    return mn + (v / 65535.0) * (mx - mn)


def _dequant_i16(v: int, mn: float, mx: float) -> float:
    """int16 → float (legacy global-buffer format)."""
    return mn + ((v + 32768) / 65536.0) * (mx - mn)


def _compute_face_normal(v0, v1, v2):
    """Compute face normal from 3 vertex positions."""
    ax, ay, az = v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]
    bx, by, bz = v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length > 1e-8:
        return (nx / length, ny / length, nz / length)
    return (0.0, 1.0, 0.0)


def _compute_smooth_normals(vertices, faces):
    """Compute per-vertex smooth normals by averaging adjacent face normals."""
    normals = [[0.0, 0.0, 0.0] for _ in range(len(vertices))]
    for a, b, c in faces:
        if a < len(vertices) and b < len(vertices) and c < len(vertices):
            fn = _compute_face_normal(vertices[a], vertices[b], vertices[c])
            for idx in (a, b, c):
                normals[idx][0] += fn[0]
                normals[idx][1] += fn[1]
                normals[idx][2] += fn[2]
    result = []
    for n in normals:
        length = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
        if length > 1e-8:
            result.append((n[0] / length, n[1] / length, n[2] / length))
        else:
            result.append((0.0, 1.0, 0.0))
    return result


# ── Stride detection ─────────────────────────────────────────────────

def _find_local_stride(data: bytes, geom_off: int, voff: int, n_verts: int, n_idx: int):
    """Detect vertex stride for per-mesh layout where indices follow vertex data."""
    for stride in STRIDE_CANDIDATES:
        vert_start = geom_off + voff
        idx_off = vert_start + n_verts * stride
        if idx_off + n_idx * 2 > len(data):
            continue
        # Validate: all index values must be < n_verts
        valid = True
        for j in range(min(n_idx, 100)):  # sample first 100 for speed
            val = struct.unpack_from("<H", data, idx_off + j * 2)[0]
            if val >= n_verts:
                valid = False
                break
        if valid:
            # Full validation on remaining
            if n_idx > 100:
                valid = all(
                    struct.unpack_from("<H", data, idx_off + j * 2)[0] < n_verts
                    for j in range(100, n_idx)
                )
            if valid:
                return stride, idx_off
    return None, None


# ── PAM Parser ───────────────────────────────────────────────────────

def parse_pam(data: bytes, filename: str = "") -> ParsedMesh:
    """Parse a .pam static mesh file."""
    if len(data) < 0x40 or data[:4] != PAR_MAGIC:
        raise ValueError(f"Not a valid PAM file: bad magic {data[:4]!r}")

    result = ParsedMesh(path=filename, format="pam")
    result.bbox_min = struct.unpack_from("<fff", data, HDR_BBOX_MIN)
    result.bbox_max = struct.unpack_from("<fff", data, HDR_BBOX_MAX)
    geom_off = struct.unpack_from("<I", data, HDR_GEOM_OFF)[0]
    mesh_count = struct.unpack_from("<I", data, HDR_MESH_COUNT)[0]
    bmin, bmax = result.bbox_min, result.bbox_max

    # Read submesh table
    raw_entries = []
    for i in range(mesh_count):
        off = SUBMESH_TABLE + i * SUBMESH_STRIDE
        if off + SUBMESH_STRIDE > len(data):
            break
        nv = struct.unpack_from("<I", data, off)[0]
        ni = struct.unpack_from("<I", data, off + 4)[0]
        ve = struct.unpack_from("<I", data, off + 8)[0]
        ie = struct.unpack_from("<I", data, off + 12)[0]
        tex = data[off + SUBMESH_TEX_OFF:off + SUBMESH_TEX_OFF + 256].split(b"\x00")[0].decode("ascii", "replace")
        mat = data[off + SUBMESH_MAT_OFF:off + SUBMESH_MAT_OFF + 256].split(b"\x00")[0].decode("ascii", "replace")
        raw_entries.append({"i": i, "nv": nv, "ni": ni, "ve": ve, "ie": ie, "tex": tex, "mat": mat})

    # Detect combined-buffer layout
    is_combined = False
    if mesh_count > 1:
        ve_acc = ie_acc = 0
        is_combined = True
        for r in raw_entries:
            if r["ve"] != ve_acc or r["ie"] != ie_acc:
                is_combined = False
                break
            ve_acc += r["nv"]
            ie_acc += r["ni"]

    if is_combined:
        _parse_combined_buffer(data, raw_entries, geom_off, bmin, bmax, result)
    else:
        _parse_independent_meshes(data, raw_entries, geom_off, bmin, bmax, result)

    # Fallback: if no vertices found, try scanning for vertex+index blocks
    # This handles "breakable" PAMs and other extended layouts that have
    # extra data (physics/destruction metadata) before the actual geometry
    if result.total_vertices == 0 and mesh_count > 0:
        _parse_scan_fallback(data, raw_entries, geom_off, bmin, bmax, result)

    # Compute normals for all submeshes
    for sm in result.submeshes:
        sm.normals = _compute_smooth_normals(sm.vertices, sm.faces)

    result.total_vertices = sum(len(sm.vertices) for sm in result.submeshes)
    result.total_faces = sum(len(sm.faces) for sm in result.submeshes)
    result.has_uvs = any(sm.uvs for sm in result.submeshes)

    logger.info("Parsed PAM %s: %d submeshes, %d verts, %d faces",
                filename, len(result.submeshes), result.total_vertices, result.total_faces)
    return result


def _parse_independent_meshes(data, entries, geom_off, bmin, bmax, result):
    """Parse PAM with per-submesh or global vertex buffers."""
    idx_avail = (len(data) - PAM_IDX_OFF) // 2

    for r in entries:
        i, nv, ni, voff, ioff = r["i"], r["nv"], r["ni"], r["ve"], r["ie"]
        tex, mat = r["tex"], r["mat"]

        # Try local layout first
        stride, idx_off = _find_local_stride(data, geom_off, voff, nv, ni)

        if stride is not None:
            verts, uvs, faces = _extract_local_mesh(data, geom_off, voff, stride, idx_off, nv, ni, bmin, bmax)
        elif ioff + ni <= idx_avail:
            verts, uvs, faces = _extract_global_mesh(data, geom_off, ni, ioff, bmin, bmax)
        else:
            continue

        sm = SubMesh(
            name=f"mesh_{i:02d}_{mat or str(i)}",
            material=mat, texture=tex,
            vertices=verts, uvs=uvs, faces=faces,
            vertex_count=len(verts), face_count=len(faces),
        )
        result.submeshes.append(sm)


def _parse_scan_fallback(data, entries, geom_off, bmin, bmax, result):
    """Fallback parser: scan for vertex+index blocks in extended-layout PAMs.

    Breakable/destructible PAMs often have extra metadata (physics, destruction
    fragments) between the header and the actual geometry. This scanner probes
    the region after geom_off to locate the real vertex positions (uint16
    quantized) and matching index block.
    """
    total_v = sum(r["nv"] for r in entries)
    total_i = sum(r["ni"] for r in entries)
    if total_v < 3 or total_i < 3:
        return

    search_limit = min(len(data) - 100, geom_off + min(len(data) // 2, 2000000))

    # Scan for a block of u16 values that look like quantized vertex positions
    # (spread across the 0-65535 range), followed by valid indices.
    # Step by 2 in small files, step by 4 in large files for speed.
    step = 2 if (search_limit - geom_off) < 500000 else 4
    for scan_start in range(geom_off, search_limit, step):
        # Quick check: read 10 potential XYZ triples (stride 6)
        if scan_start + 60 > len(data):
            break
        vals = [struct.unpack_from("<H", data, scan_start + j * 2)[0] for j in range(30)]
        spread = max(vals) - min(vals)
        if spread < 5000:
            continue

        # Found candidate vertex data. Try common strides
        for try_stride in [6, 8, 10, 12, 14, 16, 20, 24, 28, 32]:
            test_idx_off = scan_start + total_v * try_stride
            if test_idx_off + total_i * 2 > len(data):
                continue

            # Validate: first 50 indices must be < total_v
            valid = True
            for j in range(min(50, total_i)):
                v = struct.unpack_from("<H", data, test_idx_off + j * 2)[0]
                if v >= total_v:
                    valid = False
                    break
            if not valid:
                continue

            # Full validation on a larger sample
            valid = all(
                struct.unpack_from("<H", data, test_idx_off + j * 2)[0] < total_v
                for j in range(min(total_i, 500))
            )
            if not valid:
                continue

            # Found valid layout! Parse as combined buffer from this offset
            logger.info("Scan fallback: found vertex data at 0x%X stride=%d for %s",
                        scan_start, try_stride, entries[0].get("tex", ""))

            has_uv = try_stride >= 12
            idx_base = test_idx_off

            for r in entries:
                nv, ni = r["nv"], r["ni"]
                vert_base = scan_start + r["ve"] * try_stride
                idx_off = idx_base + r["ie"] * 2

                indices = [struct.unpack_from("<H", data, idx_off + j * 2)[0]
                           for j in range(ni)]
                if not indices:
                    continue

                unique = sorted(set(indices))
                idx_map = {gi: li for li, gi in enumerate(unique)}

                verts, uvs = [], []
                for gi in unique:
                    foff = vert_base + gi * try_stride
                    if foff + 6 > len(data):
                        break
                    xu, yu, zu = struct.unpack_from("<HHH", data, foff)
                    verts.append((_dequant_u16(xu, bmin[0], bmax[0]),
                                  _dequant_u16(yu, bmin[1], bmax[1]),
                                  _dequant_u16(zu, bmin[2], bmax[2])))
                    if has_uv and foff + 12 <= len(data):
                        u = struct.unpack_from("<e", data, foff + 8)[0]
                        v = struct.unpack_from("<e", data, foff + 10)[0]
                        uvs.append((u, v))

                faces = []
                for j in range(0, ni - 2, 3):
                    a, b, c = indices[j], indices[j + 1], indices[j + 2]
                    if a in idx_map and b in idx_map and c in idx_map:
                        faces.append((idx_map[a], idx_map[b], idx_map[c]))

                sm = SubMesh(
                    name=f"mesh_{r['i']:02d}_{r['mat'] or str(r['i'])}",
                    material=r["mat"], texture=r["tex"],
                    vertices=verts, uvs=uvs, faces=faces,
                    vertex_count=len(verts), face_count=len(faces),
                )
                result.submeshes.append(sm)

            result.total_vertices = sum(len(sm.vertices) for sm in result.submeshes)
            result.total_faces = sum(len(sm.faces) for sm in result.submeshes)
            result.has_uvs = any(sm.uvs for sm in result.submeshes)
            return  # Done

    # Second pass: scan BACKWARD from end of file for the index block
    # This handles files where extra per-vertex data creates non-integer strides
    for scan_end_off in range(len(data) - 2, geom_off + total_v * 6, -2):
        test_start = scan_end_off - total_i * 2 + 2
        if test_start < geom_off:
            break

        # Quick check first index
        first_val = struct.unpack_from("<H", data, test_start)[0]
        if first_val >= total_v:
            continue

        # Check first 30 indices
        valid = True
        for j in range(min(30, total_i)):
            v = struct.unpack_from("<H", data, test_start + j * 2)[0]
            if v >= total_v:
                valid = False
                break
        if not valid:
            continue

        # Deeper validation
        valid = all(
            struct.unpack_from("<H", data, test_start + j * 2)[0] < total_v
            for j in range(min(total_i, 300))
        )
        if not valid:
            continue

        # Full validation
        valid = all(
            struct.unpack_from("<H", data, test_start + j * 2)[0] < total_v
            for j in range(total_i)
        )
        if not valid:
            continue

        # Found index block! Calculate vertex region
        vert_region = test_start - geom_off
        # Try common strides that fit
        best_stride = None
        for try_stride in [6, 8, 10, 12, 14, 16, 20, 24, 28, 32]:
            expected_end = geom_off + total_v * try_stride
            # Allow up to 16KB padding between vertex data and index data
            if expected_end <= test_start and (test_start - expected_end) < 16384:
                best_stride = try_stride
                break

        if best_stride is None:
            # Use floor division of vert_region / total_v
            best_stride = vert_region // total_v
            if best_stride < 6:
                best_stride = 6

        has_uv = best_stride >= 12
        idx_base = test_start
        logger.info("Backward scan: found idx at 0x%X stride=%d for %d verts",
                    test_start, best_stride, total_v)

        for r in entries:
            nv, ni = r["nv"], r["ni"]
            vert_base = geom_off + r["ve"] * best_stride
            idx_off = idx_base + r["ie"] * 2

            indices = [struct.unpack_from("<H", data, idx_off + j * 2)[0]
                       for j in range(ni)]
            if not indices:
                continue

            unique = sorted(set(indices))
            idx_map = {gi: li for li, gi in enumerate(unique)}

            verts, uvs = [], []
            for gi in unique:
                foff = vert_base + gi * best_stride
                if foff + 6 > len(data):
                    break
                xu, yu, zu = struct.unpack_from("<HHH", data, foff)
                verts.append((_dequant_u16(xu, bmin[0], bmax[0]),
                              _dequant_u16(yu, bmin[1], bmax[1]),
                              _dequant_u16(zu, bmin[2], bmax[2])))
                if has_uv and foff + 12 <= len(data):
                    u = struct.unpack_from("<e", data, foff + 8)[0]
                    v = struct.unpack_from("<e", data, foff + 10)[0]
                    uvs.append((u, v))

            faces = []
            for j in range(0, ni - 2, 3):
                a, b, c = indices[j], indices[j + 1], indices[j + 2]
                if a in idx_map and b in idx_map and c in idx_map:
                    faces.append((idx_map[a], idx_map[b], idx_map[c]))

            sm = SubMesh(
                name=f"mesh_{r['i']:02d}_{r['mat'] or str(r['i'])}",
                material=r["mat"], texture=r["tex"],
                vertices=verts, uvs=uvs, faces=faces,
                vertex_count=len(verts), face_count=len(faces),
            )
            result.submeshes.append(sm)

        result.total_vertices = sum(len(sm.vertices) for sm in result.submeshes)
        result.total_faces = sum(len(sm.faces) for sm in result.submeshes)
        result.has_uvs = any(sm.uvs for sm in result.submeshes)
        return

    logger.debug("Scan fallback: no valid vertex block found after 0x%X", geom_off)


def _parse_combined_buffer(data, entries, geom_off, bmin, bmax, result):
    """Parse PAM with shared vertex + index buffer."""
    total_verts = sum(r["nv"] for r in entries)
    total_idx = sum(r["ni"] for r in entries)
    avail = len(data) - geom_off

    target = (avail - total_idx * 2) / total_verts if total_verts else 0
    stride = min(STRIDE_CANDIDATES, key=lambda s: abs(s - target))
    if geom_off + total_verts * stride + total_idx * 2 > len(data):
        return

    idx_base = geom_off + total_verts * stride

    for r in entries:
        nv, ni = r["nv"], r["ni"]
        vert_base = geom_off + r["ve"] * stride
        idx_off = idx_base + r["ie"] * 2
        tex, mat = r["tex"], r["mat"]

        indices = [struct.unpack_from("<H", data, idx_off + j * 2)[0] for j in range(ni)]
        if not indices:
            continue

        unique = sorted(set(indices))
        idx_map = {gi: li for li, gi in enumerate(unique)}
        has_uv = stride >= 12

        verts, uvs = [], []
        for gi in unique:
            foff = vert_base + gi * stride
            if foff + 6 > len(data):
                break
            xu, yu, zu = struct.unpack_from("<HHH", data, foff)
            verts.append((_dequant_u16(xu, bmin[0], bmax[0]),
                          _dequant_u16(yu, bmin[1], bmax[1]),
                          _dequant_u16(zu, bmin[2], bmax[2])))
            if has_uv and foff + 12 <= len(data):
                u = struct.unpack_from("<e", data, foff + 8)[0]
                v = struct.unpack_from("<e", data, foff + 10)[0]
                uvs.append((u, v))

        faces = []
        for j in range(0, ni - 2, 3):
            a, b, c = indices[j], indices[j + 1], indices[j + 2]
            if a in idx_map and b in idx_map and c in idx_map:
                faces.append((idx_map[a], idx_map[b], idx_map[c]))

        sm = SubMesh(
            name=f"mesh_{r['i']:02d}_{mat or str(r['i'])}",
            material=mat, texture=tex,
            vertices=verts, uvs=uvs, faces=faces,
            vertex_count=len(verts), face_count=len(faces),
        )
        result.submeshes.append(sm)


def _extract_local_mesh(data, geom_off, voff, stride, idx_off, nv, ni, bmin, bmax):
    """Extract vertices/uvs/faces from local (per-mesh) layout."""
    indices = [struct.unpack_from("<H", data, idx_off + j * 2)[0] for j in range(ni)]
    unique = sorted(set(indices))
    idx_map = {gi: li for li, gi in enumerate(unique)}
    has_uv = stride >= 12

    verts, uvs = [], []
    for gi in unique:
        foff = geom_off + voff + gi * stride
        if foff + 6 > len(data):
            break
        xu, yu, zu = struct.unpack_from("<HHH", data, foff)
        verts.append((_dequant_u16(xu, bmin[0], bmax[0]),
                      _dequant_u16(yu, bmin[1], bmax[1]),
                      _dequant_u16(zu, bmin[2], bmax[2])))
        if has_uv and foff + 12 <= len(data):
            u = struct.unpack_from("<e", data, foff + 8)[0]
            v = struct.unpack_from("<e", data, foff + 10)[0]
            uvs.append((u, v))

    faces = []
    for j in range(0, ni - 2, 3):
        a, b, c = indices[j], indices[j + 1], indices[j + 2]
        if a in idx_map and b in idx_map and c in idx_map:
            faces.append((idx_map[a], idx_map[b], idx_map[c]))

    return verts, uvs, faces


def _extract_global_mesh(data, geom_off, ni, ioff, bmin, bmax):
    """Extract vertices/uvs/faces from global (prefab) layout."""
    indices = [struct.unpack_from("<H", data, PAM_IDX_OFF + (ioff + j) * 2)[0] for j in range(ni)]
    unique = sorted(set(indices))
    idx_map = {gi: li for li, gi in enumerate(unique)}

    verts = []
    for gi in unique:
        li = gi - GLOBAL_VERT_BASE
        foff = geom_off + li * 6
        if foff + 6 > len(data):
            break
        xi, yi, zi = struct.unpack_from("<hhh", data, foff)
        verts.append((_dequant_i16(xi, bmin[0], bmax[0]),
                      _dequant_i16(yi, bmin[1], bmax[1]),
                      _dequant_i16(zi, bmin[2], bmax[2])))

    faces = []
    for j in range(0, ni - 2, 3):
        a, b, c = indices[j], indices[j + 1], indices[j + 2]
        if a in idx_map and b in idx_map and c in idx_map:
            faces.append((idx_map[a], idx_map[b], idx_map[c]))

    return verts, [], faces


# ── PAMLOD Parser ────────────────────────────────────────────────────

def parse_pamlod(data: bytes, filename: str = "", lod_level: int = 0) -> ParsedMesh:
    """Parse a .pamlod LOD mesh file. lod_level=0 is highest quality."""
    result = ParsedMesh(path=filename, format="pamlod")

    lod_count = struct.unpack_from("<I", data, PAMLOD_LOD_COUNT)[0]
    geom_off = struct.unpack_from("<I", data, PAMLOD_GEOM_OFF)[0]
    if lod_count == 0 or geom_off == 0 or geom_off >= len(data):
        return result

    result.bbox_min = struct.unpack_from("<fff", data, PAMLOD_BBOX_MIN)
    result.bbox_max = struct.unpack_from("<fff", data, PAMLOD_BBOX_MAX)
    bmin, bmax = result.bbox_min, result.bbox_max

    # Locate LOD entries by scanning for .dds texture strings
    entries = []
    search_region = data[PAMLOD_ENTRY_TABLE:geom_off]
    for m in re.finditer(rb"[^\x00]{1,255}\.dds\x00", search_region):
        tex_start = PAMLOD_ENTRY_TABLE + m.start()
        nv_off = tex_start - 0x10
        if nv_off < PAMLOD_ENTRY_TABLE:
            continue
        nv = struct.unpack_from("<I", data, nv_off)[0]
        ni = struct.unpack_from("<I", data, nv_off + 4)[0]
        if not (1 <= nv <= 131072 and ni > 0 and ni % 3 == 0):
            continue
        voff = struct.unpack_from("<I", data, tex_start - 0x08)[0]
        ioff = struct.unpack_from("<I", data, tex_start - 0x04)[0]
        tex = data[tex_start:tex_start + 256].split(b"\x00")[0].decode("ascii", "replace")
        mat_start = tex_start + 0x100
        mat = data[mat_start:mat_start + 256].split(b"\x00")[0].decode("ascii", "replace") if mat_start < geom_off else ""
        entries.append({"nv": nv, "ni": ni, "voff": voff, "ioff": ioff,
                        "tex_start": tex_start, "tex": tex, "mat": mat})

    entries.sort(key=lambda e: e["tex_start"])

    # Group into LOD levels
    lod_groups = []
    cur_group, ve_acc, ie_acc = [], 0, 0
    for e in entries:
        if e["voff"] == ve_acc and e["ioff"] == ie_acc:
            cur_group.append(e)
            ve_acc += e["nv"]
            ie_acc += e["ni"]
        else:
            if cur_group:
                lod_groups.append(cur_group)
            cur_group = [e]
            ve_acc = e["nv"]
            ie_acc = e["ni"]
    if cur_group:
        lod_groups.append(cur_group)
    lod_groups = lod_groups[:lod_count]

    if not lod_groups:
        return result

    # Parse each LOD level
    cur = geom_off
    for lod_i, group in enumerate(lod_groups):
        total_nv = sum(e["nv"] for e in group)
        total_ni = sum(e["ni"] for e in group)

        # Find stride with padding scan
        found_base = found_stride = found_idx_off = None
        for pad in range(0, 64, 2):
            base = cur + pad
            for stride in STRIDE_CANDIDATES:
                cand = base + total_nv * stride
                if cand + total_ni * 2 > len(data):
                    continue
                if all(struct.unpack_from("<H", data, cand + j * 2)[0] < total_nv
                       for j in range(min(total_ni, 100))):
                    found_base = base
                    found_stride = stride
                    found_idx_off = cand
                    break
            if found_base is not None:
                break

        if found_base is None:
            result.lod_levels.append([])
            cur += 2
            continue

        # Parse submeshes for this LOD
        lod_submeshes = []
        vert_offset = 0
        has_uv = found_stride >= 12

        all_verts, all_uvs, all_faces = [], [], []
        for e in group:
            nv_e, ni_e = e["nv"], e["ni"]
            vert_base_e = found_base + e["voff"] * found_stride
            idx_off_e = found_idx_off + e["ioff"] * 2

            indices = [struct.unpack_from("<H", data, idx_off_e + j * 2)[0] for j in range(ni_e)]
            unique = sorted(set(indices))
            idx_map = {gi: li + vert_offset for li, gi in enumerate(unique)}

            for gi in unique:
                foff = vert_base_e + gi * found_stride
                if foff + 6 > len(data):
                    break
                xu, yu, zu = struct.unpack_from("<HHH", data, foff)
                all_verts.append((_dequant_u16(xu, bmin[0], bmax[0]),
                                  _dequant_u16(yu, bmin[1], bmax[1]),
                                  _dequant_u16(zu, bmin[2], bmax[2])))
                if has_uv and foff + 12 <= len(data):
                    u = struct.unpack_from("<e", data, foff + 8)[0]
                    v = struct.unpack_from("<e", data, foff + 10)[0]
                    all_uvs.append((u, v))

            for j in range(0, ni_e - 2, 3):
                a, b, c = indices[j], indices[j + 1], indices[j + 2]
                if a in idx_map and b in idx_map and c in idx_map:
                    all_faces.append((idx_map[a], idx_map[b], idx_map[c]))

            vert_offset += len(unique)

        mat_name = group[0]["mat"] or f"lod{lod_i}"
        sm = SubMesh(
            name=f"lod{lod_i:02d}_{mat_name}",
            material=mat_name,
            texture=group[0]["tex"],
            vertices=all_verts, uvs=all_uvs, faces=all_faces,
            normals=_compute_smooth_normals(all_verts, all_faces),
            vertex_count=len(all_verts), face_count=len(all_faces),
        )
        lod_submeshes.append(sm)
        result.lod_levels.append(lod_submeshes)
        cur = found_idx_off + total_ni * 2

    # Use requested LOD level as the main submeshes
    if lod_level < len(result.lod_levels) and result.lod_levels[lod_level]:
        result.submeshes = result.lod_levels[lod_level]
    elif result.lod_levels:
        # Fallback to first non-empty LOD
        for lod in result.lod_levels:
            if lod:
                result.submeshes = lod
                break

    result.total_vertices = sum(len(sm.vertices) for sm in result.submeshes)
    result.total_faces = sum(len(sm.faces) for sm in result.submeshes)
    result.has_uvs = any(sm.uvs for sm in result.submeshes)

    logger.info("Parsed PAMLOD %s: %d LODs, using LOD %d (%d verts, %d faces)",
                filename, len(result.lod_levels), lod_level,
                result.total_vertices, result.total_faces)
    return result


# ── PAC Parser (skinned mesh) ────────────────────────────────────────

def parse_pac(data: bytes, filename: str = "") -> ParsedMesh:
    """Parse a .pac skinned character mesh (PAR v9.3.1 format).

    PAC format (reverse-engineered):
      Header: 80 bytes
        [0x00] 4B: 'PAR ' magic
        [0x04] 4B: version (0x01000903 = v9.3.1)
        [0x08] 8B: timestamp/hash
        [0x10] 4B: always 0
        [0x14] 8B: section 0 size (metadata + vertex positions + bone weights)
        [0x1C] 8B: section 1 size
        [0x24] 8B: section 2 size
        [0x2C] 8B: section 3 size
        [0x34] 8B: section 4 size (index buffer for triangle strips)

      Section 0 (at offset 80):
        - Section offset table
        - Asset name strings
        - Vertex data: stream of [float3 position] [float3 normal] [bone_id]
          with FFFFFFFF separating vertices and FFFFFFFE in a second pass
        - Bone transform data

      Section 4 (last section):
        - Index buffer: triangle strips with 0xFFFF restart markers

    Vertices are stored as float32 triplets in section 0, interleaved with
    bone indices (uint32 < 500) and terminated by 0xFFFFFFFF markers.
    """
    if len(data) < 0x50 or data[:4] != PAR_MAGIC:
        raise ValueError(f"Not a valid PAC file: bad magic {data[:4]!r}")

    result = ParsedMesh(path=filename, format="pac")

    # Read section sizes from header — try uint64 first, then uint32
    sec_sizes = []
    header_size = 80

    # Try uint64 layout: 5 × 8 bytes at 0x14-0x3B
    for off in range(0x14, 0x3C, 8):
        sec_sizes.append(struct.unpack_from("<Q", data, off)[0])
    total_sec = sum(sec_sizes)

    if abs(total_sec + header_size - len(data)) > 100:
        # Try uint32 layout: 5 × 4 bytes at 0x14-0x27, header = 40 bytes
        sec_sizes = []
        for off in range(0x14, 0x28, 4):
            sec_sizes.append(struct.unpack_from("<I", data, off)[0])
        total_sec = sum(sec_sizes)
        header_size = 40

    if abs(total_sec + header_size - len(data)) > 100:
        # Try 10 × uint32 at 0x14 (some files have more sections)
        sec_sizes = []
        for off in range(0x14, 0x3C, 4):
            sec_sizes.append(struct.unpack_from("<I", data, off)[0])
        total_sec = sum(sec_sizes)
        header_size = 0x3C

    if abs(total_sec + header_size - len(data)) > 100:
        # Nothing matched — fall back to vertex scan on entire file
        logger.debug("PAC %s: section layout unknown, scanning full file", filename)
        sec_sizes = [len(data) - 0x50]
        header_size = 0x50

    sec0_off = header_size
    sec0_size = sec_sizes[0] if sec_sizes else len(data) - header_size

    # ── Extract vertex positions from section 0 ──
    # Scan for float3 triplets separated by FFFFFFFF terminators
    verts, bones_per_vert = _pac_extract_vertices(data, sec0_off, sec0_size)

    if not verts:
        logger.debug("PAC %s: no vertices found in section 0, trying PAM fallback", filename)
        return _pac_fallback_pam(data, filename)

    # Compute bounding box from extracted vertices
    if verts:
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        zs = [v[2] for v in verts]
        result.bbox_min = (min(xs), min(ys), min(zs))
        result.bbox_max = (max(xs), max(ys), max(zs))

    # ── Extract faces from the last section (triangle strips) ──
    last_sec_size = sec_sizes[-1]
    last_sec_off = len(data) - last_sec_size
    faces = _pac_extract_faces_from_strips(data, last_sec_off, last_sec_size, len(verts))

    # Find asset name for the submesh
    mat_name = _pac_extract_name(data, sec0_off, sec0_size) or Path(filename).stem

    sm = SubMesh(
        name=mat_name,
        material=mat_name,
        texture="",
        vertices=verts,
        uvs=[],
        faces=faces,
        bone_indices=bones_per_vert,
        bone_weights=[],
        normals=_compute_smooth_normals(verts, faces),
        vertex_count=len(verts),
        face_count=len(faces),
    )
    result.submeshes.append(sm)
    result.total_vertices = len(verts)
    result.total_faces = len(faces)
    result.has_bones = any(bones_per_vert)

    logger.info("Parsed PAC %s: %d verts, %d faces, %d bones_entries",
                filename, len(verts), len(faces), sum(len(b) for b in bones_per_vert))
    return result


def _pac_extract_vertices(data: bytes, sec_off: int, sec_size: int):
    """Extract float32 vertex positions from PAC section 0.

    Scans the section for float3 triplets interleaved with bone indices
    and FFFFFFFF terminators.  Returns (positions, bones_per_vertex).
    """
    end = sec_off + sec_size
    verts = []
    bones_per_vert = []

    # Find where vertex data starts by looking for the first valid float3
    # after the asset name strings (skip first ~100 bytes of metadata)
    scan_start = sec_off + 80  # skip offset table + names (approximate)

    # Scan for start of vertex float stream
    vertex_start = None
    for off in range(scan_start, min(end - 12, sec_off + 1000)):
        x, y, z = struct.unpack_from("<fff", data, off)
        if (not math.isnan(x) and not math.isnan(y) and not math.isnan(z) and
                abs(x) < 50 and abs(y) < 50 and abs(z) < 50 and
                (abs(x) + abs(y) + abs(z)) > 0.001):
            # Check if next 4 bytes are a bone_id or FFFFFFFF (valid successor)
            next4 = struct.unpack_from("<I", data, off + 12)[0]
            if next4 == 0xFFFFFFFF or next4 < 500:
                vertex_start = off
                break
            # Or another float3 follows
            nx, ny, nz = struct.unpack_from("<fff", data, off + 12)
            if abs(nx) < 50 and abs(ny) < 50 and abs(nz) < 50:
                vertex_start = off
                break

    if vertex_start is None:
        return [], []

    # Parse the vertex stream
    off = vertex_start
    current_pos = None
    current_bones = []
    seen_positions = {}  # (x,y,z) rounded → vertex index

    while off < end - 4:
        raw = struct.unpack_from("<I", data, off)[0]

        # FFFFFFFF = end of current vertex's bone chain
        if raw == 0xFFFFFFFF:
            if current_pos is not None:
                key = (round(current_pos[0], 5), round(current_pos[1], 5), round(current_pos[2], 5))
                if key not in seen_positions:
                    seen_positions[key] = len(verts)
                    verts.append(current_pos)
                    bones_per_vert.append(tuple(current_bones))
                current_pos = None
                current_bones = []
            off += 4
            continue

        # FFFFFFFE = second pass terminator (different semantic, treat like FFFFFFFF)
        if raw == 0xFFFFFFFE:
            if current_pos is not None:
                key = (round(current_pos[0], 5), round(current_pos[1], 5), round(current_pos[2], 5))
                if key not in seen_positions:
                    seen_positions[key] = len(verts)
                    verts.append(current_pos)
                    bones_per_vert.append(tuple(current_bones))
                current_pos = None
                current_bones = []
            off += 4
            continue

        # Small integer = bone index
        if raw < 500:
            current_bones.append(raw)
            off += 4
            continue

        # Also check uint16 bone index
        val16 = struct.unpack_from("<H", data, off)[0]
        if val16 < 500 and off + 2 <= end:
            # Peek ahead: does a valid float3 follow at +2?
            if off + 14 <= end:
                fx, fy, fz = struct.unpack_from("<fff", data, off + 2)
                if abs(fx) < 50 and abs(fy) < 50 and abs(fz) < 50 and not math.isnan(fx):
                    current_bones.append(val16)
                    off += 2
                    continue

        # Try reading as float3 position
        if off + 12 <= end:
            x, y, z = struct.unpack_from("<fff", data, off)
            if (not math.isnan(x) and not math.isnan(y) and not math.isnan(z) and
                    abs(x) < 50 and abs(y) < 50 and abs(z) < 50):
                # Valid position — this is either a new vertex or repeated position
                if current_pos is None or (
                    abs(x - current_pos[0]) > 0.0001 or
                    abs(y - current_pos[1]) > 0.0001 or
                    abs(z - current_pos[2]) > 0.0001
                ):
                    # New position — save previous if exists
                    if current_pos is not None:
                        key = (round(current_pos[0], 5), round(current_pos[1], 5), round(current_pos[2], 5))
                        if key not in seen_positions:
                            seen_positions[key] = len(verts)
                            verts.append(current_pos)
                            bones_per_vert.append(tuple(current_bones))
                        current_bones = []
                    current_pos = (x, y, z)
                # else: same position repeated for another bone assignment, skip
                off += 12
                continue

        # Unknown data — skip 4 bytes
        off += 4

    # Save last vertex
    if current_pos is not None:
        key = (round(current_pos[0], 5), round(current_pos[1], 5), round(current_pos[2], 5))
        if key not in seen_positions:
            verts.append(current_pos)
            bones_per_vert.append(tuple(current_bones))

    return verts, bones_per_vert


def _pac_extract_faces_from_strips(data: bytes, idx_off: int, idx_size: int, vert_count: int):
    """Extract triangle faces from PAC triangle strip index buffer.

    PAC uses triangle strips with 0xFFFF as restart markers.
    """
    faces = []
    n_indices = idx_size // 2
    strip = []

    for j in range(n_indices):
        val = struct.unpack_from("<H", data, idx_off + j * 2)[0]
        if val == 0xFFFF:
            # Convert current strip to triangles
            for k in range(len(strip) - 2):
                a, b, c = strip[k], strip[k + 1], strip[k + 2]
                if a != b and b != c and a != c:  # degenerate triangle check
                    if a < vert_count and b < vert_count and c < vert_count:
                        if k % 2 == 0:
                            faces.append((a, b, c))
                        else:
                            faces.append((a, c, b))  # flip winding for odd triangles
            strip = []
        else:
            if val < vert_count:
                strip.append(val)

    # Process last strip
    for k in range(len(strip) - 2):
        a, b, c = strip[k], strip[k + 1], strip[k + 2]
        if a != b and b != c and a != c:
            if a < vert_count and b < vert_count and c < vert_count:
                if k % 2 == 0:
                    faces.append((a, b, c))
                else:
                    faces.append((a, c, b))

    return faces


def _pac_extract_name(data: bytes, sec_off: int, sec_size: int) -> str:
    """Extract asset name from PAC section 0."""
    # Names start around offset 0x78 in the section (after offset table)
    search = data[sec_off + 32:sec_off + min(sec_size, 256)]
    for prefix in [b"CD_", b"cd_"]:
        idx = search.find(prefix)
        if idx >= 0:
            start = sec_off + 32 + idx
            end = data.find(b"\x00", start, start + 128)
            if end > start:
                return data[start:end].decode("ascii", "replace")
    return ""


def _pac_fallback_pam(data: bytes, filename: str) -> ParsedMesh:
    """Fallback: try parsing PAC as PAM (works for some small PAC files)."""
    try:
        return parse_pam(data, filename)
    except Exception:
        return ParsedMesh(path=filename, format="pac")


# ── Auto-detect and parse ────────────────────────────────────────────

def parse_mesh(data: bytes, filename: str = "") -> ParsedMesh:
    """Auto-detect file type and parse accordingly."""
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pamlod":
        return parse_pamlod(data, filename)
    elif ext == ".pac":
        return parse_pac(data, filename)
    else:
        return parse_pam(data, filename)


def is_mesh_file(path: str) -> bool:
    """Check if a file path is a supported mesh format."""
    ext = os.path.splitext(path.lower())[1]
    return ext in (".pam", ".pamlod", ".pac")
