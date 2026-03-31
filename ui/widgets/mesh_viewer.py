"""3D mesh preview widget using QPainter (no OpenGL dependency).

Renders a wireframe + solid preview of ParsedMesh objects with
mouse rotation and zoom. Uses software rendering via QPainter
for maximum compatibility — works on any system without GPU drivers.
"""

import math
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QBrush, QWheelEvent, QMouseEvent


class MeshViewer(QWidget):
    """3D mesh preview with rotation and zoom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._vertices = []  # list of (x, y, z)
        self._faces = []     # list of (a, b, c)
        self._normals = []   # list of (nx, ny, nz) per face
        self._rot_x = -25.0
        self._rot_y = 35.0
        self._zoom = 1.0
        self._last_mouse = None
        self._center = (0, 0, 0)
        self._scale = 1.0
        self._info_text = ""
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    def set_mesh(self, vertices, faces, info_text=""):
        """Load mesh data for rendering."""
        self._vertices = list(vertices)
        self._faces = list(faces)
        self._info_text = info_text

        if not self._vertices:
            self.update()
            return

        # Compute bounding box and center
        xs = [v[0] for v in self._vertices]
        ys = [v[1] for v in self._vertices]
        zs = [v[2] for v in self._vertices]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)

        self._center = (
            (min_x + max_x) / 2,
            (min_y + max_y) / 2,
            (min_z + max_z) / 2,
        )

        # Scale to fit in view
        extent = max(max_x - min_x, max_y - min_y, max_z - min_z, 0.001)
        self._scale = 1.0 / extent
        self._zoom = 1.0

        # Compute per-face normals for basic shading
        self._normals = []
        for a, b, c in self._faces:
            if a < len(self._vertices) and b < len(self._vertices) and c < len(self._vertices):
                v0, v1, v2 = self._vertices[a], self._vertices[b], self._vertices[c]
                nx = (v1[1] - v0[1]) * (v2[2] - v0[2]) - (v1[2] - v0[2]) * (v2[1] - v0[1])
                ny = (v1[2] - v0[2]) * (v2[0] - v0[0]) - (v1[0] - v0[0]) * (v2[2] - v0[2])
                nz = (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v1[1] - v0[1]) * (v2[0] - v0[0])
                length = math.sqrt(nx * nx + ny * ny + nz * nz)
                if length > 1e-8:
                    self._normals.append((nx / length, ny / length, nz / length))
                else:
                    self._normals.append((0, 1, 0))
            else:
                self._normals.append((0, 1, 0))

        self.update()

    def clear(self):
        self._vertices = []
        self._faces = []
        self._normals = []
        self._info_text = ""
        self.update()

    def _project(self, x, y, z):
        """Project 3D point to 2D screen space with rotation."""
        # Center
        x -= self._center[0]
        y -= self._center[1]
        z -= self._center[2]

        # Scale
        s = self._scale * self._zoom * min(self.width(), self.height()) * 0.35
        x *= s
        y *= s
        z *= s

        # Rotate Y
        ry = math.radians(self._rot_y)
        cos_y, sin_y = math.cos(ry), math.sin(ry)
        x2 = x * cos_y + z * sin_y
        z2 = -x * sin_y + z * cos_y

        # Rotate X
        rx = math.radians(self._rot_x)
        cos_x, sin_x = math.cos(rx), math.sin(rx)
        y2 = y * cos_x - z2 * sin_x
        z3 = y * sin_x + z2 * cos_x

        # Orthographic projection
        cx = self.width() / 2
        cy = self.height() / 2
        return QPointF(cx + x2, cy - y2), z3

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Background
        p.fillRect(self.rect(), QColor(24, 24, 37))

        if not self._vertices or not self._faces:
            p.setPen(QColor(108, 112, 134))
            p.drawText(self.rect(), Qt.AlignCenter, self._info_text or "No mesh loaded")
            p.end()
            return

        # Light direction (fixed, from top-right-front)
        light = (0.3, 0.7, 0.5)
        ln = math.sqrt(sum(l * l for l in light))
        light = tuple(l / ln for l in light)

        # Rotate light with view so shading is consistent
        ry = math.radians(self._rot_y)
        rx = math.radians(self._rot_x)

        # Sort faces by depth (painter's algorithm)
        face_depths = []
        for i, (a, b, c) in enumerate(self._faces):
            if a >= len(self._vertices) or b >= len(self._vertices) or c >= len(self._vertices):
                continue
            v0, v1, v2 = self._vertices[a], self._vertices[b], self._vertices[c]
            _, z0 = self._project(*v0)
            _, z1 = self._project(*v1)
            _, z2 = self._project(*v2)
            avg_z = (z0 + z1 + z2) / 3
            face_depths.append((avg_z, i, a, b, c))

        face_depths.sort(key=lambda f: f[0])  # back to front

        # Draw filled faces with basic shading
        for avg_z, fi, a, b, c in face_depths:
            p0, _ = self._project(*self._vertices[a])
            p1, _ = self._project(*self._vertices[b])
            p2, _ = self._project(*self._vertices[c])

            # Compute shading from face normal
            if fi < len(self._normals):
                n = self._normals[fi]
                dot = max(0.15, n[0] * light[0] + n[1] * light[1] + n[2] * light[2])
            else:
                dot = 0.5

            # Base color: steel blue with shading
            r = int(min(255, 80 + 100 * dot))
            g = int(min(255, 120 + 80 * dot))
            b_col = int(min(255, 180 + 60 * dot))

            p.setBrush(QBrush(QColor(r, g, b_col, 220)))
            p.setPen(QPen(QColor(40, 42, 54), 0.5))

            from PySide6.QtGui import QPolygonF
            poly = QPolygonF([p0, p1, p2])
            p.drawPolygon(poly)

        # Info text overlay
        p.setPen(QColor(166, 173, 200))
        p.drawText(8, 16, self._info_text)
        p.setPen(QColor(108, 112, 134))
        p.drawText(8, self.height() - 8, "Drag to rotate | Scroll to zoom")

        p.end()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self._last_mouse = event.position()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._last_mouse and event.buttons() & Qt.LeftButton:
            dx = event.position().x() - self._last_mouse.x()
            dy = event.position().y() - self._last_mouse.y()
            self._rot_y += dx * 0.5
            self._rot_x += dy * 0.5
            self._rot_x = max(-90, min(90, self._rot_x))
            self._last_mouse = event.position()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._last_mouse = None

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom *= 1.15
        else:
            self._zoom /= 1.15
        self._zoom = max(0.1, min(20.0, self._zoom))
        self.update()
