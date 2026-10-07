"""Maps a tracked point in the camera frame to a screen position."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Box:
    """A region of the camera frame, in normalized 0..1 coordinates."""

    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def centered(cls, width: float, height: float, cx: float = 0.5, cy: float = 0.5) -> "Box":
        return cls(cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)


class ScreenMapper:
    """Stretches a small "active box" of the camera frame over the whole
    screen, so small head and eye movements cover every corner.

    The box can be learned with ``Calibration``: the user sweeps through
    the area they can comfortably reach, and that becomes the box.
    """

    def __init__(self, screen_w: int, screen_h: int, box: Box | None = None) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.box = box or Box.centered(0.25, 0.20)

    def to_screen(self, nx: float, ny: float) -> tuple[int, int]:
        b = self.box
        u = (nx - b.left) / max(b.right - b.left, 1e-6)
        v = (ny - b.top) / max(b.bottom - b.top, 1e-6)
        u = min(max(u, 0.0), 1.0)
        v = min(max(v, 0.0), 1.0)
        # Stay one pixel off the edges: PyAutoGUI's fail-safe fires when
        # the cursor hits a corner, and that should stay a deliberate escape.
        x = 1 + u * (self.screen_w - 3)
        y = 1 + v * (self.screen_h - 3)
        return int(round(x)), int(round(y))


class Calibration:
    """Collects samples while the user sweeps their reach, then builds a box."""

    def __init__(self, margin: float = 0.08) -> None:
        self.margin = margin
        self.xs: list[float] = []
        self.ys: list[float] = []

    def add(self, nx: float, ny: float) -> None:
        self.xs.append(nx)
        self.ys.append(ny)

    def result(self) -> Box | None:
        if len(self.xs) < 15:
            return None
        xs, ys = sorted(self.xs), sorted(self.ys)
        # Trim the extreme 5% so one bad frame can't stretch the box.
        k = max(1, len(xs) // 20)
        left, right = xs[k], xs[-k - 1]
        top, bottom = ys[k], ys[-k - 1]
        if right - left < 0.02 or bottom - top < 0.02:
            return None  # the user barely moved; keep the old box
        # Shrink slightly so the screen edges are reachable without straining.
        mx, my = (right - left) * self.margin, (bottom - top) * self.margin
        return Box(left + mx, top + my, right - mx, bottom - my)
