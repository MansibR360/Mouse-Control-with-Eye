"""Turns eye openness and cursor stillness into clicks.

- Wink left eye   -> left click
- Wink right eye  -> right click
- Close both eyes for a moment -> pause / resume tracking
- Dwell (hold the cursor still) -> left click, for users who can't wink

Normal blinks close both eyes briefly, so they never trigger anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


def eye_aspect_ratio(top, bottom, inner, outer) -> float:
    """Eye height over eye width. Roughly 0.25–0.35 open and below 0.12 closed,
    but it varies by person, so ``EyeState`` compares against each user's
    own baseline."""
    width = math.dist(inner, outer)
    return math.dist(top, bottom) / width if width else 0.0


class EyeState:
    """Tracks one eye's openness relative to its own learned baseline."""

    def __init__(self, closed_ratio: float = 0.55) -> None:
        self.closed_ratio = closed_ratio
        self.baseline: Optional[float] = None

    def update(self, ear: float) -> float:
        """Returns openness: about 1.0 when normally open, near 0 when closed."""
        if self.baseline is None:
            self.baseline = ear
        openness = ear / self.baseline if self.baseline else 1.0
        # Learn the open-eye baseline slowly, and only from open-looking frames.
        if openness > 0.8:
            self.baseline = 0.98 * self.baseline + 0.02 * ear
        return openness

    def is_closed(self, openness: float) -> bool:
        return openness < self.closed_ratio


@dataclass
class Action:
    kind: str  # "left_click", "right_click", "toggle_pause"


class WinkDetector:
    def __init__(self, wink_time: float = 0.22, pause_time: float = 1.0) -> None:
        self.wink_time = wink_time
        self.pause_time = pause_time
        self._state = "open"  # open | left | right | both
        self._since = 0.0
        self._fired = False

    def update(self, left_closed: bool, right_closed: bool, t: float) -> Optional[Action]:
        if left_closed and right_closed:
            state = "both"
        elif left_closed:
            state = "left"
        elif right_closed:
            state = "right"
        else:
            state = "open"

        if state != self._state:
            self._state, self._since, self._fired = state, t, False
            return None
        if self._fired or state == "open":
            return None

        held = t - self._since
        if state == "both" and held >= self.pause_time:
            self._fired = True
            return Action("toggle_pause")
        if state in ("left", "right") and held >= self.wink_time:
            self._fired = True
            return Action("left_click" if state == "left" else "right_click")
        return None

    @property
    def eyes_moving(self) -> bool:
        """True while any eye is closing. Eyelids shift the iris landmarks,
        so the cursor is frozen meanwhile to keep the click on target."""
        return self._state != "open"


class DwellClicker:
    """Clicks when the cursor stays within ``radius`` pixels for ``dwell_time``."""

    def __init__(self, dwell_time: float = 1.2, radius: float = 30.0) -> None:
        self.dwell_time = dwell_time
        self.radius = radius
        self._anchor: Optional[tuple[float, float]] = None
        self._since = 0.0
        self._armed = True
        self.progress = 0.0  # 0..1, for the on-screen ring

    def update(self, x: float, y: float, t: float) -> bool:
        if self._anchor is None or math.dist(self._anchor, (x, y)) > self.radius:
            self._anchor, self._since, self._armed = (x, y), t, True
            self.progress = 0.0
            return False
        if not self._armed:
            return False
        self.progress = min((t - self._since) / self.dwell_time, 1.0)
        if self.progress >= 1.0:
            self._armed = False  # move away before the next dwell click
            self.progress = 0.0
            return True
        return False
