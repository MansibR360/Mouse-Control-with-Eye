"""Hands-free mouse: webcam in, cursor and clicks out."""

from __future__ import annotations

import argparse
import time

from . import __version__
from .filters import PointFilter
from .gestures import DwellClicker, EyeState, WinkDetector
from .mapping import Calibration, ScreenMapper

CALIBRATION_SECONDS = 4.0
GREEN, AMBER, RED, WHITE = (80, 200, 120), (0, 190, 255), (60, 60, 230), (240, 240, 240)

HELP = "Wink L/R: click   Close eyes 1s: pause   C: calibrate   D: dwell   P: pause   Q: quit"


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="eye-mouse", description="Control the mouse with your eyes and face.")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    p.add_argument("--pointer", choices=("iris", "nose"), default="iris",
                   help="what drives the cursor: iris position or nose tip (steadier)")
    p.add_argument("--smoothing", type=float, default=1.0,
                   help="lower is smoother but laggier (One Euro min cutoff, default: 1.0)")
    p.add_argument("--dwell", action="store_true", help="start with dwell clicking on")
    p.add_argument("--dwell-time", type=float, default=1.2, help="seconds to hold still for a dwell click")
    p.add_argument("--no-clicks", action="store_true", help="move the cursor only; ignore winks")
    p.add_argument("--no-preview", action="store_true", help="don't show the camera window")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> None:
    import cv2 as cv
    import pyautogui as pg

    from .tracker import FaceTracker

    pg.PAUSE = 0  # PyAutoGUI sleeps 0.1 s after every call by default
    screen_w, screen_h = pg.size()

    cam = cv.VideoCapture(args.camera)
    if not cam.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}. Try --camera 1.")

    tracker = FaceTracker(args.pointer)
    mapper = ScreenMapper(screen_w, screen_h)
    smooth = PointFilter(min_cutoff=args.smoothing, beta=0.02)
    left_eye, right_eye = EyeState(), EyeState()
    winks = WinkDetector()
    dweller = DwellClicker(args.dwell_time)

    paused = False
    dwell_on = args.dwell
    calibration: Calibration | None = None
    calibration_end = 0.0
    status, status_until = "Press C to calibrate", time.monotonic() + 5
    fps, last = 0.0, time.monotonic()

    def flash(message: str) -> None:
        nonlocal status, status_until
        status, status_until = message, time.monotonic() + 2

    print(f"Eye Mouse {__version__}: screen {screen_w}x{screen_h}, pointer={args.pointer}")
    print(HELP)

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            now = time.monotonic()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now

            frame = cv.flip(frame, 1)
            face = tracker.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

            if face:
                l_open = left_eye.update(face.left_ear)
                r_open = right_eye.update(face.right_ear)
                action = winks.update(left_eye.is_closed(l_open), right_eye.is_closed(r_open), now)

                if action and action.kind == "toggle_pause":
                    paused = not paused
                    smooth.reset()
                    flash("Paused" if paused else "Resumed")
                elif action and not paused and not args.no_clicks:
                    if action.kind == "left_click":
                        pg.click()
                        flash("Left click")
                    else:
                        pg.click(button="right")
                        flash("Right click")

                if calibration is not None:
                    calibration.add(*face.pointer)
                    if now >= calibration_end:
                        box = calibration.result()
                        if box:
                            mapper.box = box
                            flash("Calibrated")
                        else:
                            flash("Calibration needs more movement, press C to retry")
                        calibration = None
                elif not paused and not winks.eyes_moving:
                    sx, sy = mapper.to_screen(*face.pointer)
                    fx, fy = smooth(sx, sy, now)
                    pg.moveTo(fx, fy)
                    if dwell_on and not args.no_clicks and dweller.update(fx, fy, now):
                        pg.click()
                        flash("Dwell click")

            if not args.no_preview:
                draw_hud(cv, frame, face, mapper, paused, dwell_on, dweller.progress,
                         calibration_end - now if calibration else None, fps,
                         status if now < status_until else None)
                cv.imshow("Eye Mouse", frame)

            key = cv.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("p"):
                paused = not paused
                smooth.reset()
                flash("Paused" if paused else "Resumed")
            elif key == ord("d"):
                dwell_on = not dwell_on
                flash(f"Dwell click {'on' if dwell_on else 'off'}")
            elif key == ord("c"):
                calibration = Calibration()
                calibration_end = now + CALIBRATION_SECONDS
                smooth.reset()
    finally:
        tracker.close()
        cam.release()
        cv.destroyAllWindows()


def draw_hud(cv, frame, face, mapper, paused, dwell_on, dwell_progress, calib_left, fps, status) -> None:
    h, w = frame.shape[:2]
    b = mapper.box
    cv.rectangle(frame, (int(b.left * w), int(b.top * h)), (int(b.right * w), int(b.bottom * h)),
                 AMBER if calib_left is not None else (120, 120, 120), 1)
    if face:
        for p in face.iris_px:
            cv.circle(frame, p, 3, GREEN, -1)
        px, py = int(face.pointer[0] * w), int(face.pointer[1] * h)
        cv.drawMarker(frame, (px, py), WHITE, cv.MARKER_CROSS, 14, 1)
        if dwell_on and dwell_progress > 0:
            cv.ellipse(frame, (px, py), (18, 18), -90, 0, int(360 * dwell_progress), AMBER, 2)
    else:
        status = status or "No face detected"

    state, color = ("PAUSED", RED) if paused else ("TRACKING", GREEN)
    cv.rectangle(frame, (0, 0), (w, 30), (20, 20, 20), -1)
    cv.putText(frame, state, (10, 21), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv.putText(frame, f"dwell {'on' if dwell_on else 'off'}   {fps:4.0f} fps", (130, 21),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    if calib_left is not None:
        status = f"Calibrating: move to reach all screen edges ({calib_left:.0f}s)"
    if status:
        cv.putText(frame, status, (10, h - 14), cv.FONT_HERSHEY_SIMPLEX, 0.55, AMBER, 2)


def main(argv=None) -> None:
    try:
        run(parse_args(argv))
    except KeyboardInterrupt:
        pass
