# Eye Mouse

**A hands-free mouse that runs on any webcam.** Move the cursor with your eyes, wink to click, and hold still to dwell-click. No extra hardware.

Built for people who can't comfortably use a mouse (RSI, motor impairments, or hands that are busy), and for anyone curious what face tracking can do on a normal laptop.

## Features

- **Gaze and head pointing:** MediaPipe Face Mesh tracks 478 face landmarks, including the irises, in real time
- **Wink to click:** left wink for left click, right wink for right click. Normal blinks are ignored
- **Dwell click:** keep the cursor still for a moment to click, for users who can't wink reliably
- **Close both eyes to pause:** hold them closed for about 1 second to stop or resume tracking
- **Personal calibration:** press `C`, sweep your comfortable range, and that range is stretched over the whole screen
- **Jitter-free movement:** a One Euro filter smooths tiny tremors without making fast moves feel laggy
- **Per-user eye baseline:** blink detection adapts to your own eye shape instead of a fixed threshold
- **Steady clicks:** the cursor freezes while an eye is closing, so winking doesn't nudge it off target
- **Live HUD:** tracking state, active area, dwell progress ring and FPS

## How it works

```
webcam ─▶ OpenCV ─▶ MediaPipe Face Mesh ─┬─▶ iris / nose point ─▶ calibration box ─▶ One Euro filter ─▶ cursor
                                          └─▶ eye aspect ratio ─▶ wink / blink detector ──────────────────▶ clicks
```

Eye openness is measured as the **eye aspect ratio** (lid gap ÷ eye width) and compared with each eye's learned open baseline. A wink is one eye closed while the other stays open for at least 0.22 s. A natural blink closes both eyes, so it never clicks.

## Requirements

- Python 3.9–3.12 (MediaPipe doesn't publish wheels for newer Python yet)
- A webcam
- Windows, macOS or Linux. On macOS, give your terminal Camera and Accessibility permission

## Quick start

```bash
git clone https://github.com/MansibR360/Mouse-Control-with-Eye.git
cd Mouse-Control-with-Eye
pip install -r requirements.txt
python -m eye_mouse
```

1. Sit about an arm's length from the camera with your face lit from the front.
2. Press **C** and, over 4 seconds, move your eyes and head to reach all four screen edges.
3. Wink to click. Press **Q** to quit.

| Key | Action |
|---|---|
| `C` | Calibrate |
| `D` | Toggle dwell click |
| `P` | Pause or resume |
| `Q` / `Esc` | Quit |

**Emergency stop:** slam the real mouse into any screen corner (PyAutoGUI's fail-safe).

### Options

| Flag | Default | Description |
|---|---|---|
| `--pointer iris\|nose` | `iris` | Drive the cursor with the irises or the nose tip (steadier) |
| `--smoothing` | `1.0` | Lower is smoother, higher is snappier |
| `--dwell` | off | Start with dwell clicking enabled |
| `--dwell-time` | `1.2` | Seconds to hold still for a dwell click |
| `--no-clicks` | off | Move the cursor only |
| `--camera` | `0` | Webcam index |
| `--no-preview` | off | Hide the camera window |

## Project structure

```
eye_mouse/
├── tracker.py    # MediaPipe Face Mesh → pointer + eye ratios
├── mapping.py    # Active box → screen, calibration
├── filters.py    # One Euro filter
├── gestures.py   # Wink, blink-pause and dwell detection
└── app.py        # Camera loop, HUD, keyboard controls
tests/            # Filter, mapping and gesture tests (no camera needed)
```

## Development

```bash
python -m unittest discover -s tests -t .
```

## License

[MIT](LICENSE) © Mansib Yasir
