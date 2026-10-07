import unittest

from eye_mouse.filters import OneEuroFilter
from eye_mouse.gestures import DwellClicker, EyeState, WinkDetector, eye_aspect_ratio
from eye_mouse.mapping import Box, Calibration, ScreenMapper


class FilterTest(unittest.TestCase):
    def test_steady_input_settles(self):
        f = OneEuroFilter()
        out = [f(100.0, i / 30) for i in range(30)]
        self.assertAlmostEqual(out[-1], 100.0)

    def test_jitter_is_reduced(self):
        f = OneEuroFilter(min_cutoff=0.5)
        noisy = [100 + (5 if i % 2 else -5) for i in range(60)]
        out = [f(x, i / 30) for i, x in enumerate(noisy)]
        self.assertLess(max(out[30:]) - min(out[30:]), 10 * 0.5)


class MapperTest(unittest.TestCase):
    def test_box_corners_map_to_screen_corners(self):
        m = ScreenMapper(1920, 1080, Box(0.4, 0.4, 0.6, 0.6))
        self.assertEqual(m.to_screen(0.4, 0.4), (1, 1))
        self.assertEqual(m.to_screen(0.6, 0.6), (1918, 1078))
        self.assertEqual(m.to_screen(0.5, 0.5), (960, 540))

    def test_outside_box_is_clamped_off_the_failsafe_corner(self):
        m = ScreenMapper(1920, 1080)
        self.assertEqual(m.to_screen(0.0, 0.0), (1, 1))

    def test_calibration_builds_a_box_from_the_sweep(self):
        c = Calibration(margin=0)
        for i in range(100):
            c.add(0.3 + 0.4 * (i % 10) / 9, 0.35 + 0.3 * (i // 10) / 9)
        box = c.result()
        self.assertAlmostEqual(box.left, 0.3, delta=0.05)
        self.assertAlmostEqual(box.right, 0.7, delta=0.05)

    def test_calibration_without_movement_is_rejected(self):
        c = Calibration()
        for _ in range(50):
            c.add(0.5, 0.5)
        self.assertIsNone(c.result())


class GestureTest(unittest.TestCase):
    def test_eye_aspect_ratio(self):
        self.assertAlmostEqual(eye_aspect_ratio((5, 0), (5, 3), (0, 1), (10, 1)), 0.3)

    def test_eye_state_learns_baseline(self):
        eye = EyeState()
        for _ in range(20):
            eye.update(0.30)
        self.assertTrue(eye.is_closed(eye.update(0.10)))
        self.assertFalse(eye.is_closed(eye.update(0.29)))

    def test_left_wink_clicks_once(self):
        w = WinkDetector(wink_time=0.2)
        actions = [w.update(True, False, i / 30) for i in range(30)]
        kinds = [a.kind for a in actions if a]
        self.assertEqual(kinds, ["left_click"])

    def test_normal_blink_does_nothing(self):
        w = WinkDetector()
        frames = [(False, False)] * 5 + [(True, True)] * 4 + [(False, False)] * 5
        actions = [w.update(l, r, i / 30) for i, (l, r) in enumerate(frames)]
        self.assertFalse(any(actions))

    def test_long_close_toggles_pause(self):
        w = WinkDetector(pause_time=1.0)
        actions = [w.update(True, True, i / 30) for i in range(40)]
        self.assertEqual([a.kind for a in actions if a], ["toggle_pause"])

    def test_dwell_clicks_once_then_needs_movement(self):
        d = DwellClicker(dwell_time=1.0, radius=20)
        clicks = sum(d.update(500, 500, i / 30) for i in range(90))
        self.assertEqual(clicks, 1)
        d.update(600, 600, 3.1)
        clicks = sum(d.update(600, 600, 3.1 + i / 30) for i in range(40))
        self.assertEqual(clicks, 1)


if __name__ == "__main__":
    unittest.main()
