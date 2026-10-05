import queue
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from app import Camera


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.camera = Camera(0, queue.Queue())

    def test_single_image_snapshot_uses_the_current_frame(self):
        self.camera.latest = Image.new("RGB", (640, 480), "blue")
        self.camera.samples.append((40, Image.new("RGB", (640, 480), "red")))
        with patch("app.time.monotonic", return_value=42):
            images = self.camera.snapshot(1)
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0].getpixel((0, 0)), (0, 0, 255))
        images[0].putpixel((0, 0), (255, 255, 255))
        self.assertEqual(self.camera.latest.getpixel((0, 0)), (0, 0, 255))

    def test_multiple_images_are_recent_and_chronological(self):
        self.camera.latest = Image.new("RGB", (10, 10), "blue")
        self.camera.samples.extend(
            [
                (20, Image.new("RGB", (10, 10), "red")),
                (40, Image.new("RGB", (10, 10), "green")),
                (41, Image.new("RGB", (10, 10), "yellow")),
            ]
        )
        with patch("app.time.monotonic", return_value=42):
            images = self.camera.snapshot(3)
        self.assertEqual(
            [image.getpixel((0, 0)) for image in images],
            [(0, 128, 0), (255, 255, 0), (0, 0, 255)],
        )

    def test_fps_uses_actual_capture_times_and_expires_when_capture_stops(self):
        self.camera.frame_times.extend([9, 9.5, 10])
        with patch("app.time.monotonic", return_value=10):
            self.assertEqual(self.camera.fps(), 2)
        with patch("app.time.monotonic", return_value=12):
            self.assertEqual(self.camera.fps(), 0)

    def test_snapshot_preserves_aspect_ratio_and_limits_resolution(self):
        self.camera.latest = Image.new("RGB", (1920, 1080))
        self.assertEqual(self.camera.snapshot(1)[0].size, (640, 360))
        self.assertEqual(self.camera.latest.size, (1920, 1080))

    def test_capture_preserves_full_hd_and_requests_25fps_but_inference_stays_small(self):
        frame = np.empty((1080, 1920, 3), dtype=np.uint8)
        frame[:] = (10, 20, 30)

        class SimulatedCapture:
            released = False
            reads = 0
            settings = {}

            def isOpened(self):
                return True

            def set(self, prop, value):
                self.settings[prop] = value
                return True

            def get(self, prop):
                return self.settings.get(prop, 0)

            def read(self):
                self.reads += 1
                return (True, frame) if self.reads == 1 else (False, None)

            def release(self):
                self.released = True

        capture = SimulatedCapture()
        with patch("app.cv2.VideoCapture", return_value=capture):
            self.camera.thread.start()
            try:
                self.camera.thread.join(timeout=3)
                self.assertFalse(self.camera.thread.is_alive())
            finally:
                self.camera.stop.set()
                self.camera.thread.join(timeout=3)
        self.assertTrue(capture.released)
        self.assertEqual(self.camera.latest.size, (1920, 1080))
        self.assertEqual(self.camera.snapshot(1)[0].size, (640, 360))
        self.assertEqual(capture.settings[3], 1920)
        self.assertEqual(capture.settings[4], 1080)
        self.assertEqual(capture.settings[5], 25)
        self.assertEqual(self.camera.latest.getpixel((0, 0)), (30, 20, 10))
        events = []
        while not self.camera.events.empty():
            events.append(self.camera.events.get_nowait()[0])
        self.assertEqual(events, ["camera_ready", "camera_error"])

    def test_driver_fallback_reports_real_frame_size_not_requested_size(self):
        from unittest.mock import Mock

        import cv2

        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: 30 if prop == cv2.CAP_PROP_FPS else 0

        def read():
            self.camera.stop.set()
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        capture.read.side_effect = read
        with patch("app.cv2.VideoCapture", return_value=capture):
            self.camera._capture()
        kind, _, settings = self.camera.events.get_nowait()
        self.assertEqual(kind, "camera_ready")
        self.assertEqual(settings["requested"], (1920, 1080, 25))
        self.assertEqual((settings["width"], settings["height"], settings["fps"]), (640, 480, 30))
        capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
