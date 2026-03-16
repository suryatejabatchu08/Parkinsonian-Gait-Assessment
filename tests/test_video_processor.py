"""
test_video_processor.py — Unit tests for video preprocessing.
"""

import sys
import os
import numpy as np
import cv2
import pytest
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.video_processor import VideoProcessor


@pytest.fixture
def sample_video_path():
    """Create a minimal test video (10 frames of colored noise)."""
    path = os.path.join(tempfile.gettempdir(), "test_pgsi_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 30, (320, 240))
    for i in range(30):
        frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        # Draw a white rectangle as a "person" for background subtraction testing
        cv2.rectangle(frame, (100 + i, 50), (220 + i, 220), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    yield path
    if os.path.exists(path):
        os.remove(path)


class TestVideoProcessor:

    def test_init(self, sample_video_path):
        vp = VideoProcessor(sample_video_path)
        assert vp.frame_count == 30
        assert vp.width == 320
        assert vp.height == 240
        vp.release()

    def test_unsupported_format(self, tmp_path):
        fake_path = str(tmp_path / "video.xyz")
        with open(fake_path, "w") as f:
            f.write("not a video")
        with pytest.raises(ValueError, match="Unsupported format"):
            VideoProcessor(fake_path)

    def test_extract_frames(self, sample_video_path):
        vp = VideoProcessor(sample_video_path)
        frames = vp.extract_frames(normalize_fps=False)
        assert len(frames) == 30
        assert frames[0].shape == (240, 320, 3)
        vp.release()

    def test_resize_frame(self, sample_video_path):
        vp = VideoProcessor(sample_video_path)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        resized = vp.resize_frame(frame)
        assert resized.shape[1] == 640  # TARGET_WIDTH
        vp.release()

    def test_clahe(self, sample_video_path):
        vp = VideoProcessor(sample_video_path)
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        enhanced = vp.apply_clahe(frame)
        assert enhanced.shape == frame.shape
        vp.release()

    def test_background_subtraction(self, sample_video_path):
        vp = VideoProcessor(sample_video_path)
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mask = vp.subtract_background(frame)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        vp.release()

    def test_preprocess_all(self, sample_video_path):
        vp = VideoProcessor(sample_video_path)
        frames = vp.preprocess_all()
        assert len(frames) > 0
        assert frames[0].ndim == 3
        vp.release()

    def test_duration(self, sample_video_path):
        vp = VideoProcessor(sample_video_path)
        assert vp.duration_seconds == pytest.approx(1.0, abs=0.1)
        vp.release()
