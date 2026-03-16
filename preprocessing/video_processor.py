"""
video_processor.py — Stage 1: Video Preprocessing Pipeline

Responsibilities:
  • Decode video to per-frame NumPy arrays
  • Normalize frame rate to TARGET_FPS
  • Apply CLAHE contrast enhancement
  • Background subtraction via MOG2
  • Crop / pad frames to subject bounding box
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Generator

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TARGET_FPS,
    TARGET_WIDTH,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE,
    BG_SUBTRACTOR_HISTORY,
    BG_SUBTRACTOR_THRESHOLD,
    SUPPORTED_VIDEO_EXTENSIONS,
)


class VideoProcessor:
    """Handles all video preprocessing for the PGSI pipeline."""

    def __init__(self, video_path: str):
        ext = os.path.splitext(video_path)[1].lower()
        if ext not in SUPPORTED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{ext}'. Supported: {SUPPORTED_VIDEO_EXTENSIONS}"
            )
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # CLAHE for contrast enhancement
        self._clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE
        )

        # Background subtractor
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=BG_SUBTRACTOR_HISTORY,
            varThreshold=BG_SUBTRACTOR_THRESHOLD,
            detectShadows=True,
        )

    # ── public API ──────────────────────────────────────

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.original_fps if self.original_fps else 0.0

    def extract_frames(self, normalize_fps: bool = True) -> List[np.ndarray]:
        """Return all frames as a list of BGR NumPy arrays, optionally
        resampled to TARGET_FPS."""
        frames: List[np.ndarray] = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        if normalize_fps and self.original_fps != TARGET_FPS:
            frame_interval = self.original_fps / TARGET_FPS
            next_target = 0.0
            idx = 0
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                if idx >= int(next_target):
                    frames.append(frame)
                    next_target += frame_interval
                idx += 1
        else:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frames.append(frame)

        return frames

    def frame_generator(self) -> Generator[np.ndarray, None, None]:
        """Yield frames one at a time (memory-efficient)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize keeping aspect ratio, width = TARGET_WIDTH."""
        h, w = frame.shape[:2]
        scale = TARGET_WIDTH / w
        new_h = int(h * scale)
        return cv2.resize(frame, (TARGET_WIDTH, new_h), interpolation=cv2.INTER_AREA)

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement on the luminance channel."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def subtract_background(self, frame: np.ndarray) -> np.ndarray:
        """Return foreground mask using MOG2 background subtraction."""
        fg_mask = self._bg_sub.apply(frame)
        # Remove shadows (shadow pixels == 127 in MOG2)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        return fg_mask

    def get_subject_bbox(self, fg_mask: np.ndarray, margin: int = 20) -> Optional[Tuple[int, int, int, int]]:
        """Find bounding box of the largest foreground contour.
        Returns (x, y, w, h) or None if no subject detected."""
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        # Add margin
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(fg_mask.shape[1] - x, w + 2 * margin)
        h = min(fg_mask.shape[0] - y, h + 2 * margin)
        return (x, y, w, h)

    def crop_to_subject(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop frame to subject bounding box."""
        x, y, w, h = bbox
        return frame[y : y + h, x : x + w]

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Full single-frame preprocessing: resize → CLAHE → return."""
        frame = self.resize_frame(frame)
        frame = self.apply_clahe(frame)
        return frame

    def preprocess_all(self, use_bg_subtraction: bool = False) -> List[np.ndarray]:
        """Run full preprocessing pipeline on all frames.
        Returns list of preprocessed BGR frames at TARGET_FPS.

        Args:
            use_bg_subtraction: If True, apply MOG2 background subtraction
                and crop frames to the detected subject bounding box.
        """
        raw_frames = self.extract_frames(normalize_fps=True)
        processed: List[np.ndarray] = []
        for frame in raw_frames:
            frame = self.preprocess_frame(frame)
            if use_bg_subtraction:
                fg_mask = self.subtract_background(frame)
                bbox = self.get_subject_bbox(fg_mask)
                if bbox is not None:
                    frame = self.crop_to_subject(frame, bbox)
            processed.append(frame)
        return processed

    def release(self):
        """Release the video capture resource."""
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.release()

    def __repr__(self) -> str:
        return (
            f"VideoProcessor(path='{self.video_path}', "
            f"fps={self.original_fps}, frames={self.frame_count}, "
            f"size={self.width}x{self.height})"
        )
