from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

# Try pupil_apriltags first (often provides prebuilt wheels), else fallback to apriltag
try:
    from pupil_apriltags import Detector as PupilDetector  # type: ignore
except Exception:
    PupilDetector = None  # type: ignore

try:
    import apriltag  # type: ignore
except Exception:
    apriltag = None  # type: ignore


class AprilTagDetector:
    """Wrapper around AprilTag detectors with sane defaults.

    Prefers pupil_apriltags if available; otherwise uses apriltag.
    """

    def __init__(self) -> None:
        self.backend = None
        if PupilDetector is not None:
            # pupil_apriltags API
            self.backend = "pupil"
            self.detector = PupilDetector(families="tag36h11")
            return
        if apriltag is not None:
            # apriltag API
            self.backend = "apriltag"
            options = apriltag.DetectorOptions(families="tag36h11")
            self.detector = apriltag.Detector(options)
            return
        raise ImportError(
            "No AprilTag backend installed. Install either 'pupil-apriltags' (recommended) or 'apriltag'."
        )

    def detect_full(self, gray: np.ndarray) -> List[Dict]:
        """Return list of normalized detection dicts with id, center, and corners.

        Output schema per detection:
          {
            'tag_id': int,
            'center': (x_px: float, y_px: float),
            'corners': [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
          }
        """
        dets = self.detector.detect(gray)

        results: List[Dict] = []
        for d in dets:
            # Normalize across backends
            tag_id = getattr(d, "tag_id", None)
            if tag_id is None:
                tag_id = getattr(d, "id", None)
            if tag_id is None and isinstance(d, dict):
                tag_id = d.get("tag_id", d.get("id"))

            center = getattr(d, "center", None)
            if center is None and isinstance(d, dict):
                center = d.get("center")

            corners = getattr(d, "corners", None)
            if corners is None and isinstance(d, dict):
                corners = d.get("corners")

            if corners is None and center is not None:
                # Try inferring center-only detections — corners unavailable
                cx, cy = float(center[0]), float(center[1])
                results.append({
                    "tag_id": int(tag_id) if tag_id is not None else -1,
                    "center": (cx, cy),
                    "corners": None,
                })
                continue

            if tag_id is None or corners is None:
                continue

            corners_np = np.asarray(corners, dtype=float)
            if corners_np.shape != (4, 2):
                # Attempt to reshape if possible
                corners_np = corners_np.reshape(4, 2)
            if center is None:
                center_np = corners_np.mean(axis=0)
            else:
                center_np = np.asarray(center, dtype=float)

            results.append({
                "tag_id": int(tag_id),
                "center": (float(center_np[0]), float(center_np[1])),
                "corners": [(float(x), float(y)) for x, y in corners_np.tolist()],
            })
        return results


def iterate_detections_full(cap: cv2.VideoCapture, fps: float) -> Iterable[Tuple[int, float, List[Dict]]]:
    """Yield (frame_idx, t_sec, detection_dicts) for each frame with corners."""
    detector = AprilTagDetector()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector.detect_full(gray)
        t_sec = frame_idx / float(fps) if fps else 0.0
        yield frame_idx, t_sec, dets
        frame_idx += 1


# Backward-compatible centers-only iterator used elsewhere in the codebase
def iterate_detections(cap: cv2.VideoCapture, fps: float) -> Iterable[Tuple[int, float, List[Tuple[int, float, float]]]]:
    detector = AprilTagDetector()
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets_full = detector.detect_full(gray)
        centers: List[Tuple[int, float, float]] = []
        for d in dets_full:
            if d.get("center") is None:
                continue
            tag_id = int(d["tag_id"]) if d.get("tag_id") is not None else -1
            cx, cy = d["center"]
            centers.append((tag_id, float(cx), float(cy)))
        t_sec = frame_idx / float(fps) if fps else 0.0
        yield frame_idx, t_sec, centers
        frame_idx += 1


