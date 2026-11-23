"""The security camera's eyes: OpenCV motion detection, a small multi-object tracker, the annotated monitor
picture and the pixel-to-floor geometry. numpy and OpenCV only, no Unreal: it runs inside the game (the
`vision` dependency group) and in the unit tests alike.

    detector = MotionDetector()
    boxes = detector.detect(frame)          # BGR uint8 frame from the camera's render target
    tracks = tracker.update(boxes)          # stable ids, ages, a primary target
    picture = annotate(frame, tracks, ...)  # what the in-world monitor shows
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

Image = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in pixels: left, top, width, height."""

    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def foot(self) -> tuple[float, float]:
        """The bottom centre: where the object touches the floor."""
        return (self.x + self.w / 2.0, float(self.y + self.h))


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes (0 when they do not overlap)."""
    left, top = max(a.x, b.x), max(a.y, b.y)
    right, bottom = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0, right - left) * max(0, bottom - top)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def merge_boxes(boxes: Sequence[Box], gap: int = 4) -> list[Box]:
    """Merge boxes that overlap or nearly touch (a walker's body and head are one object)."""
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        out: list[Box] = []
        while merged:
            current = merged.pop()
            for index, other in enumerate(merged):
                if (
                    current.x - gap <= other.x + other.w
                    and other.x - gap <= current.x + current.w
                    and current.y - gap <= other.y + other.h
                    and other.y - gap <= current.y + current.h
                ):
                    left, top = min(current.x, other.x), min(current.y, other.y)
                    right = max(current.x + current.w, other.x + other.w)
                    bottom = max(current.y + current.h, other.y + other.h)
                    merged[index] = Box(left, top, right - left, bottom - top)
                    changed = True
                    break
            else:
                out.append(current)
        merged = out
    return sorted(merged, key=lambda box: (box.x, box.y))


class MotionDetector:
    """Background subtraction (MOG2), a morphological clean-up, contours, boxes."""

    def __init__(self, *, min_area: int = 40, history: int = 400, threshold: float = 24.0) -> None:
        self.min_area = min_area
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=threshold, detectShadows=False
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.mask: Image | None = None

    def detect(self, frame: Image, learning_rate: float = -1.0) -> list[Box]:
        mask = self._subtractor.apply(frame, learningRate=learning_rate)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)
        self.mask = cast(Image, mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [Box(*map(int, cv2.boundingRect(c))) for c in contours if cv2.contourArea(c) >= self.min_area]
        return merge_boxes(boxes)


@dataclass
class Track:
    id: int
    box: Box
    hits: int = 1  # frames it was seen in
    missed: int = 0  # frames since it was last seen
    trail: list[tuple[float, float]] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.hits >= 3


class Tracker:
    """Greedy IoU/centre-distance matching: boxes keep their id from frame to frame."""

    def __init__(self, *, max_missed: int = 6, max_jump: float = 40.0, trail: int = 12) -> None:
        self.max_missed = max_missed
        self.max_jump = max_jump
        self.trail = trail
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, boxes: Iterable[Box]) -> list[Track]:
        pending = list(boxes)
        pairs: list[tuple[float, int, int]] = []
        for t_index, track in enumerate(self.tracks):
            tx, ty = track.box.center
            for b_index, box in enumerate(pending):
                bx, by = box.center
                distance = math.hypot(bx - tx, by - ty)
                overlap = iou(track.box, box)
                if overlap > 0.05 or distance <= self.max_jump:
                    pairs.append((overlap - distance / 1000.0, t_index, b_index))
        used_tracks: set[int] = set()
        used_boxes: set[int] = set()
        for _, t_index, b_index in sorted(pairs, reverse=True):
            if t_index in used_tracks or b_index in used_boxes:
                continue
            used_tracks.add(t_index)
            used_boxes.add(b_index)
            track = self.tracks[t_index]
            track.box, track.hits, track.missed = pending[b_index], track.hits + 1, 0
        for t_index, track in enumerate(self.tracks):
            if t_index not in used_tracks:
                track.missed += 1
        for b_index, box in enumerate(pending):
            if b_index not in used_boxes:
                self.tracks.append(Track(self._next_id, box))
                self._next_id += 1
        self.tracks = [track for track in self.tracks if track.missed <= self.max_missed]
        for track in self.tracks:
            if track.missed == 0:
                track.trail = [*track.trail, track.box.center][-self.trail :]
        return self.visible()

    def visible(self) -> list[Track]:
        return [track for track in self.tracks if track.confirmed and track.missed == 0]

    def primary(self) -> Track | None:
        """The target the turret follows: the longest-tracked visible object, ties to the biggest."""
        visible = self.visible()
        return max(visible, key=lambda t: (t.hits, t.box.area)) if visible else None


# -- geometry: from a pixel to a point on the floor ----------------------------------------------------------


def rotator_axes(pitch: float, yaw: float, roll: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unreal's forward (X), right (Y) and up (Z) unit vectors of a rotator in degrees (FRotationMatrix)."""
    sp, cp = math.sin(math.radians(pitch)), math.cos(math.radians(pitch))
    sy, cy = math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
    sr, cr = math.sin(math.radians(roll)), math.cos(math.radians(roll))
    forward = np.array([cp * cy, cp * sy, sp])
    right = np.array([sr * sp * cy - cr * sy, sr * sp * sy + cr * cy, -sr * cp])
    up = np.array([-(cr * sp * cy + sr * sy), cy * sr - cr * sp * sy, cr * cp])
    return forward, right, up


@dataclass(frozen=True)
class CameraModel:
    """A pinhole camera like a SceneCaptureComponent2D: pose, horizontal FOV and image size."""

    location: tuple[float, float, float]
    rotation: tuple[float, float, float]  # pitch, yaw, roll in degrees
    fov: float  # horizontal, degrees
    width: int
    height: int

    def ray(self, u: float, v: float) -> np.ndarray:
        """The unit direction through pixel (u, v) (origin at the image's top left)."""
        forward, right, up = rotator_axes(*self.rotation)
        tan_half = math.tan(math.radians(self.fov) / 2.0)
        x = (2.0 * u / self.width - 1.0) * tan_half
        y = (1.0 - 2.0 * v / self.height) * tan_half * self.height / self.width
        direction = forward + right * x + up * y
        return direction / np.linalg.norm(direction)

    def to_floor(self, u: float, v: float, floor_z: float = 0.0) -> tuple[float, float, float] | None:
        """Where the ray through (u, v) meets the plane z = floor_z, or None if it never does."""
        direction = self.ray(u, v)
        if direction[2] >= -1e-6:
            return None
        origin = np.asarray(self.location, dtype=float)
        distance = (floor_z - origin[2]) / direction[2]
        point = origin + direction * distance
        return (float(point[0]), float(point[1]), float(point[2]))

    def project(self, point: Sequence[float]) -> tuple[float, float] | None:
        """The pixel a world point lands on, or None behind the camera (the inverse of ray())."""
        forward, right, up = rotator_axes(*self.rotation)
        offset = np.asarray(point, dtype=float) - np.asarray(self.location, dtype=float)
        depth = float(offset @ forward)
        if depth <= 1e-6:
            return None
        tan_half = math.tan(math.radians(self.fov) / 2.0)
        x = float(offset @ right) / depth / tan_half
        y = float(offset @ up) / depth / tan_half / (self.height / self.width)
        return ((x + 1.0) * self.width / 2.0, (1.0 - y) * self.height / 2.0)


# -- the monitor picture -------------------------------------------------------------------------------------

GREEN = (90, 230, 90)
RED = (60, 60, 255)
AMBER = (40, 190, 255)


def annotate(
    frame: Image,
    tracks: Sequence[Track],
    primary: Track | None,
    *,
    alarm: bool,
    clock: float,
    scale: int = 2,
    label: str = "CAM 01",
) -> Image:
    """The monitor's picture: the camera frame upscaled, a box and trail per tracked object, the target
    marked, a status bar, a red frame while the alarm is on."""
    picture = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    picture[::3] = (picture[::3].astype(np.float32) * 0.8).astype(np.uint8)  # scan lines
    font = cv2.FONT_HERSHEY_SIMPLEX
    for track in tracks:
        colour = RED if primary is not None and track.id == primary.id else GREEN
        box = track.box
        top_left = (box.x * scale, box.y * scale)
        bottom_right = ((box.x + box.w) * scale, (box.y + box.h) * scale)
        cv2.rectangle(picture, top_left, bottom_right, colour, 2)
        tag = f"ID {track.id}" + (" TARGET" if colour == RED else "")
        cv2.putText(picture, tag, (top_left[0], max(12, top_left[1] - 5)), font, 0.45, colour, 1, cv2.LINE_AA)
        points = np.array([(x * scale, y * scale) for x, y in track.trail], dtype=np.int32)
        if len(points) > 1:
            cv2.polylines(picture, [points], False, AMBER, 1, cv2.LINE_AA)
    height, width = picture.shape[:2]
    cv2.rectangle(picture, (0, 0), (width, 22), (20, 20, 20), -1)
    minutes, seconds = divmod(int(clock), 60)
    status = "ALARM - TARGET LOCKED" if alarm else "CLEAR"
    cv2.putText(
        picture, f"{label}  {minutes:02d}:{seconds:02d}", (6, 16), font, 0.5, (230, 230, 230), 1, cv2.LINE_AA
    )
    cv2.putText(
        picture,
        f"OBJ {len(tracks)}  {status}",
        (width - 260, 16),
        font,
        0.5,
        RED if alarm else GREEN,
        1,
        cv2.LINE_AA,
    )
    if alarm and int(clock * 4) % 2 == 0:
        cv2.rectangle(picture, (1, 1), (width - 2, height - 2), RED, 4)
    if int(clock * 2) % 2 == 0:
        cv2.circle(picture, (width - 14, height - 14), 6, RED, -1)  # recording dot
    return cast(Image, picture)


def encode_jpeg(picture: Image, quality: int = 88) -> bytes:
    """A JPEG of a BGR picture for ImportBufferAsTexture2D: ~10x smaller than a BMP, so the bytes cross
    into Unreal in a fraction of the time, and the engine decodes it natively."""
    ok, data = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return data.tobytes()


# -- the whole pipeline --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sighting:
    """What one camera frame produced: the monitor picture and what the gameplay should do."""

    picture: bytes  # JPEG
    alarm: bool
    target: tuple[float, float, float] | None  # a floor point to aim at, while the alarm is on
    tracks: tuple[int, ...]
    clock: float


class Pipeline:
    """Detector, tracker, floor projection, alarm hold and the annotated picture, frame after frame.

    Pure OpenCV/numpy, so it runs in a worker thread: cv2 releases the GIL while it works, and the game
    thread only hands frames over and collects Sightings."""

    def __init__(
        self,
        camera: CameraModel,
        bounds: tuple[float, float, float, float],  # x_min, x_max, y_min, y_max of the floor
        *,
        settle_frames: int = 8,
        hold: float = 1.5,
        aim_height: float = 90.0,
    ) -> None:
        self.camera = camera
        self.bounds = bounds
        self.settle_frames = settle_frames
        self.hold = hold
        self.aim_height = aim_height
        self.detector = MotionDetector()
        self.tracker = Tracker()
        self.frames = 0
        self.alarm_until = -math.inf
        self.target: tuple[float, float, float] | None = None

    def process(self, frame: Image, clock: float) -> Sighting:
        self.frames += 1
        tracks = self.tracker.update(self.detector.detect(frame))
        primary = self.tracker.primary()
        if primary is not None and self.frames > self.settle_frames:  # the background model needs a moment
            point = self.camera.to_floor(*primary.box.foot)
            if point is not None:
                x_min, x_max, y_min, y_max = self.bounds
                self.target = (
                    min(max(point[0], x_min), x_max),
                    min(max(point[1], y_min), y_max),
                    self.aim_height,
                )
                self.alarm_until = clock + self.hold
        alarm = clock < self.alarm_until
        picture = annotate(frame, tracks, primary if alarm else None, alarm=alarm, clock=clock)
        return Sighting(
            encode_jpeg(picture), alarm, self.target if alarm else None, tuple(t.id for t in tracks), clock
        )
