"""Security camera example: the parts that need no engine (OpenCV detection and tracking, the camera
geometry, the monitor picture, the pipeline) and the shared showreel helpers."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from examples import showreel, stage
from examples.security_camera import vision

H, W = 144, 256


def _scene(x: int, y: int = 60, size: int = 18) -> np.ndarray:
    """A grey room with a textured floor and one bright square at (x, y)."""
    rng = np.random.default_rng(1)
    frame = np.full((H, W, 3), 70, np.uint8)
    frame[90:] = 50 + rng.integers(0, 6, (H - 90, W, 3), dtype=np.uint8)
    frame[y : y + size, x : x + size] = (40, 140, 230)
    return frame


# -- boxes ---------------------------------------------------------------------------------------------------


def test_box_geometry_and_iou() -> None:
    a, b = vision.Box(0, 0, 10, 10), vision.Box(5, 0, 10, 10)
    assert a.area == 100 and a.center == (5.0, 5.0) and a.foot == (5.0, 10.0)
    assert vision.iou(a, b) == pytest.approx(50 / 150)
    assert vision.iou(a, vision.Box(20, 20, 5, 5)) == 0.0
    assert vision.iou(vision.Box(0, 0, 0, 0), vision.Box(0, 0, 0, 0)) == 0.0


def test_merge_boxes_joins_touching_parts_only() -> None:
    body, head, other = vision.Box(10, 20, 10, 20), vision.Box(11, 12, 8, 7), vision.Box(100, 20, 10, 10)
    merged = vision.merge_boxes([body, head, other])
    assert merged == [vision.Box(10, 12, 10, 28), other]
    assert vision.merge_boxes([]) == []


# -- detection and tracking ----------------------------------------------------------------------------------


def test_detector_finds_the_moving_square_and_ignores_the_still_room() -> None:
    detector = vision.MotionDetector(min_area=30)
    for _ in range(30):
        detector.detect(_scene(20))  # the background model learns the room
    still = detector.detect(_scene(20))
    assert still == []
    boxes = detector.detect(_scene(150))
    assert len(boxes) >= 1
    # the square where it is now (and the floor it uncovered, until the model forgets it)
    assert any(box.x <= 159 <= box.x + box.w and box.y <= 69 <= box.y + box.h for box in boxes), boxes
    assert detector.mask is not None and detector.mask.shape == (H, W)


def test_tracker_keeps_ids_and_picks_the_longest_tracked_target() -> None:
    tracker = vision.Tracker(max_missed=2)
    for step in range(5):
        tracks = tracker.update([vision.Box(10 + 4 * step, 10, 12, 30)])
    assert [t.id for t in tracks] == [1] and tracks[0].hits == 5
    assert len(tracks[0].trail) == 5
    tracker.update([vision.Box(30, 10, 12, 30), vision.Box(200, 50, 20, 20)])  # a newcomer
    tracks = tracker.update([vision.Box(34, 10, 12, 30), vision.Box(202, 50, 20, 20)])
    assert [t.id for t in tracks] == [1]  # the newcomer is not confirmed yet (2 hits)
    tracks = tracker.update([vision.Box(38, 10, 12, 30), vision.Box(204, 50, 20, 20)])
    assert sorted(t.id for t in tracks) == [1, 2]
    primary = tracker.primary()
    assert primary is not None and primary.id == 1
    for _ in range(3):
        tracker.update([])
    assert tracker.tracks == [] and tracker.primary() is None


def test_tracker_does_not_jump_to_a_far_box() -> None:
    tracker = vision.Tracker(max_jump=30.0)
    tracker.update([vision.Box(0, 0, 10, 10)])
    tracker.update([vision.Box(120, 0, 10, 10)])
    assert sorted(t.id for t in tracker.tracks) == [1, 2]


# -- geometry ------------------------------------------------------------------------------------------------


def test_rotator_axes_match_unreal() -> None:
    forward, right, up = vision.rotator_axes(0.0, 90.0)
    assert np.allclose(forward, [0, 1, 0]) and np.allclose(right, [-1, 0, 0]) and np.allclose(up, [0, 0, 1])
    forward, right, up = vision.rotator_axes(-90.0, 0.0)
    assert np.allclose(forward, [0, 0, -1]) and np.allclose(right, [0, 1, 0])
    for pitch, yaw, roll in ((-30.0, 45.0, 0.0), (10.0, -120.0, 20.0)):
        axes = np.array(vision.rotator_axes(pitch, yaw, roll))
        assert np.allclose(axes @ axes.T, np.eye(3))  # orthonormal


def test_look_at_and_turn_towards() -> None:
    pitch, yaw, roll = stage.look_at((0, 0, 100), (100, 100, 0))
    assert yaw == pytest.approx(45.0) and pitch == pytest.approx(-math.degrees(math.atan2(100, 141.42)), 1e-3)
    assert roll == 0.0
    assert stage.turn_towards(170.0, -170.0, 5.0) == pytest.approx(175.0)  # the short way, across 180
    assert stage.turn_towards(10.0, 12.0, 5.0) == pytest.approx(12.0)
    assert stage.turn_towards(10.0, -50.0, 5.0) == pytest.approx(5.0)


def test_camera_model_projects_and_back_projects_floor_points() -> None:
    location = (700.0, 500.0, 430.0)
    camera = vision.CameraModel(location, stage.look_at(location, (0.0, 0.0, 0.0)), 90.0, W, H)
    centre = camera.to_floor(W / 2, H / 2)
    assert centre is not None and np.allclose(centre, (0.0, 0.0, 0.0), atol=1e-6)
    for point in ((-300.0, 120.0, 0.0), (250.0, -200.0, 0.0)):
        pixel = camera.project(point)
        assert pixel is not None
        back = camera.to_floor(*pixel)
        assert back is not None and np.allclose(back, point, atol=1e-6)
    assert camera.project((2000.0, 2000.0, 0.0)) is None  # behind the camera
    top = camera.to_floor(W / 2, 0.0)
    assert top is None or top[2] == pytest.approx(0.0)
    level = vision.CameraModel((0.0, 0.0, 100.0), (10.0, 0.0, 0.0), 90.0, W, H)
    assert level.to_floor(W / 2, H / 2) is None  # looking up never meets the floor


# -- the monitor and the pipeline ----------------------------------------------------------------------------


def test_annotate_draws_boxes_and_the_alarm_frame() -> None:
    frame = np.full((H, W, 3), 60, np.uint8)
    track = vision.Track(3, vision.Box(40, 40, 30, 50), hits=4, trail=[(50.0, 60.0), (55.0, 65.0)])
    calm = vision.annotate(frame, [track], None, alarm=False, clock=1.6)
    alarm = vision.annotate(frame, [track], track, alarm=True, clock=2.0)
    assert calm.shape == alarm.shape == (H * 2, W * 2, 3)
    red = (alarm[:, :, 2] > 200) & (alarm[:, :, 1] < 90)
    assert red[78:82, 80:140].any()  # the target's box is red
    assert red[H, :6].any()  # and so is the alarm frame
    assert not ((calm[:, :, 2] > 200) & (calm[:, :, 1] < 90))[H, :6].any()
    assert (frame == 60).all()  # the camera frame itself is untouched


def test_encode_jpeg_round_trips() -> None:
    picture = np.zeros((32, 48, 3), np.uint8)
    picture[:, 24:] = (0, 0, 255)
    data = vision.encode_jpeg(picture)
    assert data[:3] == b"\xff\xd8\xff"
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and decoded.shape == picture.shape and decoded[16, 40, 2] > 200


def test_pipeline_raises_the_alarm_and_aims_at_the_floor() -> None:
    location = (740.0, 560.0, 430.0)
    camera = vision.CameraModel(location, stage.look_at(location, (-180.0, -150.0, 0.0)), 90.0, W, H)
    pipeline = vision.Pipeline(camera, (-800.0, 800.0, -600.0, 600.0), settle_frames=5, hold=1.0)
    clock = 0.0
    for _ in range(20):  # an empty room: no alarm
        clock += 0.125
        sighting = pipeline.process(_scene(20), clock)
    assert not sighting.alarm and sighting.target is None and sighting.picture[:2] == b"\xff\xd8"
    for step in range(8):  # something walks in
        clock += 0.125
        sighting = pipeline.process(_scene(120 + 6 * step), clock)
    assert sighting.alarm and sighting.tracks
    assert sighting.target is not None
    x, y, z = sighting.target
    assert -800.0 <= x <= 800.0 and -600.0 <= y <= 600.0 and z == pipeline.aim_height
    for _ in range(20):  # it stops: the alarm holds a second, then clears
        clock += 0.125
        sighting = pipeline.process(_scene(162), clock)
    assert not sighting.alarm and sighting.target is None


# -- showreel ------------------------------------------------------------------------------------------------


def test_percentile_and_perf_meter() -> None:
    assert showreel.percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.5) == 3.0
    assert showreel.percentile([1.0], 0.99) == 1.0
    with pytest.raises(ValueError):
        showreel.percentile([], 0.5)
    now = [0.0]
    meter = showreel.PerfMeter(clock=lambda: now[0])
    for step in range(5):
        now[0] = step * 0.010
        meter.frame()
        with meter.measure("vision"):
            now[0] += 0.002
    summary = meter.summary()
    assert summary["frames"] == 4 and summary["fps"] == pytest.approx(100.0)
    assert summary["systems"]["vision"]["calls"] == 5
    assert summary["systems"]["vision"]["mean_ms"] == pytest.approx(2.0)
    assert summary["systems"]["vision"]["ms_per_frame"] == pytest.approx(2.5)
    meter.reset()
    assert meter.summary() == {"frames": 0, "systems": {}}


def test_ffmpeg_commands() -> None:
    clip = showreel.clip_command("ffmpeg", Path("frames"), 30, Path("out.mp4"))
    assert clip[0] == "ffmpeg" and str(Path("frames") / "frame_%05d.png") in clip
    assert clip[clip.index("-framerate") + 1] == "30" and clip[-1] == "out.mp4"
    assert "yuv420p" in clip
    still = showreel.still_command("ffmpeg", Path("a.png"), Path("a.jpg"), width=640)
    assert "scale=640:-2" in still and still[-1] == "a.jpg"
