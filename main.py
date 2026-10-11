"""
Assistive Desktop Navigator
===========================
Hands-free OS mouse control for users with severe motor impairments.

Architecture
------------
Layer A  Screen coordinate mapping (webcam → monitor pixels)
Layer B  Exponential Moving Average (EMA) tremor / spasm filter
Layer C  Ultra-low-fatigue micro-gesture mode (index-tip vs base joint)
Layer D  Air Clutch protocol (2 s steady fist lock / unlock)
Layer E  Dwell-time click + slow vertical-wave document scroll

Requires a webcam. Mouse injection is disabled until the Air Clutch unlocks.
Press M (or the HUD button) to toggle micro-gesture mode. Press Q / Esc to quit.
"""

from __future__ import annotations

import math
import sys
import threading
import time
import tkinter as tk
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from PIL import Image, ImageTk
class EMA:
    def __init__(self, span=12):  # Reduced span from 20 to 12 makes cursor response faster
        self.span = span
        self.alpha = 2.0 / (span + 1.0)
        self.value_x = None
        self.value_y = None
        # Threshold to ignore sudden, large accidental jumps (in normalized coordinates)
        self.max_jump_threshold = 0.15 

    def update(self, nx, ny):
        if self.value_x is None or self.value_y is None:
            self.value_x = nx
            self.value_y = ny
        else:
            # Check the distance of the movement
            distance = ((nx - self.value_x) ** 2 + (ny - self.value_y) ** 2) ** 0.5
            
            # If it's a massive, accidental quick twitch, suppress it by blending heavily with old position
            if distance > self.max_jump_threshold:
                effective_alpha = self.alpha * 0.2  # Reject most of the sudden jump
            else:
                effective_alpha = self.alpha

            self.value_x = effective_alpha * nx + (1.0 - effective_alpha) * self.value_x
            self.value_y = effective_alpha * ny + (1.0 - effective_alpha) * self.value_y
        
        return self.value_x, self.value_y, 0.0
# ---------------------------------------------------------------------------
# Global runtime flag (Layer C) — toggled by hotkey or HUD button
# ---------------------------------------------------------------------------
micro_gesture_mode: bool = False

# ---------------------------------------------------------------------------
# Safety: never let PyAutoGUI abort on a corner flick; Air Clutch is the
# accident-proof gate. Zero pause so dwell / scroll stay responsive.
# ---------------------------------------------------------------------------
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
EMA_SPAN = 20
EMA_ALPHA = 2/(EMA_SPAN+1)
# Windows: map logical pixels 1:1 with the physical desktop (DPI scaling).
if sys.platform == "win32":
    try:
        from ctypes import windll

        windll.user32.SetProcessDPIAware()
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
HUD_VIDEO_WIDTH = 480

# Velocity-adaptive low-pass (replaces fixed-alpha EMA).
ADAPTIVE_ALPHA_MIN = 0.05  # targeting / tremor: heavy damping
ADAPTIVE_ALPHA_MAX = 0.35  # fast sweep: cut cursor lag
ADAPTIVE_VEL_SLOW_PX = 5.0  # at/below this inter-frame jump -> min alpha
ADAPTIVE_VEL_FAST_PX = 90.0  # at/above this jump -> max alpha
SUBPIXEL_DEADZONE_PX = 4.0  # lock cursor if destination moved < 4 px

# Mapping: ignore the outer 12% of the camera so edge landmarks stay reachable.
FRAME_MARGIN = 0.12

CLUTCH_HOLD_SECONDS = 2.0
CLUTCH_STILL_NORM = 0.045  # palm must stay this still while holding the fist
CLUTCH_COOLDOWN_SECONDS = 1.4

DWELL_SECONDS = 0.25
DWELL_RADIUS_PX = 35  # 25 px cushion + snap; timer does not reset inside this radius
DWELL_COOLDOWN_SECONDS = 0.5

SCROLL_WINDOW = 8
SCROLL_VY_MIN = 0.012  # normalized units / second  (slow wave floor)
SCROLL_VY_MAX = 0.55  # reject fast flails
SCROLL_GAIN = 420.0
SCROLL_COOLDOWN = 0.05

MICRO_DEADZONE_DEG = 3.0
MICRO_TILT_ENGAGE_DEG = 5.0
MICRO_SPATIAL_DEADZONE = 0.004  # ~noise floor in MCP-local space
MICRO_TWITCH_NORM = 0.028  # ~1 cm at typical webcam distance
MICRO_SIGMOID_K = 1.35  # tanh curve: small twitch glides, large flick saturates
MICRO_REACH = 0.55  # saturated 1 cm-class twitch covers this fraction of the desktop
MICRO_ANGLE_REF_DEG = 28.0

SPASM_LOG_THRESHOLD = 0.045  # raw vs EMA delta that counts as a suppressed spasm
LOG_MAX_LINES = 8
HUD_REFRESH_MS = 16
STATUS_LOCKED = "🔒 LOCKED - GESTURES PAUSED"
STATUS_ACTIVE = "🟢 ACTIVE - CONTROLLING OS"

INDEX_TIP = 8
INDEX_PIP = 6
INDEX_MCP = 5
WRIST = 0
FINGER_TIP_MCP = ((8, 5), (12, 9), (16, 13), (20, 17))

# MediaPipe 0.10.31+ / 1.x removed mp.solutions. Use the Tasks Hand Landmarker.
APP_DIR = Path(__file__).resolve().parent
HAND_MODEL_PATH = APP_DIR / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _xy(landmarks, index: int) -> np.ndarray:
    lm = landmarks[index]
    return np.array([lm.x, lm.y], dtype=np.float64)


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _angle_deg(vec: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vec[1]), float(vec[0])))


def _angle_delta_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def _map_norm_to_screen(
    nx: float, ny: float, screen_w: int, screen_h: int
) -> Tuple[int, int]:
    """Layer A — map a camera-normalized point onto absolute monitor pixels."""
    usable = 1.0 - 2.0 * FRAME_MARGIN
    mx = _clamp01((nx - FRAME_MARGIN) / usable)
    my = _clamp01((ny - FRAME_MARGIN) / usable)
    return int(mx * (screen_w - 1)), int(my * (screen_h - 1))


def ensure_hand_landmarker_model() -> Path:
    """Download the Tasks .task bundle once if it is not already next to app.py."""
    if HAND_MODEL_PATH.is_file() and HAND_MODEL_PATH.stat().st_size > 1_000_000:
        return HAND_MODEL_PATH
    print(f"[Init] Downloading Hand Landmarker model -> {HAND_MODEL_PATH}", flush=True)
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            "Could not download hand_landmarker.task. Check your network, then retry. "
            f"URL: {HAND_MODEL_URL}\nOriginal error: {exc}"
        ) from exc
    if not HAND_MODEL_PATH.is_file() or HAND_MODEL_PATH.stat().st_size < 1_000_000:
        raise RuntimeError("hand_landmarker.task download looks incomplete. Delete it and retry.")
    return HAND_MODEL_PATH


def create_hand_landmarker():
    """Build a VIDEO-mode HandLandmarker (replaces the removed mp.solutions.hands)."""
    model_path = str(ensure_hand_landmarker_model())
    try:
        base_options_cls = mp.tasks.BaseOptions
        landmarker_cls = mp.tasks.vision.HandLandmarker
        options_cls = mp.tasks.vision.HandLandmarkerOptions
        running_mode = mp.tasks.vision.RunningMode.VIDEO
    except AttributeError:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options_cls = mp_python.BaseOptions
        landmarker_cls = mp_vision.HandLandmarker
        options_cls = mp_vision.HandLandmarkerOptions
        running_mode = mp_vision.RunningMode.VIDEO

    options = options_cls(
        base_options=base_options_cls(model_asset_path=model_path),
        running_mode=running_mode,
        num_hands=1,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )
    return landmarker_cls.create_from_options(options)


def draw_hand_mesh(frame_bgr: np.ndarray, landmarks: Sequence) -> None:
    """Green joint mesh — drawing_utils shipped with mp.solutions and is gone in 1.x."""
    h, w = frame_bgr.shape[:2]
    pts = []
    for lm in landmarks:
        x = int(_clamp01(float(lm.x)) * (w - 1))
        y = int(_clamp01(float(lm.y)) * (h - 1))
        pts.append((x, y))
        cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], (40, 220, 40), 2, lineType=cv2.LINE_AA)


def is_fist(landmarks) -> bool:
    """Closed hand: four fingertips pulled in toward the wrist past their MCPs."""
    wrist = _xy(landmarks, WRIST)
    folded = 0
    for tip_i, mcp_i in FINGER_TIP_MCP:
        if _dist(_xy(landmarks, tip_i), wrist) < _dist(_xy(landmarks, mcp_i), wrist) * 1.18:
            folded += 1
    thumb_closed = _dist(_xy(landmarks, 4), wrist) < _dist(_xy(landmarks, 2), wrist) * 1.40
    return folded >= 4 and thumb_closed


def _palm_local(landmarks) -> Tuple[np.ndarray, float, float]:
    """Index tip in the wrist→MCP palm frame. Cancels whole-hand translation."""
    wrist = _xy(landmarks, WRIST)
    mcp = _xy(landmarks, INDEX_MCP)
    tip = _xy(landmarks, INDEX_TIP)
    pip = _xy(landmarks, INDEX_PIP)
    axis = mcp - wrist
    axis_n = float(np.linalg.norm(axis))
    if axis_n < 1e-9:
        axis_u = np.array([0.0, 1.0], dtype=np.float64)
    else:
        axis_u = axis / axis_n
    perp_u = np.array([-axis_u[1], axis_u[0]], dtype=np.float64)
    rel = tip - mcp  # Landmark 8 relative to base knuckle (5), never absolute webcam XY
    local = np.array([float(np.dot(rel, axis_u)), float(np.dot(rel, perp_u))], dtype=np.float64)
    knuckle_angle = _angle_deg(rel)
    pip_angle = _angle_deg(tip - pip)
    return local, knuckle_angle, pip_angle


def _tanh_scale(value: float, ref: float) -> float:
    """Power-law-like sigmoid: tiny deltas glide, large flicks saturate."""
    if ref <= 1e-12:
        return 0.0
    return math.tanh((value / ref) * MICRO_SIGMOID_K)


# ---------------------------------------------------------------------------
# Layer B — velocity-adaptive low-pass (pixel space, no fixed EMA alpha)
# ---------------------------------------------------------------------------
class AdaptiveLowPass:
    """
    One-pole low-pass whose alpha tracks inter-frame pixel velocity.

    Slow / targeting  -> alpha = 0.05 (kill tremor and camera jitter)
    Fast / sweeping   -> alpha = 0.35 (no trailing lag)
    Destination jump < 4 px -> hold the locked coordinate (sub-pixel dead-zone)
    """

    __slots__ = ("_x", "_y", "_prev_raw_x", "_prev_raw_y", "last_alpha", "ready")

    def __init__(self) -> None:
        self._x = 0.0
        self._y = 0.0
        self._prev_raw_x: Optional[float] = None
        self._prev_raw_y: Optional[float] = None
        self.last_alpha = ADAPTIVE_ALPHA_MIN
        self.ready = False

    def reset(self) -> None:
        self._prev_raw_x = None
        self._prev_raw_y = None
        self.ready = False
        self.last_alpha = ADAPTIVE_ALPHA_MIN

    def update(self, x_px: float, y_px: float) -> Tuple[float, float, float, bool]:
        if not self.ready or self._prev_raw_x is None or self._prev_raw_y is None:
            self._x = x_px
            self._y = y_px
            self._prev_raw_x = x_px
            self._prev_raw_y = y_px
            self.ready = True
            self.last_alpha = ADAPTIVE_ALPHA_MIN
            return x_px, y_px, 0.0, False

        dist = math.hypot(x_px - self._prev_raw_x, y_px - self._prev_raw_y)
        self._prev_raw_x = x_px
        self._prev_raw_y = y_px

        span = ADAPTIVE_VEL_FAST_PX - ADAPTIVE_VEL_SLOW_PX
        t = (dist - ADAPTIVE_VEL_SLOW_PX) / span
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        t = t * t * (3.0 - 2.0 * t)  # smoothstep, 3 mul, no numpy
        alpha = ADAPTIVE_ALPHA_MIN + (ADAPTIVE_ALPHA_MAX - ADAPTIVE_ALPHA_MIN) * t
        self.last_alpha = alpha

        filt_x = alpha * x_px + (1.0 - alpha) * self._x
        filt_y = alpha * y_px + (1.0 - alpha) * self._y
        jump = math.hypot(filt_x - self._x, filt_y - self._y)
        locked = jump < SUBPIXEL_DEADZONE_PX
        if not locked:
            self._x = filt_x
            self._y = filt_y
        return self._x, self._y, dist, locked


# ---------------------------------------------------------------------------
# Layer C — relative micro-gesture kinematics (MCP / wrist frame + tanh curve)
# ---------------------------------------------------------------------------
class MicroGestureTracker:
    """
    Never uses absolute webcam coordinates.

    Control vector = Index Tip (8) minus Index MCP (5), expressed in the
    wrist(0)→knuckle(5) palm basis. Only angular / local spatial deltas from
    the rest pose drive the cursor, through a tanh (sigmoid) curve.
    """

    __slots__ = ("rest_local", "rest_knuckle_angle", "rest_pip_angle", "rest_cursor")

    def __init__(self) -> None:
        self.rest_local: Optional[np.ndarray] = None
        self.rest_knuckle_angle: Optional[float] = None
        self.rest_pip_angle: Optional[float] = None
        self.rest_cursor = np.array([0.5, 0.5], dtype=np.float64)

    def capture_rest(self, landmarks, cursor_norm: Tuple[float, float]) -> None:
        local, knuckle_angle, pip_angle = _palm_local(landmarks)
        self.rest_local = local.copy()
        self.rest_knuckle_angle = knuckle_angle
        self.rest_pip_angle = pip_angle
        self.rest_cursor = np.array(cursor_norm, dtype=np.float64)

    def clear(self) -> None:
        self.rest_local = None
        self.rest_knuckle_angle = None
        self.rest_pip_angle = None

    def step(self, landmarks) -> Tuple[float, float]:
        if self.rest_local is None or self.rest_knuckle_angle is None or self.rest_pip_angle is None:
            self.capture_rest(landmarks, (float(self.rest_cursor[0]), float(self.rest_cursor[1])))

        local, knuckle_angle, pip_angle = _palm_local(landmarks)
        d_local = local - self.rest_local
        mag = float(np.linalg.norm(d_local))
        if mag < MICRO_SPATIAL_DEADZONE:
            d_local = np.zeros(2, dtype=np.float64)
        else:
            d_local *= (mag - MICRO_SPATIAL_DEADZONE) / mag

        d_knuckle = _angle_delta_deg(knuckle_angle, self.rest_knuckle_angle)
        d_pip = _angle_delta_deg(pip_angle, self.rest_pip_angle)
        if abs(d_knuckle) < MICRO_DEADZONE_DEG:
            d_knuckle = 0.0
        if abs(d_pip) < MICRO_DEADZONE_DEG:
            d_pip = 0.0
        # >5° micro-tilt engages extra angular reach; 3–5° stays spatial-only.
        if abs(d_knuckle) < MICRO_TILT_ENGAGE_DEG:
            d_knuckle *= 0.25

        offset_x = _tanh_scale(float(d_local[1]), MICRO_TWITCH_NORM) * MICRO_REACH
        offset_y = _tanh_scale(float(d_local[0]), MICRO_TWITCH_NORM) * MICRO_REACH
        offset_x += 0.22 * MICRO_REACH * _tanh_scale(d_knuckle, MICRO_ANGLE_REF_DEG)
        offset_y += 0.16 * MICRO_REACH * _tanh_scale(d_pip, MICRO_ANGLE_REF_DEG)

        tx = _clamp01(float(self.rest_cursor[0]) + offset_x)
        ty = _clamp01(float(self.rest_cursor[1]) + offset_y)
        return tx, ty


# ---------------------------------------------------------------------------
# Shared state between the capture thread and the Tkinter HUD
# ---------------------------------------------------------------------------
@dataclass
class SharedState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    frame_bgr: Optional[np.ndarray] = None
    locked: bool = True
    clutch_progress: float = 0.0
    dwell_progress: float = 0.0
    log_lines: List[str] = field(default_factory=list)
    running: bool = True
    fps: float = 0.0
    hand_present: bool = False
    last_screen_xy: Tuple[int, int] = (0, 0)


class SafetyLog:
    def __init__(self, state: SharedState) -> None:
        self._state = state
        self._last_key: Optional[str] = None
        self._last_time = 0.0

    def push(self, message: str, *, key: Optional[str] = None, min_interval: float = 0.45) -> None:
        now = time.monotonic()
        if key is not None and key == self._last_key and (now - self._last_time) < min_interval:
            return
        self._last_key = key
        self._last_time = now
        stamp = time.strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with self._state.lock:
            lines = list(self._state.log_lines)
            lines.append(line)
            self._state.log_lines = lines[-LOG_MAX_LINES:]


# ---------------------------------------------------------------------------
# Vision + control loop (background thread)
# ---------------------------------------------------------------------------
class NavigatorEngine:
    def __init__(self, state: SharedState, log: SafetyLog) -> None:
        self.state = state
        self.log = log
        self.screen_w, self.screen_h = pyautogui.size()
        self.lpf = AdaptiveLowPass()
        self.micro = MicroGestureTracker()
        self.hands = create_hand_landmarker()
        self._frame_ts_ms = 0
        self._last_move: Optional[Tuple[int, int]] = None
        print(
            "[Filter] Adaptive low-pass initialized "
            "(alpha 0.05-0.35 | 4px sub-pixel lock | 25px dwell snap)",
            flush=True,
        )

        self._clutch_origin: Optional[np.ndarray] = None
        self._clutch_started: Optional[float] = None
        self._clutch_cool_until = 0.0
        self._dwell_anchor: Optional[Tuple[int, int]] = None
        self._dwell_started: Optional[float] = None
        self._dwell_cool_until = 0.0
        self._tip_y: Deque[Tuple[float, float, float]] = deque(maxlen=SCROLL_WINDOW)
        self._last_scroll_t = 0.0
        self._was_micro = False
        self._norm_cursor = (0.5, 0.5)
        self._last_fps_t = time.monotonic()
        self._fps_frames = 0
        self._fps = 0.0
        self.ema = EMA(span=20)

    def close(self) -> None:
        self.hands.close()

    def process_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        global micro_gesture_mode

        frame = cv2.flip(frame_bgr, 1)  # Layer A — mirror so motion matches intuition
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._frame_ts_ms += 33  # ~30 FPS clock; Tasks requires strictly increasing timestamps
        result = self.hands.detect_for_video(mp_image, self._frame_ts_ms)

        now = time.monotonic()
        self._fps_frames += 1
        if now - self._last_fps_t >= 0.5:
            self._fps = self._fps_frames / (now - self._last_fps_t)
            self._fps_frames = 0
            self._last_fps_t = now

        locked = True
        with self.state.lock:
            locked = self.state.locked

        hand_present = bool(result.hand_landmarks)
        clutch_progress = 0.0
        dwell_progress = 0.0
        landmarks = None

        if hand_present:
            landmarks = result.hand_landmarks[0]
            draw_hand_mesh(frame, landmarks)
            clutch_progress = self._update_air_clutch(landmarks, locked, now)
            with self.state.lock:
                locked = self.state.locked

            if locked:
                self.lpf.reset()
                self._reset_dwell()
                self._tip_y.clear()
            else:
                dwell_progress = self._control_unlocked(landmarks, now)
        else:
            self._clutch_origin = None
            self._clutch_started = None
            if not locked:
                self.log.push("[Track] Hand lost - cursor held", key="lost", min_interval=1.2)

        # Overlay HUD chrome on the camera tile (also mirrored into Tkinter).
        banner = STATUS_LOCKED if locked else STATUS_ACTIVE
        color = (40, 40, 200) if locked else (40, 170, 40)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), color, -1)
        cv2.putText(
            frame,
            banner,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"FPS {self._fps:.0f} | Micro {'ON' if micro_gesture_mode else 'OFF'}",
            (12, frame.shape[0] - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        if clutch_progress > 0.0:
            w = int(frame.shape[1] * clutch_progress)
            cv2.rectangle(frame, (0, 44), (w, 56), (0, 220, 255), -1)

        with self.state.lock:
            self.state.frame_bgr = frame
            self.state.clutch_progress = clutch_progress
            self.state.dwell_progress = dwell_progress
            self.state.fps = self._fps
            self.state.hand_present = hand_present

        return frame

    # ----- Layer D: Air Clutch ------------------------------------------------
    def _update_air_clutch(self, landmarks, locked: bool, now: float) -> float:
        if now < self._clutch_cool_until:
            return 0.0

        palm = _xy(landmarks, WRIST)
        if not is_fist(landmarks):
            self._clutch_origin = None
            self._clutch_started = None
            return 0.0

        if self._clutch_started is None or self._clutch_origin is None:
            self._clutch_started = now
            self._clutch_origin = palm.copy()
            return 0.0

        if _dist(palm, self._clutch_origin) > CLUTCH_STILL_NORM:
            self._clutch_started = now
            self._clutch_origin = palm.copy()
            self.log.push("[Clutch] Hold steadier to toggle", key="clutch-still", min_interval=0.8)
            return 0.0

        elapsed = now - self._clutch_started
        progress = min(1.0, elapsed / CLUTCH_HOLD_SECONDS)
        if elapsed >= CLUTCH_HOLD_SECONDS:
            new_locked = not locked
            with self.state.lock:
                self.state.locked = new_locked
            self._clutch_started = None
            self._clutch_origin = None
            self._clutch_cool_until = now + CLUTCH_COOLDOWN_SECONDS
            self._reset_dwell()
            self.lpf.reset()
            self._last_move = None
            if new_locked:
                self.micro.clear()
                self.log.push("[Clutch] Toggled -> locked")
                print("[SYSTEM LOCKED - MOUSE CONTROL DISABLED]", flush=True)
            else:
                print("[SYSTEM ACTIVE - MOUSE CONTROL ENABLED]", flush=True)
                self.log.push("[Clutch] Toggled -> active")
                tip = _xy(landmarks, INDEX_TIP)
                self._norm_cursor = (float(tip[0]), float(tip[1]))
                mapped_x, mapped_y = _map_norm_to_screen(
                    float(tip[0]), float(tip[1]), self.screen_w, self.screen_h
                )
                rest_cursor = (
                    mapped_x / max(self.screen_w - 1, 1),
                    mapped_y / max(self.screen_h - 1, 1),
                )
                if micro_gesture_mode:
                    self.micro.capture_rest(landmarks, rest_cursor)
                    self._norm_cursor = rest_cursor
                    self._was_micro = True
            return 0.0
        return progress

    # ----- Layers A / B / C / E when unlocked --------------------------------
    def _control_unlocked(self, landmarks, now: float) -> float:
        global micro_gesture_mode

        if micro_gesture_mode and not self._was_micro:
            mapped_x, mapped_y = _map_norm_to_screen(
                self._norm_cursor[0], self._norm_cursor[1], self.screen_w, self.screen_h
            )
            rest_cursor = (
                mapped_x / max(self.screen_w - 1, 1),
                mapped_y / max(self.screen_h - 1, 1),
            )
            self.micro.capture_rest(landmarks, rest_cursor)
            self.ema.reset()
            self.log.push("[Micro] High-sensitivity mode ON")
        elif not micro_gesture_mode and self._was_micro:
            self.micro.clear()
            self.ema.reset()
            self.log.push("[Micro] Full-sweep mapping ON")
        self._was_micro = micro_gesture_mode

        if micro_gesture_mode:
            nx, ny = self.micro.step(landmarks)
        else:
            tip = _xy(landmarks, INDEX_TIP)
            nx, ny = float(tip[0]), float(tip[1])

        sx, sy, spasm = self.ema.update(nx, ny)
        self._norm_cursor = (sx, sy)
        if spasm > SPASM_LOG_THRESHOLD and len(self.ema.history) >= 3:
            self.log.push("[Filter] Spasm Suppressed", key="spasm", min_interval=0.7)

        if micro_gesture_mode:
            screen_x = int(_clamp01(sx) * (self.screen_w - 1))
            screen_y = int(_clamp01(sy) * (self.screen_h - 1))
        else:
            screen_x, screen_y = _map_norm_to_screen(sx, sy, self.screen_w, self.screen_h)

        with self.state.lock:
            self.state.last_screen_xy = (screen_x, screen_y)

        scrolling = self._maybe_scroll(landmarks, now)
        if scrolling:
            self._reset_dwell()
            pyautogui.moveTo(screen_x, screen_y, duration=0)
            return 0.0

        pyautogui.moveTo(screen_x, screen_y, duration=0)
        return self._maybe_dwell(screen_x, screen_y, now)

    def _maybe_scroll(self, landmarks, now: float) -> bool:
        """Layer E — slow vertical waving of the index tip → document scroll."""
        tip = _xy(landmarks, INDEX_TIP)
        self._tip_y.append((now, float(tip[0]), float(tip[1])))
        if len(self._tip_y) < 4:
            return False

        t0, x0, y0 = self._tip_y[0]
        t1, x1, y1 = self._tip_y[-1]
        dt = max(t1 - t0, 1e-6)
        vx = (x1 - x0) / dt
        vy = (y1 - y0) / dt  # +vy = tip moving down in image space

        # Slow vertical wave: Y-velocity in band, X-wander clearly smaller than Y.
        if not (SCROLL_VY_MIN <= abs(vy) <= SCROLL_VY_MAX):
            return False
        if abs(vy) < abs(vx) * 1.6:
            return False
        if now - self._last_scroll_t < SCROLL_COOLDOWN:
            return True

        amount = int(max(-12, min(12, -vy * SCROLL_GAIN * dt)))
        if amount == 0:
            return False
        pyautogui.scroll(amount)
        self._last_scroll_t = now
        direction = "up" if amount > 0 else "down"
        self.log.push(f"[Scroll] Vertical document scroll {direction}", key="scroll", min_interval=0.6)
        return True

    def _maybe_dwell(self, screen_x: int, screen_y: int, now: float) -> float:
        """Layer E — stay still 1.5 s → native left click. No finger-click gestures."""
        if now < self._dwell_cool_until:
            return 0.0

        point = (screen_x, screen_y)
        if self._dwell_anchor is None or self._dwell_started is None:
            self._dwell_anchor = point
            self._dwell_started = now
            return 0.0

        dx = point[0] - self._dwell_anchor[0]
        dy = point[1] - self._dwell_anchor[1]
        if (dx * dx + dy * dy) ** 0.5 > DWELL_RADIUS_PX:
            self._dwell_anchor = point
            self._dwell_started = now
            return 0.0

        elapsed = now - self._dwell_started
        remaining = DWELL_SECONDS - elapsed
        if remaining > 0:
            if remaining <= 0.40:
                self.log.push(
                    f"[Dwell] Clicking in {remaining:.1f}s",
                    key="dwell",
                    min_interval=0.35,
                )
            return min(1.0, elapsed / DWELL_SECONDS)

        pyautogui.click()
        self.log.push("[Dwell] Left click")
        self._dwell_cool_until = now + DWELL_COOLDOWN_SECONDS
        self._reset_dwell()
        return 0.0

    def _reset_dwell(self) -> None:
        self._dwell_anchor = None
        self._dwell_started = None


def capture_loop(engine: NavigatorEngine, state: SharedState, cap: cv2.VideoCapture) -> None:
    while True:
        with state.lock:
            if not state.running:
                break
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        engine.process_frame(frame)


# ---------------------------------------------------------------------------
# Tkinter HUD overlay (corner of the desktop)
# ---------------------------------------------------------------------------
class NavigatorHUD:
    def __init__(self) -> None:
        self.state = SharedState()
        self.log = SafetyLog(self.state)
        self.engine = NavigatorEngine(self.state, self.log)

        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Unable to open webcam (index 0). Plug in a camera and retry.")

        self.root = tk.Tk()
        self.root.title("Assistive Desktop Navigator")
        self.root.configure(bg="#111318")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)

        screen_w = self.root.winfo_screenwidth()
        self.root.geometry(f"+{max(8, screen_w - 520)}+12")

        self.banner = tk.Label(
            self.root,
            text=STATUS_LOCKED,
            font=("Segoe UI", 16, "bold"),
            fg="#fff5f5",
            bg="#b42318",
            padx=12,
            pady=10,
        )
        self.banner.pack(fill=tk.X)

        self.video = tk.Label(self.root, bg="#000000")
        self.video.pack(padx=8, pady=8)

        meta = tk.Frame(self.root, bg="#111318")
        meta.pack(fill=tk.X, padx=10)
        self.meta_lbl = tk.Label(
            meta,
            text="Boot: SYSTEM LOCKED — hold a steady fist for 2s to engage.",
            font=("Segoe UI", 10),
            fg="#d0d5dd",
            bg="#111318",
            anchor="w",
            justify=tk.LEFT,
        )
        self.meta_lbl.pack(fill=tk.X)

        btn_row = tk.Frame(self.root, bg="#111318")
        btn_row.pack(fill=tk.X, padx=10, pady=(6, 4))
        self.micro_btn = tk.Button(
            btn_row,
            text="Micro-Gesture: OFF  (M)",
            command=self.toggle_micro,
            font=("Segoe UI", 10, "bold"),
            bg="#1d2939",
            fg="#f5f5f5",
            activebackground="#344054",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=6,
        )
        self.micro_btn.pack(side=tk.LEFT)
        tk.Button(
            btn_row,
            text="Quit  (Q)",
            command=self.shutdown,
            font=("Segoe UI", 10, "bold"),
            bg="#3f1f1f",
            fg="#f5f5f5",
            activebackground="#5c2b2b",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=6,
        ).pack(side=tk.RIGHT)

        tk.Label(
            self.root,
            text="Safety Log Feed",
            font=("Segoe UI", 11, "bold"),
            fg="#98a2b3",
            bg="#111318",
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(8, 0))

        self.log_box = tk.Text(
            self.root,
            height=8,
            width=58,
            font=("Consolas", 10),
            bg="#0d1117",
            fg="#7af8a2",
            insertbackground="#7af8a2",
            relief=tk.FLAT,
            state=tk.DISABLED,
        )
        self.log_box.pack(fill=tk.BOTH, padx=10, pady=(2, 10))

        self._photo: Optional[ImageTk.PhotoImage] = None
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
        self.root.bind("<KeyPress-m>", lambda _e: self.toggle_micro())
        self.root.bind("<KeyPress-M>", lambda _e: self.toggle_micro())
        self.root.bind("<KeyPress-q>", lambda _e: self.shutdown())
        self.root.bind("<KeyPress-Q>", lambda _e: self.shutdown())
        self.root.bind("<Escape>", lambda _e: self.shutdown())

        self.log.push("[Boot] System Locked - mouse disconnected")
        self.log.push("[Hint] Hold a steady FIST for 2.0s to clutch")
        print("[SYSTEM LOCKED - MOUSE CONTROL DISABLED]", flush=True)
        
        print(
            f"[Init] Desktop {self.engine.screen_w}x{self.engine.screen_h} | "
            f"EMA span={EMA_SPAN} alpha={EMA_ALPHA:.3f}",
            flush=True,
        )

        self.worker = threading.Thread(
            target=capture_loop,
            args=(self.engine, self.state, self.cap),
            daemon=True,
            name="navigator-capture",
        )
        self.worker.start()
        self.root.after(HUD_REFRESH_MS, self._tick)

    def toggle_micro(self) -> None:
        global micro_gesture_mode
        micro_gesture_mode = not micro_gesture_mode
        label = "ON" if micro_gesture_mode else "OFF"
        self.micro_btn.configure(text=f"Micro-Gesture: {label}  (M)")
        self.log.push(f"[Micro] Toggled {label}", key="micro-btn", min_interval=0.2)

    def shutdown(self) -> None:
        with self.state.lock:
            self.state.running = False
        try:
            self.cap.release()
        except Exception:
            pass
        try:
            self.engine.close()
        except Exception:
            pass
        self.root.destroy()

    def _tick(self) -> None:
        global micro_gesture_mode
        with self.state.lock:
            frame = None if self.state.frame_bgr is None else self.state.frame_bgr.copy()
            locked = self.state.locked
            clutch = self.state.clutch_progress
            dwell = self.state.dwell_progress
            fps = self.state.fps
            hand = self.state.hand_present
            lines = list(self.state.log_lines)
            xy = self.state.last_screen_xy

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = HUD_VIDEO_WIDTH / float(w)
            rgb = cv2.resize(rgb, (HUD_VIDEO_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
            image = Image.fromarray(rgb)
            self._photo = ImageTk.PhotoImage(image=image)
            self.video.configure(image=self._photo)

        if locked:
            self.banner.configure(text=STATUS_LOCKED, bg="#b42318", fg="#fff5f5")
        else:
            self.banner.configure(text=STATUS_ACTIVE, bg="#079455", fg="#f6ffed")

        clutch_txt = f"Clutch {clutch * 100:.0f}%" if clutch > 0 else "Clutch idle"
        dwell_txt = f"Dwell {dwell * 100:.0f}%" if dwell > 0 else "Dwell idle"
        hand_txt = "hand OK" if hand else "no hand"
        micro_txt = "MICRO" if micro_gesture_mode else "SWEEP"
        self.meta_lbl.configure(
            text=(
                f"{hand_txt}  |  {micro_txt}  |  {clutch_txt}  |  {dwell_txt}  |  "
                f"cursor {xy[0]},{xy[1]}  |  {fps:.0f} fps"
            )
        )

        self.log_box.configure(state=tk.NORMAL)
        self.log_box.delete("1.0", tk.END)
        self.log_box.insert(tk.END, "\n".join(lines) if lines else "(awaiting events)")
        self.log_box.configure(state=tk.DISABLED)

        if self.state.running:
            self.root.after(HUD_REFRESH_MS, self._tick)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    try:
        NavigatorHUD().run()
    except RuntimeError as exc:
        print(f"[Fatal] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
