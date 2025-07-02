"""Hand tracking using MediaPipe for gesture recognition and 3D positioning."""

import threading
from typing import Optional, Tuple, Any, Deque
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
import logging

logger = logging.getLogger(__name__)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Thread-local storage for MediaPipe instances
_thread_local = threading.local()

# Hand landmark indices
INDEX_FINGER_TIP = 8
THUMB_TIP = 4
MIDDLE_FINGER_TIP = 12
RING_FINGER_TIP = 16
PINKY_TIP = 20


def _get_thread_hands_detector():
    """Get thread-local MediaPipe hands detector."""
    if (
        not hasattr(_thread_local, "hands_detector")
        or _thread_local.hands_detector is None
    ):
        _thread_local.hands_detector = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.25,
        )
    return _thread_local.hands_detector


def get_mediapipe_hand_data(
    frame: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[Any]]:
    """Detect index finger tip using MediaPipe (convenience function).

    Args:
        frame: Input image as BGR numpy array

    Returns:
        Tuple of (index_finger_tip_position, mediapipe_results)
    """
    hands_detector = _get_thread_hands_detector()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands_detector.process(rgb)

    index_finger_tip_pos = None
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        tip_landmark = hand_landmarks.landmark[INDEX_FINGER_TIP]
        h, w = frame.shape[:2]
        index_finger_tip_pos = np.array(
            [tip_landmark.x * w, tip_landmark.y * h], dtype=np.float32
        )

    return index_finger_tip_pos, results


def draw_hand_landmarks(
    image: np.ndarray,
    results: Any,
    connections_color: Tuple[int, int, int] = (0, 255, 0),
    landmarks_color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Draw hand landmarks and connections on image.

    Args:
        image: Input image to draw on
        results: MediaPipe results object
        connections_color: Color for hand connections (BGR)
        landmarks_color: Color for landmarks (BGR)

    Returns:
        Image with drawn landmarks
    """
    annotated_image = image.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

    return annotated_image


def is_wave_gesture(
    x_coordinates: deque,
    amplitude_threshold: float = 0.2,
    min_coordinates: int = 10,
    min_velocity_points: int = 10,
    min_zero_crossings: int = 5,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Detect wave gesture from x-coordinate trail.

    Args:
        x_coordinates: Deque of x-coordinate values
        amplitude_threshold: Minimum amplitude for wave detection
        min_coordinates: Minimum number of coordinate points needed
        min_velocity_points: Minimum velocity points for analysis
        min_zero_crossings: Minimum zero crossings for wave detection

    Returns:
        Tuple of (is_wave_detected, velocity_array)
    """
    if len(x_coordinates) < min_coordinates:
        return False, None

    x_array = np.array(list(x_coordinates))
    velocity = np.diff(x_array)

    if len(velocity) < min_velocity_points:
        return False, None

    # Find zero crossings in velocity (direction changes)
    zero_crossings_indices = np.where(
        np.sign(velocity[:-1]) * np.sign(velocity[1:]) < 0
    )[0]
    num_zero_crossings = len(zero_crossings_indices)

    # Calculate amplitude
    amplitude = x_array.max() - x_array.min()

    # Detect wave
    wave_detected = (
        num_zero_crossings >= min_zero_crossings and amplitude > amplitude_threshold
    )

    return wave_detected, velocity if wave_detected else None


def update_landmark_trail(
    current_trail: deque,
    mediapipe_results: Any,
    landmark_id: int,
    last_seen_time: Optional[float],
    current_time: float,
    timeout_duration: float = 2.0,
) -> Tuple[Optional[float], bool, Optional[float]]:
    """Update landmark trail with timeout management.

    Args:
        current_trail: Deque to store landmark coordinates
        mediapipe_results: MediaPipe results object
        landmark_id: Landmark index to track
        last_seen_time: Last time landmark was detected
        current_time: Current timestamp
        timeout_duration: Time after which to clear trail if no detection

    Returns:
        Tuple of (updated_last_seen_time, trail_was_cleared, extracted_coordinate)
    """
    trail_cleared = False
    extracted_coord = None

    if mediapipe_results and mediapipe_results.multi_hand_landmarks:
        hand_landmarks = mediapipe_results.multi_hand_landmarks[0]
        try:
            extracted_coord = hand_landmarks.landmark[landmark_id].x
            current_trail.append(extracted_coord)
            updated_last_seen_time = current_time
        except IndexError:
            logger.warning(f"Warning: Landmark index {landmark_id} is invalid.")
            updated_last_seen_time = last_seen_time
    else:
        updated_last_seen_time = last_seen_time
        if last_seen_time is not None and (
            current_time - last_seen_time > timeout_duration
        ):
            if current_trail:
                current_trail.clear()
                trail_cleared = True
            updated_last_seen_time = None

    return updated_last_seen_time, trail_cleared, extracted_coord


def close_mediapipe_hands():
    """Close all MediaPipe hands detector instances."""
    if hasattr(_thread_local, "hands_detector") and _thread_local.hands_detector:
        _thread_local.hands_detector.close()
        _thread_local.hands_detector = None
