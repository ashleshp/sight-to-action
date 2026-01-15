"""- vision heuristics
- opencv pipeline
"""

from dataclasses import dataclass
from typing import List, Literal, Tuple

import cv2
import numpy as np

from . import config

Distance = Literal["near", "medium", "far"]
Motion = Literal["static", "moving"]
ObjectType = Literal["pedestrian", "vehicle", "obstacle"]
TrafficLight = Literal["red", "green", "unknown"]
Visibility = Literal["clear", "low"]
Environment = Literal["urban", "indoor", "open area"]


@dataclass
class DetectedObject:
    type: ObjectType
    distance: Distance
    motion: Motion
    bbox: Tuple[int, int, int, int]


def _load_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return image


def _infer_visibility(gray: np.ndarray) -> Visibility:
    mean_brightness = float(np.mean(gray))
    return "low" if mean_brightness < config.VISIBILITY_BRIGHTNESS_THRESHOLD else "clear"


def _detect_traffic_light(image: np.ndarray) -> TrafficLight:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_lower1 = np.array([0, 70, 50])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 70, 50])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([90, 255, 255])

    red_mask = cv2.inRange(hsv, red_lower1, red_upper1) + cv2.inRange(hsv, red_lower2, red_upper2)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    red_score = float(cv2.countNonZero(red_mask) + 1)
    green_score = float(cv2.countNonZero(green_mask) + 1)

    if red_score / green_score > config.TRAFFIC_RED_THRESHOLD:
        return "red"
    if green_score / red_score > config.TRAFFIC_GREEN_THRESHOLD:
        return "green"
    return "unknown"


def _classify_distance(area: float) -> Distance:
    if area >= config.CONTOUR_AREA_NEAR:
        return "near"
    if area >= config.CONTOUR_AREA_MEDIUM:
        return "medium"
    return "far"


def _classify_object_type(width: float, height: float, area: float) -> ObjectType:
    aspect = width / float(height or 1)
    if area < config.CONTOUR_AREA_MEDIUM * 0.8:
        return "pedestrian"
    if aspect > 1.2:
        return "vehicle"
    return "obstacle"


def _detect_objects(gray: np.ndarray) -> List[DetectedObject]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objects: List[DetectedObject] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < config.CONTOUR_AREA_MEDIUM * 0.5:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        distance = _classify_distance(area)
        obj_type = _classify_object_type(w, h, area)
        objects.append(DetectedObject(type=obj_type, distance=distance, motion="static", bbox=(x, y, w, h)))
    return objects


def _infer_environment(objects: List[DetectedObject], traffic_light: TrafficLight) -> Environment:
    if traffic_light != "unknown":
        return "urban"
    if len(objects) > 2:
        return "urban"
    return "open area"


def analyze_image(path: str) -> dict:
    image = _load_image(path)
    return analyze_frame(image)


def analyze_frame(image: np.ndarray) -> dict:
    if image is None or not hasattr(image, "shape"):
        raise ValueError("Invalid image frame provided.")
    return _analyze(image)


def _analyze(image: np.ndarray) -> dict:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    visibility = _infer_visibility(gray)
    traffic_light = _detect_traffic_light(image)
    objects = _detect_objects(gray)
    environment = _infer_environment(objects, traffic_light)

    return {
        "objects": [
            {"type": obj.type, "distance": obj.distance, "motion": obj.motion} for obj in objects
        ],
        "traffic_light": traffic_light,
        "visibility": visibility,
        "environment": environment,
    }
