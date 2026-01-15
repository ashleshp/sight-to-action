"""- rule agents
- risk decision explain
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal

from . import config

RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
Action = Literal["PROCEED", "SLOW_DOWN", "STOP"]


@dataclass
class RiskOutput:
    level: RiskLevel
    factors: List[str]


def risk_agent(scene: Dict[str, Any]) -> RiskOutput:
    objects = scene.get("objects", []) or []
    visibility = scene.get("visibility", "clear")
    traffic_light = scene.get("traffic_light", "unknown")

    factors: List[str] = []
    level: RiskLevel = "LOW"

    for obj in objects:
        distance = obj.get("distance", "far")
        motion = obj.get("motion", "static")
        obj_type = obj.get("type", "obstacle")

        if distance == "near":
            level = "HIGH"
            factors.append(f"near {obj_type}")
        elif distance == "medium" and level != "HIGH":
            level = "MEDIUM"
            factors.append(f"medium {obj_type}")

        if motion == "moving" and obj_type in {"pedestrian", "vehicle"}:
            if level == "LOW":
                level = "MEDIUM"
            factors.append(f"moving {obj_type}")

    if traffic_light == "red":
        level = "HIGH"
        factors.append("traffic light red")
    elif traffic_light == "unknown" and level == "LOW":
        level = "MEDIUM"
        factors.append("traffic light unknown")

    if visibility == "low" and level != "HIGH":
        level = "MEDIUM"
        factors.append("low visibility")

    # Remove duplicates while preserving order.
    seen = set()
    deduped = []
    for f in factors:
        if f not in seen:
            deduped.append(f)
            seen.add(f)
    return RiskOutput(level=level, factors=deduped)


def decision_agent(scene: Dict[str, Any], risk: RiskOutput) -> Action:
    traffic_light = scene.get("traffic_light", "unknown")
    visibility = scene.get("visibility", "clear")
    has_objects = len(scene.get("objects", []) or []) > 0

    if traffic_light == "red":
        return "STOP"
    if risk.level == "HIGH":
        return "STOP"
    if risk.level == "MEDIUM":
        return "SLOW_DOWN"
    if visibility == "low":
        return "SLOW_DOWN"
    if not has_objects and traffic_light == "green":
        return "PROCEED"
    if not has_objects:
        return "SLOW_DOWN"
    return "PROCEED"


def explanation_agent(action: Action, risk: RiskOutput) -> str:
    if not risk.factors:
        if action == "PROCEED":
            return "Proceed; no hazards detected."
        if action == "SLOW_DOWN":
            return "Slow down as a precaution in low-information conditions."
        return "Stop due to missing or uncertain scene data."

    factors_text = ", ".join(risk.factors)
    if action == "STOP":
        return f"Stop because of {factors_text}."
    if action == "SLOW_DOWN":
        return f"Slow down; watch for {factors_text}."
    return f"Proceed with caution; noted {factors_text}."
