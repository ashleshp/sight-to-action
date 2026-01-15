"""- pipeline
- scene to decision
"""

from typing import Any, Dict

from . import config, schema, vision
from .agents import RiskOutput
from .graph import run_graph


def _run(scene: Dict[str, Any]) -> Dict[str, Any]:
    scene_normalized = schema.normalize_scene(scene)
    graph_result = run_graph(scene_normalized)
    risk = graph_result.get("risk")
    action = graph_result.get("action", config.DEFAULT_ACTION_ON_MISSING)
    explanation = graph_result.get("explanation", "")

    risk_payload = (
        {"level": risk.level, "factors": risk.factors}
        if isinstance(risk, RiskOutput)
        else {"level": "LOW", "factors": []}
    )
    return {
        "scene": scene_normalized,
        "risk": risk_payload,
        "action": action,
        "explanation": explanation,
    }


def evaluate_image(path: str) -> Dict[str, Any]:
    try:
        raw_scene = vision.analyze_image(path)
        return _run(raw_scene)
    except Exception:
        return {
            "scene": {},
            "risk": {"level": "HIGH", "factors": ["vision failure"]},
            "action": config.DEFAULT_ACTION_ON_ERROR,
            "explanation": "Fallback decision because processing failed.",
        }


def evaluate_frame(frame) -> Dict[str, Any]:
    try:
        raw_scene = vision.analyze_frame(frame)
        return _run(raw_scene)
    except Exception:
        return {
            "scene": {},
            "risk": {"level": "HIGH", "factors": ["vision failure"]},
            "action": config.DEFAULT_ACTION_ON_ERROR,
            "explanation": "Fallback decision because processing failed.",
        }
