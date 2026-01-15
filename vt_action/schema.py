"""Scene normalization utilities."""

from typing import Any, Dict, List

DEFAULT_SCENE = {
    "objects": [],
    "traffic_light": "unknown",
    "visibility": "clear",
    "environment": "open area",
}


def normalize_scene(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize and sort scene facts to keep them deterministic."""
    scene = {**DEFAULT_SCENE, **(raw or {})}
    objects: List[Dict[str, str]] = scene.get("objects", []) or []
    scene["objects"] = sorted(
        [
            {
                "type": obj.get("type", "obstacle"),
                "distance": obj.get("distance", "far"),
                "motion": obj.get("motion", "static"),
            }
            for obj in objects
        ],
        key=lambda o: (o["type"], o["distance"], o["motion"]),
    )
    return scene
