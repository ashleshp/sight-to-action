"""- tests
- pipeline
"""
import cv2
import numpy as np

from vt_action.pipeline import evaluate_image


def make_demo_image(tmp_path):
    canvas = np.zeros((300, 300, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)
    cv2.circle(canvas, (40, 40), 15, (0, 0, 255), -1)  # red light
    cv2.rectangle(canvas, (140, 160), (200, 230), (0, 0, 255), -1)  # obstacle
    path = tmp_path / "demo.png"
    cv2.imwrite(str(path), canvas)
    return path


def test_pipeline_runs_and_returns_fields(tmp_path):
    img_path = make_demo_image(tmp_path)
    result = evaluate_image(str(img_path))

    assert result["action"] in {"PROCEED", "SLOW_DOWN", "STOP"}
    assert "explanation" in result and result["explanation"]
    assert "scene" in result and "traffic_light" in result["scene"]
    assert "risk" in result and "level" in result["risk"]


def test_missing_image_triggers_fallback(tmp_path):
    missing_path = tmp_path / "nope.png"
    result = evaluate_image(str(missing_path))
    assert result["action"] == "SLOW_DOWN"
    assert "fallback" in result["explanation"].lower()
