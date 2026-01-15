"""- streamlit ui
- image upload
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from vt_action.pipeline import evaluate_image


def _write_temp_file(upload) -> Path:
    suffix = Path(upload.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getbuffer())
        return Path(tmp.name)


def _generate_demo_image() -> Path:
    """Create a simple synthetic scene with a red light and obstacle."""
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    cv2.rectangle(canvas, (250, 250), (350, 360), (0, 0, 255), -1)  # obstacle
    cv2.circle(canvas, (100, 80), 20, (0, 0, 255), -1)  # red light
    cv2.putText(canvas, "Demo Scene", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    path = Path(tempfile.mkstemp(suffix=".png")[1])
    cv2.imwrite(str(path), canvas)
    return path


st.set_page_config(page_title="Sight to Action", page_icon="🤖", layout="centered")
st.title("Sight to Action")
st.caption("OpenCV facts → LangGraph agents → Safe decision + explanation")

upload = st.file_uploader("Upload an RGB image", type=["png", "jpg", "jpeg"])
use_demo = st.checkbox("Use demo scene", value=not upload)

image_path = None
if upload:
    image_path = _write_temp_file(upload)
elif use_demo:
    image_path = _generate_demo_image()

if image_path:
    st.image(str(image_path), caption="Input image")
    result = evaluate_image(str(image_path))
    st.subheader("Decision")
    st.markdown(f"**Action:** {result['action']}")
    st.markdown(f"**Explanation:** {result['explanation']}")

    st.subheader("Scene Facts")
    st.json(result["scene"])

    st.subheader("Risk Assessment")
    st.json(result["risk"])
else:
    st.info("Upload an image or use the demo scene to run the pipeline.")
