import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(page_title="Industrial Defect Inspection", layout="wide")

st.title("🏭 Real-Time PCB Surface Defect Inspection")
st.markdown("Automated edge inspection pipeline running ONNX inference to detect circuit manufacturing anomalies.")

API_URL = "http://127.0.0.1:8000/v1/inspect"

# Real PCB Defect Classes
DEFECT_CLASSES = {
    0: "missing_hole",
    1: "mouse_bite",
    2: "open_circuit",
    3: "short",
    4: "spur",
    5: "spurious_copper"
}

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload an inspection frame...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Component Surface", use_container_width=True)

with col2:
    if uploaded_file is not None:
        if st.button("Run Defect Inspection", type="primary"):
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            with st.spinner("Processing through dynamic batcher & ONNX runtime..."):
                try:
                    response = requests.post(API_URL, files=files)
                    if response.status_code == 200:
                        data = response.json()
                        
                        if data["status"] == "DEFECT_DETECTED":
                            st.error(f"⚠️ Status: {data['status']} ({data['defects_found']} anomalies identified)")
                        else:
                            st.success(f"✅ Status: {data['status']} (Zero defects found)")

                        draw_img = image.copy()
                        draw = ImageDraw.Draw(draw_img)

                        for det in data["detections"]:
                            ymin, xmin, ymax, xmax = det["bbox"]
                            class_name = DEFECT_CLASSES.get(det["class_id"], f"Defect-{det['class_id']}")
                            label = f"{class_name} ({det['confidence']:.2f})"
                            
                            draw.rectangle([xmin, ymin, xmax, ymax], outline="#FF0033", width=3)
                            draw.text((xmin + 4, max(0, ymin - 15)), label, fill="#FF0033")

                        st.image(draw_img, caption="Inspection Results Overlay", use_container_width=True)
                        st.json(data)
                    else:
                        st.error(f"Inference error {response.status_code}: {response.text}")
                except Exception as e:
                    st.error(f"Connection failure: {e}")
