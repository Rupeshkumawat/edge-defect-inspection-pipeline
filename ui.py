import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io

st.set_page_config(page_title="Edge Defect Inspection", layout="wide")

st.title("🔍 Edge-Optimized Visual Defect Inspection")
st.markdown("Upload a test frame to run inference through the containerized ONNX microservice.")

API_URL = "http://127.0.0.1:8000/v1/inspect"

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Input Frame", use_container_width=True)

with col2:
    if uploaded_file is not None:
        if st.button("Run Inspection", type="primary"):
            uploaded_file.seek(0)
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            with st.spinner("Processing through micro-batcher..."):
                try:
                    response = requests.post(API_URL, files=files)
                    if response.status_code == 200:
                        data = response.json()
                        st.subheader(f"Status: {data['status']}")
                        st.write(f"**Defects / Objects Found:** {data['defects_found']}")

                        # Draw bounding boxes
                        draw_img = image.copy()
                        draw = ImageDraw.Draw(draw_img)

                        for det in data["detections"]:
                            ymin, xmin, ymax, xmax = det["bbox"]
                            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
                            label = f"Class {det['class_id']}: {det['confidence']:.2f}"
                            draw.text((xmin + 4, max(0, ymin - 12)), label, fill="red")

                        st.image(draw_img, caption="Inference Detections", use_container_width=True)
                        st.json(data)
                    else:
                        st.error(f"Inference server error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")
