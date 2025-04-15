import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load a better YOLOv8 model (change to yolov8n.pt if using low spec machine)
model = YOLO("yolov8m.pt")

# Classes we want to track
vehicle_classes = ['car', 'motorbike', 'bus', 'truck', 'tuktuk']

# Map from COCO to our labels
class_map = {
    'car': 'car',
    'motorcycle': 'motorbike',
    'bus': 'bus',
    'truck': 'truck'
}

# Streamlit UI
st.set_page_config(page_title="Real-Time Traffic Monitor", layout="wide")
st.title("🚦 Real-Time Traffic Monitoring App")
st.sidebar.title("🎥 Video Source")

# Select input
source_type = st.sidebar.radio("Select Input Source:", ["Webcam", "IP Camera", "Upload Video"])
video_source = None
uploaded_video = None

if source_type == "Webcam":
    st.sidebar.info("Using your webcam.")
    video_source = 0

elif source_type == "IP Camera":
    ip_url = st.sidebar.text_input("Enter IP camera URL (e.g., http://192.168.0.101:8080/video):")
    if ip_url:
        video_source = ip_url

elif source_type == "Upload Video":
    uploaded_video = st.sidebar.file_uploader("Upload video file", type=["mp4", "avi", "mov"])
    if uploaded_video:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())
        video_source = temp_file.name

start_button = st.sidebar.button("🚀 Start Monitoring")
frame_placeholder = st.empty()
count_placeholder = st.empty()

if start_button and video_source is not None:
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        st.error("❌ Unable to open video source.")
    else:
        st.success("✅ Monitoring started. Press 'Stop' to end.")
        stop_button = st.button("🛑 Stop")

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.warning("🔁 Stream ended or unreadable frame.")
                break

            results = model(frame, stream=False)[0]
            annotated_frame = frame.copy()
            vehicle_count = {cls: 0 for cls in vehicle_classes}

            for box in results.boxes:
                cls_id = int(box.cls[0])
                coco_class = model.names[cls_id]

                # Check class map
                if coco_class in class_map:
                    # Get bounding box
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    width = xyxy[2] - xyxy[0]
                    height = xyxy[3] - xyxy[1]
                    area = width * height

                    # Tuktuk override if small truck
                    if coco_class == 'truck' and area < 5000:
                        label = 'tuktuk'
                    else:
                        label = class_map.get(coco_class)

                    # Count and draw
                    if label:
                        vehicle_count[label] += 1
                        cv2.rectangle(annotated_frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, label, (xyxy[0], xyxy[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display frame
            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(annotated_frame_rgb, channels="RGB", use_container_width=True)

            # Display counts
            #count_placeholder.markdown("### 📊 Vehicle Counts:")
            #for vehicle, count in vehicle_count.items():
             #   count_placeholder.markdown(f"- **{vehicle.capitalize()}**: {count}")

        cap.release()
