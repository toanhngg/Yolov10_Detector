import streamlit as st
import cv2
print(cv2.__version__)

import numpy as np
from ultralytics import YOLOv10
import supervision as sv
# Tải mô hình YOLO
model = YOLOv10('D:\\Download\\best.pt')

# Thiết lập giao diện Streamlit
st.title("YOLOv10 Object Detection")
st.write("Upload an image to detect objects using YOLOv10.")

# Tải ảnh từ người dùng
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Đọc ảnh từ người dùng
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Hiển thị ảnh gốc
    st.image(image, channels="BGR", caption="Uploaded Image")

    # Thực hiện dự đoán
    results = model(image)[0]
   # detections = results.pandas().xyxy[0].to_dict(orient="records")
    detections = sv.Detections.from_ultralytics(results)
    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

      # Hiển thị ảnh với bounding box
st.image(annotated_image, channels="BGR", caption=f"Detected Image={detections}")
