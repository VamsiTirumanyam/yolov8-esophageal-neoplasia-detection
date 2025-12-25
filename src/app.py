import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np 

# Load the YOLO model
model = YOLO('src/best.pt')

# Set confidence threshold
conf_thresh = 0.25

# Streamlit app
st.title("Esophagus EndoScopy with YoloV8")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Predict using the YOLO model
    results = model.predict(image, stream=True)

    # Iterate over the generator
    for result in results:
        # Extract bounding boxes and classes
        boxes = result.boxes.xyxy.cpu().numpy()
        names = result.names
        classes = result.boxes.cls.cpu().numpy()

        # Iterate over boxes and classes
        for box, cls in zip(boxes, classes):
            r = [int(coord) for coord in box[:4]]  # use only the first 4 coordinates
            class_label = names[cls]  # get the class label from the dictionary

            # Draw boxes on the image
            cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)

            # Display label on the image
            label_text = f"Class: {class_label}"
            cv2.putText(image, label_text, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image in Streamlit
    st.image(image, channels="BGR", caption="Predicted Image", use_column_width=True)

# Note: This code assumes that you have the necessary libraries installed and configured for Streamlit and YOLO.
