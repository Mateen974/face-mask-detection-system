
import os
import threading

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import av
import mediapipe as mp

from PIL import Image
from torchvision import models, transforms
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Face Mask Detection System",
    page_icon=None,
    layout="wide"
)


# ==========================================
# 2. APPLICATION STYLING
# ==========================================

st.markdown("""
<style>

.main-title {
    font-size: 32px;
    font-weight: 700;
    color: #1F2937;
    text-align: center;
    margin-bottom: 8px;
}

.sub-title {
    text-align: center;
    color: #6B7280;
    font-size: 16px;
    margin-bottom: 30px;
}

.footer {
    text-align: center;
    color: #9CA3AF;
    font-size: 13px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. APPLICATION HEADER
# ==========================================

st.markdown(
    '<div class="main-title">'
    'Face Mask Detection System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Real-Time Face Mask Classification Using Deep Learning'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# 4. DEVICE CONFIGURATION
# ==========================================

device = torch.device("cpu")

torch.set_num_threads(2)


# ==========================================
# 5. MODEL LOADING
# ==========================================

@st.cache_resource
def load_model():

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    model_path = os.path.join(
        base_dir,
        "best_mask_model_fp16.pth"
    )

    if not os.path.isfile(model_path):

        raise FileNotFoundError(
            "Model file not found. "
            "Please upload best_mask_model_fp16.pth "
            "to the GitHub repository."
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    model = models.resnet18(
        weights=None
    )

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.float()

    model.eval()

    return model, checkpoint["classes"]


try:

    model, classes = load_model()

except Exception as error:

    st.error(
        f"Model loading error: {error}"
    )

    st.stop()


# ==========================================
# 6. IMAGE PREPROCESSING
# ==========================================

transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

])


# ==========================================
# 7. MEDIAPIPE FACE DETECTOR
# ==========================================

@st.cache_resource
def load_face_detector():

    return mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )


face_detector = load_face_detector()

detector_lock = threading.Lock()


# ==========================================
# 8. MASK PREDICTION
# ==========================================

def predict_mask(face):

    rgb_face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb_face
    )

    input_tensor = transform(
        pil_image
    ).unsqueeze(0)

    with torch.inference_mode():

        output = model(input_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, prediction = probabilities.max(
            dim=1
        )

    predicted_class = classes[
        prediction.item()
    ]

    confidence_score = (
        confidence.item() * 100
    )

    return predicted_class, confidence_score


# ==========================================
# 9. FACE DETECTION
# ==========================================

def detect_mask(frame):

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    with detector_lock:

        results = face_detector.process(
            rgb_frame
        )

    height, width, _ = frame.shape

    if results.detections:

        for detection in results.detections:

            bbox = (
                detection.location_data
                .relative_bounding_box
            )

            x1 = max(
                0,
                int(bbox.xmin * width)
            )

            y1 = max(
                0,
                int(bbox.ymin * height)
            )

            x2 = min(
                width,
                int((bbox.xmin + bbox.width) * width)
            )

            y2 = min(
                height,
                int((bbox.ymin + bbox.height) * height)
            )

            if x2 <= x1 or y2 <= y1:
                continue

            face = frame[
                y1:y2,
                x1:x2
            ]

            if face.size == 0:
                continue

            predicted_class, confidence = predict_mask(
                face
            )

            if predicted_class == "with_mask":

                color = (0, 180, 0)

                label = (
                    f"With Mask: {confidence:.1f}%"
                )

            else:

                color = (0, 0, 220)

                label = (
                    f"Without Mask: {confidence:.1f}%"
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

    return frame


# ==========================================
# 10. CAMERA CALLBACK
# ==========================================

def video_frame_callback(frame):

    image = frame.to_ndarray(
        format="bgr24"
    )

    processed_image = detect_mask(
        image
    )

    return av.VideoFrame.from_ndarray(
        processed_image,
        format="bgr24"
    )


# ==========================================
# 11. SIDEBAR
# ==========================================

with st.sidebar:

    st.header("Project Information")

    st.write("Model: ResNet18")

    st.write("Framework: PyTorch")

    st.write("Face Detection: MediaPipe")

    st.write("Interface: Streamlit")

    st.divider()

    st.subheader("Classification Classes")

    st.write("With Mask")

    st.write("Without Mask")


# ==========================================
# 12. APPLICATION TABS
# ==========================================

camera_tab, upload_tab = st.tabs([

    "Live Camera Detection",

    "Image Classification"

])


# ==========================================
# 13. LIVE CAMERA
# ==========================================

with camera_tab:

    st.subheader(
        "Real-Time Face Mask Detection"
    )

    st.write(
        "Start the camera and allow browser "
        "permission to detect face masks."
    )

    webrtc_streamer(

        key="face-mask-detection",

        mode=WebRtcMode.SENDRECV,

        video_frame_callback=video_frame_callback,

        media_stream_constraints={
            "video": True,
            "audio": False
        },

        async_processing=True

    )


# ==========================================
# 14. IMAGE UPLOAD
# ==========================================

with upload_tab:

    st.subheader(
        "Image-Based Mask Detection"
    )

    uploaded_file = st.file_uploader(

        "Choose an image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]

    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_array = np.array(
            image
        )

        frame = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        result = detect_mask(
            frame
        )

        result_image = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            result_image,
            caption="Face Mask Detection Result",
            use_container_width=True
        )


# ==========================================
# 15. FOOTER
# ==========================================

st.divider()

st.markdown(
    '<div class="footer">'
    'Face Mask Detection System | '
    'PyTorch | ResNet18 | MediaPipe | OpenCV'
    '</div>',
    unsafe_allow_html=True
)
