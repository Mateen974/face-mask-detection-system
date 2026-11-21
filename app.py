
import streamlit as st
import torch
import torch.nn as nn
import cv2
import av
import numpy as np
import os

from PIL import Image
from torchvision import models, transforms
from streamlit_webrtc import webrtc_streamer, WebRtcMode


# Application Configuration

st.set_page_config(
    page_title="Face Mask Detection System",
    layout="wide"
)


# Application Styling

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


# Application Header

st.markdown(
    '<div class="main-title">Face Mask Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Real-Time Face Mask Classification Using Deep Learning'
    '</div>',
    unsafe_allow_html=True
)


# Device Configuration

device = torch.device("cpu")


# Load Trained Model

@st.cache_resource
def load_model():

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "best_mask_model_fp16.pth"
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True
    )

    model = models.resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 2)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.float()
    model.to(device)
    model.eval()

    return model, checkpoint["classes"]


try:
    model, classes = load_model()

except Exception as error:
    st.error(f"Unable to load trained model: {error}")
    st.stop()


# Image Preprocessing

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Face Detection Model

cascade_path = os.path.join(
    os.path.dirname(cv2.__file__),
    "data",
    "haarcascade_frontalface_default.xml"
)

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    st.error("OpenCV face detection classifier could not be loaded.")
    st.stop()


# Mask Classification

def predict_mask(face):

    rgb_face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(rgb_face)

    input_tensor = transform(
        pil_image
    ).unsqueeze(0).to(device)

    with torch.inference_mode():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = probabilities.max(dim=1)

    predicted_class = classes[prediction.item()]
    confidence_score = confidence.item() * 100

    return predicted_class, confidence_score


# Face Detection and Prediction

def detect_mask(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        predicted_class, confidence = predict_mask(face)

        if predicted_class == "with_mask":

            color = (0, 180, 0)
            label = f"With Mask: {confidence:.2f}%"

        else:

            color = (0, 0, 220)
            label = f"Without Mask: {confidence:.2f}%"

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            color,
            2
        )

        cv2.putText(
            frame,
            label,
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

    return frame


# Live Camera Processing

def video_frame_callback(frame):

    image = frame.to_ndarray(format="bgr24")

    processed_image = detect_mask(image)

    return av.VideoFrame.from_ndarray(
        processed_image,
        format="bgr24"
    )


# Sidebar

with st.sidebar:

    st.header("Project Information")

    st.write("Model Architecture: ResNet18")
    st.write("Framework: PyTorch")
    st.write("Face Detection: OpenCV")
    st.write("Interface: Streamlit")

    st.divider()

    st.write("Classification Categories")
    st.write("1. With Mask")
    st.write("2. Without Mask")


# Main Application Tabs

camera_tab, upload_tab = st.tabs([
    "Live Camera Detection",
    "Image Classification"
])


# Live Camera Section

with camera_tab:

    st.subheader("Real-Time Face Mask Detection")

    st.write(
        "Start the camera and allow browser permission "
        "to perform real-time face mask detection."
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


# Image Upload Section

with upload_tab:

    st.subheader("Image-Based Mask Detection")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_array = np.array(image)

        frame = cv2.cvtColor(
            image_array,
            cv2.COLOR_RGB2BGR
        )

        result = detect_mask(frame)

        result_image = cv2.cvtColor(
            result,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            result_image,
            caption="Classification Result",
            use_container_width=True
        )


# Footer

st.divider()

st.markdown(
    '<div class="footer">'
    'Face Mask Detection System | '
    'Powered by PyTorch, ResNet18 and OpenCV'
    '</div>',
    unsafe_allow_html=True
)
