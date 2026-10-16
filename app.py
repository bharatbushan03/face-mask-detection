import threading

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.detector import FaceMaskDetector

# Shared with the WebRTC callback thread. Mutate in place; do not rebind.
_detect_lock = threading.Lock()
_cam = {"detector": None, "conf": 0.5, "iou": 0.4}


def _video_frame_callback(frame):
    detector = _cam["detector"]
    if detector is None:
        return frame

    img = frame.to_ndarray(format="rgb24")
    try:
        with _detect_lock:
            out, info = detector.detect(
                img.copy(),
                conf_thresh=_cam["conf"],
                iou_thresh=_cam["iou"],
            )
            summary = detector.get_detection_summary(info)
    except Exception:
        return frame

    text = (
        f"Faces {summary['total_faces']}  "
        f"Mask {summary['with_mask']}  "
        f"No mask {summary['without_mask']}  "
        f"{summary['compliance_rate']:.0f}%"
    )
    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return av.VideoFrame.from_ndarray(out, format="rgb24")


def main():
    st.set_page_config(
        page_title="Face Mask Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Face Mask Detection System")
    st.markdown("""
    This application detects faces and determines whether people are wearing masks.
    Uses OpenCV DNN with a pre-trained SSD model for fast and accurate detection.
    """)

    # Sidebar for settings
    st.sidebar.header("Settings")
    
    # Confidence threshold slider
    conf_thresh = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Minimum confidence score for detection"
    )
    
    # IOU threshold slider
    iou_thresh = st.sidebar.slider(
        "IOU Threshold (NMS)",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.05,
        help="IOU threshold for Non-Maximum Suppression",
        key="iou_thresh"
    )
    
    # Input source selection
    st.sidebar.header("Input Source")
    
    input_source = st.sidebar.radio(
        "Choose Input Source",
        ["Camera", "Upload Image", "Upload Video"]
    )
    
    # Initialize detector
    @st.cache_resource
    def load_detector():
        with st.spinner("Loading face mask detection model..."):
            return FaceMaskDetector()
    
    try:
        detector = load_detector()
    except Exception as e:
        st.error(f"Failed to load detector: {str(e)}")
        return
    
    # Process based on input source
    if input_source == "Camera":
        process_webcam(detector, conf_thresh, iou_thresh)
    elif input_source == "Upload Image":
        process_image_upload(detector, conf_thresh, iou_thresh)
    elif input_source == "Upload Video":
        process_video_upload(detector, conf_thresh, iou_thresh)


def show_detection(detector, image_rgb, conf_thresh, iou_thresh):
    with st.spinner("Detecting faces and masks..."):
        image_with_detections, detection_info = detector.detect(
            image_rgb,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh
        )

    st.subheader("Detection Results")
    st.image(image_with_detections, channels="RGB", width=800)

    summary = detector.get_detection_summary(detection_info)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Faces", summary['total_faces'])
    col2.metric("With Mask", summary['with_mask'])
    col3.metric("Without Mask", summary['without_mask'])
    col4.metric("Compliance Rate", f"{summary['compliance_rate']:.1f}%")

    if detection_info:
        st.subheader("Detailed Detections")
        for i, detection in enumerate(detection_info):
            class_id, conf, xmin, ymin, xmax, ymax = detection
            class_name = "Mask" if class_id == 0 else "No Mask"
            st.write(f"**Face {i+1}:** {class_name} (Confidence: {conf:.2f})")


def process_webcam(detector, conf_thresh, iou_thresh):
    st.header("Camera Detection")
    st.info("Click Start and allow the camera. Your browser sends the live video to the server for detection.")

    _cam["detector"] = detector
    _cam["conf"] = conf_thresh
    _cam["iou"] = iou_thresh

    # ponytail: browser WebRTC, not cv2.VideoCapture. Render has no webcam. STUN only; add a TURN server if the stream stays on Connecting.
    webrtc_streamer(
        key="mask-camera",
        video_frame_callback=_video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        async_processing=True,
    )


def process_image_upload(detector, conf_thresh, iou_thresh):
    st.header("Image Upload Detection")
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        key="image_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                st.error("Failed to decode uploaded image")
                return
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.subheader("Original Image")
            st.image(image_rgb, channels="RGB", width=800)
            show_detection(detector, image_rgb, conf_thresh, iou_thresh)
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")


def process_video_upload(detector, conf_thresh, iou_thresh):
    st.header("Video Upload Detection")
    
    uploaded_file = st.file_uploader(
        "Choose a video...",
        type=['mp4', 'avi', 'mov', 'mkv'],
        key="video_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Save uploaded video temporarily
            temp_video_path = "temp_video.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Open video
            cap = cv2.VideoCapture(temp_video_path)
            
            if not cap.isOpened():
                st.error("Could not open video file")
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)
                return
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            st.info(f"Video loaded: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
            
            # Frame selection slider
            frame_idx = st.slider(
                "Select Frame",
                min_value=0,
                max_value=max(0, total_frames - 1),
                value=0,
                step=1,
                key="frame_slider"
            )
            
            # Process selected frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect
                with st.spinner("Detecting..."):
                    frame_with_detections, detection_info = detector.detect(
                        frame_rgb,
                        conf_thresh=conf_thresh,
                        iou_thresh=iou_thresh
                    )
                
                # Display
                st.image(frame_with_detections, channels="RGB", width=800)
                
                # Statistics
                summary = detector.get_detection_summary(detection_info)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Faces", summary['total_faces'])
                col2.metric("With Mask", summary['with_mask'])
                col3.metric("Without Mask", summary['without_mask'])
                col4.metric("Compliance Rate", f"{summary['compliance_rate']:.1f}%")
            else:
                st.error("Failed to read frame from video")
            
            cap.release()
            
            # Clean up
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            # Clean up on error
            if 'temp_video_path' in locals() and os.path.exists(temp_video_path):
                os.remove(temp_video_path)


if __name__ == "__main__":
    main()
