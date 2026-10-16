import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.detector import FaceMaskDetector


def main():
    st.set_page_config(
        page_title="Face Mask Detection",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("Face Mask Detection System")
    st.markdown("""
    This application detects faces in real-time and determines whether people are wearing masks.
    Uses OpenCV DNN with a pre-trained SSD model for fast and accurate detection.
    """)
    
    # Check if running on cloud (multiple methods)
    is_cloud = (
        os.environ.get('STREAMLIT_SERVER_TYPE') == 'cloud' or
        os.environ.get('STREAMLIT_RUNTIME_ENV') == 'cloud' or
        'STREAMLIT_SHARING_BASE_URL' in os.environ
    )
    
    # Show cloud warning
    if is_cloud:
        st.warning("""
        **🌐 Running on Streamlit Cloud**
        
        The webcam feature is not available on cloud deployments. Please use the **Image Upload** or **Video Upload** features instead.
        
        For webcam functionality, run the app locally:
        ```bash
        streamlit run app.py
        ```
        """)
    
    # Initialize session state
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    if 'should_stop' not in st.session_state:
        st.session_state.should_stop = False
    
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
    
    if is_cloud:
        input_source = st.sidebar.radio(
            "Choose Input Source",
            ["Upload Image", "Upload Video"]
        )
    else:
        input_source = st.sidebar.radio(
            "Choose Input Source",
            ["Live Webcam", "Upload Image", "Upload Video"]
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
    if input_source == "Live Webcam":
        process_webcam(detector, conf_thresh, iou_thresh)
    elif input_source == "Upload Image":
        process_image_upload(detector, conf_thresh, iou_thresh)
    elif input_source == "Upload Video":
        process_video_upload(detector, conf_thresh, iou_thresh)


def process_webcam(detector, conf_thresh, iou_thresh):
    st.header("📷 Live Webcam Detection")
    st.info("Click 'Start Camera' to begin detection. Press 'Stop' to end.")
    
    # Camera index selector
    camera_index = st.selectbox("Select Camera", [0, 1, 2], index=0, help="Try different indices if camera doesn't work")
    
    # Placeholder for video feed
    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    
    # Start/Stop button
    start_camera = st.button("Start Camera", key="start_webcam")
    stop_camera = st.button("Stop Camera", key="stop_webcam")
    
    if start_camera:
        st.session_state.camera_running = True
        st.session_state.should_stop = False
    
    if stop_camera:
        st.session_state.camera_running = False
        st.session_state.should_stop = True
        st.info("Camera stopped.")
        return
    
    # Process video if camera is running
    if st.session_state.camera_running:
        # Try to open camera with different backends
        cap = None
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_ANY]
        
        for backend in backends:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                st.success(f"Camera opened with backend: {backend}")
                break
        
        if cap is None or not cap.isOpened():
            st.error("Could not open webcam. Please try:")
            st.error("- Select a different camera index from the dropdown")
            st.error("- Close other applications using the camera")
            st.error("- Check camera permissions in your system settings")
            st.error("- Ensure camera drivers are installed")
            st.session_state.camera_running = False
            if cap is not None:
                cap.release()
            return
        
        st.success("Camera started! Processing frames...")
        
        frame_count = 0
        start_time = time.time()
        
        while st.session_state.camera_running and not st.session_state.should_stop:
            ret, frame = cap.read()
            
            if not ret:
                st.error("Failed to read frame from camera")
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces and masks
            frame_with_detections, detection_info = detector.detect(
                frame_rgb,
                conf_thresh=conf_thresh,
                iou_thresh=iou_thresh
            )
            
            # Get statistics
            summary = detector.get_detection_summary(detection_info)
            
            # Display frame
            frame_placeholder.image(frame_with_detections, channels="RGB", width=800)
            
            # Display statistics
            with stats_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Faces", summary['total_faces'])
                col2.metric("With Mask", summary['with_mask'])
                col3.metric("Without Mask", summary['without_mask'])
                col4.metric("Compliance Rate", f"{summary['compliance_rate']:.1f}%")
            
            frame_count += 1
            
            # Calculate FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                st.caption(f"FPS: {fps:.2f}")
            
            # Small delay to prevent UI freeze
            time.sleep(0.01)
        
        cap.release()
        st.session_state.camera_running = False
        st.info("Camera stopped.")


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
            
            # Display original image
            st.subheader("Original Image")
            st.image(image_rgb, channels="RGB", width=800)
            
            # Detect
            with st.spinner("Detecting faces and masks..."):
                image_with_detections, detection_info = detector.detect(
                    image_rgb,
                    conf_thresh=conf_thresh,
                    iou_thresh=iou_thresh
                )
            
            # Display results
            st.subheader("Detection Results")
            st.image(image_with_detections, channels="RGB", width=800)
            
            # Show statistics
            summary = detector.get_detection_summary(detection_info)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Faces", summary['total_faces'])
            col2.metric("With Mask", summary['with_mask'])
            col3.metric("Without Mask", summary['without_mask'])
            col4.metric("Compliance Rate", f"{summary['compliance_rate']:.1f}%")
            
            # Show detailed detections
            if detection_info:
                st.subheader("Detailed Detections")
                for i, detection in enumerate(detection_info):
                    class_id, conf, xmin, ymin, xmax, ymax = detection
                    class_name = "Mask" if class_id == 0 else "No Mask"
                    st.write(f"**Face {i+1}:** {class_name} (Confidence: {conf:.2f})")
        
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
