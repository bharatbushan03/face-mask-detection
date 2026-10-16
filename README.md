# Face Mask Detection System

A real-time face mask detection system with a modern Streamlit web interface. This application uses OpenCV DNN with a pre-trained SSD model to detect faces and determine whether people are wearing masks.

## Features

- **🎨 Modern UI**: Beautiful gradient design with card-based layout and smooth animations
- **📹 Live Webcam Detection** (Local Only): Real-time camera feed with instant mask detection
- **🖼️ Image Upload**: Upload static images for mask detection
- **🎬 Video Upload**: Analyze videos frame by frame
- **⚙️ Adjustable Thresholds**: Customize confidence and IOU thresholds
- **📊 Statistics Dashboard**: Real-time statistics with beautiful metric cards
- **🎯 Visual Feedback**: Color-coded bounding boxes (Green = Mask, Red = No Mask)
- **✨ Icons & Emojis**: Intuitive visual elements throughout the interface

## Project Structure

```
FaceMaskDetection-master/
├── app.py                 # Main Streamlit application
├── backend/
│   ├── __init__.py
│   └── detector.py        # Face mask detection module
├── models/                # Pre-trained model files
│   ├── face_mask_detection.caffemodel
│   └── face_mask_detection.prototxt
├── utils/                 # Utility functions
│   ├── anchor_generator.py
│   ├── anchor_decode.py
│   └── nms.py
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup Steps

1. **Clone or download the project**

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

This will install:
- opencv-python (4.5.0+)
- opencv-contrib-python (4.5.0+)
- numpy (1.19.0+)
- streamlit (1.20.0+)
- Pillow (8.0.0+)

## Running the Application

### Start the Streamlit App

```bash
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Usage

1. **Select Input Source** (from the sidebar):
   - **Live Webcam**: Real-time detection from your camera
   - **Upload Image**: Upload and analyze static images
   - **Upload Video**: Upload and analyze videos

2. **Adjust Settings** (from the sidebar):
   - **Confidence Threshold**: Minimum confidence score for detection (0.1 - 0.9)
   - **IOU Threshold**: Threshold for Non-Maximum Suppression (0.1 - 0.9)

3. **View Results**:
   - Real-time bounding boxes around detected faces
   - Color coding: Green for masks, Red for no masks
   - Confidence scores displayed above each detection
   - Statistics dashboard showing:
     - Total faces detected
     - Number with masks
     - Number without masks
     - Compliance rate percentage

## Model Information

The system uses a pre-trained SSD (Single Shot Detector) model with the following specifications:

- **Input Size**: 260x260 pixels
- **Backbone**: Lightweight CNN with 8 convolutional layers
- **Total Parameters**: ~1.01M
- **Total Layers**: 24 (including location and classification layers)
- **Output Classes**: 
  - 0: Mask (person wearing mask)
  - 1: NoMask (person not wearing mask)

### Anchor Configuration

| Layer | Feature Map | Anchor Sizes | Aspect Ratios |
|-------|-------------|--------------|---------------|
| 1     | 33x33       | 0.04, 0.056  | 1, 0.62, 0.42 |
| 2     | 17x17       | 0.08, 0.11   | 1, 0.62, 0.42 |
| 3     | 9x9         | 0.16, 0.22   | 1, 0.62, 0.42 |
| 4     | 5x5         | 0.32, 0.45   | 1, 0.62, 0.42 |
| 5     | 3x3         | 0.64, 0.72   | 1, 0.62, 0.42 |

## Deployment to Streamlit Cloud

### Prerequisites

1. **Push code to GitHub**:
   - Create a new GitHub repository
   - Push all project files to the repository
   - Ensure `requirements.txt` and `packages.txt` are included

2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub account
   - Select your repository
   - Main file path: `app.py`
   - Click "Deploy"

### Important Notes for Cloud Deployment

- **Webcam feature**: Live webcam detection will not work on Streamlit Cloud due to browser security restrictions. Use the Image Upload or Video Upload features instead.
- **Model files**: Ensure model files in `models/` directory are committed to the repository
- **Dependencies**: The `packages.txt` file includes OpenCV system dependencies required for the cloud environment

### Local Deployment

For full webcam functionality, run the app locally:

```bash
streamlit run app.py
```

## Troubleshooting

### Camera Not Accessible (Local Only)

If the webcam is not accessible:
- Ensure no other application is using the camera
- Check browser permissions for camera access
- Try restarting the browser
- On Windows, ensure camera drivers are installed

### Model Loading Issues

If the model fails to load:
- Verify that model files exist in the `models/` directory
- Check that `face_mask_detection.caffemodel` and `face_mask_detection.prototxt` are present
- Ensure sufficient disk space and memory

### Performance Issues

If the application is slow:
- Reduce the confidence threshold to filter out weak detections
- Lower the resolution if using a high-resolution camera
- Close other applications to free up system resources

### OpenCV Version Compatibility

The code is compatible with OpenCV 4.5.0 and above. If you encounter issues:
```bash
pip install --upgrade opencv-python opencv-contrib-python
```

## Development

### Backend Module

The backend (`backend/detector.py`) provides:

- `FaceMaskDetector` class for mask detection
- `detect()` method for processing images
- `get_detection_summary()` for statistics

### Frontend

The Streamlit app (`app.py`) provides:

- Webcam integration
- Image/video upload handling
- Real-time visualization
- Interactive controls

## License

This project is open source. Please refer to the LICENSE file for details.
