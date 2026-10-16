# Face Mask Detection System

A real-time face mask detection system with a modern Streamlit web interface. This application uses OpenCV DNN with a pre-trained SSD model to detect faces and determine whether people are wearing masks.

## Features

- **🎨 Modern UI**: Beautiful gradient design with card-based layout and smooth animations
- **📹 Camera**: Live browser camera with mask detection on each frame
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
   - **Camera**: Click Start and allow the browser camera for live detection
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

## Deployment

You can deploy this application using either of the following platforms:

### Option 1: Deploy to Render (Recommended)

Render provides excellent Python support with a free tier.

#### Quick Setup

1. **Push code to GitHub**:
   ```bash
   git add .
   git commit -m "Add Render deployment configuration"
   git push origin master
   ```

2. **Create Render account** at [render.com](https://render.com)

3. **Create Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Configure:
     - **Runtime**: Python
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
   - Add environment variable: `PORT` = `8501`
   - Click "Create Web Service"

For detailed instructions, see [DEPLOYMENT_RENDER.md](DEPLOYMENT_RENDER.md)

### Option 2: Deploy to Streamlit Cloud

#### Prerequisites

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

- **Camera**: Live video from the visitor's browser. Open the site over HTTPS and allow the camera prompt. The server does not have a webcam.
- **Model files**: Ensure model files in `models/` directory are committed to the repository
- **OpenCV**: Render uses `opencv-python-headless` for server environments without display capabilities

## Troubleshooting

### Camera blocked

Allow camera access for the site in the browser. If the prompt was dismissed, reset it from the lock icon in the address bar and reload.

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
