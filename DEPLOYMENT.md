# Deployment Guide

## Deploy to Streamlit Cloud

### Step 1: Prepare Your Repository

1. **Create a GitHub repository** for your project
2. **Push all files** to the repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Sign in with your GitHub account
4. Fill in the deployment form:
   - **Repository**: Select your GitHub repository
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy"**

### Step 3: Verify Deployment

- Streamlit Cloud will automatically install dependencies from `requirements.txt`
- System dependencies from `packages.txt` will be installed
- The app will be available at: `https://YOUR_APP_NAME.streamlit.app`

## Important Notes

### Camera on Cloud

The **Camera** option streams the visitor's browser camera over WebRTC. Detection runs on the server. The site must be served over HTTPS.

### Required Files for Deployment

Ensure these files are in your repository:
- ✅ `app.py` - Main application
- ✅ `requirements.txt` - Python dependencies
- ✅ `packages.txt` - System dependencies (OpenCV)
- ✅ `backend/` - Detection module
- ✅ `models/` - Pre-trained model files
- ✅ `utils/` - Utility functions

### File Size Limits

- Streamlit Cloud has a 1GB repository size limit
- Model files are relatively small (~5MB total)
- No issues expected with current setup

## Local Deployment

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Troubleshooting

### Deployment Fails

1. Check that all files are committed to GitHub
2. Verify `requirements.txt` and `packages.txt` are present
3. Ensure model files in `models/` directory are included
4. Check Streamlit Cloud logs for specific error messages

### App Runs but Detection Fails

1. Verify model files are correctly uploaded
2. Check that `face_mask_detection.caffemodel` and `face_mask_detection.prototxt` exist
3. Review deployment logs for import errors

### OpenCV Issues on Cloud

The `packages.txt` file includes:
```
libopencv-dev
```

This ensures OpenCV system dependencies are installed on the cloud environment.

## Alternative Deployment Options

### Using ngrok (for local development)

1. Install ngrok: https://ngrok.com/download
2. Run ngrok:
   ```bash
   ngrok http 8501
   ```
3. Share the ngrok URL with others (temporary access)

### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t face-mask-detection .
docker run -p 8501:8501 face-mask-detection
```
