# Deployment Guide - Render

This guide explains how to deploy the Face Mask Detection app to Render.

## Prerequisites

- A Render account (free at [render.com](https://render.com))
- GitHub account with the code pushed to a repository

## Deployment Steps

### 1. Push Code to GitHub

Ensure your code is pushed to GitHub:
```bash
git add .
git commit -m "Add Render deployment configuration"
git push origin master
```

### 2. Create Render Account

1. Go to [render.com](https://render.com)
2. Sign up for a free account
3. Connect your GitHub account

### 3. Create New Web Service

1. Click **"New +"** button
2. Select **"Web Service"**
3. Connect your GitHub repository (`face-mask-detection`)
4. Configure:

   **Name**: `face-mask-detection`
   
   **Branch**: `master`
   
   **Runtime**: `Python`
   
   **Build Command**: `pip install -r requirements.txt`
   
   **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

### 4. Environment Variables

Add the following environment variable:

- **Key**: `PORT`
- **Value**: `8501`

### 5. Deploy

Click **"Create Web Service"** and wait for the deployment to complete.

## Important Notes

### Camera

Render has no webcam, so `cv2.VideoCapture` cannot open one. The **Camera** option uses the browser camera (`st.camera_input`): the visitor allows the prompt, takes a photo, and the server runs detection on that image.

The site must be opened over HTTPS (Render provides this). The browser will ask for camera permission.

### OpenCV Headless

We use `opencv-python-headless` instead of `opencv-python` because:
- Render doesn't have display/GUI capabilities
- Headless version doesn't require X11 libraries
- Better suited for server environments

### Automatic Deployment

Render will automatically redeploy when you push changes to the `master` branch.

## Troubleshooting

### Build Fails

- Check the build logs in Render dashboard
- Ensure `requirements.txt` is in the repository root
- Verify Python version compatibility

### App Crashes on Startup

- Check the Render logs for errors
- Ensure all model files are in the repository
- Verify the start command is correct

### Camera permission denied

Allow camera access in the browser prompt for the Render site. If it was blocked, reset it from the lock icon in the address bar.

## Accessing Your App

After successful deployment, Render will provide:
- A public URL (e.g., `https://face-mask-detection.onrender.com`)
- HTTPS is automatically enabled
- The app will be live and accessible

## Free Tier Limitations

Render's free tier has:
- 512 MB RAM
- Limited CPU
- Sleeps after 15 minutes of inactivity
- Takes ~30 seconds to wake up

For production use, consider upgrading to a paid plan.

## Cost

- **Free tier**: $0/month (with limitations)
- **Starter**: $7/month (better performance)
- **Standard**: $25/month (production-ready)

## Monitoring

- View logs in Render dashboard
- Monitor resource usage
- Set up alerts for errors
