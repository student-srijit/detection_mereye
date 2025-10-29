# MarEye Threat Detection API

Production-ready Flask API for detecting threats in marine images and videos using YOLOv8.

## 🚀 Features

- **Image Detection**: Detect threats in uploaded images
- **Video Detection**: Process videos frame-by-frame for threat detection
- **Multi-class Recognition**: Detects mines, submarines, AUVs, ROVs, and divers
- **Annotated Results**: Returns annotated images with bounding boxes
- **CORS Enabled**: Seamlessly integrates with Vercel frontend
- **Production Ready**: Optimized for deployment on Render

## 🛠️ Technology Stack

- **Framework**: Flask 3.0.0
- **ML Model**: YOLOv8 (Ultralytics)
- **Image Processing**: OpenCV, Pillow
- **Server**: Gunicorn
- **Deployment**: Render

## 📋 Prerequisites

- Python 3.11+
- `best.pt` model file (YOLOv8 trained model)
- Render account (for deployment)

## 🏃 Quick Start (Local Development)

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd detection
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Make sure you have the model file
Ensure `best.pt` is in the root directory.

### 5. Run the development server
```bash
python app.py
```

The API will be available at `http://localhost:10000`

## 🌐 Deploy to Render

### Method 1: Using Render Dashboard (Recommended)

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - MarEye Detection API"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Create a New Web Service on Render**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure the service**
   - **Name**: `mareye-threat-detection`
   - **Environment**: `Python 3`
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 300`

4. **Add Environment Variables** (Optional)
   - `PYTHON_VERSION`: `3.11.0`
   - `PORT`: `10000` (Render will override this automatically)

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete (may take 5-10 minutes)

### Method 2: Using render.yaml

If you have `render.yaml` in your repo, Render will automatically detect and use it.

1. Push code to GitHub
2. Connect repository to Render
3. Render will auto-configure using `render.yaml`

## 🔌 API Endpoints

### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "classes": ["Mines", "Submarine", "auv-rov", "divers", "mayin"],
  "supported_formats": {
    "images": ["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
    "videos": ["mp4", "avi", "mov", "mkv", "webm"]
  }
}
```

### 2. Detect Threats in Image
```http
POST /api/detect/image
Content-Type: multipart/form-data

file: <image-file>
```

**Response:**
```json
{
  "success": true,
  "type": "image",
  "filename": "ocean.jpg",
  "threat_count": 2,
  "overall_threat_level": "HIGH",
  "threats": [
    {
      "id": 1,
      "class": "Submarine",
      "confidence": 0.89,
      "confidence_percentage": 89.5,
      "threat_level": "HIGH",
      "bounding_box": {
        "x1": 120.5,
        "y1": 200.3,
        "x2": 350.8,
        "y2": 400.6
      }
    }
  ],
  "annotated_image": "data:image/jpeg;base64,..."
}
```

### 3. Detect Threats in Video
```http
POST /api/detect/video
Content-Type: multipart/form-data

file: <video-file>
frame_interval: 30 (optional, default: 30)
```

**Response:**
```json
{
  "success": true,
  "type": "video",
  "filename": "underwater.mp4",
  "video_metadata": {
    "duration_seconds": 45.5,
    "fps": 30,
    "total_frames": 1365,
    "processed_frames": 46
  },
  "total_detections": 12,
  "overall_threat_level": "MEDIUM",
  "frames_with_threats": [...]
}
```

### 4. Unified Detection Endpoint
```http
POST /api/detect
Content-Type: multipart/form-data

file: <image-or-video-file>
```

Automatically detects file type and processes accordingly.

## 🔗 Frontend Integration

### Update your Vercel frontend

Add this API service file to your frontend:

**File: `lib/api/detectionService.ts`**

```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_DETECTION_API_URL || 'https://your-render-app.onrender.com';

export async function detectThreats(file: File): Promise<DetectionResult> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/detect`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Detection failed');
  }

  return await response.json();
}
```

**Add to `.env.local` in frontend:**
```env
NEXT_PUBLIC_DETECTION_API_URL=https://your-app-name.onrender.com
```

## 📊 Model Information

- **Model**: YOLOv8 (Custom trained)
- **Classes**: 5 threat categories
  - Mines
  - Submarine
  - AUV/ROV
  - Divers
  - Mayin (mines variant)
- **Confidence Threshold**: 0.3 (30%)
- **Input**: Images (up to 50MB) and Videos

## 🔒 Security

- CORS enabled for specific origins only
- File size limits enforced (50MB max)
- Secure file handling with UUID-based naming
- Temporary files automatically cleaned up
- File type validation

## 🐛 Troubleshooting

### Model not loading
- Ensure `best.pt` is in the root directory
- Check file permissions
- Verify file is not corrupted

### Out of memory errors
- Reduce worker count in Procfile
- Process fewer video frames (increase `frame_interval`)
- Upgrade Render plan

### CORS errors
- Check frontend origin is in allowed origins
- Verify API URL in frontend environment variables

### Deployment fails
- Check Render logs for specific errors
- Ensure all dependencies are in `requirements.txt`
- Verify Python version compatibility

## 📝 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `10000` | Server port (Render sets this automatically) |
| `PYTHON_VERSION` | `3.11.0` | Python runtime version |
| `CONFIDENCE_THRESHOLD` | `0.3` | Model detection confidence threshold |

## 🚦 Performance

- **Image Processing**: ~2-5 seconds per image
- **Video Processing**: ~30-60 seconds for 30-second video (depends on frame interval)
- **Concurrent Requests**: Supports 2 workers (configurable)
- **Timeout**: 300 seconds (5 minutes)

## 📞 Support

For issues or questions:
- Check Render deployment logs
- Verify API health endpoint: `https://your-app.onrender.com/health`
- Review frontend console for CORS or network errors

## 📄 License

Part of the MarEye Marine Security Platform

---

**Note**: After deploying to Render, update the frontend environment variable `NEXT_PUBLIC_DETECTION_API_URL` with your Render app URL and redeploy your Vercel frontend.

