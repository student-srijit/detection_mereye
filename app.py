#!/usr/bin/env python3
"""
MarEye Threat Detection API
Production-ready Flask API for detecting threats in marine images and videos
Deployed on Render, integrates with Vercel frontend
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import uuid
import cv2
import tempfile
import shutil
from pathlib import Path
from threat_detector import ThreatDetector
import base64
from PIL import Image
import io
import traceback

# Initialize Flask app
app = Flask(__name__)

# Configure CORS for Vercel frontend
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "https://mareye-frontend.vercel.app",
            "https://*.vercel.app"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize threat detector
print("🚀 Initializing Threat Detector...")
detector = ThreatDetector(
    model_path="best.pt",
    confidence_threshold=0.3
)

def allowed_file(filename, allowed_extensions):
    """Check if file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_file_extension(filename):
    """Get file extension"""
    return filename.rsplit('.', 1)[1].lower() if '.' in filename else ''

def process_video(video_path, frame_interval=30):
    """
    Process video and detect threats in frames
    
    Args:
        video_path: Path to video file
        frame_interval: Process every Nth frame (default: 30)
        
    Returns:
        Dict: Video processing results with detections from all frames
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                'success': False,
                'error': 'Failed to open video file'
            }
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        frame_results = []
        frame_count = 0
        processed_count = 0
        
        # Create temp directory for frames
        temp_dir = tempfile.mkdtemp()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame
            if frame_count % frame_interval == 0:
                # Save frame temporarily
                frame_path = os.path.join(temp_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                
                # Detect threats in frame
                result = detector.detect_threats(frame_path)
                
                if result['success'] and result['threat_count'] > 0:
                    frame_results.append({
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps,
                        'threats': result['threats'],
                        'threat_count': result['threat_count'],
                        'threat_level': result['overall_threat_level']
                    })
                
                # Clean up frame
                os.remove(frame_path)
                processed_count += 1
            
            frame_count += 1
        
        cap.release()
        shutil.rmtree(temp_dir)
        
        # Aggregate results
        total_threats = sum(f['threat_count'] for f in frame_results)
        threat_levels = [f['threat_level'] for f in frame_results]
        
        # Determine overall video threat level
        if 'CRITICAL' in threat_levels:
            overall_level = 'CRITICAL'
        elif 'HIGH' in threat_levels:
            overall_level = 'HIGH'
        elif 'MEDIUM' in threat_levels:
            overall_level = 'MEDIUM'
        elif 'LOW' in threat_levels or 'MINIMAL' in threat_levels:
            overall_level = 'LOW'
        else:
            overall_level = 'NONE'
        
        return {
            'success': True,
            'type': 'video',
            'video_metadata': {
                'duration_seconds': round(duration, 2),
                'fps': fps,
                'total_frames': total_frames,
                'processed_frames': processed_count,
                'frame_interval': frame_interval,
                'resolution': f"{width}x{height}"
            },
            'total_detections': len(frame_results),
            'total_threats': total_threats,
            'overall_threat_level': overall_level,
            'frames_with_threats': frame_results,
            'summary': {
                'frames_analyzed': processed_count,
                'frames_with_detections': len(frame_results),
                'detection_rate': round(len(frame_results) / processed_count * 100, 2) if processed_count > 0 else 0
            }
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': f'Video processing error: {str(e)}'
        }

@app.route('/')
def index():
    """API information endpoint"""
    return jsonify({
        'service': 'MarEye Threat Detection API',
        'version': '1.0.0',
        'status': 'operational',
        'endpoints': {
            'health_check': '/health',
            'detect_image': '/api/detect/image (POST)',
            'detect_video': '/api/detect/video (POST)',
            'detect_unified': '/api/detect (POST)',
            'model_info': '/api/model/info (GET)'
        },
        'documentation': 'Send POST request with image/video file to /api/detect'
    })

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'mareye-threat-detection',
        'model_loaded': detector.model is not None,
        'model_path': detector.model_path,
        'confidence_threshold': detector.confidence_threshold,
        'classes': list(detector.class_names.values()) if detector.class_names else [],
        'supported_formats': {
            'images': list(ALLOWED_IMAGE_EXTENSIONS),
            'videos': list(ALLOWED_VIDEO_EXTENSIONS)
        }
    })

@app.route('/api/model/info')
def model_info():
    """Get model information"""
    return jsonify({
        'success': True,
        'model_path': detector.model_path,
        'confidence_threshold': detector.confidence_threshold,
        'classes': detector.class_names,
        'class_count': len(detector.class_names) if detector.class_names else 0
    })

@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    """Detect threats in uploaded image"""
    try:
        # Check if file is present
        if 'file' not in request.files and 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided. Please upload an image.'
            }), 400
        
        # Get file
        file = request.files.get('file') or request.files.get('image')
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_IMAGE_EXTENSIONS)}'
            }), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(filepath)
        
        # Detect threats
        result = detector.detect_threats(filepath)
        
        # Add file type to result
        result['type'] = 'image'
        result['filename'] = filename
        
        # Create annotated image if threats found
        annotated_path = None
        if result['success'] and result.get('threat_count', 0) > 0:
            annotated_filename = f"annotated_{unique_filename}"
            annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
            detector.create_annotated_image(filepath, result, annotated_path)
            
            # Convert annotated image to base64
            if os.path.exists(annotated_path):
                with open(annotated_path, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    result['annotated_image'] = f"data:image/jpeg;base64,{img_data}"
                os.remove(annotated_path)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in detect_image: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/detect/video', methods=['POST'])
def detect_video():
    """Detect threats in uploaded video"""
    try:
        # Check if file is present
        if 'file' not in request.files and 'video' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided. Please upload a video.'
            }), 400
        
        # Get file
        file = request.files.get('file') or request.files.get('video')
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file type
        if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_VIDEO_EXTENSIONS)}'
            }), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save file
        file.save(filepath)
        
        # Get frame interval from request (default: 30)
        frame_interval = int(request.form.get('frame_interval', 30))
        
        # Process video
        result = process_video(filepath, frame_interval)
        result['filename'] = filename
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in detect_video: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/detect', methods=['POST'])
def detect_unified():
    """Unified endpoint for both image and video detection"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Determine file type
        ext = get_file_extension(file.filename)
        
        if ext in ALLOWED_IMAGE_EXTENSIONS:
            # Process as image
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            file.save(filepath)
            result = detector.detect_threats(filepath)
            result['type'] = 'image'
            result['filename'] = filename
            
            # Add annotated image
            if result['success'] and result.get('threat_count', 0) > 0:
                annotated_filename = f"annotated_{unique_filename}"
                annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
                detector.create_annotated_image(filepath, result, annotated_path)
                
                if os.path.exists(annotated_path):
                    with open(annotated_path, 'rb') as img_file:
                        img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        result['annotated_image'] = f"data:image/jpeg;base64,{img_data}"
                    os.remove(annotated_path)
            
            os.remove(filepath)
            return jsonify(result)
            
        elif ext in ALLOWED_VIDEO_EXTENSIONS:
            # Process as video
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            file.save(filepath)
            frame_interval = int(request.form.get('frame_interval', 30))
            result = process_video(filepath, frame_interval)
            result['filename'] = filename
            
            os.remove(filepath)
            return jsonify(result)
            
        else:
            return jsonify({
                'success': False,
                'error': f'Unsupported file type. Allowed: images ({", ".join(ALLOWED_IMAGE_EXTENSIONS)}) or videos ({", ".join(ALLOWED_VIDEO_EXTENSIONS)})'
            }), 400
        
    except Exception as e:
        print(f"Error in detect_unified: {traceback.format_exc()}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024)}MB'
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    print("=" * 60)
    print("🌊 MarEye Threat Detection API")
    print("=" * 60)
    print(f"🚀 Starting server on port {port}")
    print(f"📊 Model: {detector.model_path}")
    print(f"🎯 Confidence threshold: {detector.confidence_threshold}")
    print(f"🔍 Classes: {list(detector.class_names.values())}")
    print("=" * 60)
    
    # Use 0.0.0.0 for Render deployment
    app.run(host='0.0.0.0', port=port, debug=False)

