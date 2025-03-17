from flask import render_template, Response, jsonify, request, url_for
from app import app
from app.camera import Camera
import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Configure upload folder
UPLOAD_FOLDER = 'app/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize models with weights_only=True
yolo_model = YOLO('yolov8s.pt')

# Initialize MiDaS for depth estimation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
    midas.to(device)
    midas.eval()

    # Load transforms
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
    transform = midas_transforms.small_transform
except Exception as e:
    print(f"Error loading MiDaS model: {e}")
    # Fallback to a simpler approach if MiDaS fails to load
    midas = None
    transform = None

def process_frame(frame):
    # Make a copy of the original frame
    original_frame = frame.copy()
    
    # Object Detection
    yolo_results = yolo_model(frame)
    detection_frame = yolo_results[0].plot()
    
    # Depth Estimation
    if midas is not None and transform is not None:
        try:
            input_batch = transform(original_frame).to(device)
            
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=original_frame.shape[:2],
                    mode='bicubic',
                    align_corners=False
                ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
            depth_map = (depth_map * 255).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
            
            # Combine object detection and depth map
            combined_img = cv2.addWeighted(detection_frame, 0.7, depth_colored, 0.3, 0)
            
            return {
                'combined': combined_img,
                'detection': detection_frame,
                'depth': depth_colored
            }
        except Exception as e:
            print(f"Error in depth estimation: {e}")
            return {
                'combined': detection_frame,
                'detection': detection_frame,
                'depth': original_frame  # Fallback to original frame if depth fails
            }
    else:
        # If MiDaS is not available, return detection only
        return {
            'combined': detection_frame,
            'detection': detection_frame,
            'depth': original_frame  # Fallback to original frame
        }

camera = Camera()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(camera.generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream')
def start_stream():
    return camera.start_stream()

@app.route('/stop_stream')
def stop_stream():
    return camera.stop_stream()

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'})
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the uploaded file
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Process image
            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get all processed frames
            processed_results = process_frame(img)
            
            # Convert back to BGR for saving
            combined_img = cv2.cvtColor(processed_results['combined'], cv2.COLOR_RGB2BGR)
            detection_img = cv2.cvtColor(processed_results['detection'], cv2.COLOR_RGB2BGR)
            depth_img = cv2.cvtColor(processed_results['depth'], cv2.COLOR_RGB2BGR)
            
            # Save all processed images
            combined_filename = 'combined_' + filename
            detection_filename = 'detection_' + filename
            depth_filename = 'depth_' + filename
            
            combined_filepath = os.path.join(app.config['UPLOAD_FOLDER'], combined_filename)
            detection_filepath = os.path.join(app.config['UPLOAD_FOLDER'], detection_filename)
            depth_filepath = os.path.join(app.config['UPLOAD_FOLDER'], depth_filename)
            
            cv2.imwrite(combined_filepath, combined_img)
            cv2.imwrite(detection_filepath, detection_img)
            cv2.imwrite(depth_filepath, depth_img)
            
            # Generate URLs for all images
            combined_url = url_for('static', filename=f'uploads/{combined_filename}')
            detection_url = url_for('static', filename=f'uploads/{detection_filename}')
            depth_url = url_for('static', filename=f'uploads/{depth_filename}')
            
            return jsonify({
                'status': 'success',
                'combined_url': combined_url,
                'detection_url': detection_url,
                'depth_url': depth_url
            })
        else:
            # For video files, we'll process it frame by frame
            result_url = url_for('process_video', filename=filename)
            
            return jsonify({
                'status': 'success',
                'combined_url': result_url,
                'detection_url': result_url + '?mode=detection',
                'depth_url': result_url + '?mode=depth'
            })
    
    return jsonify({'status': 'error', 'message': 'File type not allowed'})

@app.route('/process_video/<filename>')
def process_video(filename):
    mode = request.args.get('mode', 'combined')
    
    def generate_frames():
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cap = cv2.VideoCapture(video_path)
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_results = process_frame(frame)
            
            # Select the appropriate output based on mode
            if mode == 'detection':
                output_frame = processed_results['detection']
            elif mode == 'depth':
                output_frame = processed_results['depth']
            else:  # combined or any other value
                output_frame = processed_results['combined']
                
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        cap.release()
    
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')