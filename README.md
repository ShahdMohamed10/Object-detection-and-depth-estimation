# Object Detection and Depth Estimation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.3-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.0.196-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-orange.svg)

A sophisticated computer vision application that combines real-time object detection using YOLOv8 and depth estimation using MiDaS. This web-based system provides an intuitive interface for processing both live video streams and uploaded media.

![Project Demo](https://via.placeholder.com/800x400?text=Object+Detection+and+Depth+Estimation+Demo)

## 🌟 Features

- **Real-time Object Detection**: Identify and classify objects in video streams using YOLOv8
- **Depth Estimation**: Generate depth maps to understand spatial relationships in scenes
- **Combined Visualization**: View object detection and depth estimation results simultaneously
- **Media Upload**: Process images and videos from your local storage
- **Responsive Web Interface**: Modern, user-friendly UI built with Bootstrap 5
- **Live Camera Feed**: Connect to your webcam for real-time processing

## 🛠️ Technologies Used

- **Backend**: Flask, Python 3.8+
- **Computer Vision**: YOLOv8, MiDaS, OpenCV
- **Deep Learning**: PyTorch, Torchvision
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Deployment**: Docker-ready (optional)

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam (for live stream functionality)
- CUDA-compatible GPU (recommended for optimal performance)

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ShahdMohamed10/Object-detection-and-depth-estimation.git
   cd "Object-detection-and-depth-estimation"
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download pre-trained models**:
   - YOLOv8s model is included in the repository
   - MiDaS model will be downloaded automatically on first run

## 🏃‍♂️ Usage

1. **Start the application**:
   ```bash
   python run.py
   ```

2. **Access the web interface**:
   Open your browser and navigate to `http://localhost:5000`

3. **Using the application**:
   - Click "Start Stream" to begin processing from your webcam
   - Use the upload section to process images or videos from your device
   - View results in the combined view, or switch between object detection and depth map tabs

## ⚙️ Configuration

You can customize the application behavior by modifying `config.py`:

- `CAMERA_INDEX`: Select which camera to use (default: 0)
- `YOLO_CONFIDENCE`: Set the confidence threshold for object detection (default: 0.45)
- `FRAME_SKIP`: Process every nth frame to improve performance (default: 2)
- `FRAME_WIDTH` and `FRAME_HEIGHT`: Set the resolution for processing

## 📊 Project Structure

```
object-detection-and-depth-estimation/
├── app/                    # Flask application
│   ├── __init__.py         # App initialization
│   ├── routes.py           # API endpoints
│   ├── models/             # ML model wrappers
│   ├── static/             # Static assets
│   └── templates/          # HTML templates
├── venv/                   # Virtual environment
├── yolov8s.pt              # Pre-trained YOLOv8 model
├── config.py               # Application configuration
├── requirements.txt        # Dependencies
├── run.py                  # Application entry point
└── README.md               # Project documentation
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📬 Contact

Shahd Mohamed - [@ShahdMohamed10](https://github.com/ShahdMohamed10) - [LinkedIn](https://www.linkedin.com/in/shahd-mohamed-123a68277/)

Project Link: [https://github.com/ShahdMohamed10/Object-detection-and-depth-estimation](https://github.com/ShahdMohamed10/Object-detection-and-depth-estimation)

---

Made with ❤️ by Shahd Mohamed

