# Setup Instructions

## Prerequisites

Before setting up the Integrated Navigation System, ensure you have the following prerequisites installed:

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)
- CUDA-compatible GPU (recommended for optimal performance)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/integrated-navigation-system.git
   cd integrated-navigation-system
   ```

2. **Set up a virtual environment (recommended)**

   ```bash
   # Create a virtual environment
   python -m venv venv
   
   # Activate the virtual environment
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional dependencies for GPU support (optional)**

   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
   ```

## Configuration

1. **Camera Configuration**

   Edit the `config/camera_config.yaml` file to match your camera parameters:

   ```yaml
   camera:
     width: 1920
     height: 1080
     focal_length: 1000
     height: 1.5  # camera height in meters
     pitch: 0.1   # camera pitch in radians
     yaw: 0.0     # camera yaw in radians
   ```

2. **Map Configuration**

   Edit the `config/map_config.yaml` file to set up map integration:

   ```yaml
   map:
     source: "osm"  # Options: "osm", "custom"
     api_key: "your_api_key_here"  # Only needed for some map providers
     cache_dir: "data/maps/cache"
     default_region: [37.7749, -122.4194]  # Default lat, lon
   ```

3. **Priority Configuration**

   Edit the `config/priority_config.yaml` file to customize object priorities:

   ```yaml
   priorities:
     pedestrian: 10
     traffic_light: 8
     stop_sign: 8
     vehicle: 6
     road_sign: 5
     lane_marking: 4
     construction: 7
   ```

## Download Pre-trained Models

1. **Object Detection Models**

   ```bash
   # Create models directory if it doesn't exist
   mkdir -p data/models
   
   # Download YOLOv5 model
   wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt -P data/models/
   ```

2. **Lane Detection Model**

   ```bash
   wget https://github.com/example/lane-detection-model/releases/download/v1.0/lane_detection.pt -P data/models/
   ```

## Verify Installation

Run the test suite to verify that everything is set up correctly:

```bash
python -m pytest tests/
```

If all tests pass, the system is correctly installed and configured.

## Troubleshooting

### Common Issues

1. **PyYAML Import Error**

   If you encounter `Import "yaml" could not be resolved from source`, run:
   
   ```bash
   pip install pyyaml
   ```

2. **GPU Not Detected**

   If CUDA isn't being detected, verify your GPU drivers are installed and compatible with the PyTorch version.

3. **Map Data Not Loading**

   Ensure you have internet connectivity for online map sources, or verify that your custom map data is correctly formatted.

4. **Camera Connection Issues**

   Check that your camera device is properly connected and that you have the correct permissions to access it.

For additional help, check the project's GitHub issues or contact the development team.