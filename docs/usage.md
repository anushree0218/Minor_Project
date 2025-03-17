# Usage Examples

## Basic Usage

The Integrated Navigation System can be used in several different configurations depending on your needs. Below are common usage examples to help you get started.

### Running the Main System

To start the complete navigation system with default settings:

```bash
python src/main.py
```

This will initialize all components: perception, mapping, and navigation using the default configuration files.

### Using Command-line Arguments

The system supports various command-line arguments to customize its behavior:

```bash
# Run with a specific camera source
python src/main.py --camera-source 0  # Use camera index 0

# Run with a test video
python src/main.py --video-source data/test_videos/downtown_drive.mp4

# Run with a specific map region
python src/main.py --map-region 37.7749,-122.4194

# Run with higher detection confidence
python src/main.py --detection-threshold 0.7
```

### Using Custom Configuration Files

You can specify custom configuration files:

```bash
python src/main.py --config-dir path/to/custom/config
```

## Advanced Usage

### Map Change Detection

To specifically run the map change detection module:

```bash
python src/mapping/change_detector.py --video-source data/test_videos/construction_zone.mp4
```

This will analyze the video for changes compared to the map data and output detected discrepancies.

### Route Planning

To test the route planning module independently:

```bash
python src/navigation/route_planner.py --start 37.7749,-122.4194 --end 37.7833,-122.4167
```

This will calculate and visualize a route between the specified start and end points.

## Using Individual Components

### Object Detection

To run only the object detection module on an image or video:

```bash
python src/perception/object_detector.py --source data/test_videos/traffic.mp4 --output output/detections
```

### Lane Detection

To run only the lane detection module:

```bash
python src/perception/lane_detector.py --source data/test_videos/highway.mp4 --output output/lanes
```

### Map Matching

To test the visual-map matching functionality:

```bash
python src/mapping/map_matcher.py --image data/test_images/intersection.jpg --position 37.7749,-122.4194
```

## API Usage

The system can also be used as a library in your own Python code:

```python
from src.perception.object_detector import ObjectDetector
from src.perception.lane_detector import LaneDetector
from src.mapping.map_loader import MapLoader
from src.navigation.route_planner import RoutePlanner

# Initialize components
object_detector = ObjectDetector(model_path="data/models/yolov5s.pt")
lane_detector = LaneDetector(model_path="data/models/lane_detection.pt")
map_loader = MapLoader(config_path="config/map_config.yaml")
route_planner = RoutePlanner(map_loader=map_loader)

# Process a frame
frame = cv2.imread("path/to/image.jpg")
detections = object_detector.detect(frame)
lanes = lane_detector.detect(frame)

# Plan a route
start_pos = (37.7749, -122.4194)
end_pos = (37.7833, -122.4167)
route = route_planner.plan_route(start_pos, end_pos)

# Visualize results
# ...
```

## Data Collection Mode

To collect data for training and testing:

```bash
python src/main.py --data-collection --output-dir data/collected
```

This will save camera frames, detection results, and navigation data to the specified directory.

## Batch Processing

To process multiple video files in batch mode:

```bash
python src/main.py --batch-mode --input-dir data/test_videos --output-dir output/results
```

## Performance Optimization

For improved performance on resource-constrained devices:

```bash
python src/main.py --optimization-level high
```

This will use a more efficient model variant and reduce processing resolution.

## Debugging and Visualization

To run the system with detailed visualization for debugging:

```bash
python src/main.py --debug --visualization full
```

This will display additional windows showing intermediate processing steps and system state.

## Integration with External Systems

The navigation system can be integrated with external systems via its API. Example code for integration with a robotics framework:

```python
# Example integration code
from src.main import NavigationSystem
from robot_framework import RobotControl

# Initialize navigation system
nav_system = NavigationSystem(config_dir="config/")

# Initialize robot controller
robot = RobotControl()

# Setup callback for navigation decisions
def on_navigation_decision(decision):
    if decision["action"] == "TURN_LEFT":
        robot.turn_left(decision["angle"])
    elif decision["action"] == "TURN_RIGHT":
        robot.turn_right(decision["angle"])
    elif decision["action"] == "STOP":
        robot.stop()
    # ...

# Register callback
nav_system.register_decision_callback(on_navigation_decision)

# Start navigation to destination
nav_system.navigate_to(destination=(37.7833, -122.4167))
```

For more detailed examples and API documentation, refer to the `docs/architecture.md` file.