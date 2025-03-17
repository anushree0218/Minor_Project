import cv2
import numpy as np
from typing import Dict, List, Tuple
import yaml

class ObjectDetector:
    """Object detector with priority-based processing"""
    
    def __init__(self, model_path: str, priority_config_path: str):
        """Initialize detector with model and priority config"""
        # Load model
        self.model = self._load_model(model_path)
        
        # Load priority configuration
        with open(priority_config_path, 'r') as f:
            self.priority_config = yaml.safe_load(f)
        
        # Set up class labels
        self.classes = self.priority_config.get('classes', [])
        self.class_priorities = self.priority_config.get('priorities', {})
        
        # Confidence thresholds
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
    
    def _load_model(self, model_path: str):
        """Load detection model"""
        # For this example, using OpenCV's DNN module with a pre-trained model
        # Real implementation would use a more sophisticated model (YOLO, SSD, etc.)
        net = cv2.dnn.readNetFromDarknet(
            f"{model_path}/yolov4-tiny.cfg", 
            f"{model_path}/yolov4-tiny.weights"
        )
        
        # Set backend and target
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    
    def detect(self, frame: np.ndarray, context: Dict = None) -> List[Dict]:
        """
        Detect objects in frame with priority-based processing
        
        Args:
            frame: Input image
            context: Current driving context (intersection, highway, etc.)
            
        Returns:
            List of detections with priority scores
        """
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (416, 416), swapRB=True, crop=False
        )
        self.model.setInput(blob)
        
        # Get output layers
        layer_names = self.model.getLayerNames()
        output_layers = [layer_names[i - 1] for i in self.model.getUnconnectedOutLayers()]
        
        # Forward pass
        outputs = self.model.forward(output_layers)
        
        # Process detections
        detections = self._process_detections(outputs, frame, context)
        return detections
    
    def _process_detections(self, outputs, frame, context):
        """Process raw detections into structured results with priorities"""
        height, width = frame.shape[:2]
        boxes = []
        confidences = []
        class_ids = []
        
        # Process each detection
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Convert to pixel coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Calculate top-left corner
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence_threshold, self.nms_threshold
        )
        
        # Calculate priorities based on context
        detections = []
        for i in indices:
            if isinstance(i, list):  # Handle different OpenCV versions
                i = i[0]
            
            box = boxes[i]
            class_id = class_ids[i]
            confidence = confidences[i]
            
            # Get class name
            class_name = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
            
            # Calculate priority based on context
            priority = self._calculate_priority(class_name, context)
            
            detections.append({
                'box': box,
                'class_id': class_id,
                'class_name': class_name,
                'confidence': confidence,
                'priority': priority
            })
        
        # Sort by priority
        detections.sort(key=lambda x: x['priority'], reverse=True)
        return detections
    
    def _calculate_priority(self, class_name: str, context: Dict = None) -> float:
        """Calculate object priority based on context"""
        # Base priority from config
        base_priority = self.class_priorities.get(class_name, 0.5)
        
        if context is None:
            return base_priority
        
        # Context-based priority adjustment
        context_type = context.get('type', 'default')
        
        # Priority multipliers for different contexts
        context_multipliers = {
            'intersection': {
                'traffic_light': 2.0,
                'stop_sign': 2.0,
                'crosswalk': 1.8,
                'car': 1.5,
                'person': 1.8
            },
            'highway': {
                'lane_marking': 1.5,
                'car': 1.3,
                'truck': 1.5,
                'road_sign': 1.2
            },
            'residential': {
                'person': 2.0,
                'bicycle': 1.8,
                'car': 1.2,
                'stop_sign': 1.5
            },
            'default': {
                # Default multipliers
            }
        }
        
        # Get multiplier for this class in this context
        multiplier = context_multipliers.get(context_type, {}).get(class_name, 1.0)
        
        # Apply distance-based priority adjustment if available
        if 'distances' in context and class_name in context['distances']:
            distance = context['distances'][class_name]
            # Closer objects get higher priority, with a non-linear relationship
            distance_factor = 1.0 / (1.0 + 0.1 * distance)
            multiplier *= distance_factor
        
        return base_priority * multiplier