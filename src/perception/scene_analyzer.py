# src/perception/scene_analyzer.py
import cv2
import numpy as np
from collections import Counter, deque

class SceneAnalyzer:
    def _init_(self, config=None):
        """
        Initialize scene analyzer to classify current driving context
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Scene classification thresholds
        self.intersection_traffic_light_threshold = self.config.get('intersection_traffic_light_threshold', 0.8)
        self.intersection_sign_threshold = self.config.get('intersection_sign_threshold', 0.7)
        self.highway_lane_threshold = self.config.get('highway_lane_threshold', 0.8)
        
        # Historical context
        self.scene_history = deque(maxlen=self.config.get('history_length', 10))
        self.speed_history = deque(maxlen=self.config.get('history_length', 10))
        
        # Priority mask dimensions
        self.mask_width = self.config.get('mask_width', 640)
        self.mask_height = self.config.get('mask_height', 480)
        
        # Scene transition smoothing
        self.last_scene = None
        self.scene_stability_count = 0
        self.min_stability_count = self.config.get('min_stability_count', 3)
        
    def analyze(self, detections, lanes, vehicle_speed=None):
        """
        Analyze current scene to determine driving context
        
        Args:
            detections: List of detected objects from ObjectDetector
            lanes: Lane detection results from LaneDetector
            vehicle_speed: Optional vehicle speed in km/h
            
        Returns:
            scene_context: Dictionary with scene classification
                          {'type': 'intersection|highway|urban|rural',
                           'confidence': 0.85,
                           'priority_regions': priority_mask}
        """
        # Count relevant objects
        object_counts = Counter()
        for detection in detections:
            object_counts[detection['class_name']] += 1
        
        # Calculate lane confidence
        lane_confidence = (lanes['confidence']['left'] + lanes['confidence']['right']) / 2
        
        # Store vehicle speed in history
        if vehicle_speed is not None:
            self.speed_history.append(vehicle_speed)
        
        # Calculate average speed
        avg_speed = sum(self.speed_history) / len(self.speed_history) if self.speed_history else 0
        
        # Detect scene type
        scene_type, confidence = self._classify_scene(object_counts, lane_confidence, avg_speed)
        
        # Apply temporal smoothing
        scene_type, confidence = self._apply_temporal_smoothing(scene_type, confidence)
        
        # Store in history
        self.scene_history.append(scene_type)
        
        # Generate priority regions mask based on scene type
        priority_mask = self._generate_priority_mask(scene_type, object_counts, lanes)
        
        return {
            'type': scene_type,
            'confidence': confidence,
            'priority_regions': priority_mask
        }
    
    def _classify_scene(self, object_counts, lane_confidence, avg_speed):
        """
        Classify the current scene based on detected objects and context
        
        Args:
            object_counts: Counter of detected objects
            lane_confidence: Confidence in lane detection
            avg_speed: Average vehicle speed
            
        Returns:
            scene_type: String identifying scene type
            confidence: Confidence in classification
        """
        # Check for intersection
        traffic_light_count = object_counts.get('traffic light', 0)
        stop_sign_count = object_counts.get('stop sign', 0)
        
        # Strong indicators of intersection
        if traffic_light_count > 0 or stop_sign_count > 0:
            intersection_confidence = min(1.0, 0.6 + 0.2 * traffic_light_count + 0.2 * stop_sign_count)
            return 'intersection', intersection_confidence
        
        # Check for highway
        if lane_confidence > self.highway_lane_threshold and avg_speed > 80:
            # High speed with clear lanes suggests highway
            highway_confidence = min(1.0, 0.5 + 0.25 * (lane_confidence - 0.5) + 0.25 * min(1.0, avg_speed / 100))
            return 'highway', highway_confidence
        
        # Check for urban environment
        building_indicators = sum([
            object_counts.get('car', 0), 
            object_counts.get('truck', 0),
            object_counts.get('bus', 0),
            object_counts.get('person', 0)
        ])
        
        if building_indicators > 3:
            urban_confidence = min(1.0, 0.5 + 0.1 * building_indicators)
            return 'urban', urban_confidence
        
        # Default to rural if no strong indicators
        return 'rural', 0.6
    
    def _apply_temporal_smoothing(self, scene_type, confidence):
        """
        Apply temporal smoothing to scene classification
        
        Args:
            scene_type: Current scene classification
            confidence: Confidence in classification
            
        Returns:
            smoothed_scene_type: Smoothed scene type
            smoothed_confidence: Smoothed confidence
        """
        if not self.scene_history:
            return scene_type, confidence
        
        # Check if scene matches previous scene
        if scene_type == self.last_scene:
            self.scene_stability_count += 1
        else:
            self.scene_stability_count = 0
        
        # If scene is stable, return it
        if self.scene_stability_count >= self.min_stability_count:
            self.last_scene = scene_type
            return scene_type, min(1.0, confidence + 0.1)
        
        # If scene is unstable but confidence is high, return new scene
        if confidence > 0.8:
            self.last_scene = scene_type
            return scene_type, confidence
        
        # Otherwise, return most common scene from history
        scene_counter = Counter(self.scene_history)
        most_common_scene = scene_counter.most_common(1)[0][0]
        
        # If current scene matches most common, boost confidence
        if scene_type == most_common_scene:
            adjusted_confidence = min(1.0, confidence + 0.05)
            self.last_scene = scene_type
            return scene_type, adjusted_confidence
        
        # Otherwise, return most common scene with reduced confidence
        self.last_scene = most_common_scene
        return most_common_scene, 0.7
    
    def _generate_priority_mask(self, scene_type, object_counts, lanes):
        """
        Generate priority regions mask based on scene type
        
        Args:
            scene_type: Classified scene type
            object_counts: Counter of detected objects
            lanes: Lane detection results
            
        Returns:
            priority_mask: Numpy array with priority regions (0-255)
        """
        # Create empty mask
        mask = np.zeros((self.mask_height, self.mask_width), dtype=np.uint8)
        
        # Different priority patterns based on scene type
        if scene_type == 'intersection':
            # At intersections, focus on front and sides
            cv2.rectangle(mask, (0, 0), (self.mask_width, int(self.mask_height * 0.4)), 100, -1)  # Top area
            cv2.rectangle(mask, (0, int(self.mask_height * 0.4)), (self.mask_width, int(self.mask_height * 0.7)), 200, -1)  # Middle area
            cv2.rectangle(mask, (0, int(self.mask_height * 0.7)), (self.mask_width, self.mask_height), 255, -1)  # Bottom area
            
            # Add side focus
            cv2.rectangle(mask, (0, 0), (int(self.mask_width * 0.3), self.mask_height), 200, -1)  # Left side
            cv2.rectangle(mask, (int(self.mask_width * 0.7), 0), (self.mask_width, self.mask_height), 200, -1)  # Right side
            
        elif scene_type == 'highway':
            # On highways, focus on forward lane and distant objects
            cv2.rectangle(mask, (0, 0), (self.mask_width, int(self.mask_height * 0.3)), 200, -1)  # Top area (distant)
            
            # Central driving lane
            center_width = int(self.mask_width * 0.4)
            center_x = int(self.mask_width * 0.5 - center_width * 0.5)
            cv2.rectangle(mask, (center_x, 0), (center_x + center_width, self.mask_height), 255, -1)
            
            # Lower area for immediate actions
            cv2.rectangle(mask, (0, int(self.mask_height * 0.7)), (self.mask_width, self.mask_height), 220, -1)
            
        elif scene_type == 'urban':
            # In urban areas, focus on road and sidewalks
            cv2.rectangle(mask, (0, 0), (self.mask_width, int(self.mask_height * 0.4)), 150, -1)  # Top area
            cv2.rectangle(mask, (0, int(self.mask_height * 0.4)), (self.mask_width, int(self.mask_height * 0.8)), 200, -1)  # Middle area
            cv2.rectangle(mask, (0, int(self.mask_height * 0.8)), (self.mask_width, self.mask_height), 255, -1)  # Bottom area
            
            # Sidewalks
            cv2.rectangle(mask, (0, 0), (int(self.mask_width * 0.2), self.mask_height), 200, -1)  # Left side
            cv2.rectangle(mask, (int(self.mask_width * 0.8), 0), (self.mask_width, self.mask_height), 200, -1)  # Right side
            
        else:  # Rural
            # In rural areas, focus on road and surroundings
            cv2.rectangle(mask, (0, 0), (self.mask_width, int(self.mask_height * 0.5)), 150, -1)  # Top area
            cv2.rectangle(mask, (0, int(self.mask_height * 0.5)), (self.mask_width, self.mask_height), 200, -1)  # Bottom area
            
            # Central area for road
            center_width = int(self.mask_width * 0.5)
            center_x = int(self.mask_width * 0.5 - center_width * 0.5)
            cv2.rectangle(mask, (center_x, 0), (center_x + center_width, self.mask_height), 220, -1)
        
        return mask