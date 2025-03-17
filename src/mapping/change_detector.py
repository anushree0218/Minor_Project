import numpy as np
import cv2
from collections import deque
import time

class ChangeDetector:
    def __init__(self, map_loader, config=None):
        """
        Initialize change detector for detecting map changes
        
        Args:
            map_loader: MapLoader instance
            config: Configuration dictionary
        """
        self.map_loader = map_loader
        self.config = config or {}
        
        # Configuration parameters
        self.detection_threshold = self.config.get('detection_threshold', 0.8)
        self.min_detections = self.config.get('min_detections', 3)
        self.temporal_window = self.config.get('temporal_window', 10)
        
        # Recent detections
        self.recent_changes = deque(maxlen=self.temporal_window)
        
        # Store detected changes
        self.detected_changes = []
    
    def detect_changes(self, detections, current_position, camera_params=None):
        """
        Detect changes in the environment compared to map data
        
        Args:
            detections: List of detected objects
            current_position: Current GPS position (lat, lon)
            camera_params: Optional camera parameters for projections
            
        Returns:
            changes: List of detected changes
            confidence: Confidence in the changes
        """
        if self.map_loader.graph is None or not current_position:
            return [], 0.0
        
        # Get nearby map features
        nearby_features = self._get_nearby_map_features(current_position)
        
        # Compare detected objects with map features
        changes = []
        
        for detection in detections:
            change_type = None
            confidence = 0.0
            
            # Convert detection to world coordinates
            world_pos = self._detection_to_world(detection, current_position, camera_params)
            
            if world_pos:
                # Check if detection matches a map feature
                matched, match_type, match_distance = self._match_to_map_feature(world_pos, detection, nearby_features)
                
                if not matched and detection['score'] > self.detection_threshold:
                    # This could be a new feature not on the map
                    change_type = 'new_feature'
                    confidence = detection['score']
                
                elif matched and match_type != detection['class_name']:
                    # Feature exists but type is different (e.g., road closure, construction)
                    change_type = 'type_changed'
                    confidence = detection['score'] * (1.0 - match_distance)
            
            # If change detected, add to list
            if change_type:
                change = {
                    'type': change_type,
                    'position': world_pos,
                    'object_type': detection['class_name'],
                    'confidence': confidence,
                    'timestamp': time.time()
                }
                changes.append(change)
                
                # Add to recent changes for temporal consistency
                self.recent_changes.append(change)
        
        # Apply temporal filtering
        consistent_changes = self._apply_temporal_filtering()
        
        # Calculate overall confidence
        avg_confidence = np.mean([c['confidence'] for c in consistent_changes]) if consistent_changes else 0.0
        
        # Store new consistent changes
        for change in consistent_changes:
            if change not in self.detected_changes:
                self.detected_changes.append(change)
        
        return consistent_changes, avg_confidence
    
    def _get_nearby_map_features(self, position, radius=100):
        """
        Get map features near a position
        
        Args:
            position: (lat, lon) position
            radius: Search radius in meters
            
        Returns:
            features: List of nearby map features
        """
        if not position or self.map_loader.graph is None:
            return []
            
        lat, lon = position
        
        # Get nearby nodes from the map graph
        nearby_nodes = []
        for node_id, node_data in self.map_loader.graph.nodes(data=True):
            node_lat = node_data.get('lat')
            node_lon = node_data.get('lon')
            
            if node_lat is not None and node_lon is not None:
                # Calculate distance
                distance = self._haversine_distance(lat, lon, node_lat, node_lon)
                if distance <= radius:
                    nearby_nodes.append((node_id, node_data, distance))
        
        # Get edges connected to nearby nodes
        nearby_features = []
        for node_id, node_data, _ in nearby_nodes:
            # Add node features
            if 'feature_type' in node_data:
                nearby_features.append({
                    'id': node_id,
                    'type': node_data['feature_type'],
                    'position': (node_data['lat'], node_data['lon']),
                    'properties': node_data
                })
            
            # Add connected edges
            for _, target, edge_data in self.map_loader.graph.edges(node_id, data=True):
                if 'feature_type' in edge_data:
                    nearby_features.append({
                        'id': f"{node_id}-{target}",
                        'type': edge_data['feature_type'],
                        'source': node_id,
                        'target': target,
                        'properties': edge_data
                    })
        
        return nearby_features
    
    def _detection_to_world(self, detection, current_position, camera_params):
        """
        Convert a detection in image coordinates to world coordinates
        
        Args:
            detection: Detection object with bounding box and class
            current_position: Current GPS position (lat, lon)
            camera_params: Camera parameters for projection
            
        Returns:
            world_pos: World position (lat, lon) or None if conversion fails
        """
        if not camera_params or 'bbox' not in detection:
            return None
            
        try:
            # Extract bounding box
            x1, y1, x2, y2 = detection['bbox']
            
            # Calculate bottom center of bounding box (assuming it's on the ground)
            image_x = (x1 + x2) / 2
            image_y = y2
            
            # Extract camera parameters
            focal_length = camera_params.get('focal_length', 1000)
            camera_height = camera_params.get('height', 1.5)  # meters
            pitch = camera_params.get('pitch', 0)  # radians
            yaw = camera_params.get('yaw', 0)  # radians
            
            # Image center
            image_width = camera_params.get('width', 1920)
            image_height = camera_params.get('height', 1080)
            cx = image_width / 2
            cy = image_height / 2
            
            # Convert to normalized image coordinates
            normalized_x = (image_x - cx) / focal_length
            normalized_y = (image_y - cy) / focal_length
            
            # Apply pitch correction
            pitch_corrected_y = np.tan(np.arctan(normalized_y) + pitch)
            
            # Calculate distance to object
            # Simple projection model: distance = camera_height / tan(pitch_angle)
            distance = camera_height / pitch_corrected_y if pitch_corrected_y > 0 else 0
            
            # If distance is unreasonable, return None
            if distance <= 0 or distance > 200:  # Limit to 200m
                return None
                
            # Apply yaw to get direction
            bearing = np.arctan(normalized_x) + yaw
            
            # Convert to lat, lon
            lat, lon = self._project_position(current_position, distance, bearing)
            
            return (lat, lon)
            
        except Exception as e:
            print(f"Error converting detection to world coordinates: {e}")
            return None
    
    def _match_to_map_feature(self, world_pos, detection, nearby_features):
        """
        Match a detected object to nearby map features
        
        Args:
            world_pos: World position (lat, lon)
            detection: Detection object
            nearby_features: List of nearby map features
            
        Returns:
            matched: Boolean indicating if a match was found
            match_type: Type of the matched feature (if matched)
            match_distance: Distance to the matched feature (if matched)
        """
        if not world_pos or not nearby_features:
            return False, None, float('inf')
            
        # Object type mapping
        # Map detection class names to map feature types
        type_mapping = {
            'traffic_light': ['traffic_signal', 'traffic_light'],
            'stop_sign': ['stop_sign', 'stop'],
            'road_sign': ['sign', 'information'],
            'construction': ['construction', 'roadwork'],
            'pothole': ['hazard', 'damage'],
            # Add more mappings as needed
        }
        
        # Get compatible feature types for this detection
        detection_type = detection['class_name']
        compatible_types = type_mapping.get(detection_type, [detection_type])
        
        # Find the closest matching feature
        best_match = None
        best_distance = float('inf')
        
        for feature in nearby_features:
            # Check if feature type is compatible
            feature_type = feature.get('type')
            
            # For point features, calculate distance
            if 'position' in feature:
                feature_pos = feature['position']
                distance = self._haversine_distance(
                    world_pos[0], world_pos[1], 
                    feature_pos[0], feature_pos[1]
                )
                
                # Update best match if this is closer
                if distance < best_distance:
                    best_match = feature
                    best_distance = distance
            
            # For line features (roads, etc.), calculate distance to line
            elif 'source' in feature and 'target' in feature:
                source_node = self.map_loader.graph.nodes[feature['source']]
                target_node = self.map_loader.graph.nodes[feature['target']]
                
                if 'lat' in source_node and 'lon' in source_node and 'lat' in target_node and 'lon' in target_node:
                    source_pos = (source_node['lat'], source_node['lon'])
                    target_pos = (target_node['lat'], target_node['lon'])
                    
                    # Calculate distance to line segment
                    distance = self._point_to_line_distance(world_pos, source_pos, target_pos)
                    
                    # Update best match if this is closer
                    if distance < best_distance:
                        best_match = feature
                        best_distance = distance
        
        # Determine if we have a match
        # Use a distance threshold based on detection type
        distance_threshold = 10  # meters
        
        # Adjust threshold based on object type
        if detection_type in ['building', 'road']:
            distance_threshold = 20
        elif detection_type in ['traffic_light', 'stop_sign']:
            distance_threshold = 5
            
        if best_match and best_distance <= distance_threshold:
            return True, best_match['type'], best_distance
        else:
            return False, None, float('inf')
    
    def _apply_temporal_filtering(self):
        """
        Apply temporal filtering to recent changes
        
        Returns:
            consistent_changes: List of temporally consistent changes
        """
        if not self.recent_changes:
            return []
            
        # Group changes by position and type
        change_groups = {}
        
        for change in self.recent_changes:
            pos = change['position']
            obj_type = change['object_type']
            change_type = change['type']
            
            # Create a key for this change
            # Group by position (rounded to reduce floating point issues)
            key = (
                round(pos[0], 6),
                round(pos[1], 6),
                obj_type,
                change_type
            )
            
            if key not in change_groups:
                change_groups[key] = []
                
            change_groups[key].append(change)
        
        # Filter changes that appear consistently
        consistent_changes = []
        
        for group in change_groups.values():
            if len(group) >= self.min_detections:
                # Sort by timestamp and take the most recent
                most_recent = sorted(group, key=lambda x: x['timestamp'], reverse=True)[0]
                
                # Calculate average confidence
                avg_confidence = np.mean([c['confidence'] for c in group])
                
                # Update confidence based on number of detections
                confidence_boost = min(1.0, len(group) / self.temporal_window)
                final_confidence = avg_confidence * (1.0 + 0.2 * confidence_boost)
                
                # Create a copy with updated confidence
                change = most_recent.copy()
                change['confidence'] = min(1.0, final_confidence)
                consistent_changes.append(change)
        
        return consistent_changes
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        
        Args:
            lat1, lon1: Coordinates of first point
            lat2, lon2: Coordinates of second point
            
        Returns:
            distance: Distance between points in meters
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371000  # Radius of earth in meters
        
        return c * r
    
    def _project_position(self, start_pos, distance, bearing):
        """
        Project a position given a starting point, distance, and bearing
        
        Args:
            start_pos: Starting position (lat, lon)
            distance: Distance to project in meters
            bearing: Bearing in radians
            
        Returns:
            new_pos: New position (lat, lon)
        """
        lat1, lon1 = start_pos
        
        # Convert to radians
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        
        # Earth radius in meters
        R = 6371000
        
        # Calculate new latitude
        lat2 = np.arcsin(np.sin(lat1) * np.cos(distance/R) +
                          np.cos(lat1) * np.sin(distance/R) * np.cos(bearing))
        
        # Calculate new longitude
        lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(distance/R) * np.cos(lat1),
                                  np.cos(distance/R) - np.sin(lat1) * np.sin(lat2))
        
        # Convert back to degrees
        lat2 = np.degrees(lat2)
        lon2 = np.degrees(lon2)
        
        return (lat2, lon2)
    
    def _point_to_line_distance(self, point, line_start, line_end):
        """
        Calculate the shortest distance from a point to a line segment
        
        Args:
            point: The point (lat, lon)
            line_start: Start of line segment (lat, lon)
            line_end: End of line segment (lat, lon)
            
        Returns:
            distance: Shortest distance in meters
        """
        # Convert all to cartesian coordinates for simplicity
        # This is an approximation that works for small distances
        p_lat, p_lon = point
        s_lat, s_lon = line_start
        e_lat, e_lon = line_end