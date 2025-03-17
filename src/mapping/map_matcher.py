# src/mapping/map_matcher.py
import cv2
import numpy as np
from sklearn.neighbors import NearestNeighbors
import osmnx as ox

class MapMatcher:
    def __init__(self, map_loader):
        """
        Initialize map matcher for visual-map matching
        
        Args:
            map_loader: MapLoader instance with loaded map
        """
        self.map_loader = map_loader
        self.last_match_position = None
        self.match_confidence = 0.0
        
    def match_visual_features(self, landmark_features, current_position=None, heading=None):
        """
        Match visual features to map features
        
        Args:
            landmark_features: List of detected landmarks with positions
            current_position: Optional current GPS position (lat, lon)
            heading: Optional current heading in degrees
            
        Returns:
            matched_position: (lat, lon) of best match
            confidence: Match confidence from 0 to 1
        """
        if self.map_loader.graph is None:
            return None, 0.0
        
        if not landmark_features:
            return current_position, 0.3  # Low confidence if no landmarks
        
        # If we have current position, search around it first
        search_radius = 100  # meters
        landmarks_to_match = []
        
        if current_position:
            # Extract map features around current position
            lat, lon = current_position
            nearby_nodes = ox.distance.nearest_nodes(
                self.map_loader.graph, 
                X=[lon], 
                Y=[lat], 
                return_dist=True
            )
            
            # Get nodes within search radius
            nearby_nodes_filtered = []
            for node, dist in zip(nearby_nodes[0], nearby_nodes[1]):
                if dist <= search_radius:
                    nearby_nodes_filtered.append(node)
            
            # Extract node coordinates
            map_points = []
            for node in nearby_nodes_filtered:
                node_data = self.map_loader.graph.nodes[node]
                map_points.append((node_data['y'], node_data['x']))  # lat, lon
            
            landmarks_to_match = map_points
        
        # If no current position or not enough nearby landmarks, use all map nodes
        if not landmarks_to_match:
            # Extract all nodes
            map_points = []
            for node, data in self.map_loader.graph.nodes(data=True):
                map_points.append((data['y'], data['x']))  # lat, lon
            
            landmarks_to_match = map_points
        
        # Transform landmarks to world coordinates based on current position and heading
        if current_position and heading:
            world_landmarks = self._transform_to_world(landmark_features, current_position, heading)
        else:
            # Without position/heading, we can only use relative positions
            world_landmarks = landmark_features
        
        # Match using nearest neighbors
        if len(landmarks_to_match) > 0 and len(world_landmarks) > 0:
            # Create nearest neighbors model
            nn_model = NearestNeighbors(n_neighbors=1).fit(landmarks_to_match)
            
            # Find nearest map points for each landmark
            distances, indices = nn_model.kneighbors(world_landmarks)
            
            # Calculate weighted average position based on confidence
            weighted_lat = 0
            weighted_lon = 0
            total_weight = 0
            
            for i, (distance, idx) in enumerate(zip(distances, indices)):
                weight = 1.0 / (1.0 + distance[0])  # Higher weight for closer matches
                map_point = landmarks_to_match[idx[0]]
                
                weighted_lat += map_point[0] * weight
                weighted_lon += map_point[1] * weight
                total_weight += weight
            
            if total_weight > 0:
                matched_lat = weighted_lat / total_weight
                matched_lon = weighted_lon / total_weight
                
                # Calculate confidence based on average distance
                avg_distance = np.mean(distances)
                confidence = 1.0 / (1.0 + avg_distance)
                
                self.last_match_position = (matched_lat, matched_lon)
                self.match_confidence = confidence
                
                return (matched_lat, matched_lon), confidence
        
        # If no match found, return current position with low confidence
        if current_position:
            return current_position, 0.2
        
        return None, 0.0
    
    def _transform_to_world(self, landmarks, current_position, heading):
        """
        Transform visual landmarks to world coordinates
        
        Args:
            landmarks: List of landmarks with relative positions (x, y) in meters
            current_position: Current GPS position (lat, lon)
            heading: Current heading in degrees
            
        Returns:
            world_landmarks: List of landmarks with global positions (lat, lon)
        """
        world_landmarks = []
        
        # Convert heading to radians
        heading_rad = np.radians(heading)
        
        # Current position
        current_lat, current_lon = current_position
        
        for landmark in landmarks:
            # Get relative position in meters
            rel_x, rel_y = landmark
            
            # Rotate based on heading
            rotated_x = rel_x * np.cos(heading_rad) - rel_y * np.sin(heading_rad)
            rotated_y = rel_x * np.sin(heading_rad) + rel_y * np.cos(heading_rad)
            
            # Convert to lat/lon (approximate)
            # 111,111 meters per degree of latitude
            # 111,111 * cos(latitude) meters per degree of longitude
            lat_change = rotated_y / 111111
            lon_change = rotated_x / (111111 * np.cos(np.radians(current_lat)))
            
            landmark_lat = current_lat + lat_change
            landmark_lon = current_lon + lon_change
            
            world_landmarks.append((landmark_lat, landmark_lon))
        
        return world_landmarks