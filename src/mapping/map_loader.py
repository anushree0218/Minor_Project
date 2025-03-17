# src/mapping/map_loader.py
import osmnx as ox
import numpy as np
import yaml
from geopy.distance import geodesic

class MapLoader:
    def __init__(self, config_path=None):
        """
        Initialize map loader for navigation
        
        Args:
            config_path: Path to map configuration file
        """
        # Default configuration
        self.config = {
            'cache_folder': './data/maps',
            'default_network_type': 'drive',
            'default_dist': 1000,  # 1km radius
            'simplify': True,
            'retain_all': False,
            'map_source': 'osm'  # OpenStreetMap
        }
        
        # Load configuration if provided
        if config_path:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                self.config.update(loaded_config)
        
        # Set cache folder
        ox.config(use_cache=True, cache_folder=self.config['cache_folder'])
        
        self.graph = None
        self.current_location = None
        self.map_bounds = None
    
    def load_map_by_location(self, lat, lon, dist=None):
        """
        Load map around a specific location
        
        Args:
            lat: Latitude
            lon: Longitude
            dist: Distance in meters to extend from center point
            
        Returns:
            loaded: Boolean indicating success
        """
        if dist is None:
            dist = self.config['default_dist']
        
        try:
            self.graph = ox.graph_from_point(
                (lat, lon),
                dist=dist,
                network_type=self.config['default_network_type'],
                simplify=self.config['simplify'],
                retain_all=self.config['retain_all']
            )
            
            self.current_location = (lat, lon)
            
            # Calculate bounds
            north, south, east, west = ox.utils_graph.graph_to_bbox(self.graph)
            self.map_bounds = {
                'north': north,
                'south': south,
                'east': east,
                'west': west
            }
            
            return True
        
        except Exception as e:
            print(f"Error loading map: {e}")
            return False
    
    def load_map_by_address(self, address, dist=None):
        """
        Load map around a specific address
        
        Args:
            address: Address string
            dist: Distance in meters to extend from center point
            
        Returns:
            loaded: Boolean indicating success
        """
        if dist is None:
            dist = self.config['default_dist']
        
        try:
            # Get coordinates from address
            location = ox.geocode(address)
            return self.load_map_by_location(location[0], location[1], dist)
        
        except Exception as e:
            print(f"Error loading map by address: {e}")
            return False
    
    def is_location_in_bounds(self, lat, lon, buffer=0.01):
        """
        Check if a location is within the loaded map bounds
        
        Args:
            lat: Latitude to check
            lon: Longitude to check
            buffer: Buffer percentage to extend the check
            
        Returns:
            in_bounds: Boolean indicating if location is in bounds
        """
        if self.map_bounds is None:
            return False
        
        # Apply buffer to bounds
        north = self.map_bounds['north'] + buffer
        south = self.map_bounds['south'] - buffer
        east = self.map_bounds['east'] + buffer
        west = self.map_bounds['west'] - buffer
        
        return south <= lat <= north and west <= lon <= east
    
    def get_nearest_node(self, lat, lon):
        """
        Get the nearest node to a location
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            node_id: ID of nearest node
        """
        if self.graph is None:
            return None
        
        try:
            return ox.distance.nearest_nodes(self.graph, lon, lat)
        except Exception as e:
            print(f"Error finding nearest node: {e}")
            return None
    
    def get_nearest_edge(self, lat, lon):
        """
        Get the nearest edge to a location
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            edge: Nearest edge as (u, v, key)
        """
        if self.graph is None:
            return None
        
        try:
            return ox.distance.nearest_edges(self.graph, lon, lat)
        except Exception as e:
            print(f"Error finding nearest edge: {e}")
            return None
    
    def get_route(self, start_lat, start_lon, end_lat, end_lon):
        """
        Calculate route between two points
        
        Args:
            start_lat: Starting latitude
            start_lon: Starting longitude
            end_lat: Ending latitude
            end_lon: Ending longitude
            
        Returns:
            route: List of node IDs forming the route
            distance: Total distance in meters
        """
        if self.graph is None:
            return None, 0
        
        try:
            # Get nearest nodes
            start_node = self.get_nearest_node(start_lat, start_lon)
            end_node = self.get_nearest_node(end_lat, end_lon)
            
            if start_node is None or end_node is None:
                return None, 0
            
            # Calculate shortest path
            route = ox.shortest_path(self.graph, start_node, end_node, weight='length')
            
            # Calculate total distance
            distance = sum(ox.utils_graph.get_route_edge_attributes(self.graph, route, 'length'))
            
            return route, distance
        except Exception as e:
            print(f"Error calculating route: {e}")
            return None, 0