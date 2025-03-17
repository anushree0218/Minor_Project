import cv2
import numpy as np

class LaneDetector:
    def __init__(self, config=None):
        """
        Initialize lane detector
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        
        # Default parameters
        self.canny_low = self.config.get('canny_low', 50)
        self.canny_high = self.config.get('canny_high', 150)
        self.hough_rho = self.config.get('hough_rho', 1)
        self.hough_theta = self.config.get('hough_theta', np.pi/180)
        self.hough_threshold = self.config.get('hough_threshold', 20)
        self.hough_min_line_length = self.config.get('hough_min_line_length', 20)
        self.hough_max_line_gap = self.config.get('hough_max_line_gap', 300)
        
        # Confidence calculation parameters
        self.min_lane_points = self.config.get('min_lane_points', 5)
        self.max_lane_points = self.config.get('max_lane_points', 50)
        
    def detect(self, image):
        """
        Detect lane markings in image
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            lanes: Dictionary with lane information
                  {'left': [(x1,y1,x2,y2),...], 'right': [(x1,y1,x2,y2),...],
                   'confidence': {'left': 0.8, 'right': 0.9}}
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Define region of interest (lower part of image)
        height, width = gray.shape
        roi_vertices = np.array([
            [0, height],
            [0, height * 0.6],
            [width, height * 0.6],
            [width, height]
        ], dtype=np.int32)
        
        # Create ROI mask
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [roi_vertices], 255)
        masked_gray = cv2.bitwise_and(gray, mask)
        
        # Apply edge detection
        edges = cv2.Canny(masked_gray, self.canny_low, self.canny_high)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges, self.hough_rho, self.hough_theta, 
                               self.hough_threshold, None, 
                               self.hough_min_line_length, self.hough_max_line_gap)
        
        # Process detected lines
        left_lines = []
        right_lines = []
        left_confidence = 0.0
        right_confidence = 0.0
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Skip horizontal lines
                if abs(y2 - y1) < 5:
                    continue
                
                # Calculate slope
                slope = (y2 - y1) / (x2 - x1 + 1e-6)  # Avoid division by zero
                
                # Filter by slope - negative slope for left lanes, positive for right
                if slope < -0.35:
                    left_lines.append([x1, y1, x2, y2])
                elif slope > 0.35:
                    right_lines.append([x1, y1, x2, y2])
        
        # Calculate confidence based on number of detected line segments
        left_confidence = self._calculate_confidence(len(left_lines))
        right_confidence = self._calculate_confidence(len(right_lines))
        
        # Return detected lanes
        return {
            'left': left_lines,
            'right': right_lines,
            'confidence': {
                'left': left_confidence,
                'right': right_confidence
            }
        }
    
    def _calculate_confidence(self, num_points):
        """
        Calculate confidence level based on number of detected line segments
        
        Args:
            num_points: Number of line segments detected
            
        Returns:
            confidence: Value between 0.0 and 1.0
        """
        if num_points < self.min_lane_points:
            # Too few points, low confidence
            return max(0.0, num_points / self.min_lane_points * 0.5)
        elif num_points > self.max_lane_points:
            # Many points but could be noise, cap confidence
            return 0.95
        else:
            # Scale confidence between 0.5 and 0.95
            normalized = (num_points - self.min_lane_points) / (self.max_lane_points - self.min_lane_points)
            return 0.5 + normalized * 0.45