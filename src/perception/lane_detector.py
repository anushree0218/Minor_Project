import cv2
import numpy as np
from typing import Dict, List, Tuple

class LaneDetector:
    """Lane detection and analysis"""
    
    def __init__(self, config: Dict = None):
        """Initialize lane detector with configuration"""
        self.config = config or {}
        
        # Parameters for lane detection
        self.canny_low = self.config.get('canny_low', 50)
        self.canny_high = self.config.get('canny_high', 150)
        self.roi_vertices = self.config.get('roi_vertices', None)
        self.hough_rho = self.config.get('hough_rho', 1)
        self.hough_theta = self.config.get('hough_theta', np.pi/180)
        self.hough_threshold = self.config.get('hough_threshold', 20)
        self.hough_min_line_length = self.config.get('hough_min_line_length', 20)
        self.hough_max_line_gap = self.config.get('hough_max_line_gap', 300)
    
    def detect_lanes(self, frame: np.ndarray) -> Dict:
        """
        Detect lanes in the frame
        
        Args:
            frame: Input image
            
        Returns:
            Dict with lane information
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blur, self.canny_low, self.canny_high)
        
        # Define region of interest if not provided
        if self.roi_vertices is None:
            height, width = frame.shape[:2]
            self.roi_vertices = np.array([
                [(0, height), (width/2, height/2), (width, height)]
            ], dtype=np.int32)
        
        # Apply region of interest mask
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, self.roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Apply Hough transform to detect lines
        lines = cv2.HoughLinesP(
            masked_edges, 
            self.hough_rho, 
            self.hough_theta, 
            self.hough_threshold, 
            minLineLength=self.hough_min_line_length, 
            maxLineGap=self.hough_max_line_gap
        )
        
        # Process detected lines
        left_lines, right_lines = self._separate_lines(lines)
        
        # Fit lane lines
        left_fit = self._fit_lane(left_lines, frame.shape[:2])
        right_fit = self._fit_lane(right_lines, frame.shape[:2])
        
        # Calculate lane curvature and position
        curvature = self._calculate_curvature(left_fit, right_fit)
        position = self._calculate_position(left_fit, right_fit, frame.shape[1])
        
        # Calculate confidence in lane detection
        confidence = self._calculate_confidence(left_lines, right_lines)
        
        return {
            'left_fit': left_fit,
            'right_fit': right_fit,
            'curvature': curvature,
            'position': position,
            'confidence': confidence
        }
    
    def _separate_lines(self, lines):
        """Separate lines into left and right lanes"""
        left_lines = []
        right_lines = []
        
        if lines is None:
            return left_lines, right_lines
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue  # Skip vertical lines
            
            slope = (y2 - y1) / (x2 - x1)
            
            # Filter based on slope
            if abs(slope) < 0.5:
                continue  # Ignore horizontal lines
            
            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)
        
        return left_lines, right_lines
    
    def _fit_lane(self, lines, img_shape):
        """Fit a polynomial to the lane lines"""
        if not lines:
            return None
        
        x_coords = []
        y_coords = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        if not x_coords or not y_coords:
            return None
        
        # Fit a polynomial
        return np.polyfit(y_coords, x_coords, 2)
    
    def _calculate_curvature(self, left_fit, right_fit):
        """Calculate lane curvature"""
        if left_fit is None or right_fit is None:
            return float('inf')
        
        # Average the curvature of both lines
        left_curvature = abs(left_fit[0])
        right_curvature = abs(right_fit[0])
        
        # Average curvature - higher value = straighter lane
        avg_curvature = (left_curvature + right_curvature) / 2
        return avg_curvature
    
    def _calculate_position(self, left_fit, right_fit, img_width):
        """Calculate vehicle position relative to lane center"""
        if left_fit is None or right_fit is None:
            return 0
        
        # Calculate lane position at the bottom of the image
        y = img_width  # Using bottom of image
        left_x = left_fit[0] * y**2 + left_fit[1] * y + left_fit[2]
        right_x = right_fit[0] * y**2 + right_fit[1] * y + right_fit[2]
        
        # Calculate lane center
        lane_center = (left_x + right_x) / 2
        
        # Calculate vehicle position (assuming camera is mounted at center)
        vehicle_position = img_width / 2
        
        # Positive = right of center, negative = left of center
        position = vehicle_position - lane_center
        return position
    
    def _calculate_confidence(self, left_lines, right_lines):
        """Calculate confidence in lane detection"""
        # Higher confidence if we have more lines
        left_count = len(left_lines) if left_lines else 0
        right_count = len(right_lines) if right_lines else 0
        
        # Basic confidence calculation
        if left_count == 0 and right_count == 0:
            return 0.0
        
        # More confidence if we have both left and right lines
        if left_count > 0 and right_count > 0:
            base_confidence = 0.7
        else:
            base_confidence = 0.4
        
        # Add confidence based on number of lines (diminishing returns)
        line_confidence = min(0.3, (left_count + right_count) / 20)
        
        return base_confidence + line_confidence
    
    def visualize_lanes(self, frame: np.ndarray, lane_info: Dict) -> np.ndarray:
        """
        Visualize detected lanes on the frame
        
        Args:
            frame: Input image
            lane_info: Lane information from detect_lanes
            
        Returns:
            Frame with lane visualization
        """
        visualization = frame.copy()
        height, width = frame.shape[:2]
        
        # Create an overlay for the lanes
        overlay = np.zeros_like(frame)
        
        # Draw lanes if detected
        if lane_info['left_fit'] is not None and lane_info['right_fit'] is not None:
            # Generate points for the lane polygon
            ploty = np.linspace(0, height-1, num=height)
            left_fitx = lane_info['left_fit'][0]*ploty**2 + lane_info['left_fit'][1]*ploty + lane_info['left_fit'][2]
            right_fitx = lane_info['right_fit'][0]*ploty**2 + lane_info['right_fit'][1]*ploty + lane_info['right_fit'][2]
            
            # Create points for polygon
            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))
            
            # Draw lane area
            cv2.fillPoly(overlay, np.int32([pts]), (0, 255, 0))
            
            # Draw lane lines
            for i in range(height-1):
                if 0 <= left_fitx[i] < width and 0 <= left_fitx[i+1] < width:
                    cv2.line(overlay, (int(left_fitx[i]), int(ploty[i])), 
                             (int(left_fitx[i+1]), int(ploty[i+1])), (255, 0, 0), 5)
                
                if 0 <= right_fitx[i] < width and 0 <= right_fitx[i+1] < width:
                    cv2.line(overlay, (int(right_fitx[i]), int(ploty[i])), 
                             (int(right_fitx[i+1]), int(ploty[i+1])), (0, 0, 255), 5)
        
        # Apply the overlay with transparency
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0, visualization)
        
        # Add text information
        info_text = [
            f"Lane Curvature: {lane_info['curvature']:.2f}",
            f"Vehicle Position: {lane_info['position']:.2f}px",
            f"Detection Confidence: {lane_info['confidence']:.2f}"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(
                visualization, 
                text, 
                (10, y_offset + i*30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2
            )
        
        return visualization