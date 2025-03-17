# src/perception/camera.py
import cv2
import numpy as np
import yaml
import time

class Camera:
    def _init_(self, config_path=None):
        """
        Initialize camera interface for capturing or loading video
        
        Args:
            config_path: Path to camera configuration YAML file
        """
        # Default configuration
        self.config = {
            'source': 0,  # Default to first webcam
            'width': 640,
            'height': 480,
            'fps': 30,
            'calibration': {
                'camera_matrix': None,
                'dist_coeffs': None
            }
        }
        
        # Load configuration if provided
        if config_path:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                self.config.update(loaded_config)
        
        self.capture = None
        self.is_file = False
        self.frame_count = 0
        self.last_frame_time = 0
        
    def open(self):
        """Open camera or video file"""
        source = self.config['source']
        
        # Check if source is a file path (string)
        if isinstance(source, str):
            self.is_file = True
            self.capture = cv2.VideoCapture(source)
            if not self.capture.isOpened():
                raise ValueError(f"Failed to open video file: {source}")
        else:
            # Source is a camera index
            self.is_file = False
            self.capture = cv2.VideoCapture(source)
            if not self.capture.isOpened():
                raise ValueError(f"Failed to open camera at index: {source}")
            
            # Set camera properties
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])
            self.capture.set(cv2.CAP_PROP_FPS, self.config['fps'])
        
        return self.capture.isOpened()
    
    def read(self):
        """
        Read a frame from the camera or video file
        
        Returns:
            frame: The captured frame (None if no frame available)
            timestamp: Time of capture
        """
        if self.capture is None or not self.capture.isOpened():
            return None, 0
        
        # Read frame
        ret, frame = self.capture.read()
        current_time = time.time()
        
        if not ret:
            return None, 0
        
        # Apply calibration if available
        if (self.config['calibration']['camera_matrix'] is not None and 
            self.config['calibration']['dist_coeffs'] is not None):
            camera_matrix = np.array(self.config['calibration']['camera_matrix'])
            dist_coeffs = np.array(self.config['calibration']['dist_coeffs'])
            frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
        
        self.frame_count += 1
        self.last_frame_time = current_time
        
        return frame, current_time
    
    def release(self):
        """Release camera resources"""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
    
    def _del_(self):
        """Destructor to ensure camera is released"""
        self.release()
    
    def get_fps(self):
        """Get actual FPS based on timestamps"""
        if self.frame_count < 2:
            return self.config['fps']
        
        elapsed = self.last_frame_time - self.start_time
        if elapsed > 0:
            return self.frame_count / elapsed
        return 0
    
    def is_opened(self):
        """Check if camera/video is open"""
        return self.capture is not None and self.capture.isOpened()

    def calibrate(self, chessboard_size=(9, 6), square_size=0.025, num_frames=20):
        """
        Calibrate camera using chessboard pattern
        
        Args:
            chessboard_size: Number of inner corners in chessboard (width, height)
            square_size: Size of each square in meters
            num_frames: Number of frames to use for calibration
        
        Returns:
            success: Whether calibration was successful
        """
        if self.capture is None:
            self.open()
            
        if not self.capture.isOpened():
            return False
        
        # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        objp *= square_size  # Convert to meters
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        found_count = 0
        
        print("Calibrating camera. Please show the chessboard pattern...")
        
        while found_count < num_frames:
            ret, frame = self.capture.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                found_count += 1
                
                # Draw corners
                cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
                cv2.putText(frame, f"Captured: {found_count}/{num_frames}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                print(f"Captured {found_count}/{num_frames} calibration frames")
            
            cv2.imshow('Calibration', frame)
            key = cv2.waitKey(500)  # Wait for 500ms
            if key == 27:  # ESC key
                break
        
        cv2.destroyWindow('Calibration')
        
        if found_count < 5:
            print("Not enough calibration frames captured")
            return False
        
        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        if ret:
            # Save calibration parameters
            self.config['calibration']['camera_matrix'] = camera_matrix.tolist()
            self.config['calibration']['dist_coeffs'] = dist_coeffs.tolist()
            
            print("Camera calibration successful")
            return True
        else:
            print("Camera calibration failed")
            return False