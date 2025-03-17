import cv2
import torch
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

class ObjectDetector:
    def __init__(self, model_path=None, confidence_threshold=0.5):
        """
        Initialize object detector with YOLO or Faster R-CNN
        
        Args:
            model_path: Path to pretrained model weights
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        
        # Load pretrained model or custom model
        if model_path:
            # Load custom model
            self.model = torch.load(model_path)
        else:
            # Use pretrained model
            self.model = fasterrcnn_resnet50_fpn(pretrained=True)
            
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            
        # Object classes of interest for navigation
        self.navigation_classes = {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 
            4: 'bus', 5: 'truck', 6: 'traffic light', 7: 'stop sign',
            8: 'parking meter', 9: 'fire hydrant'
        }
        
    def detect(self, image, priority_regions=None):
        """
        Detect objects in an image
        
        Args:
            image: RGB image as numpy array
            priority_regions: Optional mask of regions to prioritize
            
        Returns:
            detections: List of dictionaries with detected objects
                        [{'box': [x1,y1,x2,y2], 'class': class_id, 
                          'score': confidence, 'priority': priority_level}]
        """
        # Convert to tensor format
        img_tensor = F.to_tensor(image).unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
            
        # Run inference
        with torch.no_grad():
            predictions = self.model(img_tensor)
            
        # Process detections
        detections = []
        for i, pred in enumerate(predictions):
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            # Filter by confidence
            valid_detections = scores >= self.confidence_threshold
            boxes = boxes[valid_detections]
            scores = scores[valid_detections]
            labels = labels[valid_detections]
            
            # Create detection objects
            for box, score, label in zip(boxes, scores, labels):
                if label in self.navigation_classes:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Assign priority based on object class and location
                    priority = self._assign_priority(label, [x1, y1, x2, y2], priority_regions)
                    
                    detections.append({
                        'box': [x1, y1, x2, y2],
                        'class': int(label),
                        'class_name': self.navigation_classes.get(int(label), 'unknown'),
                        'score': float(score),
                        'priority': priority
                    })
        
        return detections
    
    def _assign_priority(self, class_id, box, priority_regions=None):
        """
        Assign priority to detection based on class and location
        
        Args:
            class_id: Object class ID
            box: [x1, y1, x2, y2] bounding box
            priority_regions: Optional priority mask
            
        Returns:
            priority: Priority level (1-10, where 10 is highest)
        """
        # Base priorities for different object types
        high_priority_classes = [6, 7]  # traffic light, stop sign
        medium_priority_classes = [0, 1, 2, 3, 4, 5]  # vehicles and people
        low_priority_classes = [8, 9]  # less critical for immediate navigation
        
        # Default priority based on class
        if class_id in high_priority_classes:
            base_priority = 8
        elif class_id in medium_priority_classes:
            base_priority = 5
        elif class_id in low_priority_classes:
            base_priority = 3
        else:
            base_priority = 2
            
        # Adjust by region priority if available
        if priority_regions is not None:
            x1, y1, x2, y2 = box
            box_center_x = (x1 + x2) // 2
            box_center_y = (y1 + y2) // 2
            
            if 0 <= box_center_y < priority_regions.shape[0] and 0 <= box_center_x < priority_regions.shape[1]:
                region_priority = priority_regions[box_center_y, box_center_x] / 255 * 2  # Scale to 0-2
                base_priority += region_priority
                
        return min(10, max(1, base_priority))  # Clamp to range 1-10