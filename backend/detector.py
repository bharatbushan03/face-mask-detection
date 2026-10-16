import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression


class FaceMaskDetector:
    def __init__(self, model_path=None, prototxt_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'models', 'face_mask_detection.caffemodel')
        if prototxt_path is None:
            prototxt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                        'models', 'face_mask_detection.prototxt')
        
        # Load the model
        self.net = cv2.dnn.readNet(model_path, prototxt_path)
        
        # Anchor configuration for 260x260 input
        self.feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
        self.anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        self.anchor_ratios = [[1, 0.62, 0.42]] * 5
        
        # Generate anchors
        self.anchors = generate_anchors(self.feature_map_sizes, self.anchor_sizes, self.anchor_ratios)
        self.anchors_exp = np.expand_dims(self.anchors, axis=0)
        
        # Class mapping
        self.id2class = {0: 'Mask', 1: 'NoMask'}
        self.colors = ((0, 255, 0), (255, 0, 0))  # Green for mask, Red for no mask
        
    def get_output_names(self):
        layers_names = self.net.getLayerNames()
        unconnected_out_layers = self.net.getUnconnectedOutLayers()
        
        # Handle different OpenCV versions
        if len(unconnected_out_layers) > 0 and isinstance(unconnected_out_layers[0], (list, tuple, np.ndarray)):
            return [layers_names[i[0] - 1] for i in unconnected_out_layers]
        else:
            return [layers_names[i - 1] for i in unconnected_out_layers]
    
    def detect(self, image, conf_thresh=0.5, iou_thresh=0.4, target_shape=(260, 260)):
        """
        Detect faces and mask status in the image.
        
        Args:
            image: Input image (numpy array)
            conf_thresh: Confidence threshold for detection
            iou_thresh: IOU threshold for NMS
            target_shape: Input shape for the model
            
        Returns:
            Tuple of (image_with_detections, detection_info)
            where detection_info is a list of [class_id, confidence, xmin, ymin, xmax, ymax]
        """
        height, width, _ = image.shape
        
        # Prepare input blob
        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=target_shape)
        self.net.setInput(blob)
        
        # Forward pass
        y_bboxes_output, y_cls_output = self.net.forward(self.get_output_names())
        
        # Decode bounding boxes
        y_bboxes = decode_bbox(self.anchors_exp, y_bboxes_output)[0]
        y_cls = y_cls_output[0]
        
        # Get max scores and classes
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)
        
        # Apply NMS
        keep_idxs = single_class_non_max_suppression(
            y_bboxes, 
            bbox_max_scores, 
            conf_thresh=conf_thresh, 
            iou_thresh=iou_thresh
        )
        
        detection_info = []
        
        # Draw results
        for idx in keep_idxs:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            bbox = y_bboxes[idx]
            
            # Clip coordinates
            xmin = max(0, int(bbox[0] * width))
            ymin = max(0, int(bbox[1] * height))
            xmax = min(int(bbox[2] * width), width)
            ymax = min(int(bbox[3] * height), height)
            
            # Draw bounding box
            color = self.colors[class_id]
            thickness = max(2, int(0.002 * (height + width) * 0.5))
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
            
            # Draw label
            label = f"{self.id2class[class_id]}: {conf:.2f}"
            cv2.putText(image, label, (xmin + 2, ymin - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            detection_info.append([class_id, conf, xmin, ymin, xmax, ymax])
        
        return image, detection_info
    
    def get_detection_summary(self, detection_info):
        """
        Get a summary of detections.
        
        Args:
            detection_info: List of detection results
            
        Returns:
            Dictionary with summary statistics
        """
        mask_count = sum(1 for d in detection_info if d[0] == 0)
        no_mask_count = sum(1 for d in detection_info if d[0] == 1)
        
        return {
            'total_faces': len(detection_info),
            'with_mask': mask_count,
            'without_mask': no_mask_count,
            'compliance_rate': mask_count / len(detection_info) * 100 if detection_info else 0
        }
