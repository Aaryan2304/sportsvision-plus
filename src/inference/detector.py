"""
Object detection utilities for SportsVision+.

This module provides wrappers for YOLO11 (player/ball detection) and
Roboflow pitch keypoint detection models.

Based on: Roboflow Sports library and Ultralytics YOLO
Notebook Reference: football_ai.ipynb Cells 11, 19, 38, 64
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import cv2
import numpy as np
import supervision as sv


# Class ID mappings for SportsVision+ detection
CLASS_NAMES = {
    0: "ball",
    1: "goalkeeper",
    2: "player",
    3: "referee"
}

# Reverse mapping
CLASS_IDS = {v: k for k, v in CLASS_NAMES.items()}

# Convenience constants
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3


class ObjectDetector:
    """
    YOLO-based object detector for players, goalkeepers, referees, and ball.
    
    Wraps Ultralytics YOLO model with convenience methods for
    soccer-specific detection and filtering.
    
    Attributes:
        model: Loaded YOLO model.
        confidence (float): Detection confidence threshold.
        device (str): Device for inference ('cuda', 'cpu', 'mps').
    """
    
    def __init__(
        self,
        model_path: str,
        confidence: float = 0.3,
        device: str = "cuda",
        imgsz: int = 1280
    ):
        """
        Initialize the ObjectDetector.
        
        Args:
            model_path (str): Path to YOLO model weights (.pt file).
            confidence (float): Detection confidence threshold (0-1).
            device (str): Device for inference ('cuda', 'cpu', 'mps').
            imgsz (int): Input image size for YOLO inference.
        
        Raises:
            FileNotFoundError: If model file doesn't exist.
            ImportError: If ultralytics is not installed.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Ultralytics is required. Install with: pip install ultralytics"
            )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(model_path)
        self.model.to(device)
        self.confidence = confidence
        self.device = device
        self.imgsz = imgsz
    
    def detect(
        self,
        frame: np.ndarray,
        verbose: bool = False
    ) -> sv.Detections:
        """
        Run object detection on a frame.
        
        Args:
            frame (np.ndarray): Input frame in BGR format.
            verbose (bool): Whether to print YOLO inference details.
        
        Returns:
            sv.Detections: Detected objects with bounding boxes, confidence, class IDs.
        """
        results = self.model(
            frame,
            imgsz=self.imgsz,
            conf=self.confidence,
            verbose=verbose
        )[0]
        
        return sv.Detections.from_ultralytics(results)
    
    def detect_and_filter(
        self,
        frame: np.ndarray,
        nms_threshold: float = 0.5,
        pad_ball: int = 10,
        verbose: bool = False
    ) -> Tuple[sv.Detections, sv.Detections]:
        """
        Detect objects and separate ball from other detections.
        
        Applies NMS to non-ball detections and pads ball bounding boxes
        for better visualization.
        
        Args:
            frame (np.ndarray): Input frame in BGR format.
            nms_threshold (float): NMS IoU threshold for non-ball detections.
            pad_ball (int): Pixels to pad ball bounding box.
            verbose (bool): Print inference details.
        
        Returns:
            Tuple[sv.Detections, sv.Detections]: 
                - all_detections: Players, goalkeepers, referees (with NMS)
                - ball_detections: Ball detections (padded)
        """
        detections = self.detect(frame, verbose=verbose)
        
        # Separate ball detections
        ball_mask = detections.class_id == BALL_CLASS_ID
        ball_detections = detections[ball_mask]
        all_detections = detections[~ball_mask]
        
        # Pad ball bounding boxes (create new Detections to avoid in-place mutation)
        if len(ball_detections) > 0:
            padded_xyxy = sv.pad_boxes(
                xyxy=ball_detections.xyxy,
                px=pad_ball
            )
            ball_detections = sv.Detections(
                xyxy=padded_xyxy,
                class_id=ball_detections.class_id,
                confidence=ball_detections.confidence,
                tracker_id=ball_detections.tracker_id if hasattr(ball_detections, 'tracker_id') else None,
                data=ball_detections.data if hasattr(ball_detections, 'data') else {}
            )
        
        # Apply NMS to non-ball detections
        if len(all_detections) > 0:
            all_detections = all_detections.with_nms(
                threshold=nms_threshold,
                class_agnostic=True
            )
        
        return all_detections, ball_detections
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        return CLASS_NAMES.get(class_id, f"unknown_{class_id}")
    
    @property
    def class_names(self) -> Dict[int, str]:
        """Get class ID to name mapping."""
        return CLASS_NAMES.copy()


class PitchDetector:
    """
    Keypoint detector for soccer pitch field lines.
    
    Uses a specialized YOLO-pose model trained on pitch keypoints
    to detect field reference points for homography calculation.
    
    The model detects 32 keypoints corresponding to field markings:
    corners, penalty box vertices, center circle points, etc.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        roboflow_api_key: Optional[str] = None,
        roboflow_model_id: str = "football-field-detection-f07vi-pjo7k/1",
        confidence: float = 0.3,
        device: str = "cuda",
        use_roboflow: bool = False
    ):
        """
        Initialize the PitchDetector.
        
        Can use either a local YOLO model or Roboflow hosted model.
        
        Args:
            model_path (str): Path to local pitch detection model.
            roboflow_api_key (str): Roboflow API key (if using hosted model).
            roboflow_model_id (str): Roboflow model ID.
            confidence (float): Detection confidence threshold.
            device (str): Device for inference.
            use_roboflow (bool): Whether to use Roboflow hosted model.
        
        Raises:
            ValueError: If neither model_path nor Roboflow credentials provided.
        """
        self.confidence = confidence
        self.device = device
        self.use_roboflow = use_roboflow
        self.model = None
        
        if use_roboflow:
            if roboflow_api_key is None:
                raise ValueError(
                    "roboflow_api_key required when use_roboflow=True"
                )
            try:
                from inference import get_model
                self.model = get_model(
                    model_id=roboflow_model_id,
                    api_key=roboflow_api_key
                )
            except ImportError:
                raise ImportError(
                    "Roboflow inference is required. "
                    "Install with: pip install inference"
                )
        else:
            if model_path is None:
                raise ValueError(
                    "model_path required when use_roboflow=False"
                )
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            try:
                from ultralytics import YOLO
                self.model = YOLO(model_path)
                self.model.to(device)
            except ImportError:
                raise ImportError(
                    "Ultralytics is required. Install with: pip install ultralytics"
                )
    
    def detect(
        self,
        frame: np.ndarray,
        verbose: bool = False
    ) -> sv.KeyPoints:
        """
        Detect pitch keypoints in a frame.
        
        Args:
            frame (np.ndarray): Input frame in BGR format.
            verbose (bool): Print inference details.
        
        Returns:
            sv.KeyPoints: Detected keypoints with coordinates and confidence.
        """
        if self.use_roboflow:
            result = self.model.infer(frame, confidence=self.confidence)[0]
            return sv.KeyPoints.from_inference(result)
        else:
            result = self.model(frame, conf=self.confidence, verbose=verbose)[0]
            return sv.KeyPoints.from_ultralytics(result)
    
    def detect_with_filter(
        self,
        frame: np.ndarray,
        min_confidence: float = 0.5,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and filter keypoints by confidence.
        
        Returns only keypoints with confidence above threshold,
        useful for robust homography estimation.
        
        Args:
            frame (np.ndarray): Input frame in BGR format.
            min_confidence (float): Minimum keypoint confidence.
            verbose (bool): Print inference details.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Filtered keypoint coordinates (N, 2)
                - Boolean mask indicating which keypoints passed filter
        """
        keypoints = self.detect(frame, verbose=verbose)
        
        if len(keypoints) == 0 or keypoints.xy.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32), np.array([], dtype=bool)
        
        # Filter by confidence
        confidence_mask = keypoints.confidence[0] > min_confidence
        filtered_points = keypoints.xy[0][confidence_mask]
        
        return filtered_points.astype(np.float32), confidence_mask


class CombinedDetector:
    """
    Combined detector for both objects and pitch keypoints.
    
    Provides a unified interface for all detection needs in the pipeline.
    Manages model loading and provides optimized detection methods.
    """
    
    def __init__(
        self,
        object_model_path: str,
        pitch_model_path: Optional[str] = None,
        roboflow_api_key: Optional[str] = None,
        confidence: float = 0.3,
        device: str = "cuda",
        imgsz: int = 1280
    ):
        """
        Initialize the CombinedDetector.
        
        Args:
            object_model_path (str): Path to player/ball YOLO model.
            pitch_model_path (str): Path to pitch keypoint model (optional).
            roboflow_api_key (str): Roboflow API key (if using hosted pitch model).
            confidence (float): Detection confidence threshold.
            device (str): Device for inference.
            imgsz (int): Input image size.
        """
        # Initialize object detector
        self.object_detector = ObjectDetector(
            model_path=object_model_path,
            confidence=confidence,
            device=device,
            imgsz=imgsz
        )
        
        # Initialize pitch detector if credentials provided
        self.pitch_detector: Optional[PitchDetector] = None
        if pitch_model_path is not None:
            self.pitch_detector = PitchDetector(
                model_path=pitch_model_path,
                confidence=confidence,
                device=device,
                use_roboflow=False
            )
        elif roboflow_api_key is not None:
            self.pitch_detector = PitchDetector(
                roboflow_api_key=roboflow_api_key,
                confidence=confidence,
                use_roboflow=True
            )
        
        self.confidence = confidence
        self.device = device
    
    def detect_all(
        self,
        frame: np.ndarray,
        detect_pitch: bool = True,
        pitch_confidence: float = 0.5,
        nms_threshold: float = 0.5,
        pad_ball: int = 10,
        verbose: bool = False
    ) -> Dict:
        """
        Perform all detections on a frame.
        
        Args:
            frame (np.ndarray): Input frame in BGR format.
            detect_pitch (bool): Whether to detect pitch keypoints.
            pitch_confidence (float): Minimum confidence for pitch keypoints.
            nms_threshold (float): NMS threshold for object detections.
            pad_ball (int): Padding for ball bounding boxes.
            verbose (bool): Print inference details.
        
        Returns:
            Dict containing:
                - 'all_detections': Players, goalkeepers, referees
                - 'ball_detections': Ball detections
                - 'pitch_keypoints': Filtered pitch keypoints (if detect_pitch=True)
                - 'pitch_mask': Boolean mask of valid keypoints
        """
        # Object detection
        all_detections, ball_detections = self.object_detector.detect_and_filter(
            frame,
            nms_threshold=nms_threshold,
            pad_ball=pad_ball,
            verbose=verbose
        )
        
        result = {
            'all_detections': all_detections,
            'ball_detections': ball_detections,
            'pitch_keypoints': None,
            'pitch_mask': None
        }
        
        # Pitch detection
        if detect_pitch and self.pitch_detector is not None:
            pitch_points, pitch_mask = self.pitch_detector.detect_with_filter(
                frame,
                min_confidence=pitch_confidence,
                verbose=verbose
            )
            result['pitch_keypoints'] = pitch_points
            result['pitch_mask'] = pitch_mask
        
        return result
    
    def has_pitch_detector(self) -> bool:
        """Check if pitch detector is available."""
        return self.pitch_detector is not None


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract image crops from detections.
    
    Helper function for team classification - extracts player bounding boxes
    from the frame.
    
    Args:
        frame (np.ndarray): Source frame in BGR format.
        detections (sv.Detections): Detections with bounding boxes.
    
    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def filter_by_class(
    detections: sv.Detections,
    class_ids: Union[int, List[int]]
) -> sv.Detections:
    """
    Filter detections by class ID(s).
    
    Args:
        detections (sv.Detections): Input detections.
        class_ids (int or List[int]): Class ID(s) to keep.
    
    Returns:
        sv.Detections: Filtered detections.
    """
    if isinstance(class_ids, int):
        class_ids = [class_ids]
    
    mask = np.isin(detections.class_id, class_ids)
    return detections[mask]
