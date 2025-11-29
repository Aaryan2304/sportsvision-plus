"""
Geometry utilities for SportsVision+.

This module provides the ViewTransformer class for homography-based perspective
transformation, enabling mapping of camera coordinates to 2D pitch coordinates.

Based on: Roboflow Sports library (github.com/roboflow/sports)
Notebook Reference: football_ai.ipynb Cell 78
"""

import cv2
import numpy as np
from typing import Tuple
import numpy.typing as npt


class ViewTransformer:
    """
    A class to perform perspective transformation using homography.
    
    This transformer maps points from a source plane (camera view) to a target plane
    (2D pitch coordinates) using a homography matrix computed from corresponding points.
    
    Attributes:
        m (np.ndarray): The 3x3 homography matrix.
    
    Example:
        >>> source_points = np.array([[100, 200], [300, 200], [300, 400], [100, 400]])
        >>> target_points = np.array([[0, 0], [105, 0], [105, 68], [0, 68]])
        >>> transformer = ViewTransformer(source_points, target_points)
        >>> camera_points = np.array([[150, 250], [250, 350]])
        >>> pitch_points = transformer.transform_points(camera_points)
    """
    
    def __init__(
        self,
        source: npt.NDArray[np.float32],
        target: npt.NDArray[np.float32]
    ) -> None:
        """
        Initialize the ViewTransformer with source and target point correspondences.
        
        Args:
            source (np.ndarray): Array of points in the source plane (camera view).
                                 Shape: (N, 2) where N >= 4 for homography.
            target (np.ndarray): Array of corresponding points in the target plane
                                 (pitch coordinates). Shape: (N, 2).
        
        Raises:
            ValueError: If source and target have different lengths or fewer than 4 points.
        """
        if len(source) != len(target):
            raise ValueError("Source and target must have the same number of points.")
        if len(source) < 4:
            raise ValueError("At least 4 point correspondences are required for homography.")
        
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(source, target)
        
        if self.m is None:
            raise ValueError("Failed to compute homography. Points may be collinear or degenerate.")
    
    def transform_points(
        self,
        points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        Transform points from source plane to target plane using the homography matrix.
        
        Args:
            points (np.ndarray): Array of points to transform. Shape: (N, 2).
        
        Returns:
            np.ndarray: Transformed points in target plane coordinates. Shape: (N, 2).
        
        Raises:
            ValueError: If points are not 2D coordinates.
        """
        # Handle empty input
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        
        # Handle single point case
        if points.ndim == 1:
            points = points.reshape(1, 2)
        
        # Validate 2D coordinates
        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates (shape: N x 2).")
        
        # Reshape for cv2.perspectiveTransform: (N, 1, 2)
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        
        # Apply perspective transformation
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        
        if transformed_points is None:
            return np.empty((0, 2), dtype=np.float32)
        
        return transformed_points.reshape(-1, 2).astype(np.float32)
    
    def transform_image(
        self,
        image: npt.NDArray[np.uint8],
        resolution_wh: Tuple[int, int]
    ) -> npt.NDArray[np.uint8]:
        """
        Transform an entire image using the homography matrix.
        
        Args:
            image (np.ndarray): Input image to transform.
            resolution_wh (Tuple[int, int]): Output resolution as (width, height).
        
        Returns:
            np.ndarray: Warped image with the specified resolution.
        """
        return cv2.warpPerspective(image, self.m, resolution_wh)
    
    @property
    def homography_matrix(self) -> np.ndarray:
        """Get the computed homography matrix."""
        return self.m.copy()


def get_foot_position(bbox: np.ndarray) -> np.ndarray:
    """
    Get the foot position (bottom center) from bounding boxes.
    
    This is useful for determining player positions on the pitch since
    feet touch the ground plane.
    
    Args:
        bbox (np.ndarray): Bounding boxes in format [x1, y1, x2, y2].
                          Shape: (N, 4) or (4,).
    
    Returns:
        np.ndarray: Bottom center points. Shape: (N, 2) or (2,).
    """
    if bbox.ndim == 1:
        x_center = (bbox[0] + bbox[2]) / 2
        y_bottom = bbox[3]
        return np.array([x_center, y_bottom], dtype=np.float32)
    
    x_center = (bbox[:, 0] + bbox[:, 2]) / 2
    y_bottom = bbox[:, 3]
    return np.column_stack([x_center, y_bottom]).astype(np.float32)


def compute_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two points.
    
    Args:
        point1 (np.ndarray): First point (x, y).
        point2 (np.ndarray): Second point (x, y).
    
    Returns:
        float: Euclidean distance between the points.
    """
    return float(np.linalg.norm(point1 - point2))


def compute_distances_batch(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between two sets of points.
    
    Args:
        points1 (np.ndarray): First set of points. Shape: (N, 2).
        points2 (np.ndarray): Second set of points. Shape: (M, 2).
    
    Returns:
        np.ndarray: Distance matrix. Shape: (N, M).
    """
    # Broadcasting: (N, 1, 2) - (1, M, 2) -> (N, M, 2)
    diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)
