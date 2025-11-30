"""
Team classification utilities for SportsVision+.

This module provides a simplified KMeans-based team classifier that uses
average jersey colors for fast real-time classification.

IMPORTANT: This is a SIMPLIFIED version optimized for real-time performance.
The original Roboflow Sports library uses SigLIP + UMAP + KMeans which is 
too slow for 30 FPS real-time inference on RTX 3050/3060/4050.

Our approach:
- Extract the upper body region (jersey) from player crops
- Compute mean color in LAB color space (better for color discrimination)
- Use KMeans clustering on LAB colors

Based on: Roboflow Sports library concept, but simplified for real-time
Notebook Reference: football_ai.ipynb Cells 31-35, 51
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """
    Fast team classifier using jersey color clustering.
    
    Uses KMeans on LAB color features extracted from player crops.
    Optimized for real-time inference at 30 FPS.
    
    Attributes:
        cluster_model (KMeans): Fitted KMeans model for team classification.
        team_colors (np.ndarray): Representative LAB colors for each team.
    """
    
    def __init__(self, n_clusters: int = 2):
        """
        Initialize the TeamClassifier.
        
        Args:
            n_clusters (int): Number of teams to classify (default: 2).
        """
        self.n_clusters = n_clusters
        # Use n_init='auto' for compatibility with scikit-learn 1.4+
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.team_colors: Optional[np.ndarray] = None
        self._is_fitted = False
    
    def _extract_jersey_region(
        self,
        crop: np.ndarray,
        top_ratio: float = 0.1,
        bottom_ratio: float = 0.5
    ) -> np.ndarray:
        """
        Extract the jersey region from a player crop.
        
        The jersey is typically in the upper-middle portion of the bounding box.
        We skip the head area and focus on the torso.
        
        Args:
            crop (np.ndarray): Player crop image (BGR).
            top_ratio (float): Skip this ratio from the top (head area).
            bottom_ratio (float): Take up to this ratio from the top (torso).
        
        Returns:
            np.ndarray: Jersey region crop.
        """
        h, w = crop.shape[:2]
        top = int(h * top_ratio)
        bottom = int(h * bottom_ratio)
        
        # Also crop horizontally to focus on center (avoid arms at edges)
        left = int(w * 0.2)
        right = int(w * 0.8)
        
        return crop[top:bottom, left:right]
    
    def _extract_color_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract LAB color features from a crop.
        
        LAB color space is perceptually uniform and better for
        distinguishing between different jersey colors.
        
        Args:
            crop (np.ndarray): Image crop in BGR format.
        
        Returns:
            np.ndarray: Mean LAB color values (L, A, B).
        """
        if crop.size == 0:
            return np.zeros(3, dtype=np.float32)
        
        # Convert to LAB
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        
        # Compute mean color
        mean_color = lab.mean(axis=(0, 1))
        
        return mean_color.astype(np.float32)
    
    def _extract_features_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract color features from a batch of crops.
        
        Args:
            crops (List[np.ndarray]): List of player crop images.
        
        Returns:
            np.ndarray: Feature matrix of shape (N, 3).
        """
        features = []
        for crop in crops:
            jersey = self._extract_jersey_region(crop)
            if jersey.size > 0:
                feature = self._extract_color_features(jersey)
            else:
                feature = self._extract_color_features(crop)
            features.append(feature)
        
        return np.array(features, dtype=np.float32)
    
    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier on a collection of player crops.
        
        This should be called with crops from multiple frames to ensure
        both teams are represented in the training data.
        
        Args:
            crops (List[np.ndarray]): List of player crop images.
        """
        if len(crops) < self.n_clusters:
            raise ValueError(
                f"Need at least {self.n_clusters} crops to fit classifier, "
                f"got {len(crops)}"
            )
        
        # Extract features
        features = self._extract_features_batch(crops)
        
        # Fit KMeans
        self.cluster_model.fit(features)
        
        # Store team colors (cluster centers)
        self.team_colors = self.cluster_model.cluster_centers_.copy()
        self._is_fitted = True
    
    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict team IDs for a list of player crops.
        
        Args:
            crops (List[np.ndarray]): List of player crop images.
        
        Returns:
            np.ndarray: Predicted team IDs (0 or 1) for each crop.
        """
        if not self._is_fitted:
            raise RuntimeError("Classifier must be fitted before prediction. Call fit() first.")
        
        if len(crops) == 0:
            return np.array([], dtype=np.int32)
        
        # Extract features
        features = self._extract_features_batch(crops)
        
        # Predict clusters
        predictions = self.cluster_model.predict(features)
        
        return predictions.astype(np.int32)
    
    def get_team_colors_bgr(self) -> List[Tuple[int, int, int]]:
        """
        Get the representative BGR colors for each team.
        
        Useful for visualization (drawing with team colors).
        
        Returns:
            List[Tuple[int, int, int]]: List of BGR color tuples.
        """
        if self.team_colors is None:
            return [(255, 191, 0), (147, 20, 255)]  # Default: Light Blue/Cyan, Magenta/Pink
        
        bgr_colors = []
        for lab_color in self.team_colors:
            # Convert single LAB pixel to BGR
            lab_pixel = np.uint8([[lab_color]])
            bgr_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2BGR)
            bgr_colors.append(tuple(map(int, bgr_pixel[0, 0])))
        
        return bgr_colors
    
    @property
    def is_fitted(self) -> bool:
        """Check if the classifier has been fitted."""
        return self._is_fitted


class TeamClassifierAdvanced:
    """
    Advanced team classifier using histogram-based color features.
    
    More robust than simple mean color but still fast enough for real-time.
    Uses LAB color histograms for better discrimination.
    """
    
    def __init__(
        self,
        n_clusters: int = 2,
        hist_bins: int = 16,
        use_histogram: bool = True
    ):
        """
        Initialize the advanced classifier.
        
        Args:
            n_clusters (int): Number of teams (default: 2).
            hist_bins (int): Number of bins for color histograms.
            use_histogram (bool): Whether to use histogram features.
        """
        self.n_clusters = n_clusters
        self.hist_bins = hist_bins
        self.use_histogram = use_histogram
        # Use n_init='auto' for compatibility with scikit-learn 1.4+
        self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self._is_fitted = False
    
    def _extract_jersey_region(
        self,
        crop: np.ndarray,
        top_ratio: float = 0.1,
        bottom_ratio: float = 0.5
    ) -> np.ndarray:
        """Extract jersey region from player crop."""
        h, w = crop.shape[:2]
        top = int(h * top_ratio)
        bottom = int(h * bottom_ratio)
        left = int(w * 0.2)
        right = int(w * 0.8)
        return crop[top:bottom, left:right]
    
    def _extract_histogram_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract LAB color histogram features from a crop.
        
        Args:
            crop (np.ndarray): Image crop in BGR format.
        
        Returns:
            np.ndarray: Flattened histogram feature vector.
        """
        if crop.size == 0:
            return np.zeros(self.hist_bins * 3, dtype=np.float32)
        
        # Convert to LAB
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        
        # Compute histograms for each channel
        features = []
        for i in range(3):
            hist = cv2.calcHist([lab], [i], None, [self.hist_bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.append(hist)
        
        return np.concatenate(features).astype(np.float32)
    
    def _extract_features_batch(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract features from a batch of crops."""
        features = []
        for crop in crops:
            jersey = self._extract_jersey_region(crop)
            if jersey.size > 0:
                if self.use_histogram:
                    feature = self._extract_histogram_features(jersey)
                else:
                    lab = cv2.cvtColor(jersey, cv2.COLOR_BGR2LAB)
                    feature = lab.mean(axis=(0, 1)).astype(np.float32)
            else:
                feature = np.zeros(self.hist_bins * 3 if self.use_histogram else 3, dtype=np.float32)
            features.append(feature)
        
        return np.array(features, dtype=np.float32)
    
    def fit(self, crops: List[np.ndarray]) -> None:
        """Fit the classifier on player crops."""
        if len(crops) < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} crops")
        
        features = self._extract_features_batch(crops)
        self.cluster_model.fit(features)
        self._is_fitted = True
    
    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """Predict team IDs for player crops."""
        if not self._is_fitted:
            raise RuntimeError("Call fit() before predict()")
        
        if len(crops) == 0:
            return np.array([], dtype=np.int32)
        
        features = self._extract_features_batch(crops)
        return self.cluster_model.predict(features).astype(np.int32)
    
    @property
    def is_fitted(self) -> bool:
        """Check if classifier is fitted."""
        return self._is_fitted
