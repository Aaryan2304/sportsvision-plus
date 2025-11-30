"""
Tracker utilities for SportsVision+.

This module provides ByteTrack integration with ball smoothing/anti-teleport logic
to handle erratic ball detections and interpolate missing frames.

Based on: Roboflow Sports library (github.com/roboflow/sports)
Notebook Reference: football_ai.ipynb Cells 29, 57
"""

from collections import deque
from typing import Optional, Tuple

import numpy as np
import supervision as sv


class BallTracker:
    """
    A tracker for the soccer ball with anti-teleport smoothing.
    
    Maintains a buffer of recent ball positions and uses centroid-based
    selection to reject outlier detections. Also supports linear interpolation
    for missing frames.
    
    Attributes:
        buffer (deque): Recent ball positions for smoothing.
        max_distance (float): Maximum allowed movement per frame in pixels.
        last_position (np.ndarray): Last known ball position for interpolation.
        missing_frames (int): Count of consecutive frames without detection.
    """
    
    def __init__(
        self,
        buffer_size: int = 10,
        max_distance: float = 500.0,
        max_interpolation_frames: int = 5,
        relaxed_distance_multiplier: float = 2.0
    ):
        """
        Initialize the BallTracker.
        
        Args:
            buffer_size (int): Size of the position history buffer.
            max_distance (float): Maximum allowed pixel displacement per frame.
                                  Detections exceeding this are rejected as teleports.
            max_interpolation_frames (int): Max frames to interpolate missing ball.
            relaxed_distance_multiplier (float): Multiplier for max_distance when all
                                                  detections are teleports but we need
                                                  to pick the best candidate.
        """
        self.buffer: deque = deque(maxlen=buffer_size)
        self.max_distance = max_distance
        self.max_interpolation_frames = max_interpolation_frames
        self.relaxed_distance_multiplier = relaxed_distance_multiplier
        self.last_position: Optional[np.ndarray] = None
        self.last_velocity: Optional[np.ndarray] = None
        self.missing_frames: int = 0
    
    def _compute_centroid(self) -> Optional[np.ndarray]:
        """Compute the centroid of buffered positions."""
        if len(self.buffer) == 0:
            return None
        # Use vstack to handle buffer entries with varying numbers of detections
        return np.mean(np.vstack(list(self.buffer)), axis=0)
    
    def _is_valid_position(self, position: np.ndarray) -> bool:
        """
        Check if a position is valid (not a teleport).
        
        Args:
            position (np.ndarray): Candidate ball position (x, y).
        
        Returns:
            bool: True if position is within acceptable distance of history.
        """
        if self.last_position is None:
            return True
        
        distance = np.linalg.norm(position - self.last_position)
        return distance <= self.max_distance
    
    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracker with new detections and return filtered detection.
        
        This method:
        1. Rejects detections that "teleport" (move too far in one frame)
        2. Selects the detection closest to the historical centroid
        3. Interpolates position if detection is missing for a few frames
        
        Args:
            detections (sv.Detections): Ball detections for current frame.
        
        Returns:
            sv.Detections: Filtered ball detection (single or empty).
        """
        # Get candidate positions
        if len(detections) == 0:
            xy = np.empty((0, 2), dtype=np.float32)
        else:
            xy = detections.get_anchors_coordinates(sv.Position.CENTER)
        
        # No detections - try interpolation
        if len(detections) == 0:
            self.missing_frames += 1
            
            # Interpolate if we have velocity and haven't exceeded max frames
            if (self.last_position is not None and 
                self.last_velocity is not None and 
                self.missing_frames <= self.max_interpolation_frames):
                
                # Predict next position using velocity
                predicted_position = self.last_position + self.last_velocity
                self.last_position = predicted_position
                
                # Return empty detections (caller can use get_interpolated_position)
                return detections
            
            return detections
        
        # Reset missing frame counter
        self.missing_frames = 0
        
        # Filter out teleporting detections
        valid_mask = np.array([self._is_valid_position(pos) for pos in xy])
        
        if not np.any(valid_mask):
            # All detections are teleports - keep the one closest to centroid
            centroid = self._compute_centroid()
            if centroid is not None:
                distances = np.linalg.norm(xy - centroid, axis=1)
                best_idx = np.argmin(distances)
                
                # Only accept if within relaxed distance threshold of centroid
                if distances[best_idx] <= self.max_distance * self.relaxed_distance_multiplier:
                    valid_mask[best_idx] = True
        
        if not np.any(valid_mask):
            # No valid detections at all
            return sv.Detections.empty()
        
        # Filter detections
        valid_indices = np.where(valid_mask)[0]
        filtered_detections = detections[valid_indices]
        filtered_xy = xy[valid_mask]
        
        # Select detection closest to centroid
        centroid = self._compute_centroid()
        if centroid is not None and len(filtered_xy) > 1:
            distances = np.linalg.norm(filtered_xy - centroid, axis=1)
            best_idx = np.argmin(distances)
            # Use [[idx]] fancy indexing to maintain sv.Detections object structure
            selected_detection = filtered_detections[[best_idx]]
            selected_position = filtered_xy[best_idx]
        else:
            selected_detection = filtered_detections[[0]]
            selected_position = filtered_xy[0]
        
        # Update velocity for interpolation
        if self.last_position is not None:
            self.last_velocity = selected_position - self.last_position
        
        self.last_position = selected_position.copy()
        
        # Add validated position to buffer (only after filtering teleports)
        # Store as 1D array (2,) for consistency with np.vstack in _compute_centroid
        self.buffer.append(selected_position.copy())
        
        return selected_detection
    
    def get_interpolated_position(self) -> Optional[np.ndarray]:
        """
        Get the interpolated ball position when detection is missing.
        
        Returns:
            Optional[np.ndarray]: Interpolated (x, y) position or None.
        """
        if self.missing_frames > 0 and self.last_position is not None:
            return self.last_position.copy()
        return None
    
    def reset(self) -> None:
        """Reset the tracker state."""
        self.buffer.clear()
        self.last_position = None
        self.last_velocity = None
        self.missing_frames = 0


class PlayerTracker:
    """
    Wrapper around supervision's ByteTrack for player tracking.
    
    Provides additional functionality for managing track IDs and
    filtering specific object classes.
    """
    
    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 3
    ):
        """
        Initialize the PlayerTracker.
        
        Args:
            track_activation_threshold (float): Detection confidence threshold for
                                                 creating new tracks.
            lost_track_buffer (int): Number of frames to keep lost tracks alive.
            minimum_matching_threshold (float): IoU threshold for matching detections.
            frame_rate (int): Video frame rate (used for track lifetime).
            minimum_consecutive_frames (int): Min frames before track is confirmed.
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames
        )
    
    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracker with new detections.
        
        Args:
            detections (sv.Detections): Current frame detections.
        
        Returns:
            sv.Detections: Detections with assigned tracker IDs.
        """
        return self.tracker.update_with_detections(detections)
    
    def reset(self) -> None:
        """Reset the tracker state."""
        self.tracker.reset()


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.ndarray,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Assign team IDs to goalkeepers based on proximity to team centroids.
    
    Goalkeepers are assigned to the team whose players are closer to them
    on the pitch. This is because goalkeepers often wear different jerseys
    than their teammates, making color-based classification unreliable.
    
    Args:
        players (sv.Detections): Detected players (excluding goalkeepers).
        players_team_id (np.ndarray): Team ID (0 or 1) for each player.
        goalkeepers (sv.Detections): Detected goalkeepers.
    
    Returns:
        np.ndarray: Team IDs for each goalkeeper.
    
    Based on: Roboflow sports library resolve_goalkeepers_team_id function.
    Notebook Reference: football_ai.ipynb Cell 57.
    """
    if len(goalkeepers) == 0:
        return np.array([], dtype=np.int32)
    
    if len(players) == 0:
        # No players to compare - assign all to team 0
        return np.zeros(len(goalkeepers), dtype=np.int32)
    
    # Get positions
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    
    # Compute team masks
    team_0_mask = players_team_id == 0
    team_1_mask = players_team_id == 1
    
    has_team_0 = np.any(team_0_mask)
    has_team_1 = np.any(team_1_mask)
    
    # Handle edge cases where one or both teams have no players detected
    if not has_team_0 and not has_team_1:
        # No players for either team: assign all goalkeepers to team 0 (default)
        return np.zeros(len(goalkeepers), dtype=np.int32)
    elif not has_team_0:
        # Only team 1 has players: assign all goalkeepers to team 1
        return np.ones(len(goalkeepers), dtype=np.int32)
    elif not has_team_1:
        # Only team 0 has players: assign all goalkeepers to team 0
        return np.zeros(len(goalkeepers), dtype=np.int32)
    
    # Both teams have players: compute centroids and assign by proximity
    team_0_centroid = players_xy[team_0_mask].mean(axis=0)
    team_1_centroid = players_xy[team_1_mask].mean(axis=0)
    
    # Assign each goalkeeper to nearest team
    goalkeeper_team_ids = []
    for gk_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(gk_xy - team_0_centroid)
        dist_1 = np.linalg.norm(gk_xy - team_1_centroid)
        goalkeeper_team_ids.append(0 if dist_0 < dist_1 else 1)
    
    return np.array(goalkeeper_team_ids, dtype=np.int32)
