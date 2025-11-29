"""
Visualization utilities for SportsVision+.

This module provides functions for drawing soccer pitch visualizations,
player positions, and detection annotations.

Based on: Roboflow Sports library (github.com/roboflow/sports)
Notebook Reference: football_ai.ipynb Cells 73-76, 79, 83
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import supervision as sv


# =============================================================================
# Soccer Pitch Configuration
# =============================================================================

@dataclass
class SoccerPitchConfiguration:
    """
    Configuration for standard soccer pitch dimensions.
    
    All measurements are in centimeters (cm) for precision.
    Default values are based on the Roboflow Sports library configuration and the
    football_ai.ipynb notebook reference. These may differ from official FIFA standards.
    
    Attributes:
        width (int): Pitch width in cm (touchline to touchline). Default: 7000 (70m)
        length (int): Pitch length in cm (goal line to goal line). Default: 12000 (120m)
        penalty_box_width (int): Width of penalty area. Default: 4100 (41m)
        penalty_box_length (int): Length of penalty area. Default: 2015 (20.15m)
        goal_box_width (int): Width of goal area. Default: 1832 (18.32m)
        goal_box_length (int): Length of goal area. Default: 550 (5.5m)
        centre_circle_radius (int): Radius of center circle. Default: 915 (9.15m)
        penalty_spot_distance (int): Distance from goal to penalty spot. Default: 1100 (11m)
    """
    width: int = 7000  # [cm] - 70 meters
    length: int = 12000  # [cm] - 120 meters
    penalty_box_width: int = 4100  # [cm]
    penalty_box_length: int = 2015  # [cm]
    goal_box_width: int = 1832  # [cm]
    goal_box_length: int = 550  # [cm]
    centre_circle_radius: int = 915  # [cm]
    penalty_spot_distance: int = 1100  # [cm]
    
    @property
    def vertices(self) -> List[Tuple[int, int]]:
        """
        Get all key vertices of the pitch.
        
        Returns 32 key points that define the pitch layout including:
        - Corner points
        - Penalty box corners
        - Goal box corners
        - Center line points
        - Penalty spots
        - Center circle intersection points
        """
        return [
            (0, 0),  # 1 - Top left corner
            (0, (self.width - self.penalty_box_width) / 2),  # 2
            (0, (self.width - self.goal_box_width) / 2),  # 3
            (0, (self.width + self.goal_box_width) / 2),  # 4
            (0, (self.width + self.penalty_box_width) / 2),  # 5
            (0, self.width),  # 6 - Bottom left corner
            (self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 7
            (self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 8
            (self.penalty_spot_distance, self.width / 2),  # 9 - Left penalty spot
            (self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 10
            (self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 11
            (self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 12
            (self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 13
            (self.length / 2, 0),  # 14 - Center line top
            (self.length / 2, self.width / 2 - self.centre_circle_radius),  # 15
            (self.length / 2, self.width / 2 + self.centre_circle_radius),  # 16
            (self.length / 2, self.width),  # 17 - Center line bottom
            (self.length - self.penalty_box_length, (self.width - self.penalty_box_width) / 2),  # 18
            (self.length - self.penalty_box_length, (self.width - self.goal_box_width) / 2),  # 19
            (self.length - self.penalty_box_length, (self.width + self.goal_box_width) / 2),  # 20
            (self.length - self.penalty_box_length, (self.width + self.penalty_box_width) / 2),  # 21
            (self.length - self.penalty_spot_distance, self.width / 2),  # 22 - Right penalty spot
            (self.length - self.goal_box_length, (self.width - self.goal_box_width) / 2),  # 23
            (self.length - self.goal_box_length, (self.width + self.goal_box_width) / 2),  # 24
            (self.length, 0),  # 25 - Top right corner
            (self.length, (self.width - self.penalty_box_width) / 2),  # 26
            (self.length, (self.width - self.goal_box_width) / 2),  # 27
            (self.length, (self.width + self.goal_box_width) / 2),  # 28
            (self.length, (self.width + self.penalty_box_width) / 2),  # 29
            (self.length, self.width),  # 30 - Bottom right corner
            (self.length / 2 - self.centre_circle_radius, self.width / 2),  # 31 - Center circle left
            (self.length / 2 + self.centre_circle_radius, self.width / 2),  # 32 - Center circle right
        ]
    
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [
        # Left side
        (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
        # Left penalty box
        (10, 11), (11, 12), (12, 13), 
        # Center line
        (14, 15), (15, 16), (16, 17),
        # Right penalty box
        (18, 19), (19, 20), (20, 21), (23, 24),
        # Right side
        (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
        # Horizontal connections
        (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
        (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30)
    ])
    
    labels: List[str] = field(default_factory=lambda: [
        "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
        "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
        "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
        "14", "19"
    ])


# =============================================================================
# Color Definitions
# =============================================================================

class Colors:
    """Standard colors for visualization."""
    PITCH_GREEN = (34, 139, 34)  # Forest green (BGR)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    ORANGE = (0, 165, 255)
    
    # Team colors
    TEAM_A = (255, 191, 0)  # Deep sky blue (BGR)
    TEAM_B = (147, 20, 255)  # Deep pink (BGR)
    REFEREE = (0, 215, 255)  # Gold (BGR)
    BALL = (255, 255, 255)  # White
    GOALKEEPER_A = (255, 99, 71)  # Tomato (BGR)
    GOALKEEPER_B = (0, 255, 127)  # Spring green (BGR)


# =============================================================================
# Pitch Drawing Functions
# =============================================================================

def draw_pitch(
    config: SoccerPitchConfiguration = None,
    background_color: Tuple[int, int, int] = Colors.PITCH_GREEN,
    line_color: Tuple[int, int, int] = Colors.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1
) -> np.ndarray:
    """
    Draw a soccer pitch with standard markings.
    
    Args:
        config (SoccerPitchConfiguration): Pitch configuration. Uses default if None.
        background_color (Tuple[int, int, int]): Background color in BGR format.
        line_color (Tuple[int, int, int]): Line color in BGR format.
        padding (int): Padding around the pitch in pixels.
        line_thickness (int): Thickness of pitch lines in pixels.
        point_radius (int): Radius of penalty spot markers.
        scale (float): Scaling factor for pitch dimensions.
    
    Returns:
        np.ndarray: Image of the soccer pitch.
    """
    if config is None:
        config = SoccerPitchConfiguration()
    
    # Calculate scaled dimensions
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)
    
    # Create base image with background color
    pitch_image = np.ones(
        (scaled_width + 2 * padding, scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color, dtype=np.uint8)
    
    # Draw all edges (pitch lines)
    for start, end in config.edges:
        point1 = (
            int(config.vertices[start - 1][0] * scale) + padding,
            int(config.vertices[start - 1][1] * scale) + padding
        )
        point2 = (
            int(config.vertices[end - 1][0] * scale) + padding,
            int(config.vertices[end - 1][1] * scale) + padding
        )
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color,
            thickness=line_thickness
        )
    
    # Draw center circle
    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        img=pitch_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=line_color,
        thickness=line_thickness
    )
    
    # Draw penalty spots
    penalty_spots = [
        (scaled_penalty_spot_distance + padding, scaled_width // 2 + padding),
        (scaled_length - scaled_penalty_spot_distance + padding, scaled_width // 2 + padding)
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=pitch_image,
            center=spot,
            radius=point_radius,
            color=line_color,
            thickness=-1  # Filled circle
        )
    
    # Draw center spot
    cv2.circle(
        img=pitch_image,
        center=centre_circle_center,
        radius=point_radius,
        color=line_color,
        thickness=-1
    )
    
    return pitch_image


def draw_points_on_pitch(
    xy: np.ndarray,
    config: SoccerPitchConfiguration = None,
    face_color: Tuple[int, int, int] = Colors.RED,
    edge_color: Tuple[int, int, int] = Colors.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Draw points (players/ball) on a soccer pitch.
    
    Args:
        xy (np.ndarray): Array of (x, y) coordinates in pitch units (cm).
        config (SoccerPitchConfiguration): Pitch configuration.
        face_color (Tuple[int, int, int]): Fill color for points in BGR.
        edge_color (Tuple[int, int, int]): Edge color for points in BGR.
        radius (int): Radius of points in pixels.
        thickness (int): Thickness of point edges.
        padding (int): Padding around the pitch.
        scale (float): Scaling factor.
        pitch (np.ndarray): Existing pitch image to draw on. Creates new if None.
    
    Returns:
        np.ndarray: Pitch image with points drawn.
    """
    if config is None:
        config = SoccerPitchConfiguration()
    
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)
    else:
        pitch = pitch.copy()
    
    for point in xy:
        if np.isnan(point).any():
            continue
        
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        
        # Draw filled circle (face)
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=face_color,
            thickness=-1
        )
        
        # Draw edge
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=edge_color,
            thickness=thickness
        )
    
    return pitch


def draw_radar(
    player_positions: np.ndarray,
    team_ids: np.ndarray,
    ball_position: Optional[np.ndarray] = None,
    referee_positions: Optional[np.ndarray] = None,
    config: SoccerPitchConfiguration = None,
    scale: float = 0.1,
    padding: int = 50,
    player_radius: int = 16,
    ball_radius: int = 10
) -> np.ndarray:
    """
    Draw a complete radar/minimap with all players and ball.
    
    Args:
        player_positions (np.ndarray): Player positions in pitch coordinates (N, 2).
        team_ids (np.ndarray): Team ID for each player (0 or 1).
        ball_position (np.ndarray): Ball position in pitch coordinates (2,).
        referee_positions (np.ndarray): Referee positions (M, 2).
        config (SoccerPitchConfiguration): Pitch configuration.
        scale (float): Scaling factor.
        padding (int): Padding around pitch.
        player_radius (int): Radius for player markers.
        ball_radius (int): Radius for ball marker.
    
    Returns:
        np.ndarray: Complete radar image.
    """
    if config is None:
        config = SoccerPitchConfiguration()
    
    # Draw base pitch
    radar = draw_pitch(config=config, padding=padding, scale=scale)
    
    # Draw ball if available
    if ball_position is not None and len(ball_position) > 0:
        radar = draw_points_on_pitch(
            xy=ball_position.reshape(-1, 2),
            config=config,
            face_color=Colors.BALL,
            edge_color=Colors.BLACK,
            radius=ball_radius,
            padding=padding,
            scale=scale,
            pitch=radar
        )
    
    # Draw referees if available
    if referee_positions is not None and len(referee_positions) > 0:
        radar = draw_points_on_pitch(
            xy=referee_positions,
            config=config,
            face_color=Colors.REFEREE,
            edge_color=Colors.BLACK,
            radius=player_radius,
            padding=padding,
            scale=scale,
            pitch=radar
        )
    
    # Draw Team A players
    team_a_mask = team_ids == 0
    if np.any(team_a_mask):
        radar = draw_points_on_pitch(
            xy=player_positions[team_a_mask],
            config=config,
            face_color=Colors.TEAM_A,
            edge_color=Colors.BLACK,
            radius=player_radius,
            padding=padding,
            scale=scale,
            pitch=radar
        )
    
    # Draw Team B players
    team_b_mask = team_ids == 1
    if np.any(team_b_mask):
        radar = draw_points_on_pitch(
            xy=player_positions[team_b_mask],
            config=config,
            face_color=Colors.TEAM_B,
            edge_color=Colors.BLACK,
            radius=player_radius,
            padding=padding,
            scale=scale,
            pitch=radar
        )
    
    return radar


# =============================================================================
# Frame Annotation Functions
# =============================================================================

class FrameAnnotator:
    """
    Annotator for drawing detections on video frames.
    
    Uses supervision library annotators with predefined styling.
    """
    
    def __init__(
        self,
        team_colors: List[Tuple[int, int, int]] = None,
        box_thickness: int = 2,
        text_thickness: int = 1,
        text_scale: float = 0.5
    ):
        """
        Initialize the frame annotator.
        
        Args:
            team_colors (List): Colors for [Team A, Team B, Referee, Ball].
            box_thickness (int): Thickness of bounding boxes.
            text_thickness (int): Thickness of label text.
            text_scale (float): Scale of label text.
        """
        if team_colors is None:
            team_colors = [
                Colors.TEAM_A,
                Colors.TEAM_B,
                Colors.REFEREE,
                Colors.BALL
            ]
        
        # Convert BGR tuples to sv.Color objects
        sv_colors = [sv.Color(*c[::-1]) for c in team_colors]  # BGR to RGB
        
        self.color_palette = sv.ColorPalette(colors=sv_colors)
        
        self.box_annotator = sv.BoxAnnotator(
            color=self.color_palette,
            thickness=box_thickness
        )
        
        self.label_annotator = sv.LabelAnnotator(
            color=self.color_palette,
            text_color=sv.Color.WHITE,
            text_thickness=text_thickness,
            text_scale=text_scale,
            text_padding=5
        )
        
        self.ellipse_annotator = sv.EllipseAnnotator(
            color=self.color_palette,
            thickness=box_thickness
        )
    
    def annotate(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        labels: Optional[List[str]] = None,
        color_lookup: Optional[np.ndarray] = None,
        use_ellipse: bool = False
    ) -> np.ndarray:
        """
        Annotate a frame with detections.
        
        Args:
            frame (np.ndarray): Input frame.
            detections (sv.Detections): Detected objects.
            labels (List[str]): Labels for each detection.
            color_lookup (np.ndarray): Custom color indices for each detection.
            use_ellipse (bool): Use ellipse instead of box annotation.
        
        Returns:
            np.ndarray: Annotated frame.
        """
        annotated = frame.copy()
        
        if use_ellipse:
            annotated = self.ellipse_annotator.annotate(
                scene=annotated,
                detections=detections,
                custom_color_lookup=color_lookup
            )
        else:
            annotated = self.box_annotator.annotate(
                scene=annotated,
                detections=detections,
                custom_color_lookup=color_lookup
            )
        
        if labels is not None:
            annotated = self.label_annotator.annotate(
                scene=annotated,
                detections=detections,
                labels=labels,
                custom_color_lookup=color_lookup
            )
        
        return annotated


def combine_views(
    camera_frame: np.ndarray,
    radar_frame: np.ndarray,
    layout: str = "side_by_side",
    radar_scale: float = 0.4,
    radar_opacity: float = 0.8,
    radar_position: str = "bottom_right"
) -> np.ndarray:
    """
    Combine camera view with radar/minimap.
    
    Args:
        camera_frame (np.ndarray): Main camera view.
        radar_frame (np.ndarray): Radar/minimap view.
        layout (str): "side_by_side" or "overlay".
        radar_scale (float): Scale factor for radar in overlay mode.
        radar_opacity (float): Opacity for radar overlay.
        radar_position (str): Position for overlay ("bottom_right", "bottom_left", 
                              "top_right", "top_left").
    
    Returns:
        np.ndarray: Combined visualization.
    """
    if layout == "side_by_side":
        # Resize radar to match camera height
        cam_h, cam_w = camera_frame.shape[:2]
        radar_h, radar_w = radar_frame.shape[:2]
        
        # Scale radar to match camera height
        new_radar_h = cam_h
        new_radar_w = int(radar_w * (cam_h / radar_h))
        radar_resized = cv2.resize(radar_frame, (new_radar_w, new_radar_h))
        
        # Combine horizontally
        combined = np.hstack([camera_frame, radar_resized])
        
    elif layout == "overlay":
        combined = camera_frame.copy()
        cam_h, cam_w = camera_frame.shape[:2]
        
        # Resize radar
        radar_h, radar_w = radar_frame.shape[:2]
        new_radar_w = int(cam_w * radar_scale)
        new_radar_h = int(radar_h * (new_radar_w / radar_w))
        radar_resized = cv2.resize(radar_frame, (new_radar_w, new_radar_h))
        
        # Determine position
        if radar_position == "bottom_right":
            x_offset = cam_w - new_radar_w - 10
            y_offset = cam_h - new_radar_h - 10
        elif radar_position == "bottom_left":
            x_offset = 10
            y_offset = cam_h - new_radar_h - 10
        elif radar_position == "top_right":
            x_offset = cam_w - new_radar_w - 10
            y_offset = 10
        else:  # top_left
            x_offset = 10
            y_offset = 10
        
        # Ensure radar fits within frame bounds
        if x_offset + new_radar_w > cam_w or y_offset + new_radar_h > cam_h:
            raise ValueError(
                f"Radar overlay ({new_radar_w}x{new_radar_h}) at position "
                f"({x_offset}, {y_offset}) exceeds camera frame dimensions ({cam_w}x{cam_h}). "
                f"Reduce radar_scale or adjust radar_position."
            )
        
        # Create overlay region
        roi = combined[y_offset:y_offset + new_radar_h, x_offset:x_offset + new_radar_w]
        
        # Blend radar with ROI
        blended = cv2.addWeighted(roi, 1 - radar_opacity, radar_resized, radar_opacity, 0)
        combined[y_offset:y_offset + new_radar_h, x_offset:x_offset + new_radar_w] = blended
        
    else:
        raise ValueError(f"Unknown layout: {layout}. Use 'side_by_side' or 'overlay'.")
    
    return combined


def draw_fps(
    frame: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = Colors.WHITE,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw FPS counter on frame.
    
    Args:
        frame (np.ndarray): Input frame.
        fps (float): Frames per second value.
        position (Tuple[int, int]): Position for FPS text.
        font_scale (float): Font scale.
        color (Tuple[int, int, int]): Text color in BGR.
        thickness (int): Text thickness.
    
    Returns:
        np.ndarray: Frame with FPS overlay.
    """
    annotated = frame.copy()
    text = f"FPS: {fps:.1f}"
    
    # Draw background rectangle
    (text_w, text_h), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(
        annotated,
        (position[0] - 5, position[1] - text_h - 5),
        (position[0] + text_w + 5, position[1] + 5),
        (0, 0, 0),
        -1
    )
    
    # Draw text
    cv2.putText(
        annotated,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness
    )
    
    return annotated
