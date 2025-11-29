# Utilities Layer
"""
Geometry (Homography/ViewTransformer) and Visualization utilities.
"""

from .geometry import (
    ViewTransformer,
    get_foot_position,
    compute_distance,
    compute_distances_batch
)

from .viz import (
    # Configuration
    SoccerPitchConfiguration,
    Colors,
    
    # Pitch Drawing
    draw_pitch,
    draw_points_on_pitch,
    draw_radar,
    
    # Frame Annotation
    FrameAnnotator,
    combine_views,
    draw_fps
)

__all__ = [
    # Geometry
    "ViewTransformer",
    "get_foot_position",
    "compute_distance",
    "compute_distances_batch",
    
    # Visualization
    "SoccerPitchConfiguration",
    "Colors",
    "draw_pitch",
    "draw_points_on_pitch",
    "draw_radar",
    "FrameAnnotator",
    "combine_views",
    "draw_fps"
]
