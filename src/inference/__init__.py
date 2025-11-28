# Inference Layer
"""
Core AI Logic: Detection, Tracking, Team Classification, and Pipeline Orchestration.

Phase 3 Components:
- detector.py: YOLO11 + Roboflow Pitch Model wrappers
- tracker.py: ByteTrack + Ball Anti-Teleport smoothing
- team.py: KMeans-based jersey color team classification

Phase 4 (Coming):
- pipeline.py: Main orchestration loop
"""

# Detector classes
from .detector import (
    ObjectDetector,
    PitchDetector,
    CombinedDetector,
    get_crops,
    filter_by_class,
    # Constants
    CLASS_NAMES,
    CLASS_IDS,
    BALL_CLASS_ID,
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    REFEREE_CLASS_ID,
)

# Tracker classes
from .tracker import (
    BallTracker,
    PlayerTracker,
    resolve_goalkeepers_team_id,
)

# Team classification
from .team import (
    TeamClassifier,
    TeamClassifierAdvanced,
)

# Pipeline (Phase 4 - placeholder)
# from .pipeline import SportsPipeline

__all__ = [
    # Detectors
    "ObjectDetector",
    "PitchDetector",
    "CombinedDetector",
    "get_crops",
    "filter_by_class",
    # Constants
    "CLASS_NAMES",
    "CLASS_IDS",
    "BALL_CLASS_ID",
    "GOALKEEPER_CLASS_ID",
    "PLAYER_CLASS_ID",
    "REFEREE_CLASS_ID",
    # Trackers
    "BallTracker",
    "PlayerTracker",
    "resolve_goalkeepers_team_id",
    # Team
    "TeamClassifier",
    "TeamClassifierAdvanced",
    # Pipeline (Phase 4)
    # "SportsPipeline",
]
