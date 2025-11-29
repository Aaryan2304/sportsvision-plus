# 📋 SportsVision+ Phase 4: Assembly & Launch - Revised Plan

**Created:** November 29, 2025  
**Branch:** `feature/Phase-4`  
**Goal:** Complete the real-time football analytics system with desktop application

---

## 🎯 Phase 4 Overview

| Component | Description | Priority |
|-----------|-------------|----------|
| **Training Optimizations** | GPU acceleration, quantization, export | High |
| **Stats Engine** | Real-time statistics computation | High |
| **Core Pipeline** | Main processing loop | High |
| **Desktop App** | PyQt6 transparent overlay application | High |

---

## 1️⃣ Training Optimizations

### Purpose
Optimize YOLO model training and inference for RTX 3050/3060/4050 laptops.

### File: `scripts/train.py`

### Techniques to Implement

#### A. CUDA Acceleration
```python
import torch

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Train with GPU
model = YOLO("yolo11n.pt")
model.train(
    data="data/processed/dataset.yaml",
    epochs=100,
    imgsz=1280,
    device=0,  # GPU index
    workers=8,
    cache=True  # Cache images in RAM/VRAM
)
```

#### B. Mixed Precision Training (FP16)
```python
# Enable Automatic Mixed Precision (AMP)
# Reduces VRAM usage by ~50%, speeds up training by ~2x

model.train(
    data="data/processed/dataset.yaml",
    epochs=100,
    amp=True,  # Enable FP16 mixed precision
    half=True  # Use FP16 for inference
)
```

#### C. ONNX Export
```python
# Export for cross-platform deployment
model.export(
    format="onnx",
    imgsz=1280,
    simplify=True,
    dynamic=True,  # Dynamic batch size
    opset=17
)
# Output: runs/detect/train/weights/best.onnx
```

#### D. TensorRT Export (NVIDIA Optimized)
```python
# Maximum inference speed on NVIDIA GPUs
model.export(
    format="engine",  # TensorRT
    imgsz=1280,
    half=True,  # FP16 TensorRT
    device=0
)
# Output: runs/detect/train/weights/best.engine
```

#### E. Model Quantization (INT8)
```python
# Smallest model size, fastest inference (slight accuracy loss)
model.export(
    format="onnx",
    imgsz=1280,
    int8=True,  # INT8 quantization
    data="data/processed/dataset.yaml"  # Calibration data
)
```

#### F. Pruning (Optional - Advanced)
```python
# Remove redundant weights for smaller model
# Requires fine-tuning after pruning
from ultralytics import YOLO

model = YOLO("best.pt")
# Pruning is done during training with sparsity parameter
model.train(
    data="data/processed/dataset.yaml",
    epochs=50,
    # Sparsity training requires custom implementation
)
```

### Training Script Template
```python
# scripts/train.py
"""
SportsVision+ Model Training Script
Optimized for RTX 3050/3060/4050 laptops
"""

from ultralytics import YOLO
import torch
import os

def train_model(
    data_yaml: str = "data/processed/dataset.yaml",
    model_size: str = "yolo11n.pt",  # n=nano, s=small, m=medium
    epochs: int = 100,
    imgsz: int = 1280,
    batch_size: int = -1,  # Auto-batch
    use_amp: bool = True,
    export_formats: list = ["onnx", "engine"]
):
    """Train YOLO model with optimizations."""
    
    # Check GPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU required for training.")
    
    print(f"🚀 Training on: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    model = YOLO(model_size)
    
    # Train with optimizations
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=0,
        workers=8,
        cache=True,
        amp=use_amp,
        half=True,
        patience=20,  # Early stopping
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,
        verbose=True
    )
    
    # Export optimized models
    best_model = YOLO(results.save_dir / "weights/best.pt")
    
    for fmt in export_formats:
        print(f"📦 Exporting to {fmt}...")
        if fmt == "engine":
            best_model.export(format=fmt, imgsz=imgsz, half=True, device=0)
        elif fmt == "onnx":
            best_model.export(format=fmt, imgsz=imgsz, simplify=True)
    
    print("✅ Training complete!")
    return results

if __name__ == "__main__":
    train_model()
```

---

## 2️⃣ Stats Engine

### Purpose
Compute real-time match statistics from detection and tracking data.

### File: `src/inference/stats.py`

### Classes to Implement

#### A. PossessionTracker
```python
class PossessionTracker:
    """
    Track ball possession percentage for each team.
    
    Logic:
    - Each frame, find which team's player is closest to the ball
    - Accumulate possession time for that team
    - Calculate percentage over rolling window or full match
    """
    
    def __init__(self, smoothing_window: int = 30):
        self.team_a_frames = 0
        self.team_b_frames = 0
        self.total_frames = 0
        self.smoothing_window = smoothing_window
        self.history = deque(maxlen=smoothing_window)
    
    def update(
        self,
        ball_xy: np.ndarray,           # Ball position on pitch
        team_a_xy: np.ndarray,         # Team A player positions
        team_b_xy: np.ndarray          # Team B player positions
    ) -> Tuple[float, float]:
        """
        Update possession based on current frame.
        
        Returns:
            Tuple[float, float]: (team_a_possession%, team_b_possession%)
        """
        # Find closest player to ball
        if len(ball_xy) == 0:
            return self.get_possession()
        
        ball = ball_xy[0]
        
        # Distances to each team
        dist_a = np.min(np.linalg.norm(team_a_xy - ball, axis=1)) if len(team_a_xy) > 0 else float('inf')
        dist_b = np.min(np.linalg.norm(team_b_xy - ball, axis=1)) if len(team_b_xy) > 0 else float('inf')
        
        # Assign possession
        if dist_a < dist_b:
            self.team_a_frames += 1
            self.history.append(0)
        else:
            self.team_b_frames += 1
            self.history.append(1)
        
        self.total_frames += 1
        return self.get_possession()
    
    def get_possession(self) -> Tuple[float, float]:
        """Get current possession percentages."""
        if self.total_frames == 0:
            return (50.0, 50.0)
        
        pct_a = (self.team_a_frames / self.total_frames) * 100
        pct_b = (self.team_b_frames / self.total_frames) * 100
        return (round(pct_a, 1), round(pct_b, 1))
```

#### B. AttackCounter
```python
class AttackCounter:
    """
    Count attacks (ball entering attacking third of pitch).
    
    Logic:
    - Divide pitch into 3 horizontal zones
    - When ball crosses into opponent's third, count as attack
    - Track direction of play for each team
    """
    
    def __init__(self, pitch_length: int = 12000):  # cm
        self.pitch_length = pitch_length
        self.third = pitch_length / 3
        self.team_a_attacks = 0
        self.team_b_attacks = 0
        self.last_zone = None
        self.ball_owner = None
    
    def update(
        self,
        ball_xy: np.ndarray,
        ball_owner_team: int  # 0 or 1
    ) -> Tuple[int, int]:
        """
        Update attack count based on ball position.
        
        Returns:
            Tuple[int, int]: (team_a_attacks, team_b_attacks)
        """
        if len(ball_xy) == 0:
            return (self.team_a_attacks, self.team_b_attacks)
        
        ball_x = ball_xy[0][0]  # X position on pitch (length axis)
        
        # Determine zone (0=left third, 1=middle, 2=right third)
        if ball_x < self.third:
            zone = 0
        elif ball_x < 2 * self.third:
            zone = 1
        else:
            zone = 2
        
        # Check for new attack
        if self.last_zone is not None and zone != self.last_zone:
            # Team A attacks right side (zone 2)
            if ball_owner_team == 0 and zone == 2 and self.last_zone != 2:
                self.team_a_attacks += 1
            # Team B attacks left side (zone 0)
            elif ball_owner_team == 1 and zone == 0 and self.last_zone != 0:
                self.team_b_attacks += 1
        
        self.last_zone = zone
        return (self.team_a_attacks, self.team_b_attacks)
```

#### C. ShotDetector
```python
class ShotDetector:
    """
    Detect shots on goal based on ball velocity and position.
    
    Logic:
    - Track ball velocity (difference between frames)
    - If ball is in attacking third AND moving fast toward goal → shot
    - If trajectory intersects goal box → shot on target
    """
    
    def __init__(
        self,
        pitch_length: int = 12000,
        pitch_width: int = 7000,
        goal_width: int = 732,  # 7.32m standard goal
        velocity_threshold: float = 500  # cm/frame threshold for "fast"
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.goal_width = goal_width
        self.velocity_threshold = velocity_threshold
        
        self.team_a_shots = 0
        self.team_b_shots = 0
        self.team_a_on_target = 0
        self.team_b_on_target = 0
        
        self.last_ball_xy = None
        self.cooldown = 0  # Prevent multiple counts for same shot
    
    def update(
        self,
        ball_xy: np.ndarray,
        ball_owner_team: int
    ) -> Dict[str, int]:
        """
        Detect if current frame contains a shot.
        
        Returns:
            Dict with shots and shots_on_target for each team
        """
        if self.cooldown > 0:
            self.cooldown -= 1
        
        if len(ball_xy) == 0 or self.last_ball_xy is None:
            self.last_ball_xy = ball_xy[0] if len(ball_xy) > 0 else None
            return self.get_stats()
        
        current = ball_xy[0]
        velocity = current - self.last_ball_xy
        speed = np.linalg.norm(velocity)
        
        # Check if ball is moving fast toward a goal
        if speed > self.velocity_threshold and self.cooldown == 0:
            ball_x, ball_y = current
            goal_y_center = self.pitch_width / 2
            goal_y_min = goal_y_center - self.goal_width / 2
            goal_y_max = goal_y_center + self.goal_width / 2
            
            # Team A shooting at right goal (x = pitch_length)
            if ball_owner_team == 0 and ball_x > self.pitch_length * 0.66 and velocity[0] > 0:
                self.team_a_shots += 1
                if goal_y_min < ball_y < goal_y_max:
                    self.team_a_on_target += 1
                self.cooldown = 30  # 1 second at 30fps
            
            # Team B shooting at left goal (x = 0)
            elif ball_owner_team == 1 and ball_x < self.pitch_length * 0.33 and velocity[0] < 0:
                self.team_b_shots += 1
                if goal_y_min < ball_y < goal_y_max:
                    self.team_b_on_target += 1
                self.cooldown = 30
        
        self.last_ball_xy = current
        return self.get_stats()
    
    def get_stats(self) -> Dict[str, int]:
        return {
            "team_a_shots": self.team_a_shots,
            "team_b_shots": self.team_b_shots,
            "team_a_on_target": self.team_a_on_target,
            "team_b_on_target": self.team_b_on_target
        }
```

#### D. FormationDetector
```python
class FormationDetector:
    """
    Detect team formation from player positions.
    
    Logic:
    - Exclude goalkeeper (furthest back player)
    - Cluster remaining players by Y position (depth on pitch)
    - Count players in each cluster line
    - Return formation string (e.g., "4-3-3", "4-4-2")
    
    Visualization:
    - Display formation string (e.g., "4-3-3" vs "4-4-2")
    - Draw lines connecting players in the same formation line
    - Lines help visualize team shape and gaps
    """
    
    def __init__(self, min_line_gap: float = 500):  # cm between formation lines
        self.min_line_gap = min_line_gap
        self.team_a_formation = "---"
        self.team_b_formation = "---"
        self.history_a = deque(maxlen=30)  # Smooth over 1 second
        self.history_b = deque(maxlen=30)
        
        # Store line assignments for visualization
        self.team_a_lines = []  # List of lists: [[player1_xy, player2_xy], [player3_xy, ...]]
        self.team_b_lines = []
    
    def _detect_formation(self, player_xy: np.ndarray, attacking_right: bool) -> Tuple[str, List[List[np.ndarray]]]:
        """
        Detect formation for a single team.
        
        Args:
            player_xy: Player positions (N, 2)
            attacking_right: True if team attacks right side
        
        Returns:
            Tuple[str, List, bool]: (formation_string, lines_of_players, is_partial)
            - formation_string: e.g., "4-3-3" or "Partial (3)" for partial views
            - lines_of_players: List of player groups per formation line
            - is_partial: True if fewer than 7 players visible
        """
        # Handle partial visibility (broadcast showing only part of pitch)
        if len(player_xy) < 3:
            return ("---", [], True)
        
        is_partial = len(player_xy) < 7
        
        # Sort by X position (depth)
        if attacking_right:
            sort_indices = np.argsort(player_xy[:, 0])
        else:
            sort_indices = np.argsort(-player_xy[:, 0])
        
        sorted_xy = player_xy[sort_indices]
        
        # For partial views, don't exclude goalkeeper (we don't know who's GK)
        if is_partial:
            outfield = sorted_xy  # Include all visible players
        else:
            # Exclude goalkeeper (first player after sorting)
            outfield = sorted_xy[1:]
        
        if len(outfield) < 2:
            return ("---", [], True)
        
        # Cluster by X position into lines
        x_positions = outfield[:, 0]
        line_counts = []
        line_players = []  # Store actual player positions per line
        current_line_indices = [0]
        
        for i, x in enumerate(x_positions[1:], start=1):
            mean_x = np.mean(x_positions[np.array(current_line_indices)])
            if abs(x - mean_x) < self.min_line_gap:
                current_line_indices.append(i)
            else:
                line_counts.append(len(current_line_indices))
                line_players.append(outfield[current_line_indices].tolist())
                current_line_indices = [i]
        
        line_counts.append(len(current_line_indices))
        line_players.append(outfield[current_line_indices].tolist())
        
        # Format as string (defenders first)
        if not attacking_right:
            line_counts = line_counts[::-1]
            line_players = line_players[::-1]
        
        if is_partial:
            # For partial views, show "Partial (N visible)" instead of formation
            formation_str = f"Partial ({len(player_xy)})"
        else:
            formation_str = "-".join(str(n) for n in line_counts)
        
        return (formation_str, line_players, is_partial)
    
    def update(
        self,
        team_a_xy: np.ndarray,
        team_b_xy: np.ndarray
    ) -> Tuple[str, str]:
        """
        Update formation detection.
        
        Returns:
            Tuple[str, str]: (team_a_formation, team_b_formation)
        """
        # Team A attacks right, Team B attacks left
        formation_a, lines_a, partial_a = self._detect_formation(team_a_xy, attacking_right=True)
        formation_b, lines_b, partial_b = self._detect_formation(team_b_xy, attacking_right=False)
        
        # Always store lines for visualization (even partial views)
        self.team_a_lines = lines_a
        self.team_b_lines = lines_b
        self.team_a_partial = partial_a
        self.team_b_partial = partial_b
        
        # Only add to history if full formation detected (for smoothing)
        if not partial_a:
            self.history_a.append(formation_a)
        if not partial_b:
            self.history_b.append(formation_b)
        
        # Return most common formation in history (smoothing)
        # For partial views, return the partial string directly
        if partial_a or len(self.history_a) == 0:
            self.team_a_formation = formation_a
        else:
            self.team_a_formation = max(set(self.history_a), key=list(self.history_a).count)
        
        if partial_b or len(self.history_b) == 0:
            self.team_b_formation = formation_b
        else:
            self.team_b_formation = max(set(self.history_b), key=list(self.history_b).count)
        
        return (self.team_a_formation, self.team_b_formation)
    
    def get_formation_lines(self) -> Dict[str, List[List[Tuple[float, float]]]]:
        """
        Get player positions grouped by formation line for visualization.
        
        Note: Lines are ALWAYS returned when players are visible, even in partial views.
        This allows the UI to draw connecting lines between visible players.
        
        Returns:
            Dict with 'team_a' and 'team_b' lists of formation lines.
            Each line is a list of (x, y) player positions to connect.
        """
        return {
            "team_a": self.team_a_lines,
            "team_b": self.team_b_lines,
            "team_a_partial": getattr(self, 'team_a_partial', False),
            "team_b_partial": getattr(self, 'team_b_partial', False)
        }
```

#### E. HeatmapAccumulator
```python
class HeatmapAccumulator:
    """
    Accumulate player positions into a heatmap grid.
    
    Logic:
    - Divide pitch into grid (e.g., 12x8 cells)
    - Each frame, increment cell counts where players are
    - Normalize for visualization
    
    Visualization Options:
    - Team Heatmap (default): Show all players of a team
    - Individual Player Heatmap: Select player by tracker ID from dropdown
    - Color scheme: Blue (low) → Green → Yellow → Red (high)
    """
    
    def __init__(
        self,
        pitch_length: int = 12000,
        pitch_width: int = 7000,
        grid_cols: int = 12,
        grid_rows: int = 8
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        
        self.cell_width = pitch_length / grid_cols
        self.cell_height = pitch_width / grid_rows
        
        # Team heatmaps
        self.team_a_heatmap = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        self.team_b_heatmap = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        self.ball_heatmap = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        
        # Individual player heatmaps (keyed by tracker_id)
        self.player_heatmaps: Dict[int, np.ndarray] = {}
        self.player_team_mapping: Dict[int, str] = {}  # tracker_id -> "a" or "b"
    
    def update(
        self,
        team_a_xy: np.ndarray,
        team_b_xy: np.ndarray,
        team_a_ids: np.ndarray = None,
        team_b_ids: np.ndarray = None,
        ball_xy: np.ndarray = None
    ) -> None:
        """
        Update heatmaps with current positions.
        
        Args:
            team_a_xy: Team A player positions (N, 2)
            team_b_xy: Team B player positions (M, 2)
            team_a_ids: Tracker IDs for Team A players (for individual heatmaps)
            team_b_ids: Tracker IDs for Team B players (for individual heatmaps)
            ball_xy: Ball position (1, 2)
        """
        self._add_to_heatmap(team_a_xy, self.team_a_heatmap)
        self._add_to_heatmap(team_b_xy, self.team_b_heatmap)
        
        # Update individual player heatmaps
        if team_a_ids is not None:
            for pos, pid in zip(team_a_xy, team_a_ids):
                self._add_player_to_heatmap(pid, pos, "a")
        
        if team_b_ids is not None:
            for pos, pid in zip(team_b_xy, team_b_ids):
                self._add_player_to_heatmap(pid, pos, "b")
        
        if ball_xy is not None and len(ball_xy) > 0:
            self._add_to_heatmap(ball_xy, self.ball_heatmap)
    
    def _add_player_to_heatmap(self, tracker_id: int, pos: np.ndarray, team: str) -> None:
        """Add single player position to their individual heatmap."""
        if tracker_id not in self.player_heatmaps:
            self.player_heatmaps[tracker_id] = np.zeros(
                (self.grid_rows, self.grid_cols), dtype=np.float32
            )
            self.player_team_mapping[tracker_id] = team
        
        x, y = pos
        col = int(np.clip(x / self.cell_width, 0, self.grid_cols - 1))
        row = int(np.clip(y / self.cell_height, 0, self.grid_rows - 1))
        self.player_heatmaps[tracker_id][row, col] += 1
    
    def _add_to_heatmap(self, positions: np.ndarray, heatmap: np.ndarray) -> None:
        """Add positions to a heatmap grid."""
        for pos in positions:
            x, y = pos
            col = int(np.clip(x / self.cell_width, 0, self.grid_cols - 1))
            row = int(np.clip(y / self.cell_height, 0, self.grid_rows - 1))
            heatmap[row, col] += 1
    
    def get_normalized_heatmap(self, team: str = "a") -> np.ndarray:
        """Get normalized team heatmap (0-255 for visualization)."""
        if team == "a":
            hm = self.team_a_heatmap
        elif team == "b":
            hm = self.team_b_heatmap
        else:
            hm = self.ball_heatmap
        
        if hm.max() > 0:
            return (hm / hm.max() * 255).astype(np.uint8)
        return np.zeros_like(hm, dtype=np.uint8)
    
    def get_player_heatmap(self, tracker_id: int) -> Optional[np.ndarray]:
        """Get normalized heatmap for a specific player."""
        if tracker_id not in self.player_heatmaps:
            return None
        
        hm = self.player_heatmaps[tracker_id]
        if hm.max() > 0:
            return (hm / hm.max() * 255).astype(np.uint8)
        return np.zeros_like(hm, dtype=np.uint8)
    
    def get_available_players(self) -> List[Dict]:
        """
        Get list of tracked players for dropdown selection.
        
        Returns:
            List of dicts with 'id' and 'team' keys
        """
        return [
            {"id": pid, "team": team}
            for pid, team in self.player_team_mapping.items()
        ]
    
    def reset(self) -> None:
        """Reset all heatmaps."""
        self.team_a_heatmap.fill(0)
        self.team_b_heatmap.fill(0)
        self.ball_heatmap.fill(0)
        self.player_heatmaps.clear()
        self.player_team_mapping.clear()
```

#### F. StatsAggregator
```python
class StatsAggregator:
    """
    Central aggregator for all statistics.
    
    Combines all stat trackers and provides unified interface.
    """
    
    def __init__(self):
        self.possession = PossessionTracker()
        self.attacks = AttackCounter()
        self.shots = ShotDetector()
        self.formation = FormationDetector()
        self.heatmap = HeatmapAccumulator()
        
        self.frame_count = 0
        self.start_time = None
    
    def update(
        self,
        team_a_pitch_xy: np.ndarray,
        team_b_pitch_xy: np.ndarray,
        ball_pitch_xy: np.ndarray,
        ball_owner_team: int = None
    ) -> Dict:
        """
        Update all statistics with current frame data.
        
        All coordinates should be in pitch coordinates (cm).
        
        Returns:
            Dict with all current statistics
        """
        import time
        
        if self.start_time is None:
            self.start_time = time.time()
        
        self.frame_count += 1
        
        # Determine ball owner if not provided
        if ball_owner_team is None and len(ball_pitch_xy) > 0:
            ball = ball_pitch_xy[0]
            dist_a = np.min(np.linalg.norm(team_a_pitch_xy - ball, axis=1)) if len(team_a_pitch_xy) > 0 else float('inf')
            dist_b = np.min(np.linalg.norm(team_b_pitch_xy - ball, axis=1)) if len(team_b_pitch_xy) > 0 else float('inf')
            ball_owner_team = 0 if dist_a < dist_b else 1
        
        # Update all trackers
        possession = self.possession.update(ball_pitch_xy, team_a_pitch_xy, team_b_pitch_xy)
        attacks = self.attacks.update(ball_pitch_xy, ball_owner_team)
        shot_stats = self.shots.update(ball_pitch_xy, ball_owner_team)
        formations = self.formation.update(team_a_pitch_xy, team_b_pitch_xy)
        self.heatmap.update(team_a_pitch_xy, team_b_pitch_xy, ball_pitch_xy)
        
        return {
            "frame": self.frame_count,
            "elapsed_time": time.time() - self.start_time,
            "possession": {
                "team_a": possession[0],
                "team_b": possession[1]
            },
            "attacks": {
                "team_a": attacks[0],
                "team_b": attacks[1]
            },
            "shots": shot_stats,
            "formation": {
                "team_a": formations[0],
                "team_b": formations[1]
            }
        }
    
    def get_summary(self) -> Dict:
        """Get full statistics summary."""
        return {
            "possession": self.possession.get_possession(),
            "attacks": (self.attacks.team_a_attacks, self.attacks.team_b_attacks),
            "shots": self.shots.get_stats(),
            "formation": (self.formation.team_a_formation, self.formation.team_b_formation),
            "heatmaps": {
                "team_a": self.heatmap.get_normalized_heatmap("a"),
                "team_b": self.heatmap.get_normalized_heatmap("b"),
                "ball": self.heatmap.get_normalized_heatmap("ball")
            }
        }
```

---

## 3️⃣ Core Pipeline

### Purpose
Main processing loop that connects all components.

### File: `src/inference/pipeline.py`

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    SportsPipeline                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Screen   │ → │ Detector │ → │ Tracker  │ → │  Team    │ │
│  │ Capture  │   │  (YOLO)  │   │(ByteTrack)│   │Classifier│ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│       │                                             │       │
│       ▼                                             ▼       │
│  ┌──────────┐                              ┌──────────────┐ │
│  │  Frame   │                              │    View      │ │
│  │ Buffer   │                              │ Transformer  │ │
│  └──────────┘                              └──────────────┘ │
│       │                                             │       │
│       ▼                                             ▼       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                  Stats Aggregator                     │  │
│  │  (Possession, Attacks, Shots, Formation, Heatmap)    │  │
│  └──────────────────────────────────────────────────────┘  │
│                              │                              │
│                              ▼                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    Output Queue                       │  │
│  │  → Annotated Frame + Radar + Stats → UI              │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Class Template
```python
class SportsPipeline:
    """
    Main processing pipeline for SportsVision+.
    
    Connects:
    - Screen capture → Detection → Tracking → Team classification
    - View transformation → Stats computation → Visualization
    """
    
    def __init__(
        self,
        detector: ObjectDetector,
        team_classifier: TeamClassifier,
        config: SoccerPitchConfiguration = None
    ):
        self.detector = detector
        self.tracker = PlayerTracker()
        self.ball_tracker = BallTracker()
        self.team_classifier = team_classifier
        self.config = config or SoccerPitchConfiguration()
        
        self.stats = StatsAggregator()
        self.annotator = FrameAnnotator()
        self.view_transformer = None  # Set when pitch is detected
        
        self.is_running = False
        self.output_queue = Queue(maxsize=2)  # For UI consumption
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame through the full pipeline.
        
        Returns:
            Dict containing:
            - annotated_frame: Frame with player/ball annotations
            - radar: 2D minimap visualization
            - stats: Current statistics
        """
        # 1. Detection
        all_detections, ball_detections = self.detector.detect_and_filter(frame)
        
        # 2. Tracking
        all_detections = self.tracker.update(all_detections)
        ball_detections = self.ball_tracker.update(ball_detections)
        
        # 3. Separate by class
        players = all_detections[all_detections.class_id == PLAYER_CLASS_ID]
        goalkeepers = all_detections[all_detections.class_id == GOALKEEPER_CLASS_ID]
        referees = all_detections[all_detections.class_id == REFEREE_CLASS_ID]
        
        # 4. Team classification
        if len(players) > 0:
            crops = get_crops(frame, players)
            team_ids = self.team_classifier.predict(crops)
            players.class_id = team_ids
        
        # 5. Resolve goalkeeper teams
        if len(goalkeepers) > 0 and len(players) > 0:
            gk_team_ids = resolve_goalkeepers_team_id(players, team_ids, goalkeepers)
            goalkeepers.class_id = gk_team_ids
        
        # 6. Build color lookup for visualization
        color_lookup = np.concatenate([
            players.class_id if len(players) > 0 else np.array([]),
            goalkeepers.class_id if len(goalkeepers) > 0 else np.array([]),
            np.full(len(referees), 2)  # Referee color index
        ]).astype(int)
        
        merged_detections = sv.Detections.merge([players, goalkeepers, referees])
        
        # 7. View transformation (if pitch detected)
        stats_data = None
        if self.view_transformer is not None:
            # Transform to pitch coordinates
            team_a_mask = color_lookup == 0
            team_b_mask = color_lookup == 1
            
            frame_xy = merged_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            pitch_xy = self.view_transformer.transform_points(frame_xy)
            
            team_a_pitch = pitch_xy[team_a_mask]
            team_b_pitch = pitch_xy[team_b_mask]
            
            ball_frame_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            ball_pitch_xy = self.view_transformer.transform_points(ball_frame_xy)
            
            # 8. Update stats
            stats_data = self.stats.update(team_a_pitch, team_b_pitch, ball_pitch_xy)
        
        # 9. Annotation
        labels = [f"#{tid}" for tid in merged_detections.tracker_id]
        annotated = self.annotator.annotate_full(
            frame, merged_detections, ball_detections, labels, color_lookup
        )
        
        # 10. Generate radar
        radar = draw_radar(...)  # Using viz.py functions
        
        return {
            "annotated_frame": annotated,
            "radar": radar,
            "stats": stats_data or self.stats.get_summary()
        }
```

---

## 4️⃣ Desktop Application

### Application Purpose
PyQt6 desktop application with transparent overlay for real-time analysis.

### File Structure

```
src/app/
├── __init__.py
├── main.py              # Entry point
├── capture.py           # Screen capture module
├── overlay.py           # Transparent overlay window
├── hotkeys.py           # Global hotkey handling
├── settings.py          # Display preferences/settings
└── widgets/
    ├── __init__.py
    ├── stats_panel.py   # Statistics display
    ├── minimap.py       # 2D radar widget
    ├── heatmap.py       # Heatmap overlay
    └── settings_panel.py # Settings toggle panel
```

### Display Settings (User Customizable)

Users can toggle which stats/visualizations they want to see. Settings are persisted to a JSON file.

```python
# src/app/settings.py
from dataclasses import dataclass, field, asdict
from typing import Dict
import json
from pathlib import Path

@dataclass
class DisplaySettings:
    """
    User-configurable display preferences.
    
    Each boolean controls whether that visualization is shown.
    Users can toggle these via the settings panel (accessed with F10).
    """
    
    # Stats Panel
    show_stats_panel: bool = True
    show_possession: bool = True
    show_attacks: bool = True
    show_shots: bool = True
    show_shots_on_target: bool = True
    show_gk_saves: bool = True
    
    # Formation
    show_formation: bool = True
    show_formation_lines: bool = True  # Lines connecting players
    
    # Heatmap
    show_heatmap: bool = True
    
    # Minimap / Radar
    show_minimap: bool = True
    
    # Quick presets
    @classmethod
    def stats_only(cls) -> "DisplaySettings":
        """Preset: Only show numerical stats."""
        return cls(
            show_stats_panel=True,
            show_formation=True,
            show_formation_lines=False,
            show_heatmap=False,
            show_minimap=False
        )
    
    @classmethod
    def minimal(cls) -> "DisplaySettings":
        """Preset: Minimal - just possession and minimap."""
        return cls(
            show_stats_panel=True,
            show_possession=True,
            show_attacks=False,
            show_shots=False,
            show_shots_on_target=False,
            show_gk_saves=False,
            show_formation=False,
            show_formation_lines=False,
            show_heatmap=False,
            show_minimap=True
        )
    
    @classmethod
    def heatmap_only(cls) -> "DisplaySettings":
        """Preset: Only show heatmap."""
        return cls(
            show_stats_panel=False,
            show_formation=False,
            show_formation_lines=False,
            show_heatmap=True,
            show_minimap=False
        )
    
    def save(self, path: Path = None):
        """Save settings to JSON file."""
        path = path or Path.home() / ".sportsvision" / "settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path = None) -> "DisplaySettings":
        """Load settings from JSON file, or return defaults."""
        path = path or Path.home() / ".sportsvision" / "settings.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        return cls()  # Return defaults
```

### Settings Panel Widget

```python
# src/app/widgets/settings_panel.py
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, 
    QLabel, QPushButton, QComboBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

class SettingsPanel(QWidget):
    """
    Settings panel for toggling visualizations.
    Accessed via F10 hotkey.
    """
    
    settings_changed = pyqtSignal(object)  # Emits DisplaySettings
    
    def __init__(self, settings: DisplaySettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setup_ui()
        
        # Semi-transparent dark background
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 30, 30, 230);
                color: white;
                font-size: 12px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QPushButton {
                background-color: #4a4a4a;
                border: 1px solid #666;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a5a5a;
            }
        """)
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("⚙️ Display Settings")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Presets dropdown
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom", "All Stats", "Minimal", "Stats Only", "Heatmap Only"])
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        preset_layout.addWidget(self.preset_combo)
        layout.addLayout(preset_layout)
        
        # Separator
        layout.addWidget(self._separator())
        
        # Stats section
        layout.addWidget(QLabel("📊 Statistics"))
        self.cb_stats = self._checkbox("Show Stats Panel", "show_stats_panel")
        layout.addWidget(self.cb_stats)
        
        stats_indent = QWidget()
        stats_layout = QVBoxLayout(stats_indent)
        stats_layout.setContentsMargins(20, 0, 0, 0)
        self.cb_possession = self._checkbox("Possession", "show_possession")
        self.cb_attacks = self._checkbox("Attacks", "show_attacks")
        self.cb_shots = self._checkbox("Shots", "show_shots")
        self.cb_on_target = self._checkbox("Shots on Target", "show_shots_on_target")
        self.cb_saves = self._checkbox("GK Saves", "show_gk_saves")
        for cb in [self.cb_possession, self.cb_attacks, self.cb_shots, self.cb_on_target, self.cb_saves]:
            stats_layout.addWidget(cb)
        layout.addWidget(stats_indent)
        
        # Separator
        layout.addWidget(self._separator())
        
        # Formation section
        layout.addWidget(QLabel("📐 Formation"))
        self.cb_formation = self._checkbox("Show Formation", "show_formation")
        self.cb_formation_lines = self._checkbox("Show Player Lines", "show_formation_lines")
        layout.addWidget(self.cb_formation)
        layout.addWidget(self.cb_formation_lines)
        
        # Separator
        layout.addWidget(self._separator())
        
        # Visualizations section
        layout.addWidget(QLabel("🗺️ Visualizations"))
        self.cb_heatmap = self._checkbox("Show Heatmap", "show_heatmap")
        self.cb_minimap = self._checkbox("Show Minimap", "show_minimap")
        layout.addWidget(self.cb_heatmap)
        layout.addWidget(self.cb_minimap)
        
        # Save button
        layout.addWidget(self._separator())
        save_btn = QPushButton("💾 Save Settings")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)
        
        # Close hint
        hint = QLabel("Press F10 to close")
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)
    
    def _checkbox(self, label: str, setting_name: str) -> QCheckBox:
        """Create a checkbox bound to a setting."""
        cb = QCheckBox(label)
        cb.setChecked(getattr(self.settings, setting_name))
        cb.stateChanged.connect(lambda state: self._update_setting(setting_name, state == 2))
        return cb
    
    def _separator(self) -> QFrame:
        """Create a horizontal separator line."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #444;")
        return line
    
    def _update_setting(self, name: str, value: bool):
        """Update a setting value."""
        setattr(self.settings, name, value)
        self.preset_combo.setCurrentText("Custom")
        self.settings_changed.emit(self.settings)
    
    def apply_preset(self, preset_name: str):
        """Apply a preset configuration."""
        if preset_name == "Custom":
            return
        elif preset_name == "All Stats":
            self.settings = DisplaySettings()
        elif preset_name == "Minimal":
            self.settings = DisplaySettings.minimal()
        elif preset_name == "Stats Only":
            self.settings = DisplaySettings.stats_only()
        elif preset_name == "Heatmap Only":
            self.settings = DisplaySettings.heatmap_only()
        
        self._refresh_checkboxes()
        self.settings_changed.emit(self.settings)
    
    def _refresh_checkboxes(self):
        """Refresh all checkboxes to match current settings."""
        for cb, attr in [
            (self.cb_stats, "show_stats_panel"),
            (self.cb_possession, "show_possession"),
            (self.cb_attacks, "show_attacks"),
            (self.cb_shots, "show_shots"),
            (self.cb_on_target, "show_shots_on_target"),
            (self.cb_saves, "show_gk_saves"),
            (self.cb_formation, "show_formation"),
            (self.cb_formation_lines, "show_formation_lines"),
            (self.cb_heatmap, "show_heatmap"),
            (self.cb_minimap, "show_minimap"),
        ]:
            cb.blockSignals(True)
            cb.setChecked(getattr(self.settings, attr))
            cb.blockSignals(False)
    
    def save_settings(self):
        """Save settings to disk."""
        self.settings.save()
```

### Key Features

#### A. Full Screen Capture

```python
# src/app/capture.py
import mss
import numpy as np

class ScreenCapture:
    """Capture full screen at high FPS."""
    
    def __init__(self, monitor: int = 1, target_fps: int = 30):
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor]  # 0=all, 1=primary
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
    
    def capture(self) -> np.ndarray:
        """Capture current screen as numpy array (BGR)."""
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        # Convert BGRA to BGR
        return frame[:, :, :3]
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get screen resolution."""
        return (self.monitor["width"], self.monitor["height"])
```

#### B. Transparent Overlay Window

```python
# src/app/overlay.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter

class OverlayWindow(QWidget):
    """
    Transparent overlay window that sits on top of all windows.
    Hidden by default, shown with hotkey.
    """
    
    def __init__(self):
        super().__init__()
        
        # Frameless, always on top, transparent
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Full screen
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        
        # Hidden by default
        self.is_visible = False
        self.hide()
        
        # Stats to display
        self.current_stats = {}
        self.radar_image = None
    
    def toggle_visibility(self):
        """Toggle overlay visibility."""
        self.is_visible = not self.is_visible
        if self.is_visible:
            self.show()
        else:
            self.hide()
    
    def update_stats(self, stats: Dict, radar: np.ndarray):
        """Update displayed statistics."""
        self.current_stats = stats
        self.radar_image = radar
        self.update()  # Trigger repaint
    
    def paintEvent(self, event):
        """Draw overlay content."""
        if not self.is_visible:
            return
        
        painter = QPainter(self)
        # Draw stats panel, radar, etc.
        self._draw_stats_panel(painter)
        self._draw_radar(painter)
```

#### C. Main Application
```python
# src/app/main.py
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread, pyqtSignal

class ProcessingThread(QThread):
    """Background thread for video processing."""
    
    stats_updated = pyqtSignal(dict, object)  # stats, radar_image
    
    def __init__(self, pipeline: SportsPipeline, capture: ScreenCapture):
        super().__init__()
        self.pipeline = pipeline
        self.capture = capture
        self.running = True
    
    def run(self):
        while self.running:
            frame = self.capture.capture()
            result = self.pipeline.process_frame(frame)
            self.stats_updated.emit(result["stats"], result["radar"])

class SportsVisionApp:
    """Main application controller."""
    
    def __init__(self):
        self.app = QApplication(sys.argv)
        
        # Initialize components
        self.capture = ScreenCapture()
        self.pipeline = SportsPipeline(...)
        self.overlay = OverlayWindow()
        
        # Processing thread
        self.processing_thread = ProcessingThread(self.pipeline, self.capture)
        self.processing_thread.stats_updated.connect(self.overlay.update_stats)
        
        # Global hotkey (F9 to toggle)
        self.setup_hotkeys()
    
    def setup_hotkeys(self):
        """Register global hotkeys."""
        import keyboard
        keyboard.add_hotkey('f9', self.overlay.toggle_visibility)
    
    def run(self):
        """Start the application."""
        self.processing_thread.start()
        sys.exit(self.app.exec())

if __name__ == "__main__":
    app = SportsVisionApp()
    app.run()
```

### UI Layout (When Overlay is Visible)

The stats panel uses **numerical values + visual bar comparison** (similar to broadcast stats).

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              (Match Video - Pass Through)                           │
│                                                                                     │
│  ┌───────────────────────────────────────────────────────┐                         │
│  │  📊 LIVE MATCH STATS                                  │  ← Semi-transparent     │
│  │                                                       │     panel (black 60%)   │
│  │  ┌─────────────────────────────────────────────────┐ │                         │
│  │  │     Team A          STAT         Team B         │ │                         │
│  │  ├─────────────────────────────────────────────────┤ │                         │
│  │  │  62%  [████████████░░░░░░░░]  38%   Possession  │ │  ← Bar fills from      │
│  │  │   8   [████████████░░░░░░░░]   5    Attacks     │ │     both sides         │
│  │  │   4   [█████████░░░░░░░░░░░]   2    Shots       │ │                         │
│  │  │   3   [████████░░░░░░░░░░░░]   1    On Target   │ │                         │
│  │  │   0   [░░░░░░░░░░░░░░░░░░░░]   0    GK Saves    │ │                         │
│  │  └─────────────────────────────────────────────────┘ │                         │
│  │                                                       │                         │
│  │  Formation:  4-3-3          vs          4-4-2        │                         │
│  │              ─●─●─●─               ─●─●─●─●─         │  ← Formation lines      │
│  │             ─●─●─●─                  ─●─●─●─         │     connecting players  │
│  │            ─●─●─●─●─                ─●─●─●─●─        │                         │
│  │                                                       │                         │
│  │  ┌────────────────┐  ┌─────────────────────────────┐ │                         │
│  │  │ Heatmap: [▼]   │  │         2D Minimap          │ │  ← Dropdown: Team A    │
│  │  │ ┌───────────┐  │  │   ●  ●  ●     ○  ○  ○      │ │    Team B, or Player  │
│  │  │ │░░▒▒▓▓██░░ │  │  │      ●  ●  ⚽  ○  ○       │ │    by tracker ID      │
│  │  │ │░▒▓████▒░░ │  │  │   ●  ●  ●     ○  ○  ○      │ │                         │
│  │  │ │░░▒▓▓▒░░░░ │  │  │        ●           ○       │ │                         │
│  │  │ └───────────┘  │  └─────────────────────────────┘ │                         │
│  │  └────────────────┘                                   │                         │
│  └───────────────────────────────────────────────────────┘                         │
│                                                                                     │
│                                  Press F9 to hide                                   │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Stats Bar Widget Code

```python
# src/app/widgets/stats_bar.py
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QFont
from PyQt6.QtCore import Qt, QRect

class StatBarWidget(QWidget):
    """
    Visual bar comparing two values with numbers on sides.
    
    Example: 62% [████████░░░░] 38%
    """
    
    def __init__(
        self, 
        label: str = "Stat",
        team_a_color: QColor = QColor(65, 105, 225),  # Royal Blue
        team_b_color: QColor = QColor(220, 20, 60),   # Crimson
        parent=None
    ):
        super().__init__(parent)
        self.label = label
        self.team_a_color = team_a_color
        self.team_b_color = team_b_color
        self.value_a = 0
        self.value_b = 0
        self.setFixedHeight(30)
    
    def set_values(self, value_a: float, value_b: float):
        """Update bar values."""
        self.value_a = value_a
        self.value_b = value_b
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Background
        painter.fillRect(0, 0, width, height, QColor(0, 0, 0, 150))
        
        # Calculate proportions
        total = self.value_a + self.value_b
        if total > 0:
            ratio_a = self.value_a / total
        else:
            ratio_a = 0.5
        
        bar_width = int(width * 0.5)  # Central bar area
        bar_x = int(width * 0.25)
        bar_height = height - 10
        bar_y = 5
        
        # Draw Team A bar (left side, fills right)
        fill_a = int(bar_width * 0.5 * ratio_a / 0.5) if ratio_a > 0 else 0
        painter.fillRect(bar_x + bar_width//2 - fill_a, bar_y, fill_a, bar_height, self.team_a_color)
        
        # Draw Team B bar (right side, fills left)
        fill_b = int(bar_width * 0.5 * (1 - ratio_a) / 0.5) if ratio_a < 1 else 0
        painter.fillRect(bar_x + bar_width//2, bar_y, fill_b, bar_height, self.team_b_color)
        
        # Draw numbers and label
        painter.setPen(Qt.GlobalColor.white)
        painter.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        
        # Left value
        painter.drawText(QRect(0, 0, bar_x - 5, height), 
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                        str(int(self.value_a)))
        
        # Right value
        painter.drawText(QRect(bar_x + bar_width + 5, 0, width - bar_x - bar_width - 5, height),
                        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                        str(int(self.value_b)))
        
        # Center label
        painter.drawText(QRect(bar_x, 0, bar_width, height),
                        Qt.AlignmentFlag.AlignCenter,
                        self.label)
```

### Formation Lines Renderer

```python
# src/app/widgets/formation_view.py
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtCore import Qt
from typing import List, Tuple

class FormationLinesWidget(QWidget):
    """
    Draw formation with lines connecting players in same line.
    
    Visual:
        ─●─●─●─    (forwards)
       ─●─●─●─     (midfield)
      ─●─●─●─●─    (defense)
    """
    
    def __init__(self, team_color: QColor, parent=None):
        super().__init__(parent)
        self.team_color = team_color
        self.formation_lines = []  # List of player position lists per line
        self.formation_str = "---"
    
    def set_formation(self, formation_str: str, lines: List[List[Tuple[float, float]]]):
        """
        Set formation data.
        
        Args:
            formation_str: e.g., "4-3-3"
            lines: List of player groups, each group is players on same line
        """
        self.formation_str = formation_str
        self.formation_lines = lines
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        width = self.width()
        height = self.height()
        
        # Draw formation string
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(width // 2 - 20, 15, self.formation_str)
        
        if not self.formation_lines:
            return
        
        # Normalize positions to widget space
        # Each line gets a horizontal row
        line_height = (height - 30) / len(self.formation_lines)
        
        pen = QPen(self.team_color)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(self.team_color)
        
        for line_idx, players in enumerate(self.formation_lines):
            y = 25 + line_idx * line_height + line_height / 2
            
            if len(players) == 0:
                continue
            
            # Space players evenly across width
            player_spacing = width / (len(players) + 1)
            
            prev_x = None
            for p_idx, _ in enumerate(players):
                x = player_spacing * (p_idx + 1)
                
                # Draw player dot
                painter.drawEllipse(int(x - 4), int(y - 4), 8, 8)
                
                # Draw line connecting to previous player in same line
                if prev_x is not None:
                    painter.drawLine(int(prev_x), int(y), int(x), int(y))
                
                prev_x = x
```

### Heatmap Dropdown Widget

```python
# src/app/widgets/heatmap_selector.py
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QComboBox, QLabel
from PyQt6.QtGui import QImage, QPixmap
from typing import List, Dict
import numpy as np
import cv2

class HeatmapWidget(QWidget):
    """
    Heatmap display with selector for team or individual player.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.current_heatmap = None
    
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Dropdown selector
        self.selector = QComboBox()
        self.selector.addItem("Team A", "team_a")
        self.selector.addItem("Team B", "team_b")
        self.selector.addItem("Ball", "ball")
        # Individual players added dynamically
        self.selector.currentIndexChanged.connect(self.on_selection_changed)
        layout.addWidget(self.selector)
        
        # Heatmap display
        self.heatmap_label = QLabel()
        self.heatmap_label.setFixedSize(200, 140)
        layout.addWidget(self.heatmap_label)
    
    def update_players(self, players: List[Dict]):
        """
        Update dropdown with available players.
        
        Args:
            players: List of {"id": tracker_id, "team": "a" or "b"}
        """
        # Remove existing player entries (keep first 3: Team A, Team B, Ball)
        while self.selector.count() > 3:
            self.selector.removeItem(3)
        
        # Add players
        for p in sorted(players, key=lambda x: (x["team"], x["id"])):
            label = f"Player #{p['id']} (Team {'A' if p['team'] == 'a' else 'B'})"
            self.selector.addItem(label, f"player_{p['id']}")
    
    def set_heatmap(self, heatmap: np.ndarray):
        """Display heatmap array (grid_rows x grid_cols, 0-255)."""
        if heatmap is None:
            return
        
        # Apply colormap (COLORMAP_JET: blue -> green -> yellow -> red)
        colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        colored = cv2.resize(colored, (200, 140), interpolation=cv2.INTER_NEAREST)
        
        # Convert to QPixmap
        h, w, ch = colored.shape
        bytes_per_line = ch * w
        q_img = QImage(colored.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        self.heatmap_label.setPixmap(QPixmap.fromImage(q_img))
    
    def on_selection_changed(self, index):
        """Emit signal when selection changes."""
        selection = self.selector.currentData()
        # Parent should handle fetching the appropriate heatmap
        pass
```

---

## 📦 Dependencies to Add

```txt
# requirements.txt additions for Phase 4

# Desktop App
PyQt6>=6.5.0
mss>=9.0.0          # Fast screen capture
keyboard>=0.13.5     # Global hotkeys

# Already have (from Phase 1-3)
# numpy, opencv-python, supervision, ultralytics, scikit-learn, torch
```

---

## 🚀 Execution Order

1. **Create training script** with optimizations
2. **Implement stats.py** with all stat trackers
3. **Implement pipeline.py** connecting all components
4. **Build desktop app** with PyQt6
5. **Test end-to-end** with live screen capture
6. **Package as executable** with PyInstaller

---

## 📝 Notes

- **Latency Target:** < 100ms from capture to display
- **FPS Target:** 30 FPS processing
- **Hotkeys:**
  - `F9` - Toggle overlay visibility
  - `F10` - Open settings panel (toggle which stats/visualizations to show)
- **Stats are computed continuously** even when overlay is hidden
- **Formation detection:** String format (e.g., "4-3-3") + lines connecting players in same formation line
- **Partial View Handling:** When broadcast shows only part of pitch (e.g., defenders only):
  - Formation string shows "Partial (N)" where N = visible players
  - Player lines are STILL drawn connecting visible players in same depth band
  - Last known full formation is preserved in history
- **Heatmap:** Team-level (default) + Individual player heatmaps (selectable via dropdown)
- **Stats Display:** Numerical values + Visual bar comparison (like broadcast graphics)
- **Customizable Display:** Users can toggle individual stats/visualizations via settings panel
- **Annotations:** EllipseAnnotator for players, TriangleAnnotator for ball
- **Settings Persistence:** User preferences saved to `~/.sportsvision/settings.json`

---

## 📋 Visualization Summary

| Feature | Display Style | Toggleable |
|---------|---------------|------------|
| **Possession** | Percentage + Filled bar from both sides | ✅ |
| **Attacks** | Count + Bar comparison | ✅ |
| **Shots** | Count + Bar comparison | ✅ |
| **Shots on Target** | Count + Bar comparison | ✅ |
| **GK Saves** | Count + Bar comparison | ✅ |
| **Formation** | String (4-3-3) + Connected player lines | ✅ (separate toggles) |
| **Heatmap** | Color grid (JET colormap) + Dropdown selector | ✅ |
| **Minimap** | 2D pitch with player dots | ✅ |

### Quick Presets
| Preset | What's Shown |
|--------|--------------|
| **All Stats** | Everything enabled (default) |
| **Minimal** | Just possession + minimap |
| **Stats Only** | Numbers only, no heatmap/minimap |
| **Heatmap Only** | Just the heatmap visualization |
| **Minimap** | 2D pitch with dots for players + ball |

---

*This document will be updated as implementation progresses.*
