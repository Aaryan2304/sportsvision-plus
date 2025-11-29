# SportsVision+ 🏟️⚽

**End-to-end real-time football analytics system.**

A computer vision pipeline that performs real-time player detection, tracking, team classification, and tactical visualization on football broadcast footage.

![Football AI Diagram](https://media.roboflow.com/notebooks/examples/football-ai-diagram.png)

## 🎯 Project Goal

Build a real-time football analytics system that can:
- Detect players, referees, goalkeepers, and the ball
- Track objects across frames with persistent IDs
- Classify players into teams based on jersey colors
- Transform camera view to a 2D tactical radar/minimap
- Stream results via a web interface at 30 FPS

**Hardware Target:** Single laptop GPU (RTX 3050/3060/4050)

## 📁 Repository Structure

```
sportsvision-plus/
├── data/
│   ├── soccernet/           # Raw SoccerNet tracking data
│   ├── processed/           # YOLO format images/labels
│   │   ├── images/
│   │   ├── labels/
│   │   └── dataset.yaml
│   └── models/              # Trained model weights (.pt files)
├── src/
│   ├── data_engine/         # Data parsing scripts
│   │   ├── __init__.py
│   │   ├── soccernet_parser.py
│   │   └── download_soccernet.py
│   ├── inference/           # Core AI logic
│   │   ├── __init__.py
│   │   ├── detector.py      # YOLO11 + Roboflow Pitch Model
│   │   ├── tracker.py       # ByteTrack + Ball Interpolation
│   │   ├── team.py          # Team Color Classifier (KMeans)
│   │   └── pipeline.py      # Main System Loop
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── geometry.py      # Homography & ViewTransformer
│   │   └── viz.py           # Annotators & Pitch Drawing
│   └── web/
│       ├── __init__.py
│       └── main.py          # FastAPI MJPEG Streamer
├── requirements.txt
├── Dockerfile
├── README.md
└── SportsVision+.md         # Detailed architecture guide
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download SoccerNet Data

```bash
# Using the download script
python src/data_engine/download_soccernet.py -o data/soccernet

# Or manually in Python:
from SoccerNet.Downloader import SoccerNetDownloader
downloader = SoccerNetDownloader(LocalDirectory="data/soccernet")
downloader.downloadDataTask(task="tracking", split=["train", "test", "challenge"])
```

### 3. Parse Data to YOLO Format

The parser converts SoccerNet MOT format to YOLO format, handling:
- `gt/gt.txt` ground truth files
- `seqinfo.ini` for image dimensions
- `gameinfo.ini` for track-to-class mapping
- Automatic `.zip` extraction

```bash
python src/data_engine/soccernet_parser.py \
    --input data/soccernet \
    --output data/processed \
    --split
```

This will:
- Process all `SNMOT-XXX` sequences
- Convert annotations to YOLO format (normalized coordinates)
- Split into train/val sets (80/20)
- Generate `dataset.yaml` for YOLO training

### 4. Train YOLO Model

```bash
yolo detect train \
    model=yolo11n.pt \
    data=data/processed/dataset.yaml \
    epochs=20 \
    imgsz=640
```

Move the trained weights:
```bash
cp runs/detect/train/weights/best.pt data/models/
```

### 5. Run the Pipeline

```bash
uvicorn src.web.main:app --reload --host 0.0.0.0 --port 8000
```

Open browser to: `http://localhost:8000/video_feed`

## 🏗️ Architecture

### Detection Classes
| ID | Class |
|----|-------|
| 0 | Ball |
| 1 | Player |
| 2 | Referee |
| 3 | Goalkeeper |

### Pipeline Flow
```
Frame → YOLO Detection → ByteTrack → Team Classification → Homography → Visualization
                                              ↓
                                    Camera View + 2D Radar
```

### Key Components

1. **ObjectDetector** (`detector.py`)
   - Wraps YOLO11 for player/ball detection
   - Roboflow model for pitch keypoint detection

2. **BallTracker** (`tracker.py`)
   - ByteTrack for object tracking
   - Anti-teleport logic for ball smoothing
   - Missing ball interpolation

3. **TeamClassifier** (`team.py`)
   - KMeans clustering on jersey colors
   - Fast average RGB feature extraction

4. **ViewTransformer** (`geometry.py`)
   - Homography-based perspective transformation
   - Maps camera pixels to 2D pitch coordinates

5. **SportsPipeline** (`pipeline.py`)
   - Orchestrates all components
   - Real-time processing loop

## 📊 Data Sources

### SoccerNet Tracking Dataset

The project uses the [SoccerNet Tracking Dataset](https://www.soccer-net.org/) which provides annotations in **MOT (Multiple Object Tracking) format**.

**Data Structure:**
```
SNMOT-XXX/
├── gt/
│   └── gt.txt          # Ground truth: frame_id, track_id, x, y, w, h, conf, class_id, visibility
├── img1/               # Video frames as sequential images
├── seqinfo.ini         # Sequence info: image dimensions, frame rate, total frames
└── gameinfo.ini        # Track ID mapping: trackletID_X = "player team left", "ball", etc.
```

**Class Mapping:**
| SoccerNet Class | YOLO ID |
|-----------------|---------|
| Ball | 0 |
| Player (any team) | 1 |
| Referee | 2 |
| Goalkeeper | 3 |

- **Pitch Model:** [Roboflow Football Field Detection](https://universe.roboflow.com/)
- **Inference:** Any broadcast football video

## 🛠️ Dependencies

- `ultralytics` - YOLO11 object detection
- `supervision` - Detection/tracking utilities
- `inference` - Roboflow model hosting
- `opencv-python` - Computer vision operations
- `scikit-learn` - KMeans for team classification
- `fastapi` + `uvicorn` - Web streaming

## 📝 License

Apache License - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Roboflow](https://roboflow.com/) for inference tools and models
- [SoccerNet](https://www.soccer-net.org/) for tracking dataset
- [Supervision](https://github.com/roboflow/supervision) library
