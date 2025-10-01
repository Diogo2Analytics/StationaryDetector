# Stationary Detector

A computer vision system for analyzing pedestrian behavior and dwell patterns in urban environments.

![Dashboard Overview](images/dashboard-screenshot.png)
*Real-time dashboard showing multi-panel analysis of pedestrian behavior*

## Background

This project emerged from urban analytics work conducted for BipZip Lisbon and EDP Move in Braga. The methodology focuses on analyzing mixed-use spaces where cars and pedestrians coexist - particularly around schools and transit areas - to identify where people congregate and spend the most time.

The core research question was: **where do people spend more time in public spaces?** Understanding these patterns enables evidence-based urban interventions, such as converting parking spaces to seating areas, installing shade structures for parents waiting for children, or redesigning pedestrian infrastructure based on actual usage patterns.

This repository contains the computer vision and behavioral analysis components. Environmental monitoring (air quality, humidity, temperature) was conducted separately and is not included here.

<div align="center">
  <img src="images/workflow-diagram.png" alt="Analysis Workflow" width="600"/>
  <p><em>System workflow: from video input to behavioral analytics</em></p>
</div>

## Technical Approach

The system addresses a fundamental challenge in urban analytics: **how to automatically identify when unique individuals are stationary and accurately measure the duration each person spends in specific locations.**

The solution combines multiple computer vision techniques:

**Person Detection & Tracking**: Uses YOLOv8 for robust person detection with persistent ID tracking across video frames, maintaining individual identity even through temporary occlusions.

**Movement Classification**: Employs MOG2 background subtraction to distinguish between moving and stationary behavior, analyzing pixel-level changes within each person's bounding box to determine their activity state.

**Spatial Mapping**: Implements homography-based camera calibration to transform image coordinates into real-world GPS coordinates, enabling precise geographic localization of stationary events.

**Temporal Analysis**: Tracks duration metrics per individual, accumulating time spent in stationary states and exporting structured analytics for further analysis.

## System Capabilities

- Real-time video processing with multi-panel visualization dashboard
- Persistent person tracking with unique ID assignment
- Movement vs. stationary state classification with configurable sensitivity
- GPS coordinate mapping through camera calibration matrices  
- Structured data export showing temporal patterns per individual
- Configurable processing parameters for different deployment scenarios

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Diogo2Analytics/StationaryDetector.git
   cd StationaryDetector
   ```

2. Create virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install Poetry:
   ```bash
   pip3 install poetry
   ```

4. Install dependencies:
   ```bash
   poetry install
   ```

## Running it

```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Run with the included demo video
python src/stationary_detector/main.py

# Or run with your own video
python src/stationary_detector/main.py /path/to/your/video.mp4
```

Press 'q' to quit the application. Make sure to click on the video window first so it can capture the keypress.

## Project Structure

```
StationaryDetector/
├── src/
│   └── stationary_detector/
│       ├── __init__.py           # Package initialization
│       └── main.py               # Main application & dashboard
├── resources/
│   ├── camera_parameters/        # Camera calibration files
│   ├── grid/                     # Spatial grid definitions
│   ├── satellite_images/         # Aerial/satellite imagery
│   └── videos/                   # Sample video files
├── results/                      # Analysis outputs
│   └── analytics.json            # Time-based analytics data
├── pyproject.toml                # Poetry dependencies & config
└── README.md                     # This file
```

## Configuration

You can modify the key parameters at the top of `src/stationary_detector/main.py`:

```python
# Video & Processing Settings
DEFAULT_VIDEO_FILE = "resources/videos/2.mp4"    # Input video path
FORCE_FRAME = 4                                  # Processing frame rate
CONFIDENCE_THRESHOLD = 0.20                      # YOLO detection confidence
MOVEMENT_THRESHOLD = 0.18                        # Motion detection sensitivity

# Resource Paths  
SCENARIO_FILE = "resources/camera_parameters/..."     # Camera calibration
SATELLITE_IMAGE = "resources/satellite_images/..."    # Base map
GPS_CORNER_COORDINATES = [...]                        # Area boundaries

# Dashboard Layout
PANEL_WIDTH = 480                               # Dashboard panel dimensions
PANEL_HEIGHT = 360
```

## Dashboard Interface

The application provides a comprehensive real-time visualization through a 3×2 panel dashboard:

**Top Row Analysis:**
- **Original Footage**: Raw video stream for context and verification
- **Detection View**: YOLO bounding boxes with persistent person IDs and behavioral state labels
- **Movement Analysis**: MOG2 background subtraction mask showing detected motion and shadow areas

**Bottom Row Insights:**  
- **Geographic Projection**: Satellite imagery with GPS-mapped detection points showing real-world positioning
- **System Analytics**: Live log displaying detection events, movement classifications, and timing metrics
- **Threshold Visualization**: Binary movement classification output for algorithm transparency and debugging

| Original View | Detection & Tracking | Movement Analysis |
|---------------|---------------------|-------------------|
| ![Original](images/panel-original.jpg) | ![Detection](images/panel-detection.jpg) | ![Movement](images/panel-movement.jpg) |

*Dashboard panels showing different analysis views of the same scene*

## Dependencies

The system builds on several key libraries:

**Computer Vision & Machine Learning:**
- OpenCV (`opencv-python`) for core computer vision operations and video processing
- PyTorch (`torch`, `torchvision`) providing the deep learning framework
- Ultralytics (`ultralytics`) for YOLOv8 object detection implementation
- Deep Sort (`deep-sort-realtime`) enabling robust multi-object tracking

**Scientific Computing:**
- NumPy (`numpy`) for numerical operations and array processing
- SciPy (`scipy`) for scientific computing, particularly GPS coordinate interpolation
- Hachoir (`hachoir`) for video metadata extraction and analysis

**Additional utilities:** JSON handling, datetime operations, and pathlib for file system management.

Complete dependency specifications with pinned versions are available in `pyproject.toml`.

## Analytics Output

The system generates structured analytics data saved to `results/analytics.json`, containing stationary duration metrics per individual:

```json
{
  "person_id_1": 45.2,    // Total stationary time in seconds
  "person_id_2": 12.8,    // Per unique tracked individual  
  "person_id_3": 67.4     // Accumulated across entire analysis period
}
```

**Real-time Console Output:**
During processing, the system provides live feedback including:
- Individual tracking status with persistent person IDs
- Movement state classification (moving/stationary) with confidence metrics
- GPS coordinate mapping for each detection event  
- Cumulative time tracking for stationary behavior patterns
- System performance metrics and processing statistics

This dual output approach enables both real-time monitoring and post-analysis review of pedestrian behavior patterns.