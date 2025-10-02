# Stationary Detector

A computer vision multi-layer approach for analyzing pedestrian behavior and dwell patterns in urban environments.

<div align="center">
  <img src="footage/complete_dashboard_20251002_090908.gif" alt="Complete Dashboard Overview" width="1000"/>
  <p><em>Analysis dashboard showing the detection system</em></p>
</div>

## Background

This project emerged from urban analytics work conducted for BipZip Lisbon and EDP Move in Braga. The methodology focuses on analyzing mixed-use spaces where cars and pedestrians coexist - particularly around schools and transit areas - to identify where people spend the most time. Due to GDPR compliance requirements, we cannot store any video footage, only anonymized metadata values. Most work was done in public schools that lacked proper infrastructure.

## Problem to solve

The problem we wanted to solve was: **How to identify areas of public space occupied by pedestrians and measure the time they remain stationary.** Understanding these patterns enables evidence-based urban interventions, such as converting parking spaces to seating areas, installing shade structures for parents waiting for children, or redesigning pedestrian infrastructure based on actual usage patterns.

This repository contains the computer vision and analysis components. Environmental monitoring (air quality, humidity, temperature) was conducted separately and is not included here.

## Technical Approach

The system addresses a fundamental challenge in urban analytics: **how to automatically identify when unique individuals are stationary and accurately measure the duration each person spends in specific locations.**

Due to GDPR constraints, we cannot store video footage - only metadata information. For demonstrations, we use pre-recorded content, but the production system operates on live streams without retention.

The analysis follows a sequential 5-step process:

### Step 1: Camera Calibration & Setup
We obtain a satellite view of the analysis location and map the corners of the camera field of view to real-world GPS coordinates for spatial transformation. This enables conversion between pixel coordinates and geographic locations.

<div align="left">
  <img src="footage/homography1.png" alt="Camera Calibration Setup" width="600"/>
  <p><em>GPS corner mapping for camera-to-world coordinate transformation</em></p>
</div>

<div align="right">
  <img src="footage/homography2.png" alt="Homography Matrix Setup" width="600"/>
  <p><em>Homography transformation matrix calibration process</em></p>
</div>

### Step 2: Background Reference & Person Detection
The system captures the first frame where no people are visible to establish a background reference for movement detection. YOLO8 identifies individuals in pixel coordinates and distinguishes between different people. The system assigns persistent IDs to track individuals across frames. If someone is occluded for extended periods, they may be assigned a new ID upon reappearance.

<div align="center">
  <img src="footage/original_footage_20251002_090908.gif" alt="Original Footage" width="600"/>
  <p><em>Original camera input showing the monitored area</em></p>
</div>

<div align="center">
  <img src="footage/detection_view_20251002_090908.gif" alt="Person Detection" width="600"/>
  <p><em>YOLO detection with bounding boxes and persistent ID tracking</em></p>
</div>

### Step 3: Movement Classification
**MOG2 Background Subtraction**: Analyzes pixel changes within each person's bounding box. If pixel variation exceeds the 20% threshold, the person is classified as "moving"; otherwise "stationary." Frame rate is used to calculate time duration in seconds.

<div align="center">
  <img src="footage/movement_mask_20251002_090908.gif" alt="Movement Analysis" width="600"/>
  <p><em>Background subtraction highlighting motion areas in white</em></p>
</div>

<div align="center">
  <img src="footage/threshold_analysis_20251002_090908.gif" alt="Threshold Analysis" width="600"/>
  <p><em>Processed threshold analysis for movement detection</em></p>
</div>

### Step 4: Spatial & Temporal Mapping  
**Homography Coordinate Conversion**: Transforms pixel coordinates to GPS positions using the transformation matrix. **Duration Tracking**: Accumulates time spent in stationary states per individual, maintaining data for pattern analysis.

<div align="center">
  <img src="footage/satellite_view_20251002_090908.gif" alt="GPS Mapping" width="600"/>
  <p><em>Bird's eye view with GPS coordinate transformation and tracking points</em></p>
</div>

### Step 5: Data Export & Analytics
**GeoJSON Generation**: Compiles stationary events by person ID, recording locations where individuals spent time. Multiple stationary locations per person are tracked separately for behavioral analysis. **Real-time Monitoring**: Provides system logs and analytics.

<div align="center">
  <img src="footage/system_log_20251002_090908.gif" alt="System Analytics" width="600"/>
  <p><em>Real-time system logs and analytics output</em></p>
</div>


## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Diogo2Analytics/StationaryDetector.git
   cd StationaryDetector
   ```

2. Create a virtual environment:
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

## Usage

```bash
# Activate the virtual environment
source .venv/bin/activate

# Run the detector with the included demo video
python src/stationary_detector/main.py

# Controls:
# • Press 'q' or 'Q' to quit
# • Press 'ESC' to exit
# • Click on the video window to ensure it has focus
```
## Analytics Output

The system generates multiple output formats:

### 1. Simple Analytics (`analytics.json`)
Basic duration metrics per individual:
```json
{
  "person_id_1": 45.2,    // Total stationary time in seconds
  "person_id_2": 12.8,    // Per unique tracked individual  
  "person_id_3": 67.4     // Accumulated across entire analysis period
}
```

### 2. Geographic Data (`geojson.json`)
GeoJSON format with GPS coordinates for mapping and AutoCAD integration:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-9.16995835877193, 38.717420738782415]
      },
      "properties": {
        "row_id": 0,
        "tempo_de_permanencia": 1.75,
        "id_person": 126,
        "timestamp": "2024-02-17 10:16:15.900000",
        "unique_identifier": 5260585897646893104
      }
    }
  ]
}
```
### Contributor
Diogo Martins - https://www.linkedin.com/in/diogo-martins-/