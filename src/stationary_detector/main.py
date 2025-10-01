"""
Stationary Detector - Clean Object Detection System
Detects stationary people using YOLO and background subtraction.
"""

import torch
import cv2
import numpy as np
import json
import math
import os
import time
import csv
from pathlib import Path
from datetime import datetime, timedelta
from ultralytics import YOLO
from scipy.interpolate import LinearNDInterpolator
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata


# =============================================================================
# CONFIGURATION PARAMETERS - Easy to modify
# =============================================================================

# Video and Processing Settings
DEFAULT_VIDEO_FILE = "resources/videos/2.mp4"
FORCE_FRAME = 4  # Frames per second to process
FRAME_INTERVAL = 10  # Display refresh interval
DEBUG_MODE = False
GPS_COORDINATES = True

# Detection Settings
CONFIDENCE_THRESHOLD = 0.20  # YOLO confidence threshold
MOVEMENT_THRESHOLD = 0.18  # Movement detection threshold (0-1)
IMAGE_SIZE = 1920  # YOLO processing image size

# GeoJSON Generation Settings
GEOJSON_MIN_STATIONARY_FRAMES = 6  # Minimum consecutive stationary frames before creating GeoJSON entry

# Resource Paths
SCENARIO_FILE = "resources/camera_parameters/R_Andrade_Corvo.json"
SATELLITE_IMAGE = "resources/satellite_images/R. Andrade Corvo.png"
RESULTS_DIR = "results"
ANALYTICS_FILE = "analytics.json"
GEOJSON_FILE = "geojson.json"
DETAILED_CSV_FILE = "detailed_results.csv"



# Dashboard Settings
PANEL_WIDTH = 480  # Smaller panels to fit more views
PANEL_HEIGHT = 360
DASHBOARD_BORDER = 2

# MOG2 Background Subtractor Settings
MOG_HISTORY = 300
MOG_VAR_THRESHOLD = 8
MOG_DETECT_SHADOWS = True

# GPS Corner Coordinates (top-left, bottom-left, bottom-right, top-right)
GPS_CORNER_COORDINATES = [
    [38.7174781, -9.1702299],  # top-left
    [38.7171197, -9.1702293],  # bottom-left
    [38.7171223, -9.1696365],  # bottom-right
    [38.7174791, -9.1696378]   # top-right
]

# =============================================================================


# Global dictionaries to track person IDs and their stationary state
id_bboxes = {}
stationary_frame_counts = {}  # Track consecutive stationary frames per person
csv_data_buffer = []  # Buffer for CSV data


def interpolate_gps(point, satellite_width, satellite_height, corner_coordinates):
    """Convert image coordinates to GPS using interpolation."""
    try:
        x = np.array([0, 0, satellite_width, satellite_width])
        y = np.array([0, satellite_height, satellite_height, 0])
        
        latlon = LinearNDInterpolator((x, y), corner_coordinates)
        latpoint = latlon(point[0], point[1])
        lat = latpoint[0]
        lon = latpoint[1]
        
        # Validate interpolation results
        if math.isnan(lat) or math.isnan(lon):
            return None, None
            
        return lon, lat
    except Exception as e:
        return None, None


def define_stationary_box(is_moving, id, centroid, lon, lat):
    """Track stationary state for each person ID."""
    global id_bboxes
    
    detection_id = int(id)
    
    # Initialize new ID
    if detection_id not in id_bboxes:
        id_bboxes[detection_id] = {
            'current_centroid': centroid,
            'current_lon': lon,
            'current_lat': lat,
            'previous_centroid': None,
            'previous_lon': None,
            'previous_lat': None,
            'is_moving': is_moving,
            'recorded_lon': None,
            'recorded_lat': None
        }
    else:
        # Update tracking data
        if is_moving:
            id_bboxes[detection_id]['current_centroid'] = centroid
            id_bboxes[detection_id]['current_lon'] = lon
            id_bboxes[detection_id]['current_lat'] = lat
            id_bboxes[detection_id]['is_moving'] = True
        else:
            # Record first stationary position
            if id_bboxes[detection_id]['recorded_lon'] is None:
                id_bboxes[detection_id]['recorded_lon'] = lon
                id_bboxes[detection_id]['recorded_lat'] = lat
            
            # Update tracking
            id_bboxes[detection_id]['previous_centroid'] = id_bboxes[detection_id]['current_centroid']
            id_bboxes[detection_id]['previous_lon'] = id_bboxes[detection_id]['current_lon']
            id_bboxes[detection_id]['previous_lat'] = id_bboxes[detection_id]['current_lat']
            
            id_bboxes[detection_id]['current_centroid'] = centroid
            id_bboxes[detection_id]['current_lon'] = lon
            id_bboxes[detection_id]['current_lat'] = lat
            id_bboxes[detection_id]['is_moving'] = False
    
    # Return stationary coordinates
    if id_bboxes[detection_id]['is_moving']:
        return None, None
    else:
        return id_bboxes[detection_id]['recorded_lon'], id_bboxes[detection_id]['recorded_lat']


def log_detection_to_csv(person_id, bbox, confidence, detect_class, is_moving, lat, lon, 
                        seconds_frame, timestamp_current, creating_timestamp, csv_file):
    """Log detection data to CSV file similar to original BIP_ZEDCAM format."""
    
    # Convert bbox to string format like original
    bbox_str = f"[array({bbox})]" if bbox is not None else "[]"
    
    # Prepare row data
    row_data = {
        'id': person_id,
        'box': bbox_str,
        'conf': confidence,
        'detect_class': detect_class,
        'is_moving': is_moving,
        'lat': lat,
        'lon': lon,
        'seconds_frame': seconds_frame,
        'timestamp_current': timestamp_current,
        'creating_timestamp': creating_timestamp
    }
    
    # Check if CSV file exists, if not create with headers
    csv_path = Path(RESULTS_DIR) / csv_file
    file_exists = csv_path.exists()
    
    # Append to CSV
    with open(csv_path, 'a', newline='') as file:
        fieldnames = ['id', 'box', 'conf', 'detect_class', 'is_moving', 'lat', 'lon', 
                     'seconds_frame', 'timestamp_current', 'creating_timestamp']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_data)


def validate_stationary_frames(person_id, is_moving):
    """Track consecutive stationary frames and validate minimum threshold."""
    global stationary_frame_counts
    
    if person_id not in stationary_frame_counts:
        stationary_frame_counts[person_id] = {'consecutive_stationary': 0, 'is_validated': False}
    
    if not is_moving:
        # Increment stationary count
        stationary_frame_counts[person_id]['consecutive_stationary'] += 1
        
        # Check if we've reached minimum threshold
        if (stationary_frame_counts[person_id]['consecutive_stationary'] >= GEOJSON_MIN_STATIONARY_FRAMES and 
            not stationary_frame_counts[person_id]['is_validated']):
            stationary_frame_counts[person_id]['is_validated'] = True
            return True  # This is a newly validated stationary event
    else:
        # Reset if person starts moving
        stationary_frame_counts[person_id]['consecutive_stationary'] = 0
        stationary_frame_counts[person_id]['is_validated'] = False
    
    return False


def calibrate_cam(img, scenario_path: str):
    """Load camera calibration parameters and compute homography matrix."""
    with open(scenario_path, 'r') as file:
        data = json.load(file)
        source_pts = np.array(data.get("source_pts"), dtype=np.float32)
        projection_pts = np.array(data.get("projection_pts"), dtype=np.float32)
    
    homography_matrix, _ = cv2.findHomography(source_pts, projection_pts)
    return homography_matrix


def display_bev(satellite_img, detection_coords, homography_matrix):
    """Project detection onto satellite image using homography."""
    xmin, ymin, xmax, ymax = detection_coords[0]
    
    # Use bottom corners of bounding box
    bottom_corners = np.array([
        [xmin, ymax],  # bottom-left corner
        [xmax, ymax]   # bottom-right corner
    ], dtype='float32').reshape(-1, 1, 2)
    
    # Transform to satellite view
    points_out = cv2.perspectiveTransform(bottom_corners, homography_matrix)
    
    # Calculate centroid
    centroid = np.mean(points_out, axis=0).ravel()
    centroid = tuple(map(int, centroid))
    
    # Draw detection on satellite image
    cv2.circle(satellite_img, centroid, 8, (0, 0, 0), 2)  # Black outline
    cv2.circle(satellite_img, centroid, 7, (0, 255, 0), -1)  # Green fill
    
    return satellite_img, centroid


def calculate_movement_mog(frame, fgmask, detect):
    """Determine if person is moving using MOG2 background subtraction."""
    detection_id = int(detect.boxes.id)
    
    # Clean up foreground mask
    fgthres = cv2.threshold(fgmask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
    
    # Get bounding box region
    xmin, ymin, xmax, ymax = np.array(detect.boxes.xyxy)[0]
    fgthres_roi = fgthres[int(ymin):int(ymax), int(xmin):int(xmax)]
    
    # Calculate movement percentage
    white_pixel_count = np.sum(fgthres_roi == 255)
    total_pixels = fgthres_roi.size
    white_pixel_percentage = white_pixel_count / total_pixels
    
    is_moving = white_pixel_percentage > MOVEMENT_THRESHOLD
    print(f"ID {detection_id} - Moving: {is_moving} ({white_pixel_percentage:.2f})")
    
    return is_moving, white_pixel_percentage


def draw_labels(image, labels, bbox, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=1):
    """Draw labels on detected objects."""
    x_min, y_min, x_max, y_max = bbox[0]
    
    text_offset_x = x_min
    text_offset_y = y_min - 5
    
    background_color = (0, 0, 255)  # Red
    text_color = (255, 255, 255)     # White
    
    for idx, label in enumerate(labels):
        label = str(label)
        
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Ensure text stays within image bounds
        if text_offset_x + text_width > image.shape[1]:
            text_offset_x = image.shape[1] - text_width - 5
        if text_offset_y + text_height > image.shape[0]:
            text_offset_y = image.shape[0] - text_height - 5
        
        # Draw background rectangle
        rectangle_bgr = (
            (int(text_offset_x), int(text_offset_y)), 
            (int(text_offset_x) + int(text_width) + 2, int(text_offset_y) - int(text_height) - 2)
        )
        cv2.rectangle(image, rectangle_bgr[0], rectangle_bgr[1], background_color, cv2.FILLED)
        
        # Draw text
        cv2.putText(image, label, (int(text_offset_x), int(text_offset_y)), 
                   font, font_scale, text_color, thickness)
        
        text_offset_y -= text_height + 5
    
    return image


def frame_manipulation(video_cap, frame, current_frame, force_frame):
    """Control frame sampling rate."""
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / force_frame)
    
    if current_frame % frame_interval == 0:
        current_frame += 1
        return frame, current_frame
    else:
        current_frame += 1
        return None, current_frame


def get_video_info_hachoir(file_path):
    """Get video metadata using hachoir."""
    file_path = str(Path(file_path))
    mtime_timestamp = os.path.getmtime(file_path)
    mtime = datetime.fromtimestamp(mtime_timestamp)
    return mtime


def set_background(background, image_frame):
    """Set background frame for comparison."""
    if background is None:
        print("Background image set")
        background = "set"
        still_image = np.copy(image_frame)
        return still_image, background
    return None, background


def debug_mechanism(debug_mode, frame_interval, iter_flag):
    """Handle keyboard input for debugging and program control."""
    if debug_mode:
        key = cv2.waitKey(0) & 0xFF
        if key == ord(' '):  # Space to continue in debug mode
            return iter_flag
        elif key == ord('q') or key == ord('Q') or key == 27:  # 'q', 'Q' or ESC to quit
            print("\n=== Program stopped by user (q pressed) ===")
            cv2.destroyAllWindows()
            return False
    else:
        # This function is now only called in debug mode
        # Main loop handles key detection directly
        pass
    
    return iter_flag


def save_analytics(user_id, is_moving, force_frame, analytics_file, lon=None, lat=None, timestamp=None, is_newly_stationary=False):
    """Enhanced analytics tracking with GeoJSON support and frame validation."""
    # Simple analytics (existing format)
    try:
        with open(analytics_file, 'r') as file:
            analytics_table = json.load(file)
    except FileNotFoundError:
        analytics_table = {}
    
    frame_duration = 1 / force_frame
    user_id = str(user_id)
    
    if not is_moving:
        if user_id in analytics_table:
            analytics_table[user_id] += frame_duration
        else:
            analytics_table[user_id] = frame_duration
    
    with open(analytics_file, 'w') as file:
        json.dump(analytics_table, file)
    
    # GeoJSON analytics (only for newly validated stationary events with valid GPS)
    if (is_newly_stationary and lon is not None and lat is not None and 
        not math.isnan(lon) and not math.isnan(lat)):
        # Calculate accumulated duration for this validated event
        accumulated_duration = GEOJSON_MIN_STATIONARY_FRAMES * frame_duration
        save_geojson_analytics(user_id, accumulated_duration, lon, lat, timestamp)


def save_geojson_analytics(user_id, duration, lon, lat, timestamp=None):
    """Save stationary detection data in GeoJSON format."""
    # Validate coordinates - only save if valid GPS coordinates
    if (lon is None or lat is None or 
        math.isnan(lon) or math.isnan(lat) or
        not isinstance(lon, (int, float)) or not isinstance(lat, (int, float))):
        return
    
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    geojson_file = results_dir / GEOJSON_FILE
    
    # Load existing GeoJSON data
    try:
        with open(geojson_file, 'r') as file:
            geojson_data = json.load(file)
    except FileNotFoundError:
        geojson_data = {
            "type": "FeatureCollection",
            "features": []
        }
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    # Generate unique identifier
    unique_id = hash(f"{user_id}_{timestamp}_{lon}_{lat}")
    
    # Create new feature
    new_feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lat, lon]  # Coordinates in [latitude, longitude] format
        },
        "properties": {
            "row_id": len(geojson_data["features"]),
            "tempo_de_permanencia": duration,
            "id_person": int(user_id),
            "timestamp": timestamp,
            "unique_identifier": abs(unique_id)  # Make positive
        }
    }
    
    # Check if we should add or update existing feature
    existing_feature_index = None
    for i, feature in enumerate(geojson_data["features"]):
        if (feature["properties"]["id_person"] == int(user_id) and 
            abs(feature["geometry"]["coordinates"][0] - lat) < 0.0001 and 
            abs(feature["geometry"]["coordinates"][1] - lon) < 0.0001):
            existing_feature_index = i
            break
    
    if existing_feature_index is not None:
        # Update existing feature duration
        geojson_data["features"][existing_feature_index]["properties"]["tempo_de_permanencia"] += duration
        geojson_data["features"][existing_feature_index]["properties"]["timestamp"] = timestamp
    else:
        # Add new feature
        geojson_data["features"].append(new_feature)
    
    # Save updated GeoJSON
    with open(geojson_file, 'w') as file:
        json.dump(geojson_data, file, indent=2)


def create_terminal_overlay(width, height, log_messages):
    """Create a terminal-like overlay with log messages."""
    # Create black background
    terminal_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Terminal styling
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_color = (0, 255, 0)  # Green terminal text
    line_height = 20
    margin = 10
    
    # Add title
    cv2.putText(terminal_img, "DETECTION LOG", (margin, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add log messages (last 15 lines)
    start_y = 60
    for i, message in enumerate(log_messages[-15:]):
        y_pos = start_y + (i * line_height)
        if y_pos < height - 10:
            # Truncate long messages
            if len(message) > 60:
                message = message[:57] + "..."
            cv2.putText(terminal_img, message, (margin, y_pos), 
                       font, font_scale, text_color, 1)
    
    return terminal_img


def create_dashboard(original_frame, detection_frame, satellite_img, foreground_mask, frame_count, log_messages=None):
    """Create a 3x2 dashboard showing all views including shadow/movement detection."""
    if log_messages is None:
        log_messages = []
    
    # Define panel dimensions
    panel_width = PANEL_WIDTH
    panel_height = PANEL_HEIGHT
    border = DASHBOARD_BORDER
    
    # Create dashboard canvas (3 columns, 2 rows)
    dashboard_width = (panel_width * 3) + (border * 4)
    dashboard_height = (panel_height * 2) + (border * 3)
    dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
    
    # Resize all panels to standard size
    # Top row: Original footage, Detection view, Movement/Shadow mask
    original_resized = cv2.resize(original_frame, (panel_width, panel_height))
    detection_resized = cv2.resize(detection_frame, (panel_width, panel_height))
    
    # Convert foreground mask to 3-channel for display
    foreground_colored = cv2.applyColorMap(foreground_mask, cv2.COLORMAP_HOT)
    movement_resized = cv2.resize(foreground_colored, (panel_width, panel_height))
    
    # Bottom row: Satellite view, Terminal log, and a combined analysis view
    satellite_resized = cv2.resize(satellite_img, (panel_width, panel_height))
    terminal_panel = create_terminal_overlay(panel_width, panel_height, log_messages)
    
    # Create a combined analysis view (movement threshold visualization)
    fgthres = cv2.threshold(foreground_mask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
    analysis_colored = cv2.applyColorMap(fgthres, cv2.COLORMAP_VIRIDIS)
    analysis_resized = cv2.resize(analysis_colored, (panel_width, panel_height))
    
    # Add labels to each panel
    def add_panel_label(img, label, color=(255, 255, 255)):
        cv2.rectangle(img, (0, 0), (panel_width, 25), (0, 0, 0), -1)
        cv2.putText(img, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    add_panel_label(original_resized, "ORIGINAL FOOTAGE")
    add_panel_label(detection_resized, "DETECTION VIEW")
    add_panel_label(movement_resized, "MOVEMENT/SHADOW MASK", (255, 165, 0))  # Orange
    add_panel_label(satellite_resized, "SATELLITE VIEW")
    add_panel_label(terminal_panel, "SYSTEM LOG", (0, 255, 0))
    add_panel_label(analysis_resized, "THRESHOLD ANALYSIS", (255, 255, 0))  # Yellow
    
    # Place panels in dashboard (3x2 layout)
    # Top row
    # Top-left: Original footage
    dashboard[border:border+panel_height, border:border+panel_width] = original_resized
    
    # Top-center: Detection view  
    dashboard[border:border+panel_height, border*2+panel_width:border*2+panel_width*2] = detection_resized
    
    # Top-right: Movement/Shadow mask
    dashboard[border:border+panel_height, border*3+panel_width*2:border*3+panel_width*3] = movement_resized
    
    # Bottom row
    # Bottom-left: Satellite view
    dashboard[border*2+panel_height:border*2+panel_height*2, border:border+panel_width] = satellite_resized
    
    # Bottom-center: Terminal log
    dashboard[border*2+panel_height:border*2+panel_height*2, border*2+panel_width:border*2+panel_width*2] = terminal_panel
    
    # Bottom-right: Threshold analysis
    dashboard[border*2+panel_height:border*2+panel_height*2, border*3+panel_width*2:border*3+panel_width*3] = analysis_resized
    
    # Add main title and frame counter
    title_area_height = 50
    title_dashboard = np.zeros((title_area_height, dashboard_width, 3), dtype=np.uint8)
    cv2.putText(title_dashboard, f"STATIONARY DETECTOR DASHBOARD - Frame: {frame_count}", 
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add instruction text
    cv2.putText(title_dashboard, "Press 'q' to quit", 
                (dashboard_width - 150, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Combine title with dashboard
    final_dashboard = np.vstack([title_dashboard, dashboard])
    
    return final_dashboard


def main(file_to_process=None):
    """Main detection loop."""
    print("=" * 60)
    print("🎯 STATIONARY DETECTOR DASHBOARD")
    print("=" * 60)
    print("📹 Starting video processing...")
    print("⌨️  CONTROLS:")
    print("   • Press 'q' or 'Q' to quit")
    print("   • Press 'ESC' to quit")
    print("   • IMPORTANT: Click on the video window to focus it!")
    print("   • Make sure the video window is selected/active")
    print("=" * 60)
    
    # Runtime variables
    ITER = True
    SET_BACKGROUND = None
    CURRENT_FRAME = 0
    
    global id_bboxes
    id_bboxes = {}
    
    # Log messages for terminal display
    log_messages = [
        "System initialized",
        "Loading YOLO models...",
        "Starting detection loop..."
    ]
    
    # Analytics file setup
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(exist_ok=True)
    analytics_file = results_dir / ANALYTICS_FILE
    
    print(f"Configuration: FORCE_FRAME={FORCE_FRAME}, GPS_COORDINATES={GPS_COORDINATES}")
    print(f"Movement Threshold: {MOVEMENT_THRESHOLD}, Confidence: {CONFIDENCE_THRESHOLD}")
    
    # Initialize video capture
    if file_to_process is None:
        video_file = DEFAULT_VIDEO_FILE
    else:
        video_file = file_to_process
    
    if not Path(video_file).exists():
        print(f"Error: Video file {video_file} not found!")
        return
    
    video_cap = cv2.VideoCapture(video_file)
    creation_date_timestamp = get_video_info_hachoir(video_file)
    
    # Load YOLO models
    print("Loading YOLO models...")
    model_object_detection = YOLO("yolov8l.pt")
    
    # Initialize MOG background subtractor
    mog = cv2.createBackgroundSubtractorMOG2(
        history=MOG_HISTORY, 
        varThreshold=MOG_VAR_THRESHOLD, 
        detectShadows=MOG_DETECT_SHADOWS
    )
    
    print("Starting detection loop...")
    
    while ITER:
        # Load fresh satellite image each frame
        satellite_img = cv2.imread(SATELLITE_IMAGE)
        if satellite_img is None:
            print(f"Error: Could not load satellite image from {SATELLITE_IMAGE}")
            break
            
        satellite_width = satellite_img.shape[1]
        satellite_height = satellite_img.shape[0]
        
        ret, frame = video_cap.read()
        if not ret:
            print("End of video reached")
            break
        
        # Apply background subtraction
        fgmask = mog.apply(frame)
        
        # Frame sampling
        processed_frame, CURRENT_FRAME = frame_manipulation(video_cap, frame, CURRENT_FRAME, FORCE_FRAME)
        if processed_frame is None:
            continue
        
        # Set background frame
        if SET_BACKGROUND is None:
            background, SET_BACKGROUND = set_background(SET_BACKGROUND, frame)
        
        still_image = np.copy(background)
        
        # Run YOLO detection with tracking
        action_detection = model_object_detection.track(
            source=frame, 
            persist=True, 
            tracker="botsort.yaml", 
            imgsz=IMAGE_SIZE, 
            conf=CONFIDENCE_THRESHOLD, 
            classes=0,  # Person class only
            verbose=False
        )[0]
        
        # Add frame processing info to log
        if len(action_detection) > 0:
            detection_count = len([d for d in action_detection if d.boxes.id is not None])
            if detection_count > 0:
                log_messages.append(f"Frame {CURRENT_FRAME}: Detected {detection_count} person(s)")
        
        # Process each detection
        for detect in action_detection:
            if detect.boxes.id is None:
                continue
            
            # Draw detection box
            still_image = detect.plot(img=still_image, labels=False, boxes=True)
            
            # Calculate movement
            is_moving, white_pixel_percentage = calculate_movement_mog(frame, fgmask, detect)
            
            # Add to log
            person_id = int(detect.boxes.id)
            status = "MOVING" if is_moving else "STATIONARY"
            log_messages.append(f"ID {person_id}: {status} ({white_pixel_percentage:.2f})")
            
            # Create labels
            id_label = f"ID: {person_id}"
            moving_label = "moving" if is_moving else "stationary"
            moving_label_conf = f"State: {moving_label} {white_pixel_percentage:.2f}"
            
            labels = [moving_label_conf, id_label]
            still_image = draw_labels(still_image, labels=labels, 
                                    bbox=detect.boxes.xyxy.tolist(), 
                                    font=cv2.FONT_HERSHEY_SIMPLEX, 
                                    font_scale=1, thickness=2)
            
            # Camera calibration and BEV projection
            try:
                homography_matrix = calibrate_cam(frame, SCENARIO_FILE)
                satellite_img, centroid = display_bev(satellite_img, detect.boxes.xyxy.tolist(), homography_matrix)
                
                # GPS coordinates
                if GPS_COORDINATES:
                    corner_coordinates = np.array(GPS_CORNER_COORDINATES)
                    
                    x, y = centroid[0], centroid[1]
                    lon, lat = interpolate_gps((x, y), satellite_width, satellite_height, corner_coordinates)
                    
                    # Skip GPS processing if interpolation failed
                    if lon is None or lat is None:
                        log_messages.append(f"ID {person_id} GPS interpolation failed - skipping")
                        continue
                    
                    gps_msg = f"ID {person_id} GPS: ({lat:.6f}, {lon:.6f})"
                    log_messages.append(gps_msg)
                    print(gps_msg)
                    
                    # Track stationary positions
                    stationary_lon, stationary_lat = define_stationary_box(is_moving, detect.boxes.id, centroid, lon, lat)
                    
                    if stationary_lon is not None and stationary_lat is not None:
                        log_messages.append(f"ID {person_id} STATIONARY at: ({stationary_lat:.6f}, {stationary_lon:.6f})")
                    
                    # Generate timestamps
                    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    creating_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                    
                    # Log all detection data to CSV (like original BIP_ZEDCAM)
                    bbox_coords = detect.boxes.xyxy[0].cpu().numpy().tolist()
                    confidence = float(detect.boxes.conf[0].cpu().numpy())
                    detect_class = 0  # Person class
                    frame_duration = 1 / FORCE_FRAME
                    
                    log_detection_to_csv(
                        person_id=person_id,
                        bbox=bbox_coords,
                        confidence=confidence,
                        detect_class=detect_class,
                        is_moving=is_moving,
                        lat=lat,
                        lon=lon,
                        seconds_frame=frame_duration,
                        timestamp_current=current_timestamp,
                        creating_timestamp=creating_timestamp,
                        csv_file=DETAILED_CSV_FILE
                    )
                    
                    # Check if this is a validated stationary event (configurable consecutive frames)
                    is_newly_stationary = validate_stationary_frames(person_id, is_moving)
                    
                    # Use stationary coordinates for analytics if person is stationary, otherwise use current coordinates
                    analytics_lon = stationary_lon if stationary_lon is not None else lon
                    analytics_lat = stationary_lat if stationary_lat is not None else lat
                    
                    # Save analytics (simple format for backward compatibility)
                    save_analytics(int(detect.boxes.id), is_moving, FORCE_FRAME, analytics_file, 
                                 analytics_lon, analytics_lat, creating_timestamp, is_newly_stationary)
                    
            except Exception as e:
                print(f"Error in GPS/BEV processing: {e}")
        
        # Create dashboard with multiple panels
        dashboard = create_dashboard(frame, still_image, satellite_img, fgmask, CURRENT_FRAME, log_messages)
        cv2.imshow('Stationary Detector Dashboard', dashboard)
        
        # Handle user input - check for 'q' key immediately after showing frame
        # Use a longer wait time to ensure key detection works properly
        key = cv2.waitKey(100) & 0xFF
        
        # Check for quit keys
        if key == ord('q') or key == ord('Q') or key == 27:  # 'q', 'Q' or ESC to quit
            print("\n=== Program stopped by user (q pressed) ===")
            log_messages.append("USER REQUESTED EXIT - Shutting down...")
            cv2.destroyAllWindows()
            ITER = False
            break  # Exit the loop immediately
        elif key != 255:  # Any other key was pressed
            print(f"Key pressed: {key} (char: {chr(key)})")
        
        elif DEBUG_MODE:
            ITER = debug_mechanism(DEBUG_MODE, FRAME_INTERVAL, ITER)
            if not ITER:  # If debug mechanism returned False, break the loop
                break
    
    print("Detection completed!")
    
    # Cleanup resources
    try:
        video_cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Allow time for windows to close
    except:
        pass
    
    print("All resources cleaned up. Program ended.")


if __name__ == "__main__":
    main()