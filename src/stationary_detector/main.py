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
from pathlib import Path
from datetime import datetime, timedelta
from ultralytics import YOLO
from scipy.interpolate import LinearNDInterpolator
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata


# Global dictionary to track person IDs and their stationary state
id_bboxes = {}


def interpolate_gps(point, satellite_width, satellite_height, corner_coordinates):
    """Convert image coordinates to GPS using interpolation."""
    x = np.array([0, 0, satellite_width, satellite_width])
    y = np.array([0, satellite_height, satellite_height, 0])
    
    latlon = LinearNDInterpolator((x, y), corner_coordinates)
    latpoint = latlon(point[0], point[1])
    lat = latpoint[0]
    lon = latpoint[1]
    
    return lon, lat


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
    
    is_moving = white_pixel_percentage > 0.18
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
    """Handle keyboard input for debugging."""
    if debug_mode:
        key = cv2.waitKey(0)
        if key == ord(' '):
            return iter_flag
        if key == ord('q'):
            cv2.destroyAllWindows()
            return False
    
    key = cv2.waitKey(frame_interval) & 0xFF
    if key == ord('q'):
        cv2.destroyAllWindows()
        return False
    return iter_flag


def save_analytics(user_id, is_moving, force_frame, analytics_file):
    """Simple analytics tracking."""
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
    """Create a 2x2 dashboard showing all views."""
    if log_messages is None:
        log_messages = []
    
    # Define panel dimensions
    panel_width = 640
    panel_height = 480
    border = 2
    
    # Create dashboard canvas
    dashboard_width = (panel_width * 2) + (border * 3)
    dashboard_height = (panel_height * 2) + (border * 3)
    dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
    
    # Resize all panels to standard size
    # Top-left: Original footage
    original_resized = cv2.resize(original_frame, (panel_width, panel_height))
    
    # Top-right: Detection view
    detection_resized = cv2.resize(detection_frame, (panel_width, panel_height))
    
    # Bottom-left: Satellite view
    satellite_resized = cv2.resize(satellite_img, (panel_width, panel_height))
    
    # Bottom-right: Terminal log
    terminal_panel = create_terminal_overlay(panel_width, panel_height, log_messages)
    
    # Add labels to each panel
    def add_panel_label(img, label, color=(255, 255, 255)):
        cv2.rectangle(img, (0, 0), (panel_width, 30), (0, 0, 0), -1)
        cv2.putText(img, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    add_panel_label(original_resized, "ORIGINAL FOOTAGE")
    add_panel_label(detection_resized, "DETECTION VIEW")
    add_panel_label(satellite_resized, "SATELLITE VIEW")
    add_panel_label(terminal_panel, "SYSTEM LOG", (0, 255, 0))
    
    # Place panels in dashboard
    # Top-left: Original footage
    dashboard[border:border+panel_height, border:border+panel_width] = original_resized
    
    # Top-right: Detection view  
    dashboard[border:border+panel_height, border*2+panel_width:border*2+panel_width*2] = detection_resized
    
    # Bottom-left: Satellite view
    dashboard[border*2+panel_height:border*2+panel_height*2, border:border+panel_width] = satellite_resized
    
    # Bottom-right: Terminal log
    dashboard[border*2+panel_height:border*2+panel_height*2, border*2+panel_width:border*2+panel_width*2] = terminal_panel
    
    # Add main title and frame counter
    title_area_height = 50
    title_dashboard = np.zeros((title_area_height, dashboard_width, 3), dtype=np.uint8)
    cv2.putText(title_dashboard, f"STATIONARY DETECTOR DASHBOARD - Frame: {frame_count}", 
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Combine title with dashboard
    final_dashboard = np.vstack([title_dashboard, dashboard])
    
    return final_dashboard


def main(file_to_process=None):
    """Main detection loop."""
    print("Running Stationary Detector... Press 'q' to quit")
    
    # Configuration
    ITER = True
    SET_BACKGROUND = None
    FRAME_INTERVAL = 10
    DEBUG_MODE = False
    CURRENT_FRAME = 0
    FORCE_FRAME = 4  # 4 frames per second
    GPS_COORDINATES = True
    
    global id_bboxes
    id_bboxes = {}
    
    # Log messages for terminal display
    log_messages = [
        "System initialized",
        "Loading YOLO models...",
        "Starting detection loop..."
    ]
    
    # Resource paths
    scenario_field = "resources/camera_parameters/scenario6_campo_ourique_secundaria.json"
    satellite_source = "resources/satellite_images/campo_ourique_portao_secundaria.jpeg"
    bev_width = 600
    bev_height = 600
    
    # Analytics file
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    analytics_file = results_dir / "analytics.json"
    
    print(f"Configuration: FORCE_FRAME={FORCE_FRAME}, GPS_COORDINATES={GPS_COORDINATES}")
    
    # Initialize video capture
    if file_to_process is None:
        video_file = "resources/videos/2.mp4"
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
    mog = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=8, detectShadows=True)
    
    print("Starting detection loop...")
    
    while ITER:
        # Load fresh satellite image each frame
        satellite_img = cv2.imread(satellite_source)
        if satellite_img is None:
            print(f"Error: Could not load satellite image from {satellite_source}")
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
            imgsz=1920, 
            conf=0.20, 
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
                homography_matrix = calibrate_cam(frame, scenario_field)
                satellite_img, centroid = display_bev(satellite_img, detect.boxes.xyxy.tolist(), homography_matrix)
                
                # GPS coordinates
                if GPS_COORDINATES:
                    corner_coordinates = np.array([
                        [38.7174781, -9.1702299],  # top-left
                        [38.7171197, -9.1702293],  # bottom-left
                        [38.7171223, -9.1696365],  # bottom-right
                        [38.7174791, -9.1696378]   # top-right
                    ])
                    
                    x, y = centroid[0], centroid[1]
                    lon, lat = interpolate_gps((x, y), satellite_width, satellite_height, corner_coordinates)
                    gps_msg = f"ID {person_id} GPS: ({lat:.6f}, {lon:.6f})"
                    log_messages.append(gps_msg)
                    print(gps_msg)
                    
                    # Track stationary positions
                    stationary_lon, stationary_lat = define_stationary_box(is_moving, detect.boxes.id, centroid, lon, lat)
                    
                    if stationary_lon is not None and stationary_lat is not None:
                        log_messages.append(f"ID {person_id} STATIONARY at: ({stationary_lat:.6f}, {stationary_lon:.6f})")
                    
                    # Save analytics
                    save_analytics(int(detect.boxes.id), is_moving, FORCE_FRAME, analytics_file)
                    
            except Exception as e:
                print(f"Error in GPS/BEV processing: {e}")
        
        # Create dashboard with multiple panels
        dashboard = create_dashboard(frame, still_image, satellite_img, fgmask, CURRENT_FRAME, log_messages)
        cv2.imshow('Stationary Detector Dashboard', dashboard)
        
        # Handle user input
        ITER = debug_mechanism(DEBUG_MODE, FRAME_INTERVAL, ITER)
    
    print("Detection completed!")
    video_cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()