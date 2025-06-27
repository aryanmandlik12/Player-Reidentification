import cv2
import numpy as np
from ultralytics import YOLO
import json
import argparse
from pathlib import Path
import time
from player_tracker import PlayerTracker

def load_model(model_path):
    """Load YOLOv11 model"""
    try:
        model = YOLO(model_path)
        print(f"✓ Model loaded from {model_path}")
        return model
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None

def process_detections(results, confidence_threshold=0.3):
    """Process YOLO detection results"""
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence >= confidence_threshold:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Get class (assuming class 0 is player)
                    class_id = int(box.cls[0])
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id
                    }
                    detections.append(detection)
    
    return detections

def draw_tracking_results(frame, tracked_objects, frame_num):
    """Draw tracking results on frame"""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 0), (128, 128, 0),
        (0, 0, 128), (128, 0, 0), (0, 128, 128), (192, 192, 192), (255, 20, 147)
    ]
    
    for object_id, obj_data in tracked_objects.items():
        bbox = obj_data['bbox']
        centroid = obj_data['centroid']
        
        # Get color for this ID
        color = colors[object_id % len(colors)]
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw centroid
        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, color, -1)
        
        # Draw ID label
        label = f"Player {object_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Draw frame number
    cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame

def main():
    parser = argparse.ArgumentParser(description='Player Re-identification Tracker')
    parser.add_argument('--input', default='15sec_input_720p.mp4', help='Input video path')
    parser.add_argument('--output', default='tracked_output.mp4', help='Output video path')
    parser.add_argument('--model', default='yolov11_players.pt', help='Model path')
    parser.add_argument('--confidence', type=float, default=0.3, help='Detection confidence threshold')
    
    args = parser.parse_args()
    
    # Check if input video exists
    if not Path(args.input).exists():
        print(f"✗ Input video not found: {args.input}")
        print("Please ensure the video file exists in the project directory")
        return
    
    # Load model
    model = load_model(args.model)
    if model is None:
        return
    
    # Initialize tracker
    tracker = PlayerTracker(
        max_disappeared=30,
        max_distance=100,
        similarity_threshold=0.3
    )
    
    # Open video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"✗ Failed to open video: {args.input}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('tracked_output.avi', fourcc, fps, (width, height))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Tracking data storage
    tracking_data = {
        'video_info': {
            'width': width,
            'height': height,
            'fps': fps,
            'total_frames': total_frames
        },
        'frames': {}
    }
    
    frame_num = 0
    start_time = time.time()
    
    print("Starting tracking...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, verbose=False)
        detections = process_detections(results, args.confidence)
        
        # Filter only player detections (assuming class 0 is player)
        player_detections = [d for d in detections if d['class_id'] == 0]
        
        # Update tracker
        tracked_objects = tracker.update(player_detections, frame, frame_num)
        
        # Draw results
        annotated_frame = draw_tracking_results(frame.copy(), tracked_objects, frame_num)
        
        # Write frame
        out.write(annotated_frame)
        
        # Store tracking data
        frame_data = {
            'detections': len(player_detections),
            'tracked_objects': {}
        }
        
        for obj_id, obj_data in tracked_objects.items():
            frame_data['tracked_objects'][obj_id] = {
                'bbox': obj_data['bbox'],
                'centroid': obj_data['centroid'],
                'track_length': obj_data['track_length']
            }
        
        tracking_data['frames'][frame_num] = frame_data
        
        # Progress update
        if frame_num % 30 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}% - Frame {frame_num}/{total_frames}")
        
        frame_num += 1
    
    # Cleanup
    cap.release()
    out.release()
    
    # Get tracking statistics
    stats = tracker.get_track_statistics()
    tracking_data['statistics'] = stats
    
    # Save tracking data
    with open('tracking_results.json', 'w') as f:
        json.dump(tracking_data, f, indent=2, default=str)
    
    # Print results
    elapsed_time = time.time() - start_time
    print(f"\n✓ Tracking completed in {elapsed_time:.2f} seconds")
    print(f"✓ Output video saved: {args.output}")
    print(f"✓ Tracking data saved: tracking_results.json")
    print(f"\nTracking Statistics:")
    print(f"  Total players tracked: {stats['total_tracks']}")
    print(f"  Active tracks at end: {stats['active_tracks']}")
    print(f"  Re-identification events: {stats['reidentification_events']}")
    print(f"  Average track length: {np.mean(list(stats['track_lengths'].values())):.1f} frames")

if __name__ == "__main__":
    main()
