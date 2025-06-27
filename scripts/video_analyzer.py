import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import cv2

def load_tracking_data(filename='tracking_results.json'):
    """Load tracking results"""
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Tracking results file not found: {filename}")
        return None

def analyze_player_trajectories(tracking_data):
    """Analyze and visualize player trajectories"""
    if not tracking_data:
        return
    
    # Extract trajectories
    trajectories = defaultdict(list)
    
    for frame_num, frame_data in tracking_data['frames'].items():
        frame_num = int(frame_num)
        for obj_id, obj_data in frame_data['tracked_objects'].items():
            obj_id = int(obj_id)
            centroid = obj_data['centroid']
            trajectories[obj_id].append((frame_num, centroid[0], centroid[1]))
    
    # Plot trajectories
    plt.figure(figsize=(15, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(trajectories)))
    
    for i, (obj_id, trajectory) in enumerate(trajectories.items()):
        if len(trajectory) < 5:  # Skip very short tracks
            continue
        
        frames, x_coords, y_coords = zip(*trajectory)
        
        plt.plot(x_coords, y_coords, color=colors[i], linewidth=2, 
                label=f'Player {obj_id} ({len(trajectory)} frames)', alpha=0.7)
        
        # Mark start and end points
        plt.scatter(x_coords[0], y_coords[0], color=colors[i], s=100, marker='o', edgecolor='black')
        plt.scatter(x_coords[-1], y_coords[-1], color=colors[i], s=100, marker='s', edgecolor='black')
    
    plt.xlabel('X Coordinate (pixels)')
    plt.ylabel('Y Coordinate (pixels)')
    plt.title('Player Trajectories\n(Circles: Start, Squares: End)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # Invert Y axis to match image coordinates
    plt.tight_layout()
    plt.savefig('player_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Player trajectories saved: player_trajectories.png")

def analyze_player_timeline(tracking_data):
    """Create timeline showing when each player appears"""
    if not tracking_data:
        return
    
    # Extract appearance timeline
    player_frames = defaultdict(list)
    
    for frame_num, frame_data in tracking_data['frames'].items():
        frame_num = int(frame_num)
        for obj_id in frame_data['tracked_objects'].keys():
            obj_id = int(obj_id)
            player_frames[obj_id].append(frame_num)
    
    # Create timeline plot
    plt.figure(figsize=(15, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(player_frames)))
    
    for i, (obj_id, frames) in enumerate(player_frames.items()):
        # Create segments for continuous appearances
        segments = []
        current_segment = [frames[0]]
        
        for j in range(1, len(frames)):
            if frames[j] - frames[j-1] <= 2:  # Continuous if gap <= 2 frames
                current_segment.append(frames[j])
            else:
                segments.append(current_segment)
                current_segment = [frames[j]]
        segments.append(current_segment)
        
        # Plot segments
        for segment in segments:
            plt.barh(i, len(segment), left=segment[0], height=0.6, 
                    color=colors[i], alpha=0.7, edgecolor='black')
        
        # Add player label
        plt.text(-20, i, f'Player {obj_id}', va='center', ha='right')
    
    plt.xlabel('Frame Number')
    plt.ylabel('Player ID')
    plt.title('Player Appearance Timeline\n(Gaps indicate re-identification events)')
    plt.grid(True, alpha=0.3, axis='x')
    plt.xlim(0, max([max(frames) for frames in player_frames.values()]) + 10)
    plt.tight_layout()
    plt.savefig('player_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Player timeline saved: player_timeline.png")

def analyze_detection_confidence(tracking_data):
    """Analyze detection confidence over time"""
    if not tracking_data:
        return
    
    frame_numbers = []
    detection_counts = []
    
    for frame_num, frame_data in tracking_data['frames'].items():
        frame_numbers.append(int(frame_num))
        detection_counts.append(frame_data['detections'])
    
    plt.figure(figsize=(12, 6))
    plt.plot(frame_numbers, detection_counts, linewidth=2, color='blue')
    plt.fill_between(frame_numbers, detection_counts, alpha=0.3, color='blue')
    
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Detections')
    plt.title('Detection Count Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('detection_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Detection timeline saved: detection_timeline.png")

def generate_analysis_report(tracking_data):
    """Generate detailed analysis report"""
    if not tracking_data:
        return
    
    stats = tracking_data.get('statistics', {})
    video_info = tracking_data.get('video_info', {})
    
    # Calculate additional metrics
    total_frames = len(tracking_data['frames'])
    player_visibility = defaultdict(int)
    
    for frame_data in tracking_data['frames'].values():
        for obj_id in frame_data['tracked_objects'].keys():
            player_visibility[int(obj_id)] += 1
    
    # Generate report
    report = f"""
PLAYER RE-IDENTIFICATION ANALYSIS REPORT
========================================

Video Information:
- Resolution: {video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}
- FPS: {video_info.get('fps', 'N/A')}
- Total Frames: {video_info.get('total_frames', 'N/A')}
- Processed Frames: {total_frames}

Tracking Statistics:
- Total Players Detected: {stats.get('total_tracks', 'N/A')}
- Active Players at End: {stats.get('active_tracks', 'N/A')}
- Re-identification Events: {stats.get('reidentification_events', 'N/A')}

Player Visibility Analysis:
"""
    
    for player_id, visibility_count in sorted(player_visibility.items()):
        visibility_percentage = (visibility_count / total_frames) * 100
        report += f"- Player {player_id}: {visibility_count}/{total_frames} frames ({visibility_percentage:.1f}%)\n"
    
    report += f"""
Performance Metrics:
- Average Track Length: {np.mean(list(stats.get('track_lengths', {1: 1}).values())):.1f} frames
- Tracking Continuity: {(sum(player_visibility.values()) / (len(player_visibility) * total_frames)) * 100:.1f}%

Re-identification Success:
- Players with gaps (re-identified): {sum(1 for v in player_visibility.values() if v < total_frames * 0.8)}
- Continuous tracking rate: {(sum(1 for v in player_visibility.values() if v >= total_frames * 0.8) / len(player_visibility)) * 100:.1f}%

Analysis Complete: {len(player_visibility)} unique players tracked across {total_frames} frames
"""
    
    # Save report
    with open('analysis_report.txt', 'w') as f:
        f.write(report)
    
    print("✓ Analysis report saved: analysis_report.txt")
    print("\nKey Findings:")
    print(f"  - {len(player_visibility)} unique players detected")
    print(f"  - {stats.get('reidentification_events', 0)} re-identification events")
    print(f"  - Average visibility: {np.mean(list(player_visibility.values())):.1f} frames per player")

def main():
    """Main analysis function"""
    print("Loading tracking results...")
    tracking_data = load_tracking_data()
    
    if tracking_data is None:
        print("Please run the tracker first: python scripts/main_tracker.py")
        return
    
    print("Generating analysis...")
    
    # Generate visualizations
    analyze_player_trajectories(tracking_data)
    analyze_player_timeline(tracking_data)
    analyze_detection_confidence(tracking_data)
    
    # Generate report
    generate_analysis_report(tracking_data)
    
    print("\n✓ Analysis complete! Check the generated files:")
    print("  - player_trajectories.png")
    print("  - player_timeline.png") 
    print("  - detection_timeline.png")
    print("  - analysis_report.txt")

if __name__ == "__main__":
    main()
