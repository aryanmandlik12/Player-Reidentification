#!/bin/bash

echo "=== Player Re-identification System ==="
echo "Setting up environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Setup environment and download model
echo "Installing dependencies..."
python scripts/setup_environment.py

# Check if input video exists
if [ ! -f "15sec_input_720p.mp4" ]; then
    echo "⚠️  Input video '15sec_input_720p.mp4' not found!"
    echo "Please place your 15-second video in the project root directory."
    echo "You can also specify a different video:"
    echo "  python scripts/main_tracker.py --input your_video.mp4"
    exit 1
fi

# Run the tracker
echo "Starting player tracking..."
python scripts/main_tracker.py

# Run analysis
echo "Generating analysis..."
python scripts/video_analyzer.py

echo ""
echo "✅ Processing complete!"
echo "Check the following output files:"
echo "  📹 tracked_output.mp4 - Annotated video with player IDs"
echo "  📊 tracking_results.json - Raw tracking data"
echo "  📈 player_trajectories.png - Player movement paths"
echo "  📅 player_timeline.png - Appearance timeline"
echo "  📋 analysis_report.txt - Detailed analysis report"
