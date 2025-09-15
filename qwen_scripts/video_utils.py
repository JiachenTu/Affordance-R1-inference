#!/usr/bin/env python3
"""
Video Processing Utilities for Qwen2.5-VL
Helper functions for video analysis, frame extraction, and result processing
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Dict, Tuple, Optional


def get_video_info(video_path: str) -> Dict:
    """
    Extract comprehensive video metadata

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    info = {
        'path': video_path,
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
        'file_size': os.path.getsize(video_path)
    }

    info['duration_seconds'] = info['total_frames'] / info['fps'] if info['fps'] > 0 else 0
    info['duration_formatted'] = format_duration(info['duration_seconds'])

    cap.release()
    return info


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes}:{secs:05.2f}"


def extract_frames_at_timestamps(video_path: str, timestamps: List[float],
                                output_dir: str = None) -> List[str]:
    """
    Extract frames at specific timestamps

    Args:
        video_path: Path to video file
        timestamps: List of timestamps in seconds
        output_dir: Directory to save frames (optional)

    Returns:
        List of saved frame paths
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    video_name = Path(video_path).stem

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    saved_frames = []

    for i, timestamp in enumerate(timestamps):
        # Convert timestamp to frame number
        frame_number = int(timestamp * fps)

        # Set video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        # Read frame
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️  Could not read frame at {timestamp}s")
            continue

        if output_dir:
            # Save frame
            frame_filename = f"{video_name}_t{timestamp:.2f}s_frame{i+1:03d}.png"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            saved_frames.append(str(frame_path))
            print(f"   ✅ Saved frame at {timestamp:.2f}s: {frame_filename}")
        else:
            saved_frames.append(frame)

    cap.release()
    return saved_frames


def create_timeline_visualization(timestamps: List[Dict], video_duration: float,
                                output_path: str = None) -> str:
    """
    Create a timeline visualization of detected events

    Args:
        timestamps: List of timestamp dictionaries with 'timestamp_seconds' and descriptions
        video_duration: Total video duration in seconds
        output_path: Path to save visualization

    Returns:
        Path to saved visualization
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Draw timeline
    ax.hlines(y=0, xmin=0, xmax=video_duration, linewidth=3, color='lightgray')

    # Add timestamp markers
    for i, ts in enumerate(timestamps):
        time_sec = ts['timestamp_seconds']

        # Add vertical line for timestamp
        ax.vlines(x=time_sec, ymin=-0.1, ymax=0.1, linewidth=2, color='red')

        # Add text annotation
        text = ts.get('original_text', f'{time_sec:.1f}s')
        ax.text(time_sec, 0.15 + (i % 3) * 0.1, text,
               rotation=45, ha='left', va='bottom', fontsize=9)

    # Format timeline
    ax.set_xlim(-video_duration * 0.02, video_duration * 1.02)
    ax.set_ylim(-0.2, 0.5)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_title(f'Video Timeline - {len(timestamps)} Events Detected', fontsize=14)

    # Add time ticks
    num_ticks = min(20, int(video_duration // 10) + 1)
    tick_positions = np.linspace(0, video_duration, num_ticks)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([format_duration(t) for t in tick_positions], rotation=45)

    # Remove y-axis
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path
    else:
        plt.show()
        return None


def parse_grounding_results(results_file: str) -> Dict:
    """
    Parse results from grounding analysis

    Args:
        results_file: Path to JSON results file

    Returns:
        Parsed results dictionary
    """
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Extract and sort timestamps
    timestamps = results.get('extracted_timestamps', [])
    timestamps.sort(key=lambda x: x['timestamp_seconds'])

    # Calculate statistics
    stats = {
        'total_timestamps': len(timestamps),
        'video_duration': None,  # Would need video info
        'timestamp_density': None,
        'time_range': {
            'earliest': min(ts['timestamp_seconds'] for ts in timestamps) if timestamps else 0,
            'latest': max(ts['timestamp_seconds'] for ts in timestamps) if timestamps else 0
        }
    }

    return {
        'metadata': {
            'video_name': results.get('video_name'),
            'task': results.get('task'),
            'processing_time': results.get('processing_time_seconds')
        },
        'timestamps': timestamps,
        'statistics': stats,
        'raw_output': results.get('raw_output', '')
    }


def validate_video_file(video_path: str) -> bool:
    """
    Validate if video file is readable and get basic info

    Args:
        video_path: Path to video file

    Returns:
        True if valid, raises exception if not
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Check if video has frames
    ret, frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError(f"Video file appears to be empty or corrupted: {video_path}")

    cap.release()
    return True


def export_timestamps_to_formats(timestamps: List[Dict], output_dir: str,
                                base_name: str) -> Dict[str, str]:
    """
    Export timestamps to multiple formats (JSON, CSV, SRT)

    Args:
        timestamps: List of timestamp dictionaries
        output_dir: Directory to save exports
        base_name: Base name for output files

    Returns:
        Dictionary with paths to exported files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_files = {}

    # Export to JSON
    json_path = output_dir / f"{base_name}_timestamps.json"
    with open(json_path, 'w') as f:
        json.dump(timestamps, f, indent=2)
    exported_files['json'] = str(json_path)

    # Export to CSV
    csv_path = output_dir / f"{base_name}_timestamps.csv"
    with open(csv_path, 'w') as f:
        f.write("timestamp_seconds,original_text,context\n")
        for ts in timestamps:
            # Escape quotes and newlines for CSV
            context = ts.get('context', '').replace('"', '""').replace('\n', ' ')
            original = ts.get('original_text', '').replace('"', '""')
            f.write(f"{ts['timestamp_seconds']},\"{original}\",\"{context}\"\n")
    exported_files['csv'] = str(csv_path)

    # Export to SRT subtitle format
    srt_path = output_dir / f"{base_name}_timestamps.srt"
    with open(srt_path, 'w') as f:
        for i, ts in enumerate(timestamps, 1):
            start_time = format_srt_time(ts['timestamp_seconds'])
            end_time = format_srt_time(ts['timestamp_seconds'] + 3)  # 3 second duration
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            event_text = ts.get('original_text', f'Event at {ts["timestamp_seconds"]:.1f}s')
            f.write(f"{event_text}\n\n")
    exported_files['srt'] = str(srt_path)

    return exported_files


def format_srt_time(seconds: float) -> str:
    """Format time for SRT subtitle format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millisecs = int((secs % 1) * 1000)
    secs = int(secs)

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def print_video_summary(video_path: str):
    """Print formatted video information summary"""
    try:
        info = get_video_info(video_path)
        print(f"📹 Video Information:")
        print(f"   Path: {info['path']}")
        print(f"   Duration: {info['duration_formatted']} ({info['duration_seconds']:.1f}s)")
        print(f"   Resolution: {info['width']}x{info['height']}")
        print(f"   FPS: {info['fps']:.2f}")
        print(f"   Total Frames: {info['total_frames']:,}")
        print(f"   File Size: {info['file_size'] / (1024*1024):.1f} MB")
        print()
    except Exception as e:
        print(f"❌ Error reading video info: {e}")


if __name__ == "__main__":
    # Example usage
    video_path = "/shared/BIOE486/SP25/users/ukd1/misc/testset_samples/421372/42445448/42445448.mp4"

    if Path(video_path).exists():
        print_video_summary(video_path)
    else:
        print("Video file not found for testing")