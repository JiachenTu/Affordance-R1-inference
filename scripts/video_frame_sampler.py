#!/usr/bin/env python3
"""
Video Frame Sampler for Affordance-R1
Samples evenly distributed frames from a video for affordance analysis
"""

import cv2
import os
import argparse
from pathlib import Path
import numpy as np
from datetime import datetime


def sample_frames_from_video(video_path, output_dir, num_frames=16, prefix="frame"):
    """
    Sample evenly distributed frames from a video
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save sampled frames
        num_frames: Number of frames to sample
        prefix: Prefix for saved frame filenames
    
    Returns:
        List of saved frame paths
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video Info:")
    print(f"   Path: {video_path}")
    print(f"   Total Frames: {total_frames}")
    print(f"   FPS: {fps:.2f}")
    print(f"   Duration: {duration:.2f} seconds")
    print(f"   Resolution: {width}x{height}")
    print()
    
    # Calculate frame indices to sample (evenly distributed)
    if total_frames <= num_frames:
        # If video has fewer frames than requested, take all frames
        frame_indices = list(range(total_frames))
        print(f"⚠️  Video has only {total_frames} frames, sampling all of them")
    else:
        # Sample evenly across the video
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        print(f"🎯 Sampling {num_frames} frames evenly distributed across video")
    
    saved_frames = []
    
    # Extract frames
    for i, frame_idx in enumerate(frame_indices):
        # Set video position to desired frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️  Could not read frame {frame_idx}")
            continue
        
        # Convert BGR to RGB for proper color
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save frame
        timestamp_sec = frame_idx / fps if fps > 0 else frame_idx
        frame_filename = f"{prefix}_{i+1:02d}_t{timestamp_sec:.2f}s.png"
        frame_path = output_dir / frame_filename
        
        # Convert back to BGR for OpenCV saving
        cv2.imwrite(str(frame_path), frame)
        saved_frames.append(frame_path)
        
        print(f"   ✅ Saved frame {i+1}/{len(frame_indices)}: {frame_filename} (frame #{frame_idx})")
    
    cap.release()
    
    print(f"\n🎉 Successfully sampled {len(saved_frames)} frames to: {output_dir}")
    return saved_frames


def main():
    parser = argparse.ArgumentParser(description="Sample frames from video for affordance analysis")
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save sampled frames")
    parser.add_argument("--num_frames", type=int, default=16,
                       help="Number of frames to sample (default: 16)")
    parser.add_argument("--prefix", type=str, default="frame",
                       help="Prefix for saved frame filenames")
    
    args = parser.parse_args()
    
    print("🎬 Video Frame Sampler for Affordance-R1")
    print("=" * 50)
    
    try:
        saved_frames = sample_frames_from_video(
            args.video_path, 
            args.output_dir, 
            args.num_frames, 
            args.prefix
        )
        
        print(f"\n📋 Summary:")
        print(f"   Input Video: {args.video_path}")
        print(f"   Output Directory: {args.output_dir}")
        print(f"   Frames Sampled: {len(saved_frames)}")
        print(f"   Frame Files:")
        for frame_path in saved_frames:
            print(f"     - {frame_path.name}")
            
        return saved_frames
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return []


if __name__ == "__main__":
    main()