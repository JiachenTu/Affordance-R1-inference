#!/usr/bin/env python3
"""
Video Frame Affordance Analysis
Runs affordance prediction on all frames from a video
"""

import os
import subprocess
from pathlib import Path
from datetime import datetime
import json


def run_video_frame_affordance_analysis(frames_dir, results_dir, question="What actions can be performed with objects in this image?"):
    """
    Run affordance analysis on all video frames
    
    Args:
        frames_dir: Directory containing sampled frames
        results_dir: Directory to save results  
        question: Affordance question to ask
    """
    
    frames_dir = Path(frames_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all frame files
    frame_files = sorted(frames_dir.glob("*.png"))
    
    if not frame_files:
        print(f"❌ No PNG files found in {frames_dir}")
        return
    
    print(f"🎬 Video Frame Affordance Analysis")
    print(f"=" * 60)
    print(f"Frames Directory: {frames_dir}")
    print(f"Results Directory: {results_dir}")
    print(f"Question: {question}")
    print(f"Total Frames: {len(frame_files)}")
    print()
    
    script_path = "/home/jtu9/reasoning/Affordance-R1-inference/scripts/single_inference.py"
    batch_results = {
        "timestamp": datetime.now().isoformat(),
        "frames_dir": str(frames_dir),
        "results_dir": str(results_dir), 
        "question": question,
        "total_frames": len(frame_files),
        "results": {},
        "summary": {}
    }
    
    successful = 0
    failed = 0
    total_start_time = datetime.now()
    
    for i, frame_file in enumerate(frame_files, 1):
        print(f"🔍 {i}/{len(frame_files)}: Processing {frame_file.name}")
        
        frame_start_time = datetime.now()
        
        try:
            # Run single inference
            cmd = [
                "python", script_path,
                "--image_path", str(frame_file),
                "--question", question,
                "--output_dir", str(results_dir)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd="/home/jtu9/reasoning/Affordance-R1-inference",
                timeout=120  # 2 minute timeout per frame
            )
            
            frame_time = (datetime.now() - frame_start_time).total_seconds()
            
            if result.returncode == 0:
                print(f"    ✅ Completed in {frame_time:.2f}s")
                successful += 1
                batch_results["results"][frame_file.name] = {
                    "status": "success",
                    "processing_time": frame_time
                }
            else:
                print(f"    ❌ Failed with return code {result.returncode}")
                print(f"    Error: {result.stderr}")
                failed += 1
                batch_results["results"][frame_file.name] = {
                    "status": "failed",
                    "error": result.stderr,
                    "return_code": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            print(f"    ⏰ Timeout after 2 minutes")
            failed += 1
            batch_results["results"][frame_file.name] = {
                "status": "timeout",
                "error": "Processing timeout after 2 minutes"
            }
            
        except Exception as e:
            print(f"    ❌ Exception: {str(e)}")
            failed += 1
            batch_results["results"][frame_file.name] = {
                "status": "error", 
                "error": str(e)
            }
        
        print()
    
    total_time = (datetime.now() - total_start_time).total_seconds()
    
    # Generate summary
    batch_results["summary"] = {
        "total_processing_time": total_time,
        "successful_frames": successful,
        "failed_frames": failed,
        "success_rate": successful / len(frame_files) * 100 if frame_files else 0,
        "average_time_per_frame": total_time / len(frame_files) if frame_files else 0
    }
    
    # Save batch results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_results_path = results_dir / f"video_frame_batch_results_{timestamp}.json"
    
    with open(batch_results_path, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    # Print final summary
    print("📊 VIDEO FRAME AFFORDANCE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total Frames: {len(frame_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {successful/len(frame_files)*100:.1f}%")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Time per Frame: {total_time/len(frame_files):.2f} seconds")
    print(f"\nBatch results saved to: {batch_results_path}")
    
    return batch_results_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run affordance analysis on video frames")
    parser.add_argument("--frames_dir", type=str, required=True,
                       help="Directory containing sampled video frames")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory to save affordance results")
    parser.add_argument("--question", type=str, 
                       default="What actions can be performed with objects in this image?",
                       help="Affordance question to ask")
    
    args = parser.parse_args()
    
    run_video_frame_affordance_analysis(args.frames_dir, args.results_dir, args.question)


if __name__ == "__main__":
    main()