#!/usr/bin/env python3
"""
Batch Video Processing Pipeline for Qwen2.5-VL
Comprehensive video analysis with multiple tasks and question types
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

from video_utils import get_video_info, print_video_summary, create_timeline_visualization


class BatchVideoPipeline:
    """Batch processing pipeline for comprehensive video analysis"""

    def __init__(self, model_path: str, gpu_id: str = "7", max_workers: int = 1):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.max_workers = max_workers
        self.script_dir = Path(__file__).parent
        self.results = {}

    def get_analysis_tasks(self) -> Dict[str, Dict]:
        """Define comprehensive analysis tasks"""
        return {
            'video_summary': {
                'task': 'summarization',
                'description': 'Overall video content summarization',
                'question': None
            },
            'dense_captioning': {
                'task': 'captioning',
                'description': 'Detailed frame-by-frame captioning',
                'question': None
            },
            'action_grounding': {
                'task': 'grounding',
                'description': 'Temporal grounding of actions and activities',
                'question': 'When do specific actions or activities occur in this video? Provide precise timestamps for when each action starts and ends.'
            },
            'object_interaction': {
                'task': 'grounding',
                'description': 'When objects are interacted with',
                'question': 'At what times do people interact with objects in this video? Identify when objects are picked up, used, or manipulated.'
            },
            'movement_tracking': {
                'task': 'grounding',
                'description': 'Movement and motion detection',
                'question': 'When does movement or motion occur in this video? Track when people or objects move and provide timestamps.'
            },
            'scene_analysis': {
                'task': 'grounding',
                'description': 'Scene changes and transitions',
                'question': 'When do scene changes or transitions occur in this video? Identify timestamps for different scenes or camera angles.'
            },
            'temporal_sequence': {
                'task': 'grounding',
                'description': 'Complete temporal event sequence',
                'question': 'What is the temporal sequence of events in this video? List all major events with their timestamps in chronological order.'
            }
        }

    def run_single_task(self, video_path: str, task_name: str, task_config: Dict,
                       output_dir: str, timeout: int = 300) -> Dict:
        """Run a single analysis task"""
        print(f"🔍 Running {task_name}...")

        try:
            # Prepare command
            script_path = self.script_dir / "qwen_video_inference.py"
            cmd = [
                "python", str(script_path),
                "--video_path", video_path,
                "--task", task_config['task'],
                "--model_path", self.model_path,
                "--output_dir", output_dir,
                "--gpu_id", self.gpu_id,
                "--max_pixels", str(360*420),
                "--fps", "1.0"
            ]

            # Add custom question if specified
            if task_config.get('question'):
                cmd.extend(["--question", task_config['question']])

            # Run task
            start_time = datetime.now()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.script_dir)
            )
            end_time = datetime.now()

            processing_time = (end_time - start_time).total_seconds()

            if result.returncode == 0:
                print(f"  ✅ {task_name} completed in {processing_time:.1f}s")

                # Try to find and load the result file
                task_result = self._load_task_result(output_dir, task_config['task'], video_path)

                return {
                    'task_name': task_name,
                    'status': 'success',
                    'processing_time': processing_time,
                    'description': task_config['description'],
                    'result_data': task_result,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                print(f"  ❌ {task_name} failed with return code {result.returncode}")
                return {
                    'task_name': task_name,
                    'status': 'failed',
                    'processing_time': processing_time,
                    'description': task_config['description'],
                    'error': result.stderr,
                    'return_code': result.returncode,
                    'stdout': result.stdout
                }

        except subprocess.TimeoutExpired:
            print(f"  ⏰ {task_name} timed out after {timeout}s")
            return {
                'task_name': task_name,
                'status': 'timeout',
                'processing_time': timeout,
                'description': task_config['description'],
                'error': f"Task timed out after {timeout} seconds"
            }

        except Exception as e:
            print(f"  ❌ {task_name} failed with exception: {str(e)}")
            return {
                'task_name': task_name,
                'status': 'error',
                'processing_time': 0,
                'description': task_config['description'],
                'error': str(e)
            }

    def _load_task_result(self, output_dir: str, task_type: str, video_path: str) -> Dict:
        """Load the most recent result file for a task"""
        try:
            results_dir = Path(output_dir) / task_type
            if not results_dir.exists():
                return {}

            # Find most recent JSON result file
            json_files = list(results_dir.glob(f"*_{task_type}_*_results.json"))
            if not json_files:
                return {}

            latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
            with open(latest_file, 'r') as f:
                return json.load(f)

        except Exception as e:
            print(f"    ⚠️  Could not load result for {task_type}: {e}")
            return {}

    def run_batch_analysis(self, video_path: str, output_dir: str,
                          selected_tasks: List[str] = None,
                          parallel: bool = False,
                          timeout: int = 300) -> Dict:
        """Run comprehensive batch analysis"""

        # Validate video
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Get video info
        video_info = get_video_info(video_path)
        video_name = Path(video_path).stem

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get tasks to run
        all_tasks = self.get_analysis_tasks()
        if selected_tasks:
            tasks_to_run = {k: v for k, v in all_tasks.items() if k in selected_tasks}
        else:
            tasks_to_run = all_tasks

        print(f"🎬 Batch Video Analysis Pipeline")
        print(f"{'=' * 60}")
        print(f"Video: {video_path}")
        print(f"Output: {output_dir}")
        print(f"Tasks: {', '.join(tasks_to_run.keys())}")
        print(f"Parallel: {parallel}")
        print()

        print_video_summary(video_path)

        # Initialize batch results
        batch_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'video_path': video_path,
                'video_info': video_info,
                'output_dir': str(output_dir),
                'model_path': self.model_path,
                'total_tasks': len(tasks_to_run),
                'parallel_processing': parallel
            },
            'task_results': {},
            'summary': {}
        }

        start_time = datetime.now()

        # Run tasks
        if parallel and self.max_workers > 1:
            # Parallel execution
            print(f"Running {len(tasks_to_run)} tasks in parallel...")
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(tasks_to_run))) as executor:
                future_to_task = {
                    executor.submit(
                        self.run_single_task, video_path, task_name, task_config,
                        str(output_dir), timeout
                    ): task_name
                    for task_name, task_config in tasks_to_run.items()
                }

                for future in as_completed(future_to_task):
                    task_name = future_to_task[future]
                    try:
                        result = future.result()
                        batch_results['task_results'][task_name] = result
                    except Exception as e:
                        print(f"❌ Task {task_name} generated an exception: {e}")
                        batch_results['task_results'][task_name] = {
                            'task_name': task_name,
                            'status': 'error',
                            'error': str(e)
                        }
        else:
            # Sequential execution
            print(f"Running {len(tasks_to_run)} tasks sequentially...")
            for task_name, task_config in tasks_to_run.items():
                result = self.run_single_task(
                    video_path, task_name, task_config, str(output_dir), timeout
                )
                batch_results['task_results'][task_name] = result

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Generate summary
        successful_tasks = [
            name for name, result in batch_results['task_results'].items()
            if result.get('status') == 'success'
        ]
        failed_tasks = [
            name for name, result in batch_results['task_results'].items()
            if result.get('status') != 'success'
        ]

        batch_results['summary'] = {
            'total_processing_time': total_time,
            'successful_tasks': len(successful_tasks),
            'failed_tasks': len(failed_tasks),
            'success_rate': len(successful_tasks) / len(tasks_to_run) * 100,
            'successful_task_names': successful_tasks,
            'failed_task_names': failed_tasks,
            'average_task_time': total_time / len(tasks_to_run) if tasks_to_run else 0
        }

        # Save batch results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_results_path = output_dir / f"{video_name}_batch_analysis_{timestamp}.json"

        with open(batch_results_path, 'w') as f:
            json.dump(batch_results, f, indent=2)

        # Create comprehensive report
        self._generate_comprehensive_report(batch_results, output_dir, video_name)

        # Print summary
        self._print_batch_summary(batch_results)

        return batch_results

    def _generate_comprehensive_report(self, batch_results: Dict, output_dir: Path, video_name: str):
        """Generate a comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"{video_name}_comprehensive_report_{timestamp}.md"

        with open(report_path, 'w') as f:
            f.write(f"# Comprehensive Video Analysis Report\n\n")
            f.write(f"**Video:** {batch_results['metadata']['video_path']}\n")
            f.write(f"**Analysis Date:** {batch_results['metadata']['timestamp']}\n")
            f.write(f"**Model:** {batch_results['metadata']['model_path']}\n\n")

            # Video Information
            video_info = batch_results['metadata']['video_info']
            f.write(f"## Video Information\n\n")
            f.write(f"- **Duration:** {video_info.get('duration_formatted', 'N/A')}\n")
            f.write(f"- **Resolution:** {video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}\n")
            f.write(f"- **FPS:** {video_info.get('fps', 'N/A'):.2f}\n")
            f.write(f"- **Total Frames:** {video_info.get('total_frames', 'N/A'):,}\n")
            f.write(f"- **File Size:** {video_info.get('file_size', 0) / (1024*1024):.1f} MB\n\n")

            # Task Results
            f.write(f"## Analysis Results\n\n")
            for task_name, result in batch_results['task_results'].items():
                f.write(f"### {task_name.replace('_', ' ').title()}\n\n")
                f.write(f"**Status:** {result.get('status', 'unknown').title()}\n")
                f.write(f"**Processing Time:** {result.get('processing_time', 0):.1f}s\n")
                f.write(f"**Description:** {result.get('description', 'N/A')}\n\n")

                if result.get('status') == 'success' and result.get('result_data'):
                    result_data = result['result_data']
                    if 'raw_output' in result_data:
                        output_preview = result_data['raw_output'][:300] + "..." if len(result_data['raw_output']) > 300 else result_data['raw_output']
                        f.write(f"**Output Preview:**\n```\n{output_preview}\n```\n\n")

                    if 'extracted_timestamps' in result_data:
                        timestamps = result_data['extracted_timestamps']
                        f.write(f"**Timestamps Found:** {len(timestamps)}\n")
                        if timestamps:
                            f.write(f"**Sample Timestamps:**\n")
                            for ts in timestamps[:3]:  # Show first 3
                                f.write(f"- {ts['timestamp_seconds']:.1f}s: {ts.get('original_text', 'Event')}\n")
                            if len(timestamps) > 3:
                                f.write(f"- ... and {len(timestamps) - 3} more\n")
                        f.write("\n")

                elif result.get('status') != 'success':
                    f.write(f"**Error:** {result.get('error', 'Unknown error')}\n\n")

            # Summary
            summary = batch_results['summary']
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Tasks:** {batch_results['metadata']['total_tasks']}\n")
            f.write(f"- **Successful:** {summary['successful_tasks']}\n")
            f.write(f"- **Failed:** {summary['failed_tasks']}\n")
            f.write(f"- **Success Rate:** {summary['success_rate']:.1f}%\n")
            f.write(f"- **Total Processing Time:** {summary['total_processing_time']:.1f}s\n")
            f.write(f"- **Average Task Time:** {summary['average_task_time']:.1f}s\n")

        print(f"📄 Comprehensive report saved: {report_path}")

    def _print_batch_summary(self, batch_results: Dict):
        """Print formatted batch summary"""
        summary = batch_results['summary']

        print(f"\n📊 BATCH ANALYSIS SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total Tasks: {batch_results['metadata']['total_tasks']}")
        print(f"Successful: {summary['successful_tasks']}")
        print(f"Failed: {summary['failed_tasks']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Total Time: {summary['total_processing_time']:.1f} seconds")
        print(f"Average Time per Task: {summary['average_task_time']:.1f} seconds")

        if summary['successful_task_names']:
            print(f"\n✅ Successful Tasks:")
            for task in summary['successful_task_names']:
                print(f"  - {task}")

        if summary['failed_task_names']:
            print(f"\n❌ Failed Tasks:")
            for task in summary['failed_task_names']:
                print(f"  - {task}")

        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Batch video processing pipeline")
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--output_dir", type=str,
                       default="/home/jtu9/reasoning/Affordance-R1-inference/qwen_scripts/results",
                       help="Output directory for results")
    parser.add_argument("--model_path", type=str,
                       default="/home/jtu9/reasoning/models/Qwen2.5-VL-7B-Instruct",
                       help="Path to Qwen2.5-VL model")
    parser.add_argument("--tasks", type=str, nargs='+',
                       choices=[
                           'video_summary', 'dense_captioning', 'action_grounding',
                           'object_interaction', 'movement_tracking', 'scene_analysis',
                           'temporal_sequence'
                       ],
                       help="Specific tasks to run (default: all)")
    parser.add_argument("--parallel", action="store_true",
                       help="Run tasks in parallel")
    parser.add_argument("--max_workers", type=int, default=1,
                       help="Maximum number of parallel workers")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout per task in seconds")
    parser.add_argument("--gpu_id", type=str, default="7",
                       help="GPU ID to use")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = BatchVideoPipeline(
        model_path=args.model_path,
        gpu_id=args.gpu_id,
        max_workers=args.max_workers
    )

    # Run batch analysis
    try:
        results = pipeline.run_batch_analysis(
            video_path=args.video_path,
            output_dir=args.output_dir,
            selected_tasks=args.tasks,
            parallel=args.parallel,
            timeout=args.timeout
        )

        print(f"\n🎉 Batch analysis complete!")

    except Exception as e:
        print(f"❌ Batch analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()