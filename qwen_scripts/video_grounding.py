#!/usr/bin/env python3
"""
Video Grounding Specialized Script for Qwen2.5-VL
Advanced temporal grounding with specific question types and analysis
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

from video_utils import (
    get_video_info, create_timeline_visualization,
    export_timestamps_to_formats, print_video_summary
)


class VideoGroundingAnalyzer:
    """Specialized analyzer for video grounding tasks"""

    def __init__(self, model_path: str, gpu_id: str = "7"):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.grounding_questions = self._get_grounding_question_templates()

    def _get_grounding_question_templates(self) -> Dict[str, str]:
        """Predefined grounding question templates"""
        return {
            'action_detection': "When do specific actions or activities occur in this video? Please provide precise timestamps for when each action starts and ends.",
            'object_interaction': "At what times do people interact with objects in this video? Identify when objects are picked up, used, or manipulated.",
            'movement_tracking': "When does movement or motion occur in this video? Track when people or objects move and provide timestamps.",
            'scene_changes': "When do scene changes or transitions occur in this video? Identify timestamps for different scenes or camera angles.",
            'speech_events': "When do people speak or make sounds in this video? Provide timestamps for audio events.",
            'temporal_sequence': "What is the temporal sequence of events in this video? List all major events with their timestamps in chronological order.",
            'specific_moment': "When does [SPECIFIC_EVENT] happen in this video? Provide the exact timestamp.",
            'duration_analysis': "How long does each activity or event last in this video? Provide start and end timestamps for each event.",
            'periodic_events': "Are there any repeating or periodic events in this video? When do they occur?",
            'comparative_timing': "Compare the timing of different events in this video. Which events happen first, simultaneously, or in sequence?"
        }

    def generate_grounding_prompt(self, question_type: str, specific_event: str = None) -> str:
        """Generate specific grounding prompt"""
        base_prompt = self.grounding_questions.get(question_type, self.grounding_questions['temporal_sequence'])

        if specific_event and '[SPECIFIC_EVENT]' in base_prompt:
            base_prompt = base_prompt.replace('[SPECIFIC_EVENT]', specific_event)

        # Add temporal precision instructions
        temporal_instructions = (
            "\n\nImportant: Please be as precise as possible with timestamps. "
            "Use formats like '1:23' (1 minute 23 seconds), '2.5 seconds', "
            "or 'at 45 seconds' to indicate when events occur. "
            "If an event has duration, specify both start and end times."
        )

        return base_prompt + temporal_instructions

    def run_grounding_analysis(self, video_path: str, question_type: str,
                             specific_event: str = None, output_dir: str = None) -> Dict:
        """Run video grounding analysis"""
        from qwen_video_inference import main as run_inference
        import sys
        import tempfile

        # Generate grounding prompt
        prompt = self.generate_grounding_prompt(question_type, specific_event)

        # Create temporary args for inference script
        if not output_dir:
            output_dir = "/home/jtu9/reasoning/Affordance-R1-inference/qwen_scripts/results/grounding"

        # Backup original sys.argv
        original_argv = sys.argv.copy()

        try:
            # Set up arguments for inference script
            sys.argv = [
                'qwen_video_inference.py',
                '--video_path', video_path,
                '--task', 'grounding',
                '--question', prompt,
                '--output_dir', output_dir,
                '--model_path', self.model_path,
                '--gpu_id', self.gpu_id,
                '--fps', '1.0',
                '--max_pixels', str(360*420)
            ]

            # Run inference
            run_inference()

            # Find the most recent result file
            results_dir = Path(output_dir) / 'grounding'
            if results_dir.exists():
                json_files = list(results_dir.glob("*_grounding_*_results.json"))
                if json_files:
                    # Get most recent file
                    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_file, 'r') as f:
                        results = json.load(f)
                    return results

        finally:
            # Restore original sys.argv
            sys.argv = original_argv

        return {}

    def analyze_temporal_patterns(self, timestamps: List[Dict]) -> Dict:
        """Analyze temporal patterns in extracted timestamps"""
        if not timestamps:
            return {}

        times = [ts['timestamp_seconds'] for ts in timestamps]
        times.sort()

        analysis = {
            'total_events': len(timestamps),
            'time_span': {
                'start': min(times),
                'end': max(times),
                'duration': max(times) - min(times)
            },
            'intervals': [],
            'event_density': len(timestamps) / (max(times) - min(times)) if len(times) > 1 else 0
        }

        # Calculate intervals between events
        if len(times) > 1:
            intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
            analysis['intervals'] = {
                'mean': sum(intervals) / len(intervals),
                'min': min(intervals),
                'max': max(intervals),
                'all_intervals': intervals
            }

            # Detect potential patterns
            # Check for regular intervals (periodicity)
            if len(intervals) >= 3:
                # Simple periodicity check - look for similar intervals
                avg_interval = analysis['intervals']['mean']
                tolerance = avg_interval * 0.3  # 30% tolerance

                regular_intervals = [
                    interval for interval in intervals
                    if abs(interval - avg_interval) <= tolerance
                ]

                if len(regular_intervals) >= len(intervals) * 0.6:  # 60% of intervals are regular
                    analysis['periodicity'] = {
                        'detected': True,
                        'average_period': avg_interval,
                        'confidence': len(regular_intervals) / len(intervals)
                    }
                else:
                    analysis['periodicity'] = {'detected': False}

        return analysis

    def create_detailed_timeline(self, video_path: str, timestamps: List[Dict],
                               output_path: str = None) -> str:
        """Create detailed timeline with video context"""
        video_info = get_video_info(video_path)
        video_duration = video_info['duration_seconds']

        if not output_path:
            video_name = Path(video_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"/tmp/{video_name}_timeline_{timestamp}.png"

        return create_timeline_visualization(timestamps, video_duration, output_path)

    def export_grounding_results(self, results: Dict, output_dir: str) -> Dict[str, str]:
        """Export grounding results in multiple formats"""
        if not results or 'extracted_timestamps' not in results:
            return {}

        timestamps = results['extracted_timestamps']
        video_name = Path(results.get('video_name', 'video')).stem

        # Create grounding-specific output directory
        grounding_dir = Path(output_dir) / 'grounding_exports'
        grounding_dir.mkdir(parents=True, exist_ok=True)

        # Export timestamps
        exported = export_timestamps_to_formats(timestamps, grounding_dir, video_name)

        # Add grounding-specific analysis
        temporal_analysis = self.analyze_temporal_patterns(timestamps)
        analysis_path = grounding_dir / f"{video_name}_temporal_analysis.json"

        with open(analysis_path, 'w') as f:
            json.dump({
                'video_metadata': results.get('video_name'),
                'grounding_task': results.get('task'),
                'temporal_analysis': temporal_analysis,
                'raw_results': results
            }, f, indent=2)

        exported['temporal_analysis'] = str(analysis_path)
        return exported


def main():
    parser = argparse.ArgumentParser(description="Advanced video grounding with Qwen2.5-VL")
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--question_type", type=str,
                       choices=[
                           'action_detection', 'object_interaction', 'movement_tracking',
                           'scene_changes', 'speech_events', 'temporal_sequence',
                           'specific_moment', 'duration_analysis', 'periodic_events',
                           'comparative_timing'
                       ],
                       default='temporal_sequence',
                       help="Type of grounding question")
    parser.add_argument("--specific_event", type=str,
                       help="Specific event to search for (for specific_moment questions)")
    parser.add_argument("--model_path", type=str,
                       default="/home/jtu9/reasoning/models/Qwen2.5-VL-7B-Instruct",
                       help="Path to Qwen2.5-VL model")
    parser.add_argument("--output_dir", type=str,
                       default="/home/jtu9/reasoning/Affordance-R1-inference/qwen_scripts/results",
                       help="Output directory for results")
    parser.add_argument("--gpu_id", type=str, default="7",
                       help="GPU ID to use")
    parser.add_argument("--create_timeline", action="store_true",
                       help="Create timeline visualization")
    parser.add_argument("--export_formats", action="store_true",
                       help="Export results in multiple formats")

    args = parser.parse_args()

    print(f"🎯 Video Grounding Analysis")
    print(f"{'=' * 60}")
    print(f"Video: {args.video_path}")
    print(f"Question Type: {args.question_type}")
    if args.specific_event:
        print(f"Specific Event: {args.specific_event}")
    print()

    # Validate video
    print_video_summary(args.video_path)

    # Initialize analyzer
    analyzer = VideoGroundingAnalyzer(args.model_path, args.gpu_id)

    # Run grounding analysis
    print("Running grounding analysis...")
    results = analyzer.run_grounding_analysis(
        args.video_path,
        args.question_type,
        args.specific_event,
        args.output_dir
    )

    if results:
        timestamps = results.get('extracted_timestamps', [])
        print(f"✅ Analysis complete! Found {len(timestamps)} temporal events")

        # Analyze temporal patterns
        if timestamps:
            temporal_analysis = analyzer.analyze_temporal_patterns(timestamps)
            print(f"\nTemporal Pattern Analysis:")
            print(f"  Event Count: {temporal_analysis.get('total_events', 0)}")
            print(f"  Time Span: {temporal_analysis.get('time_span', {}).get('duration', 0):.1f} seconds")
            print(f"  Event Density: {temporal_analysis.get('event_density', 0):.3f} events/second")

            if 'periodicity' in temporal_analysis:
                periodicity = temporal_analysis['periodicity']
                if periodicity.get('detected'):
                    print(f"  🔄 Periodic Pattern Detected!")
                    print(f"    Average Period: {periodicity.get('average_period', 0):.1f}s")
                    print(f"    Confidence: {periodicity.get('confidence', 0):.1%}")

        # Create timeline visualization
        if args.create_timeline and timestamps:
            print("\nCreating timeline visualization...")
            timeline_path = analyzer.create_detailed_timeline(args.video_path, timestamps)
            print(f"  Timeline saved: {timeline_path}")

        # Export results
        if args.export_formats:
            print("\nExporting results in multiple formats...")
            exported = analyzer.export_grounding_results(results, args.output_dir)
            for format_type, path in exported.items():
                print(f"  {format_type.upper()}: {path}")

        print(f"\n{'=' * 60}")
        print("GROUNDING ANALYSIS COMPLETE")
        print(f"{'=' * 60}")

    else:
        print("❌ Analysis failed or no results generated")


if __name__ == "__main__":
    main()