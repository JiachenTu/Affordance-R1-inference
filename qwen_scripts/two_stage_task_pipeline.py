#!/usr/bin/env python3
"""
Two-Stage Task Pipeline - Complete End-to-End Processing
Orchestrates all three stages: Frame extraction, VLM validation, and Bbox/mask generation
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import subprocess

from enhanced_task_frame_extractor import EnhancedTaskFrameExtractor
from vlm_frame_validator import VLMFrameValidator
from task_bbox_processor import TaskBboxProcessor
from video_utils import get_video_info, format_duration


class TwoStageTaskPipeline:
    """Complete two-stage task-specific frame grounding and localization pipeline"""

    def __init__(self, qwen_model_path: str, sam_model_path: str = "facebook/sam2-hiera-large",
                 stage1_gpu: str = "1", stage2_gpu: str = "2", stage3_gpu: str = "3"):
        self.qwen_model_path = qwen_model_path
        self.sam_model_path = sam_model_path
        self.stage1_gpu = stage1_gpu
        self.stage2_gpu = stage2_gpu
        self.stage3_gpu = stage3_gpu

        # Initialize stage components
        self.extractor = EnhancedTaskFrameExtractor(qwen_model_path, stage1_gpu)
        self.validator = VLMFrameValidator(qwen_model_path, stage2_gpu)
        self.processor = TaskBboxProcessor(qwen_model_path, sam_model_path, stage3_gpu)

    def run_complete_pipeline(self, video_path: str, output_dir: str,
                            max_frames_per_task: int = 16, top_k_per_task: int = 3,
                            selected_tasks: Optional[List[str]] = None) -> Dict:
        """Run the complete two-stage pipeline"""

        print(f"🚀 Two-Stage Task-Specific Frame Grounding Pipeline")
        print(f"{'=' * 80}")

        # Validate inputs
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Get video information
        video_info = get_video_info(video_path)
        video_name = Path(video_path).stem

        print(f"📹 Video: {video_path}")
        print(f"   Duration: {format_duration(video_info['duration_seconds'])}")
        print(f"   Resolution: {video_info['width']}x{video_info['height']}")
        print(f"   FPS: {video_info['fps']:.2f}")
        print()

        print(f"⚙️  Pipeline Configuration:")
        print(f"   Max Frames per Task (Stage 1): {max_frames_per_task}")
        print(f"   Top K per Task (Stage 2): {top_k_per_task}")
        print(f"   GPU Allocation: Stage1={self.stage1_gpu}, Stage2={self.stage2_gpu}, Stage3={self.stage3_gpu}")
        if selected_tasks:
            print(f"   Selected Tasks: {', '.join(selected_tasks)}")
        else:
            print(f"   Processing: All 11 predefined tasks")
        print()

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pipeline_start_time = datetime.now()

        # Initialize pipeline results
        pipeline_results = {
            'pipeline_start_time': pipeline_start_time.isoformat(),
            'video_path': video_path,
            'video_info': video_info,
            'output_dir': str(output_dir),
            'configuration': {
                'max_frames_per_task': max_frames_per_task,
                'top_k_per_task': top_k_per_task,
                'selected_tasks': selected_tasks,
                'gpu_allocation': {
                    'stage1': self.stage1_gpu,
                    'stage2': self.stage2_gpu,
                    'stage3': self.stage3_gpu
                }
            },
            'stages': {}
        }

        try:
            # STAGE 1: High-Recall Frame Extraction
            print(f"🎯 STAGE 1: High-Recall Frame Extraction")
            print(f"{'=' * 60}")

            stage1_start = datetime.now()
            stage1_metadata_path = self.extractor.run_stage1_extraction(
                video_path=video_path,
                output_dir=str(output_dir),
                max_frames_per_task=max_frames_per_task
            )
            stage1_time = (datetime.now() - stage1_start).total_seconds()

            pipeline_results['stages']['stage1'] = {
                'status': 'completed',
                'metadata_path': stage1_metadata_path,
                'processing_time': stage1_time,
                'completion_time': datetime.now().isoformat()
            }

            print(f"✅ Stage 1 completed in {stage1_time:.1f} seconds")
            print()

            # STAGE 2: High-Accuracy VLM Validation
            print(f"🔍 STAGE 2: High-Accuracy VLM Validation")
            print(f"{'=' * 60}")

            stage2_start = datetime.now()
            stage2_metadata_path = self.validator.run_stage2_validation(
                stage1_metadata_path=stage1_metadata_path,
                output_dir=str(output_dir),
                top_k_per_task=top_k_per_task
            )
            stage2_time = (datetime.now() - stage2_start).total_seconds()

            pipeline_results['stages']['stage2'] = {
                'status': 'completed',
                'metadata_path': stage2_metadata_path,
                'processing_time': stage2_time,
                'completion_time': datetime.now().isoformat()
            }

            print(f"✅ Stage 2 completed in {stage2_time:.1f} seconds")
            print()

            # STAGE 3: Bbox/Point Extraction and Mask Generation
            print(f"🎨 STAGE 3: Bbox/Point Extraction and Mask Generation")
            print(f"{'=' * 60}")

            stage3_start = datetime.now()
            stage3_metadata_path = self.processor.run_stage3_processing(
                stage2_metadata_path=stage2_metadata_path,
                output_dir=str(output_dir)
            )
            stage3_time = (datetime.now() - stage3_start).total_seconds()

            pipeline_results['stages']['stage3'] = {
                'status': 'completed',
                'metadata_path': stage3_metadata_path,
                'processing_time': stage3_time,
                'completion_time': datetime.now().isoformat()
            }

            print(f"✅ Stage 3 completed in {stage3_time:.1f} seconds")
            print()

        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()

            # Record failure
            current_stage = len(pipeline_results['stages']) + 1
            pipeline_results['stages'][f'stage{current_stage}'] = {
                'status': 'failed',
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            }

            raise

        # Calculate total pipeline time
        pipeline_end_time = datetime.now()
        total_pipeline_time = (pipeline_end_time - pipeline_start_time).total_seconds()

        pipeline_results['pipeline_end_time'] = pipeline_end_time.isoformat()
        pipeline_results['total_processing_time'] = total_pipeline_time

        # Create comprehensive final report
        final_report_path = self.create_final_report(pipeline_results, output_dir, video_name)
        pipeline_results['final_report_path'] = final_report_path

        # Print final summary
        self.print_final_summary(pipeline_results)

        return pipeline_results

    def create_final_report(self, pipeline_results: Dict, output_dir: Path, video_name: str) -> str:
        """Create comprehensive final pipeline report"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"two_stage_pipeline_report_{video_name}_{timestamp}.md"

        with open(report_path, 'w') as f:
            f.write("# Two-Stage Task-Specific Frame Grounding - Final Report\n\n")

            # Video Information
            video_info = pipeline_results['video_info']
            f.write("## 📹 Video Information\n\n")
            f.write(f"- **Path**: `{pipeline_results['video_path']}`\n")
            f.write(f"- **Duration**: {format_duration(video_info['duration_seconds'])}\n")
            f.write(f"- **Resolution**: {video_info['width']}x{video_info['height']}\n")
            f.write(f"- **FPS**: {video_info['fps']:.2f}\n")
            f.write(f"- **Total Frames**: {video_info['total_frames']:,}\n")
            f.write(f"- **File Size**: {video_info['file_size'] / (1024*1024):.1f} MB\n\n")

            # Pipeline Configuration
            config = pipeline_results['configuration']
            f.write("## ⚙️ Pipeline Configuration\n\n")
            f.write(f"- **Max Frames per Task**: {config['max_frames_per_task']}\n")
            f.write(f"- **Top K per Task**: {config['top_k_per_task']}\n")
            f.write(f"- **GPU Allocation**: Stage1={config['gpu_allocation']['stage1']}, ")
            f.write(f"Stage2={config['gpu_allocation']['stage2']}, Stage3={config['gpu_allocation']['stage3']}\n")
            if config['selected_tasks']:
                f.write(f"- **Selected Tasks**: {', '.join(config['selected_tasks'])}\n")
            else:
                f.write(f"- **Tasks**: All 11 predefined tasks\n")
            f.write(f"\n")

            # Stage Results
            f.write("## 🔄 Stage Processing Results\n\n")

            stages = pipeline_results.get('stages', {})

            for stage_name, stage_info in stages.items():
                stage_num = stage_name.replace('stage', '')
                stage_titles = {
                    '1': 'High-Recall Frame Extraction',
                    '2': 'VLM Frame Validation',
                    '3': 'Bbox/Mask Generation'
                }

                f.write(f"### Stage {stage_num}: {stage_titles.get(stage_num, 'Unknown')}\n")
                f.write(f"- **Status**: {stage_info['status'].title()}\n")

                if stage_info['status'] == 'completed':
                    f.write(f"- **Processing Time**: {stage_info['processing_time']:.1f} seconds\n")
                    f.write(f"- **Metadata**: `{stage_info['metadata_path']}`\n")
                elif stage_info['status'] == 'failed':
                    f.write(f"- **Error**: {stage_info.get('error', 'Unknown error')}\n")

                f.write(f"- **Completion Time**: {stage_info['completion_time']}\n\n")

            # Performance Summary
            f.write("## 📊 Performance Summary\n\n")
            f.write(f"- **Total Pipeline Time**: {pipeline_results.get('total_processing_time', 0):.1f} seconds\n")

            if 'stage1' in stages and stages['stage1']['status'] == 'completed':
                f.write(f"- **Stage 1 Time**: {stages['stage1']['processing_time']:.1f}s\n")
            if 'stage2' in stages and stages['stage2']['status'] == 'completed':
                f.write(f"- **Stage 2 Time**: {stages['stage2']['processing_time']:.1f}s\n")
            if 'stage3' in stages and stages['stage3']['status'] == 'completed':
                f.write(f"- **Stage 3 Time**: {stages['stage3']['processing_time']:.1f}s\n")

            successful_stages = len([s for s in stages.values() if s['status'] == 'completed'])
            f.write(f"- **Successful Stages**: {successful_stages}/3\n")
            f.write(f"- **Pipeline Success Rate**: {successful_stages/3*100:.1f}%\n\n")

            # Output Structure
            f.write("## 📁 Output Structure\n\n")
            f.write("```\n")
            f.write(f"{pipeline_results['output_dir']}/\n")
            f.write("├── stage1_candidates/          # All extracted candidate frames\n")
            f.write("│   ├── cabinet_operations/\n")
            f.write("│   ├── tv_controls/\n")
            f.write("│   ├── room_environment/\n")
            f.write("│   └── other_actions/\n")
            f.write("├── stage2_selected/            # Top 3 validated frames per task\n")
            f.write("│   └── ... (same structure)\n")
            f.write("├── bbox_results/              # Bbox/point extraction results\n")
            f.write("│   └── ... (same structure)\n")
            f.write("├── final_masks/               # Final visualizations with masks\n")
            f.write("│   └── ... (same structure)\n")
            f.write("├── stage1_extraction_*.json   # Stage 1 metadata\n")
            f.write("├── stage2_validation_*.json   # Stage 2 metadata\n")
            f.write("├── stage3_bbox_processing_*.json  # Stage 3 metadata\n")
            f.write("└── two_stage_pipeline_report_*.md  # This report\n")
            f.write("```\n\n")

            # Usage Instructions
            f.write("## 🚀 Results Usage\n\n")
            f.write("1. **Review Final Visualizations**: Check `final_masks/` directory\n")
            f.write("2. **Examine Bbox Results**: Review JSON files in `bbox_results/`\n")
            f.write("3. **Validate Frame Selection**: Check `stage2_selected/` for top frames\n")
            f.write("4. **Analyze Pipeline Performance**: Review stage metadata files\n\n")

            # Recommendations
            f.write("## 💡 Recommendations\n\n")

            if successful_stages == 3:
                f.write("✅ **Pipeline completed successfully!**\n")
                f.write("- All stages completed without errors\n")
                f.write("- Review final visualizations for quality assessment\n")
                f.write("- Use bbox results for downstream applications\n")
            else:
                f.write("⚠️  **Pipeline completed with issues:**\n")
                if successful_stages < 3:
                    f.write("- Some stages failed - check error messages above\n")
                    f.write("- Consider adjusting GPU allocation or model paths\n")
                    f.write("- Retry failed stages individually\n")

        return str(report_path)

    def print_final_summary(self, pipeline_results: Dict):
        """Print comprehensive final summary"""

        print(f"\n{'🎉 TWO-STAGE PIPELINE COMPLETE 🎉':^80}")
        print(f"{'=' * 80}")

        # Performance metrics
        total_time = pipeline_results.get('total_processing_time', 0)
        stages = pipeline_results.get('stages', {})
        successful_stages = len([s for s in stages.values() if s['status'] == 'completed'])

        print(f"📊 PERFORMANCE SUMMARY")
        print(f"   Total Processing Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"   Successful Stages: {successful_stages}/3")
        print(f"   Pipeline Success Rate: {successful_stages/3*100:.1f}%")
        print()

        if successful_stages == 3:
            print(f"✅ ALL STAGES COMPLETED SUCCESSFULLY!")
        else:
            print(f"⚠️  PIPELINE COMPLETED WITH {3-successful_stages} FAILED STAGE(S)")

        print()

        # Stage breakdown
        if 'stage1' in stages:
            status = stages['stage1']['status']
            time_str = f" ({stages['stage1']['processing_time']:.1f}s)" if status == 'completed' else ""
            print(f"   Stage 1 (Frame Extraction): {status.upper()}{time_str}")

        if 'stage2' in stages:
            status = stages['stage2']['status']
            time_str = f" ({stages['stage2']['processing_time']:.1f}s)" if status == 'completed' else ""
            print(f"   Stage 2 (VLM Validation): {status.upper()}{time_str}")

        if 'stage3' in stages:
            status = stages['stage3']['status']
            time_str = f" ({stages['stage3']['processing_time']:.1f}s)" if status == 'completed' else ""
            print(f"   Stage 3 (Bbox/Mask Generation): {status.upper()}{time_str}")

        print()

        # Output locations
        print(f"📁 OUTPUT LOCATIONS")
        print(f"   Results Directory: {pipeline_results['output_dir']}")
        if 'final_report_path' in pipeline_results:
            print(f"   Final Report: {pipeline_results['final_report_path']}")

        if successful_stages >= 2:
            print(f"   Frame Candidates: {pipeline_results['output_dir']}/stage1_candidates/")
        if successful_stages >= 3:
            print(f"   Selected Frames: {pipeline_results['output_dir']}/stage2_selected/")
            print(f"   Final Visualizations: {pipeline_results['output_dir']}/final_masks/")

        print(f"{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="Complete two-stage task-specific frame grounding pipeline")
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--output_dir", type=str,
                       default="/home/jtu9/reasoning/Affordance-R1-inference/qwen_scripts/results/enhanced_task_grounding",
                       help="Output directory for results")
    parser.add_argument("--qwen_model_path", type=str,
                       default="/home/jtu9/reasoning/models/Qwen2.5-VL-7B-Instruct",
                       help="Path to Qwen2.5-VL model")
    parser.add_argument("--sam_model_path", type=str,
                       default="facebook/sam2-hiera-large",
                       help="Path to SAM2 model")
    parser.add_argument("--max_frames", type=int, default=16,
                       help="Maximum frames to extract per task in Stage 1")
    parser.add_argument("--top_k", type=int, default=3,
                       help="Top K frames to select per task in Stage 2")
    parser.add_argument("--tasks", type=str, nargs='+',
                       choices=[
                           'cabinet_top_left', 'cabinet_top_right', 'cabinet_bottom_left', 'cabinet_bottom_right',
                           'tv_remote_footrest', 'tv_remote_table', 'radiator_thermostat', 'window_above_radiator',
                           'ceiling_light', 'door_close', 'power_outlet'
                       ],
                       help="Specific tasks to process (default: all)")
    parser.add_argument("--stage1_gpu", type=str, default="1",
                       help="GPU ID for Stage 1")
    parser.add_argument("--stage2_gpu", type=str, default="2",
                       help="GPU ID for Stage 2")
    parser.add_argument("--stage3_gpu", type=str, default="3",
                       help="GPU ID for Stage 3")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = TwoStageTaskPipeline(
        qwen_model_path=args.qwen_model_path,
        sam_model_path=args.sam_model_path,
        stage1_gpu=args.stage1_gpu,
        stage2_gpu=args.stage2_gpu,
        stage3_gpu=args.stage3_gpu
    )

    try:
        # Run complete pipeline
        results = pipeline.run_complete_pipeline(
            video_path=args.video_path,
            output_dir=args.output_dir,
            max_frames_per_task=args.max_frames,
            top_k_per_task=args.top_k,
            selected_tasks=args.tasks
        )

        print(f"\n📋 Pipeline complete! Check the final report for detailed results.")

    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()