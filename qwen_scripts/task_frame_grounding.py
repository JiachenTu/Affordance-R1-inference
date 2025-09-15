#!/usr/bin/env python3
"""
Task-Specific Frame Grounding for Qwen2.5-VL
Identifies best frames for specific tasks and extracts them with structured mapping
"""

import argparse
import json
import os
import cv2
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import numpy as np
from PIL import Image as PILImage
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from video_utils import get_video_info, format_duration


class TaskFrameGroundingAnalyzer:
    """Specialized analyzer for task-specific frame grounding"""

    def __init__(self, model_path: str, gpu_id: str = "1"):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.model = None
        self.processor = None
        self.tasks = self._get_predefined_tasks()

    def _get_predefined_tasks(self) -> Dict[str, Dict]:
        """Define the specific tasks for frame grounding"""
        return {
            'cabinet_top_left': {
                'category': 'Cabinet Operations',
                'description': 'Open the top left cabinet door',
                'prompt_keywords': ['top left cabinet', 'upper left cabinet door', 'cabinet door top left']
            },
            'cabinet_top_right': {
                'category': 'Cabinet Operations',
                'description': 'Open the top right cabinet door',
                'prompt_keywords': ['top right cabinet', 'upper right cabinet door', 'cabinet door top right']
            },
            'cabinet_bottom_left': {
                'category': 'Cabinet Operations',
                'description': 'Open the bottom left cabinet door',
                'prompt_keywords': ['bottom left cabinet', 'lower left cabinet door', 'cabinet door bottom left']
            },
            'cabinet_bottom_right': {
                'category': 'Cabinet Operations',
                'description': 'Open the bottom right cabinet door',
                'prompt_keywords': ['bottom right cabinet', 'lower right cabinet door', 'cabinet door bottom right']
            },
            'tv_remote_footrest': {
                'category': 'TV Controls',
                'description': 'Turn on the TV using one of the remote controls on the footrest',
                'prompt_keywords': ['remote control footrest', 'footrest remote', 'remote on footrest']
            },
            'tv_remote_table': {
                'category': 'TV Controls',
                'description': 'Turn on the TV using the remote control on the small glass table',
                'prompt_keywords': ['remote control glass table', 'table remote', 'remote on table']
            },
            'radiator_thermostat': {
                'category': 'Room Environment',
                'description': 'Adjust the room\'s temperature using the radiator\'s thermostat located below the window',
                'prompt_keywords': ['radiator thermostat', 'thermostat below window', 'temperature control']
            },
            'window_above_radiator': {
                'category': 'Room Environment',
                'description': 'Open the window above the radiator',
                'prompt_keywords': ['window above radiator', 'radiator window', 'window opening']
            },
            'ceiling_light': {
                'category': 'Room Environment',
                'description': 'Turn on the ceiling light',
                'prompt_keywords': ['ceiling light', 'overhead light', 'room light switch']
            },
            'door_close': {
                'category': 'Other Actions',
                'description': 'Close the door',
                'prompt_keywords': ['door', 'close door', 'door handle']
            },
            'power_outlet': {
                'category': 'Other Actions',
                'description': 'Plug the device in one of the power outlets next to the cabinet',
                'prompt_keywords': ['power outlet', 'electrical outlet', 'outlet next to cabinet', 'socket']
            }
        }

    def load_model(self):
        """Load the Qwen2.5-VL model"""
        if self.model is None:
            print(f"Loading Qwen2.5-VL model on GPU {self.gpu_id}...")
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model.eval()

            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                padding_side="left"
            )

            print(f"✅ Model loaded successfully")

    def generate_task_grounding_prompt(self, tasks: Dict[str, Dict]) -> str:
        """Generate comprehensive prompt for task-specific frame grounding"""

        task_descriptions = []
        for task_id, task_info in tasks.items():
            category = task_info['category']
            description = task_info['description']
            task_descriptions.append(f"- {category}: {description}")

        prompt = f"""
Analyze this video and identify the BEST TIMESTAMPS for performing each of these specific tasks. For each task, find the moment in the video where the relevant objects or areas are most clearly visible and accessible.

TASKS TO ANALYZE:
{chr(10).join(task_descriptions)}

For each task, provide:
1. The exact timestamp (in seconds) when the relevant objects/areas are best visible
2. A brief explanation of why this timestamp is optimal for the task
3. What specific objects or areas are visible at that moment

Please provide precise timestamps in the format "X.X seconds" or "X:XX" and explain your reasoning for each selection.

Focus on finding frames where:
- The target objects (cabinets, remotes, switches, outlets, etc.) are clearly visible
- The objects are at a good angle for interaction
- There's minimal obstruction from other objects or camera movement
- The lighting and visibility are optimal

Format your response with clear sections for each task category.
"""

        return prompt

    def extract_task_timestamps(self, output_text: str, tasks: Dict[str, Dict]) -> Dict[str, Dict]:
        """Extract timestamps and match them to specific tasks"""

        # Extract all timestamps from the output
        timestamp_patterns = [
            r'(\d{1,2}):(\d{2}):(\d{2})',  # HH:MM:SS
            r'(\d{1,2}):(\d{2})',          # MM:SS
            r'(\d+\.?\d*)\s*seconds?',      # X.X seconds
            r'(\d+\.?\d*)\s*s\b',          # X.X s
            r'at\s*(\d+\.?\d*)',           # at X.X
            r'timestamp\s*(\d+\.?\d*)',     # timestamp X.X
        ]

        found_timestamps = []
        for pattern in timestamp_patterns:
            matches = re.finditer(pattern, output_text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 3:  # HH:MM:SS
                    h, m, s = map(int, match.groups())
                    total_seconds = h * 3600 + m * 60 + s
                elif len(match.groups()) == 2:  # MM:SS
                    m, s = map(int, match.groups())
                    total_seconds = m * 60 + s
                else:  # Single number
                    total_seconds = float(match.group(1))

                # Get context around the timestamp
                context_start = max(0, match.start() - 150)
                context_end = min(len(output_text), match.end() + 150)
                context = output_text[context_start:context_end]

                found_timestamps.append({
                    'timestamp_seconds': total_seconds,
                    'original_text': match.group(0),
                    'context': context,
                    'match_position': match.start()
                })

        # Match timestamps to tasks based on keyword presence in context
        task_mappings = {}

        for task_id, task_info in tasks.items():
            best_match = None
            best_score = 0

            for ts in found_timestamps:
                context_lower = ts['context'].lower()
                score = 0

                # Score based on keyword matches
                for keyword in task_info['prompt_keywords']:
                    if keyword.lower() in context_lower:
                        score += 1

                # Additional scoring for category keywords
                category_keywords = {
                    'Cabinet Operations': ['cabinet', 'door', 'open'],
                    'TV Controls': ['tv', 'remote', 'control', 'turn on'],
                    'Room Environment': ['window', 'light', 'temperature', 'radiator'],
                    'Other Actions': ['door', 'outlet', 'plug', 'socket']
                }

                if task_info['category'] in category_keywords:
                    for keyword in category_keywords[task_info['category']]:
                        if keyword.lower() in context_lower:
                            score += 0.5

                if score > best_score:
                    best_score = score
                    best_match = ts

            if best_match and best_score > 0:
                task_mappings[task_id] = {
                    'timestamp_seconds': best_match['timestamp_seconds'],
                    'confidence_score': best_score,
                    'context': best_match['context'],
                    'original_text': best_match['original_text'],
                    'task_info': task_info
                }

        return task_mappings

    def extract_frames_for_tasks(self, video_path: str, task_mappings: Dict[str, Dict],
                                output_dir: str) -> Dict[str, str]:
        """Extract frames at specified timestamps for each task"""

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        video_name = Path(video_path).stem

        # Create structured output directory
        frames_dir = Path(output_dir) / "task_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        extracted_frames = {}

        for task_id, task_data in task_mappings.items():
            timestamp = task_data['timestamp_seconds']
            frame_number = int(timestamp * fps)

            # Set video position
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            # Read frame
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️  Could not read frame for task {task_id} at {timestamp}s")
                continue

            # Create task category subdirectory
            category_dir = frames_dir / task_data['task_info']['category'].replace(' ', '_').lower()
            category_dir.mkdir(parents=True, exist_ok=True)

            # Save frame with descriptive filename
            frame_filename = f"{task_id}_t{timestamp:.2f}s.png"
            frame_path = category_dir / frame_filename

            cv2.imwrite(str(frame_path), frame)
            extracted_frames[task_id] = str(frame_path)

            print(f"✅ Extracted frame for '{task_data['task_info']['description']}'")
            print(f"   Timestamp: {timestamp:.2f}s | File: {frame_path}")

        cap.release()
        return extracted_frames

    def create_task_mapping_report(self, task_mappings: Dict[str, Dict],
                                  extracted_frames: Dict[str, str],
                                  video_path: str, output_dir: str) -> str:
        """Create comprehensive mapping report"""

        video_info = get_video_info(video_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(output_dir) / f"task_frame_mapping_{timestamp}.json"

        # Create detailed report
        report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'video_path': video_path,
                'video_info': video_info,
                'model_path': self.model_path,
                'total_tasks': len(self.tasks),
                'successfully_mapped': len(task_mappings),
                'frames_extracted': len(extracted_frames)
            },
            'task_mappings': {},
            'summary_by_category': {},
            'frame_paths': extracted_frames
        }

        # Organize by category
        categories = {}
        for task_id, task_data in task_mappings.items():
            category = task_data['task_info']['category']
            if category not in categories:
                categories[category] = []

            task_entry = {
                'task_id': task_id,
                'description': task_data['task_info']['description'],
                'timestamp_seconds': task_data['timestamp_seconds'],
                'timestamp_formatted': format_duration(task_data['timestamp_seconds']),
                'confidence_score': task_data['confidence_score'],
                'frame_path': extracted_frames.get(task_id, None),
                'context_preview': task_data['context'][:200] + "..." if len(task_data['context']) > 200 else task_data['context']
            }

            categories[category].append(task_entry)
            report['task_mappings'][task_id] = task_entry

        report['summary_by_category'] = categories

        # Save JSON report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        # Create human-readable markdown report
        md_report_path = Path(output_dir) / f"task_frame_mapping_{timestamp}.md"
        self._create_markdown_report(report, md_report_path, video_info)

        return str(report_path)

    def _create_markdown_report(self, report: Dict, output_path: Path, video_info: Dict):
        """Create human-readable markdown report"""

        with open(output_path, 'w') as f:
            f.write("# Task-Specific Frame Grounding Report\n\n")

            # Video information
            f.write("## 📹 Video Information\n\n")
            f.write(f"- **Video:** {report['metadata']['video_path']}\n")
            f.write(f"- **Duration:** {video_info.get('duration_formatted', 'N/A')}\n")
            f.write(f"- **Resolution:** {video_info.get('width', 'N/A')}x{video_info.get('height', 'N/A')}\n")
            f.write(f"- **FPS:** {video_info.get('fps', 'N/A'):.2f}\n")
            f.write(f"- **Total Frames:** {video_info.get('total_frames', 'N/A'):,}\n\n")

            # Summary
            f.write("## 📊 Analysis Summary\n\n")
            f.write(f"- **Total Tasks:** {report['metadata']['total_tasks']}\n")
            f.write(f"- **Successfully Mapped:** {report['metadata']['successfully_mapped']}\n")
            f.write(f"- **Frames Extracted:** {report['metadata']['frames_extracted']}\n")
            f.write(f"- **Success Rate:** {report['metadata']['successfully_mapped']/report['metadata']['total_tasks']*100:.1f}%\n\n")

            # Results by category
            f.write("## 🎯 Task Mapping Results\n\n")
            for category, tasks in report['summary_by_category'].items():
                f.write(f"### {category}\n\n")

                for task in tasks:
                    f.write(f"#### {task['description']}\n")
                    f.write(f"- **Best Frame Timestamp:** {task['timestamp_formatted']} ({task['timestamp_seconds']:.2f}s)\n")
                    f.write(f"- **Confidence Score:** {task['confidence_score']:.1f}\n")
                    f.write(f"- **Frame Path:** `{task['frame_path']}`\n")
                    f.write(f"- **Context:** {task['context_preview']}\n\n")

            # Frame directory structure
            f.write("## 📁 Extracted Frames Directory Structure\n\n")
            f.write("```\n")
            f.write("task_frames/\n")
            for category in report['summary_by_category'].keys():
                category_dir = category.replace(' ', '_').lower()
                f.write(f"├── {category_dir}/\n")
                for task in report['summary_by_category'][category]:
                    frame_file = Path(task['frame_path']).name if task['frame_path'] else "N/A"
                    f.write(f"│   └── {frame_file}\n")
            f.write("```\n\n")

            # Usage instructions
            f.write("## 🚀 Usage Instructions\n\n")
            f.write("1. **Review Task Mappings:** Check the confidence scores and timestamps\n")
            f.write("2. **Examine Extracted Frames:** Navigate to the `task_frames/` directory\n")
            f.write("3. **Verify Frame Quality:** Ensure objects are clearly visible in extracted frames\n")
            f.write("4. **Use for Training:** These frames can be used as training data for task-specific models\n")

    def run_analysis(self, video_path: str, output_dir: str,
                    selected_tasks: List[str] = None) -> str:
        """Run complete task-specific frame grounding analysis"""

        # Validate inputs
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Load model
        self.load_model()

        # Select tasks to analyze
        if selected_tasks:
            tasks_to_analyze = {k: v for k, v in self.tasks.items() if k in selected_tasks}
        else:
            tasks_to_analyze = self.tasks

        print(f"🎯 Task-Specific Frame Grounding Analysis")
        print(f"{'=' * 60}")
        print(f"Video: {video_path}")
        print(f"Tasks to Analyze: {len(tasks_to_analyze)}")
        print(f"Output Directory: {output_dir}")
        print()

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate grounding prompt
        prompt = self.generate_task_grounding_prompt(tasks_to_analyze)

        print("🔍 Analyzing video for task-specific frames...")

        # Process video with model
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    "max_pixels": 360*420,
                    "fps": 1.0
                },
                {"type": "text", "text": prompt}
            ]
        }]

        # Run inference
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        print("🤖 Generating task-specific timestamp predictions...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=3072,
                do_sample=False,
                temperature=1.0,
                repetition_penalty=1.1
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Extract task-specific timestamps
        print("🎬 Extracting and mapping timestamps to tasks...")
        task_mappings = self.extract_task_timestamps(output_text, tasks_to_analyze)

        print(f"✅ Mapped {len(task_mappings)} tasks to timestamps")

        # Extract frames for mapped tasks
        print("🖼️  Extracting frames for identified tasks...")
        extracted_frames = self.extract_frames_for_tasks(video_path, task_mappings, output_dir)

        # Create comprehensive report
        print("📊 Creating task mapping report...")
        report_path = self.create_task_mapping_report(
            task_mappings, extracted_frames, video_path, output_dir
        )

        print(f"\n{'=' * 60}")
        print("🎉 TASK FRAME GROUNDING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Tasks Analyzed: {len(tasks_to_analyze)}")
        print(f"Successfully Mapped: {len(task_mappings)}")
        print(f"Frames Extracted: {len(extracted_frames)}")
        print(f"Success Rate: {len(task_mappings)/len(tasks_to_analyze)*100:.1f}%")
        print(f"Report: {report_path}")
        print(f"Frames Directory: {output_dir}/task_frames/")

        return report_path


def main():
    parser = argparse.ArgumentParser(description="Task-specific frame grounding and extraction")
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--output_dir", type=str,
                       default="/home/jtu9/reasoning/Affordance-R1-inference/qwen_scripts/results/task_grounding",
                       help="Output directory for results")
    parser.add_argument("--model_path", type=str,
                       default="/home/jtu9/reasoning/models/Qwen2.5-VL-7B-Instruct",
                       help="Path to Qwen2.5-VL model")
    parser.add_argument("--tasks", type=str, nargs='+',
                       choices=[
                           'cabinet_top_left', 'cabinet_top_right', 'cabinet_bottom_left', 'cabinet_bottom_right',
                           'tv_remote_footrest', 'tv_remote_table', 'radiator_thermostat', 'window_above_radiator',
                           'ceiling_light', 'door_close', 'power_outlet'
                       ],
                       help="Specific tasks to analyze (default: all)")
    parser.add_argument("--gpu_id", type=str, default="1",
                       help="GPU ID to use")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = TaskFrameGroundingAnalyzer(args.model_path, args.gpu_id)

    try:
        # Run analysis
        report_path = analyzer.run_analysis(
            video_path=args.video_path,
            output_dir=args.output_dir,
            selected_tasks=args.tasks
        )

        print(f"\n📋 Analysis complete! Report saved to: {report_path}")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()