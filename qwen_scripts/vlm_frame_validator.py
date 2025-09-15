#!/usr/bin/env python3
"""
VLM Frame Validator - Stage 2: High-Accuracy Frame Selection
Uses Qwen2.5-VL to validate and rank candidate frames for each task
"""

import argparse
import json
import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from PIL import Image as PILImage
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


class VLMFrameValidator:
    """Stage 2: High-accuracy frame selection using VLM validation"""

    def __init__(self, model_path: str, gpu_id: str = "2"):
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.model = None
        self.processor = None
        self.tasks = self._get_task_definitions()

    def _get_task_definitions(self) -> Dict[str, Dict]:
        """Get detailed task definitions for validation"""
        return {
            'cabinet_top_left': {
                'category': 'Cabinet Operations',
                'description': 'Open the top left cabinet door',
                'validation_criteria': [
                    'Cabinet doors are clearly visible',
                    'Top left cabinet door is identifiable',
                    'Cabinet handles/knobs are visible',
                    'Good angle for interaction',
                    'Adequate lighting on cabinet area'
                ],
                'ideal_objects': ['cabinet door', 'handle', 'knob', 'upper cabinet', 'kitchen cabinet']
            },
            'cabinet_top_right': {
                'category': 'Cabinet Operations',
                'description': 'Open the top right cabinet door',
                'validation_criteria': [
                    'Cabinet doors are clearly visible',
                    'Top right cabinet door is identifiable',
                    'Cabinet handles/knobs are visible',
                    'Good angle for interaction',
                    'Adequate lighting on cabinet area'
                ],
                'ideal_objects': ['cabinet door', 'handle', 'knob', 'upper cabinet', 'kitchen cabinet']
            },
            'cabinet_bottom_left': {
                'category': 'Cabinet Operations',
                'description': 'Open the bottom left cabinet door',
                'validation_criteria': [
                    'Cabinet doors are clearly visible',
                    'Bottom left cabinet door is identifiable',
                    'Cabinet handles/knobs are visible',
                    'Good angle for interaction',
                    'Adequate lighting on cabinet area'
                ],
                'ideal_objects': ['cabinet door', 'handle', 'knob', 'lower cabinet', 'kitchen cabinet']
            },
            'cabinet_bottom_right': {
                'category': 'Cabinet Operations',
                'description': 'Open the bottom right cabinet door',
                'validation_criteria': [
                    'Cabinet doors are clearly visible',
                    'Bottom right cabinet door is identifiable',
                    'Cabinet handles/knobs are visible',
                    'Good angle for interaction',
                    'Adequate lighting on cabinet area'
                ],
                'ideal_objects': ['cabinet door', 'handle', 'knob', 'lower cabinet', 'kitchen cabinet']
            },
            'tv_remote_footrest': {
                'category': 'TV Controls',
                'description': 'Turn on the TV using one of the remote controls on the footrest',
                'validation_criteria': [
                    'Footrest/ottoman is clearly visible',
                    'Remote control is visible on footrest',
                    'Remote control is accessible',
                    'TV is visible in frame',
                    'Good lighting on remote area'
                ],
                'ideal_objects': ['remote control', 'footrest', 'ottoman', 'tv', 'television']
            },
            'tv_remote_table': {
                'category': 'TV Controls',
                'description': 'Turn on the TV using the remote control on the small glass table',
                'validation_criteria': [
                    'Small glass table is clearly visible',
                    'Remote control is visible on table',
                    'Remote control is accessible',
                    'TV is visible in frame',
                    'Table surface is unobstructed'
                ],
                'ideal_objects': ['remote control', 'glass table', 'coffee table', 'tv', 'television']
            },
            'radiator_thermostat': {
                'category': 'Room Environment',
                'description': 'Adjust the room\'s temperature using the radiator\'s thermostat located below the window',
                'validation_criteria': [
                    'Radiator is clearly visible',
                    'Thermostat/control is visible on radiator',
                    'Window is visible above radiator',
                    'Thermostat is accessible',
                    'Good lighting on control area'
                ],
                'ideal_objects': ['radiator', 'thermostat', 'heating control', 'window', 'temperature dial']
            },
            'window_above_radiator': {
                'category': 'Room Environment',
                'description': 'Open the window above the radiator',
                'validation_criteria': [
                    'Window is clearly visible',
                    'Window frame and handles are visible',
                    'Radiator is visible below window',
                    'Window opening mechanism is accessible',
                    'Good view of window area'
                ],
                'ideal_objects': ['window', 'window frame', 'window handle', 'radiator', 'glass']
            },
            'ceiling_light': {
                'category': 'Room Environment',
                'description': 'Turn on the ceiling light',
                'validation_criteria': [
                    'Light switch or fixture is visible',
                    'Ceiling light or lamp is visible',
                    'Switch is accessible',
                    'Good view of lighting controls',
                    'Room lighting context is clear'
                ],
                'ideal_objects': ['light switch', 'ceiling light', 'light fixture', 'lamp', 'wall switch']
            },
            'door_close': {
                'category': 'Other Actions',
                'description': 'Close the door',
                'validation_criteria': [
                    'Door is clearly visible and open',
                    'Door handle is visible',
                    'Door frame is visible',
                    'Good angle for door interaction',
                    'Clear path to door'
                ],
                'ideal_objects': ['door', 'door handle', 'door frame', 'doorway', 'entrance']
            },
            'power_outlet': {
                'category': 'Other Actions',
                'description': 'Plug the device in one of the power outlets next to the cabinet',
                'validation_criteria': [
                    'Power outlet is clearly visible',
                    'Outlet is accessible',
                    'Cabinet is visible next to outlet',
                    'Outlet appears functional',
                    'Good lighting on outlet area'
                ],
                'ideal_objects': ['power outlet', 'electrical outlet', 'socket', 'cabinet', 'wall outlet']
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

    def generate_validation_prompt(self, task_id: str, task_info: Dict) -> str:
        """Generate task-specific validation prompt"""

        criteria_text = '\n'.join([f"- {criterion}" for criterion in task_info['validation_criteria']])
        objects_text = ', '.join(task_info['ideal_objects'])

        prompt = f"""
        Evaluate this image for the specific task: "{task_info['description']}"

        Please rate this image from 1-10 for how well it supports this task:

        EVALUATION CRITERIA:
        {criteria_text}

        KEY OBJECTS TO LOOK FOR:
        {objects_text}

        RATING SCALE:
        10 = Perfect - All objects clearly visible, excellent angle, optimal lighting
        8-9 = Excellent - Most criteria met, very good for task execution
        6-7 = Good - Adequate visibility, task could be performed
        4-5 = Fair - Some objects visible but not ideal conditions
        2-3 = Poor - Objects barely visible or poor angle/lighting
        1 = Unusable - Required objects not visible or completely unsuitable

        Please provide:
        1. A numerical score (1-10)
        2. Brief explanation of why you gave this score
        3. List of visible relevant objects
        4. Any issues that affect task suitability

        Format your response as:
        Score: [1-10]
        Explanation: [brief explanation]
        Visible Objects: [list of relevant objects seen]
        Issues: [any problems or limitations]
        """

        return prompt

    def validate_frame_for_task(self, frame_path: str, task_id: str) -> Dict:
        """Validate a single frame for a specific task"""

        if task_id not in self.tasks:
            raise ValueError(f"Unknown task ID: {task_id}")

        task_info = self.tasks[task_id]
        prompt = self.generate_validation_prompt(task_id, task_info)

        # Load and process image
        image = PILImage.open(frame_path).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }]

        # Process with model
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=1024,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Parse validation response
        validation_result = self._parse_validation_response(output_text, frame_path, task_id)
        return validation_result

    def _parse_validation_response(self, response_text: str, frame_path: str, task_id: str) -> Dict:
        """Parse VLM validation response"""
        import re

        # Extract score
        score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', response_text, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 0.0

        # Extract explanation
        explanation_match = re.search(r'Explanation:\s*([^\n]+)', response_text, re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        # Extract visible objects
        objects_match = re.search(r'Visible Objects:\s*([^\n]+)', response_text, re.IGNORECASE)
        visible_objects = objects_match.group(1).strip() if objects_match else ""

        # Extract issues
        issues_match = re.search(r'Issues:\s*([^\n]+)', response_text, re.IGNORECASE)
        issues = issues_match.group(1).strip() if issues_match else ""

        return {
            'frame_path': frame_path,
            'task_id': task_id,
            'score': score,
            'explanation': explanation,
            'visible_objects': visible_objects,
            'issues': issues,
            'raw_response': response_text,
            'validation_timestamp': datetime.now().isoformat()
        }

    def validate_all_task_candidates(self, stage1_metadata_path: str, output_dir: str,
                                   top_k_per_task: int = 3) -> Dict[str, List[Dict]]:
        """Validate all candidate frames for all tasks"""

        # Load Stage 1 metadata
        with open(stage1_metadata_path, 'r') as f:
            stage1_metadata = json.load(f)

        candidates_dir = Path(output_dir) / "stage1_candidates"

        print(f"🔍 Stage 2: VLM Frame Validation")
        print(f"{'=' * 60}")
        print(f"Candidates Directory: {candidates_dir}")
        print(f"Top K per Task: {top_k_per_task}")
        print()

        all_validations = {}

        # Process each task
        for task_id, task_info in self.tasks.items():
            print(f"🎯 Validating task: {task_info['description']}")

            category_dir = candidates_dir / task_info['category'].replace(' ', '_').lower()

            # Find all candidate frames for this task
            task_frames = list(category_dir.glob(f"{task_id}_*.png"))

            if not task_frames:
                print(f"   ⚠️  No candidate frames found for {task_id}")
                all_validations[task_id] = []
                continue

            print(f"   Found {len(task_frames)} candidate frames")

            # Validate each frame
            task_validations = []
            for i, frame_path in enumerate(task_frames, 1):
                print(f"   Validating frame {i}/{len(task_frames)}: {frame_path.name}")

                try:
                    validation = self.validate_frame_for_task(str(frame_path), task_id)
                    task_validations.append(validation)
                    print(f"     Score: {validation['score']:.1f}/10")
                except Exception as e:
                    print(f"     ❌ Validation failed: {e}")

            # Sort by score and take top K
            task_validations.sort(key=lambda x: x['score'], reverse=True)
            top_validations = task_validations[:top_k_per_task]

            all_validations[task_id] = top_validations

            print(f"   ✅ Selected top {len(top_validations)} frames")
            for j, validation in enumerate(top_validations, 1):
                print(f"     #{j}: {validation['score']:.1f}/10 - {Path(validation['frame_path']).name}")
            print()

        return all_validations

    def save_selected_frames(self, validations: Dict[str, List[Dict]], output_dir: str) -> str:
        """Save top-selected frames to Stage 2 directory"""

        stage2_dir = Path(output_dir) / "stage2_selected"
        stage2_dir.mkdir(parents=True, exist_ok=True)

        selected_frames_info = {}

        for task_id, task_validations in validations.items():
            if not task_validations:
                continue

            task_info = self.tasks[task_id]
            category_dir = stage2_dir / task_info['category'].replace(' ', '_').lower()
            category_dir.mkdir(parents=True, exist_ok=True)

            task_selected = []

            for i, validation in enumerate(task_validations, 1):
                # Copy selected frame to Stage 2 directory
                original_path = Path(validation['frame_path'])
                selected_frame_name = f"{task_id}_selected_{i}_score{validation['score']:.1f}.png"
                selected_path = category_dir / selected_frame_name

                import shutil
                shutil.copy2(original_path, selected_path)

                # Update validation info
                validation_copy = validation.copy()
                validation_copy['selected_frame_path'] = str(selected_path)
                validation_copy['selection_rank'] = i
                task_selected.append(validation_copy)

            selected_frames_info[task_id] = task_selected

        # Save Stage 2 metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage2_metadata_path = stage2_dir / f"stage2_validation_{timestamp}.json"

        stage2_metadata = {
            'validation_timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'gpu_id': self.gpu_id,
            'validation_results': selected_frames_info,
            'summary': {
                'total_tasks': len(self.tasks),
                'tasks_with_selections': len([t for t in selected_frames_info.values() if t]),
                'total_selected_frames': sum(len(t) for t in selected_frames_info.values()),
                'average_score': np.mean([
                    v['score'] for task_vals in selected_frames_info.values()
                    for v in task_vals
                ]) if any(selected_frames_info.values()) else 0.0
            }
        }

        with open(stage2_metadata_path, 'w') as f:
            json.dump(stage2_metadata, f, indent=2)

        return str(stage2_metadata_path)

    def create_validation_report(self, stage2_metadata_path: str, output_dir: str) -> str:
        """Create detailed validation report"""

        with open(stage2_metadata_path, 'r') as f:
            metadata = json.load(f)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(output_dir) / f"stage2_validation_report_{timestamp}.md"

        with open(report_path, 'w') as f:
            f.write("# Stage 2: VLM Frame Validation Report\n\n")

            # Summary
            summary = metadata['summary']
            f.write("## 📊 Validation Summary\n\n")
            f.write(f"- **Total Tasks**: {summary['total_tasks']}\n")
            f.write(f"- **Tasks with Selections**: {summary['tasks_with_selections']}\n")
            f.write(f"- **Total Selected Frames**: {summary['total_selected_frames']}\n")
            f.write(f"- **Average Score**: {summary['average_score']:.2f}/10\n\n")

            # Detailed results by task
            f.write("## 🎯 Task Validation Results\n\n")

            for task_id, validations in metadata['validation_results'].items():
                if not validations:
                    continue

                task_info = self.tasks[task_id]
                f.write(f"### {task_info['description']}\n")
                f.write(f"**Category**: {task_info['category']}\n\n")

                for validation in validations:
                    f.write(f"#### Rank {validation['selection_rank']}: Score {validation['score']:.1f}/10\n")
                    f.write(f"- **Frame**: `{Path(validation['selected_frame_path']).name}`\n")
                    f.write(f"- **Explanation**: {validation['explanation']}\n")
                    f.write(f"- **Visible Objects**: {validation['visible_objects']}\n")
                    if validation['issues']:
                        f.write(f"- **Issues**: {validation['issues']}\n")
                    f.write("\n")

            # Directory structure
            f.write("## 📁 Selected Frames Directory Structure\n\n")
            f.write("```\n")
            f.write("stage2_selected/\n")
            for category in set(task_info['category'] for task_info in self.tasks.values()):
                category_dir = category.replace(' ', '_').lower()
                f.write(f"├── {category_dir}/\n")

                category_tasks = [
                    task_id for task_id, task_info in self.tasks.items()
                    if task_info['category'] == category
                ]

                for task_id in category_tasks:
                    if task_id in metadata['validation_results'] and metadata['validation_results'][task_id]:
                        f.write(f"│   ├── {task_id}_selected_1_score*.png\n")
                        f.write(f"│   ├── {task_id}_selected_2_score*.png\n")
                        f.write(f"│   └── {task_id}_selected_3_score*.png\n")
            f.write("```\n")

        return str(report_path)

    def run_stage2_validation(self, stage1_metadata_path: str, output_dir: str,
                            top_k_per_task: int = 3) -> str:
        """Run complete Stage 2: VLM frame validation"""

        # Load model
        self.load_model()

        # Validate all candidates
        print("Starting VLM validation of all candidate frames...")
        validations = self.validate_all_task_candidates(
            stage1_metadata_path, output_dir, top_k_per_task
        )

        # Save selected frames
        print("Saving top-selected frames...")
        stage2_metadata_path = self.save_selected_frames(validations, output_dir)

        # Create validation report
        print("Creating validation report...")
        report_path = self.create_validation_report(stage2_metadata_path, output_dir)

        print(f"\n{'=' * 60}")
        print(f"🎉 STAGE 2 VALIDATION COMPLETE")
        print(f"{'=' * 60}")

        total_selected = sum(len(task_vals) for task_vals in validations.values())
        avg_score = np.mean([
            v['score'] for task_vals in validations.values() for v in task_vals
        ]) if total_selected > 0 else 0.0

        print(f"Selected Frames: {total_selected}")
        print(f"Average Score: {avg_score:.2f}/10")
        print(f"Metadata: {stage2_metadata_path}")
        print(f"Report: {report_path}")
        print(f"Selected Frames Directory: {output_dir}/stage2_selected/")

        return stage2_metadata_path


def main():
    parser = argparse.ArgumentParser(description="Stage 2: VLM frame validation")
    parser.add_argument("--stage1_metadata", type=str, required=True,
                       help="Path to Stage 1 metadata JSON file")
    parser.add_argument("--output_dir", type=str,
                       default="/home/jtu9/reasoning/Affordance-R1-inference/qwen_scripts/results/enhanced_task_grounding",
                       help="Output directory for results")
    parser.add_argument("--model_path", type=str,
                       default="/home/jtu9/reasoning/models/Qwen2.5-VL-7B-Instruct",
                       help="Path to Qwen2.5-VL model")
    parser.add_argument("--top_k", type=int, default=3,
                       help="Top K frames to select per task")
    parser.add_argument("--gpu_id", type=str, default="2",
                       help="GPU ID to use")

    args = parser.parse_args()

    # Initialize validator
    validator = VLMFrameValidator(args.model_path, args.gpu_id)

    try:
        # Run Stage 2 validation
        stage2_metadata_path = validator.run_stage2_validation(
            stage1_metadata_path=args.stage1_metadata,
            output_dir=args.output_dir,
            top_k_per_task=args.top_k
        )

        print(f"\n📋 Stage 2 complete! Metadata saved to: {stage2_metadata_path}")

    except Exception as e:
        print(f"❌ Stage 2 validation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()