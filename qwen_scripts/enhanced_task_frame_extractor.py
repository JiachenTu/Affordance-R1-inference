#!/usr/bin/env python3
"""
Enhanced Task Frame Extractor - Stage 1: High-Recall Frame Extraction
Extracts up to 16 candidate frames per task with temporal diversity and quality filtering
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
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from video_utils import get_video_info, format_duration


class EnhancedTaskFrameExtractor:
    """Stage 1: High-recall frame extraction for task-specific analysis"""

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
                'keywords': ['cabinet', 'door', 'top', 'left', 'upper', 'open', 'handle', 'knob']
            },
            'cabinet_top_right': {
                'category': 'Cabinet Operations',
                'description': 'Open the top right cabinet door',
                'keywords': ['cabinet', 'door', 'top', 'right', 'upper', 'open', 'handle', 'knob']
            },
            'cabinet_bottom_left': {
                'category': 'Cabinet Operations',
                'description': 'Open the bottom left cabinet door',
                'keywords': ['cabinet', 'door', 'bottom', 'left', 'lower', 'open', 'handle', 'knob']
            },
            'cabinet_bottom_right': {
                'category': 'Cabinet Operations',
                'description': 'Open the bottom right cabinet door',
                'keywords': ['cabinet', 'door', 'bottom', 'right', 'lower', 'open', 'handle', 'knob']
            },
            'tv_remote_footrest': {
                'category': 'TV Controls',
                'description': 'Turn on the TV using one of the remote controls on the footrest',
                'keywords': ['remote', 'control', 'footrest', 'tv', 'television', 'ottoman', 'stool']
            },
            'tv_remote_table': {
                'category': 'TV Controls',
                'description': 'Turn on the TV using the remote control on the small glass table',
                'keywords': ['remote', 'control', 'table', 'glass', 'tv', 'television', 'small', 'coffee']
            },
            'radiator_thermostat': {
                'category': 'Room Environment',
                'description': 'Adjust the room\'s temperature using the radiator\'s thermostat located below the window',
                'keywords': ['radiator', 'thermostat', 'temperature', 'heating', 'window', 'below', 'control', 'dial']
            },
            'window_above_radiator': {
                'category': 'Room Environment',
                'description': 'Open the window above the radiator',
                'keywords': ['window', 'radiator', 'above', 'open', 'glass', 'frame', 'handle']
            },
            'ceiling_light': {
                'category': 'Room Environment',
                'description': 'Turn on the ceiling light',
                'keywords': ['light', 'ceiling', 'overhead', 'switch', 'lamp', 'fixture', 'illumination']
            },
            'door_close': {
                'category': 'Other Actions',
                'description': 'Close the door',
                'keywords': ['door', 'close', 'handle', 'frame', 'entrance', 'exit']
            },
            'power_outlet': {
                'category': 'Other Actions',
                'description': 'Plug the device in one of the power outlets next to the cabinet',
                'keywords': ['outlet', 'socket', 'power', 'plug', 'electrical', 'cabinet', 'wall']
            }
        }

    def load_model(self):
        """Load the Qwen2.5-VL model for initial timestamp detection"""
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

    def extract_systematic_frames(self, video_path: str, num_frames: int = 8) -> List[Tuple[float, int]]:
        """Extract frames at systematic intervals across video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        # Extract at evenly spaced intervals: 5%, 15%, 25%, 35%, 50%, 65%, 80%, 95%
        percentages = [0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.95][:num_frames]

        timestamps = []
        for pct in percentages:
            timestamp = duration * pct
            frame_idx = int(timestamp * fps)
            timestamps.append((timestamp, frame_idx))

        return timestamps

    def get_initial_timestamp_candidates(self, video_path: str) -> List[float]:
        """Use VLM to get initial timestamp candidates for all tasks"""
        prompt = f"""
        Analyze this video and identify when objects and areas relevant to these tasks are most visible:

        1. Cabinet doors (all positions) - when are cabinets clearly visible?
        2. Remote controls - when are remotes on footrest or table visible?
        3. Radiator and thermostat - when is heating equipment visible?
        4. Windows - when are windows clearly shown?
        5. Light switches or ceiling fixtures - when are lighting controls visible?
        6. Doors - when are room doors prominently shown?
        7. Power outlets - when are electrical outlets near cabinets visible?

        Provide timestamps in seconds where these objects are most clearly visible and accessible.
        Focus on moments with good lighting, clear object visibility, and minimal camera movement.
        """

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

        # Process with model
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

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=2048,
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

        # Extract timestamps from output
        return self._extract_timestamps_from_text(output_text)

    def _extract_timestamps_from_text(self, text: str) -> List[float]:
        """Extract all timestamps from text response"""
        import re

        timestamp_patterns = [
            r'(\d{1,2}):(\d{2}):(\d{2})',  # HH:MM:SS
            r'(\d{1,2}):(\d{2})',          # MM:SS
            r'(\d+\.?\d*)\s*seconds?',      # X.X seconds
            r'(\d+\.?\d*)\s*s\b',          # X.X s
            r'at\s*(\d+\.?\d*)',           # at X.X
            r'(\d+\.?\d*)s',               # X.Xs
        ]

        timestamps = []
        for pattern in timestamp_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) == 3:  # HH:MM:SS
                    h, m, s = map(int, match.groups())
                    total_seconds = h * 3600 + m * 60 + s
                elif len(match.groups()) == 2:  # MM:SS
                    m, s = map(int, match.groups())
                    total_seconds = m * 60 + s
                else:  # Single number
                    try:
                        total_seconds = float(match.group(1))
                    except (ValueError, IndexError):
                        continue

                if 0 <= total_seconds <= 3600:  # Reasonable range
                    timestamps.append(total_seconds)

        # Remove duplicates and sort
        timestamps = sorted(list(set(timestamps)))
        return timestamps

    def generate_dense_candidates(self, base_timestamps: List[float], video_duration: float,
                                window_size: float = 10.0, density: float = 2.0) -> List[float]:
        """Generate dense sampling around base timestamps"""
        candidates = set(base_timestamps)

        # Add dense samples around each base timestamp
        for timestamp in base_timestamps:
            # Sample every 2 seconds in a 10-second window around each base timestamp
            start = max(0, timestamp - window_size/2)
            end = min(video_duration, timestamp + window_size/2)

            current = start
            while current <= end:
                candidates.add(current)
                current += density

        return sorted(list(candidates))

    def filter_temporal_spacing(self, timestamps: List[float], min_spacing: float = 3.0) -> List[float]:
        """Ensure minimum temporal spacing between frames"""
        if not timestamps:
            return []

        filtered = [timestamps[0]]

        for timestamp in timestamps[1:]:
            if timestamp - filtered[-1] >= min_spacing:
                filtered.append(timestamp)

        return filtered

    def assess_frame_quality(self, frame: np.ndarray) -> float:
        """Assess frame quality based on brightness, contrast, and sharpness"""
        if frame is None or frame.size == 0:
            return 0.0

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Brightness score (avoid too dark/bright frames)
        brightness = np.mean(gray)
        brightness_score = 1.0 - abs(brightness - 127) / 127

        # Contrast score
        contrast = np.std(gray)
        contrast_score = min(contrast / 64.0, 1.0)  # Normalize to 0-1

        # Sharpness score (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        sharpness_score = min(sharpness / 1000.0, 1.0)  # Normalize

        # Combined quality score
        quality_score = (brightness_score * 0.3 + contrast_score * 0.4 + sharpness_score * 0.3)
        return quality_score

    def extract_candidate_frames(self, video_path: str, timestamps: List[float],
                                output_dir: str, max_frames: int = 16) -> Dict[str, List[str]]:
        """Extract frames at specified timestamps with quality assessment"""

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        video_name = Path(video_path).stem

        # Create output directory structure
        candidates_dir = Path(output_dir) / "stage1_candidates"
        for category in set(task_info['category'] for task_info in self.tasks.values()):
            category_dir = candidates_dir / category.replace(' ', '_').lower()
            category_dir.mkdir(parents=True, exist_ok=True)

        extracted_frames = {}
        frame_qualities = []

        # Extract and assess all frames
        for i, timestamp in enumerate(timestamps):
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            if not ret:
                continue

            # Assess frame quality
            quality = self.assess_frame_quality(frame)
            frame_qualities.append((timestamp, frame, quality, i))

        cap.release()

        # Sort by quality and take top frames
        frame_qualities.sort(key=lambda x: x[2], reverse=True)
        selected_frames = frame_qualities[:max_frames]

        # Save selected frames and organize by task relevance
        saved_frames = []
        for timestamp, frame, quality, orig_idx in selected_frames:
            frame_filename = f"candidate_{orig_idx:03d}_t{timestamp:.2f}s_q{quality:.3f}.png"

            # Save frame temporarily
            temp_path = candidates_dir / frame_filename
            cv2.imwrite(str(temp_path), frame)

            saved_frames.append({
                'timestamp': timestamp,
                'quality_score': quality,
                'frame_path': str(temp_path),
                'frame_idx': orig_idx
            })

        print(f"✅ Extracted {len(saved_frames)} candidate frames")
        return saved_frames

    def organize_frames_by_task(self, candidate_frames: List[Dict], output_dir: str) -> Dict[str, List[Dict]]:
        """Organize candidate frames by task category for easier processing"""

        candidates_dir = Path(output_dir) / "stage1_candidates"
        task_assignments = {}

        # For now, assign all frames to all tasks (broad recall)
        # Later stages will filter based on task-specific criteria
        for task_id, task_info in self.tasks.items():
            category_dir = candidates_dir / task_info['category'].replace(' ', '_').lower()
            task_frames = []

            for frame_data in candidate_frames:
                # Copy frame to task category directory
                original_path = Path(frame_data['frame_path'])
                task_frame_path = category_dir / f"{task_id}_{original_path.name}"

                # Create symbolic link or copy
                import shutil
                shutil.copy2(original_path, task_frame_path)

                task_frame_data = frame_data.copy()
                task_frame_data['task_frame_path'] = str(task_frame_path)
                task_frame_data['task_id'] = task_id
                task_frames.append(task_frame_data)

            task_assignments[task_id] = task_frames

        # Clean up temporary files
        for frame_data in candidate_frames:
            temp_path = Path(frame_data['frame_path'])
            if temp_path.exists():
                temp_path.unlink()

        return task_assignments

    def run_stage1_extraction(self, video_path: str, output_dir: str,
                            max_frames_per_task: int = 16) -> str:
        """Run complete Stage 1: High-recall frame extraction"""

        print(f"🎯 Stage 1: Enhanced Task Frame Extraction")
        print(f"{'=' * 60}")
        print(f"Video: {video_path}")
        print(f"Max Frames per Task: {max_frames_per_task}")
        print(f"Output Directory: {output_dir}")
        print()

        # Load model
        self.load_model()

        # Get video info
        video_info = get_video_info(video_path)
        video_duration = video_info['duration_seconds']

        print(f"📹 Video Duration: {format_duration(video_duration)}")

        # Step 1: Get systematic frame samples
        print(f"🔍 Step 1: Systematic frame sampling...")
        systematic_timestamps = self.extract_systematic_frames(video_path, num_frames=8)
        systematic_times = [ts[0] for ts in systematic_timestamps]
        print(f"   Systematic samples: {len(systematic_times)}")

        # Step 2: Get VLM-detected timestamps
        print(f"🤖 Step 2: VLM timestamp detection...")
        vlm_timestamps = self.get_initial_timestamp_candidates(video_path)
        print(f"   VLM detected: {len(vlm_timestamps)}")

        # Step 3: Generate dense candidates
        print(f"📊 Step 3: Dense candidate generation...")
        all_base_timestamps = list(set(systematic_times + vlm_timestamps))
        dense_candidates = self.generate_dense_candidates(
            all_base_timestamps, video_duration, window_size=10.0, density=2.0
        )
        print(f"   Dense candidates: {len(dense_candidates)}")

        # Step 4: Apply temporal spacing filter
        print(f"⏰ Step 4: Temporal spacing filter...")
        spaced_candidates = self.filter_temporal_spacing(dense_candidates, min_spacing=3.0)
        print(f"   After spacing filter: {len(spaced_candidates)}")

        # Step 5: Extract and assess frames
        print(f"🖼️  Step 5: Frame extraction and quality assessment...")
        candidate_frames = self.extract_candidate_frames(
            video_path, spaced_candidates, output_dir, max_frames=max_frames_per_task * 2
        )

        # Step 6: Organize frames by task
        print(f"📁 Step 6: Organizing frames by task...")
        task_assignments = self.organize_frames_by_task(candidate_frames, output_dir)

        # Save metadata
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_path = Path(output_dir) / f"stage1_extraction_{timestamp}.json"

        metadata = {
            'extraction_timestamp': datetime.now().isoformat(),
            'video_path': video_path,
            'video_info': video_info,
            'extraction_params': {
                'max_frames_per_task': max_frames_per_task,
                'min_temporal_spacing': 3.0,
                'dense_sampling_window': 10.0,
                'dense_sampling_density': 2.0
            },
            'extraction_stats': {
                'systematic_samples': len(systematic_times),
                'vlm_detected': len(vlm_timestamps),
                'dense_candidates': len(dense_candidates),
                'spaced_candidates': len(spaced_candidates),
                'final_extracted': len(candidate_frames)
            },
            'task_assignments': {
                task_id: len(frames) for task_id, frames in task_assignments.items()
            }
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'=' * 60}")
        print(f"🎉 STAGE 1 EXTRACTION COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total Tasks: {len(task_assignments)}")
        print(f"Frames per Task: {len(candidate_frames)} candidates each")
        print(f"Extraction Quality: Based on brightness, contrast, sharpness")
        print(f"Metadata: {metadata_path}")
        print(f"Candidates Directory: {output_dir}/stage1_candidates/")
        print(f"{'=' * 60}")

        return str(metadata_path)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Enhanced task frame extraction")
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to input video file")
    parser.add_argument("--output_dir", type=str,
                       default="/home/jtu9/reasoning/Affordance-R1-inference/qwen_scripts/results/enhanced_task_grounding",
                       help="Output directory for results")
    parser.add_argument("--model_path", type=str,
                       default="/home/jtu9/reasoning/models/Qwen2.5-VL-7B-Instruct",
                       help="Path to Qwen2.5-VL model")
    parser.add_argument("--max_frames", type=int, default=16,
                       help="Maximum frames to extract per task")
    parser.add_argument("--gpu_id", type=str, default="1",
                       help="GPU ID to use")

    args = parser.parse_args()

    # Initialize extractor
    extractor = EnhancedTaskFrameExtractor(args.model_path, args.gpu_id)

    try:
        # Run Stage 1 extraction
        metadata_path = extractor.run_stage1_extraction(
            video_path=args.video_path,
            output_dir=args.output_dir,
            max_frames_per_task=args.max_frames
        )

        print(f"\n📋 Stage 1 complete! Metadata saved to: {metadata_path}")

    except Exception as e:
        print(f"❌ Stage 1 extraction failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()