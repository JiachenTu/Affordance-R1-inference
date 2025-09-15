#!/usr/bin/env python3
"""
Task Bbox Processor - Stage 3: Precise Localization and Segmentation
Uses Qwen2.5-VL for bbox/point extraction and SAM2 for mask generation
"""

import argparse
import json
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
from PIL import Image as PILImage
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from sam2.sam2_image_predictor import SAM2ImagePredictor


class TaskBboxProcessor:
    """Stage 3: Precise bbox/point extraction and SAM2 segmentation"""

    def __init__(self, qwen_model_path: str, sam_model_path: str = "facebook/sam2-hiera-large", gpu_id: str = "3"):
        self.qwen_model_path = qwen_model_path
        self.sam_model_path = sam_model_path
        self.gpu_id = gpu_id
        self.qwen_model = None
        self.processor = None
        self.sam_model = None
        self.tasks = self._get_task_definitions()

    def _get_task_definitions(self) -> Dict[str, Dict]:
        """Get task definitions with localization prompts"""
        return {
            'cabinet_top_left': {
                'category': 'Cabinet Operations',
                'description': 'Open the top left cabinet door',
                'localization_prompt': 'Locate the top left cabinet door handle or knob that needs to be grasped to open the door. Focus on the uppermost left cabinet door.',
                'interaction_type': 'grasp_and_pull'
            },
            'cabinet_top_right': {
                'category': 'Cabinet Operations',
                'description': 'Open the top right cabinet door',
                'localization_prompt': 'Locate the top right cabinet door handle or knob that needs to be grasped to open the door. Focus on the uppermost right cabinet door.',
                'interaction_type': 'grasp_and_pull'
            },
            'cabinet_bottom_left': {
                'category': 'Cabinet Operations',
                'description': 'Open the bottom left cabinet door',
                'localization_prompt': 'Locate the bottom left cabinet door handle or knob that needs to be grasped to open the door. Focus on the lowermost left cabinet door.',
                'interaction_type': 'grasp_and_pull'
            },
            'cabinet_bottom_right': {
                'category': 'Cabinet Operations',
                'description': 'Open the bottom right cabinet door',
                'localization_prompt': 'Locate the bottom right cabinet door handle or knob that needs to be grasped to open the door. Focus on the lowermost right cabinet door.',
                'interaction_type': 'grasp_and_pull'
            },
            'tv_remote_footrest': {
                'category': 'TV Controls',
                'description': 'Turn on the TV using one of the remote controls on the footrest',
                'localization_prompt': 'Locate the remote control that is placed on the footrest or ottoman. Find the specific remote that can be picked up and used.',
                'interaction_type': 'grasp_and_use'
            },
            'tv_remote_table': {
                'category': 'TV Controls',
                'description': 'Turn on the TV using the remote control on the small glass table',
                'localization_prompt': 'Locate the remote control that is placed on the small glass table. Find the specific remote that can be picked up and used.',
                'interaction_type': 'grasp_and_use'
            },
            'radiator_thermostat': {
                'category': 'Room Environment',
                'description': 'Adjust the room\'s temperature using the radiator\'s thermostat located below the window',
                'localization_prompt': 'Locate the thermostat or temperature control dial on the radiator below the window. Find the specific control that can be turned or adjusted.',
                'interaction_type': 'turn_or_press'
            },
            'window_above_radiator': {
                'category': 'Room Environment',
                'description': 'Open the window above the radiator',
                'localization_prompt': 'Locate the window handle or latch above the radiator that needs to be operated to open the window.',
                'interaction_type': 'turn_or_pull'
            },
            'ceiling_light': {
                'category': 'Room Environment',
                'description': 'Turn on the ceiling light',
                'localization_prompt': 'Locate the light switch on the wall that controls the ceiling light. Find the specific switch that needs to be pressed.',
                'interaction_type': 'press'
            },
            'door_close': {
                'category': 'Other Actions',
                'description': 'Close the door',
                'localization_prompt': 'Locate the door handle that needs to be grasped to close the door. Focus on the handle or knob on the open door.',
                'interaction_type': 'grasp_and_push'
            },
            'power_outlet': {
                'category': 'Other Actions',
                'description': 'Plug the device in one of the power outlets next to the cabinet',
                'localization_prompt': 'Locate the power outlet or electrical socket next to the cabinet where a plug can be inserted.',
                'interaction_type': 'insert_plug'
            }
        }

    def load_models(self):
        """Load both Qwen2.5-VL and SAM2 models"""
        print(f"Loading models on GPU {self.gpu_id}...")
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_id

        # Load Qwen2.5-VL model
        if self.qwen_model is None:
            print("  Loading Qwen2.5-VL...")
            self.qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.qwen_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.qwen_model.eval()

            self.processor = AutoProcessor.from_pretrained(
                self.qwen_model_path,
                padding_side="left"
            )

        # Load SAM2 model
        if self.sam_model is None:
            print("  Loading SAM2...")
            self.sam_model = SAM2ImagePredictor.from_pretrained(self.sam_model_path)

        print(f"✅ Both models loaded successfully")

    def generate_bbox_extraction_prompt(self, task_id: str) -> str:
        """Generate Qwen2.5-VL prompt for bbox and point extraction"""
        task_info = self.tasks[task_id]

        prompt = f"""
        Task: {task_info['description']}

        {task_info['localization_prompt']}

        Please analyze this image and identify the exact location where this action should be performed.

        Instructions:
        1. Find the specific object or area mentioned in the task
        2. Determine the precise interaction point (handle, button, switch, etc.)
        3. Provide a bounding box around the target object
        4. Provide an exact interaction point coordinate

        Output the thinking process in <think> </think> tags, rethinking process in <rethink> </rethink> tags, and final answer in <answer> </answer> tags.

        Format your final answer as JSON:
        <answer>[{{"bbox_2d": [x1, y1, x2, y2], "point_2d": [x, y], "interaction_type": "{task_info['interaction_type']}"}}]</answer>

        Where:
        - bbox_2d: [left, top, right, bottom] coordinates of the bounding box around the target object
        - point_2d: [x, y] coordinates of the exact interaction point
        - interaction_type: type of interaction required

        Be as precise as possible with the coordinates.
        """

        return prompt

    def extract_bbox_points_from_response(self, output_text: str, x_factor: float, y_factor: float) -> Tuple[List[List[int]], List[List[int]], str, str]:
        """Extract bounding boxes and points from Qwen2.5-VL response"""
        pred_bboxes = []
        pred_points = []

        # Extract JSON from <answer> tags
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(1).strip()

                # Clean and fix JSON formatting
                json_str = self._clean_json_string(json_str)

                # Parse JSON
                data = json.loads(json_str)
                if isinstance(data, dict):
                    data = [data]

                # Extract and scale coordinates
                pred_bboxes = [[
                    int(item['bbox_2d'][0] * x_factor + 0.5),
                    int(item['bbox_2d'][1] * y_factor + 0.5),
                    int(item['bbox_2d'][2] * x_factor + 0.5),
                    int(item['bbox_2d'][3] * y_factor + 0.5)
                ] for item in data if 'bbox_2d' in item and len(item['bbox_2d']) == 4]

                pred_points = [[
                    int(item['point_2d'][0] * x_factor + 0.5),
                    int(item['point_2d'][1] * y_factor + 0.5)
                ] for item in data if 'point_2d' in item and len(item['point_2d']) == 2]

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                print(f"JSON parsing error: {e}")
                print(f"Raw JSON: {json_match.group(1) if json_match else 'No match'}")

        # Extract thinking processes
        think_match = re.search(r'<think>(.*?)</think>', output_text, re.DOTALL)
        think_text = think_match.group(1).strip() if think_match else ""

        rethink_match = re.search(r'<rethink>(.*?)</rethink>', output_text, re.DOTALL)
        rethink_text = rethink_match.group(1).strip() if rethink_match else ""

        return pred_bboxes, pred_points, think_text, rethink_text

    def _clean_json_string(self, json_str: str) -> str:
        """Clean and fix common JSON formatting issues"""
        # Fix common malformed patterns
        json_str = re.sub(r'"bbox_2d":\s*\[\s*\[([^\]]+)\]', r'"bbox_2d": [\1]', json_str)
        json_str = re.sub(r'"point_2d":\s*\[(\d+,\d+)}', r'"point_2d": [\1]}', json_str)
        json_str = re.sub(r',,+', ',', json_str)
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = json_str.replace('\\"', '"')

        return json_str

    def process_frame_with_qwen(self, frame_path: str, task_id: str) -> Dict:
        """Process a single frame with Qwen2.5-VL for bbox/point extraction"""

        # Load and prepare image
        image = PILImage.open(frame_path).convert("RGB")
        original_width, original_height = image.size
        resize_size = 840
        x_factor, y_factor = original_width / resize_size, original_height / resize_size
        resized_image = image.resize((resize_size, resize_size), PILImage.BILINEAR)

        # Generate task-specific prompt
        prompt = self.generate_bbox_extraction_prompt(task_id)

        # Create messages
        messages = [[{
            "role": "user",
            "content": [
                {"type": "image", "image": resized_image},
                {"type": "text", "text": prompt}
            ]
        }]]

        # Process with Qwen model
        text = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]

        inputs = self.processor(
            text=text,
            images=[resized_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate response
        with torch.no_grad():
            generated_ids = self.qwen_model.generate(
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

        # Extract bbox and points
        bboxes, points, think_text, rethink_text = self.extract_bbox_points_from_response(
            output_text, x_factor, y_factor
        )

        return {
            'frame_path': frame_path,
            'task_id': task_id,
            'original_size': (original_width, original_height),
            'processed_size': (resize_size, resize_size),
            'scaling_factors': (x_factor, y_factor),
            'predicted_bboxes': bboxes,
            'predicted_points': points,
            'thinking_process': think_text,
            'rethinking_process': rethink_text,
            'raw_output': output_text,
            'processing_timestamp': datetime.now().isoformat()
        }

    def generate_sam2_masks(self, frame_path: str, bboxes: List[List[int]], points: List[List[int]]) -> np.ndarray:
        """Generate SAM2 segmentation masks from bboxes and points"""

        # Load original image for SAM2
        image = PILImage.open(frame_path).convert("RGB")

        if not bboxes and not points:
            # Return empty mask if no predictions
            return np.zeros((image.height, image.width), dtype=bool)

        # Set image for SAM2
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_model.set_image(image)
            mask_all = np.zeros((image.height, image.width), dtype=bool)

            # Generate masks for each bbox/point pair
            for bbox, point in zip(bboxes, points):
                try:
                    masks, scores, _ = self.sam_model.predict(
                        point_coords=[point],
                        point_labels=[1],
                        box=bbox
                    )

                    # Take the highest scoring mask
                    if len(masks) > 0:
                        sorted_ind = np.argsort(scores)[::-1]
                        best_mask = masks[sorted_ind[0]].astype(bool)
                        mask_all = np.logical_or(mask_all, best_mask)

                except Exception as e:
                    print(f"     ⚠️  SAM2 prediction failed: {e}")

        return mask_all

    def create_task_visualization(self, frame_path: str, bboxes: List[List[int]], points: List[List[int]],
                                mask: np.ndarray, task_id: str, output_path: str):
        """Create visualization similar to single_inference.py"""

        # Load original image
        image = PILImage.open(frame_path).convert("RGB")
        image_array = np.array(image)

        task_info = self.tasks[task_id]

        # Create figure
        plt.figure(figsize=(18, 8))

        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(image_array)
        plt.title('Original Frame', fontsize=14)
        plt.axis('off')

        # Image with bboxes and points
        plt.subplot(1, 3, 2)
        plt.imshow(image_array)

        # Draw bounding boxes and points
        for bbox, point in zip(bboxes, points):
            x1, y1, x2, y2 = bbox
            # Bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, color='yellow', linewidth=3)
            plt.gca().add_patch(rect)

            # Interaction point
            plt.plot(point[0], point[1], 'ro', markersize=10, markeredgecolor='white', markeredgewidth=2)

        plt.title(f'Detected Objects\n{task_info["interaction_type"]}', fontsize=14)
        plt.axis('off')

        # Image with mask overlay
        plt.subplot(1, 3, 3)
        plt.imshow(image_array, alpha=0.7)

        # Draw mask overlay
        if mask.any():
            mask_overlay = np.zeros_like(image_array)
            mask_overlay[mask] = [255, 0, 0]  # Red mask
            plt.imshow(mask_overlay, alpha=0.4)

        # Draw annotations on mask view too
        for bbox, point in zip(bboxes, points):
            x1, y1, x2, y2 = bbox
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                               fill=False, color='yellow', linewidth=2)
            plt.gca().add_patch(rect)
            plt.plot(point[0], point[1], 'yo', markersize=8, markeredgecolor='black')

        plt.title('Segmentation Mask\n+ Interaction Points', fontsize=14)
        plt.axis('off')

        # Add overall title
        plt.suptitle(f'Task: {task_info["description"]}', fontsize=16, y=0.95)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def process_task_frames(self, stage2_metadata_path: str, output_dir: str) -> Dict[str, List[Dict]]:
        """Process all selected frames for bbox/point extraction and mask generation"""

        # Load Stage 2 metadata
        with open(stage2_metadata_path, 'r') as f:
            stage2_metadata = json.load(f)

        validation_results = stage2_metadata['validation_results']

        print(f"🎯 Stage 3: Bbox/Point Extraction and Mask Generation")
        print(f"{'=' * 60}")
        print(f"Processing selected frames from Stage 2...")
        print()

        # Create output directories
        bbox_results_dir = Path(output_dir) / "bbox_results"
        final_masks_dir = Path(output_dir) / "final_masks"

        for category in set(task_info['category'] for task_info in self.tasks.values()):
            category_dir_name = category.replace(' ', '_').lower()
            (bbox_results_dir / category_dir_name).mkdir(parents=True, exist_ok=True)
            (final_masks_dir / category_dir_name).mkdir(parents=True, exist_ok=True)

        all_processing_results = {}

        # Process each task
        for task_id, task_validations in validation_results.items():
            if not task_validations:
                print(f"⚠️  No validated frames for task: {task_id}")
                continue

            task_info = self.tasks[task_id]
            category_dir_name = task_info['category'].replace(' ', '_').lower()

            print(f"🔍 Processing task: {task_info['description']}")
            print(f"   Frames to process: {len(task_validations)}")

            task_results = []

            for i, validation in enumerate(task_validations, 1):
                frame_path = validation['selected_frame_path']
                frame_name = Path(frame_path).stem

                print(f"   Processing frame {i}/{len(task_validations)}: {frame_name}")

                try:
                    # Step 1: Extract bbox/points with Qwen2.5-VL
                    bbox_result = self.process_frame_with_qwen(frame_path, task_id)

                    bboxes = bbox_result['predicted_bboxes']
                    points = bbox_result['predicted_points']

                    print(f"     Found {len(bboxes)} bboxes, {len(points)} points")

                    if bboxes and points:
                        # Step 2: Generate SAM2 masks
                        print(f"     Generating SAM2 segmentation masks...")
                        mask = self.generate_sam2_masks(frame_path, bboxes, points)

                        # Step 3: Create visualization
                        viz_filename = f"{task_id}_{frame_name}_visualization.png"
                        viz_path = final_masks_dir / category_dir_name / viz_filename

                        print(f"     Creating visualization...")
                        self.create_task_visualization(
                            frame_path, bboxes, points, mask, task_id, str(viz_path)
                        )

                        # Save bbox results
                        bbox_filename = f"{task_id}_{frame_name}_bbox_results.json"
                        bbox_path = bbox_results_dir / category_dir_name / bbox_filename

                        bbox_result['mask_coverage'] = float(np.sum(mask) / mask.size)
                        bbox_result['visualization_path'] = str(viz_path)

                        with open(bbox_path, 'w') as f:
                            json.dump(bbox_result, f, indent=2)

                        task_results.append({
                            'validation_info': validation,
                            'bbox_result': bbox_result,
                            'bbox_result_path': str(bbox_path),
                            'visualization_path': str(viz_path),
                            'mask_coverage': bbox_result['mask_coverage']
                        })

                        print(f"     ✅ Completed - Mask coverage: {bbox_result['mask_coverage']:.3f}")

                    else:
                        print(f"     ❌ No valid bbox/points extracted")

                except Exception as e:
                    print(f"     ❌ Processing failed: {e}")

            all_processing_results[task_id] = task_results
            print(f"   ✅ Task completed: {len(task_results)} successful results")
            print()

        return all_processing_results

    def save_stage3_metadata(self, processing_results: Dict[str, List[Dict]], output_dir: str) -> str:
        """Save Stage 3 processing metadata"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage3_metadata_path = Path(output_dir) / f"stage3_bbox_processing_{timestamp}.json"

        # Calculate summary statistics
        total_frames = sum(len(results) for results in processing_results.values())
        successful_frames = sum(len([r for r in results if r['bbox_result']['predicted_bboxes']]) for results in processing_results.values())

        avg_mask_coverage = 0.0
        if total_frames > 0:
            all_coverages = [
                result['mask_coverage'] for results in processing_results.values()
                for result in results if result['mask_coverage'] > 0
            ]
            avg_mask_coverage = np.mean(all_coverages) if all_coverages else 0.0

        stage3_metadata = {
            'processing_timestamp': datetime.now().isoformat(),
            'qwen_model_path': self.qwen_model_path,
            'sam_model_path': self.sam_model_path,
            'gpu_id': self.gpu_id,
            'processing_results': processing_results,
            'summary': {
                'total_tasks': len(self.tasks),
                'tasks_processed': len(processing_results),
                'total_frames_processed': total_frames,
                'successful_extractions': successful_frames,
                'success_rate': successful_frames / total_frames * 100 if total_frames > 0 else 0.0,
                'average_mask_coverage': avg_mask_coverage
            }
        }

        with open(stage3_metadata_path, 'w') as f:
            json.dump(stage3_metadata, f, indent=2)

        return str(stage3_metadata_path)

    def run_stage3_processing(self, stage2_metadata_path: str, output_dir: str) -> str:
        """Run complete Stage 3: bbox/point extraction and mask generation"""

        # Load models
        self.load_models()

        # Process all task frames
        processing_results = self.process_task_frames(stage2_metadata_path, output_dir)

        # Save metadata
        stage3_metadata_path = self.save_stage3_metadata(processing_results, output_dir)

        # Print summary
        summary = {
            'total_frames': sum(len(results) for results in processing_results.values()),
            'successful': sum(len([r for r in results if r['bbox_result']['predicted_bboxes']]) for results in processing_results.values()),
        }

        print(f"\n{'=' * 60}")
        print(f"🎉 STAGE 3 BBOX/MASK PROCESSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Total Frames Processed: {summary['total_frames']}")
        print(f"Successful Extractions: {summary['successful']}")
        print(f"Success Rate: {summary['successful']/summary['total_frames']*100:.1f}%" if summary['total_frames'] > 0 else "Success Rate: 0%")
        print(f"Metadata: {stage3_metadata_path}")
        print(f"Bbox Results: {output_dir}/bbox_results/")
        print(f"Final Visualizations: {output_dir}/final_masks/")

        return stage3_metadata_path


def main():
    parser = argparse.ArgumentParser(description="Stage 3: Bbox/point extraction and mask generation")
    parser.add_argument("--stage2_metadata", type=str, required=True,
                       help="Path to Stage 2 metadata JSON file")
    parser.add_argument("--output_dir", type=str,
                       default="/home/jtu9/reasoning/Affordance-R1-inference/qwen_scripts/results/enhanced_task_grounding",
                       help="Output directory for results")
    parser.add_argument("--qwen_model_path", type=str,
                       default="/home/jtu9/reasoning/models/Qwen2.5-VL-7B-Instruct",
                       help="Path to Qwen2.5-VL model")
    parser.add_argument("--sam_model_path", type=str,
                       default="facebook/sam2-hiera-large",
                       help="Path to SAM2 model")
    parser.add_argument("--gpu_id", type=str, default="3",
                       help="GPU ID to use")

    args = parser.parse_args()

    # Initialize processor
    processor = TaskBboxProcessor(
        qwen_model_path=args.qwen_model_path,
        sam_model_path=args.sam_model_path,
        gpu_id=args.gpu_id
    )

    try:
        # Run Stage 3 processing
        stage3_metadata_path = processor.run_stage3_processing(
            stage2_metadata_path=args.stage2_metadata,
            output_dir=args.output_dir
        )

        print(f"\n📋 Stage 3 complete! Metadata saved to: {stage3_metadata_path}")

    except Exception as e:
        print(f"❌ Stage 3 processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()