#!/usr/bin/env python3
"""
Single Image Inference Script for Affordance-R1
Adapted from the original inference script with enhanced features
"""

import os
import sys
import argparse
import json
import re
from datetime import datetime
from pathlib import Path

# Add the main repo to Python path for imports
sys.path.append('/home/jtu9/reasoning/Affordance-R1')

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from sam2.sam2_image_predictor import SAM2ImagePredictor


def parse_args():
    parser = argparse.ArgumentParser(description="Run Affordance-R1 inference on a single image")
    parser.add_argument("--reasoning_model_path", type=str, 
                       default="/home/jtu9/reasoning/models/affordance-r1/huggingface",
                       help="Path to the reasoning model")
    parser.add_argument("--segmentation_model_path", type=str, 
                       default="facebook/sam2-hiera-large",
                       help="Path to the segmentation model")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to the input image")
    parser.add_argument("--question", type=str, required=True,
                       help="Affordance question to ask about the image")
    parser.add_argument("--output_dir", type=str, 
                       default="/home/jtu9/reasoning/Affordance-R1-inference/results",
                       help="Directory to save results")
    parser.add_argument("--gpu_id", type=str, default="7",
                       help="GPU ID to use")
    return parser.parse_args()



def extract_bbox_points_think(output_text, x_factor, y_factor):
    """Extract bounding boxes, points, and reasoning from model output"""
    pred_bboxes = []
    pred_points = []

    # Extract JSON from <answer> tags
    json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', output_text, re.DOTALL)
    if json_match:
        try:
            json_str = json_match.group(1).strip()

            # Fix common malformed JSON patterns from Affordance-R1 model
            # Pattern 1: Fix malformed nested arrays like [ [3,3,689,556] should be [3,3,689,556]
            json_str = re.sub(r'"bbox_2d":\s*\[\s*\[([^\]]+)\]', r'"bbox_2d": [\1]', json_str)

            # Pattern 2: Fix malformed point_2d arrays with wrong brackets
            # "point_2d": [134,139}, should be "point_2d": [134,139]}
            json_str = re.sub(r'"point_2d":\s*\[(\d+,\d+)}', r'"point_2d": [\1]}', json_str)

            # Pattern 3: Fix the severely malformed JSON structure
            # Original: [{"bbox_2d": [ [3,3,689,556], "point_2d": [125,424], "affbox_2d": [3,527,646,839],,point_2d": [75,656]}], "afffordance": [open"]

            # Step 3a: Remove invalid fields
            json_str = re.sub(r'"aff?box_2d":\s*\[[^\]]+\]', '', json_str)  # Remove affbox_2d
            json_str = re.sub(r'"aff?fordance":\s*\[[^\]]+\]', '', json_str)  # Remove malformed affordance

            # Step 3b: Fix malformed point_2d duplicates like ,,point_2d": [75,656]}
            json_str = re.sub(r',,?\s*point_2d["\s]*:\s*\[[^\]]+\]', '', json_str)

            # Step 3c: Fix double commas and trailing commas
            json_str = re.sub(r',,+', ',', json_str)  # Fix double commas
            json_str = re.sub(r',\s*}', '}', json_str)  # Fix trailing commas
            json_str = re.sub(r',\s*]', ']', json_str)  # Fix trailing commas before ]

            # Pattern 4: Fix final malformed structure like "}], "
            json_str = re.sub(r'}\],\s*$', '}]', json_str)  # Remove trailing comma after array
            json_str = re.sub(r'}\]\],\s*$', '}]', json_str)  # Fix double closing brackets with comma


            # Pattern 5: Fix malformed quotes
            json_str = json_str.replace('\\"', '"')

            # Parse the fixed JSON
            data = json.loads(json_str)

            # Handle both single dict and list of dicts
            if isinstance(data, dict):
                data = [data]

            # Extract bounding boxes with coordinate scaling
            pred_bboxes = [[
                int(item['bbox_2d'][0] * x_factor + 0.5),
                int(item['bbox_2d'][1] * y_factor + 0.5),
                int(item['bbox_2d'][2] * x_factor + 0.5),
                int(item['bbox_2d'][3] * y_factor + 0.5)
            ] for item in data if 'bbox_2d' in item and len(item['bbox_2d']) == 4]

            # Extract points with coordinate scaling
            pred_points = [[
                int(item['point_2d'][0] * x_factor + 0.5),
                int(item['point_2d'][1] * y_factor + 0.5)
            ] for item in data if 'point_2d' in item and len(item['point_2d']) == 2]

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw JSON: {json_match.group(1)}")
            pred_bboxes = []
            pred_points = []

    # Extract thinking process
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, output_text, re.DOTALL)
    think_text = think_match.group(1).strip() if think_match else ""

    # Extract rethinking process
    rethink_pattern = r'<rethink>(.*?)</rethink>'
    rethink_match = re.search(rethink_pattern, output_text, re.DOTALL)
    rethink_text = rethink_match.group(1).strip() if rethink_match else ""

    return pred_bboxes, pred_points, think_text, rethink_text

def create_visualization(image, mask_all, bboxes, points, output_path):
    """Create visualization with original image and predicted mask"""
    plt.figure(figsize=(16, 8))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image', fontsize=14)
    plt.axis('off')
    
    # Image with mask and annotations
    plt.subplot(1, 2, 2)
    plt.imshow(image, alpha=0.7)
    
    # Draw mask overlay
    mask_overlay = np.zeros_like(image)
    mask_overlay[mask_all] = [255, 0, 0]  # Red mask
    plt.imshow(mask_overlay, alpha=0.4)
    
    # Draw bounding boxes and points
    for bbox, point in zip(bboxes, points):
        # Bounding box
        x1, y1, x2, y2 = bbox
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, color='yellow', linewidth=2)
        plt.gca().add_patch(rect)
        
        # Point
        plt.plot(point[0], point[1], 'yo', markersize=8, markeredgecolor='black')
    
    plt.title('Predicted Affordance Regions', fontsize=14)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_results(output_dir, image_name, question, output_text, think_text, rethink_text, 
                bboxes, points, processing_time):
    """Save detailed results to JSON and text files"""
    
    # Create timestamped results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(image_name).stem
    
    # Save detailed JSON results
    results = {
        "timestamp": timestamp,
        "image_name": image_name,
        "question": question,
        "processing_time_seconds": processing_time,
        "raw_output": output_text,
        "thinking_process": think_text,
        "rethinking_process": rethink_text,
        "predicted_bboxes": bboxes,
        "predicted_points": points,
        "num_predictions": len(bboxes)
    }
    
    json_path = Path(output_dir) / f"{base_name}_{timestamp}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save human-readable summary
    summary_path = Path(output_dir) / f"{base_name}_{timestamp}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"AFFORDANCE-R1 INFERENCE RESULTS\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Image: {image_name}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Processing Time: {processing_time:.2f} seconds\n")
        f.write(f"Predictions Found: {len(bboxes)}\n\n")
        
        f.write(f"THINKING PROCESS:\n{'-' * 20}\n{think_text}\n\n")
        f.write(f"RETHINKING PROCESS:\n{'-' * 20}\n{rethink_text}\n\n")
        
        f.write(f"PREDICTIONS:\n{'-' * 20}\n")
        for i, (bbox, point) in enumerate(zip(bboxes, points), 1):
            f.write(f"Prediction {i}:\n")
            f.write(f"  Bounding Box: {bbox}\n")
            f.write(f"  Center Point: {point}\n\n")
        
        f.write(f"RAW MODEL OUTPUT:\n{'-' * 20}\n{output_text}\n")
    
    return json_path, summary_path


def main():
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading models...")
    start_time = datetime.now()
    
    # Load reasoning model
    reasoning_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.reasoning_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    reasoning_model.eval()
    
    # Load segmentation model
    segmentation_model = SAM2ImagePredictor.from_pretrained(args.segmentation_model_path)
    
    # Load processor
    processor = AutoProcessor.from_pretrained(args.reasoning_model_path, padding_side="left")
    
    model_load_time = (datetime.now() - start_time).total_seconds()
    print(f"Models loaded in {model_load_time:.2f} seconds")
    
    # Process image
    print(f"Processing image: {args.image_path}")
    print(f"Question: {args.question}")
    
    inference_start = datetime.now()
    
    # Load and resize image
    image = PILImage.open(args.image_path).convert("RGB")
    original_width, original_height = image.size
    resize_size = 840
    x_factor, y_factor = original_width / resize_size, original_height / resize_size
    resized_image = image.resize((resize_size, resize_size), PILImage.BILINEAR)
    
    # Create prompt
    QUESTION_TEMPLATE = (
        "Please answer \"{Question}\" with bboxs and points."
        "Analyze the functional properties of specific parts of each object in the image and carefully find all the part(s) that matches the problem."
        "Output the thinking process in <think> </think>, rethinking process in <rethink> </rethink> and final answer in <answer> </answer> tags."
        "Output the bbox(es) and point(s) and affordance tpye(s) inside the interested object(s) in JSON format."
        "i.e., <think> thinking process here </think>,"
        "<rethink> rethinking process here </rethink>,"
        "<answer>{Answer}</answer>"
    )
    
    messages = [[{
        "role": "user",
        "content": [
            {"type": "image", "image": resized_image},
            {"type": "text", "text": QUESTION_TEMPLATE.format(
                Question=args.question.lower().strip("."),
                Answer="[{\"bbox_2d\": [10,100,200,210], \"point_2d\": [30,110]}, {\"bbox_2d\": [225,296,706,786], \"point_2d\": [302,410], \"affordance\": \"grasp\"]"
            )}
        ]
    }]]
    
    # Process with reasoning model
    text = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages]
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Generate response
    print("Generating reasoning...")
    with torch.no_grad():
        generated_ids = reasoning_model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    # Log output length for debugging truncation issues
    print(f"Generated tokens: {len(generated_ids_trimmed[0])}")
    print(f"Output text length: {len(output_text)} characters")
    if not output_text.endswith('</answer>'):
        print("⚠️  Warning: Output may be truncated - missing closing </answer> tag")
    
    # Extract predictions
    bboxes, points, think_text, rethink_text = extract_bbox_points_think(
        output_text, x_factor, y_factor
    )
    
    print(f"Found {len(bboxes)} predictions")
    
    # Generate segmentation masks
    print("Generating segmentation masks...")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        mask_all = np.zeros((image.height, image.width), dtype=bool)
        segmentation_model.set_image(image)
        
        for bbox, point in zip(bboxes, points):
            masks, scores, _ = segmentation_model.predict(
                point_coords=[point],
                point_labels=[1],
                box=bbox
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            mask = masks[0].astype(bool)
            mask_all = np.logical_or(mask_all, mask)
    
    inference_time = (datetime.now() - inference_start).total_seconds()
    
    # Save results
    image_name = Path(args.image_path).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create visualization
    viz_path = output_dir / f"{Path(image_name).stem}_{timestamp}_visualization.png"
    create_visualization(image, mask_all, bboxes, points, viz_path)
    
    # Save detailed results
    json_path, summary_path = save_results(
        output_dir, image_name, args.question, output_text, 
        think_text, rethink_text, bboxes, points, inference_time
    )
    
    print(f"\nResults saved:")
    print(f"  Visualization: {viz_path}")
    print(f"  Detailed JSON: {json_path}")
    print(f"  Summary: {summary_path}")
    print(f"\nInference completed in {inference_time:.2f} seconds")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"INFERENCE SUMMARY")
    print(f"{'=' * 60}")
    print(f"Question: {args.question}")
    print(f"Predictions: {len(bboxes)}")
    print(f"\nThinking: {think_text[:100]}...")
    print(f"\nRethinking: {rethink_text[:100]}...")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()