#!/usr/bin/env python3
"""
Qwen2.5-VL Video Understanding and Grounding Script
Supports video grounding, captioning, and summarization tasks
Based on PyImageSearch implementation
"""

import os
import sys
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from PIL import Image as PILImage
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL video understanding tasks")
    parser.add_argument("--model_path", type=str,
                       default="/home/jtu9/reasoning/models/Qwen2.5-VL-7B-Instruct",
                       help="Path to the Qwen2.5-VL model")
    parser.add_argument("--video_path", type=str, required=True,
                       help="Path to the input video file")
    parser.add_argument("--task", type=str, choices=['grounding', 'captioning', 'summarization', 'custom'],
                       default='grounding', help="Type of video understanding task")
    parser.add_argument("--question", type=str,
                       help="Custom question for the video (required for custom task)")
    parser.add_argument("--output_dir", type=str,
                       default="/home/jtu9/reasoning/Affordance-R1-inference/qwen_scripts/results",
                       help="Directory to save results")
    parser.add_argument("--max_pixels", type=int, default=360*420,
                       help="Maximum pixels for video processing")
    parser.add_argument("--fps", type=float, default=1.0,
                       help="Frame rate for video sampling")
    parser.add_argument("--gpu_id", type=str, default="7",
                       help="GPU ID to use")
    return parser.parse_args()


def get_task_prompt(task, custom_question=None):
    """Generate appropriate prompt based on task type"""
    prompts = {
        'grounding': "Analyze this video and identify when specific events or actions occur. Provide precise timestamps for when things happen.",
        'captioning': "Provide a detailed frame-by-frame or segment-by-segment description of what happens in this video. Include timing information where relevant.",
        'summarization': "Provide a comprehensive summary of this video, including the main events, people, objects, and activities that occur.",
        'custom': custom_question if custom_question else "Describe what you see in this video."
    }
    return prompts.get(task, prompts['custom'])


def extract_timestamps_from_output(output_text):
    """Extract timestamp information from model output"""
    # Look for various timestamp formats
    timestamp_patterns = [
        r'(\d{1,2}):(\d{2}):(\d{2})',  # HH:MM:SS
        r'(\d{1,2}):(\d{2})',          # MM:SS
        r'at (\d+\.?\d*)\s*seconds?',   # at X seconds
        r'(\d+\.?\d*)\s*s',            # X s
        r'(\d+\.?\d*)\s*sec',          # X sec
        r'timestamp\s*(\d+\.?\d*)',     # timestamp X
    ]

    timestamps = []
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

            timestamps.append({
                'timestamp_seconds': total_seconds,
                'original_text': match.group(0),
                'context': output_text[max(0, match.start()-50):match.end()+50]
            })

    return timestamps


def save_results(output_dir, video_name, task, question, output_text, timestamps, processing_time):
    """Save comprehensive results to files"""

    # Create task-specific output directory
    task_dir = Path(output_dir) / task
    task_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamped results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(video_name).stem

    # Save detailed JSON results
    results = {
        "timestamp": timestamp,
        "video_name": video_name,
        "task": task,
        "question": question,
        "processing_time_seconds": processing_time,
        "raw_output": output_text,
        "extracted_timestamps": timestamps,
        "num_timestamps": len(timestamps)
    }

    json_path = task_dir / f"{base_name}_{task}_{timestamp}_results.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save human-readable summary
    summary_path = task_dir / f"{base_name}_{task}_{timestamp}_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"QWEN2.5-VL VIDEO {task.upper()} RESULTS\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Video: {video_name}\n")
        f.write(f"Task: {task}\n")
        f.write(f"Question: {question}\n")
        f.write(f"Processing Time: {processing_time:.2f} seconds\n")
        f.write(f"Timestamps Found: {len(timestamps)}\n\n")

        if timestamps:
            f.write(f"EXTRACTED TIMESTAMPS:\n{'-' * 30}\n")
            for i, ts in enumerate(timestamps, 1):
                f.write(f"{i}. {ts['timestamp_seconds']:.1f}s - {ts['original_text']}\n")
                f.write(f"   Context: ...{ts['context']}...\n\n")

        f.write(f"FULL MODEL OUTPUT:\n{'-' * 30}\n{output_text}\n")

    return json_path, summary_path


def main():
    args = parse_args()

    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Validate inputs
    if args.task == 'custom' and not args.question:
        raise ValueError("Custom question is required for custom task")

    if not Path(args.video_path).exists():
        raise FileNotFoundError(f"Video file not found: {args.video_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🎬 Qwen2.5-VL Video Understanding")
    print(f"{'=' * 60}")
    print(f"Video: {args.video_path}")
    print(f"Task: {args.task}")
    print(f"Model: {args.model_path}")
    print()

    print(f"Loading model...")
    start_time = datetime.now()

    # Load model and processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path, padding_side="left")

    model_load_time = (datetime.now() - start_time).total_seconds()
    print(f"Model loaded in {model_load_time:.2f} seconds")

    # Get task-specific prompt
    question = get_task_prompt(args.task, args.question)
    print(f"Question: {question}")
    print()

    # Process video
    print(f"Processing video...")
    inference_start = datetime.now()

    # Create messages for video processing
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": f"file://{args.video_path}",
                "max_pixels": args.max_pixels,
                "fps": args.fps
            },
            {"type": "text", "text": question}
        ]
    }]

    # Process with model
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Generate response
    print("Generating response...")
    with torch.no_grad():
        generated_ids = model.generate(
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
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    inference_time = (datetime.now() - inference_start).total_seconds()

    # Extract timestamps for grounding tasks
    timestamps = []
    if args.task in ['grounding', 'captioning']:
        timestamps = extract_timestamps_from_output(output_text)

    print(f"Found {len(timestamps)} timestamps" if timestamps else "Processing completed")

    # Save results
    video_name = Path(args.video_path).name
    json_path, summary_path = save_results(
        output_dir, video_name, args.task, question,
        output_text, timestamps, inference_time
    )

    print(f"\nResults saved:")
    print(f"  Detailed JSON: {json_path}")
    print(f"  Summary: {summary_path}")
    print(f"\nProcessing completed in {inference_time:.2f} seconds")

    # Print preview
    print(f"\n{'=' * 60}")
    print(f"RESPONSE PREVIEW")
    print(f"{'=' * 60}")
    print(output_text[:500] + "..." if len(output_text) > 500 else output_text)

    if timestamps:
        print(f"\nTIMESTAMPS FOUND:")
        for ts in timestamps[:5]:  # Show first 5
            print(f"  {ts['timestamp_seconds']:.1f}s - {ts['original_text']}")
        if len(timestamps) > 5:
            print(f"  ... and {len(timestamps) - 5} more")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()