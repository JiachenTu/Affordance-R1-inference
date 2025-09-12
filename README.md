# Affordance-R1 Inference Tools

Enhanced inference tools for the Affordance-R1 model with improved JSON parsing and video frame processing capabilities.

## Features

- Single image inference with enhanced output parsing
- Batch processing for multiple images
- Video frame sampling and analysis
- Robust JSON parsing for malformed model outputs
- Comprehensive result visualization and logging

## Usage

### Single Image Inference
```bash
python scripts/single_inference.py --image_path IMAGE_PATH --question "QUESTION"
```

### Video Frame Sampling
```bash
python scripts/video_frame_sampler.py --video_path VIDEO_PATH --output_dir OUTPUT_DIR --num_frames 16
```

### Batch Processing
```bash
python scripts/batch_inference.py
```

## Requirements

- Conda environment: `Affordance-R1`
- Dependencies from original Affordance-R1 repository
- OpenCV for video processing

## Model Improvements

- Increased max_new_tokens from 1024 to 3072
- Enhanced JSON parsing for malformed outputs
- Truncation detection and warnings
- Better visualization with bounding boxes and points