# Qwen2.5-VL Video Understanding Scripts

This directory contains scripts for video understanding and grounding using the Qwen2.5-VL-7B-Instruct model.

## 🚀 Quick Start

### Environment Setup
```bash
source ~/.bashrc
conda activate Affordance-R1
cd /home/jtu9/reasoning/Affordance-R1-inference/qwen_scripts
```

## 📋 Available Scripts

### 1. Core Video Inference (`qwen_video_inference.py`)
Main script for video understanding tasks.

**Usage:**
```bash
# Video Summarization
python qwen_video_inference.py \
  --video_path "/shared/BIOE486/SP25/users/ukd1/misc/testset_samples/421372/42445448/42445448.mp4" \
  --task summarization \
  --gpu_id 1

# Video Captioning
python qwen_video_inference.py \
  --video_path "/shared/BIOE486/SP25/users/ukd1/misc/testset_samples/421372/42445448/42445448.mp4" \
  --task captioning \
  --gpu_id 2

# Video Grounding
python qwen_video_inference.py \
  --video_path "/shared/BIOE486/SP25/users/ukd1/misc/testset_samples/421372/42445448/42445448.mp4" \
  --task grounding \
  --gpu_id 3

# Custom Question
python qwen_video_inference.py \
  --video_path "/shared/BIOE486/SP25/users/ukd1/misc/testset_samples/421372/42445448/42445448.mp4" \
  --task custom \
  --question "When does the camera move to show different parts of the room?" \
  --gpu_id 4
```

**Parameters:**
- `--video_path`: Path to input video file
- `--task`: Task type (`summarization`, `captioning`, `grounding`, `custom`)
- `--question`: Custom question (required for `custom` task)
- `--gpu_id`: GPU ID to use (1-7, avoid GPU 0)
- `--max_pixels`: Maximum pixels for processing (default: 360*420)
- `--fps`: Frame rate for sampling (default: 1.0)
- `--output_dir`: Output directory (default: `./results`)

### 2. Specialized Video Grounding (`video_grounding.py`)
Advanced grounding with predefined question types.

**Usage:**
```bash
# Action Detection
python video_grounding.py \
  --video_path "/shared/BIOE486/SP25/users/ukd1/misc/testset_samples/421372/42445448/42445448.mp4" \
  --question_type action_detection \
  --gpu_id 1

# Object Interaction
python video_grounding.py \
  --video_path "/shared/BIOE486/SP25/users/ukd1/misc/testset_samples/421372/42445448/42445448.mp4" \
  --question_type object_interaction \
  --gpu_id 2

# Movement Tracking
python video_grounding.py \
  --video_path "/shared/BIOE486/SP25/users/ukd1/misc/testset_samples/421372/42445448/42445448.mp4" \
  --question_type movement_tracking \
  --gpu_id 3
```

**Question Types:**
- `action_detection`: Detect specific actions and activities
- `object_interaction`: Track object interactions
- `movement_tracking`: Track movement and motion
- `scene_changes`: Identify scene transitions
- `speech_events`: Detect speech/audio events
- `temporal_sequence`: Complete event sequence
- `specific_moment`: Find specific events
- `duration_analysis`: Analyze event durations
- `periodic_events`: Detect repeating events
- `comparative_timing`: Compare event timing

### 3. Video Utilities (`video_utils.py`)
Helper functions for video processing and analysis.

**Usage:**
```bash
# Get video information
python video_utils.py
```

### 4. Task-Specific Frame Grounding (`task_frame_grounding.py`)
**LEGACY** Single-stage frame identification (superseded by enhanced two-stage system).

**Usage:**
```bash
# Analyze all predefined tasks
python task_frame_grounding.py \
  --video_path "/shared/BIOE486/SP25/users/ukd1/misc/testset_samples/421372/42445448/42445448.mp4" \
  --gpu_id 1
```

### 5. **Enhanced Two-Stage Task Grounding System** ⭐ **RECOMMENDED**

#### **5a. Complete Pipeline (`two_stage_task_pipeline.py`)**
**NEW!** Complete end-to-end two-stage system for maximum accuracy and recall.

**Usage:**
```bash
# Complete pipeline with all tasks
python two_stage_task_pipeline.py \
  --video_path "/shared/BIOE486/SP25/users/ukd1/misc/testset_samples/421372/42445448/42445448.mp4" \
  --stage1_gpu 1 --stage2_gpu 2 --stage3_gpu 3

# Specific tasks with custom parameters
python two_stage_task_pipeline.py \
  --video_path "video.mp4" \
  --tasks cabinet_top_left tv_remote_footrest power_outlet \
  --max_frames 16 --top_k 3 \
  --stage1_gpu 4 --stage2_gpu 5 --stage3_gpu 6
```

#### **5b. Individual Stage Scripts**
For advanced users who want to run stages separately:

```bash
# Stage 1: High-recall frame extraction (up to 16 candidates per task)
python enhanced_task_frame_extractor.py \
  --video_path "video.mp4" --max_frames 16 --gpu_id 1

# Stage 2: VLM validation (select top 3 per task)
python vlm_frame_validator.py \
  --stage1_metadata "stage1_extraction_*.json" --top_k 3 --gpu_id 2

# Stage 3: Bbox/point extraction + SAM2 masks
python task_bbox_processor.py \
  --stage2_metadata "stage2_validation_*.json" --gpu_id 3
```

**Enhanced Output Structure:**
```
enhanced_task_grounding/
├── stage1_candidates/      # Up to 16 candidate frames per task
│   ├── cabinet_operations/
│   ├── tv_controls/
│   ├── room_environment/
│   └── other_actions/
├── stage2_selected/        # Top 3 validated frames per task
│   └── [same structure with confidence scores]
├── bbox_results/          # Bbox/point coordinates (JSON)
│   └── [detailed localization results]
└── final_masks/           # SAM2 segmentation + visualizations
    └── [overlay images with masks and bboxes]
```

**Key Improvements:**
- **Higher Recall**: 16 candidates vs 1 frame per task
- **Better Accuracy**: VLM double-checking with 1-10 scoring
- **Precise Localization**: Bbox/point extraction + SAM2 masks
- **Rich Visualization**: Multi-panel overlays (original + bbox + mask)
- **Temporal Diversity**: Frames spread across video timeline
- **Quality Filtering**: Brightness/contrast/sharpness assessment

**Predefined Tasks:**
- **Cabinet Operations**: `cabinet_top_left`, `cabinet_top_right`, `cabinet_bottom_left`, `cabinet_bottom_right`
- **TV Controls**: `tv_remote_footrest`, `tv_remote_table`
- **Room Environment**: `radiator_thermostat`, `window_above_radiator`, `ceiling_light`
- **Other Actions**: `door_close`, `power_outlet`

### 5. Batch Processing (`batch_video_pipeline.py`)
Comprehensive batch analysis with multiple tasks.

**Usage:**
```bash
# Run selected tasks
python batch_video_pipeline.py \
  --video_path "/shared/BIOE486/SP25/users/ukd1/misc/testset_samples/421372/42445448/42445448.mp4" \
  --tasks video_summary action_grounding \
  --gpu_id 1
```

## 📊 Output Structure

Results are saved in structured directories:
```
results/
├── summarization/     # Video summaries
├── captioning/        # Frame-by-frame captions
├── grounding/         # Temporal grounding results
└── custom/           # Custom query results
```

Each task generates:
- `*_results.json`: Detailed results with metadata
- `*_summary.txt`: Human-readable summary
- Timeline visualizations (for grounding tasks)

## 🎯 Example Results

### Video Information
- **Duration:** 2:03.01 (123.0s)
- **Resolution:** 1920x1440
- **FPS:** 60.02
- **Total Frames:** 7,383
- **File Size:** 192.2 MB

### Sample Grounding Output
```
TIMESTAMPS FOUND:
  0.0s - Video begins, living room view
  29.0s - Camera transitions to Christmas tree area
  45.0s - Focus shifts to decorative items
  62.0s - Camera pans to show kitchen area
```

## 🔧 Configuration

### Model Configuration
- **Model Path:** `/home/jtu9/reasoning/models/Qwen2.5-VL-7B-Instruct`
- **Recommended GPUs:** 1-7 (avoid GPU 0)
- **Processing Time:** ~150-170 seconds per task
- **Memory Requirements:** ~10-15GB GPU memory

### Performance Tips
1. Use different GPUs (1-7) to avoid conflicts
2. Adjust `max_pixels` for memory constraints
3. Lower `fps` for faster processing
4. Use `timeout` parameter for long videos

## 📝 Notes

- Model loading takes ~3-20 seconds (varies by GPU)
- Processing time: ~150-170s per task
- Results include timestamp extraction and temporal analysis
- All scripts support the Affordance-R1 conda environment
- GPU memory usage: ~10-15GB per inference