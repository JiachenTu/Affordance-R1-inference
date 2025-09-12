# Video Frame Affordance Analysis Summary

## 📹 Video Processing Results

### **Input Video:**
- **Source**: `/shared/BIOE486/SP25/users/ukd1/misc/testset_samples/421372/42445448/42445448.mp4`
- **Properties**:
  - Duration: 123.01 seconds (2:03)
  - Total Frames: 7,383
  - Frame Rate: 60.02 FPS  
  - Resolution: 1920x1440
  - File Size: ~192 MB

---

## 🎯 **Sampled Frames (16 frames evenly distributed)**

### **Input Directory:**
```
/home/jtu9/reasoning/Affordance-R1-inference/video_frames/421372_42445448/
```

### **Frame Files:**
1. `frame_01_t0.00s.png` - Frame #0 (0.00 seconds)
2. `frame_02_t8.20s.png` - Frame #492 (8.20 seconds)
3. `frame_03_t16.39s.png` - Frame #984 (16.39 seconds)
4. `frame_04_t24.59s.png` - Frame #1476 (24.59 seconds)
5. `frame_05_t32.79s.png` - Frame #1968 (32.79 seconds)
6. `frame_06_t40.99s.png` - Frame #2460 (40.99 seconds)
7. `frame_07_t49.18s.png` - Frame #2952 (49.18 seconds)
8. `frame_08_t57.38s.png` - Frame #3444 (57.38 seconds)
9. `frame_09_t65.59s.png` - Frame #3937 (65.59 seconds)
10. `frame_10_t73.79s.png` - Frame #4429 (73.79 seconds)
11. `frame_11_t81.99s.png` - Frame #4921 (81.99 seconds)
12. `frame_12_t90.19s.png` - Frame #5413 (90.19 seconds)
13. `frame_13_t98.38s.png` - Frame #5905 (98.38 seconds)
14. `frame_14_t106.58s.png` - Frame #6397 (106.58 seconds)
15. `frame_15_t114.78s.png` - Frame #6889 (114.78 seconds)
16. `frame_16_t122.99s.png` - Frame #7382 (122.99 seconds)

---

## 🤖 **Affordance Prediction Results**

### **Output Directory:**
```
/home/jtu9/reasoning/Affordance-R1-inference/results/video_421372_42445448/
```

### **Enhanced Model Configuration:**
- **Max New Tokens**: Increased from 1024 → 3072 
- **Early Stopping**: Disabled to ensure complete outputs
- **Repetition Penalty**: 1.1 for better quality
- **Temperature**: 1.0 for consistent generation

### **Question Used:**
```
"What actions can be performed with objects in this image?"
```

### **Result Files Structure:**
For each frame, three files are generated:

#### **Example for frame_01_t0.00s.png:**
```
frame_01_t0.00s_YYYYMMDD_HHMMSS_visualization.png  # Visual result with bounding boxes
frame_01_t0.00s_YYYYMMDD_HHMMSS_results.json       # Detailed JSON with predictions  
frame_01_t0.00s_YYYYMMDD_HHMMSS_summary.txt        # Human-readable summary
```

### **Model Output Improvements:**
✅ **Longer, more complete reasoning processes**
✅ **Detailed thinking and rethinking sections**  
✅ **Multiple object detection attempts**
✅ **Complex scene analysis capabilities**

⚠️ **Current JSON Parsing Challenges:**
- Complex multi-object predictions with formatting variations
- Occasional truncated affordance labels (`"aff:action"` vs `"affordance":"action"`)
- Multiple bounding box structures requiring enhanced parsing

---

## 🛠️ **Scripts Created:**

### **1. Frame Sampling Script:**
```
/home/jtu9/reasoning/Affordance-R1-inference/scripts/video_frame_sampler.py
```
**Usage:**
```bash
python video_frame_sampler.py \
  --video_path "/shared/BIOE486/SP25/users/ukd1/misc/testset_samples/421372/42445448/42445448.mp4" \
  --output_dir "/home/jtu9/reasoning/Affordance-R1-inference/video_frames/421372_42445448" \
  --num_frames 16
```

### **2. Enhanced Inference Script:**
```
/home/jtu9/reasoning/Affordance-R1-inference/scripts/single_inference.py
```
**Key Improvements:**
- Increased max_new_tokens to 3072
- Enhanced JSON parsing for malformed outputs
- Truncation detection and logging
- Comprehensive result saving

### **3. Batch Processing Script:**
```
/home/jtu9/reasoning/Affordance-R1-inference/scripts/video_affordance_batch.py
```
**Usage:**
```bash
python video_affordance_batch.py \
  --frames_dir "/home/jtu9/reasoning/Affordance-R1-inference/video_frames/421372_42445448" \
  --results_dir "/home/jtu9/reasoning/Affordance-R1-inference/results/video_421372_42445448"
```

---

## 📊 **Model Performance Analysis:**

### **Generation Quality:**
- **Token Generation**: ~150-250 tokens per prediction
- **Output Completeness**: Significantly improved with longer max_tokens
- **Reasoning Quality**: Detailed, coherent thinking processes
- **Processing Time**: ~5-8 seconds per frame

### **Example Output Quality:**
```
<think>
The question asks for an analysis of what actions could be performed 
on the objects within the image...
</think>

<rethink>  
The objects present include chairs which might allow for sitting 
action, and other furniture pieces that could support various 
interactions...
</rethink>

<answer>[{"bbox_2d": [560,212,839,547], "point_2d": [703,345], "affordance": "sit"}]</answer>
```

---

## 🎯 **Next Steps & Recommendations:**

1. **JSON Parsing Enhancement**: Implement more robust parsing for complex multi-object scenarios
2. **Batch Processing**: Run full 16-frame batch with improved error handling
3. **Question Refinement**: Test with more specific affordance questions
4. **Temporal Analysis**: Analyze affordance changes across video timeline
5. **Performance Optimization**: Consider model quantization for faster processing

---

## 📂 **Quick Access Paths:**

### **Input:**
- **Video**: `/shared/BIOE486/SP25/users/ukd1/misc/testset_samples/421372/42445448/42445448.mp4`
- **Frames**: `/home/jtu9/reasoning/Affordance-R1-inference/video_frames/421372_42445448/`

### **Output:**
- **Results**: `/home/jtu9/reasoning/Affordance-R1-inference/results/video_421372_42445448/`
- **Scripts**: `/home/jtu9/reasoning/Affordance-R1-inference/scripts/`

### **Analysis:**
- **This Report**: `/home/jtu9/reasoning/Affordance-R1-inference/VIDEO_PROCESSING_SUMMARY.md`