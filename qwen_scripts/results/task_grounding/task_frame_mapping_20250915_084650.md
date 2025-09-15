# Task-Specific Frame Grounding Report

## 📹 Video Information

- **Video:** /shared/BIOE486/SP25/users/ukd1/misc/testset_samples/421372/42445448/42445448.mp4
- **Duration:** 2:03.01
- **Resolution:** 1920x1440
- **FPS:** 60.02
- **Total Frames:** 7,383

## 📊 Analysis Summary

- **Total Tasks:** 11
- **Successfully Mapped:** 3
- **Frames Extracted:** 3
- **Success Rate:** 27.3%

## 🎯 Task Mapping Results

### Cabinet Operations

#### Open the top left cabinet door
- **Best Frame Timestamp:** 0:45.00 (45.00s)
- **Confidence Score:** 2.5
- **Frame Path:** `/home/jtu9/reasoning/Affordance-R1-inference/qwen_scripts/results/task_grounding/task_frames/cabinet_operations/cabinet_top_left_t45.00s.png`
- **Context:** ### Task Analysis:

#### 1. **Cabinet Operations**
   - **Objective:** Open the top left cabinet door.
   - **Optimal Timestamp:** 45.0 seconds
   - **Reasoning:** At this timestamp, the cabinet doors...

### TV Controls

#### Turn on the TV using one of the remote controls on the footrest
- **Best Frame Timestamp:** 0:38.00 (38.00s)
- **Confidence Score:** 2.0
- **Frame Path:** `/home/jtu9/reasoning/Affordance-R1-inference/qwen_scripts/results/task_grounding/task_frames/tv_controls/tv_remote_footrest_t38.00s.png`
- **Context:** structions.

#### 2. **TV Controls**
   - **Objective:** Turn on the TV using one of the remote controls on the footrest.
   - **Optimal Timestamp:** 38.0 seconds
   - **Reasoning:** Around 38.0 secon...

### Other Actions

#### Plug the device in one of the power outlets next to the cabinet
- **Best Frame Timestamp:** 0:45.00 (45.00s)
- **Confidence Score:** 2.5
- **Frame Path:** `/home/jtu9/reasoning/Affordance-R1-inference/qwen_scripts/results/task_grounding/task_frames/other_actions/power_outlet_t45.00s.png`
- **Context:** ures that the power outlet is clearly visible and accessible for plugging in the device.

### Detailed Breakdown of Frames:

1. **Cabinet Operations (45.0 seconds):**
   - The cabinet doors are open, ...

## 📁 Extracted Frames Directory Structure

```
task_frames/
├── cabinet_operations/
│   └── cabinet_top_left_t45.00s.png
├── tv_controls/
│   └── tv_remote_footrest_t38.00s.png
├── other_actions/
│   └── power_outlet_t45.00s.png
```

## 🚀 Usage Instructions

1. **Review Task Mappings:** Check the confidence scores and timestamps
2. **Examine Extracted Frames:** Navigate to the `task_frames/` directory
3. **Verify Frame Quality:** Ensure objects are clearly visible in extracted frames
4. **Use for Training:** These frames can be used as training data for task-specific models
