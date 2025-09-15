#!/bin/bash

# Test Single Frame Inference Script for Affordance-R1
# Usage: ./test_single_frame.sh <frame_number> [question]
# Example: ./test_single_frame.sh 10 "What actions can be performed with objects in this image?"

set -e  # Exit on any error

# Default values
FRAME_NUMBER=${1:-10}
QUESTION=${2:-"What actions can be performed with objects in this image?"}
BASE_DIR="/home/jtu9/reasoning/Affordance-R1-inference"
FRAMES_DIR="$BASE_DIR/video_frames/421372_42445448"
RESULTS_DIR="$BASE_DIR/results/single_frame_tests"
SCRIPT_PATH="$BASE_DIR/scripts/single_inference.py"

echo "🧪 Testing Single Frame Inference for Affordance-R1"
echo "=================================================="
echo "Frame Number: $FRAME_NUMBER"
echo "Question: $QUESTION"
echo "Frames Directory: $FRAMES_DIR"
echo "Results Directory: $RESULTS_DIR"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"

# Find the frame file
FRAME_FILE=$(find "$FRAMES_DIR" -name "frame_${FRAME_NUMBER}_*.png" | head -1)

if [ -z "$FRAME_FILE" ]; then
    echo "❌ Error: Frame $FRAME_NUMBER not found!"
    echo "Available frames:"
    ls "$FRAMES_DIR" | grep "frame_" | head -10
    echo "... (showing first 10)"
    exit 1
fi

echo "🎯 Found frame file: $(basename "$FRAME_FILE")"
echo "📝 Question: $QUESTION"
echo ""

# Check if conda environment exists
if ! conda info --envs | grep -q "Affordance-R1"; then
    echo "❌ Error: Conda environment 'Affordance-R1' not found!"
    echo "Available environments:"
    conda info --envs
    exit 1
fi

echo "🚀 Starting inference..."
echo "Command: python $SCRIPT_PATH --image_path \"$FRAME_FILE\" --question \"$QUESTION\" --output_dir \"$RESULTS_DIR\""
echo ""

# Activate conda environment and run inference
start_time=$(date +%s)

if conda run -n Affordance-R1 python "$SCRIPT_PATH" \
    --image_path "$FRAME_FILE" \
    --question "$QUESTION" \
    --output_dir "$RESULTS_DIR"; then

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    echo ""
    echo "✅ Inference completed successfully in ${duration} seconds!"
    echo ""

    # Show results
    echo "📊 Generated files:"
    find "$RESULTS_DIR" -name "*$(basename "$FRAME_FILE" .png)*" -type f | while read -r file; do
        echo "   - $(basename "$file") ($(stat -c%s "$file") bytes)"
    done

    echo ""
    echo "🔍 Latest result files:"
    ls -lt "$RESULTS_DIR" | head -5

    echo ""
    echo "📂 Results directory: $RESULTS_DIR"

else
    echo ""
    echo "❌ Inference failed!"
    exit 1
fi