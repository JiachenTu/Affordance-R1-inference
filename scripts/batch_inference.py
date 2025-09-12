#!/usr/bin/env python3
"""
Batch Inference Script for Affordance-R1
Processes multiple images with predefined questions
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

# Test cases mapping images to their questions
TEST_CASES = {
    "chair.png": "Where should I sit if I want to have a rest?",
    "hammer.png": "When I want to use this hammer to solve something, where should I hold it?",
    "knife2.png": "To control the knife safely, where should I hold?",
    "knife3.png": "In order to control this knife more safely, which part should I hold?",
    "mug.png": "Which part should I grasp if I want to holding this mug for drinking?",
    "mug1.png": "Which part should I grasp if I want to holding this mug for drinking?"
}


def run_batch_inference():
    """Run inference on all test cases"""
    
    # Setup paths
    base_dir = Path("/home/jtu9/reasoning/Affordance-R1-inference")
    test_images_dir = base_dir / "test_images"
    results_dir = base_dir / "results"
    script_path = base_dir / "scripts" / "single_inference.py"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting batch inference on Affordance-R1 test cases")
    print("=" * 60)
    
    batch_results = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(TEST_CASES),
        "results": {},
        "summary": {}
    }
    
    total_start_time = datetime.now()
    
    for i, (image_name, question) in enumerate(TEST_CASES.items(), 1):
        image_path = test_images_dir / image_name
        
        if not image_path.exists():
            print(f"❌ {i}/{len(TEST_CASES)}: {image_name} - Image not found")
            batch_results["results"][image_name] = {
                "status": "failed",
                "error": "Image file not found"
            }
            continue
        
        print(f"🔍 {i}/{len(TEST_CASES)}: Processing {image_name}")
        print(f"    Question: {question}")
        
        case_start_time = datetime.now()
        
        try:
            # Run single inference
            cmd = [
                "python", str(script_path),
                "--image_path", str(image_path),
                "--question", question,
                "--output_dir", str(results_dir)
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                cwd=str(base_dir),
                timeout=300  # 5 minute timeout per image
            )
            
            case_time = (datetime.now() - case_start_time).total_seconds()
            
            if result.returncode == 0:
                print(f"    ✅ Completed in {case_time:.2f}s")
                batch_results["results"][image_name] = {
                    "status": "success",
                    "processing_time": case_time,
                    "question": question
                }
            else:
                print(f"    ❌ Failed with return code {result.returncode}")
                print(f"    Error: {result.stderr}")
                batch_results["results"][image_name] = {
                    "status": "failed",
                    "error": result.stderr,
                    "return_code": result.returncode
                }
                
        except subprocess.TimeoutExpired:
            print(f"    ⏰ Timeout after 5 minutes")
            batch_results["results"][image_name] = {
                "status": "timeout",
                "error": "Processing timeout after 5 minutes"
            }
            
        except Exception as e:
            print(f"    ❌ Exception: {str(e)}")
            batch_results["results"][image_name] = {
                "status": "error",
                "error": str(e)
            }
        
        print()
    
    total_time = (datetime.now() - total_start_time).total_seconds()
    
    # Generate summary
    successful = sum(1 for r in batch_results["results"].values() if r["status"] == "success")
    failed = len(TEST_CASES) - successful
    
    batch_results["summary"] = {
        "total_processing_time": total_time,
        "successful_cases": successful,
        "failed_cases": failed,
        "success_rate": successful / len(TEST_CASES) * 100
    }
    
    # Save batch results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_results_path = results_dir / f"batch_results_{timestamp}.json"
    
    with open(batch_results_path, 'w') as f:
        json.dump(batch_results, f, indent=2)
    
    # Print final summary
    print("📊 BATCH INFERENCE SUMMARY")
    print("=" * 60)
    print(f"Total Cases: {len(TEST_CASES)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {successful/len(TEST_CASES)*100:.1f}%")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Time per Case: {total_time/len(TEST_CASES):.2f} seconds")
    print(f"\nBatch results saved to: {batch_results_path}")
    
    return batch_results_path


def generate_summary_report(batch_results_path):
    """Generate a comprehensive summary report"""
    
    with open(batch_results_path) as f:
        batch_data = json.load(f)
    
    results_dir = Path(batch_results_path).parent
    
    # Find all individual result files
    result_files = list(results_dir.glob("*_results.json"))
    
    report_content = []
    report_content.append("# AFFORDANCE-R1 BATCH INFERENCE REPORT")
    report_content.append("=" * 60)
    report_content.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Total Test Cases: {batch_data['total_cases']}")
    report_content.append(f"Success Rate: {batch_data['summary']['success_rate']:.1f}%")
    report_content.append(f"Total Processing Time: {batch_data['summary']['total_processing_time']:.2f} seconds")
    report_content.append("")
    
    # Process each test case
    for image_name, question in TEST_CASES.items():
        report_content.append(f"## {image_name}")
        report_content.append(f"**Question:** {question}")
        
        batch_result = batch_data["results"].get(image_name, {})
        
        if batch_result.get("status") == "success":
            # Find corresponding detailed results
            matching_files = [f for f in result_files if image_name.replace('.png', '') in f.name]
            
            if matching_files:
                # Get the most recent result file
                latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                
                with open(latest_file) as f:
                    detailed_result = json.load(f)
                
                report_content.append(f"**Status:** ✅ Success")
                report_content.append(f"**Processing Time:** {detailed_result['processing_time_seconds']:.2f}s")
                report_content.append(f"**Predictions Found:** {detailed_result['num_predictions']}")
                report_content.append("")
                report_content.append(f"**Thinking Process:**")
                report_content.append(f"{detailed_result['thinking_process']}")
                report_content.append("")
                report_content.append(f"**Rethinking Process:**")
                report_content.append(f"{detailed_result['rethinking_process']}")
                report_content.append("")
                
                if detailed_result['predicted_bboxes']:
                    report_content.append(f"**Predictions:**")
                    for i, (bbox, point) in enumerate(zip(detailed_result['predicted_bboxes'], detailed_result['predicted_points']), 1):
                        report_content.append(f"  {i}. Bbox: {bbox}, Point: {point}")
                    report_content.append("")
            else:
                report_content.append(f"**Status:** ✅ Success (detailed results not found)")
                report_content.append("")
        else:
            status = batch_result.get("status", "unknown")
            error = batch_result.get("error", "Unknown error")
            report_content.append(f"**Status:** ❌ {status.title()}")
            report_content.append(f"**Error:** {error}")
            report_content.append("")
        
        report_content.append("-" * 40)
        report_content.append("")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = results_dir / f"batch_inference_report_{timestamp}.md"
    
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_content))
    
    print(f"📋 Detailed report saved to: {report_path}")
    return report_path


if __name__ == "__main__":
    print("🤖 Affordance-R1 Batch Inference Tool")
    print("This will process all test images with their corresponding questions")
    print()
    
    # Run batch inference
    batch_results_path = run_batch_inference()
    
    # Generate detailed report
    print("\n📋 Generating detailed report...")
    report_path = generate_summary_report(batch_results_path)
    
    print(f"\n✨ Batch inference completed!")
    print(f"Results are available in: /home/jtu9/reasoning/Affordance-R1-inference/results/")