# tests.py

from matplotlib import pyplot as plt
from matplotlib.path import Path
import numpy as np
import tempfile, os
from IPython.display import display, Audio, clear_output
import ipywidgets as widgets
from visualizer import DroneVisualizer
from config import config
from functions import DroneDetectionSystemFunctions
import soundfile as sf

def run_automated_tests_on_generated_files():
    """Automatically test all generated synthetic files"""
    print("🧪 AUTOMATED TESTING ON GENERATED FILES")
    print("=" * 60)

    # Initialize DroneDetectionSystemFunctions instance
    dds = DroneDetectionSystemFunctions()
    dds.load_best_model()  # Load the model once

    test_files = [
        ("3-channel recording", "recording_3channel.wav", [0.85, 0.30]),
        ("Clean drone - 3 files", ["mic1.wav", "mic2.wav", "mic3.wav"], [0.85, 0.30]),
        ("Indoor close", ["mic1_indoor.wav", "mic2_indoor.wav", "mic3_indoor.wav"], [-0.42, 1.05]),
        ("Outdoor far", ["mic1_outdoor.wav", "mic2_outdoor.wav", "mic3_outdoor.wav"], [3.2, -1.1]),
    ]

    results = []
    visualizer = DroneVisualizer(config)

    for test_name, files, true_position in test_files:
        print(f"\n🔧 Testing: {test_name}")
        print(f"📍 Expected position: {true_position}")

        try:
            # Determine if it's a list of files (3 separate channels) or a single 3-channel file
            if isinstance(files, list):
                # Use 3 separate files
                result = dds.detect_and_localize_if_drone_enhanced(
                    files[0], files[1], files[2], threshold=0.65
                )
            else:
                # Single 3-channel file
                result = dds.detect_and_localize_if_drone_enhanced(files, threshold=0.65)

            # Calculate position error if detected
            position_error = None
            if result["detected"] and "position" in result:
                estimated_pos = result["position"]
                position_error = np.linalg.norm(np.array(estimated_pos) - np.array(true_position))

            results.append({
                "test_name": test_name,
                "detected": result["detected"],
                "confidence": result["probability"],
                "position_error": position_error,
                "estimated_position": result.get("position"),
                "true_position": true_position
            })

            # Create visualization
            fig, ax = visualizer.create_environment_map(true_position=true_position)

            if result["detected"] and "position" in result:
                visualizer.add_estimated_position(
                    result["position"],
                    confidence=result["probability"],
                    error=position_error
                )
                status = "✅ DETECTED"
                error_info = f" (error: {position_error:.3f}m)" if position_error else ""
            else:
                ax.text(0.5, 0.5, "DRONE NOT DETECTED",
                        transform=ax.transAxes, ha='center', va='center',
                        fontsize=16, color='red', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8))
                status = "❌ MISSED"
                error_info = ""

            visualizer.add_localization_info(probability=result["probability"], filename=test_name)
            visualizer.show()

            print(f"   Result: {status} - Confidence: {result['probability']:.3f}{error_info}")

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "test_name": test_name,
                "detected": False,
                "confidence": 0.0,
                "position_error": None,
                "error": str(e)
            })

    # Print summary
    print("\n" + "🎯 TEST SUMMARY")
    print("=" * 60)
    detected_count = sum(1 for r in results if r["detected"])
    total_tests = len(results)

    print(f"📊 Overall: {detected_count}/{total_tests} tests passed")

    for result in results:
        status = "✅ DETECTED" if result["detected"] else "❌ MISSED"
        error_info = f" (error: {result['position_error']:.3f}m)" if result["position_error"] is not None else ""
        print(f"   {result['test_name']:25} → {status} (conf: {result['confidence']:.3f}{error_info})")

    return results

def test_drone_detection_with_file_selection(threshold=0.70, analyze_long=True, show_visualization=True):
    """
    Universal drone detection tester with file selection
    """
    print("🎯 Universal Drone Detection Tester")
    print("=" * 50)
    print("📤 Select your audio file and configure detection settings")
    print(f"🎚️ Detection threshold: {threshold:.2f}")
    if analyze_long:
        print("🔍 Long audio analysis: ENABLED")
    else:
        print("🔍 Long audio analysis: DISABLED")
    print()

    # Create widgets
    uploader = widgets.FileUpload(
        accept='.wav, .mp3',
        multiple=False,
        description="Select Audio File",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='400px')
    )

    threshold_slider = widgets.FloatSlider(
        value=threshold,
        min=0.1,
        max=1.0,
        step=0.05,
        description='Threshold:',
        continuous_update=False,
        layout=widgets.Layout(width='80%')
    )

    long_audio_checkbox = widgets.Checkbox(
        value=analyze_long,
        description='Analyze entire file (for long audio)',
        disabled=False,
        layout=widgets.Layout(width='300px')
    )

    true_x_input = widgets.FloatText(value=0.0, description='True X:', layout=widgets.Layout(width='30%'))
    true_y_input = widgets.FloatText(value=0.0, description='True Y:', layout=widgets.Layout(width='30%'))
    true_position_inputs = widgets.HBox([true_x_input, true_y_input])

    test_button = widgets.Button(
        description='Start Detection',
        button_style='success',
        layout=widgets.Layout(width='200px')
    )

    output = widgets.Output()

    # Display widgets
    display(widgets.VBox([
        uploader,
        threshold_slider,
        long_audio_checkbox,
        widgets.Label("Optional - True Position (for accuracy evaluation):"),
        true_position_inputs,
        test_button,
        output
    ]))

    def on_test_button_clicked(b):
        with output:
            clear_output(wait=True)

            if not uploader.value:
                print("❌ Please select an audio file first")
                return

            # Get values from widgets
            threshold = threshold_slider.value
            analyze_long = long_audio_checkbox.value
            true_x = true_x_input.value
            true_y = true_y_input.value

            true_position = None
            if true_x != 0.0 or true_y != 0.0:
                true_position = [true_x, true_y]

            # Process uploaded file
            uploaded = list(uploader.value.values())[0]
            filename = list(uploader.value.keys())[0]
            content = uploaded['content']

            # Save to temp file
            file_ext = Path(filename).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as f:
                f.write(content)
                tmp_path = f.name

            print(f"📄 File: {filename}")
            print(f"💾 Size: {len(content)/1024:.1f} KB")
            print(f"🎚️ Threshold: {threshold:.2f}")
            print(f"🔍 Long audio analysis: {'ENABLED' if analyze_long else 'DISABLED'}")
            if true_position:
                print(f"📍 True position: {true_position}")
            print()

            try:
                # Initialize visualizer if visualization is enabled
                visualizer = DroneVisualizer(config) if show_visualization else None

                if show_visualization:
                    # Create base visualization
                    if true_position:
                        fig, ax = visualizer.create_environment_map(true_position=true_position)
                    else:
                        fig, ax = visualizer.create_environment_map()

                # Run detection
                if analyze_long:
                    print("🔍 Using enhanced long audio analysis...")
                    # Let it decide whether to use long-audio logic
                    result = DroneDetectionSystemFunctions.detect_and_localize_if_drone_enhanced(
                        tmp_path,
                        threshold=threshold,
                        analyze_long=analyze_long
                    )
                else:
                    # Try to read audio to check channels
                    try:
                        audio, sr = sf.read(tmp_path)
                        is_3ch = audio.ndim == 2 and audio.shape[1] >= 3
                        print(f"🎛️ Channels: {audio.shape[1] if is_3ch else 1}")

                        if is_3ch:
                            print("🎯 Using 3-channel detection + localization")
                            result = DroneDetectionSystemFunctions.detect_and_localize_if_drone_enhanced(tmp_path, threshold=threshold)
                        else:
                            print("🎯 Single channel → detection only")
                            result = DroneDetectionSystemFunctions.detect_and_localize_if_drone_enhanced(tmp_path, tmp_path, tmp_path, threshold=threshold)
                    except:
                        print("🎯 Using default detection")
                        result = DroneDetectionSystemFunctions.detect_and_localize_if_drone_enhanced(tmp_path, tmp_path, tmp_path, threshold=threshold)

                # Display results
                print("\n" + "—" * 50)

                is_long_result = (
                    isinstance(result, dict)
                    and "segments" in result
                    and "detection_summary" in result
                ) 
                if is_long_result: # Long audio analysis result
                    prob = result["probability"]
                    summary = result["detection_summary"]
                    segments = result["segments"]

                    if result["detected"]:
                        print(f"DRONE DETECTED in {summary['detected_segments']} segments")
                        print(f"🎯 Max confidence: {prob:.1%}")
                        print(f"📊 Segments analyzed: {summary['total_segments']}")

                        if result.get("detected") and "best_segment" in result:
                            best = result["best_segment"]
                            print(
                                f"🎯 Best detection at "
                                f"{best['start_time']:.1f}–{best['end_time']:.1f}s "
                                f"(conf: {best['probability']:.3f})"
                            )

                        if show_visualization and visualizer:
                            # Show segment analysis plot
                            print("\n📈 Generating segment analysis plot...")
                            segment_fig = visualizer.create_segment_analysis_plot(segments, filename)
                            plt.show()

                    else:
                        print(f"NO DRONE DETECTED")
                        print(f"🎯 Max confidence: {prob:.1%}")
                        print(f"📊 Segments analyzed: {summary['total_segments']}")

                        if show_visualization and visualizer:
                            # Show segment analysis plot even for no detection
                            segment_fig = visualizer.create_segment_analysis_plot(segments, filename)
                            plt.show()

                    if show_visualization and visualizer:
                        # Add segments info to main visualization
                        segments_info = {
                            'detected': summary['detected_segments'],
                            'total': summary['total_segments'],
                            'max_confidence': prob
                        }
                        visualizer.add_localization_info(probability=prob, filename=filename, segments_info=segments_info)

                    # Print detailed segment results
                    print("\n📊 Detailed Segment Results:")
                    for segment in result["segments"]:
                        status = "DETECTED" if segment["detected"] else "CLEAN"
                        print(f"   {segment['start_time']:5.1f}s - {segment['end_time']:5.1f}s: {status} (conf: {segment['probability']:.3f})")

                else:  # Standard detection result
                    prob = result["probability"]
                    if result["detected"]:
                        pos = result["position"]
                        print(f"DRONE DETECTED")
                        print(f"🎯 Confidence: {prob:.1%}")
                        print(f"📍 Position:  x = {pos[0]:.3f} m,  y = {pos[1]:.3f} m")

                        if show_visualization and visualizer:
                            # Add estimated position to visualization
                            if true_position:
                                error = np.linalg.norm(np.array(pos) - np.array(true_position))
                                visualizer.add_estimated_position(pos, confidence=prob, error=error)
                                print(f"📏 Localization error: {error:.3f} m")
                            else:
                                visualizer.add_estimated_position(pos, confidence=prob)
                    else:
                        print(f"NO DRONE")
                        print(f"🎯 Confidence: {1-prob:.1%} background")

                    if show_visualization and visualizer:
                        visualizer.add_localization_info(probability=prob, filename=filename)

                # Show the main visualization
                if show_visualization and visualizer:
                    print("\n📊 Generating main visualization...")
                    visualizer.show()

            except Exception as e:
                print(f"❌ Error processing file: {e}")
                import traceback
                traceback.print_exc()

                # Try fallback
                print("🔄 Falling back to basic detection...")
                try:
                    result = DroneDetectionSystemFunctions.detect_and_localize_if_drone_enhanced(tmp_path, tmp_path, tmp_path, threshold=threshold)
                    print(f"Basic detection result: {result}")
                except Exception as e2:
                    print(f"❌ Fallback also failed: {e2}")

            print("—" * 50)
            print("🔊 Playing audio preview:")
            display(Audio(tmp_path))

            # Cleanup
            os.unlink(tmp_path)

            print("\n✅ Analysis complete! You can test another file by clicking 'Start Detection' again.")

    test_button.on_click(on_test_button_clicked)

# ==================== SIMPLIFIED TEST FUNCTIONS ====================

def quick_test():
    """Quick test with default settings"""
    test_drone_detection_with_file_selection(threshold=0.70, analyze_long=True, show_visualization=True)

def test_with_visualization():
    """Test with visualization (same as enhanced version)"""
    test_drone_detection_with_file_selection(threshold=0.70, analyze_long=True, show_visualization=True)

def test_without_visualization():
    """Test without visualization (faster)"""
    test_drone_detection_with_file_selection(threshold=0.70, analyze_long=True, show_visualization=False)

def test_high_sensitivity():
    """Test with high sensitivity (lower threshold)"""
    test_drone_detection_with_file_selection(threshold=0.50, analyze_long=True, show_visualization=True)

def test_low_sensitivity():
    """Test with low sensitivity (higher threshold)"""
    test_drone_detection_with_file_selection(threshold=0.85, analyze_long=True, show_visualization=True)

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # 1. Clear any existing model variable from the path string error
    model = None

    # 2. Rename path to avoid conflict
    checkpoint_path = DroneDetectionSystemFunctions().get_latest_checkpoint()

    # 3. Use the config device consistently
    device = config.DEVICE

    # 4. Load the model and EXPLICITLY move it to the device
    model = DroneDetectionSystemFunctions().load_best_model()
    model.to(device) # Force sync between model and device
    model.eval()

    # Generate test files first
    print("🎵 Generating synthetic test data...")
    DroneDetectionSystemFunctions().generate_test_files()

    print("\n🎯 Drone Detection System - Comprehensive Testing")
    print("=" * 60)

    # Run automated tests on generated files
    print("\n🚀 Running automated tests on generated files...")
    test_results = run_automated_tests_on_generated_files()

    print("\n💡 Additional Testing Options:")
    print("   quick_test() - Test your own files with default settings")
    print("   test_high_sensitivity() - Lower threshold (0.50) for more detections")
    print("   test_low_sensitivity() - Higher threshold (0.85) for fewer false positives")
    print("   test_without_visualization() - Faster analysis without plots")
    print("\n   test_drone_detection_with_file_selection() - Custom settings")

    print("\n🎉 System ready! Run quick_test() to test your own audio files.")

# ==================== USAGE EXAMPLES ====================

# After the automated tests run, you can also test your own files:
# quick_test()           # Test your own files with default settings
# test_high_sensitivity() # If you're getting false negatives
# test_low_sensitivity()  # If you're getting false positives
