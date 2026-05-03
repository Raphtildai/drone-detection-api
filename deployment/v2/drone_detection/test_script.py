from drone_detection.improved_multi_drone_demo import multi_drone_demo_enhanced

# Your audio files
audio_files = [
    "/home/tildai/Desktop/Development/drone-detection-api/deployment/v2/dataset_builders/output_v4/clean_drone_segments/DJI_drones_noise_comparson_seg_004_0015849_0019849_0001200_0004000_seg_001_0000000_0002920.wav",
    "/home/tildai/Desktop/Development/drone-detection-api/deployment/v2/dunakeszi_pipeline_ready_B/seg_006_hover_val_ch2.wav",
    "/home/tildai/Desktop/Development/drone-detection-api/deployment/v2/dataset_builders/clean/251020VITEMOROM1AT01A_0106500_0109000.wav"
]

# audio_files = [
#     "/home/tildai/Desktop/Development/drone-detection-api/deployment/v2/dunakeszi_pipeline_ready_B/seg_006_hover_val_ch0.wav",
#     "/home/tildai/Desktop/Development/drone-detection-api/deployment/v2/dunakeszi_pipeline_ready_B/seg_006_hover_val_ch1.wav",
#     "/home/tildai/Desktop/Development/drone-detection-api/deployment/v2/dunakeszi_pipeline_ready_B/seg_006_hover_val_ch2.wav"
# ]

# audio_files = [
#     "/home/tildai/Desktop/Development/drone-detection-api/deployment/v2/dunakeszi_test_segments_P/seg_032_circle_train_ch0.wav",
#     "/home/tildai/Desktop/Development/drone-detection-api/deployment/v2/dunakeszi_test_segments_P/seg_032_circle_train_ch1.wav",
#     "/home/tildai/Desktop/Development/drone-detection-api/deployment/v2/dunakeszi_test_segments_P/seg_032_circle_train_ch2.wav"
# ]

# Explicitly set positions for better separation
positions = [
    (6.0, 0.0),    # Drone 1: due east
    (-3.0, 5.2),   # Drone 2: northwest
    (-3.0, -5.2),  # Drone 3: southwest
]

result = multi_drone_demo_enhanced(
    audio_files=audio_files,
    positions=positions,
    visualize=True
)

# # Or run the quick test with synthetic drones
# from drone_detection.improved_multi_drone_demo import quick_test
# quick_test()