# -*- coding: utf-8 -*-
"""
drone_detection
───────────────
Drone detection & multi-mic localization pipeline.

Quick start
───────────
    from drone_detection import config, train_all, analyse_audio_file, launch_ui

    # Full training run
    train_all(config, det_epochs=5, loc_epochs=5, resume=False, force_rebuild_cache=True)

    # Analyse a file
    result = analyse_audio_file("drone.mp3", config)

    # Colab interactive UI
    launch_ui(config)
"""

# ── Config ────────────────────────────────────────────────────────────────────
from .config import Config, config, AUDIO_EXTS

# ── Audio processing ──────────────────────────────────────────────────────────
from .audio_processing import AudioProcessor, synthesise_drone

# ── Models ────────────────────────────────────────────────────────────────────
from .models import (
    DetectionCNN,
    LocalizationCNN,
    make_localization_model,
    FocalLoss,
    localization_loss,
)

# ── Datasets ──────────────────────────────────────────────────────────────────
from .datasets import (
    MelCacheManager,
    MelCachedDataset,
    DetectionDataset,
    LocalizationDataset,
    SyntheticLocDataset,
    SyntheticLocDatasetV2,
    DroneAudioDatasetManager,
    UaVirBASEDatasetManager,
    get_det_dataloaders,
    report_detection_split_counts,
    parse_label_json,
)

# ── Training ──────────────────────────────────────────────────────────────────
from .training import (
    TrainingLogger,
    WarmupCosineScheduler,
    DetectionTrainer,
    LocalizationTrainer,
    train_localization_v2,
    collect_val_probs,
    find_best_threshold,
    evaluate_binary_metrics,
    print_detection_report,
)

# ── Inference ─────────────────────────────────────────────────────────────────
from .inference import (
    load_detection_model,
    load_localization_model,
    reload_models,
    heuristic_detect,
    detect,
    localize,
    load_3ch,
    run_pipeline,
    analyse_audio_file,
    analyse_external_audio_robust,
)

# ── Tracking ──────────────────────────────────────────────────────────────────
from .tracking import KalmanTrack, KalmanTracker

# ── Multi-drone ───────────────────────────────────────────────────────────────
from .multidrone import localize_multi_drone

# ── Visualization ─────────────────────────────────────────────────────────────
from .visualization import (
    PLOT_STYLE,
    plot_training_logs,
    plot_confusion_matrix_styled,
    plot_polar_azimuth,
    plot_multi_drone_positions,
    plot_track_trajectory,
    plot_kalman_trajectories,
    plot_all_thesis_figures,
    plot_azimuth_mae_per_position,
    plot_val_test_comparison,
    plot_error_histogram,
    plot_predicted_vs_true,
    plot_azimuth_distance_heatmap,
    plot_training_curves,
    plot_polar_mae,
    plot_suite_results_from_data,
    plot_position_map_from_data,
)

# ── Batch testing ─────────────────────────────────────────────────────────────
from .batch_testing import batch_test_audio, diagnose_file, _verify_rms_gate

# ── Notebook / training pipeline ─────────────────────────────────────────────
from .orchestration import (
    train_detection,
    train_localization,
    train_all,
    collect_background_pool,
    generate_mixed_drone_training_audio,
    import_custom_builder_dataset,
    describe_custom_dataset,
    validate_custom_dataset_structure,
    configure_custom_dataset,
    quickstart_notebook_setup,
    quickstart_notebook_setup_v2,
    get_notebook_setup_template,
    print_notebook_setup_template,
    upload_custom_dataset_artifacts,
    audit_localization_labels,
    diagnose_uavirbase,
    verify_tdoa_accuracy,
    quick_demo,
    launch_ui,
)

# ── Utils (selected public helpers) ───────────────────────────────────────────
from .utils import (
    sigmoid,
    wrap_angle_deg,
    safe_slug,
    rms_energy,
    db_to_gain,
    normalize_peak,
    angular_error_deg,
    xy_to_azimuth_deg,
    azimuth_deg_to_xy,
    classify_detection_score,
    mix_at_snr,
    random_crop_or_loop,
    bandpass,
    gcc_phat,
    gcc_phat_peaks,
    compute_ipd_features,
    augment_waveform,
    grouped_split_paths,
    infer_group_id,
    load_audio_any,
)

__all__ = [
    # config
    "Config", "config", "AUDIO_EXTS",
    # audio
    "AudioProcessor", "synthesise_drone",
    # models
    "DetectionCNN", "LocalizationCNN",
    "make_localization_model", "FocalLoss", "localization_loss",
    # datasets
    "MelCacheManager", "MelCachedDataset", "DetectionDataset",
    "LocalizationDataset", "SyntheticLocDataset", "SyntheticLocDatasetV2",
    "DroneAudioDatasetManager", "UaVirBASEDatasetManager",
    "get_det_dataloaders", "report_detection_split_counts", "parse_label_json",
    # training
    "TrainingLogger", "WarmupCosineScheduler",
    "DetectionTrainer", "LocalizationTrainer", "train_localization_v2",
    "collect_val_probs", "find_best_threshold", "evaluate_binary_metrics",
    "print_detection_report",
    # inference
    "load_detection_model", "load_localization_model", "reload_models",
    "heuristic_detect", "detect", "localize", "load_3ch",
    "run_pipeline", "analyse_audio_file", "analyse_external_audio_robust",
    # tracking
    "KalmanTrack", "KalmanTracker",
    # multi-drone
    "localize_multi_drone",
    # visualization
    "PLOT_STYLE", "plot_training_logs", "plot_confusion_matrix_styled",
    "plot_polar_azimuth", "plot_multi_drone_positions",
    "plot_track_trajectory", "plot_kalman_trajectories",
    "plot_all_thesis_figures", "plot_azimuth_mae_per_position",
    "plot_val_test_comparison", "plot_error_histogram",
    "plot_predicted_vs_true", "plot_azimuth_distance_heatmap",
    "plot_training_curves", "plot_polar_mae",
    "plot_suite_results_from_data", "plot_position_map_from_data",
    # batch testing
    "batch_test_audio", "diagnose_file", "_verify_rms_gate",
    # notebook / pipeline
    "train_detection", "train_localization", "train_all",
    "collect_background_pool", "generate_mixed_drone_training_audio",
    "import_custom_builder_dataset", "describe_custom_dataset",
    "validate_custom_dataset_structure", "configure_custom_dataset",
    "quickstart_notebook_setup", "quickstart_notebook_setup_v2",
    "get_notebook_setup_template", "print_notebook_setup_template",
    "upload_custom_dataset_artifacts",
    "audit_localization_labels", "diagnose_uavirbase",
    "verify_tdoa_accuracy", "quick_demo", "launch_ui",
]
