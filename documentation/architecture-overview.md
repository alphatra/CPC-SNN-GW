# Architecture Overview

## System Design

The CPC-SNN-GW system represents a revolutionary architecture for gravitational wave (GW) detection, integrating Contrastive Predictive Coding (CPC) with Spiking Neural Networks (SNN) in an end-to-end neuromorphic pipeline. This design leverages self-supervised learning for robust feature extraction from raw strain data, followed by energy-efficient, biologically-inspired classification.


The system's architecture is modular and designed for maintainability, with a clear separation of concerns across data processing, model components, and training infrastructure. The core innovation lies in the seamless integration of CPC for unsupervised representation learning and SNN for event-driven, low-power signal classification, creating a pipeline that is both scientifically rigorous and computationally efficient.


## Data Flow

The data processing pipeline is designed to handle both real and synthetic gravitational wave data with high fidelity. The flow begins with the ingestion of raw strain data from the LIGO Open Science Center (LOSC) via the ReadLIGO library. The system processes the authentic GW150914 strain data, applying a 100x normalization to match realistic detector noise levels.

The data is then segmented into overlapping windows (e.g., 256 or 512 samples) to create a dataset of time-series segments. A critical step in the pipeline is the application of a stratified train/test split, which ensures balanced class representation in both training and evaluation sets, preventing the misleading results of "fake accuracy" that can occur with single-class test sets.

For data augmentation and enhancement, the system employs a glitch injection module that introduces realistic noise artifacts into the training data, improving the model's robustness to real-world detector noise. The final dataset is structured for efficient loading and processing during training.

## Component Interaction

The system's components interact through a well-defined API, ensuring loose coupling and high cohesion. The primary interaction flow is as follows:

1.  **Data Loader**: The `real_ligo_integration.py` module is responsible for data acquisition and preprocessing. It provides the `create_real_ligo_dataset` function, which returns a tuple of `(train_signals, train_labels), (test_signals, test_labels)`.

2.  **Model Pipeline**: The core model, composed of the CPC encoder, Spike Bridge, and SNN classifier, receives the preprocessed signals. The CPC encoder processes the input to generate a sequence of latent features. These features are then passed to the Spike Bridge, which converts the continuous representations into discrete spike trains using a temporal-contrast encoding scheme.

3.  **Training and Evaluation**: The `UnifiedTrainer` orchestrates the training process. It initializes the model, manages the training loop, and applies the enhanced loss function from `cpc_loss_fixes.py`. After training, the `test_evaluation.py` module is invoked to perform a comprehensive analysis on the test set, calculating real accuracy, detecting model collapse, and generating a professional summary report.

4.  **Entry Points**: The system is accessed through three primary entry points:
    *   `cli.py`: The main command-line interface, which applies a 6-stage GPU warmup to prevent CUDA timing issues before initiating training or evaluation.
    *   `enhanced_cli.py`: An advanced CLI with additional features like enhanced logging and gradient accumulation.
    *   `run_advanced_pipeline.py`: A script that executes a multi-phase training pipeline, integrating all components in a production-ready workflow.

This component interaction ensures a robust, reproducible, and scientifically valid workflow from data ingestion to model evaluation.