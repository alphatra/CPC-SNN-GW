# Real Data Integration

## ReadLIGO Library

The foundation of the system's data integrity is the direct integration with the ReadLIGO library, which provides programmatic access to authentic gravitational wave data from the LIGO Open Science Center (LOSC). This integration marks a revolutionary shift from synthetic data generation to using real, historical detector strain, specifically the landmark GW150914 event.


The `real_ligo_integration.py` module is the central component for this integration. Its primary function, `download_gw150914_data`, directly loads the HDF5 files (`H-H1_LOSC_4_V2-1126259446-32.hdf5` and `L-L1_LOSC_4_V2-1126259446-32.hdf5`) containing 32 seconds of strain data from the Hanford (H1) and Livingston (L1) detectors. The module combines the data from both detectors by taking their average, which enhances the signal-to-noise ratio (SNR) and provides a more robust input signal.


A critical preprocessing step is the application of a 100x normalization factor to the raw strain data. This normalization is essential to bring the amplitude of the real LIGO strain into a realistic range (approximately 1e-21 to 1e-23) that matches the expected levels of astrophysical signals, as opposed to the artificially high levels often used in synthetic data.


## Data Preprocessing

The data preprocessing pipeline is designed to transform the raw, continuous strain data into a structured dataset suitable for machine learning. The first step is windowing, where the continuous 2048-sample strain segment is divided into smaller, overlapping windows (e.g., 256 or 512 samples in length). This creates a sequence of discrete data points that the model can process.


Each window is then labeled based on its temporal proximity to the known GPS time of the GW150914 event (1126259462.4). Windows that contain the signal are labeled as positive (1), while those that do not are labeled as negative (0). This creates a binary classification task for the SNN.


To further enhance the dataset and improve model robustness, a glitch injection module is employed. This module, accessible through the `glitch_injector.py` module, introduces realistic noise artifacts (glitches) into the training data. These glitches simulate common detector disturbances, forcing the model to learn to distinguish true astrophysical signals from instrumental noise, a crucial capability for real-world deployment.


## Dataset Creation

The final step in the data pipeline is the creation of the complete training and evaluation dataset. The `create_real_ligo_dataset` function orchestrates this process, combining the data loading, windowing, labeling, and augmentation steps into a single, reproducible workflow.


A cornerstone of the dataset creation is the implementation of a stratified train/test split. This is achieved through the `create_stratified_split` function in the `utils/data_split.py` module. This function ensures that both the training and test sets contain a proportional representation of positive and negative examples. This is a critical safeguard against "fake accuracy," a common pitfall where a model achieves high accuracy simply by always predicting the majority class, which would be meaningless for a real detection system.


The function performs a rigorous validation check on the test set. If it detects that all labels in the test set are identical (i.e., all 0s or all 1s), it raises a `ValueError` to prevent the training process from proceeding. This ensures that any evaluation of the model's performance is based on a scientifically valid test set.


The output of this pipeline is a pair of tuples: `(train_signals, train_labels)` and `(test_signals, test_labels)`, which are then fed into the training and evaluation modules. This complete, real-data pipeline is a defining feature of the system, setting it apart from previous neuromorphic approaches that relied on synthetic data and establishing a new standard for scientific rigor in the field.