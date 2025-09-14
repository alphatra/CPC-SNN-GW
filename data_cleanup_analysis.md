# Data Folder Cleanup Analysis

## 📊 Usage Analysis

### ✅ **CURRENTLY USED FILES** (Keep these):
1. **mlgwsc_data_loader.py** - Main MLGWSC-1 dataset loader (used in inference/evaluate)
2. **gw_synthetic_generator.py** - Synthetic data generation (used in training)
3. **gw_signal_params.py** - Signal configuration (used by generator)
4. **gw_physics_engine.py** - Physics calculations (used by generator)
5. **gw_preprocessor.py** - Data preprocessing (used in multiple places)
6. **gw_dataset_builder.py** - Dataset building (used in CLI)
7. **gw_downloader.py** - GWOSC data download (used in enhanced training)
8. **preprocessing/** - Modular preprocessing components
9. **__init__.py** - Package initialization

### ❌ **UNUSED/DEPRECATED FILES** (Can be deleted):
1. **cache_manager.py** - Old caching system (replaced by cache/ module)
2. **cache_metadata.py** - Old caching metadata (replaced by cache/ module)
3. **cache_storage.py** - Old caching storage (replaced by cache/ module)
4. **glitch_injector.py** - Not used anywhere
5. **mlgwsc_dataset_loader.py** - Duplicate of mlgwsc_data_loader.py
6. **readligo_data_sources.py** - Not used (using readligo library directly)
7. **readligo_downloader.py** - Not used (using GWOSCDownloader)
8. **label_enums.py** - Not used anywhere

### ❓ **QUESTIONABLE FILES** (Need verification):
1. **pycbc_integration.py** - Referenced in CLI but module doesn't exist
2. **real_ligo_integration.py** - Referenced in CLI but module doesn't exist
3. **cache/** folder - Might be used but seems redundant
4. **builders/** folder - Not directly used, might be legacy

## 📁 Folder Structure After Cleanup:
```
data/
├── __init__.py                    # Package init
├── gw_dataset_builder.py          # Dataset building
├── gw_downloader.py               # GWOSC download
├── gw_physics_engine.py           # Physics calculations
├── gw_preprocessor.py             # Preprocessing
├── gw_signal_params.py            # Signal configs
├── gw_synthetic_generator.py      # Synthetic data
├── mlgwsc_data_loader.py          # MLGWSC-1 loader
├── preprocessing/                 # Modular preprocessing
│   ├── __init__.py
│   ├── core.py
│   ├── sampler.py
│   └── utils.py
└── gwosc_cache/                   # Cached data
    └── GW150914_H1_32s.npy
```

## 🗑️ Files to Delete:
- cache_manager.py
- cache_metadata.py
- cache_storage.py
- glitch_injector.py
- mlgwsc_dataset_loader.py
- readligo_data_sources.py
- readligo_downloader.py
- label_enums.py
- pycbc_integration.py (if exists)
- real_ligo_integration.py (if exists)
- cache/ folder (if not used)
- builders/ folder (if not used)
