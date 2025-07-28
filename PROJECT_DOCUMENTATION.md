# Radiomics Processing Pipeline

Comprehensive Project Documentation

---

## Table of Contents
1. Project Overview
2. Directory and File Structure
3. Module and File Descriptions
4. Project Policies
   - Coding Standards
   - Error Handling
   - Logging
   - Configuration
   - Extensibility
   - Testing
   - Contribution Guidelines
5. Key Classes and Methods
   - Pipeline Orchestration
   - CLI Argument Parsing
   - Data Loading
   - Preprocessing
   - Feature Extraction (ViSERA)
6. ROI Selection Modes: per_Img vs per_region
7. Value Type Argument (--value-type)
8. RTSTRUCT ROI Naming and Lesion Handling
9. Workflow and Example Usage
10. Extensibility and Best Practices
11. Testing
12. Dependencies
13. License
14. Support

---

## Project Overview

This project is a modular, extensible radiomics processing pipeline for extracting quantitative features from medical images (DICOM, NIfTI, RTSTRUCT). It supports batch and parallel processing, advanced ROI selection, and leverages the ViSERA backend for feature extraction. The codebase is organized for maintainability, testability, and ease of extension.

---

## Directory and File Structure

```
PySERATest/
├── radiomics_standalone.py        # Main entry point
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── PROJECT_DOCUMENTATION.docx     # (This file)
└── src/
    ├── __init__.py
    ├── cli/
    │   ├── __init__.py
    │   └── argument_parser.py
    ├── config/
    │   ├── __init__.py
    │   └── settings.py
    ├── data/
    │   ├── __init__.py
    │   └── dicom_loader.py
    ├── engine/
    │   └── visera/
    │       ├── __init__.py
    │       ├── getGLCMFeatures.py
    │       ├── getGLDZMFeatures.py
    │       ├── getGLRLMFeatures.py
    │       ├── getGLSZMFeatures.py
    │       ├── getHist.py
    │       ├── getIntVolHist.py
    │       ├── getMIFeatures.py
    │       ├── getMorph_1.py
    │       ├── getMorph.py
    │       ├── getNGLDMFeatures.py
    │       ├── getNGTDMFeatures.py
    │       ├── getStats.py
    │       ├── getSUVpeak.py
    │       ├── interpolation.py
    │       ├── Main.py
    │       ├── prepareVolume.py
    │       ├── Quantization.py
    │       ├── RF_main.py
    │       ├── Sera_Filter.py
    │       ├── Sera_Fusion.py
    │       ├── Sera_ReadWrite.py
    │       ├── Sera_Registration.py
    │       ├── SERASUVscalingObj.py
    │       ├── SERAutilities.py
    │       └── view_functions.py
    ├── features/
    │   ├── __init__.py
    │   └── feature_names.py
    ├── preprocessing/
    │   ├── __init__.py
    │   ├── intensity_preprocessing.py
    │   └── roi_preprocessing.py
    ├── processing/
    │   ├── __init__.py
    │   └── radiomics_processor.py
    ├── utils/
    │   ├── __init__.py
    │   ├── file_utils.py
    │   ├── image_io.py
    │   ├── mock_modules.py
    │   └── utils.py
    └── workflow/
        ├── __init__.py
        ├── work_graph/
        │   └── __init__.py
        └── work_graph_ui/
            ├── __init__.py
            └── module.py
```

---

## Module and File Descriptions

(See previous section for the full list and brief descriptions of each file.)

---

## Project Policies

### Coding Standards
- **PEP8**: All Python code should follow PEP8 style guidelines.
- **Naming**: Use descriptive, meaningful names for variables, functions, and classes. Use snake_case for functions/variables, PascalCase for classes.
- **Single Responsibility**: Each function/class should have a single, clear responsibility.
- **DRY Principle**: Avoid code duplication; use utility functions and modularization.
- **Type Annotations**: Use type hints for all function signatures and return types.
- **Docstrings**: Every public function and class should have a docstring describing its purpose, parameters, and return value.

### Error Handling
- **Consistent Exception Handling**: Use try/except blocks for all I/O and external library calls. Raise informative errors with context.
- **Graceful Degradation**: If a non-critical error occurs, log it and continue processing other items if possible.
- **User Feedback**: All errors should be logged and, if relevant, reported to the user via CLI or output files.

### Logging
- **Centralized Logging**: Use the Python `logging` module. Configure log level via CLI or config.
- **Log Granularity**: Log at INFO for major steps, DEBUG for detailed tracing, WARNING for recoverable issues, ERROR for critical failures.
- **Log Output**: Logs should be written to both console and (optionally) a file.

### Configuration Policy
- **Centralized Settings**: All default values and configuration options are defined in `src/config/settings.py`.
- **CLI Override**: Command-line arguments always override config file defaults.
- **Environment Variables**: (Optional) Support for environment variable overrides can be added for deployment.

### Extensibility
- **Modular Design**: New feature extraction modules can be added in `engine/visera/` and registered in `Main.py`.
- **Pluggable Preprocessing**: Add new preprocessing steps as separate modules in `src/preprocessing/`.
- **CLI Expansion**: Add new arguments in `src/cli/argument_parser.py` and propagate to settings and processor.

### Testing Policy
- **Unit Tests**: Each module should have unit tests for its core logic.
- **Mocking**: Use `mock_modules.py` for testing without real data.
- **Synthetic Data**: Use synthetic images/masks for reproducible tests.
- **Continuous Integration**: (Recommended) Integrate with CI tools for automated testing.

### Contribution Guidelines
- **Pull Requests**: All changes should be submitted via pull request with a clear description.
- **Code Review**: At least one other developer should review and approve before merging.
- **Documentation**: All new features and modules must be documented in this `.docx` file and with inline docstrings.
- **Testing**: All new code must include tests or demonstrate testability.

---

## Key Classes and Methods

### Pipeline Orchestration
#### `RadiomicsProcessor` (`src/processing/radiomics_processor.py`)
- **Purpose**: Orchestrates the entire radiomics pipeline.
- **Key Methods**:
    - `__init__(self, config: dict)`: Initializes processor with configuration.
    - `run(self) -> None`: Main entry point. Loads data, preprocesses, selects ROIs, extracts features, and writes output.
    - `load_images_and_masks(self) -> Tuple[List[np.ndarray], List[np.ndarray]]`: Loads images and masks from input paths.
    - `preprocess(self, images: List[np.ndarray], masks: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]`: Applies intensity and ROI preprocessing.
    - `select_rois(self, masks: List[np.ndarray]) -> List[dict]`: Selects ROIs according to mode and number. Returns list of ROI dicts with metadata.
    - `extract_features(self, images: List[np.ndarray], rois: List[dict]) -> pd.DataFrame`: Calls ViSERA backend for each ROI and aggregates results.
    - `aggregate_results(self, features: pd.DataFrame) -> None`: Writes results to Excel/CSV.
- **Notes**: Handles all error logging and reporting. Designed for batch and parallel processing.

### CLI Argument Parsing
#### `ArgumentParser` (`src/cli/argument_parser.py`)
- **Purpose**: Handles all CLI argument parsing and validation.
- **Key Methods**:
    - `parse_args(self) -> argparse.Namespace`: Parses and returns CLI arguments.
    - `add_arguments(self, parser: argparse.ArgumentParser) -> None`: Adds all supported arguments to the parser.
- **Notes**: Provides help/usage messages. Validates required arguments and types.

##### Supported Arguments (including new):
| Argument         | Type   | Default              | Description |
|------------------|--------|----------------------|-------------|
| --value-type     | str    | "APPROXIMATE_VALUE"  | Value type for feature extraction (e.g., APPROXIMATE_VALUE, EXACT_VALUE, etc). Controls how feature values are computed or reported. |

### Data Loading
#### `DicomLoader` (`src/data/dicom_loader.py`)
- **Purpose**: Loads DICOM/RTSTRUCT and converts to arrays.
- **Key Methods**:
    - `load_dicom_series(self, path: str) -> np.ndarray`: Loads a DICOM series as a 3D numpy array.
    - `load_rtstruct(self, path: str) -> dict`: Loads RTSTRUCT and returns a dict of structures.
    - `convert_rtstruct_to_mask(self, rtstruct: dict, image_shape: tuple) -> np.ndarray`: Converts RTSTRUCT to a mask array.
- **Notes**: Handles DICOM metadata extraction and error handling for missing/invalid files.

### Preprocessing Methods
#### `optimize_roi_preprocessing` (`src/preprocessing/roi_preprocessing.py`)
- **Purpose**: Cleans and filters ROI masks, finds connected components.
- **Parameters**:
    - `mask: np.ndarray`: Input mask array (multi-label or binary).
    - `min_roi_volume: int`: Minimum volume threshold for ROIs.
- **Returns**: Processed mask array with small/noisy ROIs removed.
- **Notes**: Uses `scipy.ndimage.label` for connected component analysis. Logs number of ROIs before/after filtering.

#### `find_connected_components` (`src/preprocessing/roi_preprocessing.py`)
- **Purpose**: Finds all connected components (ROIs) in a mask.
- **Parameters**:
    - `mask: np.ndarray`: Input mask array.
- **Returns**: List of ROI dicts with label, volume, centroid, and mask.

#### `filter_by_volume` (`src/preprocessing/roi_preprocessing.py`)
- **Purpose**: Removes ROIs below a volume threshold.
- **Parameters**:
    - `rois: List[dict]`: List of ROI dicts.
    - `min_volume: int`: Minimum volume threshold.
- **Returns**: Filtered list of ROIs.

### Feature Extraction (ViSERA)
#### `get_features` (in each `src/engine/visera/*.py`)
- **Purpose**: Extracts a specific set of radiomic features from an image/ROI.
- **Parameters**:
    - `image: np.ndarray`: Input image array.
    - `mask: np.ndarray`: ROI mask array.
    - `params: dict`: Extraction parameters (e.g., bin size, quantization, etc).
- **Returns**: Dict or array of feature values.
- **Notes**: Each module implements a different feature family (GLCM, GLRLM, etc). All are coordinated by `Main.py`.

#### `Main.py` (ViSERA)
- **Purpose**: Coordinates feature extraction across all modules.
- **Key Methods**:
    - `extract_all_features(image, mask, params)`: Calls each feature module and aggregates results.

---

## ROI Selection Modes: per_Img vs per_region

### Overview
The pipeline supports two intelligent ROI (Region of Interest) selection modes, each designed for different analysis scenarios:

- **per_Img**: Selects the largest ROIs across the entire image, ignoring label grouping.
- **per_region**: Groups ROIs by color (label value) and selects up to `roi_num` ROIs from each group.

### per_Img Mode
- **Description**: This mode treats all detected ROIs in the mask as a single pool, regardless of their label (color). It sorts all ROIs by volume (size) and selects the top `roi_num` largest ROIs for feature extraction.
- **Algorithm**:
    1. Detect all connected ROIs in the mask (across all labels).
    2. Sort ROIs by volume in descending order.
    3. Select the top `roi_num` ROIs.
- **Use Case Scenarios**:
    - You want to analyze the most prominent lesions or structures, regardless of their anatomical or label grouping.
    - Useful for single-tumor studies, or when only the largest regions are of interest.
- **Example**:
    - If your mask contains 10 ROIs (labels 1, 2, 3, ...), and you set `roi_num=3`, the 3 largest ROIs (by volume) are selected, regardless of their label.

### per_region Mode
- **Description**: This mode first groups ROIs by their label value (color). For each group, it sorts the ROIs by volume and selects up to `roi_num` ROIs from each group. This ensures representation from all anatomical or semantic regions present in the mask.
- **Algorithm**:
    1. For each unique label value in the mask (excluding background):
        - Detect all connected ROIs for that label.
        - Sort ROIs by volume in descending order.
        - Select up to `roi_num` ROIs from this group.
    2. Combine all selected ROIs from all groups for feature extraction.
- **Use Case Scenarios**:
    - You want to ensure that all anatomical regions or semantic classes (e.g., different tumor types, organs) are represented in the analysis.
    - Useful for multi-focal, multi-class, or multi-organ studies.
    - Ensures that small but important regions are not missed due to being outnumbered by larger ROIs from other labels.
- **Example**:
    - If your mask contains labels 1, 2, and 3, and you set `roi_num=2`, the pipeline will select up to 2 largest ROIs from each label group. If label 1 has 3 ROIs, label 2 has 1 ROI, and label 3 has 2 ROIs, the selected set will be: 2 from label 1, 1 from label 2, and 2 from label 3 (total 5 ROIs).

### Comparison Table
| Mode         | Grouping Basis | Max ROIs Selected | Scenario Example |
|--------------|----------------|-------------------|------------------|
| per_Img      | None (all ROIs together) | `roi_num` total | "Find the 3 largest lesions in the scan" |
| per_region   | By label value (color)   | Up to `roi_num` per label | "Find up to 2 largest lesions in each anatomical region" |

### Notes
- In **per_Img** mode, the total number of selected ROIs is always at most `roi_num`.
- In **per_region** mode, the total number of selected ROIs can exceed `roi_num` if there are multiple label groups.
- Both modes use connected component analysis to ensure that each ROI is a spatially contiguous region.

---

## 7. Value Type Argument (--value-type)

The `--value-type` argument controls how feature values are computed or reported throughout the pipeline. Example options include:

- `APPROXIMATE_VALUE` (default): Use approximate/fast computation (recommended for most workflows)
- `EXACT_VALUE`: Use exact/slow computation (if supported)

This argument is passed through the pipeline and can affect feature extraction and reporting. Refer to the CLI help or code documentation for all available options.

---

## 8. RTSTRUCT ROI Naming and Lesion Handling

When using RTSTRUCT masks, each ROI is loaded and processed individually. If a single ROI contains multiple disconnected lesions, each lesion is named as `ROIName_lesion_{i}` (e.g., `GTV_Primary_lesion_1`, `GTV_Primary_lesion_2`). This ensures all lesions are uniquely identified and processed, even if they belong to the same ROI. This naming convention is reflected in logs, output files, and feature tables.

---

## Workflow and Example Usage

(See previous section for workflow and example command.)

---

## Extensibility and Best Practices
- **Add new feature types**: Implement a new module in `engine/visera/` and register it in `Main.py`.
- **Add new CLI options**: Update `src/cli/argument_parser.py` and propagate to `settings.py`.
- **Testing**: Use mock modules and synthetic data for unit tests.
- **Configuration**: Centralize all defaults in `settings.py` for maintainability.
- **Error Handling**: Use consistent error handling and logging throughout.
- **Documentation**: Maintain docstrings and update this `.docx` file as the project evolves.

---

## Testing
- Modular structure supports unit and integration testing.
- Example: test ROI preprocessing with synthetic data arrays.
- Use `mock_modules.py` for isolated tests.

---

## Dependencies
- See `requirements.txt` for the full list.
- Key packages: SimpleITK, numpy, pandas, scikit-image, pydicom, nibabel, pynrrd, PySide6, matplotlib, scikit-learn, openpyxl, psutil, PyWavelets, ReliefF, mahotas, itk, scikit-optimize, scikit-learn-extra, kmodes, pyscreenshot, pyhull, dcmrtstruct2nii, rt-utils, PyGetWindow, scipy, dataclasses, setuptools

---

## License
- This project maintains the same license as the original codebase.

---

## Support
- Check logs for error messages
- Verify input file formats
- Ensure all dependencies are installed
- For further details, see docstrings and comments in each module 