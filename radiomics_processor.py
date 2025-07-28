"""
Main radiomics processing module that orchestrates the entire pipeline.
"""

import os
import sys
import logging
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
from datetime import datetime
import numpy as np
from pathlib import Path
import gc

from ..config.settings import (
    DEFAULT_RADIOICS_PARAMS, DEFAULT_MIN_ROI_VOLUME, EXPECTED_FEATURE_COUNTS,
    OUTPUT_FILENAME_TEMPLATE, get_visera_pythoncode_path, get_default_output_path
)
from ..utils.mock_modules import setup_mock_modules
from ..utils.file_utils import (
    detect_file_format, find_files_by_format, match_image_mask_pairs, ensure_directory_exists
)
from ..utils.save_params import write_to_excel
from ..data.dicom_loader import convert_dicom_to_arrays
from ..preprocessing.roi_preprocessing import optimize_roi_preprocessing, get_roi_statistics
from ..preprocessing.intensity_preprocessing import apply_intensity_preprocessing
from ..features.feature_names import get_feature_names

try:
    from ..utils.utils import (
        get_memory_usage, check_memory_available, estimate_array_memory,
        safe_array_operation, optimize_array_dtype, memory_efficient_unique,
        log_memory_usage, validate_data_integrity, memory_safe_operation,
        optimize_memory_usage_safely
    )
except ImportError:
    # Fallback if utils not available
    def get_memory_usage(): return 0.0
    def check_memory_available(required_mb): return True
    def estimate_array_memory(shape, dtype): return 0.0
    def safe_array_operation(func, *args, **kwargs): return func(*args, **kwargs)
    def optimize_array_dtype(array, target_dtype=None): return array
    def memory_efficient_unique(array): return np.unique(array)
    def log_memory_usage(operation_name): pass
    def validate_data_integrity(original, optimized, tolerance=1e-10): return True
    def memory_safe_operation(func, *args, **kwargs): return func(*args, **kwargs)
    def optimize_memory_usage_safely(array, target_memory_mb=1000): return array


class MemoryAwareRadiomicsProcessor:
    """
    Memory-aware radiomics processor for handling large medical images.
    """

    def __init__(self, memory_limit_mb: int = 1000, enable_optimization: bool = True):
        """
        Initialize memory-aware processor.

        Args:
            memory_limit_mb: Memory limit in MB
            enable_optimization: Whether to enable memory optimization
        """
        self.memory_limit_mb = memory_limit_mb
        self.enable_optimization = enable_optimization
        self.logger = logging.getLogger(__name__)

    def process_large_array(self, array: np.ndarray, operation: str, **kwargs) -> Any:
        """
        Process large arrays with memory awareness while preserving exact results.

        Args:
            array: Input array
            operation: Operation to perform
            **kwargs: Additional arguments

        Returns:
            Processed result
        """
        if not self.enable_optimization:
            return self._standard_operation(array, operation, **kwargs)

        # Estimate memory requirements
        estimated_memory = estimate_array_memory(array.shape, array.dtype)

        if estimated_memory > self.memory_limit_mb:
            return self._memory_efficient_operation(array, operation, **kwargs)
        else:
            return self._standard_operation(array, operation, **kwargs)

    def _standard_operation(self, array: np.ndarray, operation: str, **kwargs) -> Any:
        """Standard operation without memory optimization."""
        if operation == "unique":
            return np.unique(array)
        elif operation == "convolve":
            return self._convolve_standard(array, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _memory_efficient_operation(self, array: np.ndarray, operation: str, **kwargs) -> Any:
        """Memory-efficient operation for large arrays that preserves exact results."""
        if operation == "unique":
            # Always preserve exact results for unique operations
            return memory_efficient_unique(array, preserve_exact_results=True)
        elif operation == "convolve":
            return self._convolve_memory_efficient(array, **kwargs)
        else:
            raise ValueError(f"Unknown operation: {operation}")

    def _convolve_standard(self, array: np.ndarray, kernel: np.ndarray, **kwargs) -> np.ndarray:
        """Standard convolution operation."""
        from scipy.signal import convolve
        return convolve(array, kernel, **kwargs)

    def _convolve_memory_efficient(self, array: np.ndarray, kernel: np.ndarray, **kwargs) -> np.ndarray:
        """Memory-efficient convolution for large arrays without approximations."""
        # For very large arrays, try chunked convolution instead of approximation
        if array.size > 100000000:  # 100M elements
            self.logger.info("Array very large, attempting chunked convolution")
            return self._chunked_convolution(array, kernel, **kwargs)

        # Try standard convolution with memory monitoring
        try:
            return self._convolve_standard(array, kernel, **kwargs)
        except MemoryError:
            self.logger.warning("Memory error in convolution, trying chunked approach")
            return self._chunked_convolution(array, kernel, **kwargs)

    def _chunked_convolution(self, array: np.ndarray, kernel: np.ndarray, **kwargs) -> np.ndarray:
        """
        Chunked convolution that preserves results by processing in overlapping chunks.
        This maintains mathematical correctness while reducing memory usage.
        """
        try:
            from scipy.signal import convolve
            
            # Calculate overlap needed based on kernel size
            kernel_shape = np.array(kernel.shape)
            overlap = kernel_shape // 2
            
            # For 3D arrays, process slice by slice with overlap
            if array.ndim == 3:
                result = np.zeros_like(array)
                
                for z in range(array.shape[2]):
                    # Define slice bounds with overlap
                    z_start = max(0, z - overlap[2] if len(overlap) > 2 else 0)
                    z_end = min(array.shape[2], z + overlap[2] + 1 if len(overlap) > 2 else z + 1)
                    
                    # Extract slice with overlap
                    slice_data = array[:, :, z_start:z_end]
                    
                    # Convolve the slice
                    if slice_data.ndim == 3 and kernel.ndim == 3:
                        slice_result = convolve(slice_data, kernel, mode='same')
                        # Extract the center part (original slice)
                        center_idx = z - z_start
                        if center_idx < slice_result.shape[2]:
                            result[:, :, z] = slice_result[:, :, center_idx]
                    else:
                        # 2D convolution on each slice
                        kernel_2d = kernel if kernel.ndim == 2 else kernel[:, :, kernel.shape[2]//2]
                        result[:, :, z] = convolve(slice_data[:, :, -1], kernel_2d, mode='same')
                
                return result
            else:
                # For 2D arrays, use standard convolution
                return convolve(array, kernel, **kwargs)
                
        except Exception as e:
            self.logger.error(f"Error in chunked convolution: {e}")
            # As a last resort, return the original array (no convolution)
            # This preserves the data while avoiding approximations
            self.logger.warning("Convolution failed, returning original array")
            return array

    def optimize_array_for_processing(self, array: np.ndarray) -> np.ndarray:
        """
        Optimize array for processing based on memory constraints without data loss.

        Args:
            array: Input array

        Returns:
            Optimized array (same data, potentially different dtype)
        """
        if not self.enable_optimization:
            return array

        # Check memory usage
        current_memory = get_memory_usage()
        array_memory = estimate_array_memory(array.shape, array.dtype)

        if current_memory + array_memory > self.memory_limit_mb:
            # Force garbage collection first
            gc.collect()
            
            # Update memory usage after cleanup
            current_memory = get_memory_usage()

            # Use safe memory optimization that preserves data integrity
            optimized_array = optimize_memory_usage_safely(array, self.memory_limit_mb)
            return optimized_array

        return array

    def log_memory_status(self, operation: str):
        """Log current memory status."""
        memory_mb = get_memory_usage()
        self.logger.info(f"[MEMORY] {operation}: {memory_mb:.1f} MB / {self.memory_limit_mb} MB limit")


# Global memory-aware processor instance
_memory_processor = None

def get_memory_processor(memory_limit_mb: int = 1000, enable_optimization: bool = True) -> MemoryAwareRadiomicsProcessor:
    """Get or create memory-aware processor instance."""
    global _memory_processor
    if _memory_processor is None:
        _memory_processor = MemoryAwareRadiomicsProcessor(memory_limit_mb, enable_optimization)
    return _memory_processor


class RadiomicsProcessor:
    """
    Main class for orchestrating radiomics feature extraction.
    """
    
    def __init__(self, output_path: Optional[str] = None):
        """
        Initialize the radiomics processor.
        
        Args:
            output_path: Output directory path (optional)
        """
        self.output_path = output_path or get_default_output_path()
        ensure_directory_exists(self.output_path)
        
        # Set up mock modules for dependencies
        setup_mock_modules()
        
        # Import SERA module
        self._setup_sera_module()
        
        # Initialize parameters
        self.params = DEFAULT_RADIOICS_PARAMS.copy()
        self.params['radiomics_destfolder'] = self.output_path
    
    def _setup_sera_module(self) -> None:
        """Set up the SERA module for feature extraction."""
        sera_pythoncode_dir = get_visera_pythoncode_path()
        if sera_pythoncode_dir not in sys.path:
            sys.path.insert(0, sera_pythoncode_dir)
        
        try:
            import importlib
            rf_main_module = importlib.import_module('RF_main')
            self.SERA_FE_main = rf_main_module.SERA_FE_main
            logging.info("RF_main module imported successfully")
        except ImportError as e:
            logging.error(f"Failed to import RF_main module: {e}")
            raise
    
    def process_single_image_pair(self, args_tuple: Tuple) -> Optional[pd.DataFrame]:
        """
        Process a single image-mask pair.
        
        Args:
            args_tuple: Tuple containing (image_path, mask_path, params, output_folder, 
                        apply_preprocessing, min_roi_volume, folder_name, feats2out, roi_num, roi_selection_mode)
        
        Returns:
            DataFrame with extracted features or None if failed
        """
        (image_path, mask_path, params, output_folder, apply_preprocessing, 
         min_roi_volume, folder_name, feats2out, roi_num, roi_selection_mode, value_type) = args_tuple
        
        # Add unique identifier for this processing
        image_id = os.path.basename(image_path)
        logging.info(f"=== PROCESSING IMAGE {image_id} ===")
        logging.info(f"Image path: {image_path}")
        logging.info(f"Mask path: {mask_path}")
        
        try:
            start_time = time.time()
            
            # Load and convert image and mask data
            image_data = self._load_image_data(image_id, image_path, mask_path)
            if image_data is None:
                return None
            
            image_array, image_metadata, mask_array, mask_metadata = image_data
            
            # Get ROI statistics
            if isinstance(mask_array, dict):
                total_rois = 0
                for roi_name, roi_mask in mask_array.items():
                    roi_stats = get_roi_statistics(roi_mask)
                    logging.info(f"[{image_id}] RTSTRUCT ROI '{roi_name}': {roi_stats['total_voxels']} voxels")
                    total_rois += 1 if roi_stats['total_voxels'] > 0 else 0
                logging.info(f"[{image_id}] Found {total_rois} Unique ROI(s) in RTSTRUCT mask")
                if total_rois == 0:
                    logging.warning(f"[{image_id}] No ROIs found in RTSTRUCT mask")
                    return None
            else:
                roi_stats = get_roi_statistics(mask_array)
                logging.info(f"[{image_id}] Found {roi_stats['total_rois']} Unique ROI(s) in mask")
                if roi_stats['total_rois'] == 0:
                    logging.warning(f"[{image_id}] No ROIs found in mask")
                    return None
            
            # Prepare parameters for SERA
            params_copy = self._prepare_sera_parameters(params, image_array, image_metadata, 
                                                       mask_array, mask_metadata, feats2out, roi_num, roi_selection_mode)
            params_copy['value_type'] = value_type
            
            # Process with SERA
            result = self._run_sera_processing(image_id, params_copy, apply_preprocessing, min_roi_volume)
            
            end_time = time.time()
            processing_time = end_time - start_time
            logging.info(f"[{image_id}] Processing completed in {processing_time:.2f} seconds")
            
            if result is not None:
                df = result
                df.insert(0, 'PatientID', os.path.basename(image_path))
                
                # Final quality check
                self._perform_final_quality_check(image_id, df)
                
                return df
            else:
                logging.error(f"[{image_id}] FAILED: No result returned from SERA processing")
                return None
                
        except Exception as e:
            logging.error(f"[{image_id}] CRITICAL ERROR: {e}")
            import traceback
            logging.error(f"[{image_id}] Traceback: {traceback.format_exc()}")
            return None
    
    def _load_image_data(self, image_id: str, image_path: str, mask_path: str) -> Optional[Tuple]:
        """Load and convert image and mask data."""
        logging.info(f"[{image_id}] Loading and converting image and mask data...")
        image_array, image_metadata, mask_array, mask_metadata = convert_dicom_to_arrays(image_path, mask_path)

        if image_array is None or mask_array is None:
            logging.error(f"[{image_id}] CRITICAL: Failed to load image or mask data")
            return None

        # For RTSTRUCT, mask_array is a dict; for others, it's an array
        if isinstance(mask_array, dict):
            logging.info(f"[{image_id}] Successfully loaded - Image shape: {image_array.shape}, RTSTRUCT mask with {len(mask_array)} ROIs: {list(mask_array.keys())}")
        else:
            logging.info(f"[{image_id}] Successfully loaded - Image shape: {image_array.shape}, Mask shape: {mask_array.shape}")
        return image_array, image_metadata, mask_array, mask_metadata
    
    def _prepare_sera_parameters(self, params: Dict[str, Any], image_array: np.ndarray, 
                               image_metadata: Dict, mask_array: np.ndarray, 
                               mask_metadata: Dict, feats2out: int, roi_num: int, 
                               roi_selection_mode: str) -> Dict[str, Any]:
        """Prepare parameters for SERA processing."""
        params_copy = params.copy()
        params_copy['da_original'] = [image_array, image_metadata, image_metadata['format'].title(), None]
        params_copy['da_label'] = [mask_array, mask_metadata, mask_metadata['format'].title(), None]
        params_copy['radiomics_Feats2out'] = feats2out
        params_copy['radiomics_ROI_num'] = roi_num
        params_copy['radiomics_ROI_selection_mode'] = roi_selection_mode
        params_copy['value_type'] = params.get('value_type', 'EXACT_VALUE')
        return params_copy
    
    def _perform_final_quality_check(self, image_id: str, df: pd.DataFrame) -> None:
        """Perform final quality check on results."""
        nan_count = df.isnull().sum().sum()
        total_values = len(df) * (len(df.columns) - 3)  # Exclude metadata columns
        logging.info(f"[{image_id}] FINAL RESULT: {100*nan_count/total_values:.1f}% missing values")
    
    def _run_sera_processing(self, image_id: str, params: Dict[str, Any], 
                           apply_preprocessing: bool, min_roi_volume: int) -> Optional[pd.DataFrame]:
        """
        Run SERA feature extraction with preprocessing.
        
        Args:
            image_id: Image identifier for logging
            params: Processing parameters
            apply_preprocessing: Whether to apply preprocessing
            min_roi_volume: Minimum ROI volume threshold
            
        Returns:
            DataFrame with extracted features or None if failed
        """
        def optimized_sera_with_preprocessing(da_original, da_label, *args, **kwargs):
            logging.info(f"[{image_id}] Running optimized SERA function with preprocessing")
            
            # Extract parameters from args
            sera_params = self._extract_sera_parameters(args)
            
            data_original = da_original[0].copy()
            data_label = da_label[0].copy()
            VoxelSizeInfo = da_original[1]['spacing']
            
            # Add VoxelSizeInfo to sera_params so it can be accessed in feature extraction
            sera_params['VoxelSizeInfo'] = VoxelSizeInfo
            
            # Log input data shapes
            if isinstance(data_label, dict):
                roi_shapes = {roi: mask.shape for roi, mask in data_label.items()}
                logging.info(f"[{image_id}] Input data shapes - Image: {data_original.shape}, RTSTRUCT mask with {len(data_label)} ROIs: {roi_shapes}")
            else:
                logging.info(f"[{image_id}] Input data shapes - Image: {data_original.shape}, Label: {data_label.shape}")
            
            logging.info(f"[{image_id}] Voxel spacing: {VoxelSizeInfo}")
            
            # Apply preprocessing if requested
            if apply_preprocessing:
                data_original, data_label = self._apply_preprocessing_to_data(
                    image_id, data_original, data_label, min_roi_volume
                )
            
            # Extract features for each ROI
            return self._extract_features_for_rois(
                image_id, data_original, data_label, VoxelSizeInfo, min_roi_volume, sera_params
            )
        
        # Call SERA function
        result = optimized_sera_with_preprocessing(
            params['da_original'],
            params['da_label'],
            params['radiomics_BinSize'],
            params['radiomics_isotVoxSize'],
            params['radiomics_isotVoxSize2D'],
            params['radiomics_DataType'],
            params['radiomics_DiscType'],
            params['radiomics_qntz'],
            params['radiomics_VoxInterp'],
            params['radiomics_ROIInterp'],
            params['radiomics_isScale'],
            params['radiomics_isGLround'],
            params['radiomics_isReSegRng'],
            params['radiomics_isOutliers'],
            params['radiomics_isQuntzStat'],
            params['radiomics_isIsot2D'],
            params['radiomics_ReSegIntrvl01'],
            params['radiomics_ReSegIntrvl02'],
            params['radiomics_ROI_PV'],
            params['radiomics_IVH_Type'],
            params['radiomics_IVH_DiscCont'],
            params['radiomics_IVH_binSize'],
            params['radiomics_ROI_num'],
            params['radiomics_ROI_selection_mode'],
            params['radiomics_isROIsCombined'],
            params['radiomics_Feats2out'],
            params['radiomics_destfolder'],
            params.get('value_type', 'EXACT_VALUE')
        )
        
        return result
    
    def _extract_sera_parameters(self, args: Tuple) -> Dict[str, Any]:
        """Extract SERA parameters from args tuple."""
        return {
            'BinSize': args[0],
            'isotVoxSize': args[1],
            'isotVoxSize2D': args[2],
            'DataType': args[3],
            'DiscType': args[4],
            'qntz': args[5],
            'VoxInterp': args[6],
            'ROIInterp': args[7],
            'isScale': args[8],
            'isGLround': args[9],
            'isReSegRng': args[10],
            'isOutliers': args[11],
            'isQuntzStat': args[12],
            'isIsot2D': args[13],
            'ReSegIntrvl01': args[14],
            'ReSegIntrvl02': args[15],
            'ROI_PV': args[16],
            'IVH_Type': int(args[17]),
            'IVH_DiscCont': int(args[18]),
            'IVH_binSize': float(args[19]),
            'ROI_num': args[20],
            'ROI_selection_mode': args[21],
            'isROIsCombined': args[22],
            'Feats2out': args[23],
            'destfolder': args[24],
            'value_type': args[25] if len(args) > 25 else 'EXACT_VALUE'
        }
    
    def _apply_preprocessing_to_data(self, image_id: str, data_original: np.ndarray, data_label: Any, min_roi_volume: int) -> Tuple[np.ndarray, Any]:
        """
        Apply preprocessing to image and mask data with memory optimization.

        Args:
            image_id: Image identifier
            data_original: Original image data
            data_label: Label/mask data
            min_roi_volume: Minimum ROI volume threshold

        Returns:
            Tuple of (processed_image, processed_label)
        """
        logging.info(f"[{image_id}] Applying preprocessing to data...")

        # Get memory processor
        memory_processor = get_memory_processor()
        memory_processor.log_memory_status("Preprocessing start")

        # Optimize arrays for processing with data integrity validation
        original_data_copy = data_original.copy()  # Keep copy for validation
        data_original = memory_processor.optimize_array_for_processing(data_original)
        
        # Validate that optimization preserved data integrity
        if not validate_data_integrity(original_data_copy, data_original):
            logging.warning(f"[{image_id}] Data integrity check failed for image optimization, using original")
            data_original = original_data_copy

        # Apply intensity preprocessing with memory safety
        processed_original = memory_safe_operation(
            apply_intensity_preprocessing, 
            data_original, 
            data_label,
            fallback_func=lambda x, y: x  # Return original if preprocessing fails
        )

        # Process label data based on type
        if isinstance(data_label, dict):
            # RTSTRUCT case - process each ROI separately
            processed_label = {}
            for roi_name, roi_mask in data_label.items():
                logging.info(f"[{image_id}] Processing ROI: {roi_name}")

                # Optimize ROI mask for processing with validation
                original_roi_copy = roi_mask.copy()
                roi_mask = memory_processor.optimize_array_for_processing(roi_mask)
                
                # Validate optimization preserved data integrity
                if not validate_data_integrity(original_roi_copy, roi_mask):
                    logging.warning(f"[{image_id}] Data integrity check failed for ROI {roi_name}, using original")
                    roi_mask = original_roi_copy

                # Use memory-safe ROI preprocessing
                processed_roi = memory_safe_operation(
                    optimize_roi_preprocessing,
                    roi_mask,
                    min_roi_volume,
                    fallback_func=lambda x, y: x  # Return original if preprocessing fails
                )
                processed_label[roi_name] = processed_roi
        else:
            # Standard mask case with validation
            original_label_copy = data_label.copy()
            data_label = memory_processor.optimize_array_for_processing(data_label)
            
            # Validate optimization preserved data integrity
            if not validate_data_integrity(original_label_copy, data_label):
                logging.warning(f"[{image_id}] Data integrity check failed for mask optimization, using original")
                data_label = original_label_copy

            # Use memory-safe mask preprocessing
            processed_label = memory_safe_operation(
                optimize_roi_preprocessing,
                data_label,
                min_roi_volume,
                fallback_func=lambda x, y: x  # Return original if preprocessing fails
            )

        memory_processor.log_memory_status("Preprocessing end")
        logging.info(f"[{image_id}] Preprocessing completed")

        return processed_original, processed_label
    
    def _extract_features_for_rois(self, image_id: str, data_original: np.ndarray, 
                                 data_label: Any, VoxelSizeInfo: list,
                                 min_roi_volume: int, sera_params: dict) -> Optional[pd.DataFrame]:
        """
        Extract features for all ROIs in the image.
        For RTSTRUCT, data_label is a dict of {roi_name: mask_array}.
        For other masks, data_label is a single array.
        """
        if isinstance(data_label, dict):
            # RTSTRUCT: each ROI is a separate mask
            roi_num = sera_params.get('ROI_num', 10)
            roi_selection_mode = sera_params.get('ROI_selection_mode', 'per_Img')
            
            # Collect all ROIs with their volumes
            all_rois = []
            for roi_name, roi_mask in data_label.items():
                volume = np.sum(roi_mask > 0)
                if volume < min_roi_volume:
                    logging.info(f"[{image_id}] RTSTRUCT ROI '{roi_name}' skipped (volume {volume} < min_roi_volume {min_roi_volume})")
                    continue
                all_rois.append((roi_name, roi_mask, volume))
            
            if not all_rois:
                logging.error(f"[{image_id}] CRITICAL: No RTSTRUCT ROIs meet the volume threshold!")
                return None
            
            # Apply ROI selection policy for RTSTRUCT
            selected_rois = self._apply_rtstruct_roi_selection_policy(
                image_id, all_rois, roi_num, roi_selection_mode
            )
            
            # Process selected ROIs
            all_features = []
            roi_names = []
            processed_rois = []
            skipped_rois = []
            
            for roi_name, roi_mask, volume in selected_rois:
                processed_rois.append((roi_name, volume))
                feature_result = self._extract_features_for_single_roi(
                    image_id, data_original, roi_mask, roi_name, sera_params
                )
                if feature_result is not None:
                    all_features.append(feature_result)
                    roi_names.append(roi_name)
                else:
                    skipped_rois.append((roi_name, volume))
            
            if not all_features:
                logging.error(f"[{image_id}] CRITICAL: No features extracted from any RTSTRUCT ROI")
                return None
            
            return self._create_results_dataframe(
                image_id, all_features, roi_names, sera_params, processed_rois, skipped_rois, min_roi_volume
            )
        else:
            # Non-RTSTRUCT: old behavior
            all_rois = self._get_all_rois_from_mask(data_label, image_id)
            if len(all_rois) == 0:
                logging.error(f"[{image_id}] CRITICAL: No ROIs found in mask!")
                return None
            self._log_roi_volume_summary(image_id, all_rois, min_roi_volume)
            all_features, roi_names, processed_rois, skipped_rois = self._process_all_rois(
                image_id, data_original, data_label, all_rois, min_roi_volume, sera_params
            )
            if not all_features:
                logging.error(f"[{image_id}] CRITICAL: No features extracted from any ROI")
                return None
            return self._create_results_dataframe(
                image_id, all_features, roi_names, sera_params, processed_rois, skipped_rois, min_roi_volume
            )

    def _apply_rtstruct_roi_selection_policy(self, image_id: str, valid_rois: List[Tuple], 
                                           roi_num: int, selection_mode: str) -> List[Tuple]:
        """
        Apply ROI selection policy for RTSTRUCT ROIs.
        
        Args:
            image_id: Image identifier for logging
            valid_rois: List of (roi_name, mask, volume) tuples
            roi_num: Number of ROIs to select
            selection_mode: 'per_Img' or 'per_region'
            
        Returns:
            List of selected (roi_name, mask, volume) tuples
        """
        if selection_mode == "per_Img":
            return self._select_rtstruct_rois_per_image(image_id, valid_rois, roi_num)
        elif selection_mode == "per_region":
            return self._select_rtstruct_rois_per_region(image_id, valid_rois, roi_num)
        else:
            logging.warning(f"[{image_id}] Unknown selection mode: {selection_mode}. Using per_Img.")
            return self._select_rtstruct_rois_per_image(image_id, valid_rois, roi_num)
    
    def _select_rtstruct_rois_per_image(self, image_id: str, valid_rois: List[Tuple], roi_num: int) -> List[Tuple]:
        """Select RTSTRUCT ROIs per image (ignore region grouping)."""
        # Sort by volume (largest first) and take top roi_num
        sorted_rois = sorted(valid_rois, key=lambda x: x[2], reverse=True)  # x[2] is volume
        selected = sorted_rois[:roi_num]
        
        logging.info(f"[{image_id}] RTSTRUCT per-image selection: {len(selected)}/{len(valid_rois)} ROIs selected")
        for roi_name, _, volume in selected:
            logging.info(f"[{image_id}]   Selected RTSTRUCT ROI '{roi_name}': {volume} voxels")
        
        return selected
    
    def _select_rtstruct_rois_per_region(self, image_id: str, valid_rois: List[Tuple], roi_num: int) -> List[Tuple]:
        """Group RTSTRUCT ROIs by name prefix and select from each group."""
        # Group ROIs by name prefix (e.g., "GTV", "CTV", "PTV")
        region_groups = self._group_rtstruct_rois_by_region(image_id, valid_rois)
        
        logging.info(f"[{image_id}] RTSTRUCT per-region selection: {len(region_groups)} region groups found")
        logging.info(f"[{image_id}] Will select up to {roi_num} ROIs from each region group")
        
        selected_rois = []
        
        for group_idx, group_rois in enumerate(region_groups):
            # Sort group ROIs by volume and select up to roi_num from this group
            sorted_group = sorted(group_rois, key=lambda x: x[2], reverse=True)  # x[2] is volume
            logging.info(f"[{image_id}] RTSTRUCT Group {group_idx} has {len(sorted_group)} ROIs")
            # Select up to roi_num ROIs from this group (or all if less than roi_num)
            group_selected = sorted_group[:roi_num]
            selected_rois.extend(group_selected)
            
            region_name = group_rois[0][0].split('_')[0] if '_' in group_rois[0][0] else group_rois[0][0]
            logging.info(f"[{image_id}]   RTSTRUCT Region Group {group_idx + 1} ({region_name}): {len(group_selected)}/{len(group_rois)} ROIs selected")
            for roi_name, _, volume in group_selected:
                logging.info(f"[{image_id}]     Selected RTSTRUCT ROI '{roi_name}': {volume} voxels")
        
        logging.info(f"[{image_id}] Total RTSTRUCT ROIs selected across all region groups: {len(selected_rois)}")
        
        return selected_rois
    
    def _group_rtstruct_rois_by_region(self, image_id: str, valid_rois: List[Tuple]) -> List[List[Tuple]]:
        """
        Group RTSTRUCT ROIs by name prefix (e.g., "GTV", "CTV", "PTV").
        
        Args:
            image_id: Image identifier for logging
            valid_rois: List of (roi_name, mask, volume) tuples
            
        Returns:
            List of ROI groups (each group is a list of tuples)
        """
        if len(valid_rois) <= 1:
            return [valid_rois]

        # Group ROIs by name prefix (before underscore or use full name)
        region_groups = {}
        for roi_name, mask, volume in valid_rois:
            # Extract region prefix (e.g., "GTV" from "GTV_Primary")
            region_prefix = roi_name.split('_')[0] if '_' in roi_name else roi_name
            if region_prefix not in region_groups:
                region_groups[region_prefix] = []
            region_groups[region_prefix].append((roi_name, mask, volume))
        
        # Convert to list of groups
        groups = list(region_groups.values())
        
        # Sort groups by the region prefix for consistent ordering
        groups.sort(key=lambda group: group[0][0])  # Sort by first ROI's name
        
        logging.info(f"[{image_id}] Grouped {len(valid_rois)} RTSTRUCT ROIs into {len(groups)} region groups")
        for i, group in enumerate(groups):
            region_prefix = group[0][0].split('_')[0] if '_' in group[0][0] else group[0][0]
            logging.info(f"[{image_id}]   RTSTRUCT Region Group {i + 1} ({region_prefix}): {len(group)} ROIs")
            for roi_name, _, volume in group:
                logging.info(f"[{image_id}]     {roi_name}: {volume} voxels")
        
        return groups
    
    def _get_all_rois_from_mask(self, data_label: np.ndarray, image_id: str) -> List[Tuple]:
        """
        Get all ROIs from mask using connected components analysis.
        
        Args:
            data_label: Label mask array
            image_id: Image identifier for logging
            
        Returns:
            List of (label_value, roi_id, volume, mask) tuples
        """
        from scipy.ndimage import label
        
        all_rois = []
        
        # Get unique labels (excluding 0)
        unique_labels = np.unique(data_label)
        unique_labels = unique_labels[unique_labels > 0]
        
        logging.info(f"[{image_id}] Found {len(unique_labels)} unique label values: {unique_labels}")
        
        # Process each label value
        for lbl in unique_labels:
            binary_mask = (data_label == lbl)  # binary mask for current label
            labeled_array, num_features = label(binary_mask)  # connected components
            
            logging.info(f"[{image_id}] Label {lbl}: {num_features} connected ROIs found")
            
            # Process each connected component for this label
            for roi_id in range(1, num_features + 1):
                roi_mask = (labeled_array == roi_id).astype(np.float32)
                volume = np.sum(roi_mask)
                
                # Create a unique identifier for this ROI
                roi_identifier = f"label_{lbl}_lesion_{roi_id}"
                
                all_rois.append((lbl, roi_id, volume, roi_mask, roi_identifier))
        
        logging.info(f"[{image_id}] Total ROIs found: {len(all_rois)}")
        return all_rois
    
    def _log_roi_volume_summary(self, image_id: str, all_rois: List[Tuple], min_roi_volume: int) -> None:
        """Log ROI volume summary."""
        logging.info(f"[{image_id}] ROI Volume Summary (min_roi_volume = {min_roi_volume}):")
        for label_value, roi_id, volume, _, roi_identifier in all_rois:
            status = "✓ PROCESS" if volume >= min_roi_volume else "✗ SKIP"
            logging.info(f"[{image_id}]   {roi_identifier}: {volume} voxels - {status}")
    
    def _process_all_rois(self, image_id: str, data_original: np.ndarray, data_label: np.ndarray,
                         all_rois: List[Tuple], min_roi_volume: int, sera_params: Dict[str, Any]) -> Tuple:
        """Process all ROIs and return results with new selection policy."""
        # Get ROI selection parameters
        roi_num = sera_params.get('ROI_num', 10)
        roi_selection_mode = sera_params.get('ROI_selection_mode', 'per_Img')
        
        # First, filter ROIs by volume threshold
        valid_rois = []
        skipped_rois = []
        
        for label_value, roi_id, volume, mask, roi_identifier in all_rois:
            if volume < min_roi_volume:
                skipped_rois.append((roi_identifier, volume))
            else:
                valid_rois.append((label_value, roi_id, volume, mask, roi_identifier))
        
        if not valid_rois:
            logging.warning(f"[{image_id}] No ROIs meet the volume threshold ({min_roi_volume})")
            return [], [], [], skipped_rois
        
        # Apply ROI selection policy
        selected_rois = self._apply_roi_selection_policy(
            image_id, valid_rois, roi_num, roi_selection_mode
        )
        
        # Process selected ROIs
        all_features = []
        roi_names = []
        processed_rois = []
        
        for label_idx, (label_value, roi_id, volume, mask, roi_identifier) in enumerate(selected_rois):
            processed_rois.append((roi_identifier, volume))
            roi_name = f"{roi_identifier}"
            
            feature_result = self._extract_features_for_single_roi(
                image_id, data_original, mask, roi_name, sera_params
            )
            
            if feature_result is not None:
                all_features.append(feature_result)
                roi_names.append(roi_name)
        
        return all_features, roi_names, processed_rois, skipped_rois
    
    def _apply_roi_selection_policy(self, image_id: str, valid_rois: List[Tuple], 
                                  roi_num: int, selection_mode: str) -> List[Tuple]:
        """
        Apply ROI selection policy based on mode.
        
        Args:
            image_id: Image identifier for logging
            valid_rois: List of (label_value, volume, mask) tuples
            roi_num: Number of ROIs to select
            selection_mode: 'per_Img' or 'per_region'
            
        Returns:
            List of selected (label_value, volume, mask) tuples
        """
        if selection_mode == "per_Img":
            return self._select_rois_per_image(image_id, valid_rois, roi_num)
        elif selection_mode == "per_region":
            return self._select_rois_per_region(image_id, valid_rois, roi_num)
        else:
            logging.warning(f"[{image_id}] Unknown selection mode: {selection_mode}. Using per_Img.")
            return self._select_rois_per_image(image_id, valid_rois, roi_num)
    
    def _select_rois_per_image(self, image_id: str, valid_rois: List[Tuple], roi_num: int) -> List[Tuple]:
        """Select ROIs per image (ignore region grouping)."""
        # Sort by volume (largest first) and take top roi_num
        sorted_rois = sorted(valid_rois, key=lambda x: x[2], reverse=True)  # x[2] is volume
        selected = sorted_rois[:roi_num]
        
        logging.info(f"[{image_id}] Per-image selection: {len(selected)}/{len(valid_rois)} ROIs selected")
        for label_value, roi_id, volume, _, roi_identifier in selected:
            logging.info(f"[{image_id}]   Selected {roi_identifier}: {volume} voxels")
        
        return selected
    
    def _select_rois_per_region(self, image_id: str, valid_rois: List[Tuple], roi_num: int) -> List[Tuple]:
        """Group ROIs by color and select from each group."""
        # Group ROIs by color (label value)
        color_groups = self._group_rois_by_region(image_id, valid_rois)
        
        logging.info(f"[{image_id}] Per-color selection: {len(color_groups)} color groups found")
        logging.info(f"[{image_id}] Will select up to {roi_num} ROIs from each color group")
        
        selected_rois = []
        
        for group_idx, group_rois in enumerate(color_groups):
            # Sort group ROIs by volume and select up to roi_num from this group
            sorted_group = sorted(group_rois, key=lambda x: x[2], reverse=True)  # x[2] is volume
            logging.info(f"[{image_id}] Group {group_idx} has {len(sorted_group)} ROIs")
            # Select up to roi_num ROIs from this group (or all if less than roi_num)
            group_selected = sorted_group[:roi_num]
            selected_rois.extend(group_selected)
            
            label_value = group_rois[0][0]  # Get the color/label value for this group
            logging.info(f"[{image_id}]   Color Group {group_idx + 1} (Label {label_value}): {len(group_selected)}/{len(group_rois)} ROIs selected")
            for label_value, roi_id, volume, _, roi_identifier in group_selected:
                logging.info(f"[{image_id}]     Selected {roi_identifier}: {volume} voxels")
        
        logging.info(f"[{image_id}] Total ROIs selected across all color groups: {len(selected_rois)}")
        
        return selected_rois
    
    def _group_rois_by_region(self, image_id: str, valid_rois: List[Tuple]) -> List[List[Tuple]]:
        """
        Group ROIs by color (label value).
        
        Args:
            image_id: Image identifier for logging
            valid_rois: List of (label_value, roi_id, volume, mask, roi_identifier) tuples
            
        Returns:
            List of ROI groups (each group is a list of tuples)
        """
        if len(valid_rois) <= 1:
            return [valid_rois]

        # Group ROIs by label value (color)
        color_groups = {}
        for label_value, roi_id, volume, mask, roi_identifier in valid_rois:
            if label_value not in color_groups:
                color_groups[label_value] = []
            color_groups[label_value].append((label_value, roi_id, volume, mask, roi_identifier))
        
        # Convert to list of groups
        groups = list(color_groups.values())
        
        # Sort groups by the label value for consistent ordering
        groups.sort(key=lambda group: group[0][0])  # Sort by first ROI's label value
        
        logging.info(f"[{image_id}] Grouped {len(valid_rois)} ROIs into {len(groups)} color groups")
        for i, group in enumerate(groups):
            label_value = group[0][0]  # Get label value for this group
            logging.info(f"[{image_id}]   Color Group {i + 1} (Label {label_value}): {len(group)} ROIs")
            for label_value, roi_id, volume, _, roi_identifier in group:
                logging.info(f"[{image_id}]     {roi_identifier}: {volume} voxels")
        
        return groups
    
    def _extract_features_for_single_roi(self, image_id: str, data_original: np.ndarray, 
                                       mask: np.ndarray, roi_name: str, sera_params: Dict[str, Any]) -> Optional[List]:
        """Extract features for a single ROI."""
        try:
            logging.info(f"[{image_id}] Extracting features for {roi_name}")
            
            # Check mask and image properties before feature extraction
            self._validate_roi_data(image_id, data_original, mask, roi_name)
            
            # Try the feature extraction with error handling
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                
                logging.info(f"[{image_id}] Calling SERA_FE_main for {roi_name}")
                result = self._call_sera_feature_extraction(data_original, mask, sera_params, roi_name)
                logging.info(f"[{image_id}] SERA function completed for {roi_name}")
            
            if isinstance(result, list) and len(result) > 0:
                feature_vector = result[0]
                if isinstance(feature_vector, (list, np.ndarray)):
                    processed_vector = self._process_feature_vector(image_id, feature_vector, roi_name, sera_params['Feats2out'])
                    return processed_vector if processed_vector is not None else None
                else:
                    logging.warning(f"[{image_id}] Unexpected result format for {roi_name}: {type(feature_vector)}")
            else:
                logging.warning(f"[{image_id}] No features returned for {roi_name}")
                
        except Exception as e:
            logging.error(f"[{image_id}] Error processing {roi_name}: {str(e)}")
            import traceback
            logging.error(f"[{image_id}] Traceback: {traceback.format_exc()}")
        
        return None
    
    def _validate_roi_data(self, image_id: str, data_original: np.ndarray, 
                          mask: np.ndarray, roi_name: str) -> None:
        """Validate ROI data before feature extraction."""
        roi_intensities = data_original[mask > 0]
        logging.info(f"[{image_id}] ROI {roi_name} - Intensity stats: min={np.min(roi_intensities):.2f}, max={np.max(roi_intensities):.2f}, mean={np.mean(roi_intensities):.2f}, std={np.std(roi_intensities):.2f}")
        
        # Check for problematic intensity values
        if np.any(np.isnan(roi_intensities)) or np.any(np.isinf(roi_intensities)):
            logging.warning(f"[{image_id}] ROI {roi_name} contains NaN or Inf values!")
        
        if np.std(roi_intensities) == 0:
            logging.warning(f"[{image_id}] ROI {roi_name} has zero intensity variance!")
    
    def _call_sera_feature_extraction(self, data_original: np.ndarray, mask: np.ndarray, 
                                    sera_params: Dict[str, Any], roi_name: str) -> Any:
        """Call SERA feature extraction function."""
        return self.SERA_FE_main(
            data_original,
            mask,
            sera_params['VoxelSizeInfo'],
            sera_params['BinSize'],
            sera_params['DataType'],
            sera_params['isotVoxSize'],
            sera_params['isotVoxSize2D'],
            sera_params['DiscType'],
            sera_params['qntz'],
            sera_params['VoxInterp'],
            sera_params['ROIInterp'],
            sera_params['isScale'],
            sera_params['isGLround'],
            sera_params['isReSegRng'],
            sera_params['isOutliers'],
            sera_params['isQuntzStat'],
            sera_params['isIsot2D'],
            [sera_params['ReSegIntrvl01'], sera_params['ReSegIntrvl02']],
            sera_params['ROI_PV'],
            sera_params['Feats2out'],
            [sera_params['IVH_Type'], sera_params['IVH_DiscCont'], sera_params['IVH_binSize']],
            sera_params['value_type'],
            roi_name,
            sera_params['IVH_Type'],
            sera_params['IVH_DiscCont'],
            sera_params['IVH_binSize'],
            sera_params['isROIsCombined']
        )
    
    def _process_feature_vector(self, image_id: str, feature_vector: Any, 
                              roi_name: str, feats2out: int) -> Optional[List]:
        """Process and validate feature vector."""
        # Count non-NaN features
        non_nan_count = np.sum(~np.isnan(feature_vector))
        total_count = len(feature_vector)
        logging.info(f"[{image_id}] {roi_name}: Extracted {non_nan_count}/{total_count} valid features")
        
        # Adjust feature vector size if needed
        expected_features = EXPECTED_FEATURE_COUNTS.get(feats2out, 215)
        if len(feature_vector) < expected_features:
            logging.warning(f"[{image_id}] {roi_name}: Got {len(feature_vector)} features, expected {expected_features}. Padding with NaN.")
            feature_vector = list(feature_vector) + [np.nan] * (expected_features - len(feature_vector))
        elif len(feature_vector) > expected_features:
            logging.warning(f"[{image_id}] {roi_name}: Got {len(feature_vector)} features, expected {expected_features}. Truncating.")
            feature_vector = feature_vector[:expected_features]
        
        logging.info(f"[{image_id}] Successfully processed {roi_name}")
        return feature_vector
    
    def _create_results_dataframe(self, image_id: str, all_features: List[List], 
                                roi_names: List[str], sera_params: Dict[str, Any],
                                processed_rois: List[Tuple], skipped_rois: List[Tuple], 
                                min_roi_volume: int) -> Optional[pd.DataFrame]:
        """Create results DataFrame."""
        try:
            feature_names = get_feature_names(sera_params['Feats2out'])
            actual_feature_count = len(all_features[0])
            expected_feature_count = len(feature_names)
            
            logging.info(f"[{image_id}] Feature extraction summary:")
            logging.info(f"[{image_id}]   - Expected features: {expected_feature_count}")
            logging.info(f"[{image_id}]   - Actual features: {actual_feature_count}")
            logging.info(f"[{image_id}]   - ROIs processed: {len(all_features)}")
            
            # Adjust feature names if needed
            feature_names = self._adjust_feature_names(feature_names, actual_feature_count, expected_feature_count, image_id)
            
            # Handle duplicate column names by making them unique
            unique_feature_names = self._make_column_names_unique(feature_names, image_id)
            
            df = pd.DataFrame(all_features, columns=unique_feature_names)
            df['ROI'] = roi_names
            df['File'] = os.path.basename(image_id)
            df['Bin_Size_Used'] = sera_params['BinSize']
            
            # Reorder columns to match expected format
            columns = ['File', 'ROI', 'Bin_Size_Used'] + unique_feature_names
            df = df.reindex(columns=columns)
            
            # Log processing summary
            self._log_processing_summary(image_id, processed_rois, skipped_rois, min_roi_volume)
            
            return df
            
        except Exception as e:
            logging.error(f"[{image_id}] Error creating DataFrame: {str(e)}")
            import traceback
            logging.error(f"[{image_id}] Traceback: {traceback.format_exc()}")
            return None
    
    def _adjust_feature_names(self, feature_names: List[str], actual_count: int, 
                            expected_count: int, image_id: str) -> List[str]:
        """Adjust feature names based on actual feature count."""
        if actual_count != expected_count:
            logging.warning(f"[{image_id}] Feature count mismatch: expected {expected_count}, got {actual_count} features")
            
            if actual_count > expected_count:
                extended_names = feature_names.copy()
                for i in range(expected_count, actual_count):
                    extended_names.append(f"additional_feature_{i+1-expected_count}")
                feature_names = extended_names
            elif actual_count < expected_count:
                feature_names = feature_names[:actual_count]
        
        return feature_names
    
    def _make_column_names_unique(self, feature_names: List[str], image_id: str) -> List[str]:
        """Make column names unique by adding suffixes to duplicates."""
        from collections import Counter
        
        # Count occurrences of each name
        name_counts = Counter(feature_names)
        unique_names = []
        name_occurrences = {}
        
        for name in feature_names:
            if name_counts[name] > 1:
                # This is a duplicate, add suffix
                if name not in name_occurrences:
                    name_occurrences[name] = 1
                else:
                    name_occurrences[name] += 1
                unique_name = f"{name}_{name_occurrences[name]}"
                unique_names.append(unique_name)
            else:
                # This is unique, keep as is
                unique_names.append(name)
        
        # Log if duplicates were found
        duplicates = [name for name, count in name_counts.items() if count > 1]
        if duplicates:
            logging.warning(f"[{image_id}] Found duplicate feature names: {duplicates[:5]}{'...' if len(duplicates) > 5 else ''}")
            logging.warning(f"[{image_id}] Added suffixes to make column names unique")
        
        return unique_names
    
    def _log_processing_summary(self, image_id: str, processed_rois: List[Tuple], 
                              skipped_rois: List[Tuple], min_roi_volume: int) -> None:
        """
        Log processing summary with helpful suggestions.
        
        Args:
            image_id: Image identifier
            processed_rois: List of processed ROIs
            skipped_rois: List of skipped ROIs
            min_roi_volume: Minimum ROI volume threshold
        """
        logging.info(f"[{image_id}] ═══ PROCESSING SUMMARY ═══")
        logging.info(f"[{image_id}] ✓ Processed ROIs: {len(processed_rois)}")
        if processed_rois:
            processed_volumes = [vol for _, vol in processed_rois]
            logging.info(f"[{image_id}]   └─ Volume range: {min(processed_volumes)} - {max(processed_volumes)} voxels")
        
        if skipped_rois:
            logging.warning(f"[{image_id}] ✗ Skipped ROIs: {len(skipped_rois)}")
            skipped_volumes = [vol for _, vol in skipped_rois]
            min_skipped = min(skipped_volumes)
            max_skipped = max(skipped_volumes)
            logging.warning(f"[{image_id}]   └─ Skipped volume range: {min_skipped} - {max_skipped} voxels")
            
            # Provide helpful suggestions
            suggested_threshold = max(min_skipped, 10)  # At least 10 voxels minimum
            if max_skipped < min_roi_volume:
                logging.warning(f"[{image_id}] 💡 SUGGESTION: To include all ROIs, try --min-roi-volume {suggested_threshold}")
            logging.warning(f"[{image_id}] 💡 CURRENT: --min-roi-volume {min_roi_volume} (skipping {len(skipped_rois)} ROIs)")
            logging.warning(f"[{image_id}] 💡 TO INCLUDE ALL: --min-roi-volume {suggested_threshold} (would process {len(processed_rois) + len(skipped_rois)} ROIs)")
        else:
            logging.info(f"[{image_id}] ✓ All {len(processed_rois)} ROIs processed (none skipped)")
    
    def process_batch(self, image_path: Optional[str] = None, mask_path: Optional[str] = None,
                     apply_preprocessing: bool = True, min_roi_volume: int = DEFAULT_MIN_ROI_VOLUME,
                     num_workers: Optional[int] = None, disable_parallel: bool = False,
                     feats2out: int = 2, bin_size: int = 25, roi_num: int = 10, 
                     roi_selection_mode: str = "per_Img", value_type: str = "EXACT_VALUE",
                     memory_limit_mb: int = 1000, enable_memory_optimization: bool = True,
                     aggressive_memory_optimization: bool = False) -> Optional[Dict[str, Any]]:
        """
        Process a batch of images with memory-aware processing.
        
        Args:
            image_path: Path to image files or directory
            mask_path: Path to mask files or directory
            apply_preprocessing: Whether to apply preprocessing
            min_roi_volume: Minimum ROI volume threshold
            num_workers: Number of parallel workers
            disable_parallel: Whether to disable parallel processing
            feats2out: Feature extraction mode
            bin_size: Intensity discretization bin size
            roi_num: Number of ROIs to select
            roi_selection_mode: ROI selection mode
            value_type: Type of value to use
            memory_limit_mb: Memory limit in MB
            enable_memory_optimization: Whether to enable memory optimization
            aggressive_memory_optimization: Whether to enable aggressive memory optimization
            
        Returns:
            Dictionary with processing results
        """
        # Initialize memory processor with user parameters
        global _memory_processor

        # Adjust memory limit for aggressive optimization
        if aggressive_memory_optimization:
            memory_limit_mb = min(memory_limit_mb, 500)  # Cap at 500MB for aggressive mode
            logging.info(f"Aggressive memory optimization enabled - memory limit capped at {memory_limit_mb} MB")
        
        _memory_processor = MemoryAwareRadiomicsProcessor(memory_limit_mb, enable_memory_optimization)
        
        logging.info(f"Memory processor initialized with limit: {memory_limit_mb} MB, optimization: {enable_memory_optimization}, aggressive: {aggressive_memory_optimization}")
        
        # Rest of the method remains the same
        start_time = time.time()

        # Find input files
        image_files, mask_files = self._find_input_files(image_path, mask_path)

        if not image_files:
            logging.error("No image files found")
            return None
        
        if not mask_files:
            logging.error("No mask files found")
            return None
        
        logging.info(f"Found {len(image_files)} image files and {len(mask_files)} mask files")

        # Prepare parameters
        params = self.params.copy()
        params.update({
            'feats2out': feats2out,
            'bin_size': bin_size,
            'roi_num': roi_num,
            'roi_selection_mode': roi_selection_mode,
            'value_type': value_type
        })
        
        # Determine number of workers
        if disable_parallel:
            num_workers = 1
        elif num_workers is None:
            num_workers = min(mp.cpu_count(), len(image_files))
        
        # Reduce workers for aggressive memory optimization
        if aggressive_memory_optimization:
            num_workers = max(1, num_workers // 2)
            logging.info(f"Aggressive memory mode: reduced workers to {num_workers}")
        
        logging.info(f"Using {num_workers} workers for processing")

        # Prepare arguments for parallel processing
        args_list = []
        for i, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
            folder_name = f"batch_{i+1}"
            args = (image_file, mask_file, params, self.output_path, apply_preprocessing,
                   min_roi_volume, folder_name, feats2out, roi_num, roi_selection_mode, value_type)
            args_list.append(args)
        
        # Process files
        all_results = []
        if num_workers == 1:
            # Sequential processing
            for args in args_list:
                result = self.process_single_image_pair(args)
                if result is not None:
                    all_results.append(result)
        else:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(self.process_single_image_pair, args) for args in args_list]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result is not None:
                            all_results.append(result)
                    except Exception as e:
                        logging.error(f"Error in parallel processing: {e}")

        total_processing_time = time.time() - start_time
        # Generate output file      toto
        output_path = self._generate_output_file(folder_name, apply_preprocessing, disable_parallel, len(image_files))

        # Save results      toto
        if all_results:
            saved_results = self._finalize_results(all_results, output_path)
            saved_results['processing_time'] = total_processing_time
        else:
            logging.error("No results returned from radiomics processing")
            saved_results = None

        # Save arguments and parameters     toto
        self._save_parameters(output_path, [args_list[0]])
        return saved_results


    
    def _find_input_files(self, image_path: Optional[str], mask_path: Optional[str]) -> Tuple[List[str], List[str]]:
        """
        Find input image and mask files.
        
        Args:
            image_path: Path to image file or directory
            mask_path: Path to mask file or directory
            
        Returns:
            Tuple of (image_files, mask_files) lists
        """
        # Find image files
        image_files = self._find_image_files(image_path)
        if not image_files:
            return [], []
        
        # Find mask files
        mask_files = self._find_mask_files(mask_path)
        if not mask_files:
            return [], []
        
        return image_files, mask_files
    
    def _find_image_files(self, image_path: Optional[str]) -> List[str]:
        """Find image files from the given path or default directory, supporting multi-dcm and NRRD/NHDR."""
        if image_path:
            format_type = detect_file_format(image_path)
            if format_type == 'multi-dcm':
                # Return list of patient subfolders (each is a DICOM folder)
                from src.utils.file_utils import find_files_by_format
                return find_files_by_format(image_path, 'multi-dcm')
            else:
                return self._find_files_from_path(image_path, "image")
        else:
            return self._find_files_from_default_directory("image")
    
    def _find_mask_files(self, mask_path: Optional[str]) -> List[str]:
        """Find mask files from the given path or default directory, supporting NRRD/NHDR and multi-dcm."""
        if mask_path:
            format_type = detect_file_format(mask_path)
            if format_type == 'multi-dcm':
                from src.utils.file_utils import find_files_by_format
                return find_files_by_format(mask_path, 'multi-dcm')
            else:
                return self._find_files_from_path(mask_path, "mask")
        else:
            return self._find_files_from_default_directory("mask")
    
    def _find_files_from_path(self, path: str, file_type: str) -> List[str]:
        """Find files from a specific path, supporting NRRD/NHDR."""
        if os.path.isfile(path):
            files = [path]
            logging.info(f"Using single {file_type} file: {path}")
        elif os.path.isdir(path):
            file_format = detect_file_format(path)
            logging.info(f"{file_type.capitalize()} directory format detected: {file_format}")
            from src.utils.file_utils import find_files_by_format
            files = find_files_by_format(path, file_format)
            if not files:
                logging.error(f"No supported files found in {path}")
                return []
        else:
            logging.error(f"{file_type.capitalize()} input path does not exist: {path}")
            return []
        return files
    
    def _find_files_from_default_directory(self, file_type: str) -> List[str]:
        """Find files from default directory, supporting NRRD/NHDR and multi-dcm."""
        from ..config.settings import get_default_image_dir, get_default_mask_dir
        if file_type == "image":
            default_dir = get_default_image_dir()
        else:
            default_dir = get_default_mask_dir()
        if not os.path.isdir(default_dir):
            logging.error(f"Default {file_type} directory not found at {default_dir}")
            return []
        file_format = detect_file_format(default_dir)
        from src.utils.file_utils import find_files_by_format
        files = find_files_by_format(default_dir, file_format)
        if not files:
            logging.error(f"No supported files found in {default_dir}")
            return []
        return files

    def _finalize_results(self, all_results: List[pd.DataFrame], output_path: str) -> Dict[str, Any]:
        """
        Finalize and save processing results.

        Args:
            all_results: List of result DataFrames
            output_path: Path to save the final Excel file

        Returns:
            Dictionary with processing results
        """
        final_df = pd.concat(all_results, ignore_index=True)

        # Post-process the DataFrame
        final_df = self._post_process_dataframe(final_df)

        # Save results      toto
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            final_df.to_excel(writer, sheet_name="Results", index=False)

        # Log results summary
        self._log_results_summary(final_df, output_path)

        return {"out": ["Radiomics", final_df, output_path, output_path]}
    
    def _post_process_dataframe(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """Post-process the final DataFrame."""
        # Renumber ROIs to match reference format

        # Fix column names to match reference format
        if 'Bin_Size_Used' in final_df.columns:
            final_df = final_df.rename(columns={'Bin_Size_Used': 'Bin Size'})
            logging.info("Renamed 'Bin_Size_Used' column to 'Bin Size' to match reference format")
        
        # Remove 'File' column if present (not in reference) - do this AFTER ROI renumbering
        if 'File' in final_df.columns:
            final_df = final_df.drop(columns=['File'])
            logging.info("Removed 'File' column to match reference format")
        
        return final_df
    
    def _renumber_rois(self, final_df: pd.DataFrame) -> pd.DataFrame:
        """Renumber ROIs to match reference format."""
        logging.info("Renumbering ROIs to match reference format (mask_1-label X per image)")
        new_roi_names = []
        
        # Group by file and restart ROI numbering for each file
        for file_name in final_df['File'].unique():
            file_mask = final_df['File'] == file_name
            file_data = final_df[file_mask]
            
            roi_counter = 1
            for idx in file_data.index:
                new_roi_name = f"label {roi_counter} salam"
                new_roi_names.append((idx, new_roi_name))
                roi_counter += 1
            
            logging.info(f"File {file_name}: Renumbered {roi_counter-1} ROIs")
        
        # Apply the new ROI names
        for idx, new_name in new_roi_names:
            final_df.loc[idx, 'ROI'] = new_name
        
        logging.info(f"Total ROIs renumbered: {len(new_roi_names)}")
        return final_df
    
    def _generate_output_file(self, folder_name: str,
                            apply_preprocessing: bool, disable_parallel: bool, 
                            num_images: int) -> str:
        """Generate output file path and save results."""
        # Generate output filename
        preprocessing_suffix = "_preprocessed" if apply_preprocessing else ""
        parallel_suffix = "_parallel" if not disable_parallel and num_images > 1 else "_sequential"
        timestamp = datetime.now().strftime("%m-%d-%Y_%H%M%S")
        
        output_filename = OUTPUT_FILENAME_TEMPLATE.format(
            preprocessing_suffix=preprocessing_suffix,
            parallel_suffix=parallel_suffix,
            folder_name=folder_name,
            timestamp=timestamp
        )
        output_path = os.path.join(self.output_path, output_filename)

        return output_path
    
    def _save_parameters(self, excel_path, args_list) -> None:       # toto
        try:
            # Convert path to Path object for robust handling
            excel_path = Path(excel_path)
            # Create parent directories if they don't exist
            excel_path.parent.mkdir(parents=True, exist_ok=True)
            # Check if file exists
            file_exists = excel_path.is_file()

            if file_exists:
                write_to_excel(excel_path, args_list, "Parameters")

        except ValueError as e:
            logging.error(f"Value error: {e}")
            raise
        except PermissionError as e:
            logging.error(f"Permission error: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            raise


    def _log_results_summary(self, final_df: pd.DataFrame, output_path: str) -> None:
        """Log results summary."""
        # Log feature information
        feature_columns = [col for col in final_df.columns if col not in ['ROI', 'Bin Size', 'PatientID']]
        logging.info(f"Total features extracted: {len(feature_columns)}")
        
        # Log feature quality metrics
        self._log_feature_quality_metrics(final_df)
        
        # Global processing summary
        self._log_global_summary(final_df, min_roi_volume=1, total_processing_time=time.time() - time.time())
        
        logging.info(f"🎉 Optimized radiomics features extracted successfully!")
        logging.info(f"📁 Results saved to: {output_path}")
    
    def _log_feature_quality_metrics(self, final_df: pd.DataFrame) -> None:
        """Log feature quality metrics."""
        nan_count = final_df.isnull().sum()
        nan_features = nan_count[nan_count > 0]
        if len(nan_features) > 0:
            logging.warning(f"Features with missing values: {nan_features.to_dict()}")
        else:
            logging.info("No missing values found in extracted features!")
    
    def _log_global_summary(self, final_df: pd.DataFrame, min_roi_volume: int, 
                          total_processing_time: float) -> None:
        """
        Log global processing summary.
        
        Args:
            final_df: Final results DataFrame
            min_roi_volume: Minimum ROI volume used
            total_processing_time: Total processing time
        """
        logging.info("╔══════════════════════════════════════════════════════════════════╗")
        logging.info("║                        GLOBAL SUMMARY                            ║")
        logging.info("╚══════════════════════════════════════════════════════════════════╝")
        
        total_rois_processed = len(final_df)
        roi_counts = final_df['ROI'].value_counts().sort_index()
        unique_roi_types = len(roi_counts)
        
        logging.info(f"📊 FINAL RESULTS:")
        logging.info(f"   ✓ Total ROIs processed: {total_rois_processed}")
        logging.info(f"   ✓ Unique ROI types: {unique_roi_types} (mask_1-label 1 through mask_1-label {unique_roi_types})")
        
        # Estimate how many ROIs might have been skipped globally
        if min_roi_volume > 50:  # If using a high threshold
            logging.warning(f"⚠️  NOTICE: Using min_roi_volume = {min_roi_volume}")
            logging.warning(f"   💡 If you're missing expected ROIs, try a lower threshold like:")
            logging.warning(f"   💡   --min-roi-volume 50   (for small ROIs)")
            logging.warning(f"   💡   --min-roi-volume 10   (for very small ROIs)")
            logging.warning(f"   💡   --min-roi-volume 1    (to include all detected ROIs)")
        
        logging.info(f"⏱️  Total processing time: {total_processing_time:.2f} seconds") 
