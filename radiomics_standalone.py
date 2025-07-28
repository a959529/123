#!/usr/bin/env python3
"""
Refactored Radiomics Standalone Processing Pipeline

This is the main entry point for the refactored radiomics processing pipeline.
The code has been reorganized into a clean, modular structure following clean code principles.

Main improvements:
- Separation of concerns with dedicated modules
- Configuration management
- Error handling and logging
- Type hints and documentation
- Clean architecture with clear dependencies

Supported Input Formats:
-----------------------
Image Files:
  - NIfTI (.nii, .nii.gz)
  - NRRD (.nrrd, .nhdr)
  - DICOM (.dcm, .dicom)
  - Multi-dcm: Directory with subfolders (patients), each containing DICOM files
  - Any other format supported by SimpleITK
  - Type: Medical images (CT, MRI, PET, etc.)
  - Bit depth: Any supported by SimpleITK

Mask Files:
  - Same formats as image files
  - Type: Binary or labeled segmentation masks
  - Requirement: Must have same dimensions as corresponding image
  - For multi-dcm, mask may be a single DICOM file

Usage Examples:
---------------
# Single NIfTI image and mask
python radiomics_standalone.py --image_input path/to/image.nii.gz --mask_input path/to/mask.nii.gz --output results/

# DICOM series (folder) and mask
python radiomics_standalone.py --image_input path/to/dicom_folder --mask_input path/to/mask.dcm --output results/

# Multi-dcm (batch of patients)
python radiomics_standalone.py --image_input path/to/patients_dir --mask_input path/to/masks_dir --output results/

# NRRD image and mask
python radiomics_standalone.py --image_input path/to/image.nrrd --mask_input path/to/mask.nrrd --output results/

# Any format supported by SimpleITK
python radiomics_standalone.py --image_input path/to/image.ext --mask_input path/to/mask.ext --output results/

"""

import sys
import os
import logging
from typing import Optional, Dict, Any

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    from src.cli.argument_parser import parse_arguments, validate_arguments, print_usage_examples
    from src.processing.radiomics_processor import RadiomicsProcessor
    from src.config.settings import LOGGING_CONFIG
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)


def setup_logging() -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format']
    )


def main() -> int:
    """
    Main entry point for the radiomics processing pipeline.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Set up logging
        setup_logging()
        
        # Parse command line arguments
        args = parse_arguments()
        
        # Validate arguments
        if not validate_arguments(args):
            print_usage_examples()
            return 1
        
        # Create processor instance
        processor = RadiomicsProcessor(output_path=args.output)
        
        # Process with memory-aware settings
        result = processor.process_batch(
            image_path=args.image_input,
            mask_path=args.mask_input,
            apply_preprocessing=args.apply_preprocessing,
            min_roi_volume=args.min_roi_volume,
            num_workers=args.num_workers,
            disable_parallel=args.disable_parallel,
            feats2out=args.feats2out,
            bin_size=args.bin_size,
            roi_num=args.roi_num,
            roi_selection_mode=args.roi_selection_mode,
            value_type=args.value_type,
            memory_limit_mb=getattr(args, 'memory_limit', 1000),
            enable_memory_optimization=getattr(args, 'enable_memory_optimization', True),
            aggressive_memory_optimization=getattr(args, 'aggressive_memory_optimization', False)
        )
        
        if result is None:
            logging.error("Processing failed")
            return 1
        
        logging.info("Processing completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logging.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 