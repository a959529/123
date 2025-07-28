import logging
import numpy as np
from typing import Tuple, Optional
import os
import psutil
import gc
import tempfile


ALLOWED_EXTENSIONS = [".nii.gz", ".nii", ".dcm", ".dicom", ".nrrd"]


path_dir = os.getcwd()

def handle_math_operations(feature_vector, epsilon=1e-30):      # toto
    # using epsilon instead of zero to prevend division by zero and sqrt of negative values
    mask = (feature_vector <= 0.)
    feature_vector[mask] = epsilon
    return feature_vector


def synthesis_small_RoI(array, array_shape, target_shape=(2, 2, 2)):      # toto

    # Check if original shape is smaller in all dimensions
    if all(orig < target for orig, target in zip(array_shape, target_shape)):
        # Repeat the values to reach desired shape
        reps = [t // s for s, t in zip(array_shape, target_shape)]
        new_array = np.tile(array, reps)
    else:
        new_array = array  # no expansion needed
    new_shape = new_array.shape

    return new_array, new_shape


def get_memory_usage() -> float:
    """
    Get current memory usage in MB.

    Returns:
        Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def check_memory_available(required_mb: float) -> bool:
    """
    Check if required memory is available.

    Args:
        required_mb: Required memory in MB

    Returns:
        True if memory is available, False otherwise
    """
    available_memory = psutil.virtual_memory().available / 1024 / 1024
    return available_memory > required_mb * 1.5  # 50% buffer


def estimate_array_memory(shape: Tuple[int, ...], dtype: np.dtype) -> float:
    """
    Estimate memory usage for an array.

    Args:
        shape: Array shape
        dtype: Array data type

    Returns:
        Estimated memory usage in MB
    """
    element_size = dtype.itemsize
    total_elements = np.prod(shape)
    return (total_elements * element_size) / (1024 * 1024)


def safe_array_operation(func, *args, **kwargs):
    """
    Safely execute array operations with memory monitoring.

    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        Function result
    """
    try:
        return func(*args, **kwargs)
    except MemoryError as e:
        logging.error(f"Memory error in {func.__name__}: {e}")
        # Force garbage collection
        gc.collect()
        raise


def optimize_array_dtype(array: np.ndarray, target_dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Optimize array data type for memory efficiency.

    Args:
        array: Input array
        target_dtype: Target data type (if None, auto-detect)

    Returns:
        Optimized array
    """
    if target_dtype is None:
        # Auto-detect optimal dtype
        if array.dtype == np.float64:
            # Check if float32 is sufficient
            if np.all(np.isfinite(array)) and np.max(np.abs(array)) < 3.4e38:
                target_dtype = np.float32
        elif array.dtype == np.int64:
            # Check if int32 is sufficient
            if np.max(np.abs(array)) < 2**31:
                target_dtype = np.int32

    if target_dtype and target_dtype != array.dtype:
        try:
            return array.astype(target_dtype)
        except (OverflowError, ValueError):
            logging.warning(f"Could not convert array to {target_dtype}, keeping original dtype")

    return array


def log_memory_usage(operation_name: str):
    """
    Log memory usage for debugging.

    Args:
        operation_name: Name of the operation
    """
    memory_mb = get_memory_usage()
    logging.info(f"[MEMORY] {operation_name}: {memory_mb:.1f} MB")


def chunked_astype(array, dtype, chunk_size=1000000):
    """Memory-efficient type conversion using memory mapping for large arrays"""
    if array.size > chunk_size:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        out = np.memmap(temp_file.name, dtype=dtype, mode='w+', shape=array.shape)

        # Process in chunks to avoid memory spikes
        for i in range(0, array.size, chunk_size):
            end_idx = min(i + chunk_size, array.size)
            out.flat[i:end_idx] = array.flat[i:end_idx].astype(dtype)

        return out
    else:
        return array.astype(dtype)

def chunked_quantization(array, minGL, BinSize, chunk_size=1000000):
    if array.size > chunk_size:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        out = np.memmap(temp_file.name, dtype=array.dtype, mode='w+', shape=array.shape)
        for i in range(0, array.size, chunk_size):
            end_idx = min(i + chunk_size, array.size)
            chunk = array.flat[i:end_idx]
            out.flat[i:end_idx] = np.floor((chunk - minGL) / BinSize) + 1
        return np.array(out)
    else:
        return np.floor((array - minGL) / BinSize) + 1


def memory_efficient_unique(array: np.ndarray) -> np.ndarray:
    """
    Memory-efficient unique value detection.

    Args:
        array: Input array

    Returns:
        Unique values
    """
    if array.size < 1000000:  # Small arrays
        return np.unique(array)

    # For large arrays, use sampling approach
    sample_size = min(1000000, array.size // 10)
    sample_indices = np.random.choice(array.size, sample_size, replace=False)
    sample_values = array.flat[sample_indices]

    unique_sample = np.unique(sample_values)

    # Add common values that might be missing
    common_values = [0, 1, -1, np.nan]
    for val in common_values:
        if val not in unique_sample and np.any(array == val):
            unique_sample = np.append(unique_sample, val)

    return unique_sample
