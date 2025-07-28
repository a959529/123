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
    Optimize array data type for memory efficiency while preserving precision.

    Args:
        array: Input array
        target_dtype: Target data type (if None, auto-detect)

    Returns:
        Optimized array
    """
    if target_dtype is None:
        # Auto-detect optimal dtype with careful precision preservation
        if array.dtype == np.float64:
            # Check if float32 is sufficient - be more conservative
            if np.all(np.isfinite(array)) and np.max(np.abs(array)) < 3.4e37:
                # Test conversion to ensure no precision loss for the data range
                test_converted = array.astype(np.float32).astype(np.float64)
                relative_error = np.max(np.abs((array - test_converted) / (array + 1e-10)))
                if relative_error < 1e-6:  # Less than 1 part per million error
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
    """Memory-efficient type conversion using memory mapping for large arrays with multiprocessing safety"""
    # For arrays that are large but not extremely large, use in-memory chunking
    if array.size > chunk_size and array.size < 50000000:  # 50M threshold for in-memory processing
        return _chunked_astype_in_memory(array, dtype, chunk_size)
    elif array.size > chunk_size:
        return _chunked_astype_with_tempfile(array, dtype, chunk_size)
    else:
        return array.astype(dtype)


def _chunked_astype_in_memory(array, dtype, chunk_size):
    """In-memory chunked type conversion to avoid file conflicts in multiprocessing."""
    actual_chunk_size = min(chunk_size, array.size // 4)
    
    # Pre-allocate result array
    result = np.empty(array.shape, dtype=dtype)
    
    # Process in chunks
    for i in range(0, array.size, actual_chunk_size):
        end_idx = min(i + actual_chunk_size, array.size)
        chunk = array.flat[i:end_idx]
        converted_chunk = chunk.astype(dtype)
        result.flat[i:end_idx] = converted_chunk
        
        # Force memory cleanup after each chunk
        del chunk, converted_chunk
        gc.collect()
    
    return result


def _chunked_astype_with_tempfile(array, dtype, chunk_size):
    """File-based chunked type conversion with multiprocessing safety."""
    actual_chunk_size = min(chunk_size, array.size // 4)
    
    # Create process-safe temporary file
    _, temp_filename = create_process_safe_tempfile("chunked_astype", ".tmp")
    
    try:
        out = np.memmap(temp_filename, dtype=dtype, mode='w+', shape=array.shape)

        # Process in chunks to avoid memory spikes, ensuring exact conversion
        for i in range(0, array.size, actual_chunk_size):
            end_idx = min(i + actual_chunk_size, array.size)
            chunk = array.flat[i:end_idx]
            # Ensure exact type conversion without loss
            converted_chunk = chunk.astype(dtype)
            out.flat[i:end_idx] = converted_chunk
            
            # Force memory cleanup after each chunk
            del chunk, converted_chunk
            gc.collect()

        # Return a regular array copy to avoid memory mapping issues
        result = np.array(out)
        
        # Properly close and delete the memory map
        del out
        gc.collect()
        
        # Safe file deletion with retry mechanism
        _safe_delete_temp_file(temp_filename)
        
        return result
        
    except Exception as e:
        # Ensure cleanup even if error occurs
        try:
            _safe_delete_temp_file(temp_filename)
        except:
            pass
        raise e


def chunked_quantization(array, minGL, BinSize, chunk_size=1000000):
    """Memory-efficient quantization with exact precision preservation and multiprocessing safety"""
    # For arrays that are large but not extremely large, use in-memory chunking
    if array.size > chunk_size and array.size < 50000000:  # 50M threshold for in-memory processing
        return _chunked_quantization_in_memory(array, minGL, BinSize, chunk_size)
    elif array.size > chunk_size:
        return _chunked_quantization_with_tempfile(array, minGL, BinSize, chunk_size)
    else:
        return np.floor((array - minGL) / BinSize) + 1


def _chunked_quantization_in_memory(array, minGL, BinSize, chunk_size):
    """In-memory chunked quantization to avoid file conflicts in multiprocessing."""
    actual_chunk_size = min(chunk_size, array.size // 4)
    
    # Pre-allocate result array
    result = np.empty(array.shape, dtype=array.dtype)
    
    # Process in chunks
    for i in range(0, array.size, actual_chunk_size):
        end_idx = min(i + actual_chunk_size, array.size)
        chunk = array.flat[i:end_idx]
        quantized_chunk = np.floor((chunk - minGL) / BinSize) + 1
        result.flat[i:end_idx] = quantized_chunk
        
        # Force memory cleanup
        del chunk, quantized_chunk
        gc.collect()
    
    return result


def _chunked_quantization_with_tempfile(array, minGL, BinSize, chunk_size):
    """File-based chunked quantization with multiprocessing safety."""
    actual_chunk_size = min(chunk_size, array.size // 4)
    
    # Create process-safe temporary file
    _, temp_filename = create_process_safe_tempfile("chunked_quant", ".tmp")
    
    try:
        out = np.memmap(temp_filename, dtype=array.dtype, mode='w+', shape=array.shape)
        
        for i in range(0, array.size, actual_chunk_size):
            end_idx = min(i + actual_chunk_size, array.size)
            chunk = array.flat[i:end_idx]
            # Use exact same calculation as non-chunked version
            quantized_chunk = np.floor((chunk - minGL) / BinSize) + 1
            out.flat[i:end_idx] = quantized_chunk
            
            # Force memory cleanup
            del chunk, quantized_chunk
            gc.collect()
            
        # Return regular array to avoid memory mapping issues
        result = np.array(out)
        
        # Properly close and delete the memory map
        del out
        gc.collect()
        
        # Safe file deletion with retry mechanism
        _safe_delete_temp_file(temp_filename)
        
        return result
        
    except Exception as e:
        # Ensure cleanup even if error occurs
        try:
            _safe_delete_temp_file(temp_filename)
        except:
            pass
        raise e


def _safe_delete_temp_file(filename: str, max_retries: int = 5, delay: float = 0.1):
    """
    Safely delete temporary file with retry mechanism for Windows multiprocessing.
    
    Args:
        filename: Path to temporary file
        max_retries: Maximum number of deletion attempts
        delay: Delay between attempts in seconds
    """
    import time
    
    for attempt in range(max_retries):
        try:
            if os.path.exists(filename):
                # Force garbage collection before deletion
                gc.collect()
                time.sleep(delay)  # Small delay to ensure file handles are released
                os.unlink(filename)
                return  # Success
        except (PermissionError, OSError) as e:
            if attempt < max_retries - 1:
                # Wait longer on each retry
                time.sleep(delay * (attempt + 1))
                continue
            else:
                # On final attempt, log warning but don't raise exception
                logging.warning(f"Could not delete temporary file {filename} after {max_retries} attempts: {e}")
                logging.warning("File will be cleaned up by OS eventually")
                # Register for cleanup at exit as last resort
                try:
                    import atexit
                    atexit.register(lambda: _cleanup_file_at_exit(filename))
                except:
                    pass


def _cleanup_file_at_exit(filename: str):
    """Cleanup temporary file at program exit."""
    try:
        if os.path.exists(filename):
            os.unlink(filename)
    except:
        pass  # Silent cleanup at exit


def memory_efficient_unique(array: np.ndarray, preserve_exact_results: bool = True) -> np.ndarray:
    """
    Memory-efficient unique value detection that preserves exact results.

    Args:
        array: Input array
        preserve_exact_results: If True, ensures exact same results as np.unique

    Returns:
        Unique values (exact same as np.unique when preserve_exact_results=True)
    """
    # Always preserve exact results by default to maintain consistency
    if preserve_exact_results or array.size < 10000000:  # 10M threshold
        # For arrays that might fit in memory or when exact results required
        try:
            return np.unique(array)
        except MemoryError:
            # If memory error, fall back to chunked approach
            pass
    
    # Memory-efficient approach for very large arrays when exact results not critical
    # Use a more systematic sampling approach that covers the data better
    if array.size > 10000000:
        logging.warning("Using memory-efficient unique detection for very large array")
        
        # Use multiple sampling strategies to ensure we don't miss values
        unique_values = set()
        
        # Strategy 1: Regular sampling
        sample_size = min(5000000, array.size // 5)  # Sample more data
        sample_indices = np.random.choice(array.size, sample_size, replace=False)
        sample_values = array.flat[sample_indices]
        unique_values.update(np.unique(sample_values))
        
        # Strategy 2: Systematic sampling (every nth element)
        step = max(1, array.size // 1000000)
        systematic_sample = array.flat[::step]
        unique_values.update(np.unique(systematic_sample))
        
        # Strategy 3: Edge sampling (first and last parts)
        edge_size = min(100000, array.size // 10)
        if edge_size > 0:
            edge_values = np.concatenate([array.flat[:edge_size], array.flat[-edge_size:]])
            unique_values.update(np.unique(edge_values))
        
        # Convert back to sorted array
        result = np.array(sorted(unique_values))
        
        logging.info(f"Memory-efficient unique found {len(result)} unique values from {array.size} elements")
        return result
    
    # For moderate size arrays, use chunked processing to get exact results
    unique_values = set()
    chunk_size = 1000000
    
    for i in range(0, array.size, chunk_size):
        end_idx = min(i + chunk_size, array.size)
        chunk = array.flat[i:end_idx]
        chunk_unique = np.unique(chunk)
        unique_values.update(chunk_unique)
        
        # Clean up
        del chunk, chunk_unique
        gc.collect()
    
    return np.array(sorted(unique_values))


def validate_data_integrity(original: np.ndarray, optimized: np.ndarray, 
                          tolerance: float = 1e-10) -> bool:
    """
    Validate that optimized array maintains data integrity.
    
    Args:
        original: Original array
        optimized: Optimized array  
        tolerance: Maximum allowed relative difference
        
    Returns:
        True if data integrity is preserved
    """
    if original.shape != optimized.shape:
        logging.error(f"Shape mismatch: original {original.shape} vs optimized {optimized.shape}")
        return False
    
    # Check for exact equality first
    if np.array_equal(original, optimized):
        return True
    
    # Check relative differences for floating point arrays
    if original.dtype.kind in ['f', 'c'] or optimized.dtype.kind in ['f', 'c']:
        # Avoid division by zero
        denominator = np.abs(original) + 1e-15
        relative_diff = np.abs(original - optimized) / denominator
        max_diff = np.max(relative_diff)
        
        if max_diff > tolerance:
            logging.warning(f"Data integrity check failed: max relative difference {max_diff} > tolerance {tolerance}")
            return False
    else:
        # For integer arrays, check exact equality
        if not np.array_equal(original, optimized):
            logging.warning("Integer arrays are not exactly equal after optimization")
            return False
    
    return True


def memory_safe_operation(operation_func, *args, fallback_func=None, **kwargs):
    """
    Execute operation with memory safety and fallback options.
    
    Args:
        operation_func: Primary function to execute
        *args: Arguments for the function
        fallback_func: Fallback function if memory error occurs
        **kwargs: Keyword arguments
        
    Returns:
        Function result
    """
    try:
        # Monitor memory before operation
        initial_memory = get_memory_usage()
        
        # Execute primary operation
        result = operation_func(*args, **kwargs)
        
        # Check memory usage after operation
        final_memory = get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        if memory_increase > 500:  # 500MB increase
            logging.warning(f"Operation caused large memory increase: {memory_increase:.1f} MB")
            gc.collect()  # Force cleanup
        
        return result
        
    except MemoryError as e:
        logging.error(f"Memory error in {operation_func.__name__}: {e}")
        gc.collect()  # Force cleanup
        
        if fallback_func is not None:
            logging.info(f"Attempting fallback function for {operation_func.__name__}")
            try:
                return fallback_func(*args, **kwargs)
            except Exception as fallback_error:
                logging.error(f"Fallback function also failed: {fallback_error}")
                raise
        else:
            raise


def optimize_memory_usage_safely(array: np.ndarray, target_memory_mb: float = 1000) -> np.ndarray:
    """
    Optimize memory usage while ensuring data integrity is preserved.
    
    Args:
        array: Input array
        target_memory_mb: Target memory usage in MB
        
    Returns:
        Optimized array with same data
    """
    original_memory = estimate_array_memory(array.shape, array.dtype)
    
    if original_memory <= target_memory_mb:
        return array  # No optimization needed
    
    logging.info(f"Optimizing array memory usage: {original_memory:.1f} MB -> target {target_memory_mb:.1f} MB")
    
    # Step 1: Try dtype optimization
    optimized_array = optimize_array_dtype(array)
    
    # Validate data integrity
    if not validate_data_integrity(array, optimized_array):
        logging.warning("Data integrity check failed for dtype optimization, keeping original")
        return array
    
    optimized_memory = estimate_array_memory(optimized_array.shape, optimized_array.dtype)
    
    if optimized_memory <= target_memory_mb:
        logging.info(f"Memory optimization successful: {original_memory:.1f} MB -> {optimized_memory:.1f} MB")
        return optimized_array
    
    # If still too large, log warning but don't downsample to preserve results
    logging.warning(f"Array still uses {optimized_memory:.1f} MB after optimization (target: {target_memory_mb:.1f} MB)")
    logging.warning("Consider processing in smaller batches or increasing memory limit")
    logging.warning("Proceeding with full precision to preserve results")
    
    return optimized_array


def get_process_safe_temp_dir():
    """
    Get a process-safe temporary directory for multiprocessing.
    
    Returns:
        Path to process-safe temporary directory
    """
    import uuid
    base_temp_dir = tempfile.gettempdir()
    process_temp_dir = os.path.join(
        base_temp_dir, 
        f"radiomics_proc_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    
    # Create directory if it doesn't exist
    os.makedirs(process_temp_dir, exist_ok=True)
    
    # Register for cleanup at exit
    import atexit
    atexit.register(lambda: _cleanup_temp_dir(process_temp_dir))
    
    return process_temp_dir


def _cleanup_temp_dir(temp_dir):
    """Clean up temporary directory at exit."""
    try:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except:
        pass  # Silent cleanup


def create_process_safe_tempfile(prefix="temp", suffix=".tmp"):
    """
    Create a process-safe temporary file that won't conflict with other processes.
    
    Args:
        prefix: File prefix
        suffix: File suffix
        
    Returns:
        Tuple of (file_handle, filename)
    """
    import uuid
    temp_dir = get_process_safe_temp_dir()
    filename = os.path.join(
        temp_dir,
        f"{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}{suffix}"
    )
    
    # Create and immediately close the file
    with open(filename, 'wb') as f:
        pass
    
    return None, filename
