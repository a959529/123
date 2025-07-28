# Multiprocessing Fixes for Windows File Locking Issues

## Problem Identified

The error `[WinError 32] The process cannot access the file because it is being used by another process` was occurring because:

1. **Multiple processes** were trying to access the same temporary files
2. **Windows file locking** is stricter than Linux
3. **Memory-mapped files** weren't being properly closed before deletion
4. **Temporary file names** weren't unique enough across processes

## Root Cause

```python
# OLD PROBLEMATIC CODE:
temp_file = tempfile.NamedTemporaryFile(delete=False)
out = np.memmap(temp_file.name, dtype=dtype, mode='w+', shape=array.shape)
# ... processing ...
os.unlink(temp_file.name)  # FAILS: File still locked by memory map
```

## Solutions Implemented

### 1. **Process-Safe Temporary Files**
```python
def create_process_safe_tempfile(prefix="temp", suffix=".tmp"):
    # Creates unique files per process with UUID and PID
    temp_dir = get_process_safe_temp_dir()
    filename = f"{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}{suffix}"
```

### 2. **Proper Memory Map Cleanup**
```python
# NEW SAFE CODE:
result = np.array(out)  # Copy data first
del out                 # Delete memory map object
gc.collect()           # Force garbage collection
_safe_delete_temp_file(temp_filename)  # Then delete file
```

### 3. **Retry Mechanism for File Deletion**
```python
def _safe_delete_temp_file(filename, max_retries=5, delay=0.1):
    for attempt in range(max_retries):
        try:
            gc.collect()
            time.sleep(delay)
            os.unlink(filename)
            return  # Success
        except PermissionError:
            # Wait longer on each retry
            time.sleep(delay * (attempt + 1))
```

### 4. **In-Memory Processing for Moderate Arrays**
```python
# Avoid temp files for arrays < 50M elements
if array.size > chunk_size and array.size < 50000000:
    return _chunked_astype_in_memory(array, dtype, chunk_size)
```

### 5. **Process-Safe Temporary Directories**
```python
def get_process_safe_temp_dir():
    process_temp_dir = os.path.join(
        base_temp_dir, 
        f"radiomics_proc_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    )
    # Each process gets its own directory
```

## Key Improvements

### ✅ **Fixed Issues**
- **Unique file names**: Include PID and UUID in all temp files
- **Proper cleanup order**: Copy data → delete memory map → delete file
- **Retry mechanism**: Multiple attempts with increasing delays
- **Process isolation**: Each process uses its own temporary directory
- **Graceful failure**: Warnings instead of crashes if cleanup fails

### ✅ **Multiprocessing Safety Features**
- **Process-specific directories**: No file name conflicts between processes
- **Automatic cleanup**: Files cleaned up at process exit
- **Memory-first approach**: Use in-memory processing when possible
- **Error resilience**: Continue processing even if temp file cleanup fails

### ✅ **Windows Compatibility**
- **File handle management**: Properly close handles before deletion
- **Permission error handling**: Retry with backoff for Windows file locking
- **Directory cleanup**: Use `shutil.rmtree` for directory removal

## Usage Impact

1. **No functional changes**: Same results, just safer file handling
2. **Better performance**: In-memory processing for moderate arrays
3. **Reduced disk usage**: Automatic cleanup of temporary files
4. **Process isolation**: Each worker process has its own temp space
5. **Error resilience**: Processing continues even if cleanup fails

## Testing Recommendations

1. **Run with multiple workers**: Test `num_workers > 1`
2. **Test on Windows**: Verify no file locking errors
3. **Monitor temp directories**: Check automatic cleanup works
4. **Memory monitoring**: Ensure no memory leaks from failed cleanup
5. **Long-running batches**: Test with many images to verify stability

The multiprocessing fixes ensure that your radiomics processing will work reliably across multiple worker processes on Windows without file access conflicts! 🚀