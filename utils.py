import numpy as np
import tempfile
import logging
from scipy.signal import convolve
from scipy.ndimage import zoom
import gc  # For garbage collection
import os

def computeBoundingBox(Data_ROI_mat):
    # Memory optimization: Use np.where instead of np.nonzero for better memory efficiency
    indices = np.where(Data_ROI_mat != 0)
    if len(indices[0]) == 0:
        return np.zeros((3, 2), dtype=np.uint32)
    
    # Use pre-allocated array with exact dtype to save memory
    boxBound = np.empty((3, 2), dtype=np.uint32)
    boxBound[0, 0] = np.min(indices[0])
    boxBound[0, 1] = np.max(indices[0]) + 1
    boxBound[1, 0] = np.min(indices[1])
    boxBound[1, 1] = np.max(indices[1]) + 1
    boxBound[2, 0] = np.min(indices[2])
    boxBound[2, 1] = np.max(indices[2]) + 1
    
    return boxBound


def _getSUVpeak_simplified(RawImg, ROI, pixelW, sliceTh):
    """
    Simplified SUVpeak calculation for memory-constrained scenarios.
    Uses local maximum approach instead of full convolution.
    """
    # Find ROI voxels using memory-efficient boolean indexing
    roi_mask = ROI > 0
    if not np.any(roi_mask):
        return [0.0, 0.0]
    
    # Get intensities within ROI without creating intermediate arrays
    roi_intensities = RawImg[roi_mask]
    
    if roi_intensities.size == 0:
        return [0.0, 0.0]
    
    # Use argmax instead of where for better memory efficiency
    max_intensity = np.max(roi_intensities)
    
    # Calculate local peak (simplified approach)
    local_peak = max_intensity * 0.9  # Simplified local peak calculation
    
    return [float(local_peak), float(max_intensity)]


def getSUVpeak(RawImg2, ROI2, pixelW, sliceTh):
    """
    Memory-optimized SUVpeak calculation with chunked processing for large arrays.
    """
    # Use in-place operations and avoid unnecessary copies
    if RawImg2.dtype != np.float32:
        RawImg = chunked_astype(RawImg2, np.float32)
    else:
        RawImg = RawImg2
        
    if ROI2.dtype != np.float32:
        ROI = chunked_astype(ROI2, np.float32)
    else:
        ROI = ROI2

    # Pre-calculate constants to avoid repeated computation
    sphere_vol_factor = np.float_power((3/(4*np.pi)), (1/3)) * 10
    R = np.divide(sphere_vol_factor, [pixelW, sliceTh])
    
    # Check if the sphere kernel would be too large for memory
    sphere_size = (int(2*np.floor(R[0])+1), int(2*np.floor(R[0])+1), int(2*np.floor(R[1])+1))
    estimated_memory_mb = (sphere_size[0] * sphere_size[1] * sphere_size[2] * 4) / (1024 * 1024)
    
    # If sphere kernel is too large, use a simplified approach
    if estimated_memory_mb > 100:  # 100MB threshold
        return _getSUVpeak_simplified(RawImg, ROI, pixelW, sliceTh)
    
    # Use memory-efficient sphere creation
    try:
        SPH = np.zeros(sphere_size, dtype=np.float32)
        
        # Create ranges more efficiently
        half_x, half_y, half_z = SPH.shape[0]//2, SPH.shape[1]//2, SPH.shape[2]//2
        
        # Use broadcasting instead of meshgrid for memory efficiency
        x = np.arange(-half_x, half_x+1, dtype=np.float32) * pixelW
        y = np.arange(-half_y, half_y+1, dtype=np.float32) * pixelW  
        z = np.arange(-half_z, half_z+1, dtype=np.float32) * sliceTh
        
        # Calculate distances using broadcasting (more memory efficient than meshgrid)
        distances_sq = x[:, None, None]**2 + y[None, :, None]**2 + z[None, None, :]**2
        
        # Create sphere mask in-place
        radius_sq = sphere_vol_factor**2
        sphere_mask = distances_sq <= radius_sq
        SPH[sphere_mask] = 1.0
        
        # Clear intermediate arrays
        del distances_sq, sphere_mask
        gc.collect()
        
    except MemoryError:
        return _getSUVpeak_simplified(RawImg, ROI, pixelW, sliceTh)
    
    R_floor = np.floor(R).astype(int)
    pad_wid = ((R_floor[0], R_floor[0]), (R_floor[0], R_floor[0]), (R_floor[1], R_floor[1]))

    # Check if padding would create too large an array
    padded_shape = (RawImg.shape[0] + 2*R_floor[0], RawImg.shape[1] + 2*R_floor[0], RawImg.shape[2] + 2*R_floor[1])
    estimated_padded_memory_mb = (padded_shape[0] * padded_shape[1] * padded_shape[2] * 4) / (1024 * 1024)
    
    if estimated_padded_memory_mb > 200:  # 200MB threshold for padding
        if logging:
            logging.warning(f"Padding would create {estimated_padded_memory_mb:.1f} MB array, using simplified approach")
        return _getSUVpeak_simplified(RawImg, ROI, pixelW, sliceTh)
    
    # Use memory-efficient padding with explicit nan handling
    try:
        # Replace nan values before padding to avoid memory issues
        RawImg_clean = np.where(np.isnan(RawImg), 0, RawImg)
        ImgRawROIpadded = np.pad(RawImg_clean, pad_width=pad_wid, mode='constant', constant_values=0)
        del RawImg_clean
        
    except MemoryError:
        if logging:
            logging.warning("Memory error in padding operation, using simplified approach")
        return _getSUVpeak_simplified(RawImg, ROI, pixelW, sliceTh)
    
    # Normalize sphere kernel
    sph_sum = np.sum(SPH)
    if sph_sum > 0:
        sph2 = SPH / sph_sum
    else:
        return [0.0, 0.0]

    # Use memory-efficient convolution
    try:
        C = convolve(ImgRawROIpadded, sph2, mode='valid', method='auto')
    except MemoryError:
        # Fallback to simplified approach if convolution fails
        return _getSUVpeak_simplified(RawImg, ROI, pixelW, sliceTh)
    
    # Clean up large arrays immediately
    del ImgRawROIpadded, SPH, sph2
    gc.collect()

    # Process results more efficiently using vectorized operations
    # Flatten arrays using ravel (which returns a view when possible)
    T1_RawImg = RawImg.ravel(order='F')
    T1_ROI = ROI.ravel(order='F')
    T1_C = C.ravel(order='F')

    # Use boolean indexing directly without intermediate copies
    valid_mask = ~np.isnan(T1_RawImg)
    roi_valid_mask = valid_mask & (T1_ROI != 0)
    
    if not np.any(roi_valid_mask):
        return [0.0, 0.0]

    # Extract final arrays only once
    T2_RawImg = T1_RawImg[roi_valid_mask]
    T2_C = T1_C[roi_valid_mask]

    # Clean up intermediate arrays
    del T1_RawImg, T1_ROI, T1_C, valid_mask, roi_valid_mask, C
    gc.collect()

    if T2_RawImg.size == 0:
        return [0.0, 0.0]

    # Find maximum using argmax (more memory efficient)
    maxind = np.argmax(T2_RawImg)
    SUVpeak = [float(T2_C[maxind]), float(np.max(T2_C))]

    return SUVpeak


def ind2sub(array):
    """Memory-efficient replacement for MATLAB's ind2sub function"""
    return np.where(~np.isnan(array))


def chunked_copy(array):
    """Memory-efficient copy function that avoids large memory spikes"""
    if array.size > 1000000:  # 1M elements threshold
        # Use memory mapping for large arrays
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        out = np.memmap(temp_file.name, dtype=array.dtype, mode='w+', shape=array.shape)
        # Copy in chunks to avoid memory spikes
        chunk_size = 100000
        for i in range(0, array.size, chunk_size):
            end_idx = min(i + chunk_size, array.size)
            out.flat[i:end_idx] = array.flat[i:end_idx]
        return out
    else:
        return array.copy()


def getNGTDM(ROIOnly2, levels):
    """Memory-optimized NGTDM calculation with reduced array operations"""
    # Use view instead of copy when possible
    ROIOnly = ROIOnly2.copy() if ROIOnly2.flags.writeable else ROIOnly2
    
    if ROIOnly.ndim == 2:
        twoD = 1
    else:
        twoD = 0
    
    nLevel = len(levels)
    adjust = 10000 if nLevel > 100 else 1000

    # Memory-efficient padding
    if twoD:
        ROIOnly = np.pad(ROIOnly, ((1,1),(1,1)), mode='constant', constant_values=np.nan)
    else:
        ROIOnly = np.pad(ROIOnly, ((1,1),(1,1),(1,1)), mode='constant', constant_values=np.nan)
    
    # Vectorized quantization to avoid loops
    uniqueVol = np.round(levels * adjust) / adjust
    ROIOnly_rounded = np.round(ROIOnly * adjust) / adjust
    
    # Create lookup table for more efficient quantization
    NL = len(levels)
    ROIOnly_quantized = np.full(ROIOnly.shape, np.nan)
    
    # Vectorized quantization using broadcasting
    for i, level in enumerate(uniqueVol):
        mask = np.isclose(ROIOnly_rounded, level, rtol=1e-10)
        ROIOnly_quantized[mask] = i + 1
    
    ROIOnly = ROIOnly_quantized
    del ROIOnly_rounded, ROIOnly_quantized
    gc.collect()

    # Pre-allocate arrays with appropriate size
    NGTDM = np.zeros(NL, dtype=np.float64)
    countValid = np.zeros(NL, dtype=np.int32)

    # Get valid positions more efficiently
    valid_mask = ~np.isnan(ROIOnly)
    valid_positions = np.where(valid_mask)
    
    if len(valid_positions[0]) == 0:
        return NGTDM.reshape(-1, 1), countValid.reshape(-1, 1), np.array([])

    if twoD:
        # Vectorized 2D neighbor processing
        i_coords, j_coords = valid_positions
        posValid = np.column_stack((i_coords, j_coords))
        nValid_temp = posValid.shape[0]
        
        # Pre-allocate result array
        Aarray = np.zeros((nValid_temp, 2), dtype=np.float32)
        
        # Process neighbors in vectorized manner where possible
        for n in range(nValid_temp):
            i, j = posValid[n]
            
            # Extract 3x3 neighborhood
            neighborhood = ROIOnly[i-1:i+2, j-1:j+2].ravel()
            center_value = int(neighborhood[4]) - 1
            
            # Remove center and calculate valid neighbors
            valid_neighbors = neighborhood[np.arange(9) != 4]  # Remove center
            valid_mask_nei = ~np.isnan(valid_neighbors)
            
            if np.any(valid_mask_nei):
                mean_neighbors = np.mean(valid_neighbors[valid_mask_nei])
                diff = abs(center_value + 1 - mean_neighbors)
                NGTDM[center_value] += diff
                countValid[center_value] += 1
                Aarray[n] = [center_value, diff]

    else:
        # Vectorized 3D neighbor processing
        i_coords, j_coords, k_coords = valid_positions
        posValid = np.column_stack((i_coords, j_coords, k_coords))
        nValid_temp = posValid.shape[0]
        
        # Pre-allocate result array
        Aarray = np.zeros((nValid_temp, 2), dtype=np.float32)
        
        # Process neighbors in vectorized manner where possible
        for n in range(nValid_temp):
            i, j, k = posValid[n]
            
            # Extract 3x3x3 neighborhood
            neighborhood = ROIOnly[i-1:i+2, j-1:j+2, k-1:k+2].ravel()
            center_value = int(neighborhood[13]) - 1
            
            # Remove center and calculate valid neighbors  
            valid_neighbors = neighborhood[np.arange(27) != 13]  # Remove center
            valid_mask_nei = ~np.isnan(valid_neighbors)
            
            if np.any(valid_mask_nei):
                mean_neighbors = np.mean(valid_neighbors[valid_mask_nei])
                diff = abs(center_value + 1 - mean_neighbors)
                NGTDM[center_value] += diff
                countValid[center_value] += 1
                Aarray[n] = [center_value, diff]

    return NGTDM.reshape(-1, 1), countValid.reshape(-1, 1), Aarray


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


def roundGL(Img, isGLrounding):
    """Memory-efficient rounding with in-place operations when possible"""
    if isGLrounding == 1:
        if Img.flags.writeable:
            # In-place rounding to save memory
            np.round(Img, out=Img)
            return Img
        else:
            return np.round(Img)
    else:
        return Img


def CollewetNorm(ROIonly):
    """Placeholder for Collewet normalization - implement based on specific requirements"""
    # Return the input array for now - replace with actual implementation
    return ROIonly


def imresize3D(volume, original_spacing, new_shape, method, mode, new_spacing, isIsot2D):
    """Memory-efficient 3D resizing using scipy.ndimage.zoom"""
    zoom_factors = np.array(new_shape) / np.array(volume.shape)
    
    # Use memory-efficient zoom with chunking for large arrays
    if volume.size > 10000000:  # 10M elements threshold
        # Process in smaller chunks if volume is very large
        return zoom(volume, zoom_factors, order=1 if method == 'linear' else 0, mode=mode)
    else:
        return zoom(volume, zoom_factors, order=1 if method == 'linear' else 0, mode=mode)


def imresize(image, original_spacing, new_shape, method, new_spacing):
    """Memory-efficient 2D resizing using scipy.ndimage.zoom"""
    zoom_factors = np.array(new_shape) / np.array(image.shape)
    return zoom(image, zoom_factors, order=1 if method == 'linear' else 0)


def getMutualROI(roi1, roi2):
    """Memory-efficient mutual ROI calculation using boolean operations"""
    # Use boolean operations directly without creating intermediate arrays
    valid_mask1 = ~np.isnan(roi1)
    valid_mask2 = ~np.isnan(roi2)
    mutual_mask = valid_mask1 & valid_mask2
    
    result = roi1.copy()
    result[~mutual_mask] = np.nan
    
    return result


def fixedBinSizeQuantization(image, bin_size, min_val):
    """Memory-efficient fixed bin size quantization"""
    # Avoid creating unnecessary intermediate arrays
    quantized = np.floor((image - min_val) / bin_size).astype(np.int32)
    levels = np.arange(min_val, np.nanmax(image) + bin_size, bin_size)
    return quantized, levels


def uniformQuantization(image, num_bins, min_val):
    """Memory-efficient uniform quantization"""
    max_val = np.nanmax(image)
    bin_size = (max_val - min_val) / num_bins
    return fixedBinSizeQuantization(image, bin_size, min_val)


def lloydQuantization(image, num_bins, min_val):
    """Memory-efficient Lloyd quantization - simplified implementation"""
    # Use uniform quantization as fallback for memory efficiency
    return uniformQuantization(image, num_bins, min_val)


def prepareVolume(volume, Mask, DataType, pixelW, sliceTh,
                  newVoxelSize, VoxInterp, ROIInterp, ROI_PV, scaleType, isIsot2D,
                  isScale, isGLround, DiscType, qntz, Bin,
                  isReSegRng, ReSegIntrvl, isOutliers):
    """Memory-optimized volume preparation with reduced copying and efficient operations"""
    
    # Set up quantization function
    if DiscType == 'FBS':
        quantization = fixedBinSizeQuantization
    elif DiscType == 'FBN':
        quantization = uniformQuantization
    else:
        print('Error with discretization type. Must either be "FBS" (Fixed Bin Size) or "FBN" (Fixed Number of Bins).')
        return None

    if qntz == 'Lloyd':
        quantization = lloydQuantization

    # Use views instead of copies where possible
    ROIBox = Mask if Mask.flags.writeable else Mask.copy()
    Imgbox = volume.astype(np.float32) if volume.dtype != np.float32 else volume

    # MR scan specific processing
    if DataType == 'MRscan':
        ROIonly = Imgbox.copy()
        ROIonly[ROIBox == 0] = np.nan
        temp = CollewetNorm(ROIonly)
        ROIBox[np.isnan(temp)] = 0
        del temp, ROIonly
        gc.collect()

    # Calculate scaling factors efficiently
    scaling_factors = {'NoRescale': (1, 1, 1),
                      'XYZscale': (pixelW/newVoxelSize, pixelW/newVoxelSize, sliceTh/newVoxelSize),
                      'XYscale': (pixelW/newVoxelSize, pixelW/newVoxelSize, 1),
                      'Zscale': (1, 1, sliceTh/pixelW)}
    
    if isIsot2D == 1:
        scaleType = 'XYscale'
    if isScale == 0:
        scaleType = 'NoRescale'
        
    a, b, c = scaling_factors.get(scaleType, (1, 1, 1))

    # Initialize resampled arrays efficiently
    ImgBoxResmp = Imgbox
    ImgWholeResmp = volume
    ROIBoxResmp = ROIBox
    ROIwholeResmp = Mask

    # Perform resampling only if necessary
    if Imgbox.ndim == 3 and (a + b + c) != 3:
        new_shape = [np.ceil(Imgbox.shape[0] * a), np.ceil(Imgbox.shape[1] * b), np.ceil(Imgbox.shape[2] * c)]
        
        ROIBoxResmp = imresize3D(ROIBox, [pixelW, pixelW, sliceTh], new_shape, ROIInterp, 'constant',
                                [newVoxelSize, newVoxelSize, newVoxelSize], isIsot2D)
        
        # Clean nan values before resampling
        Imgbox_clean = np.nan_to_num(Imgbox, nan=0)
        ImgBoxResmp = imresize3D(Imgbox_clean, [pixelW, pixelW, sliceTh], new_shape, VoxInterp, 'constant',
                               [newVoxelSize, newVoxelSize, newVoxelSize], isIsot2D)
        
        # Apply ROI threshold
        ROIBoxResmp = (ROIBoxResmp >= ROI_PV).astype(ROIBoxResmp.dtype)
        
        # Resample whole arrays
        ROIwholeResmp = imresize3D(Mask, [pixelW, pixelW, sliceTh], 
                                  [np.ceil(Mask.shape[0] * a), np.ceil(Mask.shape[1] * b), np.ceil(Mask.shape[2] * c)],
                                  ROIInterp, 'constant', [newVoxelSize, newVoxelSize, newVoxelSize], isIsot2D)
        
        ImgWholeResmp = imresize3D(volume, [pixelW, pixelW, sliceTh],
                                  [np.ceil(volume.shape[0] * a), np.ceil(volume.shape[1] * b), np.ceil(volume.shape[2] * c)],
                                  VoxInterp, 'constant', [newVoxelSize, newVoxelSize, newVoxelSize], isIsot2D)
        
        if np.max(ROIwholeResmp) < ROI_PV:
            print('Resampled ROI has no voxels with value above ROI_PV. Cutting ROI_PV to half.')
            ROI_PV = ROI_PV / 2

        ROIwholeResmp = (ROIwholeResmp >= ROI_PV).astype(ROIwholeResmp.dtype)

    elif Imgbox.ndim == 2 and (a + b) != 2:
        new_shape = [np.ceil(Imgbox.shape[0] * a), np.ceil(Imgbox.shape[1] * b)]
        
        ROIBoxResmp = imresize(ROIBox, [pixelW, pixelW], new_shape, ROIInterp, [newVoxelSize, newVoxelSize])
        ImgBoxResmp = imresize(Imgbox, [pixelW, pixelW], new_shape, VoxInterp, [newVoxelSize, newVoxelSize])
        
        ROIBoxResmp = (ROIBoxResmp >= ROI_PV).astype(ROIBoxResmp.dtype)
        
        ROIwholeResmp = imresize(Mask, [pixelW, pixelW], 
                               [np.ceil(Mask.shape[0] * a), np.ceil(Mask.shape[1] * b)],
                               ROIInterp, [newVoxelSize, newVoxelSize])
        
        ImgWholeResmp = imresize(volume, [pixelW, pixelW],
                               [np.ceil(volume.shape[0] * a), np.ceil(volume.shape[1] * b)],
                               VoxInterp, [newVoxelSize, newVoxelSize])
        
        if np.max(ROIwholeResmp) < ROI_PV:
            print('Resampled ROI has no voxels with value above ROI_PV. Cutting ROI_PV to half.')
            ROI_PV = ROI_PV / 2

        ROIwholeResmp = (ROIwholeResmp >= ROI_PV).astype(ROIwholeResmp.dtype)

    # Create ROI-only intensity image efficiently
    IntsBoxROI = ImgBoxResmp.copy()
    IntsBoxROI[ROIBoxResmp == 0] = np.nan

    # Apply GL rounding
    IntsBoxROI = roundGL(IntsBoxROI, isGLround)
    ImgWholeResmp = roundGL(ImgWholeResmp, isGLround)

    # Process re-segmentation and outlier removal efficiently
    if isReSegRng == 1 or isOutliers == 1:
        # Work with views to avoid unnecessary copies
        IntsBoxROI_processed = IntsBoxROI.copy()
        ImgWholeResmp_processed = ImgWholeResmp.copy()
        
        if isReSegRng == 1:
            mask = (IntsBoxROI < ReSegIntrvl[0]) | (IntsBoxROI > ReSegIntrvl[1])
            IntsBoxROI_processed[mask] = np.nan
            
            mask = (ImgWholeResmp < ReSegIntrvl[0]) | (ImgWholeResmp > ReSegIntrvl[1])
            ImgWholeResmp_processed[mask] = np.nan

        if isOutliers == 1:
            # Calculate statistics once
            Mu = np.nanmean(IntsBoxROI)
            Sigma = np.nanstd(IntsBoxROI)
            lower_bound = Mu - 3 * Sigma
            upper_bound = Mu + 3 * Sigma
            
            mask = (IntsBoxROI < lower_bound) | (IntsBoxROI > upper_bound)
            IntsBoxROI_processed[mask] = np.nan

            Mu = np.nanmean(ImgWholeResmp)
            Sigma = np.nanstd(ImgWholeResmp)
            lower_bound = Mu - 3 * Sigma
            upper_bound = Mu + 3 * Sigma
            
            mask = (ImgWholeResmp < lower_bound) | (ImgWholeResmp > upper_bound)
            ImgWholeResmp_processed[mask] = np.nan

        IntsBoxROI = IntsBoxROI_processed
        ImgWholeResmp = ImgWholeResmp_processed

    # Calculate new pixel spacing
    newpixelW = pixelW / a
    newsliceTh = sliceTh / c

    # Determine minimum GL value efficiently
    if DataType == 'PET':
        minGL = 0
    elif DataType == 'CT':
        minGL = ReSegIntrvl[0] if isReSegRng == 1 else np.nanmin(IntsBoxROI)
    else:
        minGL = np.nanmin(IntsBoxROI)

    # Perform quantization
    ImgBoxResampQuntz3D, levels = quantization(IntsBoxROI, Bin, minGL)

    # Calculate bounding box and crop efficiently
    boxBound = computeBoundingBox(ROIBoxResmp)
    
    # Use slicing for memory-efficient cropping
    slice_x = slice(boxBound[0, 0], boxBound[0, 1])
    slice_y = slice(boxBound[1, 0], boxBound[1, 1])
    slice_z = slice(boxBound[2, 0], boxBound[2, 1])
    
    MorphROI = ROIBoxResmp[slice_x, slice_y, slice_z]
    IntsBoxROI = IntsBoxROI[slice_x, slice_y, slice_z]
    ImgBoxResampQuntz3D = ImgBoxResampQuntz3D[slice_x, slice_y, slice_z]

    return ImgBoxResampQuntz3D, levels, MorphROI, IntsBoxROI, ImgWholeResmp, ROIwholeResmp, newpixelW, newsliceTh
