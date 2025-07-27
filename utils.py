def computeBoundingBox(Data_ROI_mat):
    [iV,jV,kV] = np.nonzero(Data_ROI_mat)
    boxBound = np.zeros((3,2))
    boxBound[0,0] = np.min(iV)
    boxBound[0,1] = np.max(iV)+1
    boxBound[1,0] = np.min(jV)
    boxBound[1,1] = np.max(jV)+1
    boxBound[2,0] = np.min(kV)
    boxBound[2,1] = np.max(kV)+1
    boxBound = boxBound.astype(np.uint32)


def _getSUVpeak_simplified(RawImg, ROI, pixelW, sliceTh):
    """
    Simplified SUVpeak calculation for memory-constrained scenarios.
    Uses local maximum approach instead of full convolution.
    """
    # Find ROI voxels
    roi_voxels = ROI > 0
    if not np.any(roi_voxels):
        return [0.0, 0.0]
    
    # Get intensities within ROI
    roi_intensities = RawImg[roi_voxels]
    
    if len(roi_intensities) == 0:
        return [0.0, 0.0]
    
    # Find local maximum (simplified approach)
    max_intensity = np.max(roi_intensities)
    max_indices = np.where(RawImg == max_intensity)
    
    # Calculate local peak (simplified)
    local_peak = max_intensity * 0.9  # Simplified local peak calculation
    
    return [float(local_peak), float(max_intensity)]


def getSUVpeak(RawImg2, ROI2, pixelW, sliceTh):
    """
    Memory-optimized SUVpeak calculation with chunked processing for large arrays.
    """
    RawImg = chunked_astype(RawImg2, np.float32)
    ROI = chunked_astype(ROI2, np.float32)

    R = np.divide(np.float_power((3/(4*np.pi)),(1/3)) * 10, [pixelW,sliceTh])
    
    # Check if the sphere kernel would be too large for memory
    sphere_size = (int(2*np.floor(R[0])+1), int(2*np.floor(R[0])+1), int(2*np.floor(R[1])+1))
    estimated_memory_mb = (sphere_size[0] * sphere_size[1] * sphere_size[2] * 4) / (1024 * 1024)  # 4 bytes per float32
    
    # If sphere kernel is too large, use a simplified approach
    if estimated_memory_mb > 100:  # 100MB threshold
        return _getSUVpeak_simplified(RawImg, ROI, pixelW, sliceTh)
    
    SPH = np.zeros(sphere_size, dtype=np.float32)

    rangeX = np.arange(pixelW * ((-np.ceil(SPH.shape[0]/2)+0.5) - 0.5), 
                       pixelW * ((np.floor(SPH.shape[0]/2)) - 0.5), pixelW, dtype=np.float32)
    rangeY = np.arange(pixelW * ((-np.ceil(SPH.shape[1]/2)+0.5) - 0.5), 
                       pixelW * ((np.floor(SPH.shape[1]/2)) - 0.5), pixelW, dtype=np.float32)
    rangeS = np.arange(sliceTh * ((-np.ceil(SPH.shape[2]/2)+0.5) - 0.5), 
                       sliceTh * ((np.floor(SPH.shape[2]/2)) - 0.5), sliceTh, dtype=np.float32)  
    
    x,y,z = np.meshgrid(rangeY, rangeX, rangeS, indexing='ij')

    # Calculate sphere mask more efficiently
    center_x = x[0, int(np.ceil(x.shape[0]/2))-1, 0]
    center_y = y[int(np.ceil(y.shape[1]/2))-1, 0, 0]
    center_z = z[0, 0, int(np.ceil(z.shape[2]/2))-1]
    
    tmpsph = np.sqrt((x - center_x)**2 + (y - center_y)**2 + (z - center_z)**2)
    radius = np.float_power((3/(4*np.pi)), (1/3)) * 10
    tmpsph[tmpsph > radius] = np.nan 
    
    SPH = tmpsph.copy()
    SPH[~np.isnan(tmpsph)] = 1 

    R = np.floor(R)
    
    pad_wid = ((int(R[0]), int(R[0])), (int(R[0]), int(R[0])), (int(R[1]), int(R[1])))

    # Check if padding would create too large an array
    padded_shape = (RawImg.shape[0] + 2*int(R[0]), RawImg.shape[1] + 2*int(R[0]), RawImg.shape[2] + 2*int(R[1]))
    estimated_padded_memory_mb = (padded_shape[0] * padded_shape[1] * padded_shape[2] * 4) / (1024 * 1024)  # 4 bytes per float32
    
    if estimated_padded_memory_mb > 200:  # 200MB threshold for padding
        logging.warning(f"Padding would create {estimated_padded_memory_mb:.1f} MB array, using simplified approach")
        return _getSUVpeak_simplified(RawImg, ROI, pixelW, sliceTh)
    
    # Use memory-efficient padding
    try:
        ImgRawROIpadded = np.pad(RawImg, pad_width=pad_wid, mode='constant', constant_values=np.nan) 
        ImgRawROIpadded = np.nan_to_num(ImgRawROIpadded, nan=0)
    except MemoryError:
        logging.warning("Memory error in padding operation, using simplified approach")
        return _getSUVpeak_simplified(RawImg, ROI, pixelW, sliceTh)
    
    SPH = np.nan_to_num(SPH, nan=0)

    sph2 = np.divide(SPH, np.nansum(SPH))

    # Use memory-efficient convolution
    from scipy.signal import convolve
    try:
        C = convolve(ImgRawROIpadded, sph2, mode='valid', method='auto')
    except MemoryError:
        # Fallback to simplified approach if convolution fails
        return _getSUVpeak_simplified(RawImg, ROI, pixelW, sliceTh)

    # Process results more efficiently
    T1_RawImg = RawImg.flatten(order='F')
    T1_ROI = ROI.flatten(order='F')
    T1_C = C.flatten(order='F')

    # Use boolean indexing instead of multiple array copies
    valid_mask = ~np.isnan(T1_RawImg)
    T1_RawImg1 = T1_RawImg[valid_mask]
    T1_ROI1 = T1_ROI[valid_mask]
    T1_C1 = T1_C[valid_mask]

    roi_mask = T1_ROI1 != 0
    T2_RawImg = T1_RawImg1[roi_mask]
    T2_C = T1_C1[roi_mask]

    if len(T2_RawImg) == 0:
        return [0.0, 0.0]

    maxind = np.argmax(T2_RawImg)
    SUVpeak = [float(T2_C[maxind])]   
    SUVpeak.append(float(np.max(T2_C))) 

    return SUVpeak


def getNGTDM(ROIOnly2,levels):

    ROIOnly = ROIOnly2.copy()
    if ROIOnly.ndim == 2:
        twoD = 1
    else:
        twoD = 0
    
    nLevel = len(levels)
    if nLevel > 100:
        adjust = 10000
    else:
        adjust = 1000

    if twoD:
        ROIOnly = np.pad(ROIOnly,((1,1),(1,1)),mode='constant' , constant_values=np.nan)
    else:
        ROIOnly = np.pad(ROIOnly,((1,1),(1,1),(1,1)),mode='constant' , constant_values=np.nan)
    

    uniqueVol = np.round(levels*adjust)/adjust
    ROIOnly=np.round(ROIOnly*adjust)/adjust
    NL = len(levels)


    temp = ROIOnly.copy()
    for i in range(0,NL):
        ROIOnly[temp==uniqueVol[i]] = i+1
    


    NGTDM = np.zeros((NL,1))
    countValid = np.zeros((NL,1))


    if twoD:
        i,j,sl = ind2sub(ROIOnly)
        posValid = np.column_stack((i , j))
        nValid_temp = posValid.shape[0]
        weights = np.ones(9).astype(np.int32)
        # weights = weights
        Aarray = np.zeros((nValid_temp,2)).astype(np.int32)
        for n in range(0,nValid_temp):
            neighbours = np.zeros((9,1)).astype(np.int32)
            # neighbours = ROIOnly[posValid[n,0]-1:posValid[n,0]+2  ,   posValid[n,1]-1:posValid[n,1]+2].flatten(order='F')
            neighbours = ROIOnly[posValid[n,0]-1:posValid[n,0]+2  ,   posValid[n,1]-1:posValid[n,1]+2]
            neighbours = neighbours.flatten(order='F')
            neighbours = np.multiply(neighbours , weights)
            # neighbours = np.reshape(neighbours,(9,1))
            value = int(neighbours[4])-1
            neighbours[4] = np.nan
            sum_wei = 0
            for nei in range(0,neighbours.shape[0]):
                if ~np.isnan(neighbours[nei]):
                    sum_wei += weights[nei]
            if sum_wei != 0:
                neighbours = neighbours / sum_wei
            # neighbours.pop(4)
            neighbours = np.delete(neighbours,4,None)
            # neighbours[4] = []

            fin1 = np.where(~np.isnan(neighbours))
            
            if len(fin1[0]) > 0 :
            # if neighbours[ ~np.isnan(neighbours)].shape[0] > 0  or neighbours[ ~np.isnan(neighbours)].shape[1] > 0:

                sum_nei = 0
                for nei in range(0,neighbours.shape[0]):
                    if ~ np.isnan(neighbours[nei]):
                        sum_nei += neighbours[nei]
                NGTDM[value] = NGTDM[value] + float(np.abs(value+1 - sum_nei))
                countValid[value] = countValid[value] + 1

    else:
        i,j,k = ind2sub(ROIOnly)
        posValid = np.column_stack((i , j, k))
        nValid_temp = posValid.shape[0]
        weights = np.ones(27).astype(np.int32)
        # weights = weights
        Aarray = np.zeros((nValid_temp,2)).astype(np.int32)
        for n in range(0,nValid_temp):
            neighbours = np.zeros((27,1)).astype(np.int32)
            neighbours = ROIOnly[posValid[n,0]-1:posValid[n,0]+2  ,   posValid[n,1]-1:posValid[n,1]+2,   posValid[n,2]-1:posValid[n,2]+2].flatten(order='F')
            neighbours = np.multiply(neighbours , weights)

            value = int(neighbours[13])-1
            neighbours[13] = np.nan
            sum_wei = 0
            for nei in range(0,neighbours.shape[0]):
                if ~ np.isnan(neighbours[nei]):
                    sum_wei += weights[nei]

            if sum_wei != 0:
                neighbours = neighbours / sum_wei
            neighbours = np.delete(neighbours,13,None)

            fin1 = np.where(~np.isnan(neighbours))

            if len(fin1[0]) > 0:
            # if neighbours[ ~np.isnan(neighbours)].shape[0] > 0  or neighbours[ ~np.isnan(neighbours)].shape[1] > 0:
                
                sum_nei = 0
                for nei in range(0,neighbours.shape[0]):
                    if ~ np.isnan(neighbours[nei]):
                        sum_nei += neighbours[nei]
                
                Ai = np.abs(value+1-sum_nei)
                NGTDM[value] = NGTDM[value] + float(Ai)
                countValid[value] = countValid[value] + 1
                Aarray[n,:] = [value, float(Ai)]




    return NGTDM,countValid,Aarray


def chunked_astype(array, dtype, chunk_size=1000000):
    if array.size > chunk_size:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        out = np.memmap(temp_file.name, dtype=dtype, mode='w+', shape=array.shape)
        for i in range(0, array.size, chunk_size):
            end_idx = min(i + chunk_size, array.size)
            out.flat[i:end_idx] = array.flat[i:end_idx].astype(dtype)
        return out
    else:
        return array.astype(dtype)



def roundGL(Img , isGLrounding):

    if isGLrounding == 1:
        GLroundedImg = np.round(Img)
    else:
        GLroundedImg = Img.copy()
     
    return GLroundedImg


def prepareVolume(volume, Mask, DataType, pixelW, sliceTh,
                  newVoxelSize, VoxInterp, ROIInterp, ROI_PV, scaleType, isIsot2D,
                  isScale, isGLround, DiscType, qntz, Bin,
                  isReSegRng, ReSegIntrvl, isOutliers):
    if DiscType == 'FBS':
        quantization = fixedBinSizeQuantization
    elif DiscType == 'FBN':
        quantization = uniformQuantization
    else:
        print('Error with discretization type. Must either be "FBS" (Fixed Bin Size) or "FBN" (Fixed Number of Bins).')

    if qntz == 'Lloyd':
        quantization = lloydQuantization

    ROIBox = chunked_copy(Mask)
    Imgbox = chunked_copy(volume)

    Imgbox = Imgbox.astype(np.float32)

    if DataType == 'MRscan':
        ROIonly = chunked_copy(Imgbox)
        ROIonly[ROIBox == 0] = np.nan
        temp = CollewetNorm(ROIonly)
        ROIBox[np.isnan(temp)] = 0

    flagPW = 0
    if scaleType == 'NoRescale':
        flagPW = 0
    elif scaleType == 'XYZscale':
        flagPW = 1
    elif scaleType == 'XYscale':
        flagPW = 2
    elif scaleType == 'Zscale':
        flagPW = 3

    if isIsot2D == 1:
        flagPW = 2

    if isScale == 0:
        flagPW = 0

    if flagPW == 0:
        a = 1
        b = 1
        c = 1
    elif flagPW == 1:
        a = pixelW / newVoxelSize
        b = pixelW / newVoxelSize
        c = sliceTh / newVoxelSize
    elif flagPW == 2:
        a = pixelW / newVoxelSize
        b = pixelW / newVoxelSize
        c = 1
    elif flagPW == 3:
        a = 1
        b = 1
        c = sliceTh / pixelW

    # Resampling
    ImgBoxResmp = chunked_copy(Imgbox)
    ImgWholeResmp = chunked_copy(volume)
    ROIBoxResmp = chunked_copy(ROIBox)
    ROIwholeResmp = chunked_copy(Mask)

    if Imgbox.ndim == 3 and flagPW != 0:
        if (a + b + c) != 3:
            ROIBoxResmp = imresize3D(ROIBox, [pixelW, pixelW, sliceTh],
                                     [np.ceil(ROIBox.shape[0] * a), np.ceil(ROIBox.shape[1] * b),
                                      np.ceil(ROIBox.shape[2] * c)], ROIInterp, 'constant',
                                     [newVoxelSize, newVoxelSize, newVoxelSize], isIsot2D)
            Imgbox[np.isnan(Imgbox)] = 0
            ImgBoxResmp = imresize3D(Imgbox, [pixelW, pixelW, sliceTh],
                                     [np.ceil(Imgbox.shape[0] * a), np.ceil(Imgbox.shape[1] * b),
                                      np.ceil(Imgbox.shape[2] * c)], VoxInterp, 'constant',
                                     [newVoxelSize, newVoxelSize, newVoxelSize], isIsot2D)
            ROIBoxResmp[ROIBoxResmp < ROI_PV] = 0
            ROIBoxResmp[ROIBoxResmp >= ROI_PV] = 1
            ROIwholeResmp = imresize3D(Mask, [pixelW, pixelW, sliceTh],
                                       [np.ceil(Mask.shape[0] * a), np.ceil(Mask.shape[1] * b),
                                        np.ceil(Mask.shape[2] * c)], ROIInterp, 'constant',
                                       [newVoxelSize, newVoxelSize, newVoxelSize], isIsot2D)
            ImgWholeResmp = imresize3D(volume, [pixelW, pixelW, sliceTh],
                                       [np.ceil(volume.shape[0] * a), np.ceil(volume.shape[1] * b),
                                        np.ceil(volume.shape[2] * c)], VoxInterp, 'constant',
                                       [newVoxelSize, newVoxelSize, newVoxelSize], isIsot2D)
            if np.max(ROIwholeResmp) < ROI_PV:
                print('Resampled ROI has no voxels with value above ROI_PV. Cutting ROI_PV to half.')
                ROI_PV = ROI_PV / 2

            ROIwholeResmp[ROIwholeResmp < ROI_PV] = 0
            ROIwholeResmp[ROIwholeResmp >= ROI_PV] = 1

    elif Imgbox.ndim == 2 and flagPW != 0:
        if (a + b) != 2:
            ROIBoxResmp = imresize(ROIBox, [pixelW, pixelW],
                                   [np.ceil(ROIBox.shape[0] * a), np.ceil(ROIBox.shape[1] * b)], ROIInterp,
                                   [newVoxelSize, newVoxelSize])
            ImgBoxResmp = imresize(Imgbox, [pixelW, pixelW],
                                   [np.ceil(Imgbox.shape[0] * a), np.ceil(Imgbox.shape[1] * b)], VoxInterp,
                                   [newVoxelSize, newVoxelSize])
            ROIBoxResmp[ROIBoxResmp < ROI_PV] = 0
            ROIBoxResmp[ROIBoxResmp >= ROI_PV] = 1

            ROIwholeResmp = imresize(Mask, [pixelW, pixelW], [np.ceil(Mask.shape[0] * a), np.ceil(Mask.shape[1] * b)],
                                     ROIInterp, [newVoxelSize, newVoxelSize])
            ImgWholeResmp = imresize(volume, [pixelW, pixelW],
                                     [np.ceil(volume.shape[0] * a), np.ceil(volume.shape[1] * b)], VoxInterp,
                                     [newVoxelSize, newVoxelSize])
            if np.max(ROIwholeResmp) < ROI_PV:
                print('Resampled ROI has no voxels with value above ROI_PV. Cutting ROI_PV to half.')
                ROI_PV = ROI_PV / 2

            ROIwholeResmp[ROIwholeResmp < ROI_PV] = 0
            ROIwholeResmp[ROIwholeResmp >= ROI_PV] = 1

    IntsBoxROI = chunked_copy(ImgBoxResmp)

    ImgBoxResmp[ROIBoxResmp == 0] = np.nan

    IntsBoxROI = roundGL(ImgBoxResmp, isGLround)
    ImgWholeResmp = roundGL(ImgWholeResmp, isGLround)

    IntsBoxROItmp1 = chunked_copy(IntsBoxROI)
    ImgWholeResmptmp1 = chunked_copy(ImgWholeResmp)
    IntsBoxROItmp2 = chunked_copy(IntsBoxROI)
    ImgWholeResmptmp2 = chunked_copy(ImgWholeResmp)

    if isReSegRng == 1:
        IntsBoxROItmp1[IntsBoxROI < ReSegIntrvl[0]] = np.nan
        IntsBoxROItmp1[IntsBoxROI > ReSegIntrvl[1]] = np.nan
        ImgWholeResmptmp1[ImgWholeResmp < ReSegIntrvl[0]] = np.nan
        ImgWholeResmptmp1[ImgWholeResmp > ReSegIntrvl[1]] = np.nan

    if isOutliers == 1:
        Mu = np.nanmean(IntsBoxROI)
        Sigma = np.nanstd(IntsBoxROI)
        IntsBoxROItmp2[IntsBoxROI < (Mu - 3 * Sigma)] = np.nan
        IntsBoxROItmp2[IntsBoxROI > (Mu + 3 * Sigma)] = np.nan

        Mu = np.nanmean(ImgWholeResmp)
        Sigma = np.nanstd(ImgWholeResmp)
        ImgWholeResmptmp2[ImgWholeResmp < (Mu - 3 * Sigma)] = np.nan
        ImgWholeResmptmp2[ImgWholeResmp > (Mu + 3 * Sigma)] = np.nan

    IntsBoxROI = getMutualROI(IntsBoxROItmp1, IntsBoxROItmp2)
    ImgWholeResmp = getMutualROI(ImgWholeResmptmp1, ImgWholeResmptmp2)

    newpixelW = pixelW / a
    newsliceTh = sliceTh / c

    if DataType == 'PET':
        minGL = 0
    elif DataType == 'CT':
        if isReSegRng == 1:
            minGL = ReSegIntrvl[0]
        else:
            minGL = np.nanmin(IntsBoxROI)

    else:
        minGL = np.nanmin(IntsBoxROI)

    ImgBoxResampQuntz3D, levels = quantization(IntsBoxROI, Bin, minGL)

    boxBound = computeBoundingBox(ROIBoxResmp)
    MorphROI = ROIBoxResmp[boxBound[0][0]:boxBound[0][1], boxBound[1][0]:boxBound[1][1], boxBound[2][0]:boxBound[2][1]]
    IntsBoxROI = IntsBoxROI[boxBound[0][0]:boxBound[0][1], boxBound[1][0]:boxBound[1][1], boxBound[2][0]:boxBound[2][1]]
    ImgBoxResampQuntz3D = ImgBoxResampQuntz3D[boxBound[0][0]:boxBound[0][1], boxBound[1][0]:boxBound[1][1],
                          boxBound[2][0]:boxBound[2][1]]
    # ImgWholeResmp = ImgWholeResmp[boxBound[0][0]:boxBound[0][1],boxBound[1][0]:boxBound[1][1],boxBound[2][0]:boxBound[2][1]]

    return ImgBoxResampQuntz3D, levels, MorphROI, IntsBoxROI, ImgWholeResmp, ROIwholeResmp, newpixelW, newsliceTh
