#!/opt/local/bin/python

"""

This script extracts PIV vectors from a timelapse image stack using openPIV.
The images are automatically masked to incude only regions containing cells.

"""

import os
from copy import deepcopy

import numpy as np
from skimage.io._plugins import tifffile_plugin as tifffile
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes, binary_dilation
from astropy.convolution import convolve
import matplotlib.pyplot as plt

import openpiv.tools
import openpiv.pyprocess
import openpiv.scaling
import openpiv.validation
import openpiv.filters

import utility_functions as uf

def get_piv_vectors(frame_a,frame_b,dt,window_size=32,threshold=1.,scaling_factor=1,mask_a=None,noise_thresh=None,filter_bad_neighbors=False):
    """Calculates the PIV vectors from two consecutive frames using openPIV

    Parameters
    ----------
    frame_a : 2D numpy ndarray (image)
        the reference frame
    frame_b : 2D numpy ndarray (image)
        the frame to interrogate
    dt : float
        the time step between the frames
    window_size : int
        the sidelength of the interrogation window in pixels
    threshold : float
        the noise threshold for keeping the vectors
    scaling_factor : float
        the factor to scale the vector lengths (usually the inverse of pixel size)
    mask_a : 2D numpy ndarray (image)
        the image for masking the vectors
    noise_thresh : float
        a threshold for the maximum vector size (if left None, no magnitude thresholding is performed)
    filter_bad_neighbors : bool
        option to filter vectors that point in a very different direction compared to their neighbors

    Returns
    -------
    vector_data : dictionary
        a dictionary containing the x,y coordinate data and vectors (raw and interpolated)

    """

    #opens and converts the images to be read
    frame_a = frame_a.astype(np.int32)
    frame_b = frame_b.astype(np.int32)

    #normalizes intensities between frames for piv analysis
    frame_a = openpiv.pyprocess.normalize_intensity(frame_a)
    frame_b = openpiv.pyprocess.normalize_intensity(frame_b)
    frame_a = frame_a.astype(np.int32)
    frame_b = frame_b.astype(np.int32)

    #gets the piv vectors and positions, filters noise, and scales (using openpiv)
    window_size = int(window_size)
    overlap = int(window_size/2)
    search_area = window_size
    u, v, sig2noise = openpiv.pyprocess.extended_search_area_piv(frame_a, frame_b, window_size=window_size, overlap=overlap, dt=dt,
                                                                 search_area_size=search_area, sig2noise_method='peak2peak')
    x, y = openpiv.pyprocess.get_coordinates(frame_a.shape, search_area, overlap)
    u, v, mask = openpiv.validation.sig2noise_val(u, v, sig2noise, threshold=threshold)
    x_scaled, y_scaled, u, v = openpiv.scaling.uniform(x, y, u, v, scaling_factor=scaling_factor)

    #filters out vectors that are obviously too large (this is noise)
    if not noise_thresh is None:
        u, v, mask = openpiv.validation.global_val(u, v, u_thresholds=(-noise_thresh,noise_thresh), v_thresholds=(-noise_thresh,noise_thresh))

    if filter_bad_neighbors:
        #filters out abarrent vectors (pointing in a vastly different direction compared to neighbors)
        phis = -np.arctan2(v, u)
        lengths = np.linalg.norm((u,v),axis=0)
        unit_u = u / lengths
        unit_v = v / lengths

        # mean_u = ndi.uniform_filter(unit_u,size=3,mode='mirror')
        # mean_v = ndi.uniform_filter(unit_v,size=3,mode='mirror')
        kernel = [[1,1,1],
                  [1,0,1],
                  [1,1,1]]

        neighbor_u = convolve(unit_u,kernel,boundary='extend',nan_treatment='interpolate',preserve_nan=True)
        neighbor_v = convolve(unit_v,kernel,boundary='extend',nan_treatment='interpolate',preserve_nan=True)

        neighbor_mean = -np.arctan2(neighbor_v, neighbor_u)
        angular_diff = np.arctan2(np.sin(neighbor_mean - phis), np.cos(neighbor_mean - phis))
        # print(angular_diff)
        # stop
        thresh = 0.2
        mask_phi = np.abs(angular_diff) > thresh

        u[mask_phi] = np.nan
        v[mask_phi] = np.nan

        # fig,(ax1,ax2) = plt.subplots(ncols=2)
        # ax1.imshow(np.flipud(frame_a),origin='lower')
        # ax1.quiver(x, y, u, v, color='w',pivot='mid',scale=None)

        #filters out abarrent vectors (that have a very different magnitude compared to neighbors
        neighbor_lengths = convolve(lengths,kernel,boundary='extend',nan_treatment='interpolate',preserve_nan=True)
        length_diff = (neighbor_lengths - lengths) / neighbor_lengths
        # print(length_diff)
        thresh = 5
        mask_length = length_diff**2 > thresh

        u[mask_length] = np.nan
        v[mask_length] = np.nan

        #removes vectors that are not within the mask region
        u_interp = deepcopy(u)
        v_interp = deepcopy(v)

        if mask_a is None:
            mask_a = np.ones_like(frame_a)

        for i in range(len(x)):
            for j in range(len(x[i])):
                idx_y = int(round(x[i][j])) #note x and y are swapped
                idx_x = int(round(y[i][j])) #note x and y are swapped
                if not np.flipud(mask_a)[idx_x][idx_y]:
                    u[i][j] = np.nan
                    v[i][j] = np.nan

    #makes a mask if a specific mask is not given
    if mask_a is None:
        mask_a = np.ones_like(frame_a)

    for i in range(len(x)):
        for j in range(len(x[i])):
            idx_y = int(round(x[i][j])) #note x and y are swapped
            idx_x = int(round(y[i][j])) #note x and y are swapped
            if not mask_a[idx_x][idx_y]:
                u[i][j] = np.nan
                v[i][j] = np.nan

    # performs interpolation
    u_interp = deepcopy(u)
    v_interp = deepcopy(v)
    u_interp, v_interp = openpiv.filters.replace_outliers(u_interp, v_interp, method='localmean', max_iter=10, kernel_size=2)

    #removes interpolated vectors that are not within the mask region
    u_interp_masked = deepcopy(u_interp)
    v_interp_masked = deepcopy(v_interp)

    for i in range(len(x)):
        for j in range(len(x[i])):
            idx_y = int(round(x[i][j]))  # note x and y are swapped
            idx_x = int(round(y[i][j]))  # note x and y are swapped
            if not mask_a[idx_x][idx_y]:
                u_interp_masked[i][j] = np.nan
                v_interp_masked[i][j] = np.nan

    # plots to check
    # plt.imshow(np.flipud(frame_a),origin='lower')
    # plt.quiver(x, y, u, v, color='w',pivot='mid',scale=None)
    # plt.show()
    # plt.imshow(np.flipud(frame_a),origin='lower')
    # plt.quiver(x, y, u_interp_masked, v_interp_masked, color='w',pivot='mid',scale=None)
    # plt.show()

    #swap the v components so that that positive vectors point up with origin top left
    v = -v
    v_interp = -v_interp
    v_interp_masked = -v_interp_masked

    vector_data = {"x":x_scaled,"y":y_scaled,
                   "u":u,"v":v,
                   "u_interp":u_interp_masked,"v_interp":v_interp_masked}

    return vector_data

def make_mask(stk):
    """Makes a mask from an image stack by taking a brightes point projection and thresholding.

    Parameters
    ----------
    stk : 3D ndarray
        the array you want to use to make the mask

    Returns
    ----------
    bpp : 2D ndarray
        the binary mask array
    """

    bpp = np.max(stk,axis=0)
    bpp = gaussian(bpp, sigma=2)
    thresh = threshold_otsu(bpp)
    bpp = (bpp > thresh).astype('uint8')
    bpp = remove_small_objects(bpp, min_size=10)
    bpp = (binary_dilation(bpp, iterations=2)).astype('uint8')
    bpp = (binary_fill_holes(bpp)).astype('uint8')

    # #plots to test
    # plt.imshow(stk[0])
    # fig, ax = plt.subplots()
    # plt.imshow(bpp)
    # plt.show()

    return bpp

def extract_vectors(stk_path,time_int=1,px_size=1,mask=None,window_length=10,noise_thresh=0.1):
    """Extracts the x and y component vectors from images.
    Extracted data is automatically written to a new directory.

    Parameters
    ----------
    stk_path : string
        the path to the image stack to be analyzed
    time_int : float
        the time interval in min
    px_size : float
        the pixel size in um/px
    mask : 2D numpy array (or string)
        if mask is set to a 2D numpy array, it will be used to mask the PIV data
        if mask is set to 'auto', a mask will automatically be generated from the image stack
    window_length : int
        the side length of the interrogation window (in um)
    noise_thresh : float
        the noise threshold within the range (0,1]

    """

    #makes a new directory to store the data
    save_dir = os.path.splitext(stk_path)[0] + "_piv_data"
    uf.make_dir(save_dir)
    basename = os.path.basename(os.path.splitext(stk_path)[0])
    print(basename)

    #opens the image stack
    stk = tifffile.imread(stk_path)

    # checks that the input stack and mask have correct dimensions
    # and creates the mask if necessary
    if len(stk.shape) != 3:
        raise ValueError('Input stack must be 3D (2D images over time)')
    if isinstance(mask, np.ndarray):
        if stk.shape[:2] != mask.shape:
            raise ValueError('Mask must have same dimensions as stack images')
        mask = (mask > 0).astype('uint8')
    elif mask is None:
        mask = np.ones_like(stk[0])
    elif mask == 'auto':
        mask = make_mask(stk)
    else:
        raise ValueError("<<mask>> must be a 2D ndarray or 'auto'")

    #save the mask file for record-keeping
    mask_path = os.path.join(save_dir, basename + "_mask.tif")
    tifffile.imsave(mask_path, mask * 255)

    #makes empty data files for first frame
    uf.save_data_array([], os.path.join(save_dir,basename+"_t0_x.dat"))
    uf.save_data_array([], os.path.join(save_dir,basename+"_t0_y.dat"))
    uf.save_data_array([], os.path.join(save_dir,basename+"_t0_u.dat"))
    uf.save_data_array([], os.path.join(save_dir,basename+"_t0_v.dat"))

    #goes through each frame pair and gets PIV data
    for j in range(stk.shape[0]-1):
        print("Analyzing frame %i/%i"%(j+1,stk.shape[0]-1))

        frame_a = stk[j]
        frame_b = stk[j+1]

        #does the PIV analysis
        window_size = int(np.round(window_length/px_size)) #pixel equivalent
        if window_size%2!=0:
            window_size -= 1
        vector_data = get_piv_vectors(frame_a,frame_b,time_int,scaling_factor=1./px_size,mask_a=mask,
                                      window_size=window_size,noise_thresh=noise_thresh,filter_bad_neighbors=True)

        #saves data (U,V in um/min)
        uf.save_data_array(vector_data["x"],os.path.join(save_dir,basename+"_t%i_x.dat"%(j+1)))
        uf.save_data_array(vector_data["y"],os.path.join(save_dir,basename+"_t%i_y.dat"%(j+1)))
        uf.save_data_array(vector_data["u"],os.path.join(save_dir,basename+"_t%i_u.dat"%(j+1)))
        uf.save_data_array(vector_data["v"],os.path.join(save_dir,basename+"_t%i_v.dat"%(j+1)))
        uf.save_data_array(vector_data["u_interp"],os.path.join(save_dir,basename+"_t%i_u_interp.dat"%(j+1)))
        uf.save_data_array(vector_data["v_interp"],os.path.join(save_dir,basename+"_t%i_v_interp.dat"%(j+1)))

def main():
    """Sets up the analysis for extracting PIV vectors.
    You should update the image path, time interval, pixel size and interrogation window length here.
    You should not have to change anything in the rest of the script.

    """

    # #sets some initial parameters
    # stk_path = './sample_data/tumor_nuclei.tif'
    # time_int = 30 #min
    # px_size = 0.91 #um/px
    # window_len = 20 #interrogation window length in um

    stk_path = './sample_data_lab_meeting/monolayer_live.tif'
    time_int = 20 #min
    px_size = 0.275 #um/px
    window_len = 10 #interrogation window length in um

    #for automatic generation of masks during analysis, use the following:
    # extract_vectors(stk_path,time_int=time_int,px_size=px_size,window_length=window_len,mask='auto')

    #if you want to use a manually-generated mask, use the following:
    # mask_path = './sample_data/tumor_nuclei_mask.tif'
    # mask = tifffile.imread(stk_path)
    # extract_vectors(stk_path,time_int=time_int,px_size=px_size,window_length=window_len,mask=mask)

    #if you do not want to use a mask, use the following:
    extract_vectors(stk_path,time_int=time_int,px_size=px_size,window_length=window_len)


if __name__=="__main__":
    main()