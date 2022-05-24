#!/opt/local/bin/python

"""

This script reads a stack of labeled nuclei positions and performs a finite Voronoi tesselation.
A stack with the labeled cell areas is generated.

"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from skimage.io._plugins import tifffile_plugin as tifffile
from scipy.ndimage import binary_erosion, label

import morphometric_analysis as morpho
import utility_functions as uf

def estimate_borders_voronoi(frame,labels,mask):
    """Estimates cell borders using a clipped voronoi tesselation (from a labeled image)

    Parameters
    ----------
    frame : 2D numpy array
        a 2D image with the cells
    labels : 2D numpy array
        an image (usually 16bit) with the cell/object labels
    mask : 2D numpy array
        an image for clipping the tesselated cells

    Returns
    -------
    vor_labels : 2D numpy array
        a 16-bit  labeled image with the voronoi tesselated cells
    fig : matplotlib figure
        a figure containing an axis with the original frame and the tesselation overlaid
    """

    # gets a circular structuring element for binary morphology
    se = morpho.get_circular_se(radius=1)

    # gets the image dimensions and makes a new binary image
    width = frame.shape[1]
    height = frame.shape[0]
    ar = width/height
    bin = np.zeros_like(frame,dtype='uint8')

    # does the tesselation
    centroids, vertices = morpho.tesselate_voronoi(labels,mask=mask)

    # plots the resulting tesselation overlaid on the original stack
    fig, ax = plt.subplots(figsize=(6,6*ar))
    ax.imshow(frame, cmap='Greys_r', extent=(0, frame.shape[1], frame.shape[0], 0))
    for poly in vertices:
        ax.fill(*zip(*poly), alpha=0.5)
        ax.plot(*zip(*poly), ls='-', color='k', linewidth=0.5)
    ax.plot(centroids[:, 0], centroids[:, 1], 'ko', ms=3)
    ax.set_xlim(0,width)
    ax.set_ylim(height,0) #because images start with the origin top-left
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(bottom=0, left=0, top=1, right=1)

    #makes a coordinate grid for testing where the regions are located
    x, y = np.meshgrid(np.arange(frame.shape[1]), np.arange(frame.shape[0]))  # make a canvas with coordinates
    x, y = x.flatten(), y.flatten()
    coords = np.vstack((x, y)).T

    #makes a new border image from tesselation
    for verts in vertices:
        p = Path(verts)
        grid = p.contains_points(coords)
        vert_mask = grid.reshape(frame.shape[1], frame.shape[0])
        vert_mask = binary_erosion(vert_mask, structure=se, border_value=0)
        bin += vert_mask.astype('uint8')

    #labels the image
    bin = (bin > 0).astype('uint16')
    #makes a white border around the edges to prevent counting the partial cells
    #figure out why this needs to be extended to -2
    bin[:,0:2] = 1
    bin[:,-2] = 1
    bin[:,-1] = 1
    bin[0:2,:] = 1
    bin[-2,:] = 1
    bin[-1,:] = 1

    # plt.imshow(bin)
    # plt.show()

    vor_labels, no_labels = label(bin)

    #make the border cells 0 and subtract 1 from the rest
    vor_labels[vor_labels==1] == 0
    vor_labels[vor_labels!=0] -= 1

    return vor_labels, fig

def estimate_borders_stk(stk_path,label_path,mask_path):
    """Reads in and does the voronoi tesselation frame-by-frame for a stack

       Parameters
       ----------
       stk_path : string
           path to a 2D image (or 2D timeseries) with the cells
       label_path : string
           path to a 2D image (or 2D timeseries) with the labels
           (usually-bit, must match stk dimensions)
       mask : string
           path to a 2D binary images with the same xy dimensions as the stk/label images

       Returns
       -------
       None

       """

    #opens the label and mask images
    stk = tifffile.imread(stk_path)
    label_stk = tifffile.imread(label_path)
    mask = (tifffile.imread(mask_path) > 0).astype('uint8')

    # checks that the input stack and mask have same dimensions
    if len(stk.shape) == 2:
        if stk.shape != mask.shape:
            raise ValueError('Mask must have same dimensions as stack image')

    elif len(stk.shape) == 3:
        if stk.shape[1:] != mask.shape:
            raise ValueError('Mask must have same dimensions as stack images')

    else:
        raise TypeError("Image stack must be 2D or 3D!")

    #makes a directory for plotting
    plot_dir = os.path.join(os.path.splitext(stk_path)[0] + "_voronoi")
    uf.make_dir(plot_dir)
    basename = os.path.basename(plot_dir)

    #gets the clipped voronoi tesselation for each frame
    if len(stk.shape)==2: #if it's just one frame
        vor_labels, fig = estimate_borders_voronoi(stk, label_stk, mask)
        tifffile.imsave(os.path.join(plot_dir, basename + "_voronoi_labels.tif"),vor_labels)
        fig.savefig(os.path.join(plot_dir, basename + "_voronoi_labels.png"))
        plt.close()

    elif len(stk.shape)==3: #if it's a time-series stack
        vor_labels_stk = np.zeros_like(label_stk,dtype='uint16')
        for i in range(len(stk)):
            vor_labels_stk[i], fig = estimate_borders_voronoi(stk[i], label_stk[i], mask)
            fig.savefig(os.path.join(plot_dir, basename + "_voronoi_labels_frame_%i.png"%(i+1)))
            plt.close()
        tifffile.imsave(os.path.join(plot_dir, basename + "_voronoi_labels.tif"),vor_labels_stk)

    else:
        raise TypeError("Image stack must be 2D or 3D!")

def main():
    """Sets up the parameters for doing the tesselation.
    You should update the label_path and mask_path here.
    You should not have to change anything in the rest of the script.
    **Note that for your mask image, only the largest contiguous mask region can be used
      (it is ok to have holes in your mask, but only the biggest connected region will be considered)

    """

    #sets some initial parameters
    stk_path = './sample_data/tumor_nuclei_small/tumor_nuclei_small.tif'
    label_path = './sample_data/tumor_nuclei_small/tumor_nuclei_small_stardist.tif'
    mask_path = './sample_data/tumor_nuclei_small/tumor_nuclei_small_mask.tif'

    estimate_borders_stk(stk_path, label_path, mask_path)

if __name__ == "__main__":
    main()