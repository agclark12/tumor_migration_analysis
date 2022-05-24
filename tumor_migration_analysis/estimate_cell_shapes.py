#!/opt/local/bin/python

"""

This script reads a stack of labeled nuclei positions and performs a finite Voronoi tesselation.
A binary stack with the labeled cell borders is generated.

TODO:   MAKE A LABELED IMAGE, NOT JUST A BINARY (WITH A BORDER AROUND THE EDGES)
        MAKE THIS WORK FOR A 3D STACK

"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from skimage.io._plugins import tifffile_plugin as tifffile
from scipy.ndimage import binary_erosion, label
from shapely.geometry import Point, Polygon, MultiPoint

import morphometric_analysis as morpho
import utility_functions as uf

def estimate_borders_voronoi(stk_path,label_path,mask_path):

    #opens the label and mask images
    stk = tifffile.imread(stk_path)
    label_stk = tifffile.imread(label_path)
    mask = (tifffile.imread(mask_path) > 0).astype('uint8')

    #makes a directory for plotting
    plot_dir = os.path.join(os.path.splitext(stk_path)[0] + "_voronoi")
    uf.make_dir(plot_dir)
    basename = os.path.basename(plot_dir)

    #generates a new image for the labeled voronoi regions
    vor_labels_stk = np.zeros_like(label_stk)
    se = morpho.get_circular_se(radius=1)

    #does the Voronoi tesselation  this only takes the first frame.
    # need to restructure this to account for multiple frames.
    if len(stk.shape)==3: #checks if it's a time-series stack
        frame = stk[0]
        labels = label_stk[0]
    elif len(stk.shape)==2:
        frame = stk
        labels = label_stk
    else:
        raise TypeError("Image stack must be 2D or 3D!")

    # gets the image dimensions and makes a new binary image
    width = frame.shape[1]
    height = frame.shape[0]
    bin = np.zeros_like(frame,dtype='uint8')

    centroids, vertices = morpho.tesselate_voronoi(labels,mask=mask)

    # plots the result overlaid on the original stack and saves
    fig, ax = plt.subplots()
    ax.imshow(frame, cmap='Greys_r', extent=(0, frame.shape[1], frame.shape[0], 0))
    for poly in vertices:
        ax.fill(*zip(*poly), alpha=0.5)
        ax.plot(*zip(*poly), ls='-', color='k', linewidth=0.5)
    ax.plot(centroids[:, 0], centroids[:, 1], 'ko', ms=3)
    ax.set_xlim(0,width)
    ax.set_ylim(0,height)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(bottom=0, left=0, top=1, right=1)
    plt.savefig(os.path.join(plot_dir, basename + "_voronoi.png"))
    ####need to save this with a frame number (do this in a new directory)####
    plt.close()

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
    bin = (bin > 0).astype('uint8')
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



    tifffile.imsave(os.path.join(plot_dir, basename + "_voronoi_labels.tif"),vor_labels)
    ####need to save this as a whole stack####
    # plt.imshow(vor_labels[i])
    # plt.show()

    #remove the open edges to avoid analyzing the edge cells

    return vor_labels

def main():
    """Sets up the parameters for doing the tesselation.
    You should update the label_path and mask_path here.
    You should not have to change anything in the rest of the script.
    **Note that for your mask image, only the largest contiguous mask region can be used
      (it is ok to have holes in your mask, but only the biggest conected region will be considered)

    """

    #sets some initial parameters
    # stk_path = './sample_data/tumor_nuclei_small.tif'
    # label_path = './sample_data/tumor_nuclei_stardist_small.tif'
    # mask_path = './sample_data/tumor_nuclei_mask_small.tif'

    stk_path = './sample_data_lab_meeting/nuclei.tif'
    label_path = './sample_data_lab_meeting/nuclei_labeled.tif'
    mask_path = './sample_data_lab_meeting/nuclei_mask.tif'

    estimate_borders_voronoi(stk_path, label_path, mask_path)

if __name__ == "__main__":
    main()