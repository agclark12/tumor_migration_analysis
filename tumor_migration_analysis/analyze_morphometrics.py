#!/opt/local/bin/python

"""

This script analyzes morphometric data from a labeled cell image and extracts
cell sizes, aspect ratios, orientations and shape factors.

"""

import os
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import colors
from matplotlib import cm
from skimage.io._plugins import tifffile_plugin as tifffile
from skimage import color, morphology, draw, transform, measure

import utility_functions as uf
import morphometric_analysis as morpho

def analyze_frame(labels,px_size,plot_dir):

    """gets and plots finding the aspect ratio and shape factor and makes some plots

    TODO: extract size
    """

    #calculate the principal axis, aspect ratio and shape factor for each region
    print("Calculating morphometrics for all labels")
    no_labels = np.max(labels)

    centroid_list = morpho.get_centroids(labels)
    angle_list, ar_list = morpho.get_orientation_ar(labels)
    sf_list = morpho.get_sf(labels)

    # makes some plots
    print("Plotting")

    #makes a plot of the labeled image
    fig,ax = plt.subplots(figsize=(6,6))
    np.random.seed(19680801)
    cmap = colors.ListedColormap(np.random.rand(256, 3))
    cmap.set_under('k')
    ax.imshow(labels,cmap=cmap,vmin=0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(bottom=0, left=0, top=1, right=1)
    plt.savefig(os.path.join(plot_dir, 'labels.pdf'))
    plt.close()

    #makes a plot of the cells with principal axes and ar overlaid
    bin = (labels > 0).astype('uint8')
    fig,ax = plt.subplots(figsize=(6,6))
    ax.imshow(bin,cmap='gray_r')
    scale_factor = 0.25
    ax.quiver(centroid_list[:,0], centroid_list[:,1], ar_list, ar_list, units='xy', scale=scale_factor,
              scale_units='x', width=2, headlength=0, headaxislength=0, angles=angle_list * 180. / np.pi, color='r', pivot='middle')
    ax.plot(centroid_list[:,0], centroid_list[:,1],'go', ms=5)
    # for i in range(2,no_labels):
    #     ax.annotate("%i"%i,xy=centroid_list[i-1]+2,color='r',size=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(bottom=0, left=0, top=1, right=1)
    plt.savefig(os.path.join(plot_dir, 'aspect_ratio_quiver.pdf'))
    plt.close()

    #makes a plot of the aspect ratio scaled by color
    ar_img = deepcopy(labels)
    ar_img = ar_img.astype('float')
    ar_img[np.where(labels==1)] = 0

    for i in range(1,no_labels+1): #label 1 is the border cells
        points = np.where(labels == i)
        ar_img[points] = ar_list[i-1] #the ar list is from zero, labels are from 1

    fig,ax = plt.subplots(figsize=(6,6))
    cmap = plt.get_cmap('plasma')
    cmap.set_under('k')
    v_min = 1
    v_max = 2.6
    pos = ax.imshow(ar_img, cmap=cmap, vmin=v_min, vmax=v_max)
    cbar = fig.colorbar(pos, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(v_min,v_max+0.00000001,0.2))
    cbar.set_label('Aspect Ratio', rotation=90)
    ax.set_xticks([])
    ax.set_yticks([])
    # fig.subplots_adjust(bottom=0, left=0, top=1)
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(plot_dir, 'aspect_ratio.pdf'))
    plt.close()

    #makes a plot of the shape factor scaled by color
    sf_img = deepcopy(labels)
    sf_img = sf_img.astype('float')
    sf_img[np.where(sf_img==1)] = 0

    for i in range(1,no_labels+1): #label 1 is the border cells
        points = np.where(labels == i)
        sf_img[points] = sf_list[i-1]  #the ar list is from zero, labels are from 1

    fig,ax = plt.subplots(figsize=(6,6))
    cmap = plt.get_cmap('plasma')
    cmap.set_under('k')
    v_min = 3.5
    v_max = 4.6
    pos = ax.imshow(sf_img, cmap=cmap, vmin=v_min, vmax=v_max)
    cbar = fig.colorbar(pos, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(v_min,v_max+0.0000001,0.1))
    cbar.set_label('Shape Factor', rotation=90)
    ax.set_xticks([])
    ax.set_yticks([])
    # fig.subplots_adjust(bottom=0, left=0, top=1)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'shape_factor.pdf'))
    plt.close()

    ########outputs the data#########
    ########still need to get size data#######

def analyze_stk(img_path, px_size):

    #opens file and converts to grayscale
    print("Opening Image")
    stk = tifffile.imread(img_path)

    #makes a new directory for storing all of the data
    data_dir = os.path.join(os.path.splitext(img_path)[0] + "_morphometrics")
    uf.make_dir(data_dir)

    #analyzes each frame individually
    for i in range(len(stk)):

        print("Working on Frame %i/%i"%(i+1,len(stk)))
        # makes a new image directory for storing the frame data and plots
        frame_dir = os.path.join(data_dir,"frame_%i"%(i+1))
        uf.make_dir(frame_dir)

        #does the frame analysis
        analyze_frame(stk[i],px_size,frame_dir)

def main():
    """Sets up the morphometric analysis
    You should update the image path, and pixel size here.
    You should not have to change anything in the rest of the script.

    """

    #sets some initial parameters
    label_file_path = './sample_data_lab_meeting/monolayer_labels.tif'
    px_size = 0.275 #um/px

    analyze_stk(label_file_path, px_size)

if __name__ == "__main__":
    main()