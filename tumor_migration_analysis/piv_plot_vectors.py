#!/opt/local/bin/python

"""

Plots PIV vector data (just vectors, no fame)

"""

import os

import numpy as np
from skimage.io._plugins import tifffile_plugin as tifffile
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import utility_functions as uf

def plot_vectors(stk_path,px_size=1,scale_factor=0.004,scale_length=0.1,vector_width=1.5):
    """Plots PIV vectors overlaid onto the orginal image stack
    The images are automatically written to a new directory.

    Parameters
    ----------
    stk_path : string
        the path to the image stack to be analyzed
    px_size : float
        the pixel size in um/px
    scale_factor : float
        a scaling to determine the vector size
    scale_length : int
        the length of the scale vector (in scaled units, usually um/min)
    vector_width : float
        the width of the PIV vectors for the quiver plots

    """

    # opens the image stack to get the aspect ratio
    stk = tifffile.imread(stk_path)
    width = stk[0].shape[1] * px_size
    height = stk[0].shape[0] * px_size
    ar = height / width

    # finds the PIV vector data
    data_dir = os.path.splitext(stk_path)[0] + "_piv_data"
    if not os.path.isdir(data_dir):
        raise FileNotFoundError("No PIV vector data found. Please run extraction script first.")

    # get unique basename list (from x coordinate data)
    basename_list = [_[:-6] for _ in os.listdir(data_dir) if '_x.dat' in _]
    basename_list = uf.natural_sort(basename_list)

    for tag in ["", "_interp"]: #plots both the raw and interpolated data

        print("Tag = ", tag)

        #makes new directory for plotting
        plot_dir = os.path.join(data_dir,"vectors_w%sum-min_scale"%(str(scale_length).replace(".","p")) + tag)
        uf.make_dir(plot_dir)

        #goes through each time frame
        for i, basename in enumerate(basename_list):

            #sets up the plot and plots the image data underneath
            rcParams['axes.linewidth'] = 0
            fig, ax = plt.subplots(figsize=(6,6*ar))
            # ax.patch.set_facecolor('black')
            ax.imshow(stk[i],cmap='Greys_r',extent=(0,width,height,0))

            #plots each frame
            print('Grabbing frame:', basename)
            x = np.array(uf.read_file(os.path.join(data_dir, basename + "_x.dat")),dtype=float)
            y = np.array(uf.read_file(os.path.join(data_dir, basename + "_y.dat")),dtype=float)
            if "_t0" in basename:
                U = np.array(uf.read_file(os.path.join(data_dir, basename + "_u.dat")),dtype=float)
                V = np.array(uf.read_file(os.path.join(data_dir,  basename + "_v.dat")),dtype=float)
            else:
                U = np.array(uf.read_file(os.path.join(data_dir, basename + "_u%s.dat" % tag)),dtype=float)
                V = np.array(uf.read_file(os.path.join(data_dir, basename + "_v%s.dat" % tag)),dtype=float)

            #plots vectors with color code according to angle
            phis = np.arctan2(V, U) * 180. / np.pi
            plt.quiver(x, y, U, V, phis.ravel(), cmap='rainbow', clim=(-180., 180.),
                       units='xy',scale=scale_factor,scale_units='x',width=vector_width)

            #makes an arrow for scale
            cm = LinearSegmentedColormap.from_list('cm', [(1,1,1),(1,1,1)])
            plt.quiver([width-30],[height-height*0.05], [scale_length],[0],[0],cmap=cm,
                       units='xy', scale=scale_factor, scale_units='x',width=vector_width)

            #finishes plot
            ax.set_xlim(0,width)
            ax.set_ylim(0,height)
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            fig.subplots_adjust(bottom=0,left=0,top=1,right=1)

            #saves plot as png
            plt.savefig(plot_dir + '/%s_vectors.png'%basename)
            plt.close()

def main():
    """Sets up the analysis for extracting PIV vectors.
    You should update the image path and pixel size here.
    You should not have to change anything in the rest of the script.

    """

    #sets some initial parameters
    stk_path = './sample_data/tumor_nuclei_small/tumor_nuclei_small.tif'
    px_size = 0.91 #um/px
    scale_factor = 0.004 #for scaling the PIV vectors on the plots
    scale_length = 0.1 #sets the length of the scale vector on the plots (in um/min)

    plot_vectors(stk_path,px_size=px_size,scale_factor=scale_factor,scale_length=scale_length)

if __name__=="__main__":
    main()