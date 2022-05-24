#!/opt/local/bin/python

"""

This script reads TrackMate data from .xml files and analyzes the trajectories.
The dynamics parameters are written to a new csv file and histograms for
persistence and mean instantaneous speed are generated.

"""

import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from skimage.io._plugins import tifffile_plugin as tifffile

import migration_analysis as migra
import utility_functions as uf

def plot_corr_vs_dist(ax,dist,corr_mean,corr_std,color):
    """Bins correlation vs. distance data (or any 2D data for that matter)

    Parameters
    ----------
    ax : matplotlib axis
        the axis used for plotting
    dist : 1D numpy array
        the distance data (binned x coordinate)
    corr_mean : 1D numpy array
        the mean directional correlation data (binned)
    corr_mean : 1D numpy array
        the standard deviation of the directional correlation data (binned)
    color : string
        the matplotlib color used for plotting

    Returns
    -------
    ax : matplotlib axis
        the axis used for plotting

    """

    #gets rid of any nans
    nan_array = ~np.isnan(corr_mean)
    dist = dist[nan_array]
    corr_std = corr_std[nan_array]
    corr_mean = corr_mean[nan_array]

    #plot means
    ax.plot(dist, corr_mean, 'o', color=color, zorder=2, alpha=0.7)

    #plot std
    fill_y_top = np.ones(len(dist))*(corr_mean+corr_std)
    fill_y_bottom = np.ones(len(dist))*(corr_mean-corr_std)
    ax.fill_between(dist,fill_y_top,fill_y_bottom,facecolor=color,color=color,alpha=0.3,linewidth=0,zorder=1)

    ax.set_xlabel('Distance ($\mu$m)')
    ax.set_ylabel('Directional Correlation')

    return ax

def bin_corr_vs_dist(dist_list,corr_list,n_bins=50):
    """Bins correlation vs. distance data (or any 2D data for that matter)

    Parameters
    ----------
    dist_list : 1D list (or numpy array)
        the distance data
    corr_list : 1D list (or numpy array)
        the directional correlation data

    Returns
    -------
    x_vals : 1D numpy array
        the binned distance data (centered on the bin)
    H_means : 1D numpy array
        the mean of the directional correlation at each bin
    H_stds : 1D numpy array
        the standard deviation of the directional correlation at each bin
    H_lens : 1D numpy array
        the number of data points (n) at each bin

    """

    # converts to np arrays
    start_dist_list = np.array(dist_list)
    mean_corr_list = np.array(corr_list)

    # calculates the means/SDs for the binned data
    bins = np.linspace(np.min(start_dist_list), np.max(start_dist_list) + .000000001, n_bins)
    bin_id = np.digitize(start_dist_list, bins)
    H_means = np.array([np.nanmean(mean_corr_list[bin_id == i]) for i in range(1, len(bins))])
    H_stds = np.array([np.nanstd(mean_corr_list[bin_id == i]) for i in range(1, len(bins))])
    H_lens = np.array([len(mean_corr_list[bin_id == i]) for i in range(1, len(bins))])

    # adjust edges
    x_vals = np.array([(bins[i] + bins[i + 1]) / 2. for i in range(len(bins) - 1)])

    return x_vals, H_means, H_stds, H_lens

def analyze_trackmate_file(img_file_path, track_file_path, time_int=1, px_size=1):
    """Analyzes the TrackMate data

    Parameters
    ----------
    img_file_path : string
        path where the image file is located (must be a .tif file)
    track_file_path : string
        path where the TrackMate tracks file is located (.xml file)
    time_int : float
        time interval in minutes
    px_size: float
        pixel size in um/px
    """

    #makes a new directory to store the data
    save_dir = os.path.splitext(track_file_path)[0]
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    basename = os.path.basename(save_dir)
    print(basename)

    #opens and parses the TrackMate xml file
    tree = ET.parse(track_file_path)
    root = tree.getroot()

    #makes some lists for collecting the tracking data
    print("Calculating Tracking Parameters")
    traj_dict_list = []
    param_dict_list = []

    #loops through each trajectory
    for i, particle in enumerate(root.findall('particle')):

        #sets up lists for the trajectory
        t = np.zeros(int(particle.attrib['nSpots']))
        x = np.zeros_like(t)
        y = np.zeros_like(t)

        #gets the time and position values for the trajectory
        for j, detection in enumerate(particle.findall('detection')):

            t[j] = float(detection.attrib['t']) * time_int
            x[j] = float(detection.attrib['x']) * px_size
            y[j] = float(detection.attrib['y']) * px_size

        if len(t) > 3: #only analyze the tracks if there are at least 3 time points

            #appends the trajectory data to the trajectory list
            traj_dict_list.append({'track_id' : i+1, 't' : t, 'x' : x, 'y' : y})

            #gets some dynamics parameters for the trajectory and appends to the param list
            mean_inst_speed = migra.extract_mean_inst_speed(x,y,t)
            persistence = migra.extract_persistence(x,y)
            time_lag, msd, (slope, intercept) = migra.extract_msd(x,y,t)
            param_dict_list.append({'track_id' : i+1, 'mean_inst_speed' : mean_inst_speed,
                                    'persistence' : persistence, 'coeff_persist' : slope})

    #extracts the directional correlation
    print("Getting Directional Correlation from Trajectories")
    start = time.time()
    dist_list, corr_list = migra.extract_dir_corr(traj_dict_list)
    end = time.time()
    print("Time required:", end - start)

    #bins the distance and correlation data and saves
    print("Plotting and Saving Correlation Data")
    dist_means, corr_means, corr_stds, corr_lens = bin_corr_vs_dist(dist_list,corr_list)
    data_to_write = list(zip(dist_means,corr_means,corr_stds,corr_lens))
    data_to_write.insert(0,['dist_um','corr_mean','corr_std','corr_n'])
    uf.write_csv(data_to_write, os.path.join(save_dir, basename + '_corr_vs_dist.csv'))

    #plots the binned data and saves
    fig, ax = plt.subplots()
    plot_corr_vs_dist(ax,dist_means,corr_means,corr_stds,'b')
    plt.savefig(os.path.join(save_dir, basename + '_corr_vs_dist.pdf'))

    #writes params to file
    print("Plotting and Saving Dynamics Parameters")
    key_list = ['track_id','mean_inst_speed','persistence','coeff_persist']
    data_to_write = [key_list]
    for line in param_dict_list:
        data_to_write.append([line[_] for _ in key_list])
    uf.write_csv(data_to_write, os.path.join(save_dir, basename + '_params.csv'))

    # plots a histogram of the persistence
    persistence_list = np.array([_['persistence'] for _ in param_dict_list])
    fig,ax = plt.subplots()
    ax.hist(persistence_list,bins=np.linspace(0,1,20))
    ax.set_ylabel('Count')
    ax.set_xlabel('Persistence')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, basename + '_hist_persistence.pdf'))
    plt.close()

    # plots a histogram of the mean inst. speed
    speed_list = np.array([_['mean_inst_speed'] for _ in param_dict_list]) * 60. #converts to um/hour
    fig, ax = plt.subplots()
    ax.hist(speed_list,bins=np.linspace(0,20,20))
    ax.set_ylabel('Count')
    ax.set_xlabel('Mean Inst. Speed ($\mu$m/hr)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, basename + '_hist_mean_inst_speed.pdf'))
    plt.close()

    #makes a plot of the trajectories
    print("Plotting Trajectories (this will take a long time if you have >500 trajectories)")
    im_stk = tifffile.imread(img_file_path)
    height, width = im_stk[0].shape
    fig, ax = plt.subplots()
    cm = plt.get_cmap('hot')

    # goes through each trajectory
    print(len(traj_dict_list))
    for j, traj in enumerate(traj_dict_list):

        print(j)
        #sets the colors for plotting
        n = len(traj['x'])
        colors = [cm(1. * i / (n - 1)) for i in range(n - 1)]
        ax.set_prop_cycle('color', colors)

        #plots the trajectory
        for i in range(n - 1):
            ax.plot(traj['x'][i:i + 2], traj['y'][i:i + 2])

    #finishes the plot
    ax.set_xlim(0, width * px_size)
    ax.set_ylim(0, height * px_size)
    ax.set_xlabel("Distance ($\mu$m)")
    ax.set_ylabel("Distance ($\mu$m)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, basename + '_trajectories.pdf'))
    plt.close()

def main():
    """Sets up the analysis for the trajectories from TrackMate.
    You should update the image path, track file path, time interval and pixel size here.
    You should not have to change anything in the rest of the script.

    """

    #sets some initial parameters
    img_file_path = './sample_data/tumor_nuclei_small.tif'
    track_file_path = './sample_data/tumor_nuclei_small_stardist_Tracks.xml'
    time_int = 30 #min
    px_size = 0.91 #um/px

    analyze_trackmate_file(img_file_path, track_file_path, time_int, px_size)

if __name__ == "__main__":
    main()