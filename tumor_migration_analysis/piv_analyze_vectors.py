#!/opt/local/bin/python

"""

This script reads PIV vector data for different experiments and for each time point,
 analyzes the correlation length, mean flow and speed (root mean squared velocity).
 The script plots the correlation Cvv over distance delta_r and the fit to find the correlation length.
 The data is also saved in .dat files.

"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, signal
import pandas as pd

import utility_functions as uf

def fit_func(p,x):

    return np.exp(-x/p[0]) + p[1]

def err_func(p,x,y):

    return fit_func(p,x) - y

def remove_first_row_col(array):
    """Removes first array and column of a numpy array"""

    array = array[1:]
    array = np.transpose(np.transpose(array)[1:])
    return array

def convert_nans(arr):
    
    arr[arr == "nan"] = np.nan
    arr = np.asarray(arr, dtype=float)
    return arr

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def fit_corr_dist_data(delta_r_list,cvv_list):
    """Fits directional correlation vs. distance data with an exponential

     Parameters
     ----------
     delta_r_list : 1D ndarray
        the distance data
     cvv_list : 1D ndarray
         the directional correlation data

     Returns
     ----------
     corr_len : float
         the correlation length from the fit
     r_squared : float
         the r^2 value for the goodness of fit

     """

    # filters out nan values for fitting
    delta_r_list_to_fit = delta_r_list[~np.isnan(cvv_list)]
    cvv_list_to_fit = cvv_list[~np.isnan(cvv_list)]

    # performs the fit
    if len(delta_r_list_to_fit) > 3: #only fit if there is sufficient data points
        p0 = [20.,0]
        p1, success = optimize.leastsq(err_func, p0, args=(delta_r_list_to_fit, cvv_list_to_fit), maxfev=1000000)
        residuals = err_func(p1, delta_r_list_to_fit, cvv_list_to_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((cvv_list_to_fit - np.mean(cvv_list_to_fit)) ** 2)
        r_squared = 1. - (ss_res / ss_tot)
        corr_len = p1[0]
        print(corr_len, r_squared)

        # evaluate whether the fit worked
        if corr_len == p0[0]:
            p1 = [np.nan,np.nan]
            r_squared = np.nan

    else:
        p1 = [np.nan,np.nan]
        r_squared = np.nan

    return p1,r_squared

def plot_corr_dist_data(ax, delta_r_list, cvv_list, p1=None):

    # plots correlation data with fit
    ax.plot(delta_r_list, cvv_list, 'bo', zorder=1, alpha=0.5, label="$C_{vv}$")
    if p1 is not None:
        x_ = np.linspace(np.max(delta_r_list), np.min(delta_r_list), 1000)
        y_ = fit_func(p1, x_)
        corr_len = p1[0]
        try:
            ax.plot(x_, y_, 'b-', zorder=0, label=r"Exp. Fit, $\xi_{vv}$ = %i $\mu$m" % (np.round(corr_len)))
        except TypeError:
            print("Fit unsucessful. Plotting data only.")
        except ValueError:
            print("Problem with PIV vectors. Plotting data only.")

    #finishes the plot
    ax.set_xlabel(r"Distance, $\delta r$ ($\mu$m)")
    ax.set_ylabel(r"Directional Correlation, $C_{vv}$")
    ax.legend(loc='upper right')

    return ax

def plot_corr_vs_dist_mean(ax,dist,corr_mean,corr_std,color):
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

def get_corr_dist(x,y,u,v):
    """Gets the directional correlation vs. distance for an array of vectors.
    Adapted from Garcia, S (2015) Maturation et mise en compétition de monocouches cellulaires (Thesis)

    Parameters
    ----------
    x : 2D ndarray
        array of x coordinates
    y : 2D ndarray
        array of y coordinates
    u : 2D ndarray
        array of vectors along the x-axis
    v : 2D ndarray
        array of vectors along the y-axis

    Returns
    ----------
    r : 1D ndarray
        the radius (distance) component
    Cavg : 1D ndarray
        the directional correlation over r

    """

    #initializes the radius component
    mesh = x[0,1] - x[0,0]
    xmax,ymax = np.array(x.shape) * mesh
    rmax = min([xmax,ymax])
    r = np.arange(0,rmax,1)

    #computes of the correlation matrix
    Norm_matrix = np.ones(shape=x.shape)
    # Norm = signal.correlate2d(Norm_matrix,Norm_matrix)
    Norm = signal.correlate(Norm_matrix, Norm_matrix, mode='full', method='fft')

    du = u-np.nanmean(u)*np.ones(shape=u.shape)
    dv = u-np.nanmean(v)*np.ones(shape=v.shape)
    du[np.isnan(du)] = 0
    dv[np.isnan(dv)] = 0

    # CorrU = signal.correlate2d(du,du)/Norm
    # CorrV = signal.correlate2d(dv,dv)/Norm
    CorrU = signal.correlate(du, du, mode='full', method='fft') / Norm
    CorrV = signal.correlate(dv, dv, mode='full', method='fft') / Norm

    #computes the radial function
    XX,YY = np.meshgrid(np.linspace(-xmax,xmax,CorrU.shape[1]),np.linspace(-ymax,ymax,CorrU.shape[0]))
    Rho,Phi = cart2pol(XX, YY)

    Cu = np.arange(0,rmax,1) * np.nan
    Cv = np.arange(0,rmax,1) * np.nan

    for i in range(int(round(rmax))-1):

        Cu[i] = np.nanmean(CorrU[np.where(np.round(Rho) == i)])
        Cv[i] = np.nanmean(CorrV[np.where(np.round(Rho) == i)])

    #gets the averaged radial function
    Cavg = (Cu + Cv) / (Cu[0] + Cv[0])

    #plots to check
    # plt.plot(r,Cavg,'bo')
    # plt.show()

    return r,Cavg

def analyze_vectors(stk_path):
    """Analyzes dynamics from PIV vector data.
    Extracted data is automatically written to a new directory.

    Parameters
    ----------
    stk_path : string
        the path to the image stack to be analyzed


    """

    # finds the PIV vector data
    data_dir = os.path.splitext(stk_path)[0] + "_piv_data"
    if not os.path.isdir(data_dir):
        raise FileNotFoundError("No PIV vector data found. Please run extraction script first.")

    # makes directories for saving the analysis results
    save_dir = os.path.join(data_dir,"analysis")
    uf.make_dir(save_dir)
    corr_dist_plot_dir = os.path.join(save_dir,"corr_dist_plots")
    uf.make_dir(corr_dist_plot_dir)

    # get unique basename list (from x coordinate data)
    basename_list = [_[:-6] for _ in os.listdir(data_dir) if '_x.dat' in _]
    basename_list = uf.natural_sort(basename_list)

    # creates lists for saving the data
    frame_list = []
    velocity_list = []
    angle_list = []
    speed_list = []
    corr_len_list = []
    r_squared_list = []

    # creates lists for making average correlation vs. distance plot
    delta_r_master_list = []
    cvv_master_list = []

    for i, basename in enumerate(basename_list[1:]): #the 0th time point is blank

        print('Analyzing frame:', basename)

        x = np.array(uf.read_file(data_dir + "/" + basename + "_x.dat"),dtype=float)
        y = np.array(uf.read_file(data_dir + "/" + basename + "_y.dat"),dtype=float)
        U = np.array(uf.read_file(data_dir + "/" + basename + "_u.dat"),dtype=float)
        V = np.array(uf.read_file(data_dir + "/" + basename + "_v.dat"),dtype=float)
        # U = np.array(read_file(data_dir + "/" + basename + "_interp_u.dat"),dtype=float)
        # V = np.array(read_file(data_dir + "/" + basename + "_interp_v.dat"),dtype=float)

        #converts to arrays to account for nans (also convert to float arrays)
        # U = convert_nans(U)
        # V = convert_nans(V)

        #calculates the average velocity and migration angle over the whole field
        U_mean = np.nanmean(U)
        V_mean = np.nanmean(V)
        mean_veloc = np.linalg.norm((U_mean,V_mean)) * 60. #convert to um/hr
        mean_angle = np.degrees(np.arctan2(V_mean,U_mean))
        velocity_list.append(mean_veloc)
        angle_list.append(mean_angle)

        #calculates average speed (root mean squared velocity) over the whole field
        velocities = np.linalg.norm((U,V),axis=0)
        rms_velocity = np.sqrt(np.nanmean(velocities**2)) * 60. #convert to um/hr

        #calculates the directional correlation vs. distance over the whole matrix and appends for averaging later
        delta_r_list, cvv_list = get_corr_dist(x, y, U, V)
        delta_r_master_list.append(delta_r_list)
        cvv_master_list.append(cvv_list)

        # fits and plots the correlation vs. distance data
        p1, r_squared = fit_corr_dist_data(delta_r_list, cvv_list)
        corr_len = p1[0]
        fig, ax = plt.subplots(figsize=(6,4))
        plot_corr_dist_data(ax, delta_r_list, cvv_list, p1)
        plt.tight_layout()
        plt.savefig(os.path.join(corr_dist_plot_dir,"cvv_vs_delta_r_flat_frame_%i.pdf"%(i)))
        plt.close()

        #saves cvv and delta_r data
        data_to_save = list(zip(delta_r_list,cvv_list))
        data_to_save.insert(0,["delta_r","cvv"])
        uf.save_data_array(data_to_save,os.path.join(corr_dist_plot_dir,"cvv_vs_delta_r_flat_frame_%i.dat"%(i)))

        #appends speed and corr_len data
        frame_list.append(i)
        speed_list.append(rms_velocity)
        corr_len_list.append(corr_len)
        r_squared_list.append(r_squared)

    #saves all of the data
    speed_vs_corr_len_data = list(zip(frame_list, velocity_list,
                                      angle_list, speed_list,
                                      corr_len_list, r_squared_list))
    speed_vs_corr_len_data.insert(0, ["frame", "mean_velocity_um_per_hr",
                                      "mean_angle_deg", "mean_speed_um_per_hr",
                                      "corr_len_um", "r_squared"])
    uf.write_csv(speed_vs_corr_len_data,os.path.join(save_dir,"piv_analysis_data.csv"))

    #averages the correlation vs. distance data
    dist_mean, corr_mean, corr_std, corr_n = bin_corr_vs_dist(np.ravel(np.array(delta_r_master_list)),
                                                              np.ravel(np.array(cvv_master_list)))
    #adds initial points for correlation vs. distance (for plots)
    dist_mean = np.insert(dist_mean,0,0.)
    corr_mean = np.insert(corr_mean,0,1.)
    corr_std = np.insert(corr_std,0,0.)

    #plots average correrlation vs. distance data
    fig, ax = plt.subplots()
    ax = plot_corr_vs_dist_mean(ax, dist_mean, corr_mean, corr_std, 'C1')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cvv_vs_delta_r_mean.pdf"))
    plt.close()

    #plots histogram of RMS velocity and correlation length
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    #sets up the bins
    bin_start = 0
    bin_end = 16
    bins = np.linspace(bin_start, bin_end, 21)

    # does the plotting
    n, hist, patches = ax.hist(speed_list, bins=bins)
    ax.clear()
    n = [_ / np.sum(n) for _ in n]
    ax.bar(hist[:-1], n, color='blue', align='edge', width=hist[1] - hist[0])
    ax.set_xlabel("Mean Instantaneous Speed ($\mu$m/hr)")
    ax.set_ylabel("Fraction of cells")
    ax.set_xlim(bin_start, bin_end)

    # finishes the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"hist_RMS_velocity.pdf"))
    plt.close()

    # plots histogram of correlation length
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    # sets up the bins
    bin_start = 0
    bin_end = 30
    bins = np.linspace(bin_start, bin_end, 21)

    # does the plotting
    n, hist, patches = ax.hist(corr_len_list, bins=bins)
    ax.clear()
    n = [_ / np.sum(n) for _ in n]
    ax.bar(hist[:-1], n, color='blue', align='edge', width=hist[1] - hist[0])
    ax.set_xlabel("Correlation Length ($\mu$m)")
    ax.set_ylabel("Fraction of cells")
    ax.set_xlim(bin_start, bin_end)

    # finishes the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"hist_corr_len.pdf"))
    plt.close()

def main():
    """Sets up the analysis of cell dynamics from the raw PIV vectors.
    You should update the image path here.
    You should not have to change anything in the rest of the script.

    """

    stk_path = './sample_data/tumor_nuclei_small/tumor_nuclei_small.tif'
    analyze_vectors(stk_path)

if __name__=="__main__":
    main()