#!/opt/local/bin/python

"""

This script contains several functions to extract parameters from cell trajectories.
It is intended to be used as a module to incorporate the functions into a migration analysis pipeline.
Running this script directly will generate and display some plots using sample trajectory data.
The results are simply plotted, but the data can easily be exported.

"""

import itertools
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def extract_mean_inst_speed(x, y, t):
    """Returns the mean instantaneous speed for a 2D trajectory
    
    Parameters
    ----------
    x : 1D numpy array
         an array of the x positions
    y : 1D numpy array
         an array of the y positions
    t : 1D numpy array
         an array of the time points

    Returns
    -------
    mean_inst_speed : float
         the mean instantaneous speed
    """

    speeds = np.zeros(len(x) - 1)
    for i in range(len(x) - 1):
        dist = np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
        speeds[i] = dist / (t[i+1] - t[i])
    mean_inst_speed = np.mean(speeds)

    return mean_inst_speed

def extract_persistence(x, y):
    """Returns the persistence value for a 2D trajectory

    Parameters
    ----------
    x : 1D numpy array
         an array of the x positions
    y : 1D numpy array
         an array of the y positions

    Returns
    -------
    persistence : float
         the persistence of the trajectory
    """

    distances = np.zeros(len(x) - 1)
    for i in range(len(x) - 1):
        distances[i] = np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
    full_dist = np.linalg.norm((x[-1] - x[0], y[-1] - y[0]))
    persistence = full_dist / np.sum(distances)

    return persistence

def extract_msd(x, y, t):
    """Returns the mean squared displacement and power-law exponent for a 2D trajectory

    Parameters
    ----------
    x : 1D numpy array
         an array of the x positions
    y : 1D numpy array
         an array of the y positions
    t : 1D numpy array
         an array of the time points

    Returns
    -------
    msd : 1D numpy array
        the mean squared displacement
    time_lag : 1D numpy array
        the time lag of the msd
    slope : float
        the power-law exponent from the fit of the msd
    intercept : float
        the y-intercept from the power-law fit of the msd
    """

    # extracts the msd (this can be done much faster using a fft)
    msd = np.zeros_like(x) * np.nan
    for i in range(len(x)):
        displ_sq = np.zeros_like(x) * np.nan
        for j in range(len(x) - i):
            displ_sq[j] = (x[j + i] - x[j]) ** 2 + (y[j + i] - y[j]) ** 2
        msd[i] = np.nanmean(displ_sq)

    #fits to find power-law slope (linear regression of log-transform)
    result = linregress(np.log(t[1:15]), np.log(msd[1:15]))
    slope, intercept, r_value, p_value, std_err = result

    return t, msd, (slope, intercept)

def extract_dir_corr(traj_list):
    """Extracts the directional correlation vs. starting distance for a list of trajectories

    Parameters
    ----------
    traj_list : list of dictionaries
        each dictionary should be a trajectory containing: 'x' (x positions), 'y' (y positions), 't' (time points)

    Returns
    -------
    start_dist_list : 1D numpy array
        a list of the starting distances between trajectories
    mean_corr_list : 1D numpy array
        a list of the mean directional correlation corresponding to start_dist_list
    """

    #make list of all possible trajectory combinations
    iter_list = itertools.combinations(range(len(traj_list)), 2)
    # comb_list = list(itertools.islice(iter_list,0,None,1000)) # do this if you want to get only a sample
    comb_list = list(itertools.islice(iter_list, None))

    #make arrays to keep track of the starting distance and mean correlation
    mean_dist_list = []
    mean_corr_list = []

    #iterates through the combinations
    for idxs in comb_list:

        #gets the trajectories to analyze in this iteration
        traj1 = traj_list[idxs[0]]
        traj2 = traj_list[idxs[1]]

        #determines the overlapping timepoints
        traj1_overlap_idx = np.argwhere(np.isin(traj1['t'],traj2['t']))

        #only analyze if there is any overlap
        if len(traj1_overlap_idx) > 0:

            # for overlapping timepionts, makes a new array of the xyt coordinates
            traj1_overlap = np.array([traj1['x'][traj1_overlap_idx],
                                      traj1['y'][traj1_overlap_idx],
                                      traj1['t'][traj1_overlap_idx]])

            # finds the corresponding timepoints in traj2 and gets the xyt coordinates
            traj2_overlap_idx = np.argwhere(np.isin(traj2['t'],traj1['t']))
            traj2_overlap = np.array([traj2['x'][traj2_overlap_idx],
                                      traj2['y'][traj2_overlap_idx],
                                      traj2['t'][traj2_overlap_idx]])

            #calculates the mean correlation for the x,y data across the overlapping timepoints
            dist_list, corr_list = get_corr(traj1_overlap[:2,:], traj2_overlap[:2,:])
            mean_dist = np.nanmean(dist_list)
            mean_corr = np.nanmean(corr_list)

            # append to lists
            mean_dist_list.append(mean_dist)
            mean_corr_list.append(mean_corr)

    return mean_dist_list, mean_corr_list

def get_corr(traj1,traj2):
    """calculates the directional correlations between two 2D trajectories

    Parameters
    ----------
    traj1 : 2D list
        a list for trajectory one with the form [[x1, x2, xn],[y1, y2, yn]]
    traj2 : 2D list
        a list for trajectory one with the form [[x1, x2, xn],[y1, y2, yn]]

    Returns
    -------
    dist_list : 1D numpy array
        an array of the distances between the trajectories at each time point
    corr_list : 1D numpy array
        an array of the correlations at each time point

    """

    #calculates the distances between the trajectories at each time step
    dist_list = np.linalg.norm((traj2[0] - traj1[0], traj2[1] - traj1[1]),axis=0)

    #calculates the displacement angles and the directional correlations
    traj1_angles = np.arctan2(traj1[1][1:] - traj1[1][:-1], traj1[0][1:] - traj1[0][:-1])
    traj2_angles = np.arctan2(traj2[1][1:] - traj2[1][:-1], traj2[0][1:] - traj2[0][:-1])
    corr_list = np.cos(traj2_angles - traj1_angles)

    return dist_list, corr_list

def test_migration_analysis(x,y,t):
    """Calculates trajectory parameters and plots the data to display

    Parameters
    ----------
    x : 1D numpy array
        an array of the x positions
    y : 1D numpy array
        an array of the y positions
    t : 1D numpy array
        an array of the time points
    """

    print("Mean Inst. Speed:", extract_mean_inst_speed(x,y,t), "um/min")
    print("Persistence: ", extract_persistence(x,y))

    #plots the trajectory from the origin
    fig, ax = plt.subplots()
    cmap = plt.cm.get_cmap('plasma')
    n = len(x)
    colors = [cmap(1. * i / (n - 1)) for i in range(n - 1)]
    ax.set_prop_cycle('color', colors)
    for i in range(n - 1):
        ax.plot(x[i:i + 2], y[i:i + 2])
    ax.plot(x[0],y[0],'o',color=cmap(0))
    ax.plot(x[-1],y[-1],'o',color=cmap(np.inf))

    ax_max = max(max(np.sqrt(x**2)),max(np.sqrt(y**2)))
    ax_max = ax_max + ax_max * 0.1
    ax.set_xlim(-ax_max,ax_max)
    ax.set_ylim(-ax_max,ax_max)
    ax.set_xlabel('Displacement ($x$, $\mu$m)')
    ax.set_ylabel('Displacement ($y$, $\mu$m)')
    plt.tight_layout()

    #plots the msd with a power-law fit
    fig2, ax2 = plt.subplots()
    t_lag, msd, (alpha, intercept) = extract_msd(x,y,t)
    ax2.plot(t_lag,msd,'bo',label="Raw Data")
    t_fit = np.linspace(t_lag[1],t_lag[15],10)
    msd_fit = t_fit ** alpha * np.exp(intercept)
    ax2.plot(t_fit,msd_fit,'b-',label=r"$\alpha=%.2f$"%alpha)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Time Lag (min.)')
    ax2.set_ylabel('MSD ($\mu$m)')
    ax2.legend(loc='upper left')
    plt.tight_layout()

    plt.show()

def main():
    """Uses some simulated data to test the trajectory analysis"""

    #set a pixel size and time interval for the tests
    px_size = 1 #pixel size (in um, e.g.)
    time_int = 5 #time interval (in min, e.g.)

    # uses some made-up data
    print("Made-up Trajectory")
    x = np.array([0,1,3,4,3,6,8,7,10,12,14,13,12,13,15,17,19,20,21,20,19,20,19,21,15,16,19,17,16]) * px_size
    y = np.array([0,2,1,3,1,2,5,7,6,8,9,12,10,11,10,11,14,14,12,13,14,12,12,10,9,10,8,7,7]) * px_size
    t = np.arange(len(x)) * time_int
    test_migration_analysis(x,y,t)

    # uses a perfectly straight trajectory
    print("Straight Trajectory")
    x = np.arange(100)
    y = np.zeros(100)
    t = np.arange(len(x)) * time_int
    test_migration_analysis(x,y,t)

    # simulates a 2D random walk
    print("Random Walk")
    steps_x = np.random.randint(-2,3,size=(100,2))
    steps_y = np.random.randint(-2,3,size=(100,2))
    xy = np.concatenate([steps_x,steps_y]).cumsum(0)
    x = xy[:,0]
    y = xy[:,1]
    t = np.arange(len(x)) * time_int
    test_migration_analysis(x,y,t)

if __name__=="__main__":
    main()
