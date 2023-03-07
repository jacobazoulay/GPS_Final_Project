import os
import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

# THRESHOLDS
MIN_SPEED = 0.4
MAX_ZSCORE = 3

def cleanStops(stops):
    # [ (a1, b1) ... (an, bn) ]
    thresh1 = 6
    thresh2 = 20

    # for i in range(1, len(stops)):
    i = 0
    while i < len(stops)-1:
        if stops[i][1] - stops[i][0] < thresh1:
            if stops[i+1][0] - stops[i][1] < thresh2:
                stops[i][1] = stops[i+1][1]
                stops.pop(i+1)
            else:
                stops.pop(i)
        else:
            i += 1

    return stops


def average_stop_time(df):
    df = df.sort_values("UnixTimeMillis")
    df = df.dropna()
    speeds = df["SpeedMps"].values
    speeds = speeds < MIN_SPEED

    T = (df["UnixTimeMillis"].max() - df["UnixTimeMillis"].min()) / 1000

    flip = False
    stops = []
    a, b = 0, 0
    while b < len(speeds):
        if speeds[a] == 1:
            if speeds[b] == 1:
                b += 1
            else:
                stops.append([a, b - 1])
                a = b
        else:
            a += 1
            b += 1

    stops = cleanStops(stops)
    # plt.plot(df["UnixTimeMillis"], df["SpeedMps"])
    # for a,b in stops:
    #     plt.scatter([df["UnixTimeMillis"].iloc[a],df["UnixTimeMillis"].iloc[b]],
    #                 [df["SpeedMps"].iloc[a], df["SpeedMps"].iloc[b]])
    # plt.show()

    stop_times = []
    for a, b in stops:
        stop_time = df["UnixTimeMillis"].iloc[b] - df["UnixTimeMillis"].iloc[a]
        stop_times.append(stop_time)

    avg_num_stops = len(stops)/T
    avg_time_per_stop = np.mean(stop_times)
    percent_stopped = np.sum(stop_times) / T
    stop_idxs = stops

    return avg_num_stops, avg_time_per_stop, percent_stopped, stop_idxs


def get_acceleration_features(df):
    # Sorting
    df = df.sort_values("UnixTimeMillis")
    df = df.dropna()
    
    # Time in seconds
    df["UnixTimeSecond"] = df["UnixTimeMillis"] / 1000
    
    # Find acceleration
    df["acceleration"] = df["SpeedMps"].diff() / df["UnixTimeSecond"].diff()
    
    # Find accelerations that are far from std
    df = df.dropna()
    df["acceleration_zscore"] = stats.zscore(df["acceleration"])
    df = df[df["acceleration_zscore"].abs() < MAX_ZSCORE]
    
    # Features
    max_accel = df["acceleration"].max()
    min_accel = df["acceleration"].min()
    
    # Avg accel - drop anything below 0.4 m/s
    df = df[df["SpeedMps"] > MIN_SPEED]
    
    avg_accel = df["acceleration"].mean()
    
    return max_accel, min_accel, avg_accel

def get_speed_features(df):
    max_speed = df["SpeedMps"].max()
    
    # Avg speed - drop anything below 0.4 m/s
    df = df[df["SpeedMps"] > MIN_SPEED]
        
    avg_speed = df["SpeedMps"].mean()
    
    return max_speed, avg_speed

def pre_process_files(filePaths, transittype): 
    # Columns: # Stops / s, Avg Stop Duration, Avg Percent Stop Time, Max speed, Avg Speed, Max Accel, Min Accel, Avg Accel
    # Idxs:    0            1                  2                      3          4          5          6          7
    features = np.zeros((len(filePaths), 8))
    
    for idx, filePath in enumerate(filePaths):
        # Data from file
        Fix_df = pd.read_csv(filePath)
        
        # Stops
        avg_num_stops, avg_time_per_stop, percent_stopped, stop_idxs = average_stop_time(Fix_df)
        features[idx, 0:3] = np.array([avg_num_stops, avg_time_per_stop, percent_stopped])
        
        # Velocity
        max_speed, avg_speed = get_speed_features(Fix_df)
        features[idx, 3:5] = np.array([max_speed, avg_speed])
        
        # Acceleration
        max_accel, min_accel, avg_accel = get_acceleration_features(Fix_df)
        
        features[idx,5:8] = np.array([max_accel, min_accel, avg_accel])

    return {"mean": np.nanmean(features, axis=0), "var": np.nanvar(features, axis=0)}

def crawl():
    #go over each text file in directory
    main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "GNSS_Logger_Fix")
    bike = pre_process_files(glob.glob(os.path.join(main_dir, "Bike/", "*.csv")), "Bike")    
    car = pre_process_files(glob.glob(os.path.join(main_dir, "Car/", "*.csv")), "Car")
    walk = pre_process_files(glob.glob(os.path.join(main_dir, "Walk/", "*.csv")), "Walk")
    # bus = pre_process_files(glob.glob(os.path.join(main_dir, "Bus/", "*.csv")), "Bus")

if __name__ == '__main__':
    crawl()