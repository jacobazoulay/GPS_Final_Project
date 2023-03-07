import os
import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

def average_stop_time(df):
    df = df.sort_values("UnixTimeMillis")
    df = df.dropna()
    speeds = df["SpeedMps"].values
    speeds = speeds < 0.1

    flip = False
    stops = []
    a, b = 0, 0
    while b < len(speeds):
        if speeds[a] == 1:
            if speeds[b] == 1:
                b += 1
            else:
                stops.append((a, b - 1))
                a = b
        else:
            a += 1
            b += 1


    plt.plot(df["UnixTimeMillis"], df["SpeedMps"])
    for a,b in stops:
        plt.scatter(df["UnixTimeMillis"].iloc[a], df["SpeedMps"].iloc[a])
        plt.scatter(df["UnixTimeMillis"].iloc[b], df["SpeedMps"].iloc[b])

        print(df[["UnixTimeMillis", "SpeedMps"]].iloc[a-1:b+1])
    plt.show()

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
    df = df[df["acceleration_zscore"].abs() < 3]
    
    # Features
    max_accel = df["acceleration"].max()
    min_accel = df["acceleration"].min()
    avg_accel = df["acceleration"].mean()
    
    return max_accel, min_accel, avg_accel

def get_speed_features(df):
    max_speed = df["SpeedMps"].max()
    avg_speed = 0
    
    return max_speed, avg_speed

def pre_process_files(filePaths, transittype): 
    # Columns: # Stops / s, Avg Stop Duration, Max speed, Avg Speed, Max Accel, Min Accel, Avg Accel
    # Idxs:    0            1                  2          3          4          5          6 
    features = np.zeros((len(filePaths), 7))
    
    for idx, filePath in enumerate(filePaths):
        # Data from file
        Fix_df = pd.read_csv(filePath)
        
        # Stops
        # average_stop_time(Fix_df)
        
        # Velocity
        max_speed, avg_speed = get_speed_features(Fix_df)
        features[idx, 2:4] = np.array([max_speed, avg_speed])
        
        # Acceleration
        max_accel, min_accel, avg_accel = get_acceleration_features(Fix_df)
        
        features[idx,4:7] = np.array([max_accel, min_accel, avg_accel])
    
    print(np.mean(features, axis=0))
    print(np.var(features, axis=0))
    
    return features


def crawl():
    #go over each text file in directory
    main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "GNSS_Logger_Fix")
    bike = pre_process_files(glob.glob(os.path.join(main_dir, "Bike/", "*.csv")), "Bike")    
    car = pre_process_files(glob.glob(os.path.join(main_dir, "Car/", "*.csv")), "Car")
    walk = pre_process_files(glob.glob(os.path.join(main_dir, "Walk/", "*.csv")), "Walk")
    # bus = pre_process_files(glob.glob(os.path.join(main_dir, "Bus/", "*.csv")), "Bus")

if __name__ == '__main__':
    crawl()