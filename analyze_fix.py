import os
import glob

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import random
import seaborn as sns
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
warnings.filterwarnings( "ignore", module = "seaborn\..*" )

# THRESHOLDS
random.seed(1)
MIN_SPEED = 0.4
MAX_ZSCORE = 3


def cleanStops(stops):
    # [ (a1, b1) ... (an, bn) ]
    thresh1 = 6
    thresh2 = 20

    # for i in range(1, len(stops)):
    i = 0
    while i < len(stops) - 1:
        if stops[i][1] - stops[i][0] < thresh1:
            if stops[i + 1][0] - stops[i][1] < thresh2:
                stops[i][1] = stops[i + 1][1]
                stops.pop(i + 1)
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
        stop_time = (df["UnixTimeMillis"].iloc[b] - df["UnixTimeMillis"].iloc[a]) / 1000
        stop_times.append(stop_time)

    if len(stop_times) > 0:
        avg_num_stops = len(stops) / T
        avg_time_per_stop = np.mean(stop_times)
        percent_stopped = np.sum(stop_times) / T
        stop_idxs = stops

        return avg_num_stops, avg_time_per_stop, percent_stopped, stop_idxs
    else:
        return np.NaN, np.NaN, np.NaN, np.NaN


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


def split_data_set(filePaths, validation_percentage=0.1, min_seg_length=30):
    '''
        Splits the data set into training and validation sets
        Also splits individual files into slices that are at least
        of length equal to min_seg_length
    '''
    # Split files in half if they meet the minimum time
    all_files = []

    for filePath in filePaths:
        # Data from file
        data = pd.read_csv(filePath)

        # Get min and max time and calculate difference in minutes
        time_delta = (data["UnixTimeMillis"].max() - data["UnixTimeMillis"].min()) / (1000 * 60)

        if time_delta > min_seg_length*2:
            # Split into sets that have at minimum the min_time_before_split ime difference
            num_rows = data.shape[0]
            num_min_time_sets = int(np.floor(time_delta / min_seg_length))

            for i in range(num_min_time_sets):
                pre_threshold = int(i * np.floor(num_rows / num_min_time_sets))
                post_threshold = int((i + 1) * np.floor(num_rows / num_min_time_sets))
                all_files.append(data.iloc[pre_threshold:post_threshold, :])
        else:
            # Don't split in half
            all_files.append(data)

    # Split the files into training and validation sets
    random.shuffle(all_files)

    num_files = len(all_files)
    threshold = int(num_files - np.floor(num_files * validation_percentage))
    train = all_files[:threshold]
    validate = all_files[threshold:]

    return train, validate


def get_features_for_file(df):
    features = np.zeros((8,))

    # Stops
    avg_num_stops, avg_time_per_stop, percent_stopped, stop_idxs = average_stop_time(df)
    features[0:3] = np.array([avg_num_stops, avg_time_per_stop, percent_stopped])

    # Velocity
    max_speed, avg_speed = get_speed_features(df)
    features[3:5] = np.array([max_speed, avg_speed])

    # Acceleration
    max_accel, min_accel, avg_accel = get_acceleration_features(df)

    features[5:8] = np.array([max_accel, min_accel, avg_accel])

    return features


def pre_process_files(filePaths, transittype):
    # Get files
    train, test = split_data_set(filePaths, validation_percentage=0.25, min_seg_length=5)

    # Columns: # Stops / s, Avg Stop Duration, Avg Percent Stop Time, Max speed, Avg Speed, Max Accel, Min Accel, Avg Accel
    # Idxs:    0            1                  2                      3          4          5          6          7
    features = np.zeros((len(train), 8))
    for idx, Fix_df in enumerate(train):
        features[idx, :] = get_features_for_file(Fix_df)

    features_val = np.zeros((len(test), 8))
    for idx, Fix_df in enumerate(test):
        features_val[idx, :] = get_features_for_file(Fix_df)

    # print({"transittype": transittype, 'True #': len(filePaths), "# Train sets": len(train), "# Validate sets": len(test),
    #      "mean": np.nanmean(features, axis=0), "var": np.nanvar(features, axis=0)})

    return {"train": features, "test": features_val}


def calcPScore(trains, test, true_trans, weights=np.ones(8)):
    p_scores = np.zeros((3, *test.shape))
    for j, test_key in enumerate(trains):
        train = trains[test_key]
        for i in range(train.shape[1]):
            _, p_scores[j, :, i] = stats.ttest_1samp(train[:, i], test[:, i], axis=0, nan_policy='omit')

    # p_scores= bike: n1 [metric1, ... metric8]        car: n1 [metric1, ... metric8]       walk: n1 [metric1, ... metric8]
    #                             ...                                   ...                                   ...
    #                 nn [metric1, ... metric8]             nn [metric1, ... metric8]             nn [metric1, ... metric8]

    out = np.sum(p_scores, axis=2)
    pred = np.argmax(out, axis=0)

    p_scores_weighted = p_scores * weights
    out_weighted = np.sum(p_scores_weighted, axis=2)
    pred_weighted = np.argmax(out_weighted, axis=0)

    ids = {"bike": 0, "car": 1, "walk": 2}
    percent_cor = np.sum(pred == ids[true_trans]) / len(pred)
    percent_cor_weighted = np.sum(pred_weighted == ids[true_trans]) / len(pred_weighted)
    print("Unweighted % correct " + true_trans + ": " + str(percent_cor))
    print("Weighted % correct " + true_trans + ": " + str(percent_cor_weighted))

    return [percent_cor, percent_cor_weighted]


def softmax(x):
    num = np.exp(x - np.max(x, axis=0))
    den = np.exp(x - np.max(x, axis=0)).sum(axis=0)
    score = num / den
    return score


def saveFeatureData(train, test):
    for key in train:
        out = pd.DataFrame(train[key])
        out.columns = ["Stops_per_s", "Avg_Stop_Duration", "Avg_Percent_Stop_Time", "Max_speed", "Avg_Speed",
                       "Max_Accel", "Min_Accel", "Avg_Accel"]
        out.to_csv("data/Features/Feature_Data_" + key + ".csv", index=False)


def plotFeatures(train):
    headers = ["Stop Frequency", "Average Stop Duration", "Stop Time Percentage", "Max Speed", "Average Speed",
               "Max Acceleration", "Min Acceleration", "Average Acceleration"]
    units = ["Stop Frequency (Num Stops / s)", "Stop Duration (s)", "Stop Percentage (%)", "Speed (m / s)",
             "Speed (m / s)",
             "Acceleration (m / $s^2$)", "Acceleration (m / $s^2$)", "Acceleration (m / $s^2$)"]
    weights = np.zeros(8)

    for i in range(len(headers)):
        df = pd.DataFrame()
        for key in train:
            add = pd.DataFrame({key: train[key][:, i]})
            df = pd.concat([df, add], axis=1)

        fvalue, pvalue = stats.f_oneway(df['bike'].dropna(), df['car'].dropna(), df['walk'].dropna())
        print(headers[i], " (F, p): ", fvalue, pvalue)
        weights[i] = fvalue

        df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['bike', 'car', 'walk'])
        df_melt["variable"] = df_melt["variable"].str.capitalize()
        # replace column names

        df_melt.columns = ['index', 'Modality', units[i]]
        ax = sns.boxplot(x='Modality', y=units[i], data=df_melt, color='#99c2a2')
        ax = sns.swarmplot(x="Modality", y=units[i], data=df_melt, color='#7d0013')
        plt.title(headers[i])
        plt.show()

    return weights


def plotPercentCors(percent_cors):
    headers = ["Bike", "Car", "Walk"]
    titles = ["Unweighted", "Weighted"]
    percent_cors = np.array(percent_cors)
    plt.show()

    for i in range(2):
        plt.bar(headers, percent_cors[:, i])
        plt.title("Model Prediction Accuracy - " + titles[i])
        plt.ylabel("Accuracy (%)")
        plt.ylim([0,1])
        plt.show()


def crawl():
    # go over each text file in directory
    main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "GNSS_Logger_Fix")
    bike = pre_process_files(glob.glob(os.path.join(main_dir, "Bike/", "*.csv")), "Bike")
    car = pre_process_files(glob.glob(os.path.join(main_dir, "Car/", "*.csv")), "Car")
    walk = pre_process_files(glob.glob(os.path.join(main_dir, "Walk/", "*.csv")), "Walk")
    # bus = pre_process_files(glob.glob(os.path.join(main_dir, "Bus/", "*.csv")), "Bus")
    test = {"bike": bike["test"], "car": car["test"], "walk": walk["test"]}
    train = {"bike": bike["train"], "car": car["train"], "walk": walk["train"]}

    # saveFeatureData(train, test)
    weights = plotFeatures(train)

    percent_cors = []
    for key in test:
        percent_cor = calcPScore(train, test[key], key, weights)
        percent_cors.append(percent_cor)
    plotPercentCors(percent_cors)


if __name__ == '__main__':
    crawl()