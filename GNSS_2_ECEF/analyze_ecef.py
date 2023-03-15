import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
from torch import gru
from get_user_pos import plotXYZ
import matplotlib.pyplot as plt


def remove_outliers(df):
    """
    Takes pd dataframe of ECEF user position and removes outliers and empty values
    Input: pandas dataframe of ECEF x y z and time
    Output: Modified pandas dataframe
    """
    # remove NaNs 
    df_copy = df.copy()
    df_copy = df_copy.dropna() 
    df_copy = df_copy[df_copy['RxTime_s'] != 0]
    magnitudes = []
    #add distance from origin to process out outliers
    for i, row in df_copy.iterrows():
        magnitudes.append(np.sqrt((df_copy.loc[i, 'X_u'])**2 + (df_copy.loc[i, 'Y_u'])**2 + (df_copy.loc[i, 'Z_u'])**2)) 

    df_copy['magnitude'] = magnitudes


    z_scores = stats.zscore(df_copy['magnitude'])
    keep_rows = np.abs(z_scores) < 1

    df_copy = df_copy[keep_rows]

    return df_copy


def get_velocity(df):
    diff = df.diff()

    diff['Velocity'] = np.linalg.norm(np.array(diff[['X_u', 'Y_u', 'Z_u']]), axis=1)/diff['RxTime_s']
    diff = diff.dropna()
    z_scores = stats.zscore(diff['Velocity'])

    # standard z score outlier handling
    # strictness = 3
    # keep_rows = np.abs(z_scores) < strictness
    # diff = diff[keep_rows]

    # modified Z score handling with MAD (unaffected by outliers) better than raw Z score
    mad_v = np.median(np.abs(diff['Velocity']-np.median(diff['Velocity'])))
    diff['zscore'] = 0.6745 * np.abs((diff['Velocity']-np.median(diff['Velocity']))/ mad_v)
    strictness = 3.5
    diff = diff[diff['zscore'] <= strictness]

    #Tukey's method (modified IQR not affected by extreme values) performance comparable to MAD
    # Q1 = diff['Velocity'].quantile(0.25)
    # Q3 = diff['Velocity'].quantile(0.75)
    # IQR = stats.iqr(diff['Velocity'])
    # strictness = 1.5
    # lower_fence = Q1 - strictness*IQR
    # upper_fence = Q3 + strictness*IQR
    # diff = diff[(diff['Velocity'] >= lower_fence) & (diff['Velocity'] <= upper_fence)]



    plt.plot(diff['Velocity'])
    plt.show()


def pre_process_files(filePaths, transittype):
    results = {}
    
    # Data from file
    for filePath in filePaths:
        # Data
        ECEF_df = pd.read_csv(filePath)
        ECEF_df = remove_outliers(ECEF_df)
        print(filePath)
        get_velocity(ECEF_df)
        


        # fileName = os.path.splitext(os.path.basename(filePath))[0]
        # fileName = transittype +'_'+ fileName
        # plotXYZ(ECEF_df,fileName,True)
          
    return results


def crawl():
    #go over each text file in directory
    main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "GNSS_User_ECEF")
    bike = pre_process_files(glob.glob(os.path.join(main_dir, "Bike/", "*.csv")), "Bike")    
    car = pre_process_files(glob.glob(os.path.join(main_dir, "Car/", "*.csv")), "Car")
    walk = pre_process_files(glob.glob(os.path.join(main_dir, "Walk/", "*.csv")), "Walk")
    # bus = pre_process_files(glob.glob(os.path.join(main_dir, "Bus/", "*.csv")), "Bus")

if __name__ == '__main__':
    crawl()