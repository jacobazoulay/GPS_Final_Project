import os
import glob
import pandas as pd
import numpy as np
from scipy import stats
from get_user_pos import plotXYZ


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

def pre_process_files(filePaths, transittype):
    results = {}
    
    # Data from file
    for filePath in filePaths:
        # Data
        ECEF_df = pd.read_csv(filePath)
        ECEF_df = remove_outliers(ECEF_df)
        fileName = os.path.splitext(os.path.basename(filePath))[0]
        fileName = transittype +'_'+ fileName
        plotXYZ(ECEF_df,fileName,True)
          
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