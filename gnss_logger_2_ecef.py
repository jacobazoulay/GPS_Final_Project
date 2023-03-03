from fileCrawl import crawl
from get_sat_XYYB import calcPseudo, getSatXYZB
from get_user_pos import getUserXYZ
import pandas as pd
import os
from tqdm import tqdm

def gnss_logger_2_ecef():
    '''
        Takes all the GNSS Logger files and converts 
        them to the user's ECEF position.

        Also saves the position histories in csv's 
        of the same name as the associayed GNSS Logger
        file. 

        recalc_all - whether to recalculate all files
    '''
    # Get all the data and ephemerides
    gnss_data_and_ephems = crawl()
    user_ecef_dict = {}

    # Loop through each modality
    for transitType_name, transitType_data in gnss_data_and_ephems.items():
        transitType_dict = {}
        dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "GNSS_User_ECEF", transitType_name)

        # Loop through each file
        for filename, value in tqdm(transitType_data.items(), desc=transitType_name):
            print(filename)

            # TODO: Figure out why these are problematic...
            problematic_files = ["gnss_log_2023_02_15_16_25_25", "gnss_log_2023_02_21_10_26_50", "gnss_log_2023_02_15_14_47_53", "gnss_log_2023_02_22_16_25_52"] # bike

            if any(filename == name for name in problematic_files):
                continue

            # Get the dfs in the dictionary
            ephemeris = value["ephemeris"]
            gnss_data = value["data"]

            # Get the pseudorange in m
            gnss_data = calcPseudo(gnss_data)

            # Get satellite ECEF position and clock bias
            sat_xyzb = getSatXYZB(ephemeris, gnss_data)

            # TODO: Not sure why this is necessary?
            sat_xyzb.dropna(inplace=True)

            # Convert the time from nanos to seconds
            # Sort by this time
            sat_xyzb.sort_values(by=["RxTime_s"])
            sat_xyzb["RxTime_s"] = sat_xyzb["RxTime_s"]/10**9
            sat_xyzb = sat_xyzb.apply(pd.to_numeric)
            sat_xyzb["RxTime_s"] = sat_xyzb["RxTime_s"].round(0)

            # Get the user's ECEF position history
            user_ecef = getUserXYZ(sat_xyzb)

            # TODO: The first RxTime is 0? The rest look ok...
            user_ecef.drop(index=0, inplace=True)

            # Save the data
            transitType_dict[filename] = user_ecef_dict

            user_ecef.to_csv(os.path.join(dirpath, filename + '.csv'), index=False)

        user_ecef_dict[transitType_name] = transitType_dict
    
    return user_ecef_dict




if __name__=="__main__":
    gnss_logger_2_ecef()
