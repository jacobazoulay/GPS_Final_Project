from gnss_lib_py.parsers.ephemeris import EphemerisManager
from datetime import datetime, timezone

'''
gps: 1
glonass: 3
galileo: 6
beidou: 5
IRNSS: 7
QZSS: 4
SBAS: 2
UNKN: 0
'''


def svid_constnum_2_nasa_svid(row):
    '''
        SVID + constellation number in the csv file from gnss logger
        to SVID that is used by nasa
    '''
    const_number = int(row["ConstellationType"])
    svid = row["Svid"]

    const_num_2_name = {1:"G",   # GPS
                       3:"R",   # GLONASS
                       6:"E",   # Galileo
                       5:"C",   # Beidou
                       4:"J"}   # QZSS
    
    return const_num_2_name[const_number] + str(svid).zfill(2)
    
def get_nasa_ephem(target_time, satellites):
    '''
        Returns nasa ephemerides of desired satellites
        
        Parameters
        ----------
        timestamp : datetime.datetime
            Time of clock
        satellites : List
            List of satellites ['Const_IDSVID']

        Returns
        -------
        data : pd.DataFrame
            DataFrame containing ephemeris entries corresponding to timestamp
    '''
    # May want to not initialize this more than once
    manager = EphemerisManager()
    data = manager.get_ephemeris(target_time, satellites)
    
    return data

if __name__ == "__main__":
    target_time = datetime(2023, 2, 22, 12, 16, 25, tzinfo=timezone.utc)
    list1 = ['G01', 'G03']
    list2 = ['G07', 'E30', 'G04', 'G05', 'G14', 'G16', 'G08', 'G27', 'G09', 'E07', 'G30', 'G20', 'E08', 'E27', 'E19', 'E21']
    list3 = ['G07', 'R16', 'R07', 'E30', 'G04', 'R15', 'G05', 'G14', 'G16', 'G08', 'G27', 'G09', 'R05', 'E07', 'G30', 'G20', 'E08', 'E27', 'E19', 'E21', 'R09', 'R06']
    list4 = ['G07', 'G04', 'G05', 'G14', 'G16', 'G08', 'G27', 'G09', 'G30', 'G20']
    list5 = []
    data = get_nasa_ephem(target_time, list4)  
    
    print(data)