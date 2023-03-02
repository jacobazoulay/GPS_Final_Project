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


def svid_constnum_2_nasa_svid(svid, const_number):
    '''
        SVID + constellation number in the csv file from gnss logger
        to SVID that is used by nasa
    '''
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
    target_time = datetime(2021, 1, 9, 12, 0, 0, tzinfo=timezone.utc)
    data = get_nasa_ephem(target_time, ['G01', 'G03'])
    
    print(svid_constnum_2_nasa_svid(2,1))
    print(svid_constnum_2_nasa_svid(10,3))