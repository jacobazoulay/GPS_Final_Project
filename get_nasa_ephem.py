from gnss_lib_py.parsers.ephemeris import EphemerisManager
from datetime import datetime, timezone

def svid_constnum_2_nasa_svid(svid, const_number):
    '''
        SVID + constellation number in the csv file from gnss logger
        to SVID that is used by nasa
    '''
    
    pass

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
    
    print(data.columns)