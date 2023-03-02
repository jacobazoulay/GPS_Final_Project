from gnss_lib_py.parsers.ephemeris import EphemerisManager

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