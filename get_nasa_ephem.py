import math
import os 

# Compute UTC time from GNSSRaw measurements

# Get NASA ephemeris
def get_NASA_hourly_ephemeris(utc_time,dirName=""):
    '''
        Get hourly ephemeris files, 
        If a GPS ephemeris file is in dirName, with valid ephemeris for at
        least 24 svs, then read it; else download from NASA's archive of 
        Space Geodesy Data    
    '''
    
    all_GPS_eph = []
    
    # indexes are i-1 from MATLAB
    yearNumber4Digit = utc_time[0]
    yearNumber2Digit = utc_time[0] % 100
    dayNumber = day_of_year(utc_time)
    
    # TODO: Make this into path joins for safety
    hourly_z_file = "TODO: Update"
    eph_filename = hourly_z_file[0:-2]
    full_eph_filename = dirName + eph_filename
    
    # check if ephemeris file already exists (e.g. you downloaded it 'by hand')
    # and if there are fresh ephemeris for lotsa sats within 2 hours of fctSeconds
    got_gps_eph = False
    if os.path.isfile(full_eph_filename):
        # File exists locally
        all_GPS_eph = read_rinex_nav(full_eph_filename)
        _, fct_seconds = utc2gps(utc_time)
        eph_age = all_GPS_eph.gps_week * gps_constants.WEEKSEC + all_GPS_eph.Toe - fct_seconds
        
        # get index into fresh and healthy ephemeris (health bit set => unhealthy) 
        idx_fresh_and_healthy = math.abs(eph_age) < gps_constants.EPHVALIDSECONDS & ~all_GPS_eph.health
        
        # look at allEph.Fit_interval, and deal with values > 0
        good_eph_svs = "TODO"
        
        if len(good_eph_svs) >= gps_constants.MINNUMGPSEPH:
            got_gps_eph = True
        
    if not got_gps_eph:
        url = 'cddis.gsfc.nasa.gov'
        hourly_dir = 'TODO'
        
        try:
            nasa_ftp = ftp(url)
            cd(nasa_ftp, hourly_dir)
            zF = mget(nasa_ftp, hourly_z_file, dirName)
        except Exception as e:
            raise Exception(e)
        
        # Continue with unzip
        
# Helper functions
def day_of_year(utc_time):
    '''
        Return the day number of the year
        
        Inputs:
            utc_time = [year, month, day, hours, minutes, seconds]
    '''
    if len(utc_time) != 6:
        raise Exception('utcTime must be 1x6 for DayOfYear function')
    
    j_day = julian_day([utc_time[0:3],0,0,0]) # midnight morning of this day
    j_day_jan1 = julian_day([utc_time[0],1,1,0,0,0]) # midnight morning of Jan 1st
    day_number = j_day - j_day_jan1 + 1
    
    return day_number

def julian_day(utc_time):
    '''
        input: utcTime [1x6] matrix [year,month,day,hours,minutes,seconds]

        output: totalDays in Julian Days (real number of days)

        Valid input range: 1900 < year < 2100
    '''
    assert len(utc_time[0]) == 6, "utc_time does not have 6 columns in the julian_day fcn"
    
    y = utc_time[0]
    m = utc_time[1]
    d = utc_time[2]
    h = utc_time[3] + utc_time[:,4]/60 + utc_time[:,5]/3600
    
    if m <= 2:
        m = m + 12
        y = y - 1
        
    j_day = math.floor(365.25*y) + math.floor(30.6001*(m+1)) - 15 + 1720996.5 + d + h/24
    
    return j_day
    