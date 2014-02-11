#************************************************************************
# tseries.py
# 
# Handles the extraction of time-series
#
# TODO
# Split into two modules, one tseries, one power.
# Ensure power uses SPEED and DIRECTION rather than U and V
#
#************************************************************************
import os
import csv
import time, datetime
import numpy as np
import wrftools
import tools
from customexceptions import DomainError
import glob

HOUR = datetime.timedelta(0, 60*60)                 

def extract_tseries(config):
    """ Extracts time series from wrfout_xxx.nc files, and create tseries_xxx.nc files.
    This is the function called by run_forcast.py"""
    logger = wrftools.get_logger()
    logger.info('*** EXTRACTING TIME SERIES ***')
     
    wrfout_dir     = config['wrfout_dir']
    tseries_dir    = config['tseries_dir']
    init_time      = config['init_time']
    dom            = config['dom']
    fcst_file      = '%s/wrfout_d%02d_%s:00:00.nc' %(wrfout_dir, dom, init_time.strftime("%Y-%m-%d_%H")) # note we add on the nc extension here
    loc_file       = config['locations_file']
    ncl_code       = config['tseries_code']
    extract_hgts   = config['extract_hgts']
    ncl_opt_file   = config['ncl_opt_file']
    
    
    ncl_log        = config['ncl_log']
    if not os.path.exists(tseries_dir):
        os.makedirs(tseries_dir)
    
    # Always go via the netcdf file
    tseries_file = '%s/tseries_d%02d_%s.nc' % (tseries_dir, dom,init_time.strftime("%Y-%m-%d_%H"))

    os.environ['FCST_FILE']      = fcst_file
    os.environ['LOCATIONS_FILE'] = loc_file
    os.environ['NCL_OUT_DIR']    = tseries_dir
    os.environ['NCL_OUT_FILE']   = tseries_file
    os.environ['NCL_OPT_FILE']   = ncl_opt_file
    
    
    logger.debug('Setting environment variables')
    logger.debug('FCST_FILE    ----> %s'  % fcst_file)
    logger.debug('NCL_OUT_DIR  ----> %s'  % tseries_dir)
    logger.debug('NCL_OUT_FILE  ----> %s' % tseries_file)
    logger.debug('LOCATIONS_FILE ----> %s' % loc_file)
    logger.debug('NCL_OPT_FILE   ----> %s' % ncl_opt_file)
    logger.debug(extract_hgts)

    ncl_hgts = '(/%s/)' % ','.join(map(str,extract_hgts))
    
    for script in ncl_code:
        cmd  = "ncl 'extract_heights=%s'  %s >> %s 2>&1" % (ncl_hgts,script, ncl_log)
        wrftools.run_cmd(cmd, config)


def json_to_web(config):
    logger = wrftools.get_logger()

    model_run_dir = config['model_run_dir']
    init_time     = config['init_time']
    json_dir      = '%s/json' % model_run_dir
    json_files    = glob.glob('%s/*.json' % json_dir)
    json_web_dir  = wrftools.sub_date(config['json_web_dir'], init_time)
    
    logger.info('*** COPYING JSON TO WEB DIR ***')
    wrftools.transfer(json_files,json_web_dir, mode='copy', debug_level='NONE')
    logger.info('*** COPIED JSON DATA ***')



#*****************************************************************
# Read location files
#*****************************************************************

def _in_domain(lat, lon, lat2d, lon2d):
    """Tests whether (lat,lon) is within domain defined by lat2d and lon2d.
    Simply tests lat within lat2d and lon within lon2d
    Returns boolean, true if the point is within the domain """    

    logger = wrftools.get_logger()
    
    min_lon = np.min(lon2d)
    max_lon = np.max(lon2d)
    min_lat = np.min(lat2d)
    max_lat = np.max(lat2d)
    
    if (lat < min_lat) or (lat> max_lat) or (lon<min_lon) or (lon>max_lon):
          logger.debug("point (%0.3f, %0.3f) is not within domain (%0.2f, %0.3f, %0.3f, %0.3f)" %(lat, lon, min_lat, max_lat, min_lon, max_lon))
          return False
    else:
        return True
    

def _get_index(lat, lon, lat2d, lon2d):
    """ Finds the nearest mass point grid index to the point (lon, lat).
        Works but is slow as just naively searches through the arrays point
        by point. Would be much better to implement the ncl function 
        wrf_user_get_ij to use projection information to actually calculate
        the point in porjected space. 
        
        Arguments:
            @lat: the latitude of the target point
            @lon: longitude of the target point
            @lat2d: 2d array of latitudes of grid points
            @lon2d: 2d array of longitudes of grid points
            
       Returns (i,j) of nearest grid cell. Raises exception if outside"""

    logger = wrftools.get_logger()
    logger.debug("finding index of (%0.3f, %0.3f) " %(lat,lon))
    west_east   = lat2d.shape[0]
    south_north = lat2d.shape[1]    
    logger.debug("dimensions of domain are: %d south_north, %d west_east" %(south_north, west_east))
    
    if not _in_domain(lat, lon, lat2d, lon2d):
        logger.error('point (%0.3f, %0.3f) not in model domain' % (lat, lon))
        raise DomainError('point (%0.3f, %0.3f) not in model domain' %(lat, lon))
    
    
    # 
    # slow, but will work. Just search through the arrays until we 
    # hit the nearest grid point
    #
    min_dist = 10000000  # if the point is further than 10M m away, don't bother!
    min_x = 0
    min_y = 0 
    

    for x in range(west_east-1):
        for y in range(south_north-1):            
            point_lat = lat2d[x,y]
            point_lon = lon2d[x,y]
            
            d = tools.haversine(lat, lon, point_lat, point_lon)
            
            if d < min_dist:
                min_dist = d
                min_x = x
                min_y = y
    
    if min_x==0 or min_y==0 or min_x>west_east or min_y>south_north:
        logger.error("Point is on/off edge of of model domain, this should have been caught earlier!")
        raise DomainError("Point is on/off edge of of model domain")
        
    
    logger.debug('nearest grid index is x=%d, y=%d, %0.3f m away' %(min_x, min_y, min_dist))
    logger.debug('latitude, longitude of original is (%0.3f, %0.3f)' %(lat, lon))
    logger.debug('latitude, longitude of index is (%0.3f, %0.3f)' %(lat2d[min_x,min_y], lon2d[min_x,min_y]))
    
    return (min_x, min_y, min_dist)
    








