""" Script for running a WRF forecast and all of the pre-processsing, 
post-processing, visualization and verification which goes with it.
The philosophy is to keep this script as simple and clean as possible 
to represent the high-level progamme flow. 

The first argument MUST BE a configuration file which sets most of the options.  
These options can be overridden by speciying further arguments in the form: --key=value. 
Each stage in the process should try and follow the same loop structure."""


import sys
import time, datetime
import wrftools
import logging

nl      = wrftools.read_namelist(sys.argv[1])
config  = nl.settings

#************************************************
# Allow command-line arguments to override those in the namelist file
# no checking is done here, we just assume the arguments 
# are given in the correct order. They should be specified like this
#
# --option=value 
#************************************************
if len(sys.argv)>2:
    cmd_args = sys.argv[2:]
    wrftools.add_cmd_args(config, cmd_args)

#************************************************
# Logging
#************************************************
logger = wrftools.create_logger(config)
    
    

#************************************************
# Get some required settings
#************************************************
fcst_hours   = config['fcst_hours']               # forecast length
base_dir     = config['base_dir']
domain       = config['domain']
max_dom      = config['max_dom']                  # number of nested domains
fail_mode    = config['fail_mode']                # what to do on failure



#***********************************************
# Initial checks
#***********************************************
try:
    wrftools.check_config(config)
except KeyError:
    logger.error('required setting missing')
    sys.exit()



#************************************************
# Main options. Unpack from config to make sure 
# they are defined. Is this wise? What if an option 
# is not needed, why should it be defined?
#************************************************
logger.info('*** FORECAST CYCLE STARTED ***')
run_level           = config['run_level']
fail_mode           = config['fail_mode']
full_trace          = config['full_trace']
gribmaster          = config['gribmaster']
sst                 = config['sst']
wps                 = config['wps']
ungrib              = config['ungrib']
geogrid             = config['geogrid']
metgrid             = config['metgrid']
ndown               = config['ndown']
real                = config['real']
wrf                 = config['wrf']
upp                 = config['upp']
post                = config['post']
time_series         = config['tseries']
compress            = config['compress']
metadata            = config['metadata']
json                = config['json']
power               = config['power']
ncl                 = config['ncl']
scripts             = config['scripts']
met                 = config['met']
convert_grb         = config['convert_grb']
timing              = config['timing'] # produce timing information
web                 = config['web']
dispatch            = config['dispatch']
archive             = config['archive']
cleanup             = config['cleanup']


#**********************************************************
# Preparation of directories
#**********************************************************
try:
    wrftools.prepare(config)
except Exception,e:
    wrftools.handle(e, fail_mode, full_trace)
    sys.exit()

#**********************************************************
# Forecast initial times
#**********************************************************
init_times = wrftools.get_init_times(config)

#**********************************************************
# Main outer loop of forecast cycle
#**********************************************************
logger.info('Running %d hour WRF forecasts for initial times from %s to %s' %(fcst_hours,init_times[0], init_times[-1]))
for init_time in init_times:
    #
    # Update the config state to reflect initial time
    #
    config['init_time'] = init_time
    logger.info('Running forecast from initial time: %s' %init_time) 

    #
    # Gribmaster
    #
    if gribmaster:
        try:
            wrftools.run_gribmaster(config)
        except IOError, e:
            logger.error('gribmaster failed for initial time %s' % init_time)
            wrftools.handle(e, fail_mode, full_trace)
    if sst:
        wrftools.get_sst(config)


    #
    # WPS
    #
    if wps:
        #try:
        #    wrftools.prepare_wps(config)
        #except IOError, e:
        #    logger.error('WPS failed for initial time %s' %init_time)
        #    wrftools.handle(e, fail_mode, full_trace)
        try:
            wrftools.update_namelist_wps(config)            
        except IOError, e:
            wrftools.handle(e, fail_mode, full_trace)


        if ungrib:
            try:        
                wrftools.run_ungrib(config)
            except Exception, e:
                wrftools.handle(e, fail_mode, full_trace)
        
        if sst:
            try:
                wrftools.ungrib_sst(config)
            except Exception, e:
                wrftools.handle(e, fail_mode, full_trace)

        if geogrid:
            try:
                wrftools.run_geogrid(config)
            except Exception, e:
                wrftools.handle(e, fail_mode, full_trace)
        
        if metgrid:
            try:
                wrftools.run_metgrid(config)
            except Exception, e:
                wrftools.handle(e, fail_mode, full_trace)

    #
    # ndown
    #
    if ndown:
        try:
            if real:
                wrftools.run_real(config)
            wrftools.prepare_ndown(config)
            wrftools.run_ndown(config)
        except Exception, e:
            wrftools.handle(e, fail_mode, full_trace)

    
    
    #
    # WRF standard preparation
    #           
    if wrf and not ndown:
        try:
            wrftools.prepare_wrf(config)
            wrftools.update_namelist_input(config)
        except Exception, e:
            wrftools.handle(e, fail_mode, full_trace)
        
        if real:
            try:
                wrftools.run_real(config)
            except Exception, e:
                wrftools.handle(e, fail_mode, full_trace)
    
    #
    # WRF runs
    #
    if wrf:
        try:
            wrftools.run_wrf(config)
        except Exception, e:
            wrftools.handle(e, fail_mode, full_trace)
        
        if timing:
            try:
                wrftools.timing(config)
            except Exception, e:
                wrftools.handle(e, fail_mode, full_trace)
        try:
           wrftools.move_wrfout_files(config) # this will also copy namelists, logs etc
        except Exception, e:
            wrftools.handle(e, fail_mode, full_trace)
    
    #logger.warn('*** SLEEPING FOR 10 SECONDS TO ALLOW FS TIME TO SORT ITSELF OUT ***')
    #time.sleep(10)

    #
    # Post processing
    #

    if post:
        if compress:
            try:
                wrftools.compress(config)
            except Exception, e:
                wrftools.handle(e, fail_mode, full_trace)

        
        if metadata:
            try:
                wrftools.add_metadata(nl)
            except Exception, e:
                wrftools.handle(e, fail_mode, full_trace)
        
   
    
        if upp:
            for d in range(1,max_dom+1):
                try:
                    config['dom'] = d
                    wrftools.run_unipost(config)
                except Exception, e:
                    logger.error('*** FAIL TIME SERIES ***')
                    wrftools.handle(e, fail_mode, full_trace)
                    
        if convert_grb:
            for d in range(1,max_dom+1):
                try:
                    config['dom'] = d
                    wrftools.convert_grib(config)
                except Exception, e:
                    logger.error('*** FAIL GRIB CONVERSION ***')
                    wrftools.handle(e, fail_mode, full_trace)


        
                
                
    #
    # Met verification tools
    #
    if met:
        for d in range(1,max_dom+1):        
            try:
                config['dom'] = d
                wrftools.run_point_stat(config)
            except Exception, e:
                wrftools.handle(e, fail_mode, full_trace)

        
    #
    # Visualisation
    #
    if ncl:
        for d in range(1,max_dom+1):
            try:
                logger.debug('Processing domain d%02d' %d)
                config['dom'] = d
                wrftools.produce_ncl_plots(config)
            except Exception, e:
                logger.error('*** FAIL NCL ***')
                wrftools.handle(e, fail_mode, full_trace)
    
        
        if config['openlayers']:
            for d in range(1,max_dom+1):
                try:
                    logger.debug('Processing domain d%02d' %d)
                    config['dom'] = d
                    wrftools.produce_ncl_ol_plots(config)
                except Exception, e:
                    logger.error('*** FAIL NCL ***')
                    wrftools.handle(e, fail_mode, full_trace)
    
        if web:
            wrftools.transfer_to_web_dir(config)

    if time_series:
        for d in range(1,max_dom+1):
            try:
                logger.debug('Processing domain d%02d' %d)
                config['dom'] = d
                wrftools.extract_tseries(config)
            except Exception, e:
                logger.error('*** FAIL NCL TIME SERIES ***')
                wrftools.handle(e, fail_mode, full_trace)

                
    #
    # Some bug seems to be creeping in, causing the programme to 
    # fail silently around here. I'm adding a sleep statement
    # as I have a hunch this might be some kind of race condition
    #
    #logger.warn('*** SLEEPING FOR 1 SECONDS TO ENSURE TSERIES FILES ARE CLOSED ***')
    #time.sleep(1)

    
    
    
    if power:
        for d in range(1,max_dom+1):
            try:
                config['dom'] = d
                wrftools.power(config)
            except Exception, e:
                logger.error('*** FAIL POWER CONVERSION ***')
                wrftools.handle(e, fail_mode, full_trace)


    if time_series:
        for d in range(1,max_dom+1):
            try:
                config['dom'] = d
                wrftools.ncdump(config)
            except Exception, e:
                logger.error('*** FAIL TIME SERIES DUMPING  ***')
                wrftools.handle(e, fail_mode, full_trace)

    if web:
        logger.info('*** TRANSFERRING JSON TO WEB DIR ***')
        try:
            wrftools.json_to_web(config)
        except Exception,e:
            logger.error('*** FAIL TRANSFERRING JSON ***')
            wrftools.handle(e, fail_mode, full_trace)

            
    if dispatch:
        dry_run = run_level=='DUMMY'
        wrftools.dispatch.dispatch_all(config['dispatch_json'], init_time, dry_run, log_name=wrftools.LOGGER)

    if archive:
        logger.debug("moving files to longbackup")
        wrftools.archive(config)

    if cleanup:
        logger.debug("cleaning up files")
        wrftools.cleanup(config)


# Final code to get executed
logger.debug('Shutting down the logging framework')
logging.shutdown()
