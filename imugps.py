""" Functions to process Hyspex imu and gps data.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import osr, logging, numpy as np

logger = logging.getLogger(__name__)

def process_imugps(new_imugps_file, old_imugps_file, imu_offsets, map_crs):
    """ Process Hyspex IMU and GPS data.
    Notes:
        (1) IMU offsets are applied.
        (2) Grid convergence is applied to heading.
        (2) GPS longitude and latitude are converted to map easting and northing.
    Arguments:
        new_imugps_file: str
            New IMUGPS filename.
        old_imugps_file: str
            Old IMUGPS filename.
        boresight_offsets: list
            Boresight IMU offsets.
        map_crs: osr object
            Map coordinate system.
    """
    from geography import define_wgs84_crs, get_grid_convergence

    # Load the old IMU/GPS file
    old_imugps = np.loadtxt(old_imugps_file)

    # Apply boresight offsets
    new_imugps = np.zeros((old_imugps.shape[0], 11))
    new_imugps[:,0] = old_imugps[:,0]# Scan line ID
    new_imugps[:,3] = old_imugps[:,3]# Flight height
    new_imugps[:,4] = old_imugps[:,4]+imu_offsets[0]# Roll
    new_imugps[:,5] = old_imugps[:,5]+imu_offsets[1]# Pitch
    new_imugps[:,6] = old_imugps[:,6]+imu_offsets[2]# Heading
    new_imugps[:,7] = old_imugps[:,7]# GPS time
    new_imugps[:,8] = old_imugps[:,1]# Longitude
    new_imugps[:,9] = old_imugps[:,2]# Latitude

    # Apply grid convergence
    grid_convergence = get_grid_convergence(new_imugps[:,8], new_imugps[:,9], map_crs)
    new_imugps[:,6] = new_imugps[:,6]-grid_convergence
    new_imugps[:,10] = grid_convergence

    # Convert GPS longitudes and latitudes are converted to map eastings and northings
    wgs84_crs = define_wgs84_crs()
    transform = osr.CoordinateTransformation(wgs84_crs, map_crs)
    xyz = np.array(transform.TransformPoints(old_imugps[:,1:3]))
    new_imugps[:,1] = xyz[:,0]
    new_imugps[:,2] = xyz[:,1]

    # Save the new IMU/GPS data
    header = ['Map coordinate system = %s' %(map_crs.ExportToWkt()),
              'Index    Map_X    Map_Y    Map_Z    Roll    Pitch    Heading    Timestamp    Longitude    Latitude    Grid_Convergence']
    np.savetxt(new_imugps_file,
               new_imugps,
               header='\n'.join(header),
               fmt='%d    %.3f    %.3f    %.3f    %.10f    %.10f    %.10f    %.5f    %.10f    %.10f    %.10f')
    logger.info('Write IMUGPS data to %s.' %new_imugps_file)

def get_acquisition_time(header_file, imugps_file):
    """Get Hyspex image acquistion time.
    Notes:
        (1) The code is adapted from Brendan Heberlein's (bheberlein@wisc.edu) script.
    Arguments:
        header_file: str
            Hyspex header filename.
        imugps_file: str
            Hyspex imugps filename.
    Returns:
        when: datetime object
            Image acquisition time.
    """

    from datetime import datetime, timedelta
    from envi import read_envi_header

    header = read_envi_header(header_file)
    week_start = datetime.strptime(f"{header['acquisition date']} 00:00:00", "%Y-%m-%d %H:%M:%S")
    week_seconds = np.loadtxt(imugps_file)[:,7].mean()
    epoch = datetime(1980, 1, 6, 0, 0)
    gps_week = (week_start-epoch).days//7
    time_elapsed = timedelta(days=gps_week*7, seconds=week_seconds)
    when = epoch+time_elapsed

    return when