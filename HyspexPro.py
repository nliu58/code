""" Hyspx image proccessing code.
author: Nanfeng Liu (nliu58@wisc.edu)
"""

import json, glob, os, re
import numpy as np

from atmosphere import make_atm_lut
from dem import process_dem
from envi import read_envi_header
from figure import plot_image_area, plot_angle_geometry, make_quickview
from imugps import process_imugps
from geography import get_raster_crs, define_utm_crs, get_sun_position
from geometry import build_igm, build_sca, build_glt
from radiometric import build_mask, remove_smile_effect

def get_sun_angles(imugps_file, acquisition_time):
    """ Calculate solar angles.
    Arguments:
        imugps_file: str
            Hyspex raw IMUGPS filename.
        acquisition_time: datetime object
            Image acquisition datetime.
    """

    imugps = np.loadtxt(imugps_file)
    longitude, latitude = imugps[:,1].mean(), imugps[:,2].mean()
    sun_zenith, sun_azimuth = get_sun_position(longitude, latitude, acquisition_time)

    return [sun_zenith, sun_azimuth]

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
    header = read_envi_header(header_file)
    week_start = datetime.strptime(f"{header['acquisition date']} 00:00:00", "%Y-%m-%d %H:%M:%S")
    week_seconds = np.loadtxt(imugps_file)[:,7].mean()
    epoch = datetime(1980, 1, 6, 0, 0)
    gps_week = (week_start-epoch).days//7
    time_elapsed = timedelta(days=gps_week*7, seconds=week_seconds)
    when = epoch+time_elapsed

    return when

def get_map_crs(dem, imugps_file):
    """ Get map coordinate system.
    Notes:
        If `dem` is a file, the map coordinate system should be
        the same as that of the dem file; otherwise define a UTM coordinate system
        based on the longitudes and latitudes in `imugps_file`.
    Arguments:
        dem: str or float
            DEM image filename, or user-specified DEM value.
        imugps_file: str
            Hyspex raw IMUGPS filename.
    Returns:
        map_crs: osr object
            Map coordinate system.
    """

    if os.path.isfile(dem):
        map_crs = get_raster_crs(dem)
    else:
        imugps = np.loadtxt(imugps_file)
        longitude, latitude = imugps[:,1].mean(), imugps[:,2].mean()
        map_crs = define_utm_crs(longitude, latitude)

    return map_crs

class HypexPro():
    def __init__(self, config):
        # Create an output directory
        if not os.path.exists(config['Data']['output_dir']):
            os.mkdir(config['Data']['output_dir'])

        # Initialize each Hyspex flight
        self.flights = dict()
        for sensor in ['vnir', 'swir']:
            image_files = glob.glob(os.path.join(config['Data']['input_dir'], '*%s*.hyspex' %config['Sensor'][sensor]['id']))
            for image_file in image_files:
                basename = os.path.basename(image_file)

                # flight dictionary
                span = re.search('%s' %config['Sensor'][sensor]['id'], basename).span()
                index = basename[:span[0]-1]
                if not index in self.flights.keys():
                    self.flights[index] = dict()

                # flight output directory
                flight_dir = os.path.join(config['Data']['output_dir'], index)
                if not os.path.exists(flight_dir):
                    os.mkdir(flight_dir)
                self.flights[index]['output_dir'] = flight_dir

                # sensor dictionary
                if not sensor in self.flights[index].keys():
                    self.flights[index][sensor] = dict()

                # sensor output directory
                self.flights[index][sensor]['output_dir'] = os.path.join(flight_dir, sensor)
                if not os.path.exists(self.flights[index][sensor]['output_dir']):
                    os.mkdir(self.flights[index][sensor]['output_dir'])

                # sensor parameters
                self.flights[index][sensor]['id'] = config['Sensor'][sensor]['id']
                self.flights[index][sensor]['fov'] = config['Sensor'][sensor]['fov']
                self.flights[index][sensor]['ifov'] = config['Sensor'][sensor]['ifov']
                self.flights[index][sensor]['bands'] = config['Sensor'][sensor]['bands']
                self.flights[index][sensor]['samples'] = config['Sensor'][sensor]['samples']

                # image parameters
                self.flights[index][sensor]['raw_image_file'] = image_file

                # radiometric calibration parameters
                self.flights[index][sensor]['setting_file'] = config['Radiometric_Calibration']['setting_file'][sensor]

                # atmospheric correction parameters
                self.flights[index]['atm_database'] = config['Atmospheric_Correction']['atm_database']
                self.flights[index]['rtm_params'] = config['Atmospheric_Correction']['rtm_params']
                self.flights[index]['aerosol_retrieval'] = config['Atmospheric_Correction']['aerosol_retrieval']
                self.flights[index]['water_vapor_retrieval'] = config['Atmospheric_Correction']['water_vapor_retrieval']

                # geometric correction parameters
                self.flights[index][sensor]['pixel_size'] = config['Geometric_Correction']['pixel_size'][sensor]
                self.flights[index][sensor]['imu_offsets'] = config['Geometric_Correction']['imu_offsets'][sensor]
                self.flights[index][sensor]['sensor_model_file'] = config['Geometric_Correction']['sensor_model_file'][sensor]
                self.flights[index][sensor]['raw_imugps_file'] = os.path.splitext(image_file)[0]+'.txt'
                self.flights[index]['raw_dem'] = config['Geometric_Correction']['dem']

    def process_flight(self, flight_index):
        config = self.flights[flight_index]
        config['map_crs'] = get_map_crs(config['raw_dem'], config['vnir']['raw_imugps_file'])
        config['acquisition_time'] = get_acquisition_time(os.path.splitext(config['vnir']['raw_image_file'])[0]+'.hdr', config['vnir']['raw_imugps_file'])
        config['sun_angles'] = get_sun_angles(config['vnir']['raw_imugps_file'], config['acquisition_time'])

        print('Part I: Forward Geocoding')

        for sensor_index, sensor in enumerate(['vnir', 'swir']):
            print('  1.%s. %s Sensor' %(sensor_index+1, sensor.upper()))
            basename = os.path.basename(config[sensor]['raw_imugps_file'][:-len('_raw.txt')])

            print('    (1) Process IMU&GPS')
            config[sensor]['new_imugps_file'] = os.path.join(config[sensor]['output_dir'], basename+'_IMUGPS.txt')
#            process_imugps(config[sensor]['new_imugps_file'],
#                           config[sensor]['raw_imugps_file'],
#                           config[sensor]['imu_offsets'],
#                           config['map_crs'])

            print('    (2) Process DEM')
            config[sensor]['new_dem_image_file'] = os.path.join(config[sensor]['output_dir'], basename+'_DEM')
#            process_dem(config[sensor]['new_dem_image_file'],
#                        config['raw_dem'],
#                        config[sensor]['new_imugps_file'],
#                        config[sensor]['fov'],
#                        config['map_crs'],
#                        config[sensor]['pixel_size'])

            print('    (3) Build IGM')
            config[sensor]['igm_image_file'] = os.path.join(config[sensor]['output_dir'], basename+'_IGM')
#            build_igm(config[sensor]['igm_image_file'],
#                      config[sensor]['new_imugps_file'],
#                      config[sensor]['sensor_model_file'],
#                      config[sensor]['new_dem_image_file'])

            print('    (4) Plot Image Area')
            config[sensor]['image_area_file'] = os.path.join(config[sensor]['output_dir'], basename+'_ImageArea.png')
#            plot_image_area(config[sensor]['image_area_file'],
#                            config[sensor]['new_dem_image_file'],
#                            config[sensor]['igm_image_file'],
#                            config[sensor]['new_imugps_file'])

            print('    (5) Create SCA')
            config[sensor]['sca_image_file'] = os.path.join(config[sensor]['output_dir'], basename+'_SCA')
#            build_sca(config[sensor]['sca_image_file'],
#                      config[sensor]['new_imugps_file'],
#                      config[sensor]['igm_image_file'],
#                      config['sun_angles'],
#                      config['map_crs'])

            print('    (6) Plot Angle Geometry')
            config[sensor]['angle_geometry_file'] = os.path.join(config[sensor]['output_dir'], basename+'_AngleGeometry.png')
#            plot_angle_geometry(config[sensor]['angle_geometry_file'],
#                                config[sensor]['sca_image_file'])

            print('    (7) Build Geometric LUT')
            config[sensor]['glt_image_file'] = os.path.join(config[sensor]['output_dir'], basename+'_GLT')
#            build_glt(config[sensor]['glt_image_file'],
#                      config[sensor]['igm_image_file'],
#                      config[sensor]['pixel_size']/2.0,
#                      config['map_crs'])

            print('    (8) Make Qickview')
            config[sensor]['qickview_file'] = os.path.join(config[sensor]['output_dir'], basename+'_Quickview.tif')
#            make_quickview(config[sensor]['qickview_file'],
#                           config[sensor]['raw_image_file'],
#                           config[sensor]['glt_image_file'],
#                           config[sensor]['setting_file'])

        print('Part II: Make Atmospheric LUT')
        config['atm_lut_dir'] = os.path.join(config['output_dir'], 'atm_lut')
        config['atm_lut_file'] = os.path.join(config['atm_lut_dir'], 'ATM_LUT')
        make_atm_lut(config)
        print('Part III: Radiometric Calibration')
        for sensor_index, sensor in enumerate(['swir', 'vnir']):
            print('  3.%s. %s Sensor' %(sensor_index+1, sensor.upper()))

            print('    (1) Build Mask.')
            config[sensor]['mask_image_file'] = os.path.join(config[sensor]['output_dir'], basename+'_Mask')
#            build_mask(config[sensor]['mask_image_file'],
#                       config[sensor]['raw_image_file'],
#                       config[sensor]['setting_file'],
#                       config['sun_angles'][0])

            print('    (2) Remove Smile Effect')
            config[sensor]['smile_image_file'] = os.path.join(config[sensor]['output_dir'], basename+'_Smile')
            config[sensor]['wvc_model_file'] = os.path.join(config[sensor]['output_dir'], basename+'_WVCModel.png')

            return (config[sensor], config['atm_lut_file'], config['sun_angles'])
            remove_smile_effect(config[sensor],
                                config['sun_angles'][0])
        raise IOError('sssssssssss')

config_file = './config.json'
config = json.load(open(config_file, 'r'))
hyspex = HypexPro(config)
for flight_index, flight_config in hyspex.flights.items():
    config, atm_lut_file, sun_angles = hyspex.process_flight(flight_index)
#    hyspex.process_flight(flight_index)
    raise IOError('sssssssssss')