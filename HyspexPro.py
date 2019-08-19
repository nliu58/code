""" Hyspex image pre-proccessing tool.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import json, glob, os, re, logging, logging.config

from atmosphere  import make_atm_lut, build_wvc_model
from dem         import process_dem
from figure      import plot_image_area, plot_angle_geometry, make_quickview, plot_wvc_model
from imugps      import get_acquisition_time, process_imugps
from geography   import get_map_crs, get_sun_angles
from geometry    import build_igm, build_sca, build_glt
from radiometric import build_mask

class HypexPro():
    """ Notes about HyspexPro:
    (1) HyspexPro aims to do geometric and radiometric corrections on Hyspex images.
        Our Hyspex imaging system consists of two hyperspectral cameras: VNIR-1800
        (spectral range: 400-1000 nm), and SWIR-384 (960-2500 nm). For more details,
        please refer to the Hyspex website (https://www.hyspex.no/products/).
    (2) The geometric correction focuses on removing the image distortion caused by
        platform motions/attiudes, which are characterized by roll, pitch and heading
        angles. First, the ray-tracing algorithm proposed by Meyer P. (1994), published
        on Remote Sensing of Environment (49: 118-130) is adapted to calculate the
        map coordinates of each pixel of the raw Hyspex image, a process called georeferencing.
        Then, the `warp` tool provided by GDAL (Geospatial Data Abstraction Library )
        is used to resample the georeferenced image to regualr map grids.
    """
    def __init__(self, config):
        # Make an output directory
        if not os.path.exists(config['Data']['output_dir']):
            os.mkdir(config['Data']['output_dir'])

        # Initialize each Hyspex flight
        self.flights = dict()
        for sensor in ['vnir', 'swir']:
            raw_image_files = glob.glob(os.path.join(config['Data']['input_dir'], '*%s*.hyspex' %config['Sensor'][sensor]['id']))
            for raw_image_file in raw_image_files:
                basename = os.path.basename(raw_image_file)

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
                self.flights[index][sensor]['raw_image_file'] = raw_image_file

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
                self.flights[index][sensor]['raw_imugps_file'] = os.path.splitext(raw_image_file)[0]+'.txt'
                self.flights[index]['raw_dem'] = config['Geometric_Correction']['dem']

    def process_flight(self, flight_index):
        config = self.flights[flight_index]
        # log file
        config['log_file'] = os.path.join(config['output_dir'], '%s.log' %flight_index)
        logging.basicConfig(filename=config['log_file'],
                            level=logging.DEBUG,
                            format="%(asctime)s %(funcName)25s: %(message)s",
                            datefmt='%Y-%m-%dT%H:%M:%S',
                            filemode='w')
        logger = logging.getLogger()

        # map coordinate system
        config['map_crs'] = get_map_crs(config['raw_dem'], config['vnir']['raw_imugps_file'])
        logger.info('Map coordinate system: %s' %config['map_crs'].GetAttrValue('projcs'))

        # image acquisition time
        config['acquisition_time'] = get_acquisition_time(os.path.splitext(config['vnir']['raw_image_file'])[0]+'.hdr', config['vnir']['raw_imugps_file'])
        logger.info('Image acquisition time: %s' %config['acquisition_time'])

        # sun zenith and azimuth angles
        config['sun_angles'] = get_sun_angles(config['vnir']['raw_imugps_file'], config['acquisition_time'])
        logger.info('Sun zenith and azimuth [degrees]: %.2f, %.2f' %(config['sun_angles'][0], config['sun_angles'][1]))

        for sensor_index, sensor in enumerate(['swir', 'vnir']):
            basename = os.path.basename(config[sensor]['raw_imugps_file'][:-len('_raw.txt')])

            logger.info('Process %s-sensor IMUGPS data.' %(sensor.upper()))
            config[sensor]['new_imugps_file'] = os.path.join(config[sensor]['output_dir'], basename+'_IMUGPS.txt')
            process_imugps(config[sensor]['new_imugps_file'],
                           config[sensor]['raw_imugps_file'],
                           config[sensor]['imu_offsets'],
                           config['map_crs'])

            logger.info('Process %s-sensor DEM data.' %(sensor.upper()))
            config[sensor]['new_dem_image_file'] = os.path.join(config[sensor]['output_dir'], basename+'_DEM')
            process_dem(config[sensor]['new_dem_image_file'],
                        config['raw_dem'],
                        config[sensor]['new_imugps_file'],
                        config[sensor]['fov'],
                        config['map_crs'],
                        config[sensor]['pixel_size'])

            logger.info('Build %s-sensor IGM.' %(sensor.upper()))
            config[sensor]['igm_image_file'] = os.path.join(config[sensor]['output_dir'], basename+'_IGM')
            build_igm(config[sensor]['igm_image_file'],
                      config[sensor]['new_imugps_file'],
                      config[sensor]['sensor_model_file'],
                      config[sensor]['new_dem_image_file'])
            logger.info('Create %s-sensor SCA.' %(sensor.upper()))
            config[sensor]['sca_image_file'] = os.path.join(config[sensor]['output_dir'], basename+'_SCA')
            build_sca(config[sensor]['sca_image_file'],
                      config[sensor]['new_imugps_file'],
                      config[sensor]['igm_image_file'],
                      config['sun_angles'],
                      config['map_crs'])

            logger.info('Build %s-sensor geometric LUT.' %(sensor.upper()))
            config[sensor]['glt_image_file'] = os.path.join(config[sensor]['output_dir'], basename+'_GLT')
            build_glt(config[sensor]['glt_image_file'],
                      config[sensor]['igm_image_file'],
                      config[sensor]['pixel_size']/2.0,
                      config['map_crs'])

            logger.info('Make %s-sensor qickview.' %(sensor.upper()))
            config[sensor]['qickview_figure_file'] = os.path.join(config[sensor]['output_dir'], basename+'_Quickview.tif')
            make_quickview(config[sensor]['qickview_figure_file'],
                           config[sensor]['raw_image_file'],
                           config[sensor]['glt_image_file'],
                           config[sensor]['setting_file'])

            logger.info('Plot %s-sensor image area.' %(sensor.upper()))
            config[sensor]['image_area_figure_file'] = os.path.join(config[sensor]['output_dir'], basename+'_ImageArea.png')
            plot_image_area(config[sensor]['image_area_figure_file'],
                            config[sensor]['new_dem_image_file'],
                            config[sensor]['igm_image_file'],
                            config[sensor]['new_imugps_file'])
            logger.info('Plot %s-sensor angle geometry.' %(sensor.upper()))
            config[sensor]['angle_geometry_figure_file'] = os.path.join(config[sensor]['output_dir'], basename+'_AngleGeometry.png')
            plot_angle_geometry(config[sensor]['angle_geometry_figure_file'],
                                config[sensor]['sca_image_file'])

        logger.info('Make atmoshperic LUT.')
        config['atm_lut_dir'] = os.path.join(config['output_dir'], 'atm_lut')
        config['atm_lut_file'] = os.path.join(config['atm_lut_dir'], 'ATM_LUT')
        make_atm_lut(config)

        for sensor_index, sensor in enumerate(['vnir', 'swir']):
            logger.info('Build %s-sensor mask.' %(sensor.upper()))
            basename = os.path.basename(config[sensor]['raw_imugps_file'][:-len('_raw.txt')])
            config[sensor]['mask_image_file'] = os.path.join(config[sensor]['output_dir'], basename+'_Mask')
            build_mask(config[sensor]['mask_image_file'],
                       config[sensor]['raw_image_file'],
                       config[sensor]['setting_file'],
                       config['sun_angles'][0])
            logger.info('Build %s-sensor WVC model.' %(sensor.upper()))

            config[sensor]['wvc_model_file'] = os.path.join(config[sensor]['output_dir'], basename+'_WVCModel.txt')
            build_wvc_model(config[sensor]['wvc_model_file'],
                            config['atm_lut_file'],
                            os.path.splitext(config[sensor]['raw_image_file'])[0]+'.hdr',
                            config[sensor]['setting_file'])
            config[sensor]['wvc_model_figure_file'] = os.path.join(config[sensor]['output_dir'], basename+'_WVCModel.png')
            plot_wvc_model(config[sensor]['wvc_model_figure_file'],
                           config[sensor]['wvc_model_file'])

            config[sensor]['smile_image_file'] = os.path.join(config[sensor]['output_dir'], basename+'_Smile')
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
