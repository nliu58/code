import gdal,glob,json
import numpy as np, pandas as pd
from os.path import expanduser,os
home = expanduser("~")
import statsmodels.api as sm
from skimage.util.shape import view_as_blocks
import hytools as ht
from multiprocessing import Pool
from skimage import filters,measure
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage.filters import gaussian_filter
#https://geomag.nrcan.gc.ca/calc/mdcal-r-en.php?date=2018-07-24&latitude=43.653305&latitude_direction=1&longitude=89.8006193&longitude_direction=-1&grid=on
#http://www.drillingformulas.com/magnetic-declination-and-grid-convergent-and-their-applications-in-directional-drilling/
avg_vza, avg_raa = get_avg_scan_angles(config['sca_image_file'], mask_image)

def build_wvc_model(wvc_model_figure_file, atm_lut_file, sensor_waves, sensor_fwhms, vza, raa):
    """ Build water vapor model for a certain view geometry.
    Arguments:
        wvc_model_figure_file: str
            Water vapor column estimation model figure filename.
        atm_lut_file: str
            Atmosphere look-up-table file.
        sensor_waves: array
            Sensor center wavelengths.
        sensor_fwhms: array
            Sensor FWHMs.
        vza, raa: array
            View zenith angle, Relative azimuth angle.
    Returns:
        wvc_model: dict
            Column water vapor estimation model.
    """

    # Read atm lut
    atm_lut_metadata = read_metadata(atm_lut_file+'.meta')
    atm_lut_metadata['shape'] = tuple([int(v) for v in atm_lut_metadata['shape']])
    atm_lut_metadata['WVC'] = np.array([float(v) for v in atm_lut_metadata['WVC']])
    atm_lut_metadata['VZA'] = np.array([float(v) for v in atm_lut_metadata['VZA']])
    atm_lut_metadata['RAA'] = np.array([float(v) for v in atm_lut_metadata['RAA']])
    atm_lut_metadata['WAVE'] = np.array([float(v) for v in atm_lut_metadata['WAVE']])
    atm_lut = np.memmap(atm_lut_file,
                        dtype=atm_lut_metadata['dtype'],
                        mode='r',
                        shape=atm_lut_metadata['shape'])

    # Reduce atm lut dimensions.
    atm_lut = atm_lut[1,:,4,:,:,:]# Fix Rho=0.5, VIS=80.0

    # Interpolate lut to `vza` and `raa`.
    vza_lower_bound_index = np.where(atm_lut_metadata['VZA']<=vza)[0][-1]
    vza_upper_bound_index = np.where(atm_lut_metadata['VZA']>vza)[0][0]
    raa_lower_bound_index = np.where(atm_lut_metadata['RAA']<=raa)[0][-1]
    raa_upper_bound_index = np.where(atm_lut_metadata['RAA']>raa)[0][0]
    vza_lower_bound = atm_lut_metadata['VZA'][vza_lower_bound_index]
    vza_upper_bound = atm_lut_metadata['VZA'][vza_upper_bound_index]
    raa_lower_bound = atm_lut_metadata['RAA'][raa_lower_bound_index]
    raa_upper_bound = atm_lut_metadata['RAA'][raa_upper_bound_index]
    interp_rdn =  (atm_lut[:,vza_lower_bound_index,raa_lower_bound_index,:]*(raa_upper_bound-raa)*(vza_upper_bound-vza)+
                   atm_lut[:,vza_lower_bound_index,raa_upper_bound_index,:]*(raa-raa_lower_bound)*(vza_upper_bound-vza)+
                   atm_lut[:,vza_upper_bound_index,raa_lower_bound_index,:]*(raa_upper_bound-raa)*(vza-vza_lower_bound)+
                   atm_lut[:,vza_upper_bound_index,raa_upper_bound_index,:]*(raa-raa_lower_bound)*(vza-vza_lower_bound))/(
                           (raa_upper_bound-raa_lower_bound)*(vza_upper_bound-vza_lower_bound))
    atm_lut.flush()

    # Get model wavelength posistions.
    wvc_model = dict()
    the_wave, _  = get_closest_wave(sensor_waves, 650)
    if np.abs(the_wave-650)<20:# vnir sensor
        three_waves = [890, 940, 1000]
    else:# swir sensor
        three_waves = [1070, 1130, 1200]
    left_wave, left_band = get_closest_wave(sensor_waves, three_waves[0])
    middle_wave, middle_band = get_closest_wave(sensor_waves, three_waves[1])
    right_wave, right_band = get_closest_wave(sensor_waves, three_waves[2])

    left_weight = (right_wave-middle_wave)/(right_wave-left_wave)
    right_weight = (middle_wave-left_wave)/(right_wave-left_wave)

    wvc_model['bands'] = [left_band, middle_band, right_band]
    wvc_model['weights'] = [left_weight, right_weight]
    wvc_model['waves'] = [left_wave, middle_wave, right_wave]

    # Resample radiance to model wavelengths
    sample_coeff = get_resampling_coeff(atm_lut_metadata['WAVE'], sensor_waves[wvc_model['bands']], sensor_fwhms[wvc_model['bands']])
    resampled_rdn = np.dot(interp_rdn, sample_coeff)/10 # 10: mW / (m2 nm) -> mW / (cm2 um)

    # Calculate absorption depths
    depths = resampled_rdn[:,1]/(left_weight*resampled_rdn[:,0]+right_weight*resampled_rdn[:,2])

    # Save depths and wvc to the model
    wvc_model['WVC'] = list(atm_lut_metadata['WVC'])
    wvc_model['Ratio'] = list(depths)

    # Make a plot
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(wvc_model['Ratio'])*100, wvc_model['WVC'], '.-', ms=20, lw=2, color='red')
    plt.xticks([0, 20, 40, 60, 80, 100], [0, 20, 40, 60, 80, 100], fontsize=20)
    plt.yticks([0, 10, 20, 30, 40, 50], [0, 10, 20, 30, 40, 50], fontsize=20)
    plt.text(60, 50, r'$\lambda_{Left}$: %.2f nm' %left_wave, fontsize=20)
    plt.text(60, 45, r'$\lambda_{Middle}$: %.2f nm' %middle_wave, fontsize=20)
    plt.text(60, 40, r'$\lambda_{Right}$: %.2f nm' %right_wave, fontsize=20)

    plt.xlim(0, 100)
    plt.ylim(0, 55)
    plt.xlabel('APDA Ratio (%)', fontsize=20)
    plt.ylabel('WVC (mm)', fontsize=20)
    plt.savefig(wvc_model_figure_file, dpi=1000)
    plt.close()

    return wvc_model

sca_header = read_envi_header(sca_image_file+'.hdr')
    sca_image = np.memmap(sca_image_file,
                          dtype='float32',
                          mode='r',
                          offset=0,
                          shape=(sca_header['bands'], sca_header['lines'], sca_header['samples']))
    figure = plt.figure(figsize=(10, 10))
    ax = figure.add_subplot(111, projection='polar')
    ax.scatter(np.deg2rad(sca_image[1,:,:].flatten()),
               sca_image[0,:,:].flatten(),
               color='green')
    sca_image.flush()
    ax.scatter(np.deg2rad(float(sca_header['sun azimuth'])),
               float(sca_header['sun zenith']),
               color='red',
               marker='*',
               s=500)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')
    ax.tick_params(labelsize=20)
    plt.savefig(angle_geometry_file)
    plt.close()
    del sca_header, figure, ax

    # Get average scan angles
def get_gain_offset(cal_data):
    """ Get calibration gain and offset coefficients.
    Arguments:
        cal_data: dict
            Calibration data.
    Returns:
        gain, offset: array
            Calibration gain and offset coefficients.
    """
    from scipy import constants
    h = constants.Planck # Planck constant
    c = constants.c*1e+9 # Light speed nm/s
    QE = np.tile(cal_data['QE'], (cal_data['spatialSize'], 1)).transpose()
    spectralSampling = np.tile(cal_data['spectralSampling'], (cal_data['spatialSize'], 1)).transpose()
    spectralVector = np.tile(cal_data['spectralVector'], (cal_data['spatialSize'], 1)).transpose()
    gain = h*c*1e6/(cal_data['RE']*
                    QE*
                    cal_data['SF']*
                    cal_data['integrationTime']*
                    np.pi*cal_data['apertureSize']*cal_data['apertureSize']*
                    cal_data['solidAngle']*
                    spectralSampling*
                    spectralVector)
    offset = cal_data['background']

    return gain, offset

import os
import numpy as np
from envi import read_envi_header, empty_envi_header
from radio import get_cal_data, resample_solar_flux, calibrate_single_band
avg_rdn = get_avg_rdn(raw_image, cal_data, mask_image)
def remove_dark_current_outliers(dark_currents, threshold=5):
        for dark_current in dark_currents:
            outlier_indices = np.where(np.abs(dark_current-np.mean(dark_current))>threshold*np.std(dark_current))[0]
            for outlier_index in outlier_indices:
                if outlier_index==0:
                    dark_current[outlier_index] = dark_current[outlier_index+1]
                elif outlier_index==len(dark_current)-1:
                    dark_current[outlier_index] = dark_current[outlier_index-1]
                else:
                    dark_current[outlier_index] = (dark_current[outlier_index-1]+dark_current[outlier_index+1])/2.0
        return dark_currents
#    cal_data['background'] = remove_dark_current_outliers(cal_data['background'])

#        cal_data['backgroundLast'] = remove_dark_current_outliers(cal_data['backgroundLast'])
def get_avg_rdn(raw_image, cal_data, mask_image):
    """Average radiance along each column.
    Arguments:
        raw_image: 3D array
            Raw DN image data, dimensions: [lines, bands, samples].
        mask_image: 2D array
            Mask image, dimension: [lines, samples].
    Returns:
        avg_spectra: 2D array
            Averaged radiance along the column direction, dimension: [bands, samples].
    """

    avg_spectra = np.full((raw_image.shape[1], raw_image.shape[2]), 0, dtype=np.float32)
    for band_index in range(raw_image.shape[1]):
        rdn_image = raw2rdn(raw_image[:,band_index,:], cal_data, band_index)
        rdn_image = np.ma.array(rdn_image, mask=~mask_image)
        avg_spectra[band_index,:] = rdn_image.mean(axis=0)

    return avg_spectra

#avg_rdn_0 = get_avg_rdn(raw_image, cal_data, mask_image)
plt.plot(avg_rdn[18,:])
    plt.show()

    plt.plot(cal_data['spectralVector'], avg_rdn[:,8], 'g-')
    plt.plot(cal_data['spectralVector'], avg_rdn[:,9], 'r-')
    plt.plot(cal_data['spectralVector'], avg_rdn[:,10], 'b-')
    plt.xlim(1000, 1200)
    plt.show()

    plt.plot(cal_data['spectralVector'], avg_rdn_0[:,8], 'g-')
    plt.plot(cal_data['spectralVector'], avg_rdn_0[:,9], 'r-')
    plt.plot(cal_data['spectralVector'], avg_rdn_0[:,10], 'b-')

    plt.plot(cal_data['spectralVector'], avg_rdn_0, 'b-')
    plt.show()

def get_avg_rdn(raw_image, cal_data, mask_image):
    """Average radiance along each column.
    Arguments:
        raw_image: 3D array
            Raw DN image data, dimensions: [lines, bands, samples].
        mask_image: 2D array
            Mask image, dimension: [lines, samples].
    Returns:
        avg_spectra: 2D array
            Averaged radiance along the column direction, dimension: [bands, samples].
    """

    avg_spectra = np.full((raw_image.shape[1], raw_image.shape[2]), 0, dtype=np.float32)
    for band_index in range(raw_image.shape[1]):
        rdn_image = raw2rdn(raw_image[:,band_index,:], cal_data, band_index)
        rdn_image = np.ma.array(rdn_image, mask=~mask_image)
        avg_spectra[band_index,:] = rdn_image.mean(axis=0)

    return avg_spectra

def calibrate_single_band(raw_band_image, cal_data, band):
    """ Calibrate single band image to radiance.
    """
    from scipy import constants
    # Get calibration values
    lines = raw_band_image.shape[0]
    h = constants.Planck # Planck constant
    c = constants.c*1e+9 # Light speed nm/s
    RE = cal_data['RE'][band, :]
    QE = cal_data['QE'][band]
    SF = cal_data['SF']
    integration_time = cal_data['integrationTime']
    aperture_area = np.pi*cal_data['apertureSize']*cal_data['apertureSize']
    solid_angle = cal_data['solidAngle']
    wavelength_interval = cal_data['spectralSampling'][band]
    center_wavelength = cal_data['spectralVector'][band]
    number_of_frames = cal_data['numberOfFrames']
    background = cal_data['background'][band,:]
    background_last = cal_data['backgroundLast'][band,:]
    serial_number = cal_data['serialNumber']

    # Calculate gain
    gain = h*c*1e6/(RE* QE*SF*integration_time*aperture_area*solid_angle*wavelength_interval*center_wavelength)
    gain = np.tile(gain, (lines, 1)).transpose()

    # Calculate offset
    offset = np.zeros_like(gain)
    if serial_number >=3000 and serial_number<=5000:
        for line in range(lines):
            offset[line,:] = background+(background_last-background)*line/number_of_frames

    # Calibration
    index = raw_band_image==cal_data['satValue']
    rdn_band_image = ((raw_band_image.astype('float32')-offset)*gain).astype('float32')
    index = index|(rdn_band_image<0.0)
    rdn_band_image[index] = 0
    del gain, offset, index

    return rdn_band_image


def get_avg_flight_altitude(imugps_file):
    """ Get average flight altitude.
    Arguments:
        imugps_file: str
            Hyspex IMUGPS filename.
    Returns:
        avg_flight_altitude: float
            Average flight altitude.
    """

    imugps = np.loadtxt(imugps_file)
    avg_flight_altitude = imugps[:,3].mean()

    return avg_flight_altitude

def get_closest_band(waves, center_wav):
    """ Get the band id whose wavelength is closest to `center_wav`.
    Arguments:
        waves: array
            Wavelength array.
        center_wav: float
            Center wavelength.
    Returns:
        band: int
            Band ID.
    """

    band = np.argmin(np.abs(np.array(waves)-center_wav))

    return band

solar_flux_file = './solar_flux.dat'



def build_mask(mask_image_file, raw_image_file, setting_file, sun_zenith, sensor):
    """ Pre-classify Hyspex image.
    Arguments:
        preclass_image_file: str
            Pre-classification image filename.
        raw_image_file: str
            Raw Hyspex image filename.
        sun_zenith: float
            Sun zenith angle, in degrees.
        sensor: str
            Sensor type.
    """
    cos_sun_zenith = np.cos(np.deg2rad(sun_zenith))
    # Read raw image data
    raw_header = read_envi_header(os.path.splitext(raw_image_file)[0]+'.hdr')
    raw_image = np.memmap(raw_image_file,
                          dtype='int16',
                          mode='r',
                          offset=raw_header['header offset'],
                          shape=(raw_header['lines'], raw_header['bands'], raw_header['samples']))

    # Read calibration data
    cal_data = get_cal_data(raw_image_file, setting_file=setting_file)

    # Resample solar flux data
    solar_flux = resample_solar_flux(solar_flux_file, cal_data['spectralVector'], cal_data['fwhm'])
    solar_flux *= 10 # mW / (m2 nm) -> mW / (cm2 um)
    T_water = 0.05
    if sensor == 'vnir':
        blue_band = get_closest_band(cal_data['spectralVector'], 470)
        nir_band = get_closest_band(cal_data['spectralVector'], 850)
        blue_refl = calibrate_single_band(raw_image[:,blue_band,:], cal_data, blue_band)*np.pi/(solar_flux[blue_band]*cos_sun_zenith)
        nir_refl = calibrate_single_band(raw_image[:,nir_band,:], cal_data, nir_band)*np.pi/(solar_flux[nir_band]*cos_sun_zenith)
        dark_class = (blue_refl==0)|(nir_refl<=T_water)
    else:
        swir_band = get_closest_band(cal_data['spectralVector'], 1600)
        swir_refl = calibrate_single_band(raw_image[:,swir_band,:], cal_data, swir_band)*np.pi/(solar_flux[swir_band]*cos_sun_zenith)
        dark_class = swir_refl<=T_water

    preclass_header = empty_envi_header()
    preclass_header['description'] = 'Pre-classification map'
    preclass_header['samples'] = raw_header['samples']
    preclass_header['lines'] = raw_header['lines']
    preclass_header['bands'] = 2
    preclass_header['byte order'] = 0
    preclass_header['header offset'] = 0
    preclass_header['interleave'] = 'bsq'
    preclass_header['data type'] = 3
    preclass_header['band names'] = ['Image Row', 'Image Column']

def read_metadata(metadata_file):
    """ Read metadata of a binary file.
    Arguments:
        metadata_file: str
            Metadata filename.
    Returns:
        metadata: dict
            Metadata.
    """
    fid = open(metadata_file, 'r')
    for key in metadata.keys():
        if metadata[key] is None:
            continue
        if type(metadata[key]) is list:
            value = []
            for i, v in enumerate(metadata[key]):
                if (i+1)%5==0:
                    value.append(str(v)+'\n')
                else:
                    value.append(str(v))
            value = '{%s}' %(', '.join(value))
        else:
            value = str(metadata[key])
        fid.write('%s = %s\n' %(key, value))
    fid.close()

def sort_out_files(files):
    """ Sort the output files of the LibRadTran model.
    Notes:
        (1) File basename convension: rho_100_cwv_040_vis_010.out
    Arguments:
        files: list of strings
            Output filenames.
    Returns:
        files: list of strings
            Sorted filename.
    """

    file_dict = dict()
    for file in files:
        tmp = os.path.basename(file)[:-len('.out')].split('_')[1::2]
        # rho_100_cwv_040_vis_010.out
        file_dict[file] = (float(tmp[0]), float(tmp[1]), float(tmp[2]))
    files = sorted(file_dict, key=lambda k:(file_dict[k][0], file_dict[k][1], file_dict[k][2]))

    return files

#out_files = glob.glob(os.path.join(config['atm_lut_dir'], '*.out'))
#    out_files = sort_out_files(out_files)
#    rows = 0
#    fid = open(config['atm_lut_file'], 'wb')
#    for out_file in out_files:
#        data = np.loadtxt(out_file, dtype=np.float32)
#        fid.write(data.T.tostring())
#        rows += data.shape[1]
#    fid.close()
#    columns = data.shape[0]
#    del data
#    print(rows, columns)
#    header = empty_envi_header()
#    header['description'] = 'Atmospheric Look-up-table'
#    header['samples'] = columns
#    header['lines'] = rows
#    header['bands'] = 1
#    header['byte order'] = 0
#    header['header offset'] = 0
#    header['interleave'] = 'bsq'
#    header['data type'] = 4
#    header['band names'] = ['IGM Map X', 'IGM Map Y', 'IGM Map Z']


""" Functions to do atmospheric corrections.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""
#sensor_zenith = []
#        relative_azimuth = []
#        plt.figure()
#        for sensor in ['vnir', 'swir']:
#            sca_header = read_envi_header(config[sensor]['sca_image_file']+'.hdr')
#            sca_image = np.memmap(config[sensor]['sca_image_file'],
#                                  dtype=np.float32,
#                                  mode='r',
#                                  offset=0,
#                                  shape=(2, sca_header['lines'], sca_header['samples']))
#            raa = sun_azimuth-sca_image[1,:,:]
#            raa[raa<0] += 360.0
#            raa[raa>180] = 360.0-raa[raa>180]
#            sensor_zenith += list(np.arange(np.floor(sca_image[0,:,:].min()/vza_step), np.ceil(sca_image[0,:,:].max()/vza_step)+.01)*vza_step)
#            relative_azimuth += list(np.arange(np.floor(raa.min()/raa_step), np.ceil(raa.max()/raa_step)+.01)*raa_step)
#            plt.plot(sca_image[0,:,:], raa, 'b.')
#            sca_image.flush()
#            del raa
#        sensor_zenith = list(set(sensor_zenith))
#        relative_azimuth = list(set(relative_azimuth))
#        print(sensor_zenith)
#        print(relative_azimuth)
#        for sza in sensor_zenith:
#             plt.plot([sza]*len(relative_azimuth), relative_azimuth, 'r.')
#        plt.show()

#sensor_angles = []
#        plt.figure()
#        for sensor in ['vnir', 'swir']:
#            sca_header = read_envi_header(config[sensor]['sca_image_file']+'.hdr')
#            sca_image = np.memmap(config[sensor]['sca_image_file'],
#                                  dtype=np.float32,
#                                  mode='r',
#                                  offset=0,
#                                  shape=(2, sca_header['lines'], sca_header['samples']))
#            raa = sun_azimuth-sca_image[1,:,:]
#            raa[raa<0] += 360.0
#            raa[raa>180] = 360.0-raa[raa>180]
#            angles_1 = set(tuple(zip((np.floor(sca_image[0,:,:]/vza_step)*vza_step).flatten(),(np.floor(raa/raa_step)*raa_step).flatten())))
#            angles_2 = set(tuple(zip((np.floor(sca_image[0,:,:]/vza_step)*vza_step).flatten(),((np.floor(raa/raa_step)+1)*raa_step).flatten())))
#            angles_3 = set(tuple(zip(((np.floor(sca_image[0,:,:]/vza_step)+1)*vza_step).flatten(),(np.floor(raa/raa_step)*raa_step).flatten())))
#            angles_4 = set(tuple(zip(((np.floor(sca_image[0,:,:]/vza_step)+1)*vza_step).flatten(),((np.floor(raa/raa_step)+1)*raa_step).flatten())))
#            angles = set.union(angles_1, angles_2, angles_3, angles_4)
#            sensor_angles = sensor_angles+list(angles)
#            plt.plot(sca_image[0,:,:], raa, 'b.')
#            sca_image.flush()
#            del angles_1, angles_2, angles_3, angles_4, raa
#        sensor_angles = np.array(sensor_angles)
#        plt.plot(sensor_angles[:,0], sensor_angles[:,1], 'r.')
#        plt.show()

# Get sensor angle grids.
#        sensor_angles = []
#        for sensor in ['vnir', 'swir']:
#            sca_header = read_envi_header(config[sensor]['sca_image_file']+'.hdr')
#            sca_image = np.memmap(config[sensor]['sca_image_file'],
#                                  dtype=np.float32,
#                                  mode='r',
#                                  offset=0,
#                                  shape=(2, sca_header['lines'], sca_header['samples']))
#            raa = sun_azimuth-sca_image[1,:,:]
#            raa[raa<0] += 360.0
#            angles_1 = set(tuple(zip((np.floor(sca_image[0,:,:]/vza_step)*vza_step).flatten(),(np.floor(raa/raa_step)*raa_step).flatten())))
#            angles_2 = set(tuple(zip((np.floor(sca_image[0,:,:]/vza_step)*vza_step).flatten(),((np.floor(raa/raa_step)+1)*raa_step).flatten())))
#            angles_3 = set(tuple(zip(((np.floor(sca_image[0,:,:]/vza_step)+1)*vza_step).flatten(),(np.floor(raa/raa_step)*raa_step).flatten())))
#            angles_4 = set(tuple(zip(((np.floor(sca_image[0,:,:]/vza_step)+1)*vza_step).flatten(),((np.floor(raa/raa_step)+1)*raa_step).flatten())))
#            angles = set.union(angles_1, angles_2, angles_3, angles_4)
#            sensor_angles = sensor_angles+list(angles)
#            sca_image.flush()
#            del angles_1, angles_2, angles_3, angles_4, raa
#        sensor_angles = np.array(sensor_angles)

import os
import glob
import numpy as np
from dem import get_avg_elevation
from imugps import get_avg_flight_altitude
from envi import read_envi_header
import multiprocessing
import matplotlib.pyplot as plt



def calibrate_single_band(raw_band_image, cal_data, band_index):
    """ Calibrate single band image to radiance.
    Arguments:
        raw_band_image: 2D array
            Raw image of one band.
        cal_data: dict
            Calibration data.
        band: int
            Band to calibrate.
    """
    from scipy import constants

    # Get calibration values
    lines = raw_band_image.shape[0]
    h = constants.Planck # Planck constant
    c = constants.c*1e+9 # Light speed nm/s
    RE = cal_data['RE'][band_index, :]
    QE = cal_data['QE'][band_index]
    SF = cal_data['SF']
    integration_time = cal_data['integrationTime']
    aperture_area = np.pi*cal_data['apertureSize']*cal_data['apertureSize']
    solid_angle = cal_data['solidAngle']
    wavelength_interval = cal_data['spectralSampling'][band_index]
    center_wavelength = cal_data['spectralVector'][band_index]
    number_of_frames = cal_data['numberOfFrames']
    background = cal_data['background'][band_index,:]
    serial_number = cal_data['serialNumber']

    # Calculate gain
    gain = h*c*1e6/(RE* QE*SF*integration_time*aperture_area*solid_angle*wavelength_interval*center_wavelength)
    gain = np.tile(gain, (lines, 1))

    # Calculate offset
    offset = np.zeros_like(gain)
    if serial_number >=3000 and serial_number<=5000:
        background_last = cal_data['backgroundLast'][band_index,:]
        for line in range(lines):
            offset[line,:] = background+(background_last-background)*line/number_of_frames
        del background_last
    del background

    # Calibration
    mask = raw_band_image==cal_data['satValue']
    rdn_band_image = ((raw_band_image.astype('float32')-offset)*gain).astype('float32')
    mask = mask|(rdn_band_image<0.0)
    rdn_band_image[mask] = 0
    del gain, offset, mask

    return rdn_band_image

vza_step = 5
vaa_step = 15

def make_atm_lut(config, sensor):
    """ Make atmopheric look-up-tables.
    """
    rtm_config = dict()

    # Initialize radiative transfer model configurations
    if not os.path.exists(config['atm_lut_dir']):
        os.mkdir(config['atm_lut_dir'])
    sca_header = read_envi_header(config[sensor]['sca_image_file']+'.hdr')
    out_files = glob.glob(os.path.join(config['atm_lut_dir'], '*.out'))
    if len(out_files):
        tmp = os.path.basename(out_files[0]).split('_')
        elev = float(tmp[1])/1000.0
        zout = float(tmp[3])/1000.0 - elev
        sza = float(tmp[5])
        saa = float(tmp[7])
    else:
        elev = get_avg_elevation(config[sensor]['new_dem_image_file'])/1000.0
        zout = get_avg_flight_altitude(config[sensor]['new_imugps_file'])/1000.0 - elev
        sza = float(sca_header['sun zenith'])
        saa = float(sca_header['sun azimuth'])

    rtm_config['altitude'] = elev
    rtm_config['zout'] = zout
    rtm_config['sun_zenith'] = sza
    rtm_config['sun_azimuth'] = saa
    rtm_config['source_file'] = '../data/solar_flux/kurudz_0.1nm.dat'
    rtm_config['atmosphere_file'] = config['atm_mode']
    rtm_config['lambda_0'] = 400
    rtm_config['lambda_1'] = 2500
    rtm_config['o3'] = 331

    # Get sensor angle grids.
    sca_image = np.memmap(config[sensor]['sca_image_file'],
                          dtype=np.float32,
                          mode='r',
                          offset=0,
                          shape=(2, sca_header['lines'], sca_header['samples']))
    sensor_angles_1 = set(tuple(zip((np.floor(sca_image[0,:,:]/vza_step)*vza_step).flatten(),(np.floor(sca_image[1,:,:]/vaa_step)*vaa_step).flatten())))
    sensor_angles_2 = set(tuple(zip((np.floor(sca_image[0,:,:]/vza_step)*vza_step).flatten(),((np.floor(sca_image[1,:,:]/vaa_step)+1)*vaa_step).flatten())))
    sensor_angles_3 = set(tuple(zip(((np.floor(sca_image[0,:,:]/vza_step)+1)*vza_step).flatten(),(np.floor(sca_image[1,:,:]/vaa_step)*vaa_step).flatten())))
    sensor_angles_4 = set(tuple(zip(((np.floor(sca_image[0,:,:]/vza_step)+1)*vza_step).flatten(),((np.floor(sca_image[1,:,:]/vaa_step)+1)*vaa_step).flatten())))
    sensor_angles = set.union(sensor_angles_1, sensor_angles_2, sensor_angles_3, sensor_angles_4)
    sensor_angles = np.array(list(sensor_angles))
    plt.figure()
    plt.plot(sca_image[0,:,:].flatten(), sca_image[1,:,:].flatten(), 'b.', ms=2)
    plt.plot(sensor_angles[:,0], sensor_angles[:,1], 'r.', ms=10)
    plt.show()
    sca_image.flush()
    del sensor_angles_1, sensor_angles_2, sensor_angles_3, sensor_angles_4
    print(sensor_angles.shape)
    # Make atmospheric look-up-tables
    RHO = [0, 0.5, 1.0]
    CWV = [0, 5, 10, 15, 20, 25, 30, 40, 50]
    VIS = [5, 20, 40, 80, 120, 200]
#    pool = multiprocessing.Pool(processes=config['cpu_count'])
    for vza, vaa in zip(sensor_angles[:,0], sensor_angles[:,1]):
        rtm_config['sensor_zenith'] = vza
        rtm_config['sensor_azimuth'] = vaa
        for rho in RHO:
            rtm_config['albedo'] = rho
            for cwv in CWV:
                rtm_config['h2o'] = cwv
                for vis in VIS:
                    rtm_config['aerosol_visibility'] = vis
                    inp_file = os.path.join(config['atm_lut_dir'],
                                            'zout_%04d_elev_%04d_sza_%03d_saa_%03d_vza_%03d_vaa_%03d_rho_%03d_cwv_%03d_vis_%03d.inp' %(zout*1000,
                                                                                                                                       elev*1000,
                                                                                                                                       sza,
                                                                                                                                       saa,
                                                                                                                                       vza,
                                                                                                                                       vaa,
                                                                                                                                       rho*100,
                                                                                                                                       cwv,
                                                                                                                                       vis))
                    if os.path.exists(inp_file):
                        continue
                    make_inp_file(inp_file, rtm_config)
#                        out_file = inp_file[:-len('.inp')]+'.out'
#                        pool.apply_async(run_rtm, (config['rtm_dir'], inp_file, out_file, ))
#    pool.close()
#    pool.join()

def run_rtm(rtm_dir, inp_file, out_file):
    os.system('cd %s && uvspec <%s> %s' %(rtm_dir, inp_file, out_file))

def make_inp_file(file, params):
    """ Make a libradtran input file.
    Arguments:
        file: str
            The libradtran input filename.
        params: dict
            The libradtran input parameters.
    References:
        libRadtran user’s guide
        from http://www.libradtran.org/doc/libRadtran.pdf
    """
    fid = open(file, 'w')
    """
    1. Wavelength grid/band parameterization.
        wavelength:
            Usage:
                Set the wavelength range by specifying first and last wavelength in nm.
            Syntax:
                wavelength lambda_0 lambda_1
        wavelength_grid_file:
            Usage:
                Location of single column file that sets the wavelength grid used for the internal transmittance calculations.
                The wavelengths must be in nm.
            Syntax:
                wavelength_grid_file file
        mol_abs_param:
            Usage:
                Calculate integrated shortwave or longwave irradiance, or simulate satellite instrument channels.
            Syntax:
                mol_abs_param type
                There are many type options. Here, we always use `reptran` for the type. The syntax would be:
                mol_abs_param reptran fine
                It means "Representative wavelengths parameterization adapted for spectral bands".
                `fine` means 1cm−1; `coarse` means 15 cm-1;
    """
    fid.write('wavelength %s %s\n' %(params['lambda_0'], params['lambda_1']))# wavelength range
    fid.write('mol_abs_param reptran fine\n')
    """
    3. Geometry
        sza:
            Usage:
                The solar zenith angle (degrees).
            Syntax:
                sza value
        phi0:
            Usage:
                Azimuth angle of the sun (0 to 360 degrees).
                Sun in the South: 0 degrees
                Sun in the West: 90 degrees
                Sun in the North: 180 degrees
                Sun in the East: 270 degrees
            Syntax:
                phi0 value
        latitude:
            Usage:
                This option can be used to specify the latitude of the location to simulate.
            Syntax:
                latitude value
        longitude:
            Usage:
                This option can be used to specify the longitude of the location to simulate.
            Syntax:
                longitude value
        time:
            Usage:
                It specifies the time to simulate.
            Syntax
                time YYYY MM DD hh mm ss
        umu:
            Usage:
                Cosine of output polar angles in increasing order, starting with negative
                (downwelling radiance, looking upward) values (if any) and on through positive
                (upwelling radiance, looking downward) values. Must not be zero.
            Syntax:
                umu value
        zout:
            Usage:
                This option is used to specify the output altitudes in km above surface altitude.
                You can also use toa for top of atmosphere.
            Syntax:
                zout value
        altitude:
            Usage:
                Set the bottom level in the model atmosphere provided in atmosphere_file to
                be at the given altitude above sea level (km).
            Syntax:
                altitude value
    """
    fid.write('umu %s\n' %(np.cos(np.deg2rad(np.float(params['sensor_zenith'])))))
    fid.write('phi %s\n' %params['sensor_azimuth'])
    fid.write('sza %s\n' %params['sun_zenith'])
    fid.write('phi0 %s\n' %params['sun_azimuth'])
    fid.write('zout %s\n' %params['zout'])
    fid.write('altitude %s\n' %params['altitude'])
    """
    4. Atmosphere
    atmosphereic profile:
        Usage:
            Location of the atmospheric data file.
        Syntax:
            atmosphere_file file
            libRadtran provides the six standard atmospheres by Anderson et al. (1986):
                midlatitude_summer
                midlatitude_winter
                subarctic_summer
                subarctic_winter
                tropical
                US-standard
    mol_modify:
        Usage:
            Set the total column of a density profile. The column is integrated between the
            user-defined altitude and TOA (top of atmosphere).
        Syntax:
            mol_modify species column unit
            The species is one of O3, O2, H2O, CO2, NO2, BRO, OCLO, HCHO,
            O4, SO2, CH4, N2O, CO, or N2. The unit can be DU (Dobson units), CM_2 (molecules/cm2) or MM.
    aerosol_default:
        Usage:
            Set up a default aerosol according to Shettle (1989). The default properties
            are a rural type aerosol in the boundary layer, background aerosol above 2km,
            spring-summer conditions and a visibility of 50km. These settings may be
            modified with aerosol_haze, aerosol_vulcan, aerosol_season, and
            aerosol_visibility.
        Syntax:
            aerosol_default
    aerosol_season:
        Usage:
            Specify season to get appropriate aerosol profile.
        Syntax:
            aerosol_season season
            `season` is either 1 or 2:
                1 Spring-summer profile.
                2 Fall-winter profile.
    aerosol_haze:
        Usage:
            Specify the aerosol type in the lower 2 km of the atmosphere.
        Syntax:
            aerosol_haze type
            `type` is an integer identifying the following aerosol types:
                1 Rural type aerosols.
                4 Maritime type aerosols.
                5 Urban type aerosols.
                6 Tropospheric type aerosols.
    aerosol_visibility
        Usage:
            Horizontal visibility in km. Affects the profile according to Shettle (1989) and the optical thickness.
        Syntax:
            aerosol_visibility value
    crs_model:
        Usage:
            Choose between various cross sections.
        Syntax:
            crs_model species crs
            Following species are included:
            rayleigh
                Specify the Rayleigh cross section. Choose between the following Rayleigh
                scattering cross sections (crs):
                    Bodhaine
                        Bodhaine et al. (1999) Rayleigh scattering cross section using their
                        Eqs. 22-23.
                    Bodhaine29
                        Bodhaine et al. (1999) Rayleigh scattering cross section using
                        their Eq. 29.
                    Nicolet
                        Nicolet (1984) Rayleigh scattering cross section.
                    Penndorf
                        Penndorf (1957) Rayleigh scattering cross section.
                    Bodhaine et al. (1999) is default.
    """
    fid.write('atmosphere_file %s\n' %params['atmosphere_file'])
    fid.write('mol_modify O3 %s DU\n' %params['o3'])
    fid.write('mol_modify H2O %s MM\n' %params['h2o'])
    fid.write('aerosol_default\n')
    if 'aerosol_visibility' in params.keys():
        fid.write('aerosol_visibility %s\n' %params['aerosol_visibility'])
    fid.write('crs_model rayleigh bodhaine\n')
    """
    5. Radiative transfer equation solver
    rte_solver:
        disort
    """
    fid.write('rte_solver disort\n')
    """
    6. Others
    albedo:
        Usage:
            The Lambertian surface albedo
        Syntax:
            albedo value
            `value` is a number between 0.0 and 1.0, constant for all wavelengths.
    source:
        Usage:
            Set the radiation source type. The type is either solar or thermal.
        Syntax:
            source type [file]
            `type` is either solar or thermal.
    pseudospherical:
        Invokes pseudo-spherical geometry in disort/twostr. Default is plane-parallel.
    output_user:
        Usage:
            With this option the user may specify the columns desired for output:
        Syntax:
            output_user output_1 output_2 ... output_n
            lambda:
                Wavelength in nm.
            uu:
                The radiance (intensity) at umu and phi user specified angles (unit e.g. mW/(m2 nm sr))
            eglo:
                The global irradiance.
    """
    if 'albedo' in params.keys():
        fid.write('albedo %s\n' %params['albedo'])
    fid.write('source solar %s\n' %params['source_file'])
    fid.write('pseudospherical\n')
    fid.write('output_user uu \n')
    fid.write('quiet\n')

def get_glt_mask(x, y, pixel_size, x_min, y_max, rows, cols):
    """ Make a GLT mask.
    Arguments:
        x: array
            IGM map x coordinates array.
        y: array
            IGM map y coordinates array.
        pixels_size: float
            GLT image pixel size.
        x_min: float
            GLT image upper-left map x coordinate.
        y_max: float
            GLT image upper-left map y coordinate.
        rows: int
            GLT image rows.
        cols: int
            GLT image columns.
    Returns:
        glt_mask: bool array
            0: which pixels do not have glt values.
            1: which pixels have glt values.
    """
    # Get row and col index
    x, y = x.flatten(), y.flatten()
    # Remove nan values
    index = ~np.isnan(x)
    x, y = x[index], y[index]
    # Initialize dataframe
    df = pd.DataFrame()
    df['J'] = ((x-x_min)/pixel_size).astype(np.int16)
    df['I'] = ((y_max-y)/pixel_size).astype(np.int16)
    tmp_df = pd.DataFrame(dict(I=df.I, J=df.J))
    # Make masks
    polygon1 = []
    J_min = tmp_df.groupby('I')['J'].min() # Left boundary of the image
    for i, j in zip(J_min.index, J_min.values):
        polygon1.append(tuple([j, i]))
    J_max = tmp_df.groupby('I')['J'].max() # Right boundary of the image
    for i, j in zip(J_max.index.values[::-1], J_max.values[::-1]):
        polygon1.append(tuple([j, i]))
    mask1 = Image.new('L', (cols, rows), 0)
    ImageDraw.Draw(mask1).polygon(polygon1, outline=0, fill=1)
    mask1 = np.array(mask1).astype(np.bool)

    polygon2 = []
    I_min = tmp_df.groupby('J')['I'].min() # Top boundary of the image
    for j, i in zip(I_min.index, I_min.values):
        polygon2.append(tuple([j, i]))
    I_max = tmp_df.groupby('J')['I'].max() # Buttom boundary of the image
    for j, i in zip(I_max.index.values[::-1], I_max.values[::-1]):
        polygon2.append(tuple([j, i]))
    mask2 = Image.new('L', (cols, rows), 0)
    ImageDraw.Draw(mask2).polygon(polygon2, outline=1, fill=1)
    mask2 = np.array(mask2).astype(np.bool)

    glt_mask = mask1*mask2

    return glt_mask

#def build_glt(glt_image_file, igm_image_file, pixel_size, map_crs):
#    """ Create a geographic lookup table (GLT) image.
#    Notes:
#        The GLT image consists of two bands:
#            Band 0: Sample Lookup:
#                Pixel values indicate the column number of the pixel
#                in the input geometry file that belongs at the given Y location in the output image.
#            Band 1: Line Lookup:
#                Pixel values indicate the row number of the pixel
#                in the input geometry file that belongs at the given X location in the output image.
#    Arguments:
#        glt_image_file: str
#            Geographic look-up table filename.
#        igm_image_file: str
#            Input geometry filename.
#        pixel_size: float
#            Output image pixel size.
#        map_crs: osr object
#            GLT image map coordinate system.
#    """
#
#    Delta_Line = 100
#    max_distance = 5*pixel_size
#    # Read IGM map x and y values
#    igm_header = read_envi_header(igm_image_file+'.hdr')
#    igm_image = np.memmap(igm_image_file,
#                          dtype=np.float32,
#                          mode='r',
#                          offset=0,
#                          shape=(igm_header['bands'], igm_header['lines'], igm_header['samples']))
#    # Estimate output image spatial range
#    X_Min = igm_image[0,:,:].min()
#    X_Max = igm_image[0,:,:].max()
#    Y_Min = igm_image[1,:,:].min()
#    Y_Max = igm_image[1,:,:].max()
#    X_Min = np.floor(X_Min/pixel_size)*pixel_size-pixel_size
#    X_Max = np.ceil(X_Max/pixel_size)*pixel_size+pixel_size
#    Y_Min = np.floor(Y_Min/pixel_size)*pixel_size-pixel_size
#    Y_Max = np.ceil(Y_Max/pixel_size)*pixel_size+pixel_size
#
#    # Make interpolation grids
#    Y_Grids, X_Grids = np.mgrid[Y_Max:Y_Min:-pixel_size, X_Min:X_Max:pixel_size]
#    Lines, Samples = Y_Grids.shape
#    glt_image = np.memmap(glt_image_file,
#                          dtype=np.int16,
#                          offset=0,
#                          mode='w+',
#                          shape=(2, Lines, Samples))
#    glt_image[:,:,:] = -1
#    glt_mask = get_glt_mask(igm_image[0,:,:],
#                            igm_image[1,:,:],
#                            pixel_size, X_Min, Y_Max,
#                            Lines, Samples)
#    print('\tLine = ', end='')
#    for Line0 in range(0, Lines, Delta_Line):
#        print('%d, ' %Line0, end='')
#        # subset
#        Line1 = min(Line0+Delta_Line, Lines)
#        X = X_Grids[Line0:Line1,:]
#        Y = Y_Grids[Line0:Line1,:]
#        I, J = np.where(glt_mask[Line0:Line1,:])
#        # build a KD tree
#        i, j = np.where((igm_image[0,:,:]>=X.min()-2*max_distance)&
#                        (igm_image[0,:,:]<=X.max()+2*max_distance)&
#                        (igm_image[1,:,:]>=Y.min()-2*max_distance)&
#                        (igm_image[1,:,:]<=Y.max()+2*max_distance))
#        if len(i)==0:
#            continue
#        # scipy.spatial.ckDTree method
#        tree = cKDTree(list(zip(igm_image[0,i,j], igm_image[1,i,j])), leafsize=100)
#        distance, location = tree.query(list(zip(X[I, J], Y[I, J])), distance_upper_bound=max_distance)
#        index = ~np.isinf(distance)
#
#        # sklearn.neighbors.KDTree method
##        from sklearn.neighbors import KDTree
##        tree = KDTree(list(zip(igm_image[0,i,j], igm_image[1,i,j])), leaf_size=10000)
##        distance, location = tree.query(list(zip(X[I, J], Y[I, J])), return_distance=True, dualtree=True)
##        distance = distance.flatten()
##        location = location.flatten()
##        index = distance<=max_distance
#
#        # sklearn.neighbors.BallTree method
##        from sklearn.neighbors import BallTree
##        tree = BallTree(list(zip(igm_image[0,i,j], igm_image[1,i,j])), leaf_size=10000)
##        distance, location = tree.query(list(zip(X[I, J], Y[I, J])), return_distance=True, dualtree=True)
##        distance = distance.flatten()
##        location = location.flatten()
##        index = distance<=max_distance
#
#        if np.sum(index)==0:
#            continue
#        location = location[index]
#        I = I[index]
#        J = J[index]
#        # fill glt
#        glt_image[0, I+Line0, J] = i[location]
#        glt_image[1, I+Line0, J] = j[location]
#        del tree, X, Y, I, J
#    del igm_image, glt_image, glt_mask, Y_Grids, X_Grids
#    print('%d, Done!' %Lines)
#
#    glt_header = empty_envi_header()
#    glt_header['description'] = 'GLT'
#    glt_header['samples'] = Samples
#    glt_header['lines'] = Lines
#    glt_header['bands'] = 2
#    glt_header['byte order'] = 0
#    glt_header['header offset'] = 0
#    glt_header['interleave'] = 'bsq'
#    glt_header['data type'] = 2
#    glt_header['band names'] = ['Image Row', 'Image Column']
#    glt_header['coordinate system string'] = map_crs.ExportToWkt()
#    glt_header['map info'] = [map_crs.GetAttrValue('projcs'),
#              1, 1, X_Min, Y_Max, pixel_size, pixel_size, ' ', ' ',
#              map_crs.GetAttrValue('datum'), map_crs.GetAttrValue('unit')]
#    if 'utm' in map_crs.GetAttrValue('projcs').lower():
#        glt_header['map info'][7] = map_crs.GetUTMZone()
#        if Y_Max>0.0:
#            glt_header['map info'][8] = 'North'
#        else:
#            glt_header['map info'][8] = 'South'
#    glt_header_file = glt_image_file+'.hdr'
#    write_envi_header(glt_header_file, glt_header)



#def build_vrt(vrt_file, src_files):
#    """ Build VRT.
#    Arguments:
#        vrt_file: str
#            VRT filename.
#        src_files: str or list
#            Source filenames.
#    """
#    if type(src_files) == str:
#        src_files = list(src_files)
#    ds = []
#    for src_file in src_files:
#        ds.append(gdal.Open(src_file))
#    vrt = gdal.BuildVRT(vrt_file, ds)
#    del vrt


#def create_glt(igm_image_file, pixel_size, glt_image_file):
#    """ Create a geographic lookup table (GLT) image.
#    Notes:
#        The GLT image consists of two bands:
#            Band 0: Sample Lookup:
#                Pixel values indicate the column number of the pixel
#                in the input geometry file that belongs at the given Y location in the output image.
#            Band 1: Line Lookup:
#                Pixel values indicate the row number of the pixel
#                in the input geometry file that belongs at the given X location in the output image.
#    Arguments:
#        igm_file: str
#            Input geometry filename.
#        pixel_size: float
#            Output image pixel size.
#    """
#
#    Delta_Line = 500
#    max_distance = 2*pixel_size
#
#    # Read IGM map x and y values
#    igm_header = read_envi_header(igm_image_file+'.hdr')
#    igm_image = np.memmap(igm_image_file,
#                          dtype=np.float32,
#                          mode='r',
#                          offset=0,
#                          shape=(igm_header['bands'], igm_header['lines'], igm_header['samples']))
#
#    # Estimate output image spatial range
#    X_Min = igm_image[0,:,:].min()
#    X_Max = igm_image[0,:,:].max()
#    Y_Min = igm_image[1,:,:].min()
#    Y_Max = igm_image[1,:,:].max()
#    X_Min = np.floor(X_Min/pixel_size)*pixel_size-pixel_size
#    X_Max = np.ceil(X_Max/pixel_size)*pixel_size+pixel_size
#    Y_Min = np.floor(Y_Min/pixel_size)*pixel_size-pixel_size
#    Y_Max = np.ceil(Y_Max/pixel_size)*pixel_size+pixel_size
#
#    # Make interpolation grids
#    Y_Grids, X_Grids = np.mgrid[Y_Max:Y_Min:-pixel_size, X_Min:X_Max:pixel_size]
#    Lines, Samples = Y_Grids.shape
#    glt_image = np.memmap(glt_image_file,
#                          dtype=np.int16,
#                          offset=0,
#                          mode='w+',
#                          shape=(2, Lines, Samples))
#    glt_image[:,:,:] = -1
#    glt_mask = create_glt_mask(igm_image[0,:,:],
#                               igm_image[1,:,:],
#                               pixel_size, X_Min, Y_Max,
#                               Lines, Samples)
#    import datetime
#    t1 = datetime.datetime.now()
#    print(t1)
#    print('\tLine = ', end='')
#    for Line0 in range(0, Lines, Delta_Line):
#        print('%d, ' %Line0, end='')
#        # subset
#        Line1 = min(Line0+Delta_Line, Lines)
#        I, J = np.where(glt_mask[Line0:Line1,:])
#        if len(I)==0:
#            continue
#        X = X_Grids[Line0:Line1,:][I,J]
#        Y = Y_Grids[Line0:Line1,:][I,J]
#        # build a KD tree
#        i, j = np.where((igm_image[0,:,:]>=X.min()-2*max_distance)&
#                        (igm_image[0,:,:]<=X.max()+2*max_distance)&
#                        (igm_image[1,:,:]>=Y.min()-2*max_distance)&
#                        (igm_image[1,:,:]<=Y.max()+2*max_distance))
#        if len(i)==0:
#            continue
#        x = igm_image[0,i,j]
#        y = igm_image[1,i,j]
#        tree = spatial.cKDTree(list(zip(x, y)), leafsize=10000)
#        # KD tree query
#        locations = tree.query_ball_point(list(zip(X, Y)), max_distance)
#        for index, location in enumerate(locations):
#            if len(location)==0:
#                continue
#            d=(x[location]-X[index])**2+(y[location]-Y[index])**2
#            glt_image[0, I[index]+Line0, J] = j[location[np.argmin(d)]]
#            glt_image[1, I[index]+Line0, J] = i[location[np.argmin(d)]]
#        del tree, X, Y, I, J, x, y, i, j
#    del igm_image, glt_image, glt_mask, Y_Grids, X_Grids
#    print('%d, Done!' %Lines)
#
#    glt_header = empty_envi_header()
#    glt_header['description'] = 'GLT'
#    glt_header['samples'] = Samples
#    glt_header['lines'] = Lines
#    glt_header['bands'] = 2
#    glt_header['byte order'] = 0
#    glt_header['header offset'] = 0
#    glt_header['interleave'] = 'bsq'
#    glt_header['data type'] = 2
#    glt_header['band names'] = ['Image Column', 'Image Row']
#    glt_header_file = glt_image_file+'.hdr'
#    write_envi_header(glt_header, glt_header_file)
#    t2 = datetime.datetime.now()
#    print(t2)
#
#    print('Total time: %s' %(t2-t1))


#http://www.fundza.com/algorithmic/quadtree/index.html
#def create_glt(igm_image_file, pixel_size, glt_image_file):
#    """ Create a geographic lookup table (GLT) image. The GLT image consists of two bands:
#            Sample Lookup:
#                Pixel values indicate the column (sample) number of the pixel
#                in the input geometry file that belongs at the given Y location in the output image.
#            Line Lookup:
#                Pixel values indicate the row (line) number of the pixel in the input geometry file
#                that belongs at the given X location in the output image.
#    Arguments:
#        igm_file: str
#            Input geometry filename.
#        pixel_size: float
#            Output image pixel size.
#    """
#    # Read IGM map x and y values
#    igm_header = read_envi_header(igm_image_file+'.hdr')
#    igm_image = np.memmap(igm_image_file,
#                          dtype=np.float32,
#                          mode='r',
#                          offset=0,
#                          shape=(igm_header['bands'], igm_header['lines'], igm_header['samples']))
#    lines = igm_image.shape[1]
#    delta_line = 10
#    max_distance = 5*pixel_size
#
#    # Estimate output image spatial range
#    X_Min = igm_image[0,...].min()
#    X_Max = igm_image[0,...].max()
#    Y_Min = igm_image[1,...].min()
#    Y_Max = igm_image[1,...].max()
#    X_Min = np.floor(X_Min/pixel_size)*pixel_size-pixel_size
#    X_Max = np.ceil(X_Max/pixel_size)*pixel_size+pixel_size
#    Y_Min = np.floor(Y_Min/pixel_size)*pixel_size-pixel_size
#    Y_Max = np.ceil(Y_Max/pixel_size)*pixel_size+pixel_size
#
#    # Make interpolation grids
#    Y_Grids, X_Grids = np.mgrid[Y_Max:Y_Min:-pixel_size, X_Min:X_Max:pixel_size]
#    glt_image = np.memmap(glt_image_file,
#                          dtype=np.int16,
#                          offset=0,
#                          mode='w+',
#                          shape=(2, X_Grids.shape[0], Y_Grids.shape[1]))
#    glt_image[...] = -1
#    print('\tLine = ', end='')
#    for line0 in range(0, lines, delta_line):
#        print('%d, ' %line0, end='')
#        # extract a subset
#        line1 = min(line0+delta_line, lines)
#        x = igm_image[0,line0:line1,:]
#        y = igm_image[1,line0:line1,:]
#        # build a KD tree
#        tree = spatial.KDTree(list(zip(x.ravel(), y.ravel())))
#        # narrowdown interpolation range
#        I, J = np.where((X_Grids>=x.min())&(X_Grids<=x.max())&(Y_Grids>=y.min())&(Y_Grids<=y.max()))
#        # KD tree search
#        if len(I)==0:
#            continue
#        distance, location = tree.query(list(zip(X_Grids[I, J], Y_Grids[I, J])), p=2)
#        index = distance<=max_distance
#        if np.sum(index)==0:
#            continue
#        I = I[index]
#        J = J[index]
#        location = location[index]
#        rows, cols = np.unravel_index(location, x.shape)
#        # fill glt
#        glt_image[0, I, J] = rows+line0
#        glt_image[1, I, J] = cols
#        del x, y
#    del igm_image, glt_image
#    print('Done!')
#
#    glt_header = empty_envi_header()
#    glt_header['description'] = 'GLT'
#    glt_header['samples'] = X_Grids.shape[1]
#    glt_header['lines'] = X_Grids.shape[0]
#    glt_header['bands'] = 2
#    glt_header['byte order'] = 0
#    glt_header['header offset'] = 0
#    glt_header['interleave'] = 'bsq'
#    glt_header['data type'] = 2
#    glt_header['band names'] = ['Image Row', 'Image Column']
#    glt_header_file = glt_image_file+'.hdr'
#    write_envi_header(glt_header, glt_header_file)

#import pandas as pd
#from PIL import Image
#from PIL import ImageDraw
#def make_dataframe(x, y, pixel_size, x_min, y_max):
#    """ Make a dataframe.
#    """
#    # Get row and col index
#    x, y = x.flatten(), y.flatten()
#    # Remove nan values
#    index = ~np.isnan(x)
#    x, y = x[index], y[index]
#    # Initialize dataframe
#    df = pd.DataFrame()
#    J = ((x-x_min)/pixel_size).astype(np.int16)
#    I = ((y_max-y)/pixel_size).astype(np.int16)
#    df['I'], df['J'] = I, J
#
#
#    x_min = np.floor(x.min()/pixel_size)*pixel_size
#    y_max = np.ceil(y.max()/pixel_size)*pixel_size
##    df = make_dataframe(x, y, pixel_size, x_min, y_max)
#    I_MAX = df.I.max()+1
#    J_MAX = df.J.max()+1
#
#    # Find which pixels need to be interpolated
#    polygon1 = []
#    tmp_df = pd.DataFrame(dict(I=df.I, J=df.J))
#    J_min = tmp_df.groupby('I')['J'].min() # Left boundary of the image
#    for i, j in zip(J_min.index, J_min.values):
#        polygon1.append(tuple([j, i]))
#    J_max = tmp_df.groupby('I')['J'].max() # Right boundary of the image
#    for i, j in zip(J_max.index.values[::-1], J_max.values[::-1]):
#        polygon1.append(tuple([j, i]))
#    mask1 = Image.new('L', (J_MAX, I_MAX), 0)
#    ImageDraw.Draw(mask1).polygon(polygon1, outline=0, fill=1)
#    mask1 = np.array(mask1).astype(np.bool)
#    plt.imshow(mask1)
#    plt.show()
#
#    return df

#def create_glt(igm_image_file, pixel_size, glt_image_file):
#    """ Create a geographic lookup table (GLT) image. The GLT image consists of two bands:
#            Sample Lookup:
#                Pixel values indicate the column (sample) number of the pixel
#                in the input geometry file that belongs at the given Y location in the output image.
#            Line Lookup:
#                Pixel values indicate the row (line) number of the pixel in the input geometry file
#                that belongs at the given X location in the output image.
#    Arguments:
#        igm_file: str
#            Input geometry filename.
#        pixel_size: float
#            Output image pixel size.
#    """
#    # Read IGM map x and y values
#    igm_header = read_envi_header(igm_image_file+'.hdr')
#    igm_image = np.memmap(igm_image_file,
#                          dtype=np.float32,
#                          mode='r',
#                          offset=0,
#                          shape=(igm_header['bands'], igm_header['lines'], igm_header['samples']))
#    x = igm_image[0,:,:].copy()
#    y = igm_image[1,:,:].copy()
#    del igm_image
#    # Make a dataframe
#    x_min = np.floor(x.min()/pixel_size)*pixel_size
#    y_max = np.ceil(y.max()/pixel_size)*pixel_size
#    df = make_dataframe(x, y, pixel_size, x_min, y_max)
#    I_MAX = df.I.max()+1
#    J_MAX = df.J.max()+1
#
#    # Find which pixels need to be interpolated
#    polygon1 = []
#    tmp_df = pd.DataFrame(dict(I=df.I, J=df.J))
#    J_min = tmp_df.groupby('I')['J'].min() # Left boundary of the image
#    for i, j in zip(J_min.index, J_min.values):
#        polygon1.append(tuple([j, i]))
#    J_max = tmp_df.groupby('I')['J'].max() # Right boundary of the image
#    for i, j in zip(J_max.index.values[::-1], J_max.values[::-1]):
#        polygon1.append(tuple([j, i]))
#    mask1 = Image.new('L', (J_MAX, I_MAX), 0)
#    ImageDraw.Draw(mask1).polygon(polygon1, outline=0, fill=1)
#    mask1 = np.array(mask1).astype(np.bool)
#    plt.imshow(mask1)
#    plt.show()
#
#    polygon2 = []
#    I_min = tmp_df.groupby('J')['I'].min() # Top boundary of the image
#    for j, i in zip(I_min.index, I_min.values):
#        polygon2.append(tuple([j, i]))
#    I_max = tmp_df.groupby('J')['I'].max() # Buttom boundary of the image
#    for j, i in zip(I_max.index.values[::-1], I_max.values[::-1]):
#        polygon2.append(tuple([j, i]))
#    mask2 = Image.new('L', (J_MAX, I_MAX), 0)
#    ImageDraw.Draw(mask2).polygon(polygon2, outline=0, fill=1)
#    mask2 = np.array(mask2).astype(np.bool)
#    plt.imshow(mask2)
#    plt.show()
#
#    mask = mask1*mask2
##    del mask1, mask2
#    plt.imshow(mask)
#    plt.show()
#    I, J = np.where(mask)
#    mask = zip(I, J)
#
#    # Find nearest points
#    df['distance'] = (df.i-df.I).abs() + (df.j-df.J).abs()
#    df['min_distance'] = df.groupby(['I','J'])['distance'].transform('min')
#    df = df[df.min_distance == df.distance]
#
#    glt_image = np.memmap(glt_image_file,
#                          dtype=np.int16,
#                          offset=0,
#                          mode='w+',
#                          shape=(2, I_MAX, J_MAX))
#    glt_image[...] = -1
#    glt_image[0, df.I, df.J] = df.row
#    glt_image[1, df.I, df.J] = df.col
#    del glt_image
#
#    glt_header = empty_envi_header()
#    glt_header['description'] = 'GLT'
#    glt_header['samples'] = J_MAX
#    glt_header['lines'] = I_MAX
#    glt_header['bands'] = 2
#    glt_header['byte order'] = 0
#    glt_header['header offset'] = 0
#    glt_header['interleave'] = 'bsq'
#    glt_header['data type'] = 2
#    glt_header['band names'] = ['Image Row', 'Image Column']
#    glt_header_file = glt_image_file+'.hdr'
#    write_envi_header(glt_header, glt_header_file)


#import pandas as pd
#from PIL import Image
#from PIL import ImageDraw
#
#def initialize_df(x, y, pixel_size, x_min, y_max):
#    # Get row and col index
#    row, col = np.indices(x.shape)
#    row, col = row.flatten(), col.flatten()
#    x, y = x.flatten(), y.flatten()
#    # Remove nan values
#    index = ~np.isnan(x)
#    x, y = x[index], y[index]
#    row, col = row[index], col[index]
#    # Initialize dataframe
#    df = pd.DataFrame()
#    df['row'], df['col'] = row, col
#    x_min, y_max = np.floor(x.min()/pixel_size)*pixel_size, np.ceil(y.max()/pixel_size)*pixel_size
#    j = (x-x_min)/pixel_size
#    i = (y_max-y)/pixel_size
#    J = j.astype('uint16')
#    I = i.astype('uint16')
#    df['i'], df['j'] = i, j
#    df['I'], df['J'] = I, J
#    return df, x_min, y_max
#
#def make_dataframe(x, y, pixel_size, x_min, y_max):
#    """ Make a dataframe.
#    """
#    # Get row and col index
#    row, col = np.indices(x.shape)
#    row, col = row.flatten(), col.flatten()
#    x, y = x.flatten(), y.flatten()
#    # Remove nan values
#    index = ~np.isnan(x)
#    x, y = x[index], y[index]
#    row, col = row[index], col[index]
#    # Initialize dataframe
#    df = pd.DataFrame()
#    df['row'], df['col'] = row, col
#    j = (x-x_min)/pixel_size
#    i = (y_max-y)/pixel_size
#    J = j.astype('int16')
#    I = i.astype('int16')
#    df['i'], df['j'] = i, j
#    df['I'], df['J'] = I, J
#    return df
#
#
#def create_glt(igm_image_file, pixel_size, glt_image_file):
#    """ Create a geographic lookup table (GLT) image. The GLT image consists of two bands:
#            Sample Lookup:
#                Pixel values indicate the column (sample) number of the pixel
#                in the input geometry file that belongs at the given Y location in the output image.
#            Line Lookup:
#                Pixel values indicate the row (line) number of the pixel in the input geometry file
#                that belongs at the given X location in the output image.
#    Arguments:
#        igm_file: str
#            Input geometry filename.
#        pixel_size: float
#            Output image pixel size.
#    """
#    # Read IGM map x and y values
#    igm_header = read_envi_header(igm_image_file+'.hdr')
#    igm_image = np.memmap(igm_image_file,
#                          dtype=np.float32,
#                          mode='r',
#                          offset=0,
#                          shape=(igm_header['bands'], igm_header['lines'], igm_header['samples']))
#    x = igm_image[0,:,:].copy()
#    y = igm_image[1,:,:].copy()
#    del igm_image
#    # Make a dataframe
#    x_min = np.floor(x.min()/pixel_size)*pixel_size
#    y_max = np.ceil(y.max()/pixel_size)*pixel_size
#    df = make_dataframe(x, y, pixel_size, x_min, y_max)
#    I_MAX = df.I.max()+1
#    J_MAX = df.J.max()+1
#
#    # Find which pixels need to be interpolated
#    polygon1 = []
#    tmp_df = pd.DataFrame(dict(I=df.I, J=df.J))
#    J_min = tmp_df.groupby('I')['J'].min() # Left boundary of the image
#    for i, j in zip(J_min.index, J_min.values):
#        polygon1.append(tuple([j, i]))
#    J_max = tmp_df.groupby('I')['J'].max() # Right boundary of the image
#    for i, j in zip(J_max.index.values[::-1], J_max.values[::-1]):
#        polygon1.append(tuple([j, i]))
#    mask1 = Image.new('L', (J_MAX, I_MAX), 0)
#    ImageDraw.Draw(mask1).polygon(polygon1, outline=0, fill=1)
#    mask1 = np.array(mask1).astype(np.bool)
#    plt.imshow(mask1)
#    plt.show()
#
#    polygon2 = []
#    I_min = tmp_df.groupby('J')['I'].min() # Top boundary of the image
#    for j, i in zip(I_min.index, I_min.values):
#        polygon2.append(tuple([j, i]))
#    I_max = tmp_df.groupby('J')['I'].max() # Buttom boundary of the image
#    for j, i in zip(I_max.index.values[::-1], I_max.values[::-1]):
#        polygon2.append(tuple([j, i]))
#    mask2 = Image.new('L', (J_MAX, I_MAX), 0)
#    ImageDraw.Draw(mask2).polygon(polygon2, outline=0, fill=1)
#    mask2 = np.array(mask2).astype(np.bool)
#    plt.imshow(mask2)
#    plt.show()
#
#    mask = mask1*mask2
##    del mask1, mask2
#    plt.imshow(mask)
#    plt.show()
#    I, J = np.where(mask)
#    mask = zip(I, J)
#
#    # Find nearest points
#    df['distance'] = (df.i-df.I).abs() + (df.j-df.J).abs()
#    df['min_distance'] = df.groupby(['I','J'])['distance'].transform('min')
#    df = df[df.min_distance == df.distance]
#
#    glt_image = np.memmap(glt_image_file,
#                          dtype=np.int16,
#                          offset=0,
#                          mode='w+',
#                          shape=(2, I_MAX, J_MAX))
#    glt_image[...] = -1
#    glt_image[0, df.I, df.J] = df.row
#    glt_image[1, df.I, df.J] = df.col
#    del glt_image
#
#    glt_header = empty_envi_header()
#    glt_header['description'] = 'GLT'
#    glt_header['samples'] = J_MAX
#    glt_header['lines'] = I_MAX
#    glt_header['bands'] = 2
#    glt_header['byte order'] = 0
#    glt_header['header offset'] = 0
#    glt_header['interleave'] = 'bsq'
#    glt_header['data type'] = 2
#    glt_header['band names'] = ['Image Row', 'Image Column']
#    glt_header_file = glt_image_file+'.hdr'
#    write_envi_header(glt_header, glt_header_file)

input_dir = "%s/Dropbox/rs/bhi/data/l1_gcp" % home

dates = [ '20180516','20180604',
         '20180629','20180707','20180718',
         '20180725','20180813','20180910',
         '20180926','20181017','20181110']

for date in dates:

    output_dir ="%s/%s/geo_offset/" % (input_dir,date)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)


    print(date)
    images = glob.glob("%s/%s/offsets/*offset.tif" % (input_dir,date))
    images.sort()
    for image in images:

        try:
            gps_file =image.replace("_geo_offset.tif",".txt").replace('/offsets/',"/gps/")

            gps = pd.read_csv(gps_file,sep = '\t',index_col = None,header = None)
            gps.columns  = ['ext',"lon","lat","alt","roll","pitch","heading","time"]
            gps.index = gps.time
            gps.sort_index(inplace=True)
            gps.index =range(len(gps))

            slopex = gdal.Open("%s_slope_%s.tif" % (image[:-4],'x'))
            slopex = slopex.GetRasterBand(1).ReadAsArray()
            slopey = gdal.Open("%s_slope_%s.tif" % (image[:-4],'y'))
            slopey = slopey.GetRasterBand(1).ReadAsArray()

            offset = gdal.Open(image)

            source = gdal.Open(image.replace('/offsets/',"/geo_gdal/").replace('_offset','_gdal'))
            source_map = gdal.Open(image.replace('/offsets/',"/geo_gdal/").replace('_geo_offset','_map'))

            source_igm = ht.openENVI(image.replace('/offsets/',"/geo/").replace('_offset.tif','')+"_igm.bsq")
            source_igm.load_data()

            map_px =  source_map.GetRasterBand(1).ReadAsArray()
            map_line =  source_map.GetRasterBand(2).ReadAsArray()
            map_px[map_px<0]*=-1
            map_line[map_line<0]*=-1


            roll_dict= gps.roll.to_dict()
            roll_dict[-1] = 0
            roll = np.vectorize(roll_dict.get)(map_line)
            pitch_dict= gps.pitch.to_dict()
            pitch_dict[-1] = 0
            pitch = np.vectorize(pitch_dict.get)(map_line)
            head_dict= gps.heading.to_dict()
            head_dict[-1] = 0
            heading = np.sin(np.radians(np.vectorize(head_dict.get)(map_line)))

            alt_dict= gps.alt.to_dict()
            alt_dict[-1] = 0
            alt = np.vectorize(alt_dict.get)(map_line)

            x_sr,a,c,y_sr,c,d = source.GetGeoTransform()

            # Get red band
            red = source.GetRasterBand(1).ReadAsArray()
            ir = source.GetRasterBand(4).ReadAsArray()
            ndvi = (ir-red)/(ir+red)

            off_y = offset.GetRasterBand(1).ReadAsArray().astype(float)
            off_x = offset.GetRasterBand(2).ReadAsArray().astype(float)

            off_x[off_x  == -99] = np.nan
            off_x[slopex >1] = np.nan
            labels_x = measure.label(off_x,connectivity =1)
            for label in np.unique(labels_x):
                class_mask = labels_x==label
                if class_mask.sum() < 300:
                    off_x[class_mask] = np.nan
                if ndvi[class_mask].mean() < .2:
                    off_x[class_mask] = np.nan

            off_y[off_y  == -99] = np.nan
            off_y[slopey >1] = np.nan
            labels_y = measure.label(off_y,connectivity =1)
            for label in np.unique(labels_y):
                class_mask = labels_y==label
                if class_mask.sum() < 300:
                    off_y[class_mask] = np.nan
                if ndvi[class_mask].mean() < .2:
                    off_y[class_mask] = np.nan

            mask_y =  ~np.isnan(off_y)
            mask_x =  ~np.isnan(off_x)

            X = np.column_stack([map_px[mask_y],roll[mask_y],alt[mask_y]])
            y = off_y[mask_y]
            model_y = sm.OLS(y,sm.add_constant(X)).fit()

            y = off_x[mask_x]
            X = np.column_stack([map_px[mask_x],roll[mask_x],alt[mask_x]])
            model_x = sm.OLS(y,sm.add_constant(X)).fit()

            map_line_raw,map_px_raw = np.indices(source_igm.data[0].shape)
            new_x  = np.copy(source_igm.data[0])
            new_y  = np.copy(source_igm.data[1])
            new_x[new_x<0]*=-1
            new_y[new_y<0]*=-1

            flip = gps.loc[0,'lat'] < gps.loc[gps.index.max(),'lat']
            if flip:
                map_line = np.flipud(map_line)
            else:
                map_px = np.fliplr(map_px)


            roll_raw = np.vectorize(roll_dict.get)(map_line_raw)
            pitch_raw = np.vectorize(pitch_dict.get)(map_line_raw)
            heading_raw = np.sin(np.radians(np.vectorize(head_dict.get)(map_line_raw)))
            alt_raw = np.vectorize(alt_dict.get)(map_line_raw)

            i,a,b,c  = model_y.params
            y_offset = i + a*map_px_raw + b * roll_raw + c * alt_raw

            i,a,b,c  = model_x.params
            x_offset = i + a*map_px_raw + b * roll_raw + c * alt_raw

            new_y-=y_offset
            new_x+=x_offset

            source_rad = ht.openENVI(image.replace('/offsets/',"/rad/").replace('_geo_offset.tif',''))
            source_rad.load_data()

            header = source_rad.header_dict
            header['bands'] = 1
            header['lines'] =new_y.shape[0]
            header['samples'] =new_y.shape[1]
            header['data type'] = 4

            northing = "%s_northing"  % image.replace('/offsets/',"/geo_offset/").replace('_geo_offset.tif','')
            writer = ht.file_io.writeENVI(northing, header)
            writer.write_band(new_y,0)
            writer.close()

            easting = "%s_easting"  % image.replace('/offsets/',"/geo_offset/").replace('_geo_offset.tif','')
            writer = ht.file_io.writeENVI(easting, header)
            writer.write_band(new_x,0)
            writer.close()


            for coord in [northing,easting]:
                latlongData = gdal.Open(coord)

                ########################################
                ##CREATE A VRT FOR THE LATITUDE AND LONGITUDE RASTER

                #create a new file to write the VRT
                latlonVRT = open("%s.vrt" % coord,'w')

                #Begin writing XML tags........
                #More information about the meaning of VRT XML tags can be found here: http://www.gdal.org/gdal_vrttut.html
                #but most tags are self explanatory

                latlonVRT.write('<VRTDataset rasterxsize="%s" rasterysize="%s">\n' % (latlongData.RasterXSize,latlongData.RasterYSize))

                #write two bands to the VRT (latitude and longitude)
                for bandNum in [1]:

                    #opent the current band
                    band =latlongData.GetRasterBand(bandNum)

                    #Continue writing XML tags for the current band
                    latlonVRT.write('<VRTRasterBand dataType="%s" band="%s">\n' % ("Float32",bandNum))
                    latlonVRT.write('<SimpleSource>\n')
                    latlonVRT.write('<SourceFilename relativeToVRT="1">%s</SourceFilename>' % coord)
                    latlonVRT.write('<SourceBand>%s</SourceBand>\n' % bandNum)
                    latlonVRT.write('<SourceProperties RasterXSize="%s" RasterYSize="%s" DataType="%s" BlockXSize="%s" BlockYSize="%s" />'  % (band.XSize,band.YSize,"Float32",band.GetBlockSize()[0],band.GetBlockSize()[1]))
                    latlonVRT.write('<SrcRect xOff="0" yOff="0" xSize="%s" ySize="%s" />' % (band.XSize,band.YSize))
                    latlonVRT.write('<DstRect xOff="0" yOff="0" xSize="%s" ySize="%s" />' % (band.XSize,band.YSize))
                    latlonVRT.write("</SimpleSource>\n")
                    latlonVRT.write("</VRTRasterBand>\n")

                #Write closing tag
                latlonVRT.write("</VRTDataset>\n")

                #close VRT file
                latlonVRT.close()


            #########################################
            ##CREATE A VRT FOR THE RASTER TO BE WARPED

            output_dir =os.path.split(image)[0].replace('/offsets',"/geo_offset/")
            directory = os.path.split(image)[0].replace('/offsets',"/rad/")
            basename = os.path.splitext(os.path.basename(image.replace('/offset/',"/rad/").replace('_geo_offset','')))[0]

            #open the GDAL raster data
            baseData= gdal.Open(directory+basename)

            #create a new file to write the VRT
            baseVRT = open(directory +"%s.vrt" % basename,'w')

            #Begin writing XML tags
            baseVRT.write('<VRTDataset rasterXSize="%s" rasterYSize="%s">\n' % (baseData.RasterXSize,baseData.RasterYSize))
            baseVRT.write('<Metadata Domain = "GEOLOCATION">\n')
            baseVRT.write('<mdi key="X_DATASET">%s</mdi>' % easting)
            baseVRT.write('<mdi key="X_BAND">1</mdi>')
            baseVRT.write('<mdi key="Y_DATASET">%s</mdi>' % northing)
            baseVRT.write('<mdi key="Y_BAND">1</mdi>')
            baseVRT.write('<mdi key="PIXEL_OFFSET">0</mdi><mdi key="LINE_OFFSET">0</mdi><mdi key="PIXEL_STEP">1</mdi> <mdi key="LINE_STEP">1</mdi>')
            baseVRT.write('</Metadata>\n')

            #cycle through each band in the rasterbandNum
            for bandNum in range(1,baseData.RasterCount+1):

                #open the current band
                band = baseData.GetRasterBand(bandNum)

                #Continue writing XML tags for the current band
                baseVRT.write('<VRTRasterBand dataType="%s" band="%s">\n' % ("Float32",bandNum))
                baseVRT.write('<SimpleSource>\n')
                baseVRT.write('<SourceFilename relativeToVRT="1">%s</SourceFilename>' % basename)
                baseVRT.write('<SourceBand>%s</SourceBand>\n' % bandNum)
                baseVRT.write('<SourceProperties RasterXSize="%s" RasterYSize="%s" DataType="%s" BlockXSize="%s" BlockYSize="%s" />'  % (band.XSize,band.YSize,"Int32",band.GetBlockSize()[0],band.GetBlockSize()[1]))
                baseVRT.write('<SrcRect xOff="0" yOff="0" xSize="%s" ySize="%s" />' % (band.XSize,band.YSize))
                baseVRT.write('<DstRect xOff="0" yOff="0" xSize="%s" ySize="%s" />' % (band.XSize,band.YSize))
                baseVRT.write("</SimpleSource>\n")
                baseVRT.write("</VRTRasterBand>\n")

            #Write closing tag
            baseVRT.write("</VRTDataset>\n")
            #close VRT file
            baseVRT.close()
            resolution = .25

            output_filename =  "%s_geo_corr.tif" % (output_dir+ os.path.splitext(os.path.basename(basename))[0])


            if os.path.isfile(output_filename):
                os.remove(output_filename)

            os.system("gdalwarp -multi -geoloc -t_srs EPSG:32616 -tap -tr %s %s -r near %s.vrt %s"  %  (resolution,resolution,directory+basename,output_filename))

            dstFile_1m = output_filename.replace(".tif","_1m.tif")
            if os.path.isfile(dstFile_1m):
                os.remove(dstFile_1m)

            os.system("gdalwarp  -tr 1 1 -tap -r average %s %s"  %  (output_filename,dstFile_1m))

            os.remove(output_filename)



            #########################################
            ##CREATE A VRT FOR THE SCAN ANGLE RASTER TO BE WARPED
            out_directory = os.path.split(image)[0].replace('/offsets',"/geo_gdal/")
            directory = os.path.split(image)[0].replace('/offsets',"/geo/")
            basename =os.path.basename(image.replace('/offset/',"/geo/").replace('_offset.tif','_sca.bsq'))
            #open the GDAL raster data
            baseData= gdal.Open(directory+basename)
            #create a new file to write the VRT
            baseVRT = open(directory +"%s.vrt" % basename,'w')

            #Begin writing XML tags
            baseVRT.write('<VRTDataset rasterXSize="%s" rasterYSize="%s">\n' % (baseData.RasterXSize,baseData.RasterYSize))
            baseVRT.write('<Metadata Domain = "GEOLOCATION">\n')
            baseVRT.write('<mdi key="X_DATASET">%s</mdi>' % easting)
            baseVRT.write('<mdi key="X_BAND">1</mdi>')
            baseVRT.write('<mdi key="Y_DATASET">%s</mdi>' % northing)
            baseVRT.write('<mdi key="Y_BAND">1</mdi>')
            baseVRT.write('<mdi key="PIXEL_OFFSET">0</mdi><mdi key="LINE_OFFSET">0</mdi><mdi key="PIXEL_STEP">1</mdi> <mdi key="LINE_STEP">1</mdi>')
            baseVRT.write('</Metadata>\n')

            #cycle through each band in the rasterbandNum
            for bandNum in range(1,baseData.RasterCount+1):
                #open the current band
                band = baseData.GetRasterBand(bandNum)
                #Continue writing XML tags for the current band
                baseVRT.write('<VRTRasterBand dataType="%s" band="%s">\n' % ("Int32",bandNum))
                baseVRT.write('<SimpleSource>\n')
                baseVRT.write('<SourceFilename relativeToVRT="1">%s</SourceFilename>' % basename)
                baseVRT.write('<SourceBand>%s</SourceBand>\n' % bandNum)
                baseVRT.write('<SourceProperties RasterXSize="%s" RasterYSize="%s" DataType="%s" BlockXSize="%s" BlockYSize="%s" />'  % (band.XSize,band.YSize,"Int32",band.GetBlockSize()[0],band.GetBlockSize()[1]))
                baseVRT.write('<SrcRect xOff="0" yOff="0" xSize="%s" ySize="%s" />' % (band.XSize,band.YSize))
                baseVRT.write('<DstRect xOff="0" yOff="0" xSize="%s" ySize="%s" />' % (band.XSize,band.YSize))
                baseVRT.write('<NoData>0</NoData>')
                baseVRT.write("</SimpleSource>\n")
                baseVRT.write("</VRTRasterBand>\n")

            #Write closing tag
            baseVRT.write("</VRTDataset>\n")
            #close VRT file
            baseVRT.close()

            resolution = .25
            output_filename =  "%s_geo_corr.tif" % (output_dir+ os.path.splitext(os.path.basename(basename))[0])

            if os.path.isfile(output_filename):
                os.remove(output_filename)

            os.system("gdalwarp -multi -geoloc -t_srs EPSG:32616 -tap -tr %s %s -r near %s.vrt %s"  %  (resolution,resolution,directory+basename,output_filename))

            dstFile_1m = output_filename.replace(".tif","_1m.tif")
            if os.path.isfile(dstFile_1m):
                os.remove(dstFile_1m)

            os.system("gdalwarp  -tr 1 1 -tap -r average %s %s"  %  (output_filename,dstFile_1m))

            os.remove(output_filename)




        except:
            print("ERROR: %s" % image)
            continue














