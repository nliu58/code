""" Functions to do radiometric calibration.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import os
import numpy as np
from scipy import constants, signal
from envi import read_envi_header, empty_envi_header, write_envi_header
import matplotlib.pyplot as plt
from atmosphere import read_metadata
from spectral import get_resampling_coeff, estimate_fwhms_from_waves

solar_flux_file = './solar_flux.dat'

features = {429:  [420,  445],
            517:  [500,  540],
            762:  [744,  784],
            820:  [805,  835],
            940:  [920,  970],
            1130: [1100, 1160],
            1268: [1255, 1285],
            1470: [1450, 1490],
            2004: [1985, 2030],
            2055: [2040, 2080],
            2317: [2300, 2330],
            2420: [2400, 2435]}

def remove_smile_effect(config, atm_lut_file, sun_zenith):
    # Read calibration data
    cal_data = get_cal_data(config['raw_image_file'],
                            config['setting_file'])

    # Read raw image data
    raw_header = read_envi_header(os.path.splitext(config['raw_image_file'])[0]+'.hdr')
    raw_image = np.memmap(config['raw_image_file'],
                          dtype='uint16',
                          mode='r',
                          offset=raw_header['header offset'],
                          shape=(raw_header['lines'], raw_header['bands'], raw_header['samples']))

    # Read mask
    mask_header = read_envi_header(os.path.splitext(config['raw_image_file'])[0]+'.hdr')
    mask_image = np.memmap(config['mask_image_file'],
                           dtype='bool',
                           mode='r',
                           shape=(mask_header['bands'], raw_header['lines'], raw_header['samples']))

    # Get average radiance spectra with dark current corrected
    avg_rdn_01 = get_avg_rdn_01(raw_image, cal_data, mask_image)

    # Get average radiance spectra with dark current and gain corrected
    avg_rdn_02 = get_avg_rdn_02(avg_rdn_01, cal_data)

    plt.plot(cal_data['spectralVector'], avg_rdn_02, '-')
    plt.show()

    # Get average scan angles
    avg_vza, avg_raa = get_avg_scan_angles(config['sca_image_file'])

    # Build water vapor column estimation model
    """ Notes:
        (1) The view geometry does not seem to affect the model. Therefore, a vza=0 and raa=0 are used here.
    """
    cwv_model = build_wvc_model(config['wvc_model_file'], atm_lut_file, cal_data['spectralVector'], cal_data['fwhm'], 0, 0)

    # Estimate water vapor for each column
    ratios = avg_rdn_02[cwv_model['bands'][1], :]/(avg_rdn_02[cwv_model['bands'][0], :]*cwv_model['weights'][0]+avg_rdn_02[cwv_model['bands'][2], :]*cwv_model['weights'][1])
    wvcs = np.interp(ratios, cwv_model['Ratio'], cwv_model['WVC'])

    plt.plot(cwv_model['Ratio'], cwv_model['WVC'], '.-', color='red')
    plt.plot(ratios, wvcs, '.', color='blue')
    plt.show()
    cwv_model['bands']

    rdn = raw2rdn(raw_image[:,134,:], cal_data, 134)

    tmp_mask = rdn<0

    raw_image.flush()
    mask_image.flush()

    for band in range(mask_image.shape[0]):
        print(np.sum(mask_image[band,:,:])/mask_image[band,:,:].size)

def build_mask(mask_image_file, raw_image_file, setting_file, sun_zenith):
    """ Mask out bad pixels.
    Arguments:
        mask_image_file: str
            Mask image filename.
        raw_image: 3D array
            Raw DN image, dimension = [lines, bands, samples]
        cal_data: dict
            Calibration data.
        sun_zenith: float
            Sun zenith angle in degrees.
    """

    # Read calibration data
    cal_data = get_cal_data(raw_image_file, setting_file)

    # Read raw image data
    raw_header = read_envi_header(os.path.splitext(raw_image_file)[0]+'.hdr')
    raw_image = np.memmap(raw_image_file,
                          dtype='int16',
                          mode='r',
                          offset=raw_header['header offset'],
                          shape=(raw_header['lines'], raw_header['bands'], raw_header['samples']))

    """ Rule 1 for good pixels:
        At-sensor blue reflectance>0.10, and nir reflectance>0.10, and swir reflectance>0.10
    """
    solar_flux = resample_solar_flux(solar_flux_file, cal_data['spectralVector'], cal_data['fwhm'])
    solar_flux /= 10 # mW / (m2 nm) -> mW / (cm2 um)
    cos_sun_zenith = np.cos(np.deg2rad(sun_zenith))
    mask = np.full((raw_image.shape[0], raw_image.shape[2]), True, dtype=np.bool)
    # if the reflectance at 470 nm is less than 0.10, then mask these pixels.
    wave, band = get_closest_wave(cal_data['spectralVector'], 470)
    if np.abs(wave-470)<10:
        refl = raw2rdn(raw_image[:,band,:], cal_data, band)*np.pi/(solar_flux[band]*cos_sun_zenith)
        mask &= (refl>0.10)
    # if the reflectance at 850 nm is less than 0.10, then mask these pixels.
    wave, band = get_closest_wave(cal_data['spectralVector'], 850)
    if np.abs(wave-850)<10:
        refl = raw2rdn(raw_image[:,band,:], cal_data, band)*np.pi/(solar_flux[band]*cos_sun_zenith)
        mask &= (refl>0.10)
    # if the reflectance at 1600 nm is less than 0.10, then mask these pixels.
    wave, band = get_closest_wave(cal_data['spectralVector'], 1600)
    if np.abs(wave-1600)<10:
        refl = raw2rdn(raw_image[:,band,:], cal_data, band)*np.pi/(solar_flux[band]*cos_sun_zenith)
        mask &= (refl>0.10)

    """ Rule 2:
        Mask dark pixels by using radiance values.
    """
    mask_image = np.memmap(mask_image_file,
                           dtype='uint8',
                           mode='w+',
                           shape=(raw_image.shape[1], raw_image.shape[0], raw_image.shape[2]))
    for band in range(raw_image.shape[1]):
        rdn = raw2rdn(raw_image[:,band,:], cal_data, band)
        mask_image[band,:,:] = mask&(raw_image[:,band,:]<cal_data['satValue'])&(rdn>0.0)
        del rdn
    mask_image.flush()
    raw_image.flush()
    del cal_data, mask

    mask_header = empty_envi_header()
    mask_header['description'] = 'Mask 0: bad; 1: good'
    mask_header['samples'] = raw_image.shape[2]
    mask_header['lines'] = raw_image.shape[0]
    mask_header['bands'] = raw_image.shape[1]
    mask_header['byte order'] = 0
    mask_header['header offset'] = 0
    mask_header['interleave'] = 'bsq'
    mask_header['data type'] = 1
    write_envi_header(mask_image_file+'.hdr', mask_header)

def nood_transform(spectra):
    """NOOD transformation of spectra.
    References:
        Analysis of Hyperion Data with the FLAASH Atmospheric Correction Algorithm.
    Arguments:
        spectra: 1D array
            Radiance specta, dimension: [Bands].
    Returns:
        Transformed spectra.
    """
    spectra = -np.log(spectra)
    spectra = np.diff(spectra)
    spectra = np.append(spectra, spectra[-1])
    spectra = (spectra-spectra.mean())/spectra.std()

    return spectra

def get_avg_scan_angles(sca_image_file):
    """ Get average scan angles along each column.
    Arguments:
        sca_image_file: str
            Sensor view angle image filename.
    Returns:
        avg_vza, avg_raa: 1D array
            Averaged view zenith angle, averaged relative azimuth angle.
    """

    # Read scan angles
    sca_header = read_envi_header(sca_image_file+'.hdr')
    sca_image = np.memmap(sca_image_file,
                          dtype='float32',
                          mode='r',
                          offset=0,
                          shape=(sca_header['bands'], sca_header['lines'], sca_header['samples']))

    # Get average sensor view zenith angle along each column
    avg_vza = sca_image[0,:,:].mean(axis=0)

    # Get average relative azimuth angle along each column
    saa = float(sca_header['sun azimuth'])
    raa = saa-sca_image[1,:,:]
    raa[raa<0] += 360.0
    raa[raa>180] = 360.0-raa[raa>180]
    avg_raa = raa.mean(axis=0)

    sca_image.flush()
    del saa, raa

    return avg_vza, avg_raa

def build_wvc_model(wvc_model_figure_file, atm_lut_file, sensor_waves, sensor_fwhms, vza=0, raa=0):
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
        vza, raa: float
            View zenith angle, and relative azimuth angle in degrees.
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
    """ Notes: atm_lut dimensions: [RHO, CWV, VIS, VZA, RAA, WAVE]
    In general, we fix Rho=0.5, VIS=80.0, VZA=0.0, RAA=0.0
    """
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
    ratios = resampled_rdn[:,1]/(left_weight*resampled_rdn[:,0]+right_weight*resampled_rdn[:,2])

    # Save depths and wvc to the model
    index = np.argsort(ratios)
    wvc_model['WVC'] = list(atm_lut_metadata['WVC'][index])
    wvc_model['Ratio'] = list(ratios[index])

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

def get_avg_rdn_01(raw_image, cal_data, mask_image):
    """ Average radiance along each column (only dark current corrected!).
    Arguments:
        raw_image: 3D array
            Raw DN image data, dimensions: [lines, bands, samples].
        mask_image: 3D array
            Mask image, dimensions: [bands, lines, samples].
    Returns:
        avg_rdn_01: 2D array
            Averaged radiance along each column with dark current corrected, dimension: [bands, samples].
    """

    lines = raw_image.shape[0]
    bands = raw_image.shape[1]
    samples = raw_image.shape[2]

    serial_number = cal_data['serialNumber']
    number_of_frames = cal_data['numberOfFrames']
    avg_rdn_01 = np.full((bands, samples), 0, dtype=np.float32)

    for band in range(bands):
        background = cal_data['background'][band,:]
        if serial_number>=3000 and serial_number<=5000:
            offset = np.zeros((lines, samples))
            background_last = cal_data['backgroundLast'][band,:]
            for line in range(lines):
                offset[line,:] = background+(background_last-background)*line/number_of_frames
            del background_last
        else:
            offset = np.tile(background, (lines, 1))
        tmp = (raw_image[:,band,:]).astype('float')-offset # tmp.size = (lines, samples)
        tmp = np.ma.array(tmp, mask=~mask_image[band,:,:]).mean(axis=0) # tmp.size = (samples)
        for bad_sample in np.where(tmp.mask)[0]:
            if bad_sample==0:
                tmp[bad_sample] = tmp[bad_sample+1]
            elif bad_sample==samples-1:
                tmp[bad_sample] = tmp[bad_sample-1]
            else:
                tmp[bad_sample] = (tmp[bad_sample-1]+tmp[bad_sample+1])/2.0
        avg_rdn_01[band,:] = tmp
        del offset, tmp

    return avg_rdn_01

def get_avg_rdn_02(avg_rdn_01, cal_data):
    """ Average radiance along each column (both dark current and gain corrected!).
    Arguments:
        avg_rdn_01: 2D array
            Averaged radiance along each column with dark current corrected, dimension: [bands, samples].
        cal_data: dict
            Calibration data.
    Returns:
        avg_rdn_02: 2D array
            Averaged radiance along each column (both dark current and gain corrected!), dimension: [bands, samples].
    """
    samples = avg_rdn_01.shape[1]

    h = constants.Planck # Planck constant
    c = constants.c*1e+9 # Light speed nm/s
    RE = cal_data['RE'] # (bands, samples)
    QE = np.tile(cal_data['QE'], (samples,1)).transpose() # (bands)
    SF = cal_data['SF']
    integration_time = cal_data['integrationTime']
    aperture_area = np.pi*cal_data['apertureSize']*cal_data['apertureSize']
    solid_angle = cal_data['solidAngle']
    wavelength_interval = np.tile(cal_data['spectralSampling'], (samples, 1)).transpose() # (bands, samples)
    center_wavelength = np.tile(cal_data['spectralVector'], (samples, 1)).transpose() # (bands, samples)
    # Calculate gain
    gain = h*c*1e6/(RE*QE*SF*integration_time*aperture_area*solid_angle*wavelength_interval*center_wavelength)
    avg_rdn_02 = gain*avg_rdn_01*100 # in mW/(cm2*um)
    del gain
    # Smooth this radiance
    for avg_rdn in avg_rdn_02:
        relative_diff = signal.medfilt(avg_rdn)/(avg_rdn+1e-4)-1.0
        bad_columns = np.where(np.abs(relative_diff)>0.05)[0]
        for bad_column in bad_columns:
            if bad_column == 0:
                avg_rdn[bad_column] = avg_rdn[bad_column+1]
            elif bad_column == samples-1:
                avg_rdn[bad_column] = avg_rdn[bad_column-1]
            else:
                avg_rdn[bad_column] = (avg_rdn[bad_column-1]+avg_rdn[bad_column+1])/2.0

    return avg_rdn_02

def get_closest_wave(waves, center_wav):
    """ Get the band index whose wavelength is closest to `center_wav`.
    Arguments:
        waves: array
            Wavelength array.
        center_wav: float
            Center wavelength.
    Returns:
        Closest wavelength and its band index.
    """

    band_index = np.argmin(np.abs(np.array(waves)-center_wav))

    return waves[band_index], band_index

def raw2rdn(raw_image, cal_data, band_index):
    """ Convert raw DN values to physical radiance in mW/(cm2*um) for one band.
    Arguments:
        raw_image: array
            Single band raw DN image.
        cal_data: dict
            Calibration data.
        band_index: int
            Band index.
    """

    # Get calibration values
    lines = raw_image.shape[0]
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
    gain = h*c*1e6/(RE*QE*SF*integration_time*aperture_area*solid_angle*wavelength_interval*center_wavelength)
    gain = np.tile(gain, (lines, 1))

    # Calculate offset
    if serial_number >=3000 and serial_number<=5000:
        offset = np.zeros_like(gain)
        background_last = cal_data['backgroundLast'][band_index,:]
        for line in range(lines):
            offset[line,:] = background+(background_last-background)*line/number_of_frames
        del background_last
    else:
        offset = np.tile(background, (lines, 1))
    del background

    # Calibration
    rdn_image = ((raw_image.astype('float32')-offset)*gain).astype('float32')
    rdn_image = rdn_image*100.0 # in mW/(cm2*um)
    del gain, offset

    return rdn_image

def get_cal_data(image_file, setting_file=None):
    """ Read HySpex calibation data.
    Notes:
        (1) The code is adapted from Trond Løke's (trond@neo.no) script.
    Arguments:
        image_file: str
            HySpex image filename.
        setting_file: str
            Lab calibration filename.
    Returns:
        cal_data: dict
            HySpex calibration data.
    """

    cal_data = dict()
    try:
        fid = open(image_file, 'rb')
    except:
        raise IOError('Cannot open %s!' %(image_file))

    cal_data['word'] = np.fromfile(fid, dtype=np.int8, count=8)
    cal_data['hdrSize'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    cal_data['serialNumber'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    cal_data['configFile'] = np.fromfile(fid, dtype=np.int8, count=200)
    cal_data['settingFile'] = np.fromfile(fid, dtype=np.int8, count=120)

    cal_data['scalingFactor'] = np.fromfile(fid, dtype=np.float64, count=1)[0]
    cal_data['electronics'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['comsettingsElectronics'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['comportElectronics'] = np.fromfile(fid, dtype=np.int8, count=44)
    cal_data['fanSpeed'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    cal_data['backTemperature'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['Pback'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['Iback'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['Dback'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['comport'] = np.fromfile(fid, dtype=np.int8, count=64)

    cal_data['detectstring'] = np.fromfile(fid, dtype=np.int8, count=200)
    cal_data['sensor'] = np.fromfile(fid, dtype=np.int8, count=176)
    cal_data['temperature_end'] = np.fromfile(fid, dtype=np.float64, count=1)[0]
    cal_data['temperature_start'] = np.fromfile(fid, dtype=np.float64, count=1)[0]
    cal_data['temperature_calibration'] = np.fromfile(fid, dtype=np.float64, count=1)[0]

    cal_data['framegrabber'] = np.fromfile(fid, dtype=np.int8, count=200)
    cal_data['ID'] = np.fromfile(fid, dtype=np.int8, count=200)
    cal_data['supplier'] = np.fromfile(fid, dtype=np.int8, count=200)
    cal_data['leftGain'] = np.fromfile(fid, dtype=np.int8, count=32)
    cal_data['rightGain'] = np.fromfile(fid, dtype=np.int8, count=32)

    cal_data['comment'] = np.fromfile(fid, dtype=np.int8, count=200)
    cal_data['backgroundFile'] = np.fromfile(fid, dtype=np.int8, count=200)
    cal_data['recordHD'] = np.fromfile(fid, dtype=np.int8, count=1)
    cal_data['unknownPOINTER'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['serverIndex'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    cal_data['comsettings'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['numberOfBackground'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['spectralSize'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['spatialSize'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['binning'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    cal_data['detected'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['integrationTime'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['frameperiod'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['defaultR'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['defaultG'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    cal_data['defaultB'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['bitshift'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['temperatureOffset'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['shutter'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['backgroundPresent'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    cal_data['power'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['current'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['bias'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['bandwidth'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['vin'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    cal_data['vref'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['sensorVin'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['sensorVref'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['coolingTemperature'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['windowStart'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    cal_data['windowStop'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['readoutTime'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['p'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['i'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['d'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    cal_data['numberOfFrames'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['nobp'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['dw'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['EQ'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['lens'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]

    cal_data['FOVexp'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['scanningMode'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['calibAvailable'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['numberOfAvg'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['SF'] = np.fromfile(fid, dtype=np.float64, count=1)[0]

    cal_data['apertureSize'] = np.fromfile(fid, dtype=np.float64, count=1)[0]
    cal_data['pixelSizeX'] = np.fromfile(fid, dtype=np.float64, count=1)[0]
    cal_data['pixelSizeY'] = np.fromfile(fid, dtype=np.float64, count=1)[0]
    cal_data['temperature'] = np.fromfile(fid, dtype=np.float64, count=1)[0]
    cal_data['maxFramerate'] = np.fromfile(fid, dtype=np.float64, count=1)[0]

    cal_data['spectralCalibPOINTER'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    cal_data['REPOINTER'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    cal_data['QEPOINTER'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    cal_data['backgroundPOINTER'] = np.fromfile(fid, dtype=np.int32, count=1)[0]
    cal_data['badPixelsPOINTER'] = np.fromfile(fid, dtype=np.int32, count=1)[0]

    cal_data['imageFormat'] = np.fromfile(fid, dtype=np.uint32, count=1)[0]
    cal_data['spectralVector'] = np.fromfile(fid, dtype=np.float64, count=cal_data['spectralSize'])
    cal_data['QE'] = np.fromfile(fid, dtype=np.float64, count=cal_data['spectralSize'])
    cal_data['RE'] = np.fromfile(fid, dtype=np.float64, count=cal_data['spectralSize']*cal_data['spatialSize'])
    cal_data['RE'].shape = (cal_data['spectralSize'], cal_data['spatialSize'])
    cal_data['background'] = np.fromfile(fid, dtype=np.float64, count=cal_data['spectralSize']*cal_data['spatialSize'])
    cal_data['background'].shape = (cal_data['spectralSize'], cal_data['spatialSize'])

    cal_data['badPixels'] = np.fromfile(fid, dtype=np.uint32, count=cal_data['nobp'])

    if cal_data['serialNumber'] > 3000:
        cal_data['backgroundLast'] = np.fromfile(fid, dtype=np.float64, count=cal_data['spectralSize']*cal_data['spatialSize'])
        cal_data['backgroundLast'].shape = (cal_data['spectralSize'], cal_data['spatialSize'])

    fid.close()

    def from_int8array_to_string(int8_array):
        string = ''
        for int8_value in int8_array:
            if int8_value == 0:
                break
            string += chr(int8_value)
        return string

    # Convert int8array to string
    for key in cal_data:
        if cal_data[key].dtype.name == 'int8':
            cal_data[key] = from_int8array_to_string(cal_data[key])


    # Read setting
    if setting_file:
        setting = get_hyspex_setting(setting_file)
        cal_data['QE'] = np.array(setting['QE'])
        cal_data['spectralVector'] = np.array(setting['spectral_calib'])
        cal_data['RE'] = np.array(setting['RE']).reshape((setting['spectral_size'], setting['spatial_size']))

    # Add other values
    cal_data['spectralSampling'] = np.diff(cal_data['spectralVector'])
    cal_data['spectralSampling'] = np.append(cal_data['spectralSampling'], cal_data['spectralSampling'][-1])
    cal_data['solidAngle'] = cal_data['pixelSizeX']*cal_data['pixelSizeY']
    cal_data['satValue'] = np.power(2, 16.0-cal_data['bitshift']) - 1
    cal_data['fwhm'] = estimate_fwhms_from_waves(cal_data['spectralVector'])

    return cal_data

def resample_solar_flux(solar_flux_file, sensor_waves, sensor_fwhms):
    """ Resample solar flux to sensor wavelengths.
    Arguments:
        solar_flux_file: str
            Solar flux filename.
        sensor_waves: array
            Sensor wavelengths.
        sensor_fwhms: array
            Sensor FWHMs.
    Returns:
        solar_flux: array
            Resampled solar flux.
    """

    solar_flux = np.loadtxt(solar_flux_file)
    sample_coeff = get_resampling_coeff(solar_flux[:,0], sensor_waves, sensor_fwhms)
    solar_flux = np.dot(solar_flux[:,1], sample_coeff)

    return solar_flux

def get_hyspex_setting(setting_file):
    """ Read Hyspex calibration setting data.
    Arguments:
        setting_file: str
            Hyspex setting filename.
    Returns:
        setting: dict
            Hyspex setting.
    """

    setting_value_type = {"serialnumber": "int",
                          "configfile": "str",
                          "serverindex": "int",
                          "RecordHD": "str",
                          "framegrabber": "str",
                          "comsettings_electronics": "int",
                          "comsettings": "int",
                          "electronics": "int",
                          "readout_time": "float",
                          "EQ": "float",
                          "ScanningMode": "int",
                          "CalibAvailable": "int",
                          "lens": "int",
                          "FOVexp": "int",
                          "number_of_background": "int",
                          "detectstring": "str",
                          "sensor": "str",
                          "ID": "str",
                          "supplier": "str",
                          "spectral_size": "int",
                          "spatial_size": "int",
                          "default_R": "int",
                          "default_G": "int",
                          "default_B": "int",
                          "bitshift": "int",
                          "binning": "int",
                          "window_start": "int",
                          "window_stop": "int",
                          "integrationtime": "int",
                          "frameperiod": "int",
                          "nobp": "int",
                          "dw": "int",
                          "shutter": "int",
                          "SF": "float",
                          "max_framerate": "float",
                          "aperture_size":"float",
                          "pixelsize_x": "float",
                          "pixelsize_y": "float",
                          "Temperature_Calibration": "float",
                          "AIM_gain": "int",
                          "AIM_midlevel": "int",
                          "vref": "int",
                          "vin": "int",
                          "sensor_vref": "int",
                          "sensor_vin": "int",
                          "spectral_calib": "list_float",
                          "RE": "list_float",
                          "QE": "list_float",
                          "bad_pixels": "list_int"}

    trans_table = str.maketrans("\n"," ")
    setting = dict()

    fid = open(setting_file, 'r')
    line = fid.readline()
    flag = True

    while line:
        if "=" in line:
            key, value = line.split("=", 1)

            # Add field if not in the default list
            if key.strip() not in setting_value_type.keys():
                setting_value_type[key.strip()] = 'str'

            # Keep reading if value is ''
            if value.strip() == '':
                line = fid.readline()
                while (not "=" in line) and (not line.strip() is ''):
                    value += line
                    line = fid.readline()
                flag = False

            # Extract values
            val_type = setting_value_type[key.strip()]
            if val_type == "list_float":
                tmp = value.translate(trans_table).strip().split(" ")
                value= np.array([float(x) for x in tmp])
            elif val_type == "list_int":
                tmp = value.translate(trans_table).strip()
                if tmp == '':
                    value = None
                else:
                    value= np.array([int(x) for x in tmp.split(" ")])
            elif val_type == "list_str":
                value= [x.strip() for x in value.translate(trans_table).strip().split(" ")]
            elif val_type == "int":
                value = int(value.translate(trans_table))
            elif val_type == "float":
                value = float(value.translate(trans_table))
            elif val_type == "str":
                value = value.translate(trans_table).strip().lower()

            setting[key.strip()] = value

        if flag is False:
            flag = True
        else:
            line = fid.readline()
    fid.close()

    return setting