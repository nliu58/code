""" Functions to do radiometric calibration.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import os, logging
import numpy as np
from scipy      import constants, signal, optimize
from envi       import read_envi_header, empty_envi_header, write_envi_header
from spectral   import resample_spectra, estimate_fwhms_from_waves, get_closest_wave
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

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
    from atmosphere import read_wvc_model
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
    samples = raw_header['samples']

    # Read mask
    mask_header = read_envi_header(os.path.splitext(config['raw_image_file'])[0]+'.hdr')
    mask_image = np.memmap(config['mask_image_file'],
                           dtype='bool',
                           mode='r',
                           shape=(mask_header['bands'], raw_header['lines'], raw_header['samples']))

    # Get average radiance spectra with dark current corrected
    avg_rdn_01 = get_avg_rdn_01(raw_image, cal_data, mask_image)
    raw_image.flush()
    mask_image.flush()
    del raw_header, mask_header

    # Get average radiance spectra with dark current and gain corrected
    avg_rdn_02 = get_avg_rdn_02(avg_rdn_01, cal_data)

    plt.figure()
    plt.plot(cal_data['spectralVector'], avg_rdn_02, '-')
    plt.show()


    plt.figure()
    plt.plot(cal_data['spectralVector'], avg_rdn_02[:,10], '-')
    plt.show()

    # Get average scan angles
    avg_vza, avg_raa = get_avg_scan_angles(config['sca_image_file'])

    # Build water vapor column estimation model
    """ Notes:
        (1) The view geometry does not seem to affect the model.
            Therefore, a vza=0 and raa=0 are used here.
    """
    wvc_model = read_wvc_model(config['wvc_model_file'])

    # Estimate water vapor for each column
    ratio = avg_rdn_02[wvc_model['Bands'][1], :]/(avg_rdn_02[wvc_model['Bands'][0], :]*wvc_model['Weights'][0]+avg_rdn_02[wvc_model['Bands'][2], :]*wvc_model['Weights'][1])
    wvc = np.interp(ratio, wvc_model['Ratio'], wvc_model['WVC'])
    vis = 80
    plt.figure()
    plt.plot(wvc_model['Ratio'], wvc_model['WVC'], 'r.-')
    plt.plot(ratio, wvc, 'b.')
    plt.show()
    del ratio

    wvc = wvc.mean()
    # Interpolate atm lut
    lut_wave, lut_rdn_0, lut_rdn_1 = interp_atm_lut_to_angles(atm_lut_file, wvc, vis, avg_vza, avg_raa)

    plt.plot(lut_wave, lut_rdn_0, 'r-')
    plt.show()

    for wave_range in features.values():
        wave_range = [744,  784]
#        wave_range = [1255, 1285]
        wave_0, band_0 = get_closest_wave(cal_data['spectralVector'], wave_range[0])
        wave_1, band_1 = get_closest_wave(cal_data['spectralVector'], wave_range[1])
        if abs(wave_0-wave_range[0])>10 or abs(wave_1-wave_range[1])>10:
            continue

        X = []
        lower_bounds = [-5.0, -2.0]
        upper_bounds = [5.0, 2.0]
        for sample in range(0, samples, 100):
            x0 = [0,0]
            p = optimize.least_squares(err, x0,
                                       bounds=(lower_bounds, upper_bounds),
                                       args=(cal_data, avg_rdn_01,
                                             lut_wave, lut_rdn_0, lut_rdn_1,
                                             [band_0, band_1+1], sample))
            X.append(p.x)
    X = np.array(X)
    plt.plot(X[:,0], '.', color='blue')
    plt.show()

def err(x, cal_data, avg_rdn_01, lut_waves, lut_rdn_0, lut_rdn_1, band_range, sample):
    band_0 = band_range[0]
    band_1 = band_range[1]

    # Apply wavelength and fwhm shifts.
    sensor_waves = list(cal_data['spectralVector'][:band_0])+list(cal_data['spectralVector'][band_0:band_1]+x[0])+list(cal_data['spectralVector'][band_1:])
    sensor_fwhms = list(cal_data['fwhm'][:band_0])+list(cal_data['fwhm'][band_0:band_1]+x[1])+list(cal_data['fwhm'][band_1:])
    sensor_fwhms = cal_data['fwhm']
    sensor_waves = np.array(sensor_waves)
    sensor_fwhms = np.array(sensor_fwhms)

    # Get sensor radiance
    sensor_rdn = get_avg_rdn_02(avg_rdn_01[:,sample], cal_data, sample, sensor_waves)
    sensor_rdn = sensor_rdn[band_0:band_1]
    sensor_waves = sensor_waves[band_0:band_1]
    sensor_fwhms = sensor_fwhms[band_0:band_1]

    lut_rdn_0 = resample_spectra(lut_rdn_0[:,sample], lut_waves, sensor_waves, sensor_fwhms)
    lut_rdn_1 = resample_spectra(lut_rdn_1[:,sample], lut_waves, sensor_waves, sensor_fwhms)


    lut_rdn = lut_rdn_1-lut_rdn_0
    sensor_rdn = sensor_rdn-lut_rdn_0

    sensor_rdn = spectral_continuum_remove(sensor_rdn, sensor_waves)
    lut_rdn = spectral_continuum_remove(lut_rdn, sensor_waves)

    #Define costs.
    cost = (sensor_rdn-lut_rdn)**2
    return cost

def interp_atm_lut_to_angles(atm_lut_file, wvc, vis, vzas, raas):
    """ Interpolate atmosphere look-up-table radiance to different view angles.
    Notes:
        (1) surface albedos are fixed to 0 and 0.5.
        (2) constant water vapor column `wvc` and visibility `vis` are used.
    Arguments:
        atm_lut_file: str
            Atmosphere look-up-table filename.
        wvc, vis: float
            Constant water vapor column and visibility.
        vzas, raas: list of floats
            View zenith angles and relative azimuth angles.
    Returns:
        WAVE: array
            Wavelengths of the atmosphere look-up-table radiance.
        rdn_0: 2D array
            Interpolated path radiance (albedo=0.0) at different view angles.
        rdn_1: 2D array
            Interpolated at-sensor radiance (albedo=0.5) at different view angles.
    """
    from atmosphere import read_atm_lut_metadata
    atm_lut_metadata = read_atm_lut_metadata(atm_lut_file+'.meta')
    atm_lut_metadata['shape'] = tuple([int(v) for v in atm_lut_metadata['shape']])
    WVC = np.array([float(v) for v in atm_lut_metadata['WVC']])
    VIS = np.array([float(v) for v in atm_lut_metadata['VIS']])
    VZA = np.array([float(v) for v in atm_lut_metadata['VZA']])
    RAA = np.array([float(v) for v in atm_lut_metadata['RAA']])
    WAVE = np.array([float(v) for v in atm_lut_metadata['WAVE']])
    atm_lut = np.memmap(atm_lut_file,
                        dtype=atm_lut_metadata['dtype'],
                        mode='r',
                        shape=atm_lut_metadata['shape'])
    # subtract path radiance
    the_rdn_0 = atm_lut[0,...]
    the_rdn_1 = atm_lut[1,...] # shape=[WVC, VIS, VZA, RAA, WAVE]
    atm_lut.flush()

    # interpolate along water vapor column
    index0 = np.where(WVC<=wvc)[0][-1]
    index1 = np.where(WVC>wvc)[0][0]
    x0 = WVC[index0]
    x1 = WVC[index1]
    the_rdn_0 = (the_rdn_0[index0,...]*(x1-wvc) + the_rdn_0[index1,...]*(wvc-x0))/(x1-x0) # shape=[VIS, VZA, RAA, WAVE]
    the_rdn_1 = (the_rdn_1[index0,...]*(x1-wvc) + the_rdn_1[index1,...]*(wvc-x0))/(x1-x0) # shape=[VIS, VZA, RAA, WAVE]

    # interpolate along visibility
    index0 = np.where(VIS<=vis)[0][-1]
    index1 = np.where(VIS>vis)[0][0]
    x0 = VIS[index0]
    x1 = VIS[index1]
    the_rdn_0 = (the_rdn_0[index0,...]*(x1-wvc) + the_rdn_0[index1,...]*(wvc-x0))/(x1-x0) # shape=[VZA, RAA, WAVE]
    the_rdn_1 = (the_rdn_1[index0,...]*(x1-wvc) + the_rdn_1[index1,...]*(wvc-x0))/(x1-x0) # shape=[VZA, RAA, WAVE]

    # interpolate to angles
    rdn_0 = []
    rdn_1 = []
    for vza, raa in zip(vzas, raas):
        vza0_index = np.where(VZA<=vza)[0][-1]
        vza1_index = np.where(VZA>vza)[0][0]
        raa0_index = np.where(RAA<=raa)[0][-1]
        raa1_index = np.where(RAA>raa)[0][0]
        vza0 = VZA[vza0_index]
        vza1 = VZA[vza1_index]
        raa0 = RAA[raa0_index]
        raa1 = RAA[raa1_index]
        rdn_0.append((the_rdn_0[vza0_index,raa0_index,:]*(raa1-raa)*(vza1-vza)+
                      the_rdn_0[vza0_index,raa1_index,:]*(raa-raa0)*(vza1-vza)+
                      the_rdn_0[vza1_index,raa0_index,:]*(raa1-raa)*(vza-vza0)+
                      the_rdn_0[vza1_index,raa1_index,:]*(raa-raa0)*(vza-vza0))/((raa1-raa0)*(vza1-vza0)))
        rdn_1.append((the_rdn_1[vza0_index,raa0_index,:]*(raa1-raa)*(vza1-vza)+
                      the_rdn_1[vza0_index,raa1_index,:]*(raa-raa0)*(vza1-vza)+
                      the_rdn_1[vza1_index,raa0_index,:]*(raa1-raa)*(vza-vza0)+
                      the_rdn_1[vza1_index,raa1_index,:]*(raa-raa0)*(vza-vza0))/((raa1-raa0)*(vza1-vza0)))

    return WAVE, np.array(rdn_0).T, np.array(rdn_1).T

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
    bands = raw_header['bands']

    """ Rule 1 for good pixels:
        At-sensor blue reflectance>0.10, and nir reflectance>0.10, and swir reflectance>0.10
    """
    solar_flux = resample_solar_flux(solar_flux_file, cal_data['spectralVector'], cal_data['fwhm'])
    solar_flux /= 10 # mW / (m2 nm) -> mW / (cm2 um)
    cos_sun_zenith = np.cos(np.deg2rad(sun_zenith))
    mask = np.full((raw_image.shape[0], raw_image.shape[2]), True, dtype=np.bool)
    # if the reflectance at 470 nm is less than 0.10, then mask these pixels.
    wave, band = get_closest_wave(cal_data['spectralVector'], 470)
    if abs(wave-470)<10:
        refl = raw2rdn(raw_image[:,band,:], cal_data, band)*np.pi/(solar_flux[band]*cos_sun_zenith)
        mask &= (refl>0.10)
    # if the reflectance at 850 nm is less than 0.10, then mask these pixels.
    wave, band = get_closest_wave(cal_data['spectralVector'], 850)
    if abs(wave-850)<10:
        refl = raw2rdn(raw_image[:,band,:], cal_data, band)*np.pi/(solar_flux[band]*cos_sun_zenith)
        mask &= (refl>0.10)
    # if the reflectance at 1600 nm is less than 0.10, then mask these pixels.
    wave, band = get_closest_wave(cal_data['spectralVector'], 1600)
    if abs(wave-1600)<10:
        refl = raw2rdn(raw_image[:,band,:], cal_data, band)*np.pi/(solar_flux[band]*cos_sun_zenith)
        mask &= (refl>0.10)

    """ Rule 2:
        Mask dark pixels by using radiance values.
    """
    mask_image = np.memmap(mask_image_file,
                           dtype='uint8',
                           mode='w+',
                           shape=(raw_image.shape[1], raw_image.shape[0], raw_image.shape[2]))
    log_mesg = 'Band (max=%d): ' %bands
    for band in range(bands):
        if band%50==0:
            log_mesg += '%d, ' %band
        rdn = raw2rdn(raw_image[:,band,:], cal_data, band)
        mask_image[band,:,:] = mask&(raw_image[:,band,:]<cal_data['satValue'])&(rdn>0.0)
        del rdn
    mask_image.flush()
    raw_image.flush()
    del cal_data, mask
    log_mesg += '%d, Done!' %bands
    logging.info(log_mesg)

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

def spectral_continuum_remove(spectra, waves):
    """Continuum remove spectra.
    Arguments:
        spectra: 1D or 2D array
            Raw spectral data, dimension: [Bands] or [Bands, Columns].
        waves: list
            Spectral wavelengths.
    Returns:
        cont_rm_spectra: 1D or 2D array
            Continuum removed spectra, dimension: [Bands] or [Bands, Columns].
    """
    waves = np.array(waves)
    interp_spectra = (waves-waves[0])*(spectra[-1]-spectra[0])/(waves[-1]-waves[0])+spectra[0]
    cont_rmd_spectra = spectra/interp_spectra
    return cont_rmd_spectra

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
            offset = background+(background_last-background)*np.expand_dims(np.arange(lines), axis=1)/number_of_frames
            del background_last
        else:
            offset = background
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

def get_avg_rdn_02(avg_rdn_01, cal_data, sample=None, waves=None):
    """ Average radiance along each column (both dark current and gain corrected!).
    Arguments:
        avg_rdn_01: 1D or 2D array
            Averaged radiance along each column with dark current corrected, dimension: [bands, samples].
        cal_data: dict
            Calibration data.
    Returns:
        avg_rdn_02: 2D array
            Averaged radiance along each column (both dark current and gain corrected!), dimension: [bands, samples].
    """
    h = constants.Planck# Planck constant
    c = constants.c*1e+9# Light speed nm/s
    SF = cal_data['SF']
    integration_time = cal_data['integrationTime']
    aperture_area = np.pi*cal_data['apertureSize']*cal_data['apertureSize']
    solid_angle = cal_data['solidAngle']

    if avg_rdn_01.ndim==1:
        RE = cal_data['RE'][:,sample]# (bands)
        if len(waves):
            QE = np.interp(waves, cal_data['spectralVector'], cal_data['QE'])
        else:
            QE = cal_data['QE']
        wavelength_interval = cal_data['spectralSampling']
        center_wavelength = cal_data['spectralVector']# (bands)
        gain = h*c*1e6/(RE*QE*SF*integration_time*aperture_area*solid_angle*wavelength_interval*center_wavelength)
        avg_rdn_02 = gain*avg_rdn_01*100# in mW/(cm2*um)
        del gain
    else:
        samples = avg_rdn_01.shape[1]
        RE = cal_data['RE']# (bands, samples)
        if waves:
            QE = np.interp(waves, cal_data['spectralVector'], cal_data['QE'])
            QE = np.tile(QE, (samples,1)).transpose()# (bands, samples)
        else:
            QE = np.tile(cal_data['QE'], (samples,1)).transpose()# (bands, samples)
        wavelength_interval = np.tile(cal_data['spectralSampling'], (samples, 1)).transpose()# (bands, samples)
        center_wavelength = np.tile(cal_data['spectralVector'], (samples, 1)).transpose()# (bands, samples)
        gain = h*c*1e6/(RE*QE*SF*integration_time*aperture_area*solid_angle*wavelength_interval*center_wavelength)
        avg_rdn_02 = gain*avg_rdn_01*100# in mW/(cm2*um)
        del gain
        # Smooth radiance spatially
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

def raw2rdn(raw_image, cal_data, band_index, wave=None):
    """ Convert raw DN values to physical radiance in mW/(cm2*um) for one band.
    Arguments:
        raw_image: 2D array
            Single band raw DN image.
        cal_data: dict
            Calibration data.
        band_index: int
            Band index.
        wave: None or float
            User-defined wavelength.
    Returns:
        rdn_image: 2D array
            Single band radiance image.
    """

    lines = raw_image.shape[0]

    # Get calibration values
    h = constants.Planck# Planck constant
    c = constants.c*1e+9# Light speed nm/s
    RE = cal_data['RE'][band_index, :]
    if wave:
        QE = np.interp(wave, cal_data['spectralVector'], cal_data['QE'])
    else:
        QE = cal_data['QE'][band_index]
    SF = cal_data['SF']
    integration_time = cal_data['integrationTime']
    aperture_area = np.pi*cal_data['apertureSize']*cal_data['apertureSize']
    solid_angle = cal_data['solidAngle']
    wavelength_interval = cal_data['spectralSampling'][band_index]
    if wave:
        center_wavelength = wave
    else:
        center_wavelength = cal_data['spectralVector'][band_index]
    number_of_frames = cal_data['numberOfFrames']

    serial_number = cal_data['serialNumber']

    # Calculate gain
    gain = h*c*1e6/(RE*QE*SF*integration_time*aperture_area*solid_angle*wavelength_interval*center_wavelength)
    gain = np.tile(gain, (lines, 1))

    # Calculate offset
    if serial_number>=3000 and serial_number<=5000:
        offset = np.zeros_like(gain)
        background = cal_data['background'][band_index,:]
        background_last = cal_data['backgroundLast'][band_index,:]
        for line in range(lines):
            offset[line,:] = background+(background_last-background)*line/number_of_frames
        del background, background_last
    else:
        background = cal_data['background'][band_index,:]
        offset = np.tile(background, (lines, 1))
        del background

    # Calibration
    rdn_image = ((raw_image.astype('float32')-offset)*gain).astype('float32')
    rdn_image = rdn_image*100.0# in mW/(cm2*um)
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
    solar_flux = resample_spectra(solar_flux[:,1], solar_flux[:,0], sensor_waves, sensor_fwhms)

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