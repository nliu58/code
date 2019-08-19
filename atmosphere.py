""" Functions to do atmospheric corrections.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import os, logging, multiprocessing, numpy as np


logger = logging.getLogger(__name__)

""" Notes:
    (1) Pre-defined atmosphere look-up-table grids. Do not change them!
"""
vza_grid = 5 # view zenith angle grid size, in degrees
raa_grid = 15 # relative azimuth angle grid size, in degrees
RHO =  [0, 0.5, 1.0] # surface albedo
WVC =  [4, 10, 15, 20, 25, 30, 35, 40, 45, 50] # water vapor column, in mm
VIS =  [5, 10, 20, 40, 80, 120] # aerosol visibility, in km
WAVE =  list(np.arange(4000, 25001)/10) # wavelength, in nm

def make_atm_lut(config):
    """ Make an atmophere look-up-table (LUT).
    Notes:
        (1) If the `atm_database` exits, then do interpolation to generate a LUT.
            Otherwise, use the LibRadTran model to generate it.
    Arguments:
        config: dictionary
            Configuration parameters.
    """
    from envi import read_envi_header
    from dem import get_avg_elev

    # If the LUT has been created, do nothing.
    if os.path.exists(config['atm_lut_file']):
        logger.info('Write atmospheric LUT to %s.' %config['atm_lut_file'])
        return

    # Make a directory.
    if not os.path.exists(config['atm_lut_dir']):
        os.mkdir(config['atm_lut_dir'])

    # Make the LUT
    """ Notes:
        (1) If `atm_database` is None, the LibRadTran model is used to the LUT.
            Otherwise, the LUT is interpolated from the `atm_database`.
    """
    """ Step 1: Read sun zenith `sza` in degrees,
                sun azimuth `saa` in degrees,
                elevation `elev` in km,
                above-ground flight altitude `zout` in km.
    """
    sca_header = read_envi_header(config['vnir']['sca_image_file']+'.hdr')
    sza = float(sca_header['sun zenith'])
    saa = float(sca_header['sun azimuth'])
    elev = get_avg_elev(config['vnir']['new_dem_image_file'])/1000.0
    imugps = np.loadtxt(config['vnir']['new_imugps_file'])
    avg_flight_altitude = imugps[:,3].mean()
    zout = avg_flight_altitude/1000.0 - elev
    del imugps, sca_header, avg_flight_altitude

    """ Step 2: Get the LUT grids of
                view zenith angle `VZA` in degrees,
                relative azimuth angle `RAA` in degrees.
    """
    VZA = []
    RAA = []
    for sensor in ['vnir', 'swir']:
        sca_header = read_envi_header(config[sensor]['sca_image_file']+'.hdr')
        sca_image = np.memmap(config[sensor]['sca_image_file'],
                              dtype='float32',
                              mode='r',
                              offset=0,
                              shape=(2, sca_header['lines'], sca_header['samples']))
        raa = saa-sca_image[1,:,:]
        raa[raa<0] += 360.0
        raa[raa>180] = 360.0-raa[raa>180]
        VZA += list(np.arange(int(np.floor(sca_image[0,:,:].min()/vza_grid)),
                              int(np.ceil(sca_image[0,:,:].max()/vza_grid))+.01,
                              1)*vza_grid)
        RAA += list(np.arange(int(np.floor(raa.min()/raa_grid)),
                              int(np.ceil(raa.max()/raa_grid))+.01,
                              1)*raa_grid)
        sca_image.flush()
        del raa, sca_header
    VZA = sorted(list(set(VZA)))
    RAA = sorted(list(set(RAA)))

    if config['atm_database'] is None:
        """ Step 3: Use the LibRadTran model to make the LUT.
        """

        # Initialize rtm configurations
        rtm_config = dict()
        rtm_config['altitude'] = elev
        rtm_config['zout'] = zout
        rtm_config['sun_zenith'] = sza
        rtm_config['sun_azimuth'] = 0
        rtm_config['resolution'] = config['rtm_params']['resolution']
        rtm_config['source_file'] = '../data/solar_flux/kurudz_0.1nm.dat'
        rtm_config['atmosphere_file'] = config['rtm_params']['atm_mode']
        rtm_config['lambda_0'] = 400
        rtm_config['lambda_1'] = 2500
        rtm_config['o3'] = 331

        # Make atmospheric look-up-tables
        pool = multiprocessing.Pool(processes=min(config['rtm_params']['cpu_count'], multiprocessing.cpu_count()))
        rtm_config['sensor_zenith'] = VZA
        rtm_config['sensor_azimuth'] = RAA
        for rho in RHO:
            rtm_config['albedo'] = rho
            for wvc in WVC:
                rtm_config['h2o'] = wvc
                for vis in VIS:
                    rtm_config['aerosol_visibility'] = vis
                    inp_file = os.path.join(config['atm_lut_dir'], 'rho_%03d_wvc_%03d_vis_%03d.inp' %(rho*100, wvc, vis))
                    out_file = inp_file[:-len('.inp')]+'.out'
                    make_inp_file(inp_file, rtm_config)
                    if os.path.exists(out_file) and os.path.getsize(out_file):
                        continue
                    pool.apply_async(run_rtm, (os.path.join(config['rtm_params']['install_dir'], 'test'), inp_file, out_file))
        pool.close()
        pool.join()

        # Save all .out files to a binary file
        atm_lut = np.memmap(config['atm_lut_file'],
                            dtype='float32',
                            mode='w+',
                            offset=0,
                            shape=(len(RHO), len(WVC), len(VIS), len(VZA), len(RAA), len(WAVE)))
        for rho_index, rho in enumerate(RHO):
            for wvc_index, wvc in enumerate(WVC):
                for vis_index, vis in enumerate(VIS):
                    out_file = os.path.join(config['atm_lut_dir'], 'rho_%03d_wvc_%03d_vis_%03d.out' %(rho*100, wvc, vis))
                    data = np.loadtxt(out_file, dtype='float32') # data.shape = [len(WAVE), len(SZA)*len(RAA)]
                    data = np.dstack(np.split(data, len(VZA), axis=1)[::-1]) # data.shape = [len(WAVE), len(RAA), len(SZA)]
                    data = data.swapaxes(0,2) # data.shape = [len(SZA), len(RAA), len(WAVE)]
                    """ Notes:
                        (1) In the .out file, the radiance uu is arranged as:
                            umu(0),phi(0) umu(0),phi(1) ... umu(0),phi(N)
                            umu(1),phi(0) umu(1),phi(1) ... umu(0),phi(N)
                            ...
                            umu(M),phi(0) umu(M),phi(1) ... umu(M),phi(N)
                            where umu=cos(sensor_zenith), which is ascendingly ordered.
                            To ascend SZA, we need to reverse the order of the radiance.
                            That is why we have [::-1] in the `data = np.dstack(...)` step.
                    """
                    atm_lut[rho_index, wvc_index, vis_index, :, :, :] = data
                    del data, out_file
        atm_lut.flush()
        atm_lut_metadata = dict()
        atm_lut_metadata['description'] = 'Atmospheric Look-Up-Table Metadata'
        atm_lut_metadata['dtype'] = 'float32'
        atm_lut_metadata['shape'] = [len(RHO), len(WVC), len(VIS), len(VZA), len(RAA), len(WAVE)]
        atm_lut_metadata['dimension'] = ['RHO', 'WVC', 'VIS', 'VZA', 'RAA', 'WAVE']
        atm_lut_metadata['RHO'] = RHO
        atm_lut_metadata['WVC'] = WVC
        atm_lut_metadata['VIS'] = VIS
        atm_lut_metadata['VZA'] = VZA
        atm_lut_metadata['RAA'] = RAA
        atm_lut_metadata['WAVE'] = WAVE
        atm_lut_metadata['SZA'] = sza
        atm_lut_metadata['SAA'] = saa
        write_atm_lut_metadata(config['atm_lut_file']+'.meta', atm_lut_metadata)
        #TODO: remove all *.inp and *.out files.
    else:
        #TODO: do interpolations on `atm_database` to generate the LUT.
        pass
    logger.info('Write atmospheric LUT to %s.' %config['atm_lut_file'])

def build_wvc_model(wvc_model_file, atm_lut_file, raw_header_file, setting_file, vis=80, vza=0, raa=0):
    """ Build water vapor model for a certain view geometry.
    Notes:
        (1) rho = 0.5
    Arguments:
        wvc_model_figure_file: str
            Water vapor column estimation model figure filename.
        atm_lut_file: str
            Atmosphere look-up-table file.
        raw_header_file: str
            Raw Hyspex header filename.
        setting_file: str
            Calibration setting filename.
        vza, raa: float
            View zenith angle, and relative azimuth angle in degrees.
    Returns:
        wvc_model: dict
            Column water vapor estimation model.
    """
    from radiometric import get_hyspex_setting
    from spectral import get_closest_wave, resample_spectra, estimate_fwhms_from_waves

    # Get sensor wavelengths and fwhms.
    if setting_file:
        setting = get_hyspex_setting(setting_file)
        sensor_waves = np.array(setting['spectral_calib'])
        del setting
    else:
        header = raw_header_file(raw_header_file)
        sensor_waves = np.array(header['wavelength'])
        del header
    sensor_fwhms = estimate_fwhms_from_waves(sensor_waves)

    # Read atmosphere look-up-table metadata.
    atm_lut_metadata = read_atm_lut_metadata(atm_lut_file+'.meta')
    atm_lut_metadata['shape'] = tuple([int(v) for v in atm_lut_metadata['shape']])
    WVC = np.array([float(v) for v in atm_lut_metadata['WVC']])
    VIS = np.array([float(v) for v in atm_lut_metadata['VIS']])
    VZA = np.array([float(v) for v in atm_lut_metadata['VZA']])
    RAA = np.array([float(v) for v in atm_lut_metadata['RAA']])
    WAVE = np.array([float(v) for v in atm_lut_metadata['WAVE']])

    # Load atmosphere look-up-table data.
    """ Notes:
        (1) atm_lut dimensions: [RHO, WVC, VIS, VZA, RAA, WAVE]
    """
    atm_lut = np.memmap(atm_lut_file,
                        dtype=atm_lut_metadata['dtype'],
                        mode='r',
                        shape=atm_lut_metadata['shape'])

    # Subtract path radiance
    rdn = atm_lut[1,...] - atm_lut[0,...] # dimesion: [WVC, VIS, VZA, RAA, WAVE]
    atm_lut.flush()

    # Interpolate along the VIS direction
    index0 = np.where(VIS<=vis)[0][-1]
    index1 = np.where(VIS>vis)[0][0]
    x0 = VIS[index0]
    x1 = VIS[index1]
    rdn = (rdn[:,index0,...]*(x1-vis) + rdn[:,index1,...]*(vis-x0))/(x1-x0) # dimesion: [WVC, VZA, RAA, WAVE]

    # Interpolate along the VZA direction
    index0 = np.where(VZA<=vza)[0][-1]
    index1 = np.where(VZA>vza)[0][0]
    x0 = VZA[index0]
    x1 = VZA[index1]
    rdn = (rdn[:,index0,...]*(x1-vza) + rdn[:,index1,...]*(vza-x0))/(x1-x0) # dimesion: [WVC, RAA, WAVE]

    # Interpolate along the RAA direction
    index0 = np.where(RAA<=raa)[0][-1]
    index1 = np.where(RAA>raa)[0][0]
    x0 = RAA[index0]
    x1 = RAA[index1]
    rdn = (rdn[:,index0,...]*(x1-raa) + rdn[:,index1,...]*(raa-x0))/(x1-x0) # dimesion: [WVC, WAVE]

    # Get model wavelengths
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

    wvc_model['Bands'] = [left_band, middle_band, right_band]
    wvc_model['Waves'] = [left_wave, middle_wave, right_wave]
    wvc_model['Weights'] = [left_weight, right_weight]

    # Resample radiance to model wavelengths
    rdn = resample_spectra(rdn, WAVE, sensor_waves[wvc_model['Bands']], sensor_fwhms[wvc_model['Bands']])/10.0 # 10: mW / (m2 nm) -> mW / (cm2 um)

    # Calculate absorption depths
    ratio = rdn[:,1]/(left_weight*rdn[:,0]+right_weight*rdn[:,2])

    # Save depths and wvc to the model
    index = np.argsort(ratio)
    wvc_model['Ratio'] = list(ratio[index])
    wvc_model['WVC'] = list(WVC[index])

    # Save WVC model.
    write_wvc_model(wvc_model_file, wvc_model)
    logger.info('Write WVC model to %s.' %wvc_model_file)

def write_wvc_model(wvc_model_file, wvc_model):
    """ Write the WVC model to a file.
    wvc_model_file: str
        Water vapor column model filename.
    wvc_model: dict
        Water vapor column model.
    """
    fid = open(wvc_model_file, 'w')
    keys = ['Bands', 'Waves', 'Weights', 'Ratio', 'WVC']
    for key in keys:
        value = map(str, wvc_model[key])
        value = '{%s}' %(', '.join(value))
        fid.write('%s = %s\n' %(key, value))
    fid.close()

def read_wvc_model(wvc_model_file):
    """ Read WVC model parameters.
    """
    fid = open(wvc_model_file, 'r')
    trans_tab = str.maketrans(dict.fromkeys('\n{}'))
    wvc_model = dict()
    while 1:
        line = fid.readline()
        if '=' in line:
            key, value = line.split('=', 1)
            if ('{' in value) and ('}' not in value):
                while '}' not in line:
                    line = fid.readline()
                    if line.strip()[0] == ';':
                        continue
                    value += line
            key = key.strip()
            value = value.translate(trans_tab).strip()
            value = list(map(str.strip, value.split(',')))
            if key == 'Bands':
                wvc_model[key] = list(map(int, value))
            else:
                wvc_model[key] = list(map(float, value))
        if line == '':
            break
    fid.close()

    return wvc_model

def write_atm_lut_metadata(metadata_file, metadata):
    """ Write the metadata of the atmosphere look-up-table file.
    Arguments:
        metadata_file: str
            Metadata filename.
        metadata: dict
            Metadata.
    """

    fid = open(metadata_file, 'w')
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

def read_atm_lut_metadata(metadata_file):
    """ Read the metadata of the atmosphere look-up-table file.
    Arguments:
        metadata_file: str
            Metadata filename.
    Returns:
        metadata: dict
            Metadata.
    """

    fid = open(metadata_file, 'r')
    trans_tab = str.maketrans(dict.fromkeys('\n{}'))
    metadata = dict()
    while 1:
        line = fid.readline()
        if '=' in line:
            key, value = line.split('=', 1)
            if ('{' in value) and ('}' not in value):
                while '}' not in line:
                    line = fid.readline()
                    if line.strip()[0] == ';':
                        continue
                    value += line
            key = key.strip()
            if ('{' in value) and ('}' in value):
                value = value.translate(trans_tab).strip()
                value = list(map(str.strip, value.split(',')))
            else:
                value = value.translate(trans_tab).strip()
            metadata[key] = value
        if line == '':
            break
    fid.close()

    return metadata

def run_rtm(rtm_dir, inp_file, out_file):
    """ Run the LibRadTran model.
    Arguments:
        rtm_dir: str
            RTM installation directory.
        inp_file: str
            Input file of LibRadTran.
        out_file: str
            Output file of LibRadTran.
    """
    os.system('cd %s && uvspec <%s> %s' %(rtm_dir, inp_file, out_file))

def make_inp_file(file, params):
    """ Make a LibRadTran input file.
    References:
        (1) Mayer B., Kylling A., Emde C., Buras R., Hamann U., Gasteiger J., and Richter B. (2017).
            libRadtran user’s guide (version=2.0.2).
    Arguments:
        file: str
            Libradtran input filename.
        params: dict
            Libradtran input parameters.
    """

    fid = open(file, 'w')

    # wavelength range, spectral resolution
    fid.write('wavelength %s %s\n' %(params['lambda_0'], params['lambda_1']))# wavelength range
    fid.write('mol_abs_param reptran %s\n' %params['resolution'])

    # solar angles
    fid.write('sza %s\n' %params['sun_zenith'])
    fid.write('phi0 %s\n' %params['sun_azimuth'])

    # sensor angles
    umu = list(map(str, np.sort(np.cos(np.deg2rad(params['sensor_zenith'])))))
    phi = list(map(str, sorted(params['sensor_azimuth'])))
    fid.write('umu %s\n' %(' '.join(umu)))
    fid.write('phi %s\n' %(' '.join(phi)))

    # above-sea-level elevation and above-ground altitude
    fid.write('altitude %s\n' %params['altitude'])
    fid.write('zout %s\n' %params['zout'])

    # atmosphere conditions
    fid.write('atmosphere_file %s\n' %params['atmosphere_file'])
    fid.write('mol_modify O3 %s DU\n' %params['o3'])
    fid.write('mol_modify H2O %s MM\n' %params['h2o'])
    fid.write('aerosol_default\n')
    fid.write('aerosol_visibility %s\n' %params['aerosol_visibility'])
    fid.write('crs_model rayleigh bodhaine\n')
    fid.write('pseudospherical\n')

    # radiative transfer equation solver
    fid.write('rte_solver disort\n')
    fid.write('quiet\n')

    # ground conditions
    fid.write('albedo %s\n' %params['albedo'])

    # solar irradiance
    fid.write('source solar %s\n' %params['source_file'])

    # output
    fid.write('output_user uu \n')

    fid.close()