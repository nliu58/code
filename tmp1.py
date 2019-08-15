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
    mask = raw_image==cal_data['satValue']
    rdn_image = ((raw_image.astype('float32')-offset)*gain).astype('float32')
    mask = mask|(rdn_image<0.0)
    rdn_image[mask] = 0
    rdn_image = rdn_image*100.0 # in mW/(cm2*um)
    del gain, offset, mask

    return rdn_image
