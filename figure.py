""" Functions to make figures.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import logging, gdal, numpy as np, matplotlib.pyplot as plt
from envi import read_envi_header

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def plot_angle_geometry(angle_geometry_figure_file, sca_image_file):
    """ Plot sun and view geometry in a polar coordinate system.
    Arguments:
        angle_geometry_figure_file: str
            Angle geometry figure filename.
        sca_image_file: str
            Scan angle image filename.
    """

    # Read sca image data.
    sca_header = read_envi_header(sca_image_file+'.hdr')
    sca_image = np.memmap(sca_image_file,
                          dtype='float32',
                          mode='r',
                          offset=0,
                          shape=(sca_header['bands'], sca_header['lines'], sca_header['samples']))

    # Scatter-plot view geometry.
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, projection='polar')
    ax.scatter(np.deg2rad(sca_image[1,:,:].flatten()), sca_image[0,:,:].flatten(), color='green')
    sca_image.flush()

    # Scatter-plot sun geometry.
    ax.scatter(np.deg2rad(float(sca_header['sun azimuth'])), float(sca_header['sun zenith']), color='red', marker='*', s=500)
    del sca_header

    # Flip figure left to right.
    ax.set_theta_direction(-1)

    # Rotate figure by 90 degrees.
    ax.set_theta_zero_location('N')
    ax.tick_params(labelsize=20)
    plt.savefig(angle_geometry_figure_file)
    plt.close()

    del ax

    logger.info('Save angle geometry figure to %s.' %angle_geometry_figure_file)

def plot_image_area(image_area_figure_file, dem_image_file, igm_image_file, imugps_file):
    """ Plot image area.
    Arguments:
        image_area_figure_file: str
            Image area figure filename.
        dem_image_file: str
            DEM image filename.
        igm_image_file: str
            IGM image filename.
        imugps_file: str
            IMUGPS filename.
    """

    # Read DEM.
    ds = gdal.Open(dem_image_file, gdal.GA_ReadOnly)
    dem_image = ds.GetRasterBand(1).ReadAsArray()
    dem_geotransform = ds.GetGeoTransform()
    ds = None

    # Read IGM.
    igm_header = read_envi_header(igm_image_file+'.hdr')
    igm_image = np.memmap(igm_image_file,
                          dtype='float64',
                          mode='r',
                          offset=0,
                          shape=(2, igm_header['lines'], igm_header['samples']))
    cols = (igm_image[0,:,:]-dem_geotransform[0])/dem_geotransform[1]
    rows = (igm_image[1,:,:]-dem_geotransform[3])/dem_geotransform[5]
    igm_image.flush()

    # Read IMUGPS
    imugps = np.loadtxt(imugps_file)

    # Make a plot
    plt.figure(figsize=(10, 10.0*dem_image.shape[0]/dem_image.shape[1]))
    plt.imshow(dem_image, cmap='gray', vmin=dem_image.min(), vmax=dem_image.max())
    del dem_image
    plt.plot(cols[:,0],  rows[:,0],  '-', color='green', lw=4)
    plt.plot(cols[:,-1], rows[:,-1], '-', color='green', lw=4)
    plt.plot(cols[0,:],  rows[0,:],  '-', color='green', lw=4)
    plt.plot(cols[-1,:], rows[-1,:], '-', color='green', lw=4)
    cols = (imugps[:,1]-dem_geotransform[0])/dem_geotransform[1]
    rows = (imugps[:,2]-dem_geotransform[3])/dem_geotransform[5]
    plt.plot(cols, rows, '-', color='red', lw=5)
    plt.plot(cols[0], rows[0], '.', color='red', ms=25)
    plt.arrow(cols[-100], rows[-100],
              cols[-1]-cols[-100], rows[-1]-rows[-100],
              head_width=10, head_length=10,
              fc='red', ec='red')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(image_area_figure_file, dpi=1000)
    plt.close()
    del cols, rows, imugps

    logger.info('Save image area figure to %s.' %image_area_figure_file)

def linear_percent_stretch(raw_image):
    """ Do linear percent stretch.
    References:
        (1) https://www.harrisgeospatial.com/docs/BackgroundStretchTypes.html
    Arguments:
        raw_image: 2D array
            Raw image data.
    Returns:
        stretched_image: 2D array
            Percent_stretched image.
    """

    stretched_image = np.zeros(raw_image.shape, dtype='uint8')
    low = np.percentile(raw_image, 2)
    high = np.percentile(raw_image, 98)
    stretched_image[raw_image<low] = low
    stretched_image[raw_image>high] = high
    stretched_image = (raw_image.astype('float32')-low)/(high-low)*255

    return stretched_image

def make_quickview(qickview_figure_file,  raw_image_file, glt_image_file, setting_file):
    """ Make a RGB quickview image.
    Arguments:
        qickview_figure_file: str
            Quickview figure filename.
        raw_image_file: str
            Raw Hyspex image filename.
        glt_image_file: str
            GLT image filename.
    """
    from radiometric import get_cal_data, raw2rdn
    # Read Hyspex raw image
    raw_header = read_envi_header(raw_image_file[:-len('.hyspex')]+'.hdr')
    raw_image = np.memmap(raw_image_file,
                          dtype='int16',
                          mode='r',
                          offset=raw_header['header offset'],
                          shape=(raw_header['lines'], raw_header['bands'], raw_header['samples']))
    cal_data = get_cal_data(raw_image_file, setting_file)

    # Read GLT image
    glt_header = read_envi_header(glt_image_file+'.hdr')
    glt_image = np.memmap(glt_image_file,
                          dtype=np.int32,
                          mode='r',
                          offset=0,
                          shape=(2, glt_header['lines'], glt_header['samples']))

    # Write RGB image
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(qickview_figure_file,
                       glt_header['samples'], glt_header['lines'], 3,
                       gdal.GDT_Byte)
    ds.SetGeoTransform((float(glt_header['map info'][3]),
                        float(glt_header['map info'][5]),
                        0, float(glt_header['map info'][4]),
                        0, -float(glt_header['map info'][6])))
    ds.SetProjection(glt_header['coordinate system string'])
    quickview_image = np.zeros((glt_header['lines'], glt_header['samples']), dtype='uint8')
    I,J = np.where(glt_image[0,:,:]>0)
    for output_band_index, rgb_band_index in enumerate(raw_header['default bands']):
        quickview_image[:,:] = 0
        rdn_image = raw2rdn(raw_image[:,rgb_band_index-1,:], cal_data, rgb_band_index-1)
        rdn_image = linear_percent_stretch(rdn_image)
        quickview_image[I,J] = rdn_image[glt_image[0,I,J], glt_image[1,I,J]]
        ds.GetRasterBand(output_band_index+1).WriteArray(quickview_image)
        del rdn_image
    raw_image.flush()
    glt_image.flush()
    ds = None
    del cal_data, I, J, glt_header, raw_header

    logger.info('Save quickview figure to %s.' %qickview_figure_file)

def plot_wvc_model(wvc_model_figure_file, wvc_model_file):
    """ Plot the WVC model to a figure.
    wvc_model_figure_file: str
        Water vapor column model figure filename.
    wvc_model: dict
        Water vapor column model.
    """
    from atmosphere  import read_wvc_model
    wvc_model = read_wvc_model(wvc_model_file)
    plt.figure(figsize=(10, 6))
    plt.plot(np.array(wvc_model['Ratio'])*100, wvc_model['WVC'], '.-', ms=20, lw=2, color='red')
    plt.xticks([0, 20, 40, 60, 80, 100], [0, 20, 40, 60, 80, 100], fontsize=20)
    plt.yticks([0, 10, 20, 30, 40, 50], [0, 10, 20, 30, 40, 50], fontsize=20)
    plt.text(60, 50, r'$\lambda_{Left}$: %.2f nm' %wvc_model['Waves'][0], fontsize=20)
    plt.text(60, 45, r'$\lambda_{Middle}$: %.2f nm' %wvc_model['Waves'][1], fontsize=20)
    plt.text(60, 40, r'$\lambda_{Right}$: %.2f nm' %wvc_model['Waves'][2], fontsize=20)

    plt.xlim(0, 100)
    plt.ylim(0, 55)
    plt.xlabel('APDA Ratio (%)', fontsize=20)
    plt.ylabel('WVC (mm)', fontsize=20)
    plt.savefig(wvc_model_figure_file, dpi=1000)
    plt.close()

    logger.info('Save WVC model figure to %s.' %wvc_model_file)