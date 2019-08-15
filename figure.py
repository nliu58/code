""" Functions to make figures.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import gdal
import numpy as np
import matplotlib.pyplot as plt
from envi import read_envi_header
from radiometric import get_cal_data, raw2rdn

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
    ax.scatter(np.deg2rad(sca_image[1,:,:].flatten()),
                sca_image[0,:,:].flatten(),
                color='green')
    sca_image.flush()

    # Scatter-plot sun geometry.
    ax.scatter(np.deg2rad(float(sca_header['sun azimuth'])),
                float(sca_header['sun zenith']),
                color='red',
                marker='*',
                s=500)

    # Flip figure left to right.
    ax.set_theta_direction(-1)

    # Rotate figure by 90 degrees.
    ax.set_theta_zero_location('N')
    ax.tick_params(labelsize=20)
    plt.savefig(angle_geometry_figure_file)
    plt.close()

    del sca_header, ax

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
