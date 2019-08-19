""" Functions to process DEM data.
@author: Nanfeng Liu (nliu58@wisc.edu)
"""

import logging, os, sys, gdal, numpy as np

logger = logging.getLogger(__name__)

def get_avg_elev(dem_image_file):
    """ Get the average elevation of a DEM image.
    Notes:
        Pixels with a negative elevation are excluded from averaging.
    Arguments:
        dem_image_file: str
            DEM image filename.
    Returns:
        avg_elev: float
            Average elevation.
    """

    ds = gdal.Open(dem_image_file, gdal.GA_ReadOnly)
    dem_image = ds.GetRasterBand(1).ReadAsArray()
    avg_elev = dem_image[dem_image>0].mean()# negative DEM values are ignored.
    ds = None
    del dem_image

    return avg_elev

def process_dem(new_dem_image_file, old_dem, imugps_file, fov, map_crs, pixel_size):
    """ Process DEM.
    Notes:
        The input DEM should not be rotated.
    Arguments:
        new_dem_image_file: str
            Processed DEM image filename. The elevation is in meters.
        old_dem: str or float
            Input DEM image filename, or user-specified constant elevation in meters.
        imugps_file: str
            The imugps filename.
        fov: float
            Sensor field of view in degrees.
        map_crs: osr object
            Map coordinate system.
        pixel_size: float
            Image pixel size.
    """

    from envi import empty_envi_header, write_envi_header

    # Estimate the spatial area of the flight.
    """ Notes:
        (1) The first three columns of `imugps` are the flight Map X, Map Y, Height.
        (2) The `altitude` here is above the mean sea level, or the Earth ellipsoid surface.
            It is NOT the altitude above the ground surface.
            Strictly speaking, it should be subtracted by the average elevation
            before calculating `half_swath`.
        (3) A buffer of 50 m is used to ensure the processed DEM image covers the whole flight area.
    """
    imugps = np.loadtxt(imugps_file)
    altitude = imugps[:,3].max()
    buffer = 50
    half_swath = np.tan(np.deg2rad(fov/2))*altitude
    x_min = imugps[:,1].min()-half_swath-buffer
    x_max = imugps[:,1].max()+half_swath+buffer
    y_min = imugps[:,2].min()-half_swath-buffer
    y_max = imugps[:,2].max()+half_swath+buffer
    del imugps, half_swath, altitude, buffer

    # Process DEM
    """ Notes:
        (1) If `old_dem` is a filename, then clip it.
        (2) If `old_dem` is a constant, then make a flat DEM.
    """
    if type(old_dem) is str:
        # Read raw DEM
        ds = gdal.Open(old_dem, gdal.GA_ReadOnly)

        # Get image rows and columns
        geotransform = ds.GetGeoTransform()
        col_0 = int((x_min-geotransform[0])/geotransform[1])
        col_1 = int((x_max-geotransform[0])/geotransform[1])
        row_0 = int((y_max-geotransform[3])/geotransform[5])
        row_1 = int((y_min-geotransform[3])/geotransform[5])

        if (col_0>ds.RasterXSize-1 or
            row_0>ds.RasterYSize-1 or
            col_1<0 or
            row_1<0):
            logger.error('The DEM image does not cover the flight area!')
            sys.exit(1)

        col_0 = max(col_0, 0)
        row_0 = max(row_0, 0)
        col_1 = min(col_1, ds.RasterXSize-1)
        row_1 = min(row_1, ds.RasterYSize-1)
        cols  = int(col_1-col_0)
        rows  = int(row_1-row_0)

        # Read a subset of DEM
        dem_image = ds.GetRasterBand(1).ReadAsArray(col_0, row_0, cols, rows)
        dem_image = dem_image.astype('float32')

        # Write clipped DEM image
        fid = open(new_dem_image_file, 'wb')
        fid.write(dem_image.tostring())
        fid.close()
        ds = None
        del dem_image

        # Update geotransform
        geotransform = (geotransform[0]+col_0*geotransform[1],
                        geotransform[1],
                        0,
                        geotransform[3]+row_0*geotransform[5],
                        0,
                        geotransform[5])
        del col_0, col_1, row_0, row_1

    elif type(old_dem) in [int, float]:
        # Make a flat DEM
        rows = int((y_max-y_min)/pixel_size)
        cols = int((x_max-x_min)/pixel_size)
        dem_image = np.ones((rows, cols), dtype=np.float32)*old_dem

        # Write clipped DEM image
        fid = open(new_dem_image_file, 'wb')
        fid.write(dem_image.tostring())
        fid.close()
        del dem_image

        # Update geotransform
        geotransform = (x_min, pixel_size, 0, y_max, 0, -pixel_size)

    # Write clipped DEM header
    new_dem_header_file = os.path.splitext(new_dem_image_file)[0]+'.hdr'
    new_dem_header = empty_envi_header()
    new_dem_header['description'] = 'DEM, in meters'
    new_dem_header['samples'] = cols
    new_dem_header['lines'] = rows
    new_dem_header['bands'] = 1
    new_dem_header['byte order'] = 0
    new_dem_header['header offset'] = 0
    new_dem_header['interleave'] = 'bsq'
    new_dem_header['data type'] = 4
    new_dem_header['coordinate system string'] = map_crs.ExportToWkt()
    new_dem_header['map info'] = [map_crs.GetAttrValue('projcs').replace(',', ''),
                  1, 1, geotransform[0], geotransform[3], geotransform[1], geotransform[1],
                  ' ',' ', map_crs.GetAttrValue('datum').replace(',', ''), map_crs.GetAttrValue('unit')]
    if map_crs.GetAttrValue('PROJECTION').lower() == 'transverse_mercator':
        new_dem_header['map info'][7] = map_crs.GetUTMZone()
        if y_min>0.0:
            new_dem_header['map info'][8] = 'North'
        else:
            new_dem_header['map info'][8] = 'South'
    write_envi_header(new_dem_header_file, new_dem_header)
    del new_dem_header, new_dem_header_file

    logger.info('Write DEM to %s.' %new_dem_image_file)