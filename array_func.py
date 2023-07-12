# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 21:09:05 2021

@author: Jomfrago
"""

#Script to convert to geotiff matrix of slope, aspect, TRI, TPI and roughness
#that were calculated in Matlab.

#import the require librarys
import numpy as np
from osgeo import gdal, gdalconst
import rasterio
from rasterio.transform import Affine
####################
##Define functions##
####################
def get_geotiff_props(tif_path):
    """function to get properties of base raster: res_x, res_y, projection,
    height, width, x_max, x_min, y_max, y_min"""
    ds = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    # Extract projection & geotransform from input dataset
    ds_proj = ds.GetProjection()
    ds_geotrans = ds.GetGeoTransform()
    ds_res_x = ds_geotrans[1]
    ds_res_y = ds_geotrans[5]  # Note: identical to x-resolution, but negative
    ds_width = ds.RasterXSize
    ds_height = ds.RasterYSize
    # Close access to GeoTIFF file
    ds = None
    # Get bounding box of input dataset
    ds_x_min = ds_geotrans[0]
    ds_y_max = ds_geotrans[3]
    ds_x_max = ds_x_min + (ds_geotrans[1] * ds_width)
    ds_y_min = ds_y_max + (ds_geotrans[5] * ds_height)
    # Return all results as a dictionary
    return {'proj':ds_proj, 'res_x':ds_res_x, 'res_y':ds_res_y, 'x_min':ds_x_min, 'x_max':ds_x_max, 'y_min':ds_y_min, 'y_max':ds_y_max, 'width':ds_width, 'height':ds_height}

def array_to_geotiff(array, tif_path, no_data_value, props, output_format=gdal.GDT_Float32):
    """function to convert a matrix to geotiff"""
    # Get the appropriate GDAL driver
    driver = gdal.GetDriverByName('GTiff')
    # Create a new GeoTIFF file to which the array is to be written
    tif_width = props['width']
    tif_height = props['height']
    ds = driver.Create(tif_path, tif_width, tif_height, 1, output_format)
    # Set the geotransform
    tif_x_min = props['x_min']
    tif_res_x = props['res_x']
    tif_y_max = props['y_max']
    tif_res_y = props['res_y']
    ds.SetGeoTransform((tif_x_min, tif_res_x,0, tif_y_max,0, tif_res_y))
    # Set the projection
    tif_proj = props['proj']
    ds.SetProjection(tif_proj)
    # Set the no data value
    ds.GetRasterBand(1).SetNoDataValue(no_data_value)
    # Write array to GeoTIFF
    ds.GetRasterBand(1).WriteArray(array)
    ds.FlushCache()

def geotiff_to_array(tif_path):
    tif_ds = gdal.Open(tif_path, gdalconst.GA_ReadOnly)
    tif_array = np.array(tif_ds.ReadAsArray())
    tif_ds = None
    return tif_array
   
def export_kde_raster(Z, XX, YY, min_x, max_x, min_y, max_y, proj, filename):
    '''Export and save a kernel density raster.'''

    # Get resolution
    xres = (max_x - min_x) / len(XX)
    yres = (max_y - min_y) / len(YY)

    # Set transform
    transform = Affine.translation(min_x - xres / 2, min_y - yres / 2) * Affine.scale(xres, yres)

    # Export array as raster
    with rasterio.open(
            filename,
            mode = "w",
            driver = "GTiff",
            height = Z.shape[0],
            width = Z.shape[1],
            count = 1,
            dtype = Z.dtype,
            crs = proj,
            transform = transform,
    ) as new_dataset:
            new_dataset.write(Z, 1)