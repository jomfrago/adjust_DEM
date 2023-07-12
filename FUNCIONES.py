# -*- coding: utf-8 -*-
"""
Created on Sat May 27 13:05:28 2023

@author: JOMFRAGO
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 13 17:04:22 2023

@author: JOMFRAGO
"""
from osgeo import gdal, osr
import numpy as np
import geopandas as gpd
from array_func import array_to_geotiff, get_geotiff_props ,geotiff_to_array#,export_kde_raster , 
from os import remove
import os
import rasterio
from sklearn.model_selection import train_test_split 
import pandas as pd
import matplotlib.pyplot as plt
import glob
plt.style.use('seaborn')
#import ospybook as pb
#import scipy.ndimage as ndi
#from pykrige.ok import OrdinaryKriging
#from rasterio.plot import show
#import matplotlib.pyplot as plt
#from pykrige.uk import UniversalKriging

def extract_z1(in_puntos,out_puntos,dem1,dem2):
    
    # ini=tm.time()
    #open point shapefile
    pointData = gpd.read_file('%s.shp'%(in_puntos))
    print(pointData.crs)
    #pointData.plot()

    #open raster file
    ds1 = rasterio.open('%s.tif'%(dem1))
    print(ds1.crs)
    print(ds1.count)

    ds2 = rasterio.open('%s.tif'%(dem2))
    print(ds2.crs)
    print(ds2.count)

    #show point and raster on a matplotlib plot
    # fig, ax = plt.subplots(figsize=(12,12))
    # pointData.plot(ax=ax, color='orangered')
    # show(ds1, ax=ax)

    # fig, ax = plt.subplots(figsize=(12,12))
    # pointData.plot(ax=ax, color='orangered')
    # show(ds2, ax=ax)

    #extract xy from point geometry

    for point in pointData['geometry']:
        print(point.xy[0][0],point.xy[1][0])

    #create empty list for each Z_DEM
    z_ds1=[]
    z_ds2=[]

    #extract point value from raster
    for point in pointData['geometry']:
        x = point.xy[0][0]
        y = point.xy[1][0]
        row, col = ds1.index(x,y)
        row1, col1 = ds2.index(x,y)
        z1=ds1.read(1)[row,col]
        z2=ds2.read(1)[row1,col1]
        z_ds1.append(z1)
        z_ds2.append(z2)

    #save each Z_DEM in a dataframe
    z_ds1=pd.DataFrame(z_ds1)
    #z_ds1=np.matrix(z_ds1)
    z_ds2=pd.DataFrame(z_ds2)
    #z_ds2=np.matrix(z_ds2)
    # agregate Z_DEMs to a point dataframe
    pointData=pointData.assign(DEM1=z_ds1,DEM2=z_ds2)
    #calculate error between Z points and Z_DEMs
    #puntos=np.matrix(pointData)
    #real=pointData['ORTO']
    diff_ds1=pd.DataFrame(pointData['ORTO']-pointData['DEM1'])
    #diff_ds1=np.matrix(diff_ds1)
    #diff_ds1=diff_ds1[:,0]
    diff_ds2=pd.DataFrame(pointData['ORTO']-pointData['DEM2'])
    #diff_ds2=np.matrix(diff_ds2)
    #diff_ds2=diff_ds2[:,0]

    #agregate errors to point dataframe and save it as csv an shp file.
    
    pointData=pointData.assign(DIF_DEM1=diff_ds1,DIF_DEM2=diff_ds2)
    #pointData.to_csv("PUNTOS.csv")
    pointData.to_file("%s.shp"%(out_puntos))
    #point=gpd.read_file('PUNTOS.shp')
    
    
    # plt.close('all')
    # fin=tm.time()
    # tiempo=fin-ini
    
    # print("El tiempo de ejecución es de:\n",tiempo/60, "minutos")

def division_puntos(in_puntos,nc,m,out_train,out_test):
    
    # ini=tm.time()
    path=os.getcwd()
    #se divide los puntos en ajuste y validacion.y se guardan en archivo csv
    #puntos=pd.read_csv("PUNTOS.csv",sep=',')
    puntos = gpd.read_file('%s.shp'%(in_puntos))
    print(puntos.crs)
    #puntos.plot()
      
    # for i in range(5,95,5):
    #    if not os.path.exists('{}//DIST_%s'.format(path)%(i)):
    #                        os.mkdir('{}//DIST_%s'.format(path)%(i))
    
    # with os.scandir(path) as ficheros:
    #     subdirectorios = [fichero.name for fichero in ficheros if fichero.is_dir()]
    # print(subdirectorios)
    # subdrectorios=subdirectorios[0:18]
    
    # for j in reversed( range(5,95,5)):
    for k in range(nc):
        train, test = train_test_split(puntos,test_size=m/100)
        
        test.to_file('{}//%s_%s.shp'.format(path)%(out_test,k))
        train.to_file('{}//%s_%s.shp'.format(path)%(out_train,k))
    
    # fin=tm.time()
    # tiempo=fin-ini
    # print("El tiempo de ejecución es de:\n",tiempo/60, "minutos")

def calc_error(nc,in_puntos,z_points,dem_points,out_rmse,out_mae,out_mse,out_pbias):
    
    #metrics=np.array([], dtype=object)
    #metrics1=np.array()
    metri_mae=np.zeros((nc,1))
    metri_mse=np.zeros((nc,1))
    metri_rmse=np.zeros((nc,1))
    metri_pbias=np.zeros((nc,1))
    #metrics1=np.zeros((5,1))
    path=os.getcwd()
    # for l in range (5,95,5):
    for i in range(nc):
        puntos=gpd.read_file("{}//%s_%s.shp".format(path)%(in_puntos,i))
        real=puntos['%s'%(z_points)]
        real=np.transpose(np.matrix(real))
        res=puntos['%s'%(dem_points)]
        #fabdem=puntos['FABDEM']
        #res=pd.DataFrame(columns=['LIDAR','FABDEM'])
        #res['LIDAR']=lidar
        #res['FABDEM']=fabdem
        res=np.transpose(np.matrix(res))
        #mi=np.min(res,0)
        #ma=np.max(res,0)
        errores=real-res
        #errorrel=(errores/real)*100
        errorprom=np.mean(errores,0)
        mae=abs(errorprom)
        errorcuadrado=np.square(errores)
        mse=np.mean(errorcuadrado,0)
        rmse=np.power(mse,(1/2))
        pbias_deno=np.sum((real-res),0)
        pbias_nume=np.sum(real,0)
        pbias=pbias_deno/pbias_nume
        metri_mae[i,]=mae
        mae=pd.DataFrame(metri_mae)
        metri_mse[i,]=mse
        mse=pd.DataFrame(metri_mse)
        metri_rmse[i,]=rmse
        rmse=pd.DataFrame(metri_rmse)
        metri_pbias[i,]=pbias
        pbias=pd.DataFrame(metri_pbias)
        print(i)
    # print(l)
    rmse.to_csv("{}//%s.csv".format(path)%(out_rmse))
    mae.to_csv("{}//%s.csv".format(path)%(out_mae))
    pbias.to_csv("{}//%s.csv".format(path)%(out_pbias))
    mse.to_csv("{}//%s.csv".format(path)%(out_mse))


def mapa_error(nc,base,puntos,z_field,error,binario,dem):
    # ini=tm.time()
    path=os.getcwd()
    # Define el sistema de referencia de coordenadas del raster
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ###############################
    base="{}//%s.tif".format(path)%(base)
    # base_info=get_geotiff_props(base)
    #in_ds=gdal.Open(base)
    #in_band=in_ds.GetRasterBand(1) 
    
    # # # Define las dimensiones del raster
    # x_min, x_max, y_min, y_max = (base_info['x_min'], base_info['x_max'],
    #                                 base_info['y_min'], base_info['y_max']) # Rango de coordenadas en metros
    #proj=base_info['proj']
    # x_size=base_info['width']
    # y_size=base_info['height']
    # x_res=base_info['res_x']
    # y_res=base_info['res_y']
    
    ###############################
    
    # Carga los puntos desde un archivo CSV con las columnas "LAT", "LON" y "Z"
    # for l in range (5,95,5):
    for i in range(nc):      
        in_points='{}\%s_%s.shp'.format(path)%(puntos,i)
        zfield='%s'%(z_field)
        #zfiled='DIF_FABDEM'
        out_idw='{}//%s_idw_%s.tif'.format(path)%(dem,i)
        out_idnn='{}//%s_invdistnn_%s.tif'.format(path)%(dem,i)
        out_near='{}//%s_nearest_%s.tif'.format(path)%(dem,i)
        out_linear='{}//%s_linear_%s.tif'.format(path)%(dem,i)
        out_average='{}//%s_average_%s.tif'.format(path)%(dem,i)
        out_prome='{}//%s_%s_%s.tif'.format(path)%(dem,error,i)
        out_bina='{}//%s_%s_%s.tif'.format(path)%(dem,binario,i)
        #in_ref='{}//%s.tif'.format(path)%(base)
        options1=gdal.GridOptions(format='GTiff',outputType=6,zfield='%s'%(zfield),
                          noData=-9999,algorithm="invdist:power=10.0:smoothing:2.0",
                          outputSRS=srs)
        
        rasterDs1 = gdal.Grid('%s'%(out_idw),'%s'%(in_points),options=options1)
        rasterDs1.FlushCache()
        rasterBand1 = rasterDs1.GetRasterBand(1)
        # get elevation as numpy array
        rasterBand1 = rasterBand1.ReadAsArray()
        
        options2=gdal.GridOptions(format='GTiff',outputType=6,zfield='%s'%(zfield),
                          noData=-9999,algorithm="invdistnn",
                          outputSRS=srs)
        
        rasterDs2 = gdal.Grid('%s'%(out_idnn), '%s'%(in_points),options=options2)
        
        rasterBand2 = rasterDs2.GetRasterBand(1)
        # get elevation as numpy array
        rasterBand2 = rasterBand2.ReadAsArray()
        
        rasterDs3 = gdal.Grid('%s'%(out_near),'%s'%(in_points), format='GTiff',
                              noData=-9999,algorithm='nearest', 
                              zfield='%s'%(zfield),
                              outputSRS=srs,outputType=6)
        rasterDs3.FlushCache()
        
        rasterBand3 = rasterDs3.GetRasterBand(1)
        # get elevation as numpy array
        rasterBand3 = rasterBand3.ReadAsArray()
        
        rasterDs4= gdal.Grid('%s'%(out_linear), '%s'%(in_points), format='GTiff',
                              noData=-9999,algorithm='linear',
                              zfield='%s'%(zfield),
                              outputSRS=srs,outputType=6)
        rasterDs4.FlushCache()
        
        rasterBand4 = rasterDs4.GetRasterBand(1)
        rasterBand4 = rasterBand4.ReadAsArray()
        
        rasterDs5= gdal.Grid('%s'%(out_average),'%s'%(in_points), format='GTiff',
                              noData=-9999,algorithm='average',
                              zfield='%s'%(zfield),
                              outputSRS=srs,outputType=6)
        rasterDs5.FlushCache()
        
        rasterBand5 = rasterDs5.GetRasterBand(1)
        rasterBand5 = rasterBand5.ReadAsArray()
        #se saca el promedio de los 5 metodos
        
        ref='%s'%(base)
        #in_ds=gdal.Open(ref)
        #in_band=in_ds.GetRasterBand(1)
        # props=get_geotiff_props(ref)
        # points_file = gpd.read_file(in_points)
        # x=points_file['LON']
        # y=points_file['LAT']
        # z=list(points_file['%s'%(zfield)])
        # z=np.transpose(np.matrix(z))
        # xi, yi =(np.linspace(x_min, x_max,256), np.linspace(y_min, y_max,256))
        # OK = OrdinaryKriging(
        # np.array(x),
        # np.array(y),
        # z,
        # variogram_model ='linear',
        # verbose = True,
        # enable_plotting = True,
        # coordinates_type = "euclidean",
        # )
    
        # # Evaluate the method on grid
        # zi,sigma=OK.execute("grid", xi, yi)
        #array_to_geotiff(zi,"kriging_%s.tif"%(variogram), -9999, base_info)
        #OrdinaryKriging(x, y, z)
        # Export raster
        # export_kde_raster(Z=zi,XX=xi,YY=yi,
        #                     min_x = x_min, max_x = x_max, min_y = y_min, max_y = y_max,
        #                     proj =proj, filename ='%s'%(out_krig))
        
        band_prome=(rasterBand1+rasterBand2+rasterBand3+rasterBand4+rasterBand5)/5
        
        # band_prome1 = pb.make_slices(band_prome, (3, 3))
        # band_prome2 = np.ma.dstack(band_prome1)
        # rows, cols = in_band.YSize, in_band.XSize
        # band_prome3 = np.ones((rows, cols), np.float32)*-9999
        # band_prome3[1:-1, 1:-1] = np.mean(band_prome2, 2)
        
        # band_prome4 = ndi.filters.uniform_filter(
        #     band_prome3, size=3, mode='nearest')
        
        [l,m]=band_prome.shape
        
        # for o in range(l):
        #     for p in range(m):
        #         if band_prome4[o,p]<-5:
        #             band_prome4[o,p]=-9999
        #         else:
        #             band_prome4[o,p]=band_prome4[o,p]
        
        #ref='%s'%(out_idw)
        props=get_geotiff_props(ref)
        out_file="%s"%(out_prome)
        array_to_geotiff(band_prome,'{}'.format(out_file),-9999,props)
        #se crea el mapa binario
        
        band_bin=np.zeros((band_prome.shape))
        
        
        for j in range(l):
            for k in range(m):
                if band_prome[j,k]<=0:
                    band_bin[j,k]=0
                else:
                    band_bin[j,k]=1
        
        out_file="%s"%(out_bina)
        array_to_geotiff(band_bin,'{}'.format(out_file),-9999,props)
        
#se elimina los archivos que ya no son necesarios para liberar espacio en disco
        rasterDs1=None
        rasterDs2=None
        rasterDs3=None
        rasterDs4=None
        rasterDs5=None
        remove('%s'%(out_idw))
        remove('%s'%(out_idnn))
        remove('%s'%(out_near))
        remove('%s'%(out_linear))
        remove('%s'%(out_average))
        # os.remove(in_points)
 
# fin=tm.time()
# tiempo=fin-ini
        
# print("El tiempo de ejecución es de:\n",tiempo/60, "minutos")




def reproyectar(nc,base,in_dem,out_dem,in_clasify,out_clasify,in_binario,
                out_binario,in_error,out_error):
    
    #ini=tm.time()
    path=os.getcwd()
    # Define el sistema de referencia de coordenadas del raster de salida
    srs_in=osr.SpatialReference()
    srs_in.ImportFromEPSG(4326)
    
    srs_out = osr.SpatialReference()
    srs_out.ImportFromEPSG(3116)
    
    base="%s.tif"%(base)
    base_info=get_geotiff_props(base) 
    # Define las dimensiones del raster
    x_min, x_max, y_min, y_max = (base_info['x_min'], base_info['x_max'],
                                   base_info['y_min'], base_info['y_max']) # Rango de coordenadas en metros
    
    x_size=base_info['width']
    y_size=base_info['height']
    #x_res=base_info['res_x']
    #y_res=base_info['res_y']
    
    in_ds='{}\%s.tif'.format(path)%(in_dem)
    out_ds='{}\%s.tif'.format(path)%(out_dem)
    in_clas='{}\%s.tif'.format(path)%(in_clasify)
    out_clas='{}\%s.tif'.format(path)%(out_clasify)
    
    gdal.Warp('%s'%(out_ds), '%s'%(in_ds),
                        srcSRS='%s'%(srs_in),outputBounds=[x_max,y_max,x_min,y_min],
                        width=x_size,height=y_size,
                        dstSRS='%s'%(srs_out),dstNodata=-9999)
    
    gdal.Warp('%s'%(out_clas), '%s'%(in_clas),
                          srcSRS='%s'%(srs_in),outputBounds=[x_max,y_max,x_min,y_min],
                          width=x_size,height=y_size,
                          dstSRS='%s'%(srs_out),dstNodata=-9999)
    
    #os.remove('%s'%(in_clas))
    #os.remove('%s'%(in_lidar))
     
    # for l in range (5,95,5):
    for i in range(nc):
        
        
        #out_fabdem='FABDEM.tif'
        in_bina='{}\%s_%s_%s.tif'.format(path)%(in_dem,in_binario,i)
        out_bina='{}\%s_%s_%s.tif'.format(path)%(in_dem,out_binario,i)
        in_errores='{}\%s_%s_%s.tif'.format(path)%(in_dem,in_error,i)
        out_errores='{}\%s_%s_%s.tif'.format(path)%(in_dem,out_error,i)
        
        gdal.Warp('%s'%(out_bina), '%s'%(in_bina),
                            srcSRS='%s'%(srs_in),outputBounds=[x_max,y_max,x_min,y_min],
                            width=x_size,height=y_size,
                            dstSRS='%s'%(srs_out),dstNodata=-9999)
        
        gdal.Warp('%s'%(out_errores), '%s'%(in_errores),
                            srcSRS='%s'%(srs_in),outputBounds=[x_max,y_max,x_min,y_min],
                            width=x_size,height=y_size,
                            dstSRS='%s'%(srs_out),dstNodata=-9999)
        
       
        
        #se elimina los archivos en WGS84 para liberar espacio en disco. 
        
        os.remove('%s'%(in_bina))
        os.remove('%s'%(in_errores))
       

#fin=tm.time()
#tiempo=fin-ini
        
#print("El tiempo de ejecución es de:\n",tiempo/60, "minutos")
# binario_smoth=gdal.ReprojectImage(args, kwargs)

def ajuste(nc,in_dem,in_clas,in_bina,in_error,dem_out):
    
    path=os.getcwd()
    #ini=tm.time()
    
    print("CARGANDO INPUTS: DEM,MAPA CLASIFICADO,MAPA ERRORES,\
          BINARIO ERRORES,MAR,PLAYA Y RIO, Y VOLVEROS ARRAYS")
          
    dem="{}/%s.tif".format(path)%(in_dem)
    ds_dem=geotiff_to_array(dem)
    clasify="{}/%s.tif".format(path)%(in_clas)
    ds_clas=geotiff_to_array(clasify)
    #ds_error=np.nan_to_num(ds_error,copy=True,nan=-9999,posinf=None,neginf=None)
    print("Extrayendo propiedades del DEM")
    props=get_geotiff_props(dem) 
    [n,m]=ds_dem.shape
    
    #esto se hace una sola vez y se guarda el mapa raster de estas clases para
    # usarlo en las demás procesos
    print("generando los raster de las clases 1,2 y 3 (mar,playa y rio)")
    ds_1=np.zeros(ds_dem.shape)
    ds_2=np.zeros(ds_dem.shape)
    ds_3=np.zeros(ds_dem.shape)
    
    for k in range(n):
        for l in range(m):
            if ds_clas[k,l]==1:
                ds_1[k,l]=ds_dem[k,l]
            else:
                ds_1[k,l]=0
            
            if ds_clas[k,l]==2:
                ds_2[k,l]=ds_dem[k,l]
            else:
                ds_2[k,l]=0
                
            if ds_clas[k,l]==3:
                ds_3[k,l]=ds_dem[k,l]
            else:
                ds_3[k,l]=0
    
    ds_clas_1_2_3=ds_1+ds_2+ds_3
    #out_file="clas_1_2_3.tif"
    #array_to_geotiff(clas_1_2_3,'{}'.format(out_file),-9999,props)
    #out_file="mar.tif"
    #array_to_geotiff(ds_1,'{}'.format(out_file),-9999,props)
    
    #out_file="playa.tif"
    #array_to_geotiff(ds_2,'{}'.format(out_file),-9999,props)
    
    #out_file="rio.tif"
    #array_to_geotiff(ds_3,'{}'.format(out_file),-9999,props)
    
    for p in range(nc):
        
        binario="{}/%s_%s.tif".format(path)%(in_bina,p)
        ds_binario=geotiff_to_array(binario)
        error="{}/%s_%s.tif".format(path)%(in_error,p)
        ds_error=geotiff_to_array(error)
        out_dem='{}/%s_%s.tif'.format(path)%(dem_out,p)
        print("Ejecutando condicionales para cada clase, y cada valor binario (0 y 1)")
        ds_4_1=np.zeros(ds_dem.shape)
        ds_4_0=np.zeros(ds_dem.shape)
        ds_5_1=np.zeros(ds_dem.shape)
        ds_5_0=np.zeros(ds_dem.shape)
        ds_6_1=np.zeros(ds_dem.shape)
        ds_6_0=np.zeros(ds_dem.shape)
        ds_7_1=np.zeros(ds_dem.shape)
        ds_7_0=np.zeros(ds_dem.shape)
        
        # if (ds_lidar[i,j]!=9999 and ds_error[i,j]!=-9999):
        for i in range(n):
            for j in range(m):
                if (ds_dem[i,j]!=9999 and ds_error[i,j]!=-9999):
                    
                    if (ds_binario[i,j]==0 and ds_clas[i,j]==4 ):
                        ds_4_0[i,j]=ds_dem[i,j]-abs(ds_error[i,j])
                    else:
                        ds_4_0[i,j]=0
                        
                    if (ds_binario[i,j]==1 and ds_clas[i,j]==4 ):
                        ds_4_1[i,j]=ds_dem[i,j]+abs(ds_error[i,j])
                    else:
                        ds_4_1[i,j]=0
                    
                    if (ds_binario[i,j]==0 and ds_clas[i,j]==5 ):
                        ds_5_0[i,j]=ds_dem[i,j]-abs(ds_error[i,j])
                    else:
                        ds_5_0[i,j]=0
                        
                    if (ds_binario[i,j]==1 and ds_clas[i,j]==5 ):
                        ds_5_1[i,j]=ds_dem[i,j]+abs(ds_error[i,j])
                    else:
                        ds_5_1[i,j]=0
                    
                    if (ds_binario[i,j]==0 and  ds_clas[i,j]==6):
                        ds_6_0[i,j]=ds_dem[i,j]-abs(ds_error[i,j])
                    else:
                            ds_6_0[i,j]=0 
                            
                    if (ds_binario[i,j]==1 and  ds_clas[i,j]==6):
                         ds_6_1[i,j]=ds_dem[i,j]+abs(ds_error[i,j])
                    else:
                            ds_6_1[i,j]=0
                    
                    if (ds_binario[i,j]==0 and  ds_clas[i,j]==7):
                        ds_7_0[i,j]=ds_dem[i,j]-abs(ds_error[i,j])
                    else:
                            ds_7_0[i,j]=0 
                            
                    if (ds_binario[i,j]==1 and  ds_clas[i,j]==7):
                         ds_7_1[i,j]=ds_dem[i,j]+abs(ds_error[i,j])
                    else:
                            ds_7_1[i,j]=0
                        
        #guardar los de cada clase solo para probar el proceso, despues que se 
        # sepa que está funcionando bien,no es necesario guardarlo, guardar solo el 
        #raster final
        """
        print("guardadno los raster de cada clase")
        out_file="viv_0.tif"
        array_to_geotiff(ds_4_0,'{}'.format(out_file),-9999,props) 
        out_file="veg_0.tif"
        array_to_geotiff(ds_5_0,'{}'.format(out_file),-9999,props)           
        out_file="suelo_0.tif"
        array_to_geotiff(ds_6_0,'{}'.format(out_file),-9999,props)       
        out_file="via_0.tif"
        array_to_geotiff(ds_7_0,'{}'.format(out_file),-9999,props)
        out_file="viv_1.tif"
        array_to_geotiff(ds_4_1,'{}'.format(out_file),-9999,props)          
        out_file="veg_1.tif"
        array_to_geotiff(ds_5_1,'{}'.format(out_file),-9999,props)            
        out_file="suelo_1.tif"
        array_to_geotiff(ds_6_1,'{}'.format(out_file),-9999,props)          
        out_file="via_1.tif"
        array_to_geotiff(ds_7_1,'{}'.format(out_file),-9999,props)
        """
        print("Creando el modelo Corregido")
        dem_corr=ds_clas_1_2_3+ds_4_0+ds_4_1+ds_5_0+ds_5_1+ds_6_0+ds_6_1+ds_7_0+ds_7_1
        out_file="%s"%(out_dem)
        array_to_geotiff(dem_corr,'{}'.format(out_file),-9999,props)
        ds_binario=None
        ds_error=None
        os.remove('%s'%(binario))
        os.remove('%s'%(error))
        
    #print(o)
    print(p)
    
    #end=tm.time()
    #total=end-ini
    #print("TERMINA EL PROCESO","\n el tiempo de ejecucion es de:",total/60, "minutos")

def reproyect_results(nc,in_dem1,out_dem1,in_dem2,out_dem2):
    
    path=os.getcwd()
    # Define el sistema de referencia de coordenadas del raster de salida
    srs_in=osr.SpatialReference()
    srs_in.ImportFromEPSG(3116)
    
    srs_out = osr.SpatialReference()
    srs_out.ImportFromEPSG(4326)
    
    # for l in range (5,95,5):
    for i in range(nc):
        
        in_ds1='{}\%s_%s.tif'.format(path)%(in_dem1,i)
        out_ds1='{}\%s_%s.tif'.format(path)%(out_dem1,i)
        gdal.Warp('%s'%(out_ds1), '%s'%(in_ds1),
                            srcSRS='%s'%(srs_in),dstSRS='%s'%(srs_out),
                            dstNodata=-9999)
        
        in_ds2='{}\%s_%s.tif'.format(path)%(in_dem2,i)
        out_ds2='{}\%s_%s.tif'.format(path)%(out_dem2,i)
        
        gdal.Warp('%s'%(out_ds2), '%s'%(in_ds2),
                            srcSRS='%s'%(srs_in),dstSRS='%s'%(srs_out),
                            dstNodata=-9999)
        
        os.remove(in_ds1)
        os.remove(in_ds2)
def extract_z2(nc,in_points,out_points,dem1,dem2):
    
    path=os.getcwd()
    #open point shapefile
    #for i in range (5,95,5):
    for j in range(nc):
        pointData = gpd.read_file('{}//%s_%s.shp'.format(path)%(in_points,j))
        print(pointData.crs)
        #pointData.plot()
        
        #open raster file
        ds1 = rasterio.open('{}//%s_%s.tif'.format(path)%(dem1,j))
        print(ds1.crs)
        print(ds1.count)
        
        ds2 = rasterio.open('{}//%s_%s.tif'.format(path)%(dem2,j))
        print(ds2.crs)
        print(ds2.count)
        
        #show point and raster on a matplotlib plot
        # fig, ax = plt.subplots(figsize=(12,12))
        # pointData.plot(ax=ax, color='orangered')
        # show(ds1, ax=ax)
        
        # fig, ax = plt.subplots(figsize=(12,12))
        # pointData.plot(ax=ax, color='orangered')
        # show(ds1, ax=ax)
        
        #extract xy from point geometry
        
        for point in pointData['geometry']:
            print(point.xy[0][0],point.xy[1][0])
        
        #create empty list for each Z_DEM
        z_ds1=[]
        z_ds2=[]
        
        #extract point value from raster
        for point in pointData['geometry']:
            x = point.xy[0][0]
            y = point.xy[1][0]
            row, col = ds1.index(x,y)
            row1, col1 = ds2.index(x,y)
            z1=ds1.read(1)[row,col]
            z2=ds2.read(1)[row1,col1]
            z_ds1.append(z1)
            z_ds2.append(z2)
        
        #save each Z_DEM in a dataframe
        z_ds1=pd.DataFrame(z_ds1)
        z_ds2=pd.DataFrame(z_ds2)
        # agregate Z_DEMs to a point dataframe
        pointData=pointData.assign(DEM1_CORR=z_ds1,DEM2_CORR=z_ds2)
        pointData.to_file("{}//%s_%s.shp".format(path)%(out_points,j))

# def erase_shp(nc,points):
#      path=os.getcwd()
#      for i in range(nc):      
#          in_points='{}\%s_%s.shp'.format(path)%(points,i)
#          os.remove(in_points)
    
def visualize_results(dem,rmse1,rmse2,mae1,mae2,mse1,mse2,pbias1,pbias2):
    
    path=os.getcwd()
    MAE1=[]
    MAE2=[]
    MSE1=[]
    MSE2=[]
    PBIAS1=[]
    PBIAS2=[]
    RMSE1=[]
    RMSE2=[]
    
    # for i in range (nc):
    in_rmse='{}//%s.csv'.format(path)%(rmse1)
    in_mae='{}//%s.csv'.format(path)%(mae1)
    in_mse='{}//%s.csv'.format(path)%(mse1)
    in_pbias='{}//%s.csv'.format(path)%(pbias1)
    rmse=pd.read_csv(in_rmse)
    mae=pd.read_csv(in_mae)
    mse=pd.read_csv(in_mse)
    pbias=pd.read_csv(in_pbias)
    mae=mae['0']
    mse=mse['0']
    rmse=rmse['0']
    pbias=pbias['0']
    RMSE1=rmse
    MAE1=rmse
    PBIAS1=pbias
    in_rmse2='{}//%s.csv'.format(path)%(rmse2)
    in_mae2='{}//%s.csv'.format(path)%(mae2)
    in_mse2='{}//%s.csv'.format(path)%(mse2)
    in_pbias2='{}//%s.csv'.format(path)%(pbias2)
    rmse2=pd.read_csv(in_rmse2)
    mae2=pd.read_csv(in_mae2)
    mse2=pd.read_csv(in_mse2)
    #mse=(np.array(mse))
    pbias2=pd.read_csv(in_pbias2)
    mae2=mae2['0']
    mse2=mse2['0']
    rmse2=rmse2['0']
    pbias2=pbias2['0']
    RMSE2=rmse2
    MAE2=mae2
    MSE2=mse2
    PBIAS2=pbias2
    
    RMSE1=np.transpose(np.matrix(RMSE1))
    MAE1=np.transpose(np.matrix(MAE1))
    MSE1=np.transpose(np.matrix(MSE1))
    PBIAS1=np.transpose(np.matrix(PBIAS1))
        
    RMSE2=np.transpose(np.matrix(RMSE2))
    MAE2=np.transpose(np.matrix(MAE2))
    MSE2=np.transpose(np.matrix(MSE2))
    PBIAS2=np.transpose(np.matrix(PBIAS2))
    
    plt.plot(RMSE1,'o')
    plt.title('COMPARACIÓN DE RMSE EN %s EN LAS DISTINTAS ITERACIONES - DEM SIN AJUSTAR'%(dem))
    plt.ylabel('RMSE (m)')
    plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5])
   #plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],[#'95-5','90-10','85-15',
                                                               # '80-20','75-25','70-30',
                                                                #'65-35','60-40','55-45',
                                                                #'50-50','45-55','40-60',
                                                                #'35-65','30-70','25-75',
                                                                #'20-80','15-85','10-90'
                                                               #],rotation=90)
    #plt.xlabel('DISTRIBUCIÓN')
    plt.show()
    
    plt.plot(RMSE2,'o')
    plt.title('COMPARACIÓN DE RMSE EN %s EN LAS DISTINTAS ITERACIONES-VALIDACIÓN'%(dem))
    plt.ylabel('RMSE (m)')
    plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5])
    # plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],['95-5','90-10','85-15',
    #                                                            '80-20','75-25','70-30',
    #                                                            '65-35','60-40','55-45',
    #                                                            '50-50','45-55','40-60',
    #                                                            '35-65','30-70','25-75',
    #                                                            '20-80','15-85','10-90'
    #                                                            ],rotation=90)
    # plt.xlabel('DISTRIBUCIÓN')
    plt.show()
    # xticks([0, 1, 2], ['January', 'February', 'March'],
    #        rotation=20)
    plt.plot(MAE1,'o')
    plt.title('COMPARACIÓN DE MAE EN %s EN LAS DISTINTAS ITERACIONES - DEM SIN AJUSTAR'%(dem))
    plt.ylabel('MAE (m)')
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5])
    # plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],['95-5','90-10','85-15',
    #                                                            '80-20','75-25','70-30',
    #                                                            '65-35','60-40','55-45',
    #                                                            '50-50','45-55','40-60',
    #                                                            '35-65','30-70','25-75',
    #                                                            '20-80','15-85','10-90'
    #                                                            ],rotation=90)
    # plt.xlabel('DISTRIBUCIÓN')
    plt.show()
    
    plt.plot(MAE2,'o')
    plt.title('COMPARACIÓN DE MAE EN %s EN LAS DISTINTAS ITERACIONES - VALIDACIÓN'%(dem))
    plt.ylabel('MAE (m)')
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5])
    # plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],['95-5','90-10','85-15',
    #                                                            '80-20','75-25','70-30',
    #                                                            '65-35','60-40','55-45',
    #                                                            '50-50','45-55','40-60',
    #                                                            '35-65','30-70','25-75',
    #                                                            '20-80','15-85','10-90'
    #                                                            ],rotation=90)
    # plt.xlabel('DISTRIBUCIÓN')
    plt.show()
    
    plt.plot(MSE1,'o')
    plt.title('COMPARACIÓN DE MAE EN %s EN LAS DISTINTAS ITERACIONES - DEM SIN AJUSTAR'%(dem))
    plt.ylabel('MSE (m)')
    plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5])
    # plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],['95-5','90-10','85-15',
    #                                                            '80-20','75-25','70-30',
    #                                                            '65-35','60-40','55-45',
    #                                                            '50-50','45-55','40-60',
    #                                                            '35-65','30-70','25-75',
    #                                                            '20-80','15-85','10-90'
    #                                                            ],rotation=90)
    # plt.xlabel('DISTRIBUCIÓN')
    plt.show()
    
    plt.plot(MSE2,'o')
    plt.title('COMPARACIÓN DE MAE EN %s EN LAS DISTINTAS ITERACIONES - VALIDACIÓN'%(dem))
    plt.ylabel('MSE (m)')
    plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5])
    # plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],['95-5','90-10','85-15',
    #                                                            '80-20','75-25','70-30',
    #                                                            '65-35','60-40','55-45',
    #                                                            '50-50','45-55','40-60',
    #                                                            '35-65','30-70','25-75',
    #                                                            '20-80','15-85','10-90'
    #                                                            ],rotation=90)
    # plt.xlabel('DISTRIBUCIÓN')
    plt.show()
    
    plt.plot(PBIAS1,'o')
    plt.title('COMPARACIÓN DE PBIAS EN %s EN LAS DISTINTAS ITERACIONES - DEM SIN AJUSTAR'%(dem))
    plt.ylabel('PBIAS (%)')
    plt.yticks([-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])
    # plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],['95-5','90-10','85-15',
    #                                                            '80-20','75-25','70-30',
    #                                                            '65-35','60-40','55-45',
    #                                                            '50-50','45-55','40-60',
    #                                                            '35-65','30-70','25-75',
    #                                                            '20-80','15-85','10-90'
    #                                                            ],rotation=90)
    # plt.xlabel('DISTRIBUCIÓN')
    plt.show()
    
    plt.plot(PBIAS2,'o')
    plt.title('COMPARACIÓN DE PBIAS EN %s EN LAS DISTINTAS ITERACIONES - VALIDACIÓN'%(dem))
    plt.ylabel('PBIAS (%)')
    plt.yticks([-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.0,0.1,0.2,0.3,0.4,0.5])
    # plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],['95-5','90-10','85-15',
    #                                                            '80-20','75-25','70-30',
    #                                                            '65-35','60-40','55-45',
    #                                                            '50-50','45-55','40-60',
    #                                                            '35-65','30-70','25-75',
    #                                                            '20-80','15-85','10-90'
    #                                                            ],rotation=90)
    # plt.xlabel('DISTRIBUCIÓN')
    plt.show()
    plt.close('all')

def promedio_corregido(nc,patron,out_promedio):
    
    path=os.getcwd() 
    # Lista de nombres de archivos
    #lista_rasters = ["LIDAR1.tif", "LIDAR2.tif", "LIDAR3.tif"]  # Reemplaza con tus nombres de archivo
    #out_prome='%s.tif'%(out_promedio)
    # Patrón para buscar archivos raster específicos
    patron = "%s*"%(patron) 
    lista_rasters = glob.glob(f"{path}/{patron}")
    # lista_rasters=lista_rasters[0:2]
    # Variables para almacenar la suma y el número total de rasters
    suma = 0
    num_rasters = 0

    # Iterar sobre la lista de rasters
    for raster_nombre in lista_rasters:
        # Abrir el raster
        props=get_geotiff_props(raster_nombre)
        with rasterio.open(raster_nombre) as src:
            # Leer los datos del raster como un arreglo numpy
            data = src.read(1)  # Se asume que los datos están en la banda 1
            
            # Sumar los valores del raster al acumulador
            suma += data

            # Incrementar el contador de rasters
            num_rasters += 1
        #os.remove(raster_nombre)
    # Calcular el promedio dividiendo la suma total entre el número de rasters
    promedio = suma /nc
    
    out_file="%s.tif"%(out_promedio)
    array_to_geotiff(promedio,'{}'.format(out_file),-9999,props)
    
    for ds in lista_rasters:
        os.remove(ds)
        
def eliminar_shp(patron1,patron2):
    path=os.getcwd()
    patron1 = "%s*"%(patron1)
    patron2 = "%s*"%(patron2)
    lista1 = glob.glob(f"{path}/{patron1}")
    lista2 = glob.glob(f"{path}/{patron2}")
    for shp1 in lista1:
        os.remove(shp1)
    for shp2 in lista2:
        os.remove(shp2)