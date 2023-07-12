# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:34:10 2023

@author: JOMFRAGO
"""

   
import geopandas as gpd
import numpy as np
import pandas as pd
puntos=gpd.read_file('PUNTOS.shp')
real=puntos['ORTO']
real=np.transpose(np.matrix(real))
lidar=puntos['LIDAR_CORR']
fabdem=puntos['FABDEM_COR']
res=pd.DataFrame(columns=['LIDAR','FABDEM'])
res['LIDAR']=lidar
res['FABDEM']=fabdem
res=(np.matrix(res))
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
#metri_mae[i,]=mae
#mae=pd.DataFrame(metri_mae)
#metri_rmse[i,]=rmse
#rmse=pd.DataFrame(metri_rmse)
#metri_pbias[i,]=pbias
#pbias=pd.DataFrame(metri_pbias)