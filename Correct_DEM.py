# -*- coding: utf-8 -*-
"""
Created on Sat May 27 13:17:26 2023

@author: JOMFRAGO
"""
import FUNCIONES as fn
import time as tm

ini=tm.time()
#se pone los nombres de los archivos de entrada y salida sin formatos
nc=30
t=20
dem1='LIDAR'
dem2='FABDEM'
fn.extract_z1('PUNTOS_AREA','PUNTOS','%s'%(dem1),'%s'%(dem2))

fn.division_puntos('PUNTOS',nc,t,'EVAL','VALID')

fn.calc_error(nc,'EVAL','ORTO','DEM1','%s_RMSE1'%(dem1),'%s_MAE1'%(dem1),
              '%s_MSE1'%(dem1),'%s_PBIAS1'%(dem1))

fn.calc_error(nc,'EVAL','ORTO','DEM2','%s_RMSE1'%(dem2),'%s_MAE1'%(dem2),
              '%s_MSE1'%(dem2),'%s_PBIAS1'%(dem2))

fn.mapa_error(nc,'BASE2','EVAL','DIF_DEM1','PROME','BINARIO','%s'%(dem1))

fn.mapa_error(nc,'BASE2','EVAL','DIF_DEM2','PROME','BINARIO','%s'%(dem2))

fn.reproyectar(nc,'BASE','%s'%(dem1),'%s1'%(dem1),'CLASIFY','CLASIFY1','BINARIO',
                'BINARIO1','PROME','PROME1')

fn.reproyectar(nc,'BASE','%s'%(dem2),'%s1'%(dem2),'CLASIFY','CLASIFY1','BINARIO',
                'BINARIO1','PROME','PROME1')

fn.ajuste(nc,'%s1'%(dem1),'CLASIFY1','%s_BINARIO1'%(dem1),'%s_PROME1'%(dem1),
          '%s_CORR'%(dem1))

fn.ajuste(nc,'%s1'%(dem2),'CLASIFY1','%s_BINARIO1'%(dem2),'%s_PROME1'%(dem2),
          '%s_CORR'%(dem2))

fn.reproyect_results(nc,'%s_CORR'%(dem1),'%s_CORR1'%(dem1),'%s_CORR'%(dem2),
                      '%s_CORR1'%(dem2))

fn.extract_z2(nc,'VALID','VALID','%s_CORR1'%(dem1),'%s_CORR1'%(dem2))

fn.calc_error(nc,'VALID','ORTO','DEM1_CORR','%s_RMSE2'%(dem1),'%s_MAE2'%(dem1),
              '%s_MSE2'%(dem1),'%s_PBIAS2'%(dem1))

fn.calc_error(nc,'VALID','ORTO','DEM2_CORR','%s_RMSE2'%(dem2),'%s_MAE2'%(dem2),
              '%s_MSE2'%(dem2),'%s_PBIAS2'%(dem2))


fn.visualize_results('%s'%(dem1),'%s_RMSE1'%(dem1),'%s_RMSE2'%(dem1),'%s_MAE1'%(dem1),
                     '%s_MAE2'%(dem1),'%s_MSE1'%(dem1),'%s_MSE2'%(dem1),
                     '%s_PBIAS1'%(dem1),'%s_PBIAS2'%(dem1))

fn.visualize_results('%s'%(dem2),'%s_RMSE1'%(dem2),'%s_RMSE2'%(dem2),'%s_MAE1'%(dem2),
                     '%s_MAE2'%(dem2),'%s_MSE1'%(dem2),'%s_MSE2'%(dem2),
                     '%s_PBIAS1'%(dem2),'%s_PBIAS2'%(dem2))

fn.promedio_corregido(nc,'%s_CORR1'%(dem1),'%s_CORR_PROME'%(dem1))
fn.promedio_corregido(nc,'%s_CORR1'%(dem2),'%s_CORR_PROME'%(dem2))

fn.eliminar_shp('EVAL_','VALID_')

fin=tm.time()
tiempo=fin-ini
        
print("El tiempo de ejecución es de:\n",tiempo/60, "minutos")
