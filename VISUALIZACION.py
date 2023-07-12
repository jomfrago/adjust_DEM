# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:32:31 2023

@author: JOMFRAGO
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
# def visualize_results(dem,rmse1,rmse2,mae1,mae2,mse1,mse2,pbias1,pbias2):
path=os.getcwd()
dem='LIDAR'
rmse1='%s_RMSE1'%(dem)   
mse1='%s_MSE1'%(dem)
mae1='%s_MAE1'%(dem)  
pbias1='%s_PBIAS1'%(dem) 
rmse2='%s_RMSE2'%(dem)   
mse2='%s_MSE2'%(dem)  
mae2='%s_MAE2'%(dem)  
pbias2='%s_PBIAS2'%(dem)
 
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
mse=(np.array(mse))
pbias=pd.read_csv(in_pbias)
mae=mae['0']
mse=mse[:,1]
rmse=rmse['0']
pbias=pbias['0']
RMSE1=rmse
MAE1=mae
MSE1=mse
PBIAS1=pbias
in_rmse2='{}//%s.csv'.format(path)%(rmse2)
in_mae2='{}//%s.csv'.format(path)%(mae2)
in_mse2='{}//%s.csv'.format(path)%(mse2)
in_pbias2='{}//%s.csv'.format(path)%(pbias2)
rmse2=pd.read_csv(in_rmse2)
mae2=pd.read_csv(in_mae2)
mse2=pd.read_csv(in_mse2)
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

plt.show()

plt.plot(RMSE2,'o')
plt.title('COMPARACIÓN DE RMSE EN %s EN LAS DISTINTAS ITERACIONES-VALIDACIÓN'%(dem))
plt.ylabel('RMSE (m)')
plt.yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5])

plt.show()

plt.plot(MAE1,'o')
plt.title('COMPARACIÓN DE MAE EN %s EN LAS DISTINTAS ITERACIONES - DEM SIN AJUSTAR'%(dem))
plt.ylabel('MAE (m)')
plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5])

plt.show()

plt.plot(MAE2,'o')
plt.title('COMPARACIÓN DE MAE EN %s EN LAS DISTINTAS ITERACIONES - VALIDACIÓN'%(dem))
plt.ylabel('MAE (m)')
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5])

plt.show()

plt.plot(MSE1,'o')
plt.title('COMPARACIÓN DE MAE EN %s EN LAS DISTINTAS ITERACIONES - DEM SIN AJUSTAR'%(dem))
plt.ylabel('MSE (m)')
plt.yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5])

plt.show()

plt.plot(MSE2,'o')
plt.title('COMPARACIÓN DE MAE EN %s EN LAS DISTINTAS ITERACIONES - VALIDACIÓN'%(dem))
plt.ylabel('MSE (m)')
plt.yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5])

plt.show()

plt.plot(PBIAS1,'o')
plt.title('COMPARACIÓN DE PBIAS EN %s EN LAS DISTINTAS ITERACIONES - DEM SIN AJUSTAR'%(dem))
plt.ylabel('PBIAS (%)')
plt.yticks([-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7])

plt.show()

plt.plot(PBIAS2,'o')
plt.title('COMPARACIÓN DE PBIAS EN %s EN LAS DISTINTAS ITERACIONES - VALIDACIÓN'%(dem))
plt.ylabel('PBIAS (%)')
plt.yticks([-0.5,-0.4,-0.3,-0.2,-0.1,-0.05,0.0,0.1,0.2,0.3,0.4,0.5])

plt.show()
plt.close('all')
