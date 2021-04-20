
# coding: utf-8

# # Pruebas de Ruido con S.T. sintéticas
# 
# Los decibeles expresan una relación de poder, no una cantidad. Dicen cuántas veces más (dB positivo) o menos (dB negativo) pero no cuánto en términos absolutos. Los decibeles son logarítmicos, no lineales. Por ejemplo, 20 dB no es el doble de la relación de potencia de 10 dB.
# 
# La ecuación para decibelios es:
# 
# $A = 10 * log10 (P2 / P1) (dB)$
# 
# donde P1 es la potencia que se mide y P1 es la referencia con la que se compara P2.
# Para convertir de la medida de decibelios a la relación de potencia:
# 
# $P2 / P1 = 10^{(A / 10)}$

# In[1]:


import os, sys
import requests, pandas as pd, numpy as np
import time, json
import datetime
import csv
from psr import psr
from psr import methods

def fillnans(ts):
    index_missing = []
    num_nans = ts.isnull().sum()
#    index_nan = ts.isnull()
    index_missing = ts.isnull().nonzero()[0]
    #print("num_nans",num_nans,"\index_missing\n",index_missing)
    for i in index_missing:
        if i > 0:
            ts[i] = ts[i-1]
        else:
            ts[i] = np.mean(ts)
    #print("ts.isnull().sum() salida",ts.isnull().sum())
    return ts

def crearbd(nombreTs):
    niveles = [0,5]
    filename = "parametrosbd/"+nombreTs+str(datetime.datetime.now())+".csv"
    csv_file = open(filename, "w")
    columnTitleRow = "Tamano ts,Va_Fa,Out,SNR_In,m,tau,tiempo,\n"
    csv_file.write(columnTitleRow)    
    csv_file.close()    
    tiempoGlobal = time.time()
    for i in niveles:
        for j in niveles:
            for k in niveles:
                if i == 0:
                    print("Creando db   ",nombreTs,"_",i,"_",j,"_",k)
                    tiempoPorSerie = time.time()
                    with open(filename, 'a') as csv_file:
                        writer = csv.writer(csv_file)
                        filename2 = "st perturbadas/"+nombreTs+"/"+nombreTs+"_"+str(i)+"_"+str(j)+"_"+str(k)+".csv"
                        ts = pd.read_csv(filename2)
                        ts = ts['col1']
                        if ts.isnull().sum() > 0:
                            ts = fillnans(ts)
                        dm = methods.DeterministicMethod(ts)
                        m, tau = dm.deterministic_method(2, 7)
                        p = psr.PSR(ts = np.array(ts), m= m, tau=tau)
                        db = p.get_database(path="st perturbadas/"+nombreTs+"/"+nombreTs+"_"+str(i)+"_"+str(j)+"_"+str(k)+'_bd.csv')
                        row = [len(ts),i,j,k,m,tau]
                        writer.writerow(row)
                        # Ruta donde se va a guardar la ts
                    csv_file.close()
                    print("Tiempo por serie",time.time()-tiempoPorSerie)
    print("termino tiempo total",time.time()-tiempoGlobal)


#SC = ["henon","lorenz","rossler"]
#SC = ["chen_system","duffing_oscillator","halvorsen_attractor"]
#SC = ["rucklidge_attractor","shawn_van_der_pol_oscillator","simplest_cubic_flow","simplest_linear_flow"]
SC = ["seno2"]
for i in SC:
    print(i)
    crearbd(i)


# Para obtener los parametros m y tau de la series de tiempo tardo de 2:30 am termino a las 12:24 pm en total 10 horas para crear las bd
