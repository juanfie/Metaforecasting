
# coding: utf-8

# # Perturbar Series de Tiempo
# ### 1.- Quitar valores en %
# ### 2.- Agregar Outliers en %
# ### 3.- Agregar SNR en dB

# In[1]:


import os, sys
sys.path.insert(0, '/home/victort/Documentos/SEMESTRE 17-18 MAESTRIA/TESIS/Deteccion Outliers')
import outliersDetection as od
import requests, pandas as pd, numpy as np
import time, json
import bokeh
from scipy import signal, fft
from statsmodels.tsa.stattools import adfuller, acf, pacf
import statsmodels.graphics as smg
#get_ipython().run_line_magic('matplotlib', 'inline')
from bokeh.plotting import gridplot, figure, output_file, output_notebook, show
from datetime import datetime
#from __future__ import print_function
from PyAstronomy import pyasl
from pylab import *
# Biblioteca para calcular los exponentes de Lyapunov
import nolds 


# ## Valores faltantes

# In[2]:


def quitarValores(ts,porcentaje):
    cambios = []
    tam = len(ts)
    elementos = round((porcentaje/100.0)*tam)
#     print("elementos",elementos)
    indices = []
    for i in range(elementos):
        k = 0
        while k == 0:
            r=np.random.randint(tam)
            if r not in indices: 
                indices.append(r)
                k = 1
#     print("FALTANTES",indices)
    for i in indices:
        ts[i] = np.nan
    return ts


# ## Outliers

# In[28]:


def agregarOutliers(ts,porcentaje):
    cambios = []
    tam = len(ts)
    elementos = round((porcentaje/100)*tam)
#     print("elementos",elementos)
    indices = []
    for i in range(elementos):
        k = 0
        while k == 0:
            r=np.random.randint(tam)
            if r not in indices and ts[r] != np.nan: 
                indices.append(r)
                k = 1
#     print("OUTLIERS INDICES",indices)
#     media = np.mean(ts)
#     sigma = np.std(ts)*5
    for i in indices:
        ts[i] += ts[i]*2
#         print("\nts_i despues",ts[i])
#         else:
#             alpha = np.random.rand()**np.random.randint(low=2,high=5)
#             ts[i] = media*((sigma+alpha))
    return ts


# ## SNR

# In[4]:


# x = A*np.cos(2*np.pi*1000*t) + a*np.sin(2*np.pi*2000*t) + s*np.random.normal(len(t));
def funcionSeno(fs, f, T=2):
    fs_ = fs #100KHz frecuencia de muestreo
    f_ = f #1KHz frecuencia de la senal
    t_ = np.arange(0,2*(1/f_),1/fs_)
#     print(len(t_))
    x_ = np.sin(T*np.pi*f_*t_)
    return x_, t_

def agregarSNR(x,SNR_dB,t_):
    #https://www.gaussianwaves.com/gaussianwaves/wp-content/uploads/2015/06/How_to_generate_AWGN_noise.pdf
    #Matlab / Octave tiene una función incorporada llamada -awgn () con la que se puede agregar 
    #un ruido blanco para obtener la señal deseada a ruido (SNR). 
    #El uso principal de esta función es agregar AWGN a una señal limpia (SNR infinita) para obtener una señal 
    #tamano de la serie
    L = len(x)
    #SNR a escala lineal
    SNR = np.power(10,(SNR_dB/10))
    #Calcular la energía
    Esym = np.sum(np.power(abs(x),2))/(L);
    #Encontrar la densidad espectral de ruido
    N0=Esym/SNR;
    #Desviación estándar para el Ruido
    noiseSigma =np.sqrt(N0);
    #Creando el ruido
    n = np.random.normal(0,noiseSigma,L)
#     print("SNR medido en dB",validarSNRinyectado(x, n,t_))
#     print("x\n\n",x,"\n\n\n n \n\n\n",n)
    return x + n

#Funcion para calcular el SNR antes de inyectarlo
def validarSNRinyectado(sig, noise,dt):
    Signal = np.sum(np.abs(np.fft.fft(sig)*dt)**2)/len(np.fft.fft(sig))
    Noise = np.sum(np.abs(np.fft.fft(noise)*dt)**2)/len(np.fft.fft(noise))
#     print("SNR medido lineal",Signal/Noise)
    return (10 * np.log10(Signal/Noise))


# In[5]:


def plotseries(tipo,path="",ts=[]):
    # Funcion para plotear las series
    if tipo == 0:
        od.grafica2D(np.arange(len(ts)),ts,title1='TS')
    else:
        ts = pd.read_csv(path)
#         ts = ts.dropna()
        od.grafica2D(np.arange(len(ts)),ts['col1'],title1='TS')


def crearStPerturbada(nombreTs,showPlot=0):
    #nombreTs: Nombre de la serie de tiempo
    #cuantasSeries: Numero de series independientes
    niveles = [0,5,10,15,20,25]
    ###
    # print(ts[0:20])
    # Primer nivel para quitar valores
    for i in niveles:
        for j in niveles:
            for k in niveles:
#                 print("nombre",nombreTs,"_",i,"_",j,"_",k)
                ##BLOQUE PARA ABRIR Y QUITAR VALORES
                # filename = "st con ruido/seno5/ts 0.csv"
                filename = "st originales/"+nombreTs+".dat"
                ts = pd.read_csv(filename)
                ts = ts['col1']
                tamano = len(ts)
                if tamano > 20000:
                    ts = ts[0:20000]
                if i != 0:
        #             print("valores %",i)
                    ts = quitarValores(ts,i)
                
                #BLOQUE PARA AGREGAR OUTLIERS
                # valoresFaltantes(ts_aux2)
                if j != 0:
                    ts = agregarOutliers(ts,j)
                
                #BLOQUE PARA AGREGAR SNR
                # Ruta donde se va a guardar la ts
                path = "st perturbadas/"+nombreTs
                # Revisando que exista el directorio si no existe lo crea
                if not os.path.exists(path): 
                    os.mkdir(path)
                # Crea el archivo .csv donde se guardara la ts
                csv_file2 = open(path+"/"+nombreTs+'_'+str(i)+'_'+str(j)+'_'+str(k)+'.csv', "w")
#                 print("Datos faltantes",i,"% Outliers",j,"% SNR",k,"dB")
                if k != 0:
                    SNR_dB = k
                    # Usando la funcion agregarSNR para inyeectar el ruido
                    xn_ = agregarSNR(ts,SNR_dB,np.arange(len(ts)))
                    xn2 = pd.DataFrame({'col0': np.arange(len(xn_)),'col1': xn_})
                    csv_file2.write(xn2.to_csv(index=False))
                    csv_file2.close()
                    if showPlot: plotseries(0,ts=xn_)
                else:
                    xn2 = pd.DataFrame({'col0': np.arange(len(ts)),'col1': ts})
                    csv_file2.write(xn2.to_csv(index=False))
                    csv_file2.close()
                    if showPlot: plotseries(0,ts=ts)
    print("Termino")


# In[29]:


SC = ["henon","lorenz","rossler","duffing_oscillator","halvorsen_attractor",
      "rucklidge_attractor","shawn_van_der_pol_oscillator","simplest_cubic_flow","simplest_linear_flow"]
# SC = ["seno"]
for i in SC:
    crearStPerturbada(i,showPlot=0)


# In[30]:


plotseries(1,"st perturbadas/seno/seno_0_25_0.csv")

