
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
import datetime
import csv

# ## Valores faltantes

# In[2]:


def valoresFaltantes(data):
    tam = len(data)
    num_nans = round(round(data.isnull().sum())/float(tam)*100) 
#     print("Porcentaje de valores faltantes",num_nans,"tamano de la serie de tiempo",tam)
    return num_nans


# ## Outliers

# In[3]:
def calcularOutliers(nombre,original=0,faltantes=0,outliers=0,SNR_dB=0):
    ts = pd.DataFrame()
    tam = 0
    if original == 1:
        filename = "st originales/"+nombre+".dat"
        ts_aux = pd.read_csv(filename)
        ts_aux = ts_aux.dropna()
        tam = len(ts_aux)
        if tam > 20000:
            ts_aux = ts_aux[0:20000]
        ts = pd.DataFrame({'col0': np.arange(len(ts_aux)),'col1': ts_aux['col1']})
    else:
        filename = "st perturbadas/"+nombre+"/"+nombre+"_"+str(faltantes)+"_"+str(outliers)+"_"+str(SNR_dB)+".csv"
        ts = pd.read_csv(filename)
        #ts = ts.dropna()
        tam = len(ts)
        if tam > 20000:
            ts = ts[0:20000]
#         print("TSSSS\n",ts)
#     print(ts)
    Inliers,Outliers,Pos = od.obtenerOutliers(ts,k=5,showPlot=0)
#     print(len(Outliers[1]))
    tam = len(ts)
    totalOutliers = round(len(Outliers)/float(tam)*100)
#     print("totalOutliers",totalOutliers)
    return totalOutliers
# calcularOutliers("duffing_oscillator",original=1)

# ## SNR

# In[4]:


def obtener_SNR(xn,x=[],t=[],filename='SNR.csv',showPlot=0):
    """
    Parameters:
           xn.- Signal + noise
           x.- Original signal
           t.- time
           filename.- donde guardar los SNR obtenidos
           showPlot.- muestra las graficas con 1 y con 0 no
    Return:
           SNR.- Signal to noise ratio 
    """
    # Estimate noise using robust estimate
    beq = pyasl.BSEqSamp()
    # Define order of approximation (use larger values such as 2,3, or 4 for
    # faster varying or less well sampled data sets; also 0 is a valid order)
    N = 2
    # Define 'jump parameter' (use larger values such as 2,3, or 4 if correlation
    # between adjacent data point is suspected)
    j = 1
    nstd, nstdstd = beq.betaSigma(xn['col1'], N, j, returnMAD=True)
    # Obteniendo la varianza apartir de la desviacion estandar
    var_noise = np.power(nstd,2)
    # Ma va a contener la senal
    Ma = []
    # Res contiene los residuos los cuales se consideran como el ruido
    Res = []
    SNRcalculado = 0
    for window in range(2,20):
#         print("Ancho de ventana = ",window)
        # Moving Average Medias Moviles
        Ma = xn['col1'].rolling(window=window,center=True).mean()
        # Residuos de la Media movil
        Res = xn['col1'] - Ma
        # Quitando los 
        Res = Res.dropna()
        # Obteniendo la varianza del ruido
        varianza_Noise_sin = np.var(Res)      
#         restd = np.std(Ma)
        # Obteniendo las frecuencias y el espectro de potencia o densidad espectral de potencia
        fsenal, Pxx_den_senal = signal.periodogram(Ma[~np.isnan(Ma)])
        fnoise, Pxx_den_noise = signal.periodogram(Res[~np.isnan(Res)])
        # Quitando nans de la densidad espectral de potencia tanto para la senal como para el ruido
        Pxx_den_senal = Pxx_den_senal[~np.isnan(Pxx_den_senal)]
        Pxx_den_noise = Pxx_den_noise[~np.isnan(Pxx_den_noise)]

        #Si la diferencia entre la varianza de pyastronomy y la varianza del ruido extraido es menor a 1e-3
        #Podemos decir que se ha encontrado el SNR o que no contiene SNR
        if np.abs(var_noise - varianza_Noise_sin) < 1e-3:
            # Si la ventana es de 2 quiere decir que la serie no contiene ruido
            if window == 2:
                #print("No contiene ruido")
                filename.write(",,No hay ruido\n")
                SNRcalculado = "No hay ruido"
                break
            # Esta x se utilizo para las pruebas
            if len(x) > 0:
                SNRcalculado = 10*np.log10(np.mean(Pxx_den_senal)/np.mean(Pxx_den_noise))
                #print("\nSe redujo el ruido con una ventana de",window)
                #print("SNR detectado",SNRcalculado)
#                 print("SNR mediaS/desvN",1/(np.mean(Ma)/np.std(Res)))
                filename.write(str(SNRcalculado)+'\n')
                if showPlot:
                    od.grafica2D(t,Ma,t,x,
                             title1='Ruido reducido',title2='Original')
                    od.grafica2D(t,Ma,t,xn['col1'],
                             title1='Ruido reducido',title2='Ruido')
            else:
                # Se calcula el SNR = Psignal/Pnoise donde P es la potencia media https://en.wikipedia.org/wiki/Signal-to-noise_ratio
                SNRcalculado = 10*np.log10(np.mean(Pxx_den_senal)/np.mean(Pxx_den_noise))
                #print("\nSe redujo el ruido con una ventana de",window)
                #print("SNR detectado",SNRcalculado)
                filename.write(str(SNRcalculado)+'\n')
#                 print("SNR mediaS/desvN",1/(np.mean(Ma)/np.std(Res)))
                if showPlot:
                    od.grafica2D(t,Ma,t,xn['col1'],
                             title1='Noise reduced',title2='Noise signal')
            break
        if window == 19:
            SNRcalculado = 10*np.log10(np.mean(Pxx_den_senal)/np.mean(Pxx_den_noise))
            filename.write(str(SNRcalculado)+',,,Max\n')
    return SNRcalculado


#Funcion para graficar el periodograma
# import matplotlib.pyplot as plt
def plot_periodogram(df_x,title,legend,showPlot=0):
    #Obtenemos la frecuencia y densidad espectral de potencia o espectro de potencia.
    f, Pxx_den = signal.periodogram(df_x['col1'].T)
    Pxx_den = np.ravel(Pxx_den)
    Pxx_den[0] = np.nan
    #print(Pxx_den[0:10])
    if showPlot:
        p = bokeh.plotting.figure(title=title, y_axis_type="log",
               background_fill_color="#fafafa")
        p.line(x=f, y=np.ravel(Pxx_den), legend=legend, line_color="blue")
        bokeh.plotting.show(p)
    return f, Pxx_den

def variacion(Pxx_den):
    # Esta funcion recibe el espectro de potencia
    # El indice es el del elemento mas grande
    indice = np.argmax(Pxx_den[~np.isnan(Pxx_den)])
    # Si el indice se encuentra delante del 10% de la posicion quiere decir que no se puede calcular el SNR
    if indice > len(Pxx_den)*0.1:
        return False
    else:
        return True

def plotseries(tipo,path="",ts=[]):
    # Funcion para plotear las series
    if tipo == 0:
        od.grafica2D(np.arange(len(ts)),ts,title1='TS')
    else:
        ts = pd.read_csv(path)
#         ts = ts.dropna()
        od.grafica2D(np.arange(len(ts)),ts['col1'],title1='TS')


# In[5]:


def SNRTsPruebas(nombre,prueba=0,SNR_dB=-1,showPlot=0):
    path = "st con ruido/"+nombre+str(SNR_dB)
    numeroDeSeries = 1
    if prueba:
        numeroDeSeries = 10
    SNRObtenido = 0
    if SNR_dB == -1:
        #print("entra",os.path.exists)
        if not os.path.exists:
            #print("entra")
            os.mkdir(path)
        numeroDeSeries = 1
    filename = path+"/snr "+nombre+str(SNR_dB)+".csv"
    csv_file = open(filename, "w")  
    columnTitleRow = "SNR detectado, SNR no detectado, No contiene SNR, Prueba con un SNR = "+str(SNR_dB)+'\n'
    csv_file.write(columnTitleRow)
    tiempoGlobal = time.time()
    for i in range(0,numeroDeSeries):
        tiempoInicioTarea = time.time()
        #print("i",i)
        if SNR_dB != -1:
            ts = pd.read_csv("st con ruido/"+nombre+str(SNR_dB)+"/ts "+str(i)+".csv")
        else:
            #Read the file 
            ts = pd.read_csv("st originales/"+nombre+".dat")
            ts = ts
            ts = ts.dropna()
#         od.grafica2D(np.arange(len(ts)),ts['col1'],title1=nombre+" SNR "+str(SNR_dB))    
        f_ts, Pxx_den_ts = plot_periodogram(ts,nombre,nombre,showPlot=showPlot)
        varia = variacion(Pxx_den_ts)
        if varia:
            #Obtener el SNR
            SNRObtenido = obtener_SNR(ts,t=np.arange(len(ts)),filename=csv_file,showPlot=showPlot)
        else:
            #print("Varia rapidamente")
            SNRObtenido = "No se puede calcular SNR"
            csv_file.write(",No se puede calcular SNR,\n")
        tiempoFinalTarea = time.time()
        #print("Tiempo total ",tiempoFinalTarea-tiempoGlobal,"Tiempo tarea ",tiempoFinalTarea-tiempoInicioTarea)
    columnTitleRow = "=PROMEDIO(A2:A11)\n"
    csv_file.write(columnTitleRow)
    csv_file.close()
    return SNRObtenido


# In[6]:


def SNRTs(nombre,faltantes=0,outliers=0,SNR_dB=0,showPlot=0):
    tiempoGlobal = time.time()
    path = "st perturbadas/"+nombre+"/"+nombre+"_"+str(faltantes)+"_"+str(outliers)+"_"+str(SNR_dB)+".csv"
    SNRObtenido = 0
    ts = pd.read_csv(path)
    if len(ts) > 20000:
        ts = ts[0:20000]
    ts = ts.dropna()
    
    filename = "snr/"+nombre+str(SNR_dB)+".csv"
    csv_file = open(filename, "w")  
    columnTitleRow = "SNR detectado, SNR no detectado, No contiene SNR, Prueba con un SNR = "+str(SNR_dB)+'\n'
    csv_file.write(columnTitleRow)
    
#     ts = ts.dropna()
#     od.grafica2D(np.arange(len(ts)),ts['col1'],title1=nombre+" SNR "+str(SNR_dB))    
    f_ts, Pxx_den_ts = plot_periodogram(ts,nombre,nombre,showPlot=showPlot)
    varia = variacion(Pxx_den_ts)
    if varia:
        #Obtener el SNR
        SNRObtenido = obtener_SNR(ts,t=np.arange(len(ts)),filename=csv_file,showPlot=showPlot)
    else:
#         print("Varia rapidamente")
        SNRObtenido = "No se puede calcular SNR"
    csv_file.close()
    tiempoFinalTarea = time.time()
    print("Tiempo total SNRTs",tiempoFinalTarea-tiempoGlobal)
    return SNRObtenido


# ## Caos

# In[7]:


def calcularCaosPrueba(nombre,SNR_dB=-1):
    # Funcion para calcular el Caos usando la biblioteca nolds y el algoritmo de Rosenstein
#     tiempoInicial = time.time()
    filename = ""
    # Revisando si la st está contaminada con ruido o es la original
    if SNR_dB != -1:
        filename = "st con ruido/"+nombre+str(SNR_dB)+"/ts 0"+".csv"
    else:
        filename = "st originales/"+nombre+".dat"
    # Se carga la ts
    ts = pd.read_csv(filename)
    tamano = len(ts)
    # Se limita a que a lo maximo el tamano de la ts sea 20000 
    if tamano > 20000:
        tamano = 20000
    ts = ts[0:tamano]
#     print("TS\n\n",ts['col1'].shape)
    # Se calcula el exponente
    expo = nolds.lyap_r(np.ravel(ts['col1']))
    #expo = max(nolds.lyap_e(ts['col1']))
#     print("Tiempo total ",time.time()-tiempoInicial)
    return expo

def calcularCaos(nombre,faltantes=0,outliers=0,SNR_dB=0,showPlot=0):
    tiempoGlobal = time.time()
    path = "st perturbadas/"+nombre+"/"+nombre+"_"+str(faltantes)+"_"+str(outliers)+"_"+str(SNR_dB)+".csv"
    ts = pd.read_csv(path)
    if len(ts) > 20000:
        ts = ts[0:20000]
    ts = ts.dropna()
    #print("CAOS TS",ts['col1'])
    expo = nolds.lyap_r(np.ravel(ts['col1']))
    #expo = max(nolds.lyap_e(ts['col1']))
#     print("Tiempo total ",time.time()-tiempoInicial)
    return expo


# ## Obtener caracteristicas

# In[11]:


def obtenerCaracteristicas(nombreTs,showPlot=0):
    niveles = [0,5,10,15,20,25]
    tam = 0
    faltantes = 0
    outliers = 0
    snr = 0
    caos = 0
    
    tiempoInicial = time.time()
    #####
    
    filename = "caracteristicas/"+nombreTs+str(datetime.datetime.now())+".csv"
    csv_file = open(filename, "w")
    columnTitleRow = "Tamano ts,Va_Fa,Out,SNR_In,Valores Faltantes,Outliers,SNR,Caos\n"
    csv_file.write(columnTitleRow)
    csv_file.close()    
    for i in niveles:
        for j in niveles:
            for k in niveles:
                with open(filename, 'a') as csv_file:
                    writer = csv.writer(csv_file)
                    #if i != 0:
                    print("calculando   ",nombreTs,"_",i,"_",j,"_",k)
                    # filename = "st con ruido/seno5/ts 0.csv"
                    filename2 = "st perturbadas/"+nombreTs+"/"+nombreTs+"_"+str(i)+"_"+str(j)+"_"+str(k)+".csv"
                    ts_aux = pd.read_csv(filename2)
                    ts_aux = ts_aux['col1']
                    tam = len(ts_aux)
                    if tam > 20000:
                        tam = 20000
                        ts_aux = ts_aux[0:20000]
                    # Obtener los valores faltantes
                    #if i != 0:
                    faltantes = valoresFaltantes(ts_aux)
                    # Obtener los outliers
                    #if j != 0:
                    outliers = calcularOutliers(nombreTs,original=0,faltantes=i,outliers=j,SNR_dB=k)
                    #if k != 0:                    
                    snr = SNRTs(nombreTs,i,j,k,showPlot=0)
                    caos = calcularCaos(nombreTs,i,j,k,showPlot=0)
                    print("Se obtubieron",nombreTs,"_",faltantes,"_",outliers,"_",snr)
                    row = [tam,i,j,k,faltantes,outliers,snr,caos]
                    #writer.write(str(tam)+","+str(i)+","+str(j)+","+str(k)+","+str(faltantes)+","+str(outliers)+","+str(snr)+","+str(caos)+'\n')
                    writer.writerow(row)
                    # Ruta donde se va a guardar la ts
                csv_file.close()

    print("Tiempo total ",time.time()-tiempoInicial)


# In[13]:


#SC = ["seno","henon","lorenz","rossler","chen_system","duffing_oscillator"]
#SC = ["halvorsen_attractor","rucklidge_attractor","shawn_van_der_pol_oscillator","simplest_cubic_flow",
#      "simplest_linear_flow"]
SC = ["chen_system"]
#SC = ["halvorsen_attractor"]
for i in SC:
    print(i)
    obtenerCaracteristicas(i,showPlot=0)
