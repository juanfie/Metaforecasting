
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


# In[2]:


# x = A*np.cos(2*np.pi*1000*t) + a*np.sin(2*np.pi*2000*t) + s*np.random.normal(len(t));
def funcionSeno(fs, f, T=2):
    fs_ = fs #100KHz frecuencia de muestreo
    f_ = f #1KHz frecuencia de la senal
    t_ = np.arange(0,2*(1/f_),1/fs_)
    print(len(t_))
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
    print("SNR medido en dB",validarSNRinyectado(x, n,t_))
#     print("x\n\n",x,"\n\n\n n \n\n\n",n)
    return x + n

#Funcion para calcular el SNR antes de inyectarlo
def validarSNRinyectado(sig, noise,dt):
    Signal = np.sum(np.abs(np.fft.fft(sig)*dt)**2)/len(np.fft.fft(sig))
    Noise = np.sum(np.abs(np.fft.fft(noise)*dt)**2)/len(np.fft.fft(noise))
    print("SNR medido lineal",Signal/Noise)
    return (10 * np.log10(Signal/Noise))


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
        print("Ancho de ventana = ",window)
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
                print("No contiene ruido")
                filename.write(",,No SNR\n")
                SNRcalculado = "No SNR"
                break
            # Esta x se utilizo para las pruebas
            if len(x) > 0:
                SNRcalculado = 10*np.log10(np.mean(Pxx_den_senal)/np.mean(Pxx_den_noise))
                print("\nSe redujo el ruido con una ventana de",window)
                print("SNR detectado",SNRcalculado)
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
                print("\nSe redujo el ruido con una ventana de",window)
                print("SNR detectado",SNRcalculado)
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
    f, Pxx_den = signal.periodogram(df_x.T)
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

def plotseries(path):
    # Funcion para plotear las series
    ts = pd.read_csv(path)
    ts = ts.dropna()
    od.grafica2D(np.arange(len(ts)),ts['col1'],title1='TS')


def crearStRuidosa(nombreTs,cuantasSeries=1):
    #nombreTs: Nombre de la serie de tiempo
    #cuantasSeries: Numero de series independientes
    nivelesSNR = np.arange(5,26)
    for i in nivelesSNR:
        ts = pd.read_csv("st originales/"+nombreTs+".dat")
        ts = ts
        ts = ts.dropna()
        SNR_dB = i
        #print(ts[0:10])
        for j in arange(cuantasSeries):
            # Usando la funcion agregarSNR para inyeectar el ruido
            xn_ = agregarSNR(ts['col1'],SNR_dB,np.arange(len(ts)))
            # Ruta donde se va a guardar la ts
            path = "st con ruido/"+nombreTs+str(SNR_dB)
            # Revisando que exista el directorio si no existe lo crea
            if not os.path.exists(path): 
                os.mkdir(path)
            # Crea el archivo .csv donde se guardara la ts
            csv_file2 = open(path+'/ts '+str(j)+'.csv', "w")
            xn2 = pd.DataFrame({'col1': xn_})
            csv_file2.write(xn2.to_csv())
            csv_file2.close()
    print("Termino")


# ## Outliers

# In[27]:


def leerTs(nombre,original=0,SNR_dB=-1):
    ts = pd.DataFrame()
    if original == 1:
        filename = "st originales/"+nombre+".dat"
        ts_aux = pd.read_csv(filename)
        if len(ts_aux) > 20000:
            ts_aux = ts_aux[0:20000]
        ts = pd.DataFrame({'col0': np.arange(len(ts_aux)),'col1': ts_aux['col1']})
    else:
        filename = "st con ruido/"+nombre+str(SNR_dB)+"/ts 0"+".csv"
        ts = pd.read_csv(filename)
        if len(ts) > 20000:
            ts = ts[0:20000]        
#     print(ts)
    Inliers,Outliers,Pos = od.obtenerOutliers(ts,k=20,showPlot=0)
    print(len(Outliers[1]))
    return len(Outliers[1])
leerTs("duffing_oscillator",original=1)


# In[20]:


def SNRTs(nombre,prueba=0,SNR_dB=-1,showPlot=0):
    path = "st con ruido/"+nombre+str(SNR_dB)
    numeroDeSeries = 1
    if prueba:
        numeroDeSeries = 10
    SNRObtenido = 0
    if SNR_dB == -1:
        print("entra",os.path.exists)
        if not os.path.exists:
            print("entra")
            os.mkdir(path)
        numeroDeSeries = 1
    filename = path+"/snr "+nombre+str(SNR_dB)+".csv"
    csv_file = open(filename, "w")  
    columnTitleRow = "SNR detectado, SNR no detectado, No contiene SNR, Prueba con un SNR = "+str(SNR_dB)+'\n'
    csv_file.write(columnTitleRow)
    tiempoGlobal = time.time()
    for i in range(0,numeroDeSeries):
        tiempoInicioTarea = time.time()
        print("i",i)
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
            print("Varia rapidamente")
            SNRObtenido = "Nan"
            csv_file.write(",Nan,\n")
        tiempoFinalTarea = time.time()
        print("Tiempo total ",tiempoFinalTarea-tiempoGlobal,"Tiempo tarea ",tiempoFinalTarea-tiempoInicioTarea)
    columnTitleRow = "=PROMEDIO(A2:A11)\n"
    csv_file.write(columnTitleRow)
    csv_file.close()
    return SNRObtenido


# ## Caos

# In[21]:


def calcularCaos(nombre,SNR_dB=-1):
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


# In[28]:


import datetime
# Nombre de la serie de tiempo
# nameTs = "simplest_linear_flow"
# Graficar la serie de tiempo
# plotseries("st originales/"+nameTs+".dat")
# Crear n series de tiempo con SNR de 5 hasta 25
# crearStRuidosa(nameTs,10)

SC = ["lorenz"]
# SC = ["henon","lorenz","rossler","chen_system","duffing_oscillator","halvorsen_attractor",
#       "rucklidge_attractor","shawn_van_der_pol_oscillator","simplest_cubic_flow","simplest_linear_flow"]

# for i in range(4,5,1):
#     for j in SC:
#         if i == 4:
#             # Calcular SNR para st originales
#             SNRTs(j,showPlot=0)
#         else:
#             # Calcular SNR para st con ruido
#             SNRTs(j,i,showPlot=0)
# SNRTs(SC[0],15,showPlot=1)

tiempoInicial = time.time()
SNR_pasos = np.arange(4,26)
print(SNR_pasos)
for i in SC:
    caos = 0
    filename = "caracteristicas/"+i+str(datetime.datetime.now())+".csv"
    csv_file = open(filename, "w")
    columnTitleRow = "SNR_inyectado,Outliers,SNR,Caos\n"
    csv_file.write(columnTitleRow)
    for j in SNR_pasos:
        if j == 4:
            outliers = leerTs(i,original=1)
            snr = SNRTs(i,showPlot=0)
            caos = calcularCaos(i)
            csv_file.write("0,"+str(outliers)+","+str(snr)+","+str(caos)+'\n')
        else:
            outliers = leerTs(i,original=0,SNR_dB=j)
            snr = SNRTs(i,SNR_dB=j,showPlot=0)
            caos = calcularCaos(i,j)
            csv_file.write(str(j)+","+str(outliers)+","+str(snr)+","+str(caos)+'\n')
#         print(i,j,caos)
        print("Tiempo parcial ",time.time()-tiempoInicial)
    csv_file.close()

print("Tiempo total ",time.time()-tiempoInicial)
