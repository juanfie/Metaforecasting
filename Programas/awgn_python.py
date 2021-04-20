# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

def awgnPython(st,SNR_dB):
    #https://www.gaussianwaves.com/gaussianwaves/wp-content/uploads/2015/06/How_to_generate_AWGN_noise.pdf
    #Matlab / Octave tiene una función incorporada llamada -awgn () con la que se puede agregar 
    #un ruido blanco para obtener la señal deseada a ruido (SNR). 
    #El uso principal de esta función es agregar AWGN a una señal limpia (SNR infinita) para obtener una señal 
    #tamano de la serie
    #st:        Serie de tiempo
    #SNR_dB :   Ruido SNR en decibeles
    #Regresa:   Serie de tiempo ruidosa
    L = len(st)
    #SNR a escala lineal
    SNR = np.power(10,(SNR_dB/10))
    #Calcular la energía
    Esym = np.sum(np.power(abs(st),2))/(L);
    #Encontrar la densidad espectral de ruido
    N0=Esym/SNR;
    #Desviación estándar para el Ruido
    noiseSigma =np.sqrt(N0);
    #Creando el ruido
    n = np.random.normal(0,noiseSigma,L)
#     print("st\n\n",st,"\n\n\n n \n\n\n",n)
    return st + n

x = np.linspace(-np.pi, np.pi, 201)
st = np.sin(x)
stRuidosa = awgnPython(st,10)
plt.plot(x, st)
plt.plot(x, stRuidosa)
plt.show()
#print("series de tiempo\n",st[0:5],"\nserie de tiempo ruidosa\n",stRuidosa[0:5])
