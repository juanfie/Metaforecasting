print(__doc__)
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import matplotlib.font_manager

#get_ipython().run_line_magic('matplotlib', 'nbagg')
from bokeh.plotting import figure, gridplot 
from bokeh.models import Legend
from bokeh.io import output_notebook, show

from pandas import read_csv
from pandas import DataFrame
import pandas as pd

import random
#Libreria para graficar 
import ipyvolume as ipv

#Funcion para inyectar perturbaciones a la serie de tiempo, recibe
#ts.-      Serie de Tiempo
#outlier.- Porcentaje de outliers va de 0-1
#ruido.-   Porcentaje de ruido va de 0-1
def generadorPerturbaciones(ts, outlier = 0.0, ruido = 0.0, faltantes = 0.0):
    random.seed(42)
    st = ts.copy()
    cambios = []
    if outlier:
        #Se crea una lista que guardara los indices cambiados, para saber cuales se han cambiado
        #Cuantos ouliers se le van a generar
        totalOutliers = int(outlier * len(st))
        media = np.mean(st)
        sigma = np.std(st)*10
        print("Se generaron",totalOutliers,"Outliers\n","Media de la Serie de Tiempo:",media)
        for i in range(totalOutliers):
            #Se elige de manera aleatoria el dato que se va a cambiar
            cambio = np.random.randint(0,len(st))
            #Cambia los valores por outliers
            for k in range(len(cambios)):
                #Revisa si se cambio el valor en ese indice
                if cambios[k] == cambio:
                    cambio = np.random.randint(0,len(st))
                    k -=1
            alpha = np.random.rand()
            if alpha >= 0.5:
                st[cambio] = media*sigma*alpha
            else:
                st[cambio] = media*(-(sigma+alpha))
            cambios.append(cambio)
    #Perturbaciones para el ruido
    if ruido:
        print("ruido")
    #Perturbaciones para datos faltantes
    if faltantes:
        print("faltantes")
#     print(st)
    return st, cambios


#Crea una base de datos apartir de una ST. de m, recibe
#ts.-   Serie de tiempo
#m.-    m
#name.- Nombre con el que se guardara la Base de Datos
def creaBD(ts,m,name="Base de datos"):
    tamanoTS = len(ts)
    k = m-1
    resultado = pd.DataFrame()
    if tamanoTS > 0 and tamanoTS > m:
        col = "col"
        dataBase = pd.DataFrame()
        for i in range(m):
            if i == 0:
                col = col + str(i)
                resultado = pd.DataFrame({
                    col: np.ravel(ts[i:len(ts)-k])
                })
            else:
                col = col + str(i)
                dataBase = pd.DataFrame({
                    col: np.ravel(ts[i:len(ts)-k])
                })
                resultado = pd.concat([resultado, dataBase], axis=1)
            k -= 1
            col = "col"
        resultado.to_csv(name, encoding='utf-8', index=False)
        print("Se creo correctamente la Base de Datos",name)
    else:
        print("No se puede crear la Base de Datos")


def grafica2D(x1,y1,x2=[],y2=[],titulo="Grafica",title1="Serie 1",title2="Serie 2",tipo=1):    
    p = figure(title=titulo,plot_width=500, plot_height=300)
    output_notebook()
    legend_it = []

    if tipo == 1:
        c = p.line(x1, y1, color="black")
        #p.circle(x1,y1, color="red")
        legend_it.append((title1, [c]))
        if(len(y2)>0):
            c = p.line(x2, y2, color="red")
            legend_it.append((title2, [c]))
    else:
        p.line(x1,y1, color="black")
        if(len(y2)>0):
            p.circle(x2,y2, color="red")
    legend = Legend(items=legend_it, location=(0, 10))
    legend.click_policy="mute"
    p.add_layout(legend, 'above')
    
    show(p, browser=None, new='tab', notebook_handle=False, notebook_url='localhost:8888')


def obtenerOutliers(datos,k=20,contaminacion=0.1,showPlot=0):
    np.random.seed(42)
#     print("incio LocalOutlierFactor")
    clf = LocalOutlierFactor(n_neighbors=k,metric='euclidean'
                             ,algorithm = 'ball_tree',leaf_size = 5, n_jobs=-1)
#     print("fin LocalOutlierFactor \nclf",clf)
    #print("DATA/n",datos['col1'][0:130])
    datos2 = datos
    datos = datos.dropna()
    y_pred = clf.fit_predict(datos)
    #print("\n\ny_pred")
    posiciones = []
    outliers = []
    inliers = []
    dfOut = pd.DataFrame()
    dfIn = pd.DataFrame()
    ###
    mu = np.mean(datos['col1'])
    sigma = np.std(datos['col1'])
            
    lim_sup = mu+(3*sigma)
    lim_inf = mu-(3*sigma)
    print("limite supe",lim_sup,"\nlim_inf",lim_inf)
    indi_up = np.ravel(np.where(datos['col1'] > lim_sup))
    indi_low = np.ravel(np.where(datos['col1'] < lim_inf))
    print("indices",indi_up,"indi_low",indi_low)          
    total_outliers = len(indi_up) + len(indi_low)
    print("total outliers 3sigma",total_outliers)
    ###

    if(np.shape(datos)[1] == 2):
        outliers = [datos2['col0'][np.ravel(np.where(y_pred==-1))],
                    datos2['col1'][np.ravel(np.where(y_pred==-1))]]
        inliers = [datos2['col0'][np.ravel(np.where(y_pred==1))],
                    datos2['col1'][np.ravel(np.where(y_pred==1))]]
        dfOut = pd.DataFrame({'col0':outliers[0],'col1':outliers[1]})
        dfIn = pd.DataFrame({'col0':inliers[0],'col1':inliers[1]})
        print("dfOut\n\n",len(dfOut),"\n",dfOut)
        print("dfIn\n\n",len(dfIn),"\n",dfIn)
        superin = np.concatenate((indi_up, indi_low), axis=0)
        superin = np.sort(superin)
        superin = superin[::-1]
        print("superin",superin)
        #print("indi_up size",len(indi_up),"ind_low size",len(indi_low),"append ind_sup.ind_inf",len(superin))
        for i in superin:
            #print("\nOUTLIERS\ni",outliers[0].shape,"\nindices\n",indi_up)
            if not (i in outliers[0]):
                dfOut = dfOut.append({'col0': i, 'col1':datos2['col1'][i]}, ignore_index=True)
                dfIn = dfIn.drop(dfIn['col0'][i])
                #print("si entro son diferentes")
            #print("dfOut agregado\n\n",dfOut)
        #print("tam despues dfOut",len(dfOut))
        #print("tam despues dfIn",len(dfIn))
        if showPlot:
            grafica2D(dfIn['col0'],dfIn['col1'],dfOut['col0'],dfOut['col1'],
                      titulo = "Inliers vs Outliers", tipo = 2)
#         ipv.quickscatter(inliers[0], inliers[1], color = "blue", size=0.1, marker="sphere")
#         ipv.scatter(outliers[0], outliers[1], color = "red", size=0.5, marker="sphere")
#         ipv.show()
    elif(np.shape(datos)[1] == 3):
        outliers = [datos['col0'][np.ravel(np.where(y_pred==-1))],
                    datos['col1'][np.ravel(np.where(y_pred==-1))],
                    datos['col2'][np.ravel(np.where(y_pred==-1))]]
        inliers = [datos['col0'][np.ravel(np.where(y_pred==1))],
                    datos['col1'][np.ravel(np.where(y_pred==1))],
                    datos['col2'][np.ravel(np.where(y_pred==1))]]
        if showPlot:
            ipv.quickscatter(inliers[0], inliers[1], inliers[2], color = "blue", size=0.1, marker="sphere")
            ipv.scatter(outliers[0], outliers[1], outliers[2], color = "red", size=0.5, marker="sphere")
            ipv.show()

#     print("inicia negative_outlier_factor_")
    Z = -clf.negative_outlier_factor_
#     print("termina negative_outlier_factor_")
#     print("inicia _decision_function")
#     fD = clf._decision_function(datos)
#     print("termina _decision_function")
    if showPlot:
        print("Datos analizados:",len(y_pred),"\nInlliers:",len(dfIn),
              "\nOutliers",len(dfOut))
        print("Media",np.mean(Z),"\nDst",np.std(Z),
              "\nDif",np.abs(Z[np.argmax(Z)]-np.mean(Z)),
              "\nDonde estan",outliers,
              "\nn_neighbors_",clf.n_neighbors_,
              "\nIndice LOF",Z)
    return dfIn,dfOut,posiciones
