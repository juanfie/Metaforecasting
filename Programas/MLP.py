# coding: utf-8

from sklearn.neural_network import MLPClassifier
#Importamos las librerias que necesitaremos
import pandas as pd
import numpy as np
import math
import time

#Librerias sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
#Libreria matplotlib en linea y bokeh
#get_ipython().run_line_magic('matplotlib', 'inline')
#from bokeh.plotting import gridplot, figure, output_file, output_notebook, show

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping

import datetime
import csv


#calculate Symmetric mean absolute percentage error (smape)
def smape(y_true, y_predict):
    #if
    dimension = y_predict.shape
    if len(dimension) > 1:
        y_predict = np.ravel(y_predict)
    #dimension2 = y_true.shape
    #print(len(dimension2),len(dimension))
    #print("y_true",y_true[0:10],"\ny_predict",y_predict[0:10])
    #print(k)
    return 100/len(y_true) * np.sum(2 * np.abs(y_predict - y_true) / (np.abs(y_true) + np.abs(y_predict)))



def MLPTrain(nombreTs,num_col,val_fa,outliers,snr,size_train=0.7,showplot=0,paciencia=1):
    #Read dataset for training and validation
    file = "/home/victort/Documentos/SEMESTRE 17-18 MAESTRIA/TESIS/Detect Noise/st perturbadas/"+nombreTs+"/"+nombreTs+"_"+str(val_fa)+"_"+str(outliers)+"_"+str(snr)+"_bd.csv"
    cols = np.arange(0,num_col)
    look_back = num_col - 1
    dataframe = pd.read_csv(file,header=None, usecols=cols, engine='python')
    dataset = dataframe.values
#     dataset = dataset.astype('float32')
    
    if(showplot == 1):
        #Plot one dimension time series
        p = figure(title="St. Original training",plot_width=800, plot_height=500)
        output_notebook()
        p.line(np.arange(0,len(dataset)),dataset[:,look_back] , legend="ST Original 1d")
        show(p)
    
    # split into train and validation sets
    train_size = int(len(dataset) * size_train)
    trainX = dataset[0:train_size,0:look_back]            #Toma las n-1 columnas
    trainY = dataset[0:train_size,look_back:num_col]      #Toma la ultima columna
    trainY = np.asarray(np.ravel(trainY))
    
    validation_size = len(dataset) - train_size
    validationX = dataset[train_size:len(dataset),0:look_back]
    validationY = dataset[train_size:len(dataset),look_back:num_col]
    
#     print("trainX\n",trainX,"\ntrainY\n",trainY)
    # reshape input to be [samples, time steps, features]
#     trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
#     print("trainX\n",trainX,"\ntrainY\n",trainY)
#     validationX = np.reshape(validationX, (validationX.shape[0], 1, validationX.shape[1]))
    if(showplot == 1):
        print(dataset)
        print("tamano training set",train_size)
        print("tamano validation set",validation_size)
        print("Train X\n",trainX,"\nTrain Y\n",trainY)
    
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    if nombreTs == "seno":
        model.add(Dense(64, activation='relu', input_dim=num_col-1))
        model.add(Dense(32, activation='sigmoid'))
        model.add(Dense(1, activation='linear'))
    elif nombreTs == "chen_system" or nombreTs == "rossler":
        model.add(Dense(8, activation='relu', input_dim=num_col-1))
        model.add(Dense(4, activation='sigmoid'))
        model.add(Dense(1, activation='linear'))
    elif nombreTs == "duffing_oscillator" or nombreTs == "halvorsen_attractor":
        model.add(Dense(24, activation='relu', input_dim=num_col-1))
        model.add(Dense(4, activation='sigmoid'))
        model.add(Dense(1, activation='linear'))
    elif nombreTs == "henon":
        model.add(Dense(24, activation='relu', input_dim=num_col-1))
        model.add(Dense(12, activation='tanh'))
        model.add(Dense(4, activation='sigmoid'))
        model.add(Dense(1, activation='linear'))
    elif nombreTs == "lorenz":
        model.add(Dense(17, activation='sigmoid', input_dim=num_col-1))
        model.add(Dense(17, activation='sigmoid'))
        model.add(Dense(17, activation='sigmoid'))
        model.add(Dense(1, activation='linear'))
    elif nombreTs == "rucklidge_attractor" or "shawn_van_der_pol_oscillator":
        model.add(Dense(24, activation='relu', input_dim=num_col-1))
        model.add(Dense(8, activation='sigmoid'))
        model.add(Dense(1, activation='linear'))
    elif nombreTs == "simplest_cubic_flow" or "simplest_linear_flow":
        model.add(Dense(24, activation='sigmoid', input_dim=num_col-1))
        model.add(Dense(12, activation='sigmoid'))
        model.add(Dense(1, activation='linear'))
    else:
        print("Modo prueba")
        model.add(Dense(24, activation='sigmoid', input_dim=num_col-1))
        model.add(Dense(12, activation='sigmoid'))
        model.add(Dense(1, activation='linear'))   

    model.compile(optimizer='rmsprop',
              loss='mse')
    stopper = EarlyStopping(monitor='loss', patience=paciencia)
#     checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='max')
#     callbacks_list = [stopper,checkpoint]
    callbacks_list = [stopper]
    
    model.fit(trainX, trainY, epochs=500, callbacks=callbacks_list,verbose=showplot)
    
    model_json = model.to_json()
    filepath2 = "/home/victort/Documentos/SEMESTRE 17-18 MAESTRIA/TESIS/Detect Noise/modelosPrediccion/"+nombreTs+"/"+nombreTs+"_"+str(val_fa)+"_"+str(outliers)+"_"+str(snr)+"_"
    with open(filepath2+"model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filepath2+"model.h5")
    
    
    testY = np.ravel(np.transpose(validationY))
    # make predictions
    trainPredict = model.predict(trainX)
    validationPredict = np.ravel(model.predict(validationX))
    
    mseTrain = mean_squared_error(trainY, trainPredict)
    smapeTrain = smape(trainY,trainPredict)
    mseValidation = mean_squared_error(validationY, validationPredict)
#     print("\ntestY\n",testY)
#     print("\nvalidationPredict\n",validationPredict)
    smapeValidation = smape(testY,validationPredict)
    #print("mseTrain",mseTrain,"smapeTrain",smapeTrain,"mseValidation",mseValidation,"smapeValidation",smapeValidation)
#     print("MSE train", mseTrain)
#     print("MSE validation",mseValidation)
#     print("SMAPE validation",smapeValidation)

    # create and fit the LSTM network
#     filepath='/home/victort/MEGAsync/Redes Neuronales/Proyecto Final/checkpoint'
#     filepath2='/home/victort/MEGAsync/Redes Neuronales/Proyecto Final/'
    
#     model = Sequential()
#     if(dificultad == 1):
# #         filepath=filepath+'_facil.hdf5'
#         filepath2=filepath2+'facil_'
#         model.add(LSTM(10, input_shape=(1, look_back)))
#         model.add(Dense(1,activation='linear',kernel_initializer='zeros'))
#         model.compile(loss='smape', optimizer='adam')
    
#     stopper = EarlyStopping(monitor='loss', patience=paciencia)
# #     checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='max')
# #     callbacks_list = [stopper,checkpoint]
#     callbacks_list = [stopper]
#     model.fit(trainX, trainY, epochs=100, callbacks=callbacks_list, verbose=0)
    
    # serialize model to JSON
    
#     model_json = model.to_json()
#     with open(filepath2+"model.json", "w") as json_file:
#         json_file.write(model_json)
#     # serialize weights to HDF5
#     model.save_weights(filepath2+"model.h5")
    
    
    # make predictions
#     trainPredict = model.predict(trainX)
#     validationPredict = np.ravel(model.predict(validationX))

#     # calculate mean squared error
#     mseTrain = mean_squared_error(trainY, trainPredict)
#     mseValidation = mean_squared_error(validationY, validationPredict)
#     testY = np.ravel(np.transpose(validationY))
#     smapeValidation = smape(testY,validationPredict)

#     print("MSE train", mseTrain)
#     print("MSE validation",mseValidation)
#     print("SMAPE validation",smapeValidation)
    
    if(showplot == 1):
        print("Modelo guardado en disco")
        model.summary()
        # shift train predictions for plotting
        trainPredictPlot = np.zeros(np.shape(dataset)[0])
        trainPredictPlot.fill(np.nan)
        trainPredictPlot[0:len(trainPredict)] = np.ravel(trainPredict)
        # shift test predictions for plotting
        validationPredictPlot = np.zeros(np.shape(dataset)[0])
        validationPredictPlot.fill(np.nan)
        validationPredictPlot[len(trainPredict):len(dataset)] = validationPredict[:len(validationPredict)]
        # plot baseline and predictions
        p = figure(title="Fig",plot_width=800, plot_height=500)
        output_notebook()
        p.line(np.arange(0,len(dataset)),dataset[:,look_back] , legend="ST Original",color = 'black')
        p.line(np.arange(0,len(trainPredictPlot)),trainPredictPlot , legend="Train Predict")
        p.line(np.arange(0,len(validationPredictPlot)),validationPredictPlot , legend="Validation Predict", color = 'red')
        show(p, browser=None, new='tab', notebook_handle=False, notebook_url='localhost:8888')
    
    return mseTrain,smapeTrain,mseValidation,smapeValidation



# file = "/home/victort/Documentos/SEMESTRE 17-18 MAESTRIA/TESIS/Detect Noise/st perturbadas/henon/henon_0_0_5_bd.csv"
def pronostico(nombreTs,showplot=0):
    tiempoGlobal = time.time()
    niveles = [0,5,10,15,20,25]
    filename = "erroresprediccion/"+nombreTs+str(datetime.datetime.now())+".csv"
    csv_file = open(filename, "w")
    columnTitleRow = "Va_Fa,Out,SNR_In,mse entrenamiento,smape entrenamiento,mse pronostico,smape pronostico\n"
    csv_file.write(columnTitleRow)    
    csv_file.close()
    
    filename2 = "/home/victort/Documentos/SEMESTRE 17-18 MAESTRIA/TESIS/Detect Noise/parametrosbd/"+nombreTs+".csv"
    m = pd.read_csv(filename2)
    m = m['m']
    count = 0
    
    for i in niveles:
        for j in niveles:
            for k in niveles:
                tiempoLocal = time.time()
                num_col = m.pop(count)+1
                count +=1
                #print("columnas",m,count)
                if i > 10:
                    print(nombreTs,i,j,k,"columnas",num_col,"posicion",count-1)
                    with open(filename, 'a') as csv_file:
                        writer = csv.writer(csv_file)
                        #num_col = m.pop(count)+1
    #                     print("m2",m2+1)
                        #count +=1
                        
                        mseTrain,smapeTrain,mseVal,smapeVal = MLPTrain(nombreTs,num_col=num_col,val_fa=i,outliers=j,snr=k,size_train=0.7,showplot=showplot,paciencia=5)
                        row = [i,j,k,mseTrain,smapeTrain,mseVal,smapeVal]
                        writer.writerow(row)
                        print("mse",mseVal,"smape",smapeVal)
                    # Ruta donde se va a guardar la ts
                    print("tiempo local",time.time()-tiempoLocal)
                    csv_file.close()
    print("termino con",nombreTs,"en un tiempo de:",time.time()-tiempoGlobal)
    
#SC = ["henon","lorenz"]
#SC = ["chen_system","duffing_oscillator"]
#SC = ["rucklidge_attractor","shawn_van_der_pol_oscillator"]
#SC = ["simplest_cubic_flow","simplest_linear_flow"]
# Aun no acaban de crearse las db
#SC = ["rossler","halvorsen_attractor"]
SC = ["rossler"]
for i in SC:
    pronostico(i,showplot=0)

# Comienza a las 12:30pm corriendo "henon,lorenz" 12:30 am 24 horas
