########################################
# # # 06. ENTRENAMIENTO DE MODELOS # # # 
########################################
#------------#
# # INDICE # #
#------------#
# 1. Cargue de librerias
# 2. Cargue y segmentacion de datos 
# 3. Funcion para el entrenamiento de la red propuesta
# 4. Exportar el modelo estimado
# 5. Funcion consolidada


#==================================================================

#--------------------------------#
# # # 1. Cargue de librerias # # #
#--------------------------------#
import warnings
warnings.filterwarnings("ignore")

import os
import joblib
from data_tokenizer import procesar_texto
import data_loader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM
import model_rnn
import model_lstm 
import model_bilstm_attention

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.utils import class_weight


#-------------------------------------------#
# # # 2. Cargue y segmentacion de datos # # # 
#-------------------------------------------#
def division_datos(tfidf, mensajes=True):
    '''
    Carga y segmenta los datos en entrenamiento, validacion y prueba dado el tokenizador escogido.
    Guarda el scaler para normalizar datos en la prediccion.
    Argumentos:
        * tfidf: True para usar TF-IDF, False para usar CountVectorizer
        * mensajes: Por default es True. Indica si se desea imprimir un diagnostico de cantidad de filas y primeros registros de las bases finales

    Retorno:
        * x_train, x_test, x_val
        * y_train, y_test, y_val
    '''
    # Cargando datos de acuerdo a tokenizador seleccionado
    if tfidf:
        vectorizador = joblib.load('models/vectorizador_tfidf.pkl')
        x = joblib.load('models/tweets_tfidf.pkl')
    else:
        vectorizador = joblib.load('models/vectorizador_tf.pkl')
        x = joblib.load('models/tweets_tf.pkl')
    
    y = joblib.load('models/labels.pkl')
    
    # Division en train, test y validacion
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.10, random_state=42)

    if mensajes:
        print("Dimensiones de X completa:", x.shape)
        print("Dimensiones de X train:", x_train.shape)
        print("Dimensiones de X test:", x_test.shape)

        print("\nPrimeros registros X test:")
        print(x_test.toarray()[:5])

        print("\nPrimeras 5 etiquetas")
        print(y[:5])
    
    return x_train, x_test, x_val, y_train, y_test, y_val


# Prueba
# x_train, x_test, x_val, y_train, y_test, y_val = division_datos(tfidf = True, mensajes=True)


#--------------------------------------------------------------#
# # # 3. Funcion para el entrenamiento de la red propuesta # # #
#--------------------------------------------------------------#
def entrenamiento(modelo, x_train, x_val, y_train, y_val, grafico=True):
    '''
    Entrena el modelo y muestra el grafico de desempeno (si se desea).
    Argumentos:
        * modelo: Si se desea entrenar la RNN [0], LSTM [1] o la BILSTM [2]
        * x_train : Datos de entrenamiento ; entradas del modelo para el entrenamiento
        * x_val : Datos de validacion del modelo ; entradas del modelo para la validacion
        * y_train  : Datos de entrenamiento ; salida del modelo para el entrenamiento
        * y_val  : Datos de validacion ; salida del modelo para la validacion 
        * grafico : (True or False) si se desea (True) mostrar el grafico de desempeno del modelo en cuanto a funcion de perdida y metrica por epoca
    Retorno:
        * model: Retorna el modelo estimado de acuerdo a la arquitectura dada y los datos de entrenamiento y validacion
        * History: Historia del proceso de estimacion del modelo por epoca (funcion de perdida y metricas)
        * Grafico de desempeno (si se desea)
    '''
    # Separando las bases
    # x_train, x_test, x_val, y_train, y_test, y_val = division_datos(tfidf = tfidf, mensajes=True)

    # Cargar la arquitectura del modelo
    if modelo == 0:
        # Balanceando clases con smooth
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))
        
        # Arquitectura del modelo
        model = model_rnn.constr_modelo(x_train = x_train)

        # Entrenando el modelo
        print("Entrenando el modelo RNN")
        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        history = model.fit(x_train.toarray(),
                            y_train,
                            epochs=10,
                            batch_size=32,
                            validation_data=(x_val.toarray(), y_val),
                            class_weight=class_weights_dict,
                            callbacks=[early_stop])


    if modelo == 1:
        # Balanceando clases con smooth
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))
        
        # Arquitectura del modelo
        print("Entrenando el modelo LSTM")
        model = model_lstm.constr_modelo_lstm(x_train = x_train)

        # Entrenando el modelo
        history = model.fit(x_train.toarray(),
                            y_train,
                            epochs=5,
                            batch_size=32,
                            validation_data=(x_val.toarray(), y_val),
                            class_weight=class_weights_dict
                            )

    if modelo == 2:
        # Balanceando clases con smooth
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights_dict = dict(enumerate(class_weights))

        # Arquitectura del modelo
        model = model_bilstm_attention.constr_modelo(x_train = x_train)

        # Entrenando el modelo
        print("Entrenando el modelo BILSTM CON ATENCION")
        history = model.fit(x_train.toarray(),
                            y_train,
                            epochs=5, # 5,
                            batch_size=64,
                            validation_data=(x_val.toarray(), y_val),
                            class_weight=class_weights_dict)

    if grafico:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        axes[0].plot(history.history['accuracy'])
        axes[0].plot(history.history['val_accuracy'])
        axes[0].set_title('Model Accuracy')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].legend(['Train', 'Validation'], loc='upper left')

        axes[1].plot(history.history['loss'])
        axes[1].plot(history.history['val_loss'])
        axes[1].set_title('Model Loss')
        axes[1].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(['Train', 'Validation'], loc='upper left')
        plt.tight_layout()
        plt.show()
    
    return model, history
    


#----------------------------------------#
# # # 4. Exportar el modelo estimado # # #
#----------------------------------------#
def exportar_modelo(model, modelo, tfidf):
    '''
    Guarda el modelo entrenado.

    Argumentos:
        * model: Modelo estimado dada la arquitectura seleccionada
    Retorno:
        * Guardado del modelo
    '''
    # Guardar el modelo estimado
    if tfidf:
        if modelo == 0:
            modelo_path = "models/rnn_tfidf.h5"
        elif modelo == 1:
            modelo_path = "models/lstm_tfidf.h5"
        elif modelo == 2:
            modelo_path = "models/bilstm_tfidf.h5"
        else:
            print("ERROR: Modelo no definido")
    else:
        if modelo == 0:
            modelo_path = "models/rnn_tf.h5"
        elif modelo == 1:
            modelo_path = "models/lstm_tf.h5"
        elif modelo == 2:
            modelo_path = "models/bilstm_tf.h5"
        else:
            print("ERROR: Modelo no definido")

    model.save(modelo_path)
    print(f"Modelo guardado en {modelo_path} 👌")


#--------------------------------#
# # # 5. Funcion consolidada # # #
#--------------------------------#
def main(modelo, tfidf, msj=True, grafico=True):
    '''
    Ejecuta todo el proceso de carga, entrenamiento y exportacion de archivos.
    Argumentos:
        * msj: Por default es True. Indica si se desea imprimir un diagnostico de cantidad de filas y primeros registros de las bases finales
        * grafico : (True or False) si se desea (True) mostrar el grafico de desempeno del modelo en cuanto a funcion de perdida y metrica por epoca
    Retorno:
        * Modelo, funcion de escalado de datos y datos de testeo exportados
    '''
    # Cargar y segmentar datos
    print("REALIZANDO SEGMENTACION DE DATOS ✂️")
    x_train, x_test, x_val, y_train, y_test, y_val = division_datos(tfidf, mensajes=msj)
    print("")

    # Entrenar el modelo
    print("ENTRENANDO EL MODELO ⏳⌛")
    model, history = entrenamiento(modelo, x_train, x_val, y_train, y_val, grafico=grafico)
    print("MODELO ENTRENADO SATISFACTORIAMENTE!! 👌")
    print("")

    # Guardar el modelo, scaler y datos de prueba
    print("EXPORTANDO EL MODELO 🖨️")
    exportar_modelo(model, modelo, tfidf)

    print("PROCESO CULMINADO SATISFACTORIAMENTE!! 😎")

    return model


if __name__ == '__main__':
    modelo = int(input("¿qué modelo entrenará? [0: rnn, 1: lstm, 2: bilstm]: "))
    tfidf = input("Desea usar vectorizacion TF-IDF? (True / False): ").strip().lower() in ["true", "1", "yes"]
    msj = input("Desea ver detalles de las bases resultantes? (True / False): ").strip().lower() in ["true", "1", "yes"]
    grafico = input("Desea ver el grafico de entrenamiento del modelo? (True / False): ").strip().lower() in ["true", "1", "yes"]

    main(modelo, tfidf, msj, grafico)
