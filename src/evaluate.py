########################################
# # # 06. ENTRENAMIENTO DE MODELOS # # # 
########################################
#------------#
# # INDICE # #
#------------#
# 1. Cargue de librerias
# 2. Cargue del modelo y los datos de testeo
# 3. Funcion de evaluacion del modelo
# 4. Funcion consolidada


#==================================================================

#--------------------------------#
# # # 1. Cargue de librerias # # #
#--------------------------------#
import warnings
warnings.filterwarnings("ignore")

import os
import joblib 
import numpy as np
import pandas as pd
import data_loader
import matplotlib.pyplot as plt
  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score

import tensorflow as tf

import train


#----------------------------------------------------#
# # # 2. Cargue del modelo y los datos de testeo # # #
#----------------------------------------------------#
def cargar_modelo_y_datos(modelo, tfidf):
    '''
    Carga el modelo guardado y los datos de test desde la carpeta models.
    Argumentos: 
        * tfidf: True para usar TF-IDF, False para usar CountVectorizer
        * mensajes: Por default es True. Indica si se desea imprimir un diagnostico de cantidad de filas y primeros registros de las bases finales

    Retorno:
        - model: Modelo cargado.
        - x_test, y_test: Datos de prueba.
    '''
    # Cargue del modelo
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

    model = tf.keras.models.load_model(modelo_path)
    print("Modelo cargado correctamente 👌")

    # Cargue de datos test
    x_train, x_test, x_val, y_train, y_test, y_val = division_datos(tfidf = tfidf, mensajes=True)
    print(f"Datos de test cargados: {x_test.shape} registros 👌.")

    return model, x_test, y_test


#---------------------------------------------#
# # # 3. Funcion de evaluacion del modelo # # #
#---------------------------------------------#
def evaluar_modelo(model, x_test, y_test):
    '''
    Realiza la evaluacion del modelo en cuanto a metricas de desempeno y error
    Argumentos:

    Retorno: 
        * resultados: metricas de desempeno del modelo en formato df

    '''

    # Uso del modelo para predecir sobre base test
    y_pred = model.predict(x_test.toarray())
    y_pred = (y_pred > 0.5).astype(int)

    # Calcular métricas de evaluación
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(mse)
    recall = recall_score(y_test, y_pred)
    f1_score = f1_score(y_test, y_pred)

    # Guardar resultados en df
    results = pd.DataFrame({
        'ACCURACY': [accuracy],
        'PRECISION': [precision],
        'RECALL': [recall],
        'F1_SCORE': [f1_score]
    })

    return results

#--------------------------------#
# # # 4. Funcion consolidada # # #
#--------------------------------#
def main(modelo, tfidf):
    '''
    Ejecuta el proceso de validacion del modelo sobre la base de testeo
    Argumentos: 

    Retorno:
        * evaluacion: retorna la evaluacion del modelo en diferentes metricas en formato df
    '''
    model, x_test, y_test = cargar_modelo_y_datos(modelo, tfidf)
    
    evaluacion = evaluar_modelo(model, x_test, y_test)
    print(evaluacion)

    return evaluacion


if __name__ == '__main__':
    modelo = int(input("¿qué modelo entrenará? [0: rnn, 1: lstm, 2: bilstm]: "))
    tfidf = input("Desea usar vectorizacion TF-IDF? (True / False): ").strip().lower() in ["true", "1", "yes"]
    
    main(modelo, tfidf)














