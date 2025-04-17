#################################################
# # # 07. USO DEL MODELO PARA CLASIFICACION # # # 
#################################################
#------------#
# # INDICE # #
#------------#
# 1. Cargue de librerias
# 2. Cargue del modelo
# 3. Funcion de prediccion
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

import data_loader

#--------------------------------#
# # # # 2. Cargue del modelo # # #
#--------------------------------#
def cargar_modelo(modelo, tfidf):
    '''
    Carga el modelo guardado y los datos de test desde la carpeta models.
    Argumentos: 
        * tfidf: True para usar TF-IDF, False para usar CountVectorizer
        * mensajes: Por default es True. Indica si se desea imprimir un diagnostico de cantidad de filas y primeros registros de las bases finales

    Retorno:
        - model: Modelo cargado.
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

    # Cargando el modelo
    model = tf.keras.models.load_model(modelo_path)
    print("Modelo cargado correctamente 👌")

    # Cargando vectorizador
    if tfidf:
        vectorizador = joblib.load('models/vectorizador_tfidf.pkl')
    else:
        vectorizador = joblib.load('models/vectorizador_tf.pkl')

    print("Vectorizador cargado correctamente 👌")

    return model, vectorizador

#----------------------------------#
# # # 3. Funcion de prediccion # # #
#----------------------------------#
def prediccion_sentimiento(model, f_vector, tweet_nvo):
    '''
    
    '''

    

    return 




