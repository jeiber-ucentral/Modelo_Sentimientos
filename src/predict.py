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

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

nltk.download('wordnet')
nltk.download('punkt')

# Descargar stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
  
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score

import tensorflow as tf

import data_loader
from evaluate import procesar_texto
from data_loader import preprocesar_texto

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
    Recibe un tweet nuevo, lo preprocesa, lo vectoriza y lo clasifica utilizando el modelo cargado.

    Argumentos:
        * model: modelo ya cargado
        * f_vector: vectorizador ya cargado (TF o TFIDF)
        * tweet_nvo: mensaje nuevo (string) a clasificar

    Retorno:
        * prediccion: etiqueta 0 o 1 (sentimiento negativo o positivo)
    '''
    # Preprocesar el tweet
    tweet_proc = data_loader.preprocesar_texto(tweet_nvo, stop=True, lematizar=False, stemizar=False)
    
    # Vectorizar el tweet
    tweet_vect = f_vector.transform([tweet_proc])

    # Predecir
    pred = model.predict(tweet_vect.toarray())

    # Convertir la probabilidad a clase (0 o 1)
    pred = (pred > 0.5).astype(int)

    return pred[0][0]  # Retornar solo el valor (0 o 1)


#--------------------------------#
# # # 4. Funcion consolidada # # #
#--------------------------------#
def main():
    '''
    Carga el modelo y vectorizador, solicita un mensaje nuevo y muestra la predicción
    '''
    modelo = int(input("¿Qué modelo desea usar? [0: rnn, 1: lstm, 2: bilstm]: "))
    tfidf = input("¿Desea usar TF-IDF? (True/False): ").strip().lower() in ["true", "1", "yes"]
    
    model, vectorizador = cargar_modelo(modelo, tfidf)
    
    tweet_nuevo = input("Ingrese el tweet a clasificar: ")
    
    resultado = prediccion_sentimiento(model, vectorizador, tweet_nuevo)
    
    if resultado == 1:
        print("✨ El mensaje tiene **sentimiento positivo**.")
    else:
        print("💀 El mensaje tiene **sentimiento negativo**.")

if __name__ == "__main__":
    main()



