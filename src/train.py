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
# import model_rnn
# import model_lstm 
# import model_bilstm_attention

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score

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
# def entrenamiento(modelo, x_train, x_val, y_train, y_val, grafico=True):
#     '''

#     '''

#     # Cargar la arquitectura del modelo
#     if modelo == 0:
#         model = model_rnn.constr_modelo(x_train = x_train)

#     if modelo == 1:
#         model = model_lstm.constr_modelo(x_train = x_train)

#     if modelo == 2:
#         model = model_bilstm_attention.constr_modelo(x_train = x_train)
    
#     # Entrenando el modelo
#     history = model.fit(x_train, 
#                         y_train,
#                         epochs=5,
#                         batch_size=16,
#                         verbose=0,
#                         validation_data=(x_val, y_val)
#                         )

#     if grafico:
#         fig, axes = plt.subplots(1, 2, figsize=(15, 5))

#         # Grafico de MAPE
#         axes[0].plot(history.history['mape'])
#         axes[0].plot(history.history['val_mape'])
#         axes[0].set_title('Model MAPE')
#         axes[0].set_ylabel('MAPE')
#         axes[0].set_xlabel('Epoch')
#         axes[0].legend(['Train', 'Validation'], loc='upper left')

#         # Grafico de perdida (Loss)
#         axes[1].plot(history.history['loss'])
#         axes[1].plot(history.history['val_loss'])
#         axes[1].set_title('Model Loss')
#         axes[1].set_ylabel('Loss')
#         axes[1].set_xlabel('Epoch')
#         axes[1].legend(['Train', 'Validation'], loc='upper left')

#         plt.tight_layout()
#         plt.show()
    
#     return model, history
    


#----------------------------------------#
# # # 4. Exportar el modelo estimado # # #
#----------------------------------------#
# def exportar_modelo(model, modelo, tfidf):
#     '''
#     Guarda el modelo entrenado.
#
#     Argumentos:
#         * model: Modelo estimado dada la arquitectura seleccionada
#     Retorno:
#         * Guardado del modelo
#     '''
#     # Guardar el modelo estimado
#     if tfidf:
#         if model == 0:
#             modelo_path = "models/rnn_tfidf.h5"
#         elif model == 1:
#             modelo_path = "models/lstm_tfidf.h5"
#         elif model == 1:
#             modelo_path = "models/bilstm_tfidf.h5"
#         else:
#             print("ERROR: Modelo no definido")
#     else:
#         if model == 0:
#             modelo_path = "models/rnn_tf.h5"
#         elif model == 1:
#             modelo_path = "models/lstm_tf.h5"
#         elif model == 1:
#             modelo_path = "models/bilstm_tf.h5"
#         else:
#             print("ERROR: Modelo no definido")
# 
#    model.save(modelo_path)
#    print(f"Modelo guardado en {modelo_path} 👌")






# 5. Funcion consolidada

# def main(msj=True, grafico=True):
#     '''
#     Ejecuta todo el proceso de carga, entrenamiento y exportacion de archivos.
#     Argumentos:
#         * msj: Por default es True. Indica si se desea imprimir un diagnostico de cantidad de filas y primeros registros de las bases finales
#         * grafico : (True or False) si se desea (True) mostrar el grafico de desempeno del modelo en cuanto a funcion de perdida y metrica por epoca
#     Retorno:
#         * Modelo, funcion de escalado de datos y datos de testeo exportados
#     '''
#     # Cargar y segmentar datos
#     x_train, x_test, x_val, y_train, y_test, y_val, scaler = division_datos(ruta, mensajes=msj)

#     # Entrenar el modelo
#     model, history = entrenamiento(modelo, x_train, x_val, y_train, y_val, grafico=grafico)
#     print("MODELO ENTRENADO SATISFACTORIAMENTE!! 👌")

#     # Guardar el modelo, scaler y datos de prueba
#     exportar_modelo(model, scaler, x_test, y_test)

#     return model, scaler





# if __name__ == '__main__':
#     modelo = input(¿que modelo entrenara? [0: rnn, 1: lstm, 2:bilstm])
#     tfidf = input("Desea usar vectorizacion TF-IDF? (True / False): ").strip().lower() in ["true", "1", "yes"]
#     msj = input("Desea ver detalles de las bases resultantes? (True / False): ").strip().lower() in ["true", "1", "yes"]
#     grafico = input("Desea ver el grafico de entrenamiento del modelo? (True / False): ").strip().lower() in ["true", "1", "yes"]

#     main(tfidf, msj, grafico)
