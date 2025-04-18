######################################################
# # # 05. ARQUITECTURA MODELO BILSTM - ATTENTION # # # 
######################################################
#------------#
# # INDICE # #
#------------#
# 1. Cargue de librerias
# 2. Definicion de la arquitectura del modelo
# 3. 
# 4. 

#==================================================================

##### 1. Cargue de librerias #####
# Cargue de librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Layer

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, Bidirectional, LSTM
from tensorflow.keras.callbacks import EarlyStopping


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.utils import class_weight

#================================================

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')



##### 2. Definicion de la arquitectura del modelo #####
class Attention(Layer):
    def __init__(self, units, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.units = units  # Asigna units como atributo de la instancia
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features): # Only one input is needed for the Attention layer
        # hidden_with_time_axis = tf.expand_dims(hidden, 1) # Removed, as hidden state is not needed

        # Calculate attention weights based only on features
        score = tf.nn.tanh(self.W1(features))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector # Return only the context vector

    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({"units": self.units})
        return config


def constr_modelo(x_train):
    '''
    Se crea la arquitectura del modelo para su estimacion.
    Argumentos:
        * x_train: Base de entrenamieto; para conocer las dimensiones de la base en la capa de entrada
    Retorno:
        * model: Modelo propuesto con la arquitectura definida
    '''
    # # Definicion de la Arquitectura - Modelo BiLSTM con atención
    model = Sequential()
    model.add(Embedding(input_dim=x_train.shape[1], output_dim=128, input_length=130))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))  # return_sequences=True para la atención
    model.add(Attention(64))  # Capa de atención
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Compilar el modelo
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy']
                  )

    # model.compile(optimizer='adam',  # RMSprop, adam
    #             loss='mean_squared_error',  
    #             metrics=['mape']
    #             )
    
    return model