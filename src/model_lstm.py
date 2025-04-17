########################################
# # # 04. ARQUITECTURA MODELO LSTM # # # 
########################################
#------------#
# # INDICE # #
#------------#
# 1. Cargar librerias
# 2. Definición de la arquitectura
# 3. 
# 4. 

#==================================================================
### 1. Cargue Librerias ###
#-------------------------#
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

#------------------------------------------#
# # # 2. Definición de la arquitectura # # #
#------------------------------------------#


def constr_modelo_lstm(x_train):
    '''
    Se crea la arquitectura LSTM del modelo para su estimación.
    
    Argumentos:
        * x_train: Base de entrenamiento; para conocer las dimensiones de la capa de entrada.
    
    Retorno:
        * model: Modelo LSTM propuesto con la arquitectura definida
    '''
    ## Definicion de la Arquitectura
    model = Sequential()
    model.add(Embedding(input_dim=x_train.shape[1], output_dim=128))
    model.add(LSTM(128))  # Cambio principal: de GRU a LSTM
    model.add(Dense(1, activation='sigmoid'))

    # Compilación del modelo
    model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
    
    return model

