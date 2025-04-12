############################################
# # # 01. DATA LOADER AND DATA CLEANER # # # 
############################################
#------------#
# # INDICE # #
#------------#
# 1. Cargue de librerias
# 2. Funcion de cargue de la base de datos
# 3. Funcion de depuracion de los datos
# 4. Funcion consolidada

#==================================================================

#--------------------------------#
# # # 1. Cargue de librerias # # #
#--------------------------------#
import pandas as pd
import re
import string

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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import tensorflow as tf
import emoji

import warnings
warnings.filterwarnings("ignore")

#--------------------------------------------------#
# # # 2. Funcion de cargue de la base de datos # # # 
#--------------------------------------------------#
def cargue_datos():
    '''
    Realiza el cargue de la base de twits dado el enlace suministrado.
    Codigo sin argumentos, lee directamente la base del enlace.
    '''
    url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
    csv_path = tf.keras.utils.get_file("twitter_sentiment.csv", url)

    df = pd.read_csv(csv_path)
    df = df[['tweet', 'label']]

    return df

#-----------------------------------------------#
# # # 3. Funcion de depuracion de los datos # # #
#-----------------------------------------------#
# Funcion de pre-procesamiento de texto
def preprocesar_texto(texto, stop = True, lematizar = False, stemizar = False):
  '''
  Realiza el proceso de depuracion del texto suministrado. 
  Se realiza a conveniencia la omision de stopwords, lematizacion o stemizacion.
  Argumentos:
    * texto : Conversacion que se desea depurar
    * stop :  Indicador de uso u omision de stopwords. Por default es True
    * lematizar : Indicador de uso u omision de lematizacion Por default es False
    * stemizar : Indicador de uso u omision de stemizacion Por default es False

  Salida:
    * texto : Conversacion depurada
  '''
  # Convertir a minúsculas
  texto = texto.lower()

  # Reempazando emojis
  texto = emoji.replace_emoji(texto, replace='')

  # Eliminar caracteres especiales y números
  tabla_traduccion = str.maketrans('áéíóúñÁÉÍÓÚÑ', 'aeiounAEIOUN')
  texto = texto.translate(tabla_traduccion)
  texto = re.sub(r'[^a-z\s]', '', texto)

  # Quitando palabra user
  texto = re.sub(r'user', '', texto)

  # Eliminar stopwords
  if stop:
    palabras = texto.split()
    palabras = [palabra for palabra in palabras if palabra not in stop_words]
    texto = ' '.join(palabras)

  # Lematizar o stemizar (opcional)
  if lematizar:
      # Descarga el modelo de lematización si no está descargado
      lemmatizer = WordNetLemmatizer()
      palabras = texto.split()
      palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras]
      texto = ' '.join(palabras)

  if stemizar:
      # Descarga el modelo de stemización si no está descargado
      stemmer = PorterStemmer()
      palabras = texto.split()
      palabras = [stemmer.stem(palabra) for palabra in palabras]
      texto = ' '.join(palabras)

  return texto


#--------------------------------#
# # # 4. Funcion consolidada # # #
#--------------------------------#
def main():
  '''
  Funcion que realiza todo el proceso de cargue y depuracion de la base de datos
    
  Salida:
    * base de datos en formato para entrenamiento y consumo del modelo
    * Mensajes con cantidad de registros cargados, dimensiones y primeros registros de ejemplo (si se desea)
  
  '''

  # Cargue de base de datos
  df = cargue_datos()

  # Depuracion de la base
  df['tweet_clean'] = df['tweet'].apply(lambda x: preprocesar_texto(x, stop=True, lematizar=False, stemizar=False))

  print("LA BASE DE DATOS HA SIDO CARGADA Y DEPURADA CON EXITO!! 👌👌")
  print("\n")
  print(df.head())
  return df


if __name__ == '__main__':
    main()


