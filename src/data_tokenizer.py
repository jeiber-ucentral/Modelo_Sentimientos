#####################################
# # # 02. CREANDO TOKENIZADORES # # # 
#####################################
#------------#
# # INDICE # #
#------------#
# 1. Cargue de librerias y datos
# 2. Creando tokenizador
# 3. Calculando y guardando tokenizadores y bases vectorizadas
# 4. Funcion consolidada

#==================================================================

#----------------------------------------#
# # # 1. Cargue de librerias y datos # # #
#----------------------------------------#
# Librerias
import data_loader

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

import joblib
import os

import warnings
warnings.filterwarnings("ignore")

#--------------------------------#
# # # 2. Creando tokenizador # # #
#--------------------------------#
# Procesador de texto
def procesar_texto(text):
    '''
    Realiza un pre procesamient de texto y la tokenizacion del texto suministrado

    Argumentos:
        * text: texto a tokenizar
    Salida:
        * Texto tokenizado
    '''
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]

# Tokenizador
def tokenizador(df, columna_texto, tfidf_vec=False, max_features=10000):
    '''
    Tokeniza y vectoriza una columna de texto de un DataFrame.

    Argumentos:
        df: DataFrame con los textos.
        columna_texto: nombre de la columna que contiene los textos.
        tfidf_vec: True para usar TF-IDF, False para usar CountVectorizer.
        max_features: número máximo de características (tokens) a considerar.

    Salida:
        vectorizer: objeto vectorizador ya ajustado (fit).
        X: matriz de características (sparse matrix).
    '''

    # Selección del vectorizador
    if tfidf_vec:
        vectorizer = TfidfVectorizer(tokenizer=procesar_texto, max_features=max_features)
    else:
        vectorizer = CountVectorizer(tokenizer=procesar_texto, max_features=max_features)

    # Ajustar el vectorizador sobre la columna de texto del DataFrame
    X = vectorizer.fit_transform(df[columna_texto])

    return vectorizer, X

#----------------------------------------------------------------------#
# # # 3. Calculando y guardando tokenizadores y bases vectorizadas # # #
#----------------------------------------------------------------------#
def calculando_vectorizaciones(df, columna):
    '''
    Calcula las vectorizaciones (TF y TF-IDF) de una columna de texto de la base y guarda los resultados en archivos .pkl y las etiquetas.
    
    Argumentos:
        df: DataFrame con la columna de texto y etiquetas.
    
    Salida: 
        * Exporta en formato .pkl los vectorizadores, bases de texto vectorizadas y etiquetas para su uso en el entrenamiento y validacion
    '''

    # Calculando vectorizacion tf
    vectorizer_tf, X_cv = tokenizador(df, columna_texto=columna, tfidf_vec=False)

    # Calculando vectorizacion tf-idf
    vectorizer_tfidf, X_tfidf = tokenizador(df, columna_texto=columna, tfidf_vec=True)

    # Guardando etiquetas
    y = df['label']

    # Guardando archivos en .pkl
    joblib.dump(vectorizer_tf, 'models/vectorizador_tf.pkl')
    joblib.dump(X_cv, 'models/tweets_tf.pkl')

    joblib.dump(vectorizer_tfidf, 'models/vectorizador_tfidf.pkl')
    joblib.dump(X_tfidf, 'models/tweets_tfidf.pkl')

    # Guardar las etiquetas
    joblib.dump(y, 'models/labels.pkl')

    print("Archivos guardados en la carpeta 'models/'.")

#------------------------------------------------------#
# # # 4. Exportando tokenizador y base vectorizada # # #
#------------------------------------------------------#
def main():
    # Cargue del df
    df = data_loader.main()  
    
    # Calculando tokenizadores y guardando los resultados
    calculando_vectorizaciones(df, columna = "tweet_clean")

if __name__ == "__main__":
    main()

