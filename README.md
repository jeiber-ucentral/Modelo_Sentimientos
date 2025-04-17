# Análisis de Sentimiento en Tweets con RNN, LSTM y BiLSTM + Atención

## Descripción

Este proyecto desarrolla un sistema de clasificación de sentimientos en tweets mediante redes neuronales recurrentes (RNN), redes LSTM, y redes BiLSTM con mecanismos de atención. El flujo completo incluye:

- **Cargue y preprocesamiento de datos**
- **Tokenización y vectorización de texto**
- **Construcción de arquitecturas de modelos**
- **Entrenamiento y evaluación de modelos**
- **Predicción de nuevos mensajes**

---

## Estructura de Archivos

| Archivo | Descripción |
|:--------|:------------|
| [data_loader.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/data_loader.py) | Carga y depura la base de datos de tweets. |
| [data_tokenizer.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/data_tokenizer.py) | Tokeniza y vectoriza los textos usando TF o TF-IDF. |
| [model_rnn.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/model_rnn.py) | Arquitectura de Red Neuronal Recurrente (RNN). |
| [model_lstm.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/model_lstm.py) | Arquitectura de Red LSTM. |
| [model_bilstm_attention.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/model_bilstm_attention.py) | Arquitectura de Red BiLSTM con mecanismo de atención. |
| [train.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/train.py) | Entrena los modelos y guarda los modelos entrenados. |
| [evaluate.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/evaluate.py) | Evalúa los modelos sobre el conjunto de prueba. |
| [predict.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/predict.py) | Clasifica nuevos mensajes usando los modelos entrenados. |

---

## Instalación de Librerías

Antes de ejecutar el proyecto, asegúrate de instalar las siguientes librerías:

```bash
pip install pandas numpy scikit-learn tensorflow nltk matplotlib joblib emoji
```

Además, es necesario descargar algunos recursos de NLTK:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

---

## Pipeline del Proyecto

1. **Cargar y depurar datos**  
   Ejecutar [data_loader.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/data_loader.py) para cargar la base de datos y limpiar los tweets.

2. **Tokenizar y vectorizar**  
   Ejecutar [data_tokenizer.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/data_tokenizer.py) para crear los archivos `.pkl` del tokenizador y las matrices vectorizadas (TF o TF-IDF).

3. **Entrenar modelos**  
   Ejecutar [train.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/train.py) donde puedes elegir entrenar entre tres arquitecturas:
   - `0`: RNN
   - `1`: LSTM (en versión avanzada, puedes agregar su entrenamiento)
   - `2`: BiLSTM con Atención

4. **Evaluar modelos**  
   Ejecutar [evaluate.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/evaluate.py) para evaluar el desempeño del modelo en el conjunto de prueba.

5. **Clasificar nuevos tweets**  
   Ejecutar `predict.py` para predecir el sentimiento de un mensaje nuevo ingresado por el usuario.

---

## Modelos Implementados

- **RNN**: Red simple con embedding y capa de pooling.
- **LSTM**: Red con memoria de largo plazo para relaciones de largo alcance.
- **BiLSTM + Atención**: Red bidireccional con mecanismo de atención para enfocar en las partes más relevantes del texto.

---

## ¿Cómo entrenar un modelo?

Desde [predict.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/predict.py):

```bash
python train.py
```

El programa pedirá:
- Qué modelo deseas entrenar [0: rnn, 1: lstm, 2: bilstm]
- Si deseas usar vectorización TF-IDF
- Si deseas ver detalles de las bases
- Si deseas visualizar el gráfico de entrenamiento

Después, guardará el modelo en la carpeta `/models/`.

---

## ¿Cómo evaluar el modelo?

Desde [evaluate.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/evaluate.py):

```bash
python evaluate.py
```

Deberás indicar:
- Modelo [0: rnn, 1: lstm, 2: bilstm]
- Si usarás TF-IDF

El sistema mostrará métricas como:
- Accuracy
- Precision
- Recall
- F1-Score

---

## ¿Cómo predecir nuevos tweets?

Desde [predict.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/predict.py):

```bash
python predict.py
```

Debes ingresar:
- El modelo que deseas usar
- El tipo de vectorizador (TF o TF-IDF)
- El tweet que deseas clasificar

El sistema te devolverá si el sentimiento es **positivo** o **negativo**.

---

## Notas

- Los modelos entrenados se guardan en `/models/`.
- Actualmente en [train.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/train.py) no se activa el entrenamiento de LSTM ([model_lstm.py](https://github.com/jeiber-ucentral/Modelo_Sentimientos/blob/db45f9a6fe538587b82083d233aa67005d6229a4/src/model_lstm.py)), pero está disponible para ser habilitado si se desea.
- Todos los modelos usan optimizador `Adam` y función de pérdida `binary_crossentropy`.
- Se usa **EarlyStopping** para evitar sobreentrenamiento en los modelos.

---

## Autor

- Proyecto desarrollado para fines académicos y experimentales de Deep Learning aplicado a análisis de sentimientos por los estudiantes: Willian Lopez, Jeiber Díaz, Ronald Salinas, Gretel Ruiz

