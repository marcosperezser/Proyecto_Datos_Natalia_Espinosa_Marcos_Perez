# Proyecto Final Tratamiento de Datos

## Máster de Ingeniería de Telecomunicación

### Marcos Pérez Serrano 100451409 
### Natalia Espinosa Muñoz 100451577
---

### 1. Análisis de Variables de Entrada
El conjunto de datos analizado contiene información detallada de 20.130 recetas de cocina, estructurado en variables textuales y numéricas. El objetivo principal del trabajo es abordar un problema de regresión para predecir la variable `rating`, que representa la valoración otorgada por los usuarios.

#### 1.1 Análisis de las Variables Numéricas

#### Matriz de Correlación

Para analizar la relación entre estas variables numéricas y la variable objetivo `rating`, se ha realizado una matriz de correlación. Los resultados obtenidos indican que existe una fuerte correlación entre las variables nutricionales: por ejemplo, `fat` y `sodium` presentan un valor de correlación cercano a 0.99, lo que sugiere una dependencia lineal muy alta entre estas características. Del mismo modo, `protein` y `calories` también muestran correlaciones moderadas entre sí, con valores alrededor de 0.7.

#### Relación con la Variable Objetivo

Sin embargo, al analizar la relación de estas variables con `rating`, se observa que la correlación es extremadamente baja, con valores cercanos a 0.007. Esto sugiere que las características nutricionales no tienen una influencia significativa en la predicción de la valoración de las recetas.

<div align="center">
  <img src="images/Imagen1.png" alt="Matriz correlación" width="400px">
</div>


#### 1.2 Análisis variables textuales

El siguiente paso es analizar si las variables textuales ofrecen información relevante para la predicción de la variable rating. Para este análisis se deben implementar técnicas de procesamiento del lenguaje natural (NLP), como la vectorización mediante TF-IDF y Word2Vec, así como embeddings contextuales basados en modelos de tipo Transformer como BERT. La combinación de estas representaciones nos permitirá analizar cómo el contenido textual influye en la valoración de las recetas.

En cuanto a la variable categories, esta variable de entrada se basa en etiquetas discretas y predefinidas, por lo que se ha optado por utilizar la técnica MultiLabelBinarizer en lugar de métodos como TF-IDF o Word2Vec, los cuales están diseñados para procesar texto libre y extraer representaciones numéricas basadas en frecuencias o contexto.

En este análisis se ha visualizado la relación entre la variable de salida rating y algunas de las categorías presentes en la variable categories. Los resultados muestran que las categorías tienen un impacto significativo en la valoración de las recetas. El gráfico de las 20 categorías con mayor rating promedio revela que unas pocas categorías, como "Bon Appétit", "Peanut Free" y "Soy Free", obtienen valoraciones considerablemente más altas, con ratings promedio superiores a 1.5. Además, el alto número de recetas en estas categorías (por ejemplo, 9355 recetas en "Bon Appétit") sugiere que son representativas y relevantes en el conjunto de datos. Al mismo tiempo, se observan categorías con ratings más bajos presentan valores promedio inferiores a 0.6. Estas diferencias en los ratings promedio indican que las categorías pueden capturar patrones importantes en los datos. Aun teniendo alta variabilidad, la influencia general de las categorías en las valoraciones indica que actúan como indicadores relevantes para predecir la variable rating.

<div align="center">
  <img src="images/AnalisisCategorias.png" alt="Gráfica Categorias" width="55%" style="display:inline-block; margin-right:10px;">
  <img src="images/AnalisisCategorias2.png" alt="Gráfica Categorias" width="35%" style="display:inline-block;">
</div>

---

### 2. Implementación de un pipeline para el preprocesado de los textos. 
Se ha implementado un pipeline de preprocesado de textos con el objetivo de preparar los datos textuales para su posterior análisis y representación vectorial. Se han realizado en una serie de transformaciones aplicadas al texto, utilizando la librería NLTK. 

Se ha creado una función de preprocesado llamada preprocess_text, donde se ha realizado lo siguiente:

Tokenización:
-	Se utiliza la función word_tokenize de NLTK para dividir el texto en tokens individuales (palabras). Este paso permite trabajar con cada palabra de forma aislada.

Normalización del texto:
-	Se convierten todos los tokens a minúsculas para evitar duplicidades causadas por diferencias en mayúsculas y minúsculas.
-	Se eliminan caracteres especiales y se filtran únicamente tokens alfanuméricos mediante una comprobación con isalnum().

Eliminación de números:
-	Al analizar la columna directions, se ha observado que muchas recetas presentan los pasos numerados, utilizando un formato como "1.", "2.", etc.. Dado que esta información no aporta relevancia al proceso de tokenización ni al análisis del contenido del texto, se ha decidido descartar estos tokens durante el preprocesado. Para ello, se aplicó una expresión regular que identifica y elimina los números seguidos de un punto,

Eliminación de Stopwords:
-	Las stopwords (palabras muy frecuentes como "the", "is", "and") se eliminan utilizando una lista predefinida en inglés proporcionada por NLTK. 

Lematización:
-	Finalmente, se aplica la lematización mediante WordNetLemmatizer, que reduce las palabras a su forma base o raíz. Este paso ayuda a agrupar palabras con un significado similar y reducir la dimensionalidad del texto.

Además, se comprueba si la entrada es una lista (como puede ser el caso de textos tokenizados previos) y, en ese caso, se unen los elementos en una única cadena de texto. La salida de esta función es una lista de tokens limpios y lematizados, que pueden ser utilizados posteriormente en técnicas de vectorización como TF-IDF, Word2Vec, o embeddings contextuales.

---

### 3. Representación vectorial de los documentos mediante tres procedimientos diferentes:

TF-IDF
#### 3.1 TF-IDF

Una vez preprocesados los textos, se utiliza el modelo Bag of Words (BoW) para construir un diccionario de tokens. Este diccionario asigna a cada término único del corpus un identificador numérico, lo que permite representar los textos como combinaciones de estos identificadores y sus frecuencias de aparición en los documentos. 

Como no todos los términos son igualmente útiles para el análisis se aplican filtros al diccionario: se eliminan los términos que aparecen en muy pocos documentos (menos de un umbral definido, como 10), ya que probablemente no aportan información generalizable, y aquellos que aparecen en una proporción demasiado alta del corpus (más del 75% de los documentos), ya que suelen ser palabras genéricas sin valor distintivo.

Después de este filtrado, se aplica el modelo TF-IDF) sobre la representación BoW. TF-IDF pondera cada término según su frecuencia en un documento específico y su frecuencia inversa en el corpus completo. 

Los términos comunes en todo el corpus, como palabras de uso cotidiano, recibiran menos peso, mientras que los términos que son más específicos para un documento individual se consideren más importantes. El resultado es una representación numérica de los textos en forma de vectores ponderados, que reflejan la relevancia relativa de cada término en el contexto del corpus.

Se ha representado en un gráfico la distribución de tokens para la columna desc: 

<div align="center">
  <img src="images/ImagenTokenDir.jpg" alt="Distribución Token Directions" width="400px">
</div>

La distribución de tokens para la columna directions: 
<div align="center">
  <img src="images/ImagenTokenDesc.jpg" alt="Distribución Token Desc" width="400px">
</div>


#### 3.2 WORD2VEC

El proceso de vectorización con Word2Vec consiste en transformar los textos preprocesados en representaciones numéricas multidimensionales que capturan las relaciones semánticas entre palabras. Para ello, se entrena un modelo Word2Vec con los textos de descripciones, direcciones y categorías utilizando parámetros como un tamaño de vector de 100 dimensiones (vector_size=100), una ventana de contexto de 5 palabras (window=5) y un mínimo de 5 apariciones por palabra (min_count=5). 

Una vez entrenado, el modelo asigna a cada palabra un vector numérico que refleja su relación con otras palabras en función de su contexto. Para representar documentos completos, se calcula el vector promedio de las palabras que aparecen en el texto, ignorando aquellas que no están en el vocabulario del modelo. Si no hay palabras válidas en el texto, se asigna un vector de ceros.


Posteriormente, se extraen los vectores de las palabras del vocabulario y se aplica una reducción de dimensiones mediante PCA (Análisis de Componentes Principales), que permite reducir los vectores de 100 dimensiones a 2 componentes principales para visualización. Esta transformación facilita representar gráficamente las palabras en un plano 2D, donde se observan agrupaciones que reflejan las relaciones semánticas aprendidas por Word2Vec


<div align="center">
  <img src="images/Word2VecDesc.png" alt="Distribución Word2Vec Desc" width="400px">
</div>

<div align="center">
  <img src="images/Word2VecDir.png" alt="Distribución Word2Vec Dir" width="400px">
</div>

#### 3.3 BERT

El proceso para la vectorización Bert ha sido la siguiente:
1. Cargamos el modelo preentrenado: se utiliza un modelo preentrenado de BERT que se encuentra disponible en la librería transformers de Hugging Face. 
2. Preparación del texto antes de procesar el texto: se añaden [CLS] al inicio del texto y [SEP] al final, que indica la separación entre frases o el final del texto.
3. Tokenización: el texto se divide en subpalabras mediante el tokenizador de BERT. Este proceso convierte palabras completas en tokens compatibles con el vocabulario del modelo. Si el texto supera la longitud máxima admitida por BERT (512 tokens), se trunca para ajustarse a este límite.
4. Mapeo de tokens y segmentos: cada token se convierte en un índice numérico correspondiente a su posición en el vocabulario del modelo.
Adicionalmente, se crean IDs de segmentos que identifican si los tokens pertenecen a la primera o segunda oración (en tareas donde se analizan pares de frases). En este caso, cada oración se asigna alternativamente a un segmento.
5. Generación de Embeddings: el modelo BERT procesa los tokens y genera representaciones para cada uno a través de múltiples capas internas. Estas representaciones, conocidas como estados ocultos, encapsulan información contextual rica sobre cómo las palabras interactúan entre sí en el texto. Para obtener un único vector que represente el texto completo, se promedian las representaciones de todos los tokens.
6. El resultado final es una lista de embeddings, donde cada vector es una representación numérica del texto procesado.

---

### 4. Entrenamiento y evaluación de modelos de regresión

### 4.1 Entrenamiento y evaluación con RandomForest

Como primer paso en la implementación del modelo, se dividió el conjunto de datos de entrada (X) y la columna de salida ratings (Y) en subconjuntos de entrenamiento y prueba utilizando train_test_split. Se ha asignado el 80% de los datos al conjunto de entrenamiento y el 20% restante al conjunto de prueba, con el objetivo de mejorar la evaluación del modelo al evaluar conjunto de datos no vistos durante el entrenamiento.

Posteriormente, el conjunto de entrenamiento fue subdividido en un conjunto de entrenamiento y un conjunto de validación, asignando el 70% de los datos originales de entrenamiento para entrenamiento y el 30% para validación. Este conjunto de validación se utiliza para evaluar el rendimiento del modelo y seleccionar los mejores hiperparámetros garantizando así que el modelo no se ajuste en exceso al conjunto de entrenamiento.

Por tanto, para este modelo se ha realizado una búsqueda de hiperparámetros utilizando GridSearchCV con el objetivo de optimizar el rendimiento del algoritmo Random Forest y encontrar la combinación de parámetros que minimizara el error en la predicción. Existen varios hiperparámetros que afectan la capacidad de generalización y ajuste, como el número de árboles, la profundidad máxima del árbol, el número mínimo de muestras requeridas para dividir un nodo, y el número mínimo de muestras por hoja y max_features, que controla la cantidad de características consideradas en cada división.

Los mejores hiperparámetros encontrado fueron: n_estimators=500, max_depth=50, min_samples_split=2, min_samples_leaf=1 y max_features='sqrt'. Con estos valores óptimos, se entrenó un modelo final sobre el conjunto completo de entrenamiento (excluyendo la validación) para maximizar su capacidad de generalización. 

Posteriormente, se evaluó este modelo final en el conjunto de prueba.

Para cada una de las vectorizaciones, se han probado diferentes conjuntos de variables de entrada con el objetivo de identificar qué columnas ofrecen mejores resultados en las métricas de evaluación. Los experimentos se realizaron utilizando la vectorización de las siguientes columnas:
- Direcciones
- Descripciones
- Categorías: Estas se han vectorizado utilizando una estrategia de codificación multibinaria como se ha comentado en el apartado 1.
- Variables numéricas: Aunque estas variables mostraron una baja correlación con la variable de salida, se añadieron al modelo para enriquecer el conjunto de datos y aportar contexto adicional.
- 
### 4.2.1 Vectorización TF-IDF

| Variables de Entrada                                       |   MSE |   MAE |   R2  |
|------------------------------------------------------------|-------|-------|-------|
| Sin categorías, con direcciones, con descripciones         | 1.330 | 0.770 | 0.165 |
| Con categorías, con direcciones, con descripciones         | 1.297 | 0.755 | 0.185 |
| Con categorías, con direcciones                            | 1.282 | 0.750 | 0.195 |
| Sin categorías, con direcciones                            | 1.318 | 0.772 | 0.172 |

Se ha decidido utilizar como variables de entrada las categorías, las variables numéricas y únicamente las direcciones. Estos resultados nos indican que el modelo no ha logrado predecir correctamente los valores de rating. El bajo valor de R² (19.5%) indica que el modelo apenas explica una pequeña fracción de la variabilidad de la variable objetivo. Además, los errores significativos en las predicciones, representados por los valores altos de MSE y MAE, evidencian que el modelo tiene dificultades para capturar los patrones clave en los datos.

### 4.2.2 Vectorización WORD2VEC


| Variables de Entrada                            |   MSE |   MAE |   R2  |
|-------------------------------------------------|-------|-------|-------|
| Con categorías, descripciones y direcciones     | 1.294 | 0.759 | 0.187 |
| Sin categorías, descripciones y direcciones     | 1.315 | 0.773 | 0.173 |
| Con categorías y descripciones                  | 1.265 | 0.749 | 0.205 |
| Con categorías y direcciones                    | 1.339 | 0.771 | 0.159 |

Para esta vectorización, el rendimiento del modelo es ligeramente mejor al utilizar la combinación de variables numéricas, categorías y descripciones.

### 4.2.3 Entrenamiento y evaluación con vectorización BERT

Una vez aplicada la vectorización utilizando BERT, se analizaron los resultados del modelo con las representaciones vectorizadas de las descripciones y las direcciones.


### 4.2 Red Neuronal

Se ha implementado una red neuronal NN densa. Esta red está compuesta por varias capas totalmente conectadas, donde cada nodo de una capa está conectado con todos los nodos de la capa siguiente.

El modelo fue diseñado con las siguientes características:
- Capas ocultas: Dos capas ocultas con 50 y 25 neuronas, respectivamente.
- Funciones de activación: Se utilizaron funciones de activación ReLU (Rectified Linear Unit) en las capas ocultas. Esto introduce no linealidad en la red, permitiendo que el modelo aprenda relaciones más complejas en los datos.
- Capa de salida: Una única neurona en la capa de salida, diseñada para predecir un valor continuo (el objetivo de la tarea de regresión).

Para el entrenamiento de la red neuronal se ha utilizado el Error Cuadrático Medio MSE como función de pérdide donde se mide el promedio de los errores al cuadrado entre las predicciones del modelo y los valores reales, penalizando los errores grandes. El objetivo durante el entrenamiento fue minimizar esta función.

Se ha elegido el optimizador Adam, una de las técnicas de optimización más populares debido a su capacidad para adaptarse dinámicamente al tamaño del paso (learning rate) para cada parámetro.

Los hiperparámetros de tasa de aprendizaje y numero de epocas se han ido modificando con el objetivo de minimizar el MSE y por lo tanto mejorar el rendimiento del modelo. Además se han utilizado Mini-batch Training que consiste en dividir los datos en fracciones más pequeñas para actualizar los pesos del modelo de forma iterativa. 


### 4.2.1 Entrenamiento y evaluación con vectorización TF-IDF



### 4.2.2 Entrenamiento y evaluación con vectorización WORD2VEC
| Métrica   |   Valor  |
|-----------|----------|
| MSE       | 1.58178  |
| R² Score  | 0.0152   |


### 4.2.3 Entrenamiento y evaluación con vectorización BERT


### 5. Extensión 

### Traducciones

En esta extensión, se ha llevado a cabo la traducción al francés de todas las recetas cuya descripción (desc) o instrucciones (directions) contenían la palabra "French". 
Para ello se ha crado una nueva base de datos filtrando únicamente aquellas recetas en las que la palabra "French" estuviera presente en las columnas desc o directions del conjunto de datos original. Esto nos permitió enfocar el proceso de traducción en un subconjunto específico de recetas relevantes.

Para la traducción, utilizamos el modelo preentrenado T5-base, que es ampliamente reconocido por su capacidad para realizar tareas de generación de texto, incluyendo traducción automática. Implementamos la traducción mediante el uso del pipeline text2text-generation proporcionado por la librería transformers.

<div align="center">
  <img src="images/Traduciones.png" alt="Extension: Traducciones" width="900px">
</div>





