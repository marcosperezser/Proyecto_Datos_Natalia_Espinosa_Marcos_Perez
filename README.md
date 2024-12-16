# Proyecto Final

## Tratamiento de Datos

### Máster de Ingeniería de Telecomunicación

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
  <img src="images/AnalisisCategorias.png" alt="Gráfica Categorias" width="45%" style="display:inline-block; margin-right:10px;">
  <img src="images/AnalisisCategorias2.png" alt="Gráfica Categorias" width="45%" style="display:inline-block;">
</div>
