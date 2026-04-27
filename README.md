# Resumen del informe y del cuaderno: Predicción de Calidad del Agua con Keras

## 1. Propósito general

El proyecto busca predecir la calidad del agua en ríos de la India mediante técnicas de procesamiento de datos y aprendizaje automático. El indicador central es el **Water Quality Index (WQI)**, una medida compuesta que resume diferentes parámetros fisicoquímicos y microbiológicos en un único valor interpretable.

El informe presenta la explicación académica del proyecto, mientras que el cuaderno implementa el flujo práctico en Python: carga de datos, limpieza, análisis exploratorio, cálculo del WQI, visualización geográfica y entrenamiento de una red neuronal con Keras.

## 2. Datos utilizados

El conjunto de datos corresponde a mediciones de estaciones de monitoreo hídrico en India. Las variables principales son:

| Variable | Descripción |
|---|---|
| `TEMP` | Temperatura del agua superficial. |
| `DO` | Oxígeno disuelto; valores altos suelen asociarse con mejor calidad. |
| `pH` | Nivel de acidez o basicidad del agua. |
| `CONDUCTIVITY` | Capacidad de conducción eléctrica; puede indicar presencia de sales o contaminantes. |
| `BOD` | Demanda bioquímica de oxígeno; refleja carga orgánica. |
| `NITRATE_N_NITRITE_N` | Concentración de nitratos y nitritos. |
| `FECAL_COLIFORM` | Indicador de contaminación fecal. |

## 3. Flujo del cuaderno

El cuaderno está organizado en las siguientes etapas:

1. **Importación de bibliotecas:** se cargan PySpark, Pandas, Numpy, Matplotlib, Seaborn, GeoPandas y Keras.
2. **Creación de sesión Spark:** se configura una sesión local para procesar el archivo CSV.
3. **Carga de datos:** se lee `waterquality.csv` y se revisan las primeras filas.
4. **Exploración inicial:** se inspeccionan columnas, estadísticas y valores nulos.
5. **Tratamiento de tipos:** se convierten variables numéricas para permitir cálculos.
6. **Visualización exploratoria:** se grafican relaciones entre DO/pH, BOD/nitratos y conductividad/coliformes fecales.
7. **Cálculo de subíndices:** cada parámetro se transforma en una escala de calidad (`qrPH`, `qrDO`, `qrCOND`, `qrBOD`, `qrNN`, `qrFecal`).
8. **Cálculo del WQI:** se ponderan los subíndices y se obtiene el índice final.
9. **Clasificación del WQI:** se asignan categorías como Excelente, Buena, Baja, Muy Baja o Inadecuada.
10. **Visualización geográfica:** se usa GeoPandas para mapear el WQI promedio por estado de India.
11. **Modelo predictivo:** se entrena una red neuronal densa con Keras para predecir WQI.
12. **Evaluación visual:** se grafica la curva de pérdida y la comparación entre valores reales y predichos.

## 4. Cálculo del WQI

El WQI se calcula como suma ponderada de subíndices de calidad. Los pesos usados en el informe y el cuaderno son:

| Parámetro | Peso |
|---|---:|
| pH | 0.165 |
| DO | 0.281 |
| Conductividad | 0.234 |
| BOD | 0.009 |
| Nitratos/Nitritos | 0.028 |
| Coliformes fecales | 0.281 |

La fórmula general implementada es:

```text
WQI = wpH + wDO + wCOND + wBOD + wNN + wFecal
```

Donde cada componente ponderado se obtiene multiplicando el subíndice de calidad del parámetro por su peso.

## 5. Clasificación del índice

| Rango WQI | Clasificación | Interpretación |
|---|---|---|
| 0 - 25 | Excelente | Agua de mejor calidad dentro de la escala usada. |
| 25 - 50 | Buena | Requiere bajo nivel de tratamiento. |
| 50 - 75 | Baja | Requiere tratamiento convencional. |
| 75 - 100 | Muy Baja | Requiere tratamiento intensivo. |
| > 100 | Inadecuada | No apta sin tratamiento. |

## 6. Análisis exploratorio

El informe y el cuaderno incluyen gráficos que permiten observar patrones importantes:

- **DO y pH:** el pH tiende a mantenerse cerca de la neutralidad, mientras que el oxígeno disuelto presenta mayor variabilidad.
- **BOD y nitratos/nitritos:** los picos pueden sugerir cargas orgánicas o posibles efectos de actividad agrícola.
- **Conductividad y coliformes fecales:** valores extremos pueden relacionarse con mineralización, contaminación industrial o problemas de saneamiento.

## 7. Visualización geográfica

El cuaderno usa archivos shapefile de India para crear:

- Un mapa base de estados.
- Un mapa coroplético del WQI.
- Un histograma horizontal para comparar WQI por estado.

Esta parte permite pasar del análisis puramente numérico a una interpretación territorial de la calidad hídrica.

## 8. Modelo de red neuronal

El modelo implementado en Keras es una red neuronal secuencial densa con la siguiente estructura:

| Capa | Tipo | Neuronas | Activación |
|---|---|---:|---|
| Entrada | Dense | 350 | ReLU |
| Oculta 1 | Dense | 350 | ReLU |
| Oculta 2 | Dense | 350 | ReLU |
| Salida | Dense | 1 | Lineal |

La variable objetivo es `WQI`, y las variables predictoras son los seis subíndices de calidad. El modelo se entrena con:

- Optimizador: Adam.
- Función de pérdida: Mean Squared Error.
- Épocas: 200.
- Tamaño de lote: 81.
- División entrenamiento/prueba: 80% / 20%.

## 9. Principales conclusiones

El proyecto demuestra que es posible construir un flujo académico completo de Big Data y Machine Learning para analizar calidad del agua. El uso de PySpark facilita la preparación de datos, GeoPandas permite interpretar los resultados espacialmente y Keras ofrece una aproximación predictiva al WQI.

Sin embargo, el modelo debe entenderse como una guía metodológica. Para una implementación más robusta se recomienda:

- Calcular métricas de evaluación en prueba: MSE, RMSE, MAE y R².
- Ajustar hiperparámetros de forma sistemática.
- Usar regularización para evitar sobreajuste.
- Revisar la calidad de los datos y la imputación de valores faltantes.
- Ampliar el conjunto de datos con más registros temporales y estaciones.

## 10. Relación entre informe y cuaderno

El informe funciona como explicación formal del proyecto: define contexto, variables, metodología, visualizaciones, arquitectura del modelo, resultados y conclusiones. El cuaderno es la implementación práctica de esa metodología en Python. Ambos documentos se complementan: el informe justifica y explica; el cuaderno ejecuta y demuestra el proceso técnico.
