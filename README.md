# Resumen del Cuaderno — Predicción de la Calidad del Agua de la India con Keras

**Pontificia Universidad Javeriana · Procesamiento de Big Data · 2do Examen Parcial · 28/04/2026**  
**Autor:** Juan Sebastian Álvarez

-----

## 1. Objetivo general

Construir un flujo completo de análisis de calidad del agua sobre el dataset `waterquality.csv` (534 registros, ríos de la India) usando **PySpark** para el procesamiento distribuido y **Keras/TensorFlow** para predecir el **Water Quality Index (WQI)** mediante una red neuronal de regresión.

-----

## 2. Dataset

|Variable           |Descripción                  |Unidad    |
|-------------------|-----------------------------|----------|
|TEMP               |Temperatura del agua         |°C        |
|DO                 |Oxígeno disuelto             |mg/L      |
|pH                 |Acidez/basicidad             |—         |
|CONDUCTIVITY       |Conductividad eléctrica      |µS/cm     |
|BOD                |Demanda bioquímica de oxígeno|mg/L      |
|NITRATE_N_NITRITE_N|Nitratos y nitritos          |mg/L      |
|FECAL_COLIFORM     |Coliformes fecales           |MPN/100 mL|

- **534 registros** correspondientes a estaciones de monitoreo en estados de la India.
- Carga con PySpark (`SparkSession`), conversión de columnas a `FloatType`, eliminación de `TOTAL_COLIFORM` por redundancia.
- Se identificaron valores nulos con `F.isnan()` / `F.col().isNull()`; los estados sin datos en la visualización geográfica recibieron imputación con la mediana del WQI.

-----

## 3. Análisis exploratorio (Sección 5)

### Gráfica 5.1 — DO y pH

- El **pH** es muy estable (rango 7–9), típico de aguas neutras a levemente alcalinas con sustrato calcáreo como tampón.
- El **DO** presenta mayor variabilidad (4–9 mg/L), con picos > 10 mg/L (aguas turbulentas bien oxigenadas) y caídas cercanas a cero (zonas hipóxicas críticas).
- No existe correlación visual directa entre ambas variables a escala de registro; la relación biológica DO–pH opera a escala diaria (fotosíntesis intensa).
- **Impacto en WQI:** DO tiene el mayor peso del índice (0.281); caídas bajo 3 mg/L colapsan el subíndice a 0.

### Gráfica 5.2 — BOD y Nitratos/Nitritos

- **BOD:** mayoría de registros < 10 mg/L, con picos extremos > 100 mg/L en estaciones puntuales (descargas industriales o domésticas sin tratamiento).
- **Nitratos/Nitritos:** generalmente < 5 mg/L; picos ocasionales asociados a escorrentía agrícola. No superan el límite OMS de 50 mg/L en casi ningún caso.
- Picos simultáneos de BOD y NN → contaminación agropecuaria. Pico aislado de BOD → origen orgánico industrial/doméstico.

### Gráfica 5.3 — Conductividad y Coliformes Fecales

- **Conductividad:** mayoría < 500 µS/cm, picos hasta 10 000 µS/cm (contaminación inorgánica / intrusión salina).
- **Coliformes fecales:** altísima variabilidad, de 0 a decenas de miles de MPN/100 mL. Picos extremos = zonas urbanas sin saneamiento.
- Los picos de ambas variables **no coinciden espacialmente**, confirmando fuentes de contaminación distintas.
- **Riesgo sanitario:** peso del subíndice `qrFecal` = 0.281; valores > 1 000 MPN/100 mL colapsan el WQI a categoría Inadecuada.

-----

## 4. Construcción del WQI (Secciones 6 y 7)

### Fórmula

```
WQI = wpH + wDO + wCOND + wBOD + wNN + wFecal
```

### Pesos por parámetro

|Parámetro         |Subíndice|Peso     |
|------------------|---------|---------|
|pH                |qrPH     |0.165    |
|DO                |qrDO     |**0.281**|
|Conductividad     |qrCOND   |0.234    |
|BOD               |qrBOD    |0.009    |
|Nitratos/Nitritos |qrNN     |0.028    |
|Coliformes fecales|qrFecal  |**0.281**|

Los subíndices `qr` toman valores discretos: **100 · 80 · 60 · 40 · 0** según rangos de la literatura (Sutadian et al., 2016).

### Clasificación del WQI

|Rango WQI|Categoría |
|---------|----------|
|0 – 25   |Excelente |
|25 – 50  |Buena     |
|50 – 75  |Baja      |
|75 – 100 |Muy Baja  |
|≥ 100    |Inadecuada|

### Distribución observada (Gráfica 7.3)

- Distribución con **sesgo positivo** (cola a la derecha), moda en torno a WQI ≈ 55–65.
- **Casi ninguna estación alcanza la categoría Excelente** (WQI < 25).
- La mayoría cae en **Baja** y **Muy Baja**, evidenciando presión antrópica generalizada.
- Estaciones con WQI > 100 = contaminación multidimensional; intervención urgente.

### Conteo por categoría (Gráfica 7.4)

- **Baja** es la categoría dominante, seguida de **Muy Baja**.
- La proporción Excelente + Buena es mínima, indicando un problema estructural de calidad hídrica nacional.

-----

## 5. Visualización geográfica (Sección 8)

### Mapa base (8.3)

Mapa político de India cargado desde shapefile con GeoPandas. Se realizó homologación de nombres de estados entre el shapefile y el DataFrame Spark (reemplazo de `&`, conversión `initcap`, corrección de `TAMILNADU`).

### Mapa coroplético WQI (8.4)

- Paleta `Reds`; cuatro rangos: 0–25, 25–50, 50–75, 75–100+.
- Colores más intensos = peor calidad del agua.
- Estados sin datos reciben WQI mediano como imputación (limitación metodológica señalada).
- Etiquetas ajustadas con `adjust_text` para evitar superposición.
- Permite identificar territorialmente qué cuencas requieren intervención prioritaria.

### Barras horizontales WQI por estado (8.5)

- Complementa el mapa con lectura cuantitativa directa por estado.
- Facilita identificar los estados con peor y mejor calidad relativa.
- La agregación por estado puede ocultar heterogeneidad interna (ríos prístinos vs. industrializados en el mismo estado).

-----

## 6. Modelo de red neuronal con Keras (Sección 9)

### Arquitectura

```
Entrada (6) → Dense(350, ReLU) → Dense(350, ReLU) → Dense(350, ReLU) → Dense(1, linear)
```

- ~490 000 parámetros entrenables.
- Variables de entrada: `qrPH`, `qrDO`, `qrCOND`, `qrBOD`, `qrNN`, `qrFecal`.
- Variable objetivo: `WQI` (regresión continua).

### Compilación y entrenamiento

```python
optim = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
modelo01.compile(loss='mean_squared_error', optimizer=optim, metrics=['mse'])
historia01 = modelo01.fit(X_train, y_train, epochs=200, batch_size=81, verbose=0)
```

- Split 80/20 con `train_test_split(random_state=1)`.
- Conversión de DataFrames Pandas a NumPy (`float32`) antes del entrenamiento.

### Correcciones aplicadas al código original

|Error                                                     |Corrección                                                   |
|----------------------------------------------------------|-------------------------------------------------------------|
|`Sequential`, `Dense` sin importar                        |`from tensorflow.keras.models import Sequential` etc.        |
|`X_train`, `y_train` no definidos                         |Conversión explícita: `dataTrain.values.astype('float32')`   |
|`keras.optimizers.Adam(...)` flotante (objeto no asignado)|`optim = tf.keras.optimizers.Adam(...)` + pasarlo a `compile`|
|`optimizer='Adam'` (mayúscula)                            |Pasado como objeto; convención lowercase                     |
|`amsgrad=False` removido en TF 2.11+                      |Eliminado                                                    |
|Encabezados §9.3 y §9.5 duplicados                        |Eliminados                                                   |

### Curva de pérdida (9.3)

- Reducción rápida del MSE en las primeras épocas; estabilización posterior = convergencia exitosa.
- Adam adapta el learning rate por parámetro → descenso inicial eficiente.
- Sin curva de validación: no se puede detectar overfitting directamente desde esta gráfica.

### Dispersión real vs. predicho (9.5)

- Puntos concentrados en torno a la diagonal y = x → el modelo captura la tendencia del WQI.
- Mayor dispersión en extremos (WQI muy bajo o muy alto) por menor representación en entrenamiento.
- Las métricas de prueba son el indicador definitivo de generalización.
-----

## 7. Conclusiones generales (Sección 11)

### Calidad del agua en India

- La calidad es **sistemáticamente deficiente**: la mayoría de estaciones cae en categorías Baja o Muy Baja.
- Los **coliformes fecales** y el **oxígeno disuelto** son los determinantes estructurales del WQI deteriorado (peso conjunto = 0.562).
- Existe **heterogeneidad espacial** marcada: algunos estados presentan WQI notablemente más altos que otros, asociado a industrialización y densidad urbana.
- Las intervenciones de mayor impacto son: tratamiento de aguas residuales (reduce coliformes) y reoxigenación de ríos (mejora DO).

### Modelo neuronal

- La red neuronal aprende la relación subíndices → WQI con buena capacidad, aunque el problema es intrínsecamente lineal (WQI = suma ponderada).
- El valor académico es la práctica del pipeline PySpark + Keras end-to-end, no la complejidad del modelo.
- Trabajo futuro: comparar con regresión lineal, Random Forest y SVR; implementar k-fold CV; aplicar regularización Dropout + L2.

### Limitaciones metodológicas

- Imputación de WQI por mediana en estados sin datos (para visualización geográfica).
- Subíndices discretos → pérdida de información respecto a variables continuas originales.
- Dataset pequeño (534 registros) para justificar redes con ~490 000 parámetros.

-----

## 8. Referencias

1. Sutadian, A. D., Muttil, N., Yilmaz, A. G., & Perera, B. J. C. (2016). Development of river water quality indices — A review. *Environmental Monitoring and Assessment*, 188(1), 58.
1. Brown, R. M., McClelland, N. I., Deininger, R. A., & Tozer, R. G. (1970). A water quality index: Do we dare? *Water and Sewage Works*, 117(10), 339–343.
1. Chollet, F. (2021). *Deep Learning with Python* (2nd ed.). Manning Publications.
1. Zaharia, M., et al. (2016). Apache Spark: A unified engine for big data processing. *Communications of the ACM*, 59(11), 56–65.
1. OMS / WHO. (2022). *Guidelines for Drinking-Water Quality* (4th ed.). World Health Organization.

-----

*Cuaderno generado con PySpark 3.x + TensorFlow/Keras 2.x · Dataset: waterquality.csv · India · 534 registros · 7 variables fisicoquímicas*
