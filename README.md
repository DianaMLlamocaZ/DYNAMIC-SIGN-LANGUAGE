# DESCRIPCIÓN
Este proyecto se centra en la clasificación de señas dinámicas en tiempo real, implementando modelos secuenciales y utilizando los landmarks de MediaPipe Holistic como inputs para el modelo.

Importante: Es la continuación del proyecto "Static Sign Language", con el objetivo de ampliar el proyecto a señas dinámicas.

----

# ETAPAS
### ETAPA 1: CREACIÓN DEL PIPELINE AUTOMATIZADO PARA CAPTURA DE DATOS (HECHO)
#### - CAPTURA Y ALMACENAMIENTO DE DATOS EN TIEMPO REAL (LISTO)
  - La grabación de las muestras se inicia 3 segundos después de que el usuario haya ingresado una de sus manos frente a la cámara.
  -  Posteriormente, el usuario procede a realizar la seña.
  - Extracción de features: pose + hands landmarks. Si no se detecta uno de ellos, la grabación de la muestra continúa, rellenando con ceros los vectores, lo cual es importante, ya que no todas las señas involucran   movimiento de las manos en todo momento.
  - Se guarda la muestra creada en el directorio correspondiente para facilitar la carga de datos al diseñar el custom dataset.
---

#### - **NOTA (SOLUCIONADO)**:
Cuando el usuario, en la etapa de captura de datos, realiza una seña más corta que la duración total (30 frames), los vectores resultantes de los landmarks pueden contener ceros al final por la función *extract_keypoints*. Estos ceros no necesariamente significan que la mano no estaba presente en la seña, sino que indican que el gesto ya había finalizado.

Esto puede hacer que el modelo confunda 'ausencia de gesto' con 'fin anticipado del gesto', lo cual es un problema en el entrenamiento. Para solucionarlo:
- Se permite que el usuario finalice manualmente la grabación con la tecla 'a'.
- Se guarda la secuencia de datos tal como es, sin forzar a que tenga 30 frames.
- Se evita rellenar con ceros que no contengan información real.

---

#### - **IMPORTANTE**:
De lo anterior, se deduce que la cantidad de frames no será “30” (el valor máximo de cada seña, por ahora) para todos los samples, sino variable, lo que conlleva a realizar padding a las secuencias por batch antes de entrenar el modelo.

---

====
### ETAPA 2: PREPROCESAMIENTO DE DATOS (HECHO)
#### - NORMALIZACIÓN DE KEYPOINTS PARA LA INVARIANZA A 'TRASLACIÓN' (LISTO)
  - Para que el modelo generalice mejor las señas dinámicas, y no dependa exclusivamente de la 'posición' de los landmarks en el frame, se implementó la invarianza a traslación para que, independientemente de dónde se encuentren ubicados los keypoints de los landmarks, el modelo aprenda la "distribución espacial" respecto al wrist (coordenada [0,0,0]).

---
#### - **NOTA:**
  - Esto permite que el usuario tenga mayor flexibilidad de realizar la seña en cualquier posición del frame. Sin embargo, la invarianza a 'traslación' NO es lo mismo que la invarianza a 'escala' y 'rotación', por lo que si se realiza la misma seña, pero con mayor proximidad/lejanía a la cámara o con grados distintos a los que se entrenó respecto al wrist, el modelo verá una distribución diferente.
---

#### - **IMPORTANTE** (EN PROCESO):
  - Para solucionar lo anterior, estoy evaluando aplicar invarianza a escala de las coordenadas 'x' e 'y', no 'z', ya que, en algunas señas, es necesario que el modelo interprete correctamente el movimiento de los gestos en el eje 'z'.
---

====

### ETAPA 3: CREACIÓN DEL CUSTOM DATASET (HECHO)
#### - CARGA DE DATOS PREVIO AL ENTRENAMIENTO (HECHO)
  - Se crea el custom dataset leyendo cada directorio individualmente, y agrupándolos en listas que luego se convertirán a tensores para el entrenamiento del modelo.
  - Adicionalmente, se creó una función de padding. De esta manera, se asegura que todos los samples tengan el mismo sequence length.
  - Además, aplico el preprocesamiento de datos durante la creación del custom dataset para garantizar la invarianza a traslación de los keypoints.
----

#### - **NOTA (EN PROCESO)**:
- El preprocesamiento de los datos garantiza la invarianza a traslación, pero no la invarianza a escala ni rotación. Por ello, actualmente me encuentro evaluando qué método es el que permite un mejor performance y generalización del modelo en las señas para aplicar el preprocesamiento que mejor se ajuste a los gestos.

----



# ARCHIVOS
- **create_data.py**: Archivo que se encarga de la creación y el almacenamiento de datos en tiempo real, utilizando las funciones definidas en *utils_keypoints.py*.
- **utils_keypoints.py**: Archivo que contiene las funciones para dibujar los landmarks por frame, la extracción de keypoints por muestra y el preprocesamiento de los mismos para garantizar la invarianza a 'traslación'.
- **custom_dataset.py**: Archivo que contiene el código para la creación del custom dataset, leyendo cada directorio individualmente y almacenándolos en listas que se convertirán a tensores para el entrenamiento del modelo, así como el preprocesamiento de los landmarks.
- **utils_dataloader.py**: Archivo que contiene una función de "padding dinámico", para que, en cada batch, se calcule el max length y se rellenen de 0s algunos samples que no necesariamente contengan más de 30 frames.

- **training.py**: Archivo que contiene el código para el entrenamiento del modelo.
- **real_time_pre.py (en actualización)**: Archivo que contiene el código *'referencial'* para la predicción en tiempo real, *creado principalmente para saber qué mejoras realizar*.
  Por ese motivo, actualizaré este código constantemente, ya que quedan muchas mejoras por hacer respecto al suavizado temporal, y evitar que el modelo prediga clases durante la transición de una seña. Por lo tanto, *este código iré modificándolo y actualizándolo* con el objetivo de mejorar la predicción de señas dinámicas en tiempo real.
