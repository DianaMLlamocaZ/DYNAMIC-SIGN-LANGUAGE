# DESCRIPCIÓN
Este proyecto se centra en la clasificación de señas dinámicas en tiempo real implementando modelos secuenciales y utilizando los landmarks de MediaPipe Holistic como inputs para el modelo.

----

# ETAPAS
### ETAPA 1: CREACIÓN DEL PIPELINE AUTOMATIZADO PARA CAPTURA DE DATOS (HECHO)
#### - CAPTURA Y ALMACENAMIENTO DE DATOS EN TIEMPO REAL (LISTO)
  - La grabación de las muestras se inicia 3 segundos después de que el usuario haya ingresado una de sus manos frente a la cámara.
  -  Posteriormente, el usuario procede a realizar la seña.
  - Extracción de features: pose + hands landmarks. Si no se detecta uno de ellos, la grabación de la muestra continúa, rellenando con ceros los vectores, lo cual es importante, ya que no todas las señas involucran   movimiento de las manos en todo momento.
  - Se guarda la muestra creada en el directorio correspondiente para facilitar la carga de datos al diseñar el custom dataset.

**NOTA**: Cuando el usuario, dentro de la recolección de los 30 frames, no realiza ninguna seña, el vector resultante de landmarks sería sparse (muchos ceros) por la función *extract_keypoints*, pero no necesariamente los 0s indican ausencia de dicha mano en la seña, sino que indican, por ejemplo, la ‘finalización’ del gesto antes de los 30 frames. Entonces, los landmarks recolectados, y el valor 0, no reflejan necesariamente una ‘ausencia’ de la mano para la seña, sino que puede deberse a que el gesto se realizó en menos de 30 frames.

Por ello, la solución fue que, dentro de los 30 segundos de recolección de datos, el usuario tiene la opción de finalizar la grabación (presionando la tecla ‘a’) cuando cree conveniente. Así, se almacena la cantidad necesaria de frames, en vez de que hayan ‘ceros’ que puedan confundir al modelo. Un punto a considerar es que, por el momento, la cantidad máxima de frames a capturar es 30; sin embargo, el usuario puede detener la captura de datos antes de este valor.

**IMPORTANTE**: De lo anterior, se deduce que la cantidad de frames no será “30” (el valor máximo de cada seña, por ahora) para todos los samples necesariamente, sino variable, lo que conlleva a realizar padding a las secuencias por batch antes de entrenar el modelo.


====

### ETAPA 2: CREACIÓN DEL CUSTOM DATASET (SIGUIENTE PASO)
#### - CARGA DE DATOS PREVIO AL ENTRENAMIENTO (SIGUIENTE PASO)
  -
  -
----

# ARCHIVOS
- **create_data.py**: Archivo que se encarga de la creación y el almacenamiento de datos en tiempo real, utilizando las funciones definidas en *utils_keypoints.py*.
- **utils_keypoints.py**: Archivo que contiene las funciones para dibujar los landmarks por frame y la extracción de keypoints por muestra.
