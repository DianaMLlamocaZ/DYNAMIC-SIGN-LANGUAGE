# DESCRIPCIÓN
Este proyecto se centra en la clasificación de señas dinámicas en tiempo real implementando modelos secuenciales.

----

# ETAPAS
### ETAPA 1: CREACIÓN DEL PIPELINE AUTOMATIZADO PARA CAPTURA DE DATOS
#### - CAPTURA Y ALMACENAMIENTO DE DATOS EN TIEMPO REAL (LISTO)
  - La grabación de las muestras se inicia 3 segundos después de que el usuario haya ingresado una de sus manos frente a la cámara.
  -  Posteriormente, el usuario procede a realizar la seña.
  - Extracción de features: pose + hands landmarks. Si no se detecta uno de ellos, la grabación de la muestra continúa, rellenando con ceros los vectores, lo cual es importante, ya que no todas las señas involucran   movimiento de las manos.
  - Se guarda la muestra creada en el directorio correspondiente para facilitar la carga de datos al diseñar el custom dataset.

====

### ETAPA 2: CREACIÓN DEL CUSTOM DATASET
#### - CARGA DE DATOS PREVIO AL ENTRENAMIENTO (SIGUIENTE PASO)
  -
  -
----

# ARCHIVOS
- **create_data.py**: Archivo que se encarga de la creación y el almacenamiento de datos en tiempo real, utilizando las funciones definidas en *utils_keypoints.py*.
- **utils_keypoints.py**: Archivo que contiene las funciones para dibujar los landmarks por frame y la extracción de keypoints por muestra.
