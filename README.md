# DESCRIPCIÓN
Este proyecto se centra en la clasificación de señas dinámicas en tiempo real utilizando modelos secuenciales.

# ETAPAS
## ETAPA 1: CREACIÓN DEL PIPELINE AUTOMATIZADO PARA CAPTURA DE DATOS
## CAPTURA Y ALMACENAMIENTO DE DATOS EN TIEMPO REAL (LISTO)
  - La grabación de las muestras se inicia 3 segundos después de que el usuario haya ingresado una de sus manos frente a la cámara.
  -  Posteriormente, el usuario procede a realizar la seña.
  - Extracción de features: pose + hands landmarks. Si no se detecta uno de ellos, la grabación de la muestra continúa, rellenando con ceros los vectores, lo cual es importante, ya que no todas las señas involucran   movimiento de las manos.
  - Se guarda la muestra creada en el directorio correspondiente para facilitar la carga de datos al crear el custom dataset.

