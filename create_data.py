#Código que permite al usuario crear datos en real time
import mediapipe as mp
import numpy as np
import os
import cv2
import time

#Importo la función para dibujar los keypoints
from utils_keypoints import draw_lms,extract_keypoints


#Modelo holistic de mediapipe
md_holistic=mp.solutions.holistic


#Aquí almaceno los landmarks para formar los 60 frames
landmarks=[]



#Creación de la main function que creará las carpetas y guardará los datos
def main():

    #Variables auxiliares
    capturing=False
    start_wait_time=None
    texto="Iniciando. Muestra tu mano..."
    detener_capturing=False
    frames=30 #30 frames máximo por muestra


    clase=input("Ingrese la clase que quiere guardar/crear datos: ").lower().strip()
    

    #Comprobación de paths y guardado de datos
    path_main_base="./dataset"
    path_clase=f"{path_main_base}/{clase}"


    #Si NO existe esa carpeta, se crea
    if not os.path.exists(path_clase):
        print("Creando carpeta...")
        os.makedirs(path_clase,exist_ok=True)

    
    #Se activa la cámara para crear los datos
    with md_holistic.Holistic(model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic_model:
        print("Presiona 'q' para salir")
        print("Luego de mostrar tu mano, espera 3 segundos para empezar a grabar...")

        cam=cv2.VideoCapture(0)

        while True:

            #Captura de frames y flip para efecto espejo
            open,frame=cam.read()
            frame=cv2.flip(frame,1)
            

            #Conversión a RGB y predicción del modelo
            landmarks_tuple=holistic_model.process(frame[:,:,::-1])
            

            #=========#
            #INICIO DE GRABACIÓN

            #Aquí verifico si el usuario introduce cualquiera de sus manos para a partir de allí empezar a contar hasta 3 para que empiece a grabar
            hand_appear=landmarks_tuple.left_hand_landmarks or landmarks_tuple.right_hand_landmarks


            #Aparece la mano, pero no se está realizando la captura de pantalla hasta que pasen los 3 segundos
            #NOTA: Esto se ejecuta 1 sola vez. Luego de que pasen los 3 segundos y empieza a grabar, deja de ejecutarse esto.
            if hand_appear and not capturing:
                if start_wait_time is None:
                    start_wait_time=time.time()

                else:
                    elapsed=time.time()-start_wait_time

                    if elapsed>=3:
                        texto=f"Grabando!!"
                        
                        capturing=True
                        landmarks=[]

                    else:
                        texto=f"Iniciando en {int(elapsed)+1} seg. NO BAJES TU MANO."
                        


            #Aquí, si capturing=True (lo cual es cierto SIEMPRE luego de que pasen los 3 segundos) capturo los frames
            if capturing:
                
                #Extraigo los keypoints del frame
                landmark_frame=extract_keypoints(landmarks_tuple)
                landmarks.append(landmark_frame)


                #Paths:
                archivos_total_clase=len(os.listdir(path_clase))
                name_archivo=f"{clase}_{archivos_total_clase}"
                
                    
                #Ruta donde se va a guardar el archivo
                ruta_final=os.path.join(path_clase,f"{name_archivo}.npy")


                #Si se capturan los 'x' frames --> se termina la grabación
                if len(landmarks)>=frames:
                    print("Finalizado!!")
                    
                    #Guardo el numpy array
                    np.save(ruta_final,np.array(landmarks))
                    print(f"Muestra guardada {name_archivo}. Shape: {np.array(landmarks).shape}")
                    

                    break
                
                #Si el tiempo de la seña es menor al valor de la cantidad de frames máximo, el usuario, al presionar 'a', puede detener la grabación
                else:
                    detener_capturing=cv2.waitKey(1)

                    if detener_capturing==ord("a"):
                        np.save(ruta_final,np.array(landmarks))
                        print(f"Muestra guardada {name_archivo}. Shape: {np.array(landmarks).shape}")
                        break

             #=========#


            #Dibujo los landmarks
            draw_lms(frame,landmarks_tuple)
            
            
            #Coloco el texto
            cv2.putText(frame, texto, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            

            #Muestro la imagen
            cv2.imshow("Frame",frame)


            #Presionar 'q' si se quiere salir antes de iniciar la captura de keypoints
            tecla=cv2.waitKey(1)
            
            if tecla==ord("q"):
                break
    
    #print(f"path clase '{path_clase}' exists?: {os.path.exists(path_clase)}")


main()
