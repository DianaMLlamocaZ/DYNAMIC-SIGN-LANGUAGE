import numpy as np
import mediapipe as mp
import cv2
from collections import deque
import torch
import os

from utils_keypoints import extract_keypoints
from model import Modelo

#Aquí se almacenan los 30 frames para predicción
sequences=deque([],maxlen=30)

#Softmax function para convertir logits a probs
sm=torch.nn.Softmax(dim=1)

#Holistic model
md_holistic=mp.solutions.holistic

#Draw utilities
draw=mp.solutions.drawing_utils

#Cargo los pesos del modelo
weights_model=torch.load("./modelo.pth",weights_only=True)

#Instancio al modelo
modelo=Modelo(input_size=126,hidden_state_size=32,output_size=2)

#Cargo los pesos AL modelo
modelo.load_state_dict(weights_model)

#Clases
path_dataset="./dataset"
clases={clase_num:clase_str for clase_num,clase_str in enumerate(os.listdir(path_dataset))}



with md_holistic.Holistic(model_complexity=1,min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic_model:
        print("Presiona 'q' para salir")

        cam=cv2.VideoCapture(0)

        while True:

            #Captura de frames y flip para efecto espejo
            open,frame=cam.read()
            frame=cv2.flip(frame,1)
            

            #Conversión a RGB y predicción del modelo
            landmarks_tuple=holistic_model.process(frame[:,:,::-1])


            #Obtengo los keypoints (aquí ya no hay manejo de 'keypoints' en frame, ya que la función lo realiza)
            landmarks_extracted=extract_keypoints(landmarks_tuple)


            #Añado los landmarks extracted al deque
            sequences.append(landmarks_extracted)


            #El modelo fue entrenado con '30 frames', entonces aquí aplico dicha lógica y le paso al modelo cuando len == 30
            if len(sequences)==30:
                  with torch.no_grad():
                        modelo.eval()

                        #Predicción del modelo
                        seq_a_predecir=torch.tensor(np.array(sequences),dtype=torch.float32).unsqueeze(0) #shape 'sequences': [30,126] --> .unsqueeze(0) batch dimension, pues modelo es batch_first=True 
                        
                        #Le paso al modelo la secuencia
                        logits_pred=modelo(seq_a_predecir)
                        
                        #print(f"logits pred: {logits_pred}, shapee: {logits_pred.shape}")
                        #Conversión de logits a probabilidades
                        probs_pred=sm(logits_pred)


                        #Obtengo la clase 'predicha' por el modelo
                        clase_pred=torch.argmax(probs_pred,dim=1).item()
                        print(f"clase predicha: {clase_pred}")
                        
                        #Coloco la predicción en pantalla
                        cv2.putText(frame,f"Pred: {clases[clase_pred]}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)


            #Muestro la cámara
            cv2.imshow("real time pred",frame)

            #Si el usuario quiere salir de la pred in real time, presiona "q"
            tecla=cv2.waitKey(10)

            if tecla==ord("q"):
                  break