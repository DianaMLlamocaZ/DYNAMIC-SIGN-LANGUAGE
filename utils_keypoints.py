#Funciones auxiliares para la recolección de datos

import mediapipe as mp
import numpy as np

#Holistic model
mp_holistic = mp.solutions.holistic

#Draw utilities
draw=mp.solutions.drawing_utils



#Función para dibujar los keypoints del modelo: pose + hands (left,right), face
def draw_lms(frame,tuple_lms):
    #Draw connections face
    #if tuple_lms.face_landmarks:
    #    draw.draw_landmarks(frame,tuple_lms.face_landmarks,mp_holistic.FACEMESH_TESSELATION)
    
    #Draw pose connections
    if tuple_lms.pose_landmarks:
        draw.draw_landmarks(frame,tuple_lms.pose_landmarks,mp_holistic.POSE_CONNECTIONS)

    #Draw left hand connections
    if tuple_lms.left_hand_landmarks:
        draw.draw_landmarks(frame,tuple_lms.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)

    #Draw right hand connections
    if tuple_lms.right_hand_landmarks:
        draw.draw_landmarks(frame,tuple_lms.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)




#Función para extraer los keypoints: "Flatten" para que el vector sea de 1d y las coordenadas 'x' e 'y' estén en un solo array
def extract_keypoints(lms_tuple):
    
    #Landmarks left hand
    lms_left_hand=np.array([[res.x,res.y] for res in lms_tuple.left_hand_landmarks.landmark]).flatten() if lms_tuple.left_hand_landmarks else np.zeros(shape=21*2)
    
    
    #Landmarks right hand
    lms_right_hand=np.array([[res.x,res.y] for res in lms_tuple.right_hand_landmarks.landmark]).flatten() if lms_tuple.right_hand_landmarks else np.zeros(shape=21*2)


    #Landmarks pose
    lms_pose=np.array([[res.x,res.y,res.z,res.visibility] for res in lms_tuple.pose_landmarks.landmark]).flatten() if lms_tuple.pose_landmarks else np.zeros(shape=33*4)


    #Landmarks face
    #lms_face=np.array([[res.x,res.y] for res in lms_tuple.face_landmarks.landmark]).flatten() if lms_tuple.face_landmarks else np.zeros(shape=364)
    

    #Concateno, ya que el rellenar con 'ceros' igual me asegura de que se tendrán los vectores, así no se hayan mostrado en la cámara directamente
    landmarks_final=np.concatenate([lms_left_hand,lms_right_hand,lms_pose])
    

    return landmarks_final #Retorno los landmarks

    

