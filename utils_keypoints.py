#Funciones auxiliares para la recolección de datos
import torch
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
    #if tuple_lms.pose_landmarks:
    #    draw.draw_landmarks(frame,tuple_lms.pose_landmarks,mp_holistic.POSE_CONNECTIONS)

    #Draw left hand connections
    if tuple_lms.left_hand_landmarks:
        draw.draw_landmarks(frame,tuple_lms.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)

    #Draw right hand connections
    if tuple_lms.right_hand_landmarks:
        draw.draw_landmarks(frame,tuple_lms.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)



#Función para convertir las coordenadas a coordenadas relativas con respecto al wrist [0,:]
def coord_rel(kps): #kps deben estar en la forma [30,126]
    
    kps_reshape=kps.view(kps.size(0),42,3) #30 frames, 42 landmarks (2 manos), 3 coordenadas
    
    mano1=kps_reshape[:,:21,:] #(30,21,3)
    mano2=kps_reshape[:,21:,:] #(30,21,3)

    
    wrist1=mano1[:,0:1,:] #(30,1,3)
    wrist2=mano2[:,0:1,:] #(30,1,3)

    
    rel1=mano1-wrist1 #(30,1,3)
    rel2=mano2-wrist2 #(30,1,3)


    kps_rel=torch.cat([rel1,rel2],dim=1) #(30,42,3)
    return kps_rel.reshape(kps_rel.size(0),kps_rel.size(1)*kps_rel.size(2)) #(30,126)



#Función para extraer los keypoints: "Flatten" para que el vector sea de 1d y las coordenadas 'x', 'y', 'z' estén en un solo array
def extract_keypoints(lms_tuple):
    
    #Landmarks left hand
    if lms_tuple.left_hand_landmarks:
        lms_left_hand=np.array([[res.x,res.y,res.z] for res in lms_tuple.left_hand_landmarks.landmark],dtype=np.float32) #Coordenadas originales. Shape=(30,63)

    else:
        lms_left_hand=np.zeros(shape=(21,3)) #Esto es lo que causa que el modelo realice predicciones erróneas cuando NO recibe 
 


    #Landmarks right hand
    if lms_tuple.right_hand_landmarks:
        lms_right_hand=np.array([[res.x,res.y,res.z] for res in lms_tuple.right_hand_landmarks.landmark],dtype=np.float32) #Coordenadas originales
       
    else:
        lms_right_hand=np.zeros(shape=(21,3))

   
    #Landmarks pose
    #lms_pose=np.array([[res.x,res.y,res.z,res.visibility] for res in lms_tuple.pose_landmarks.landmark]).flatten() if lms_tuple.pose_landmarks else np.zeros(shape=33*4)


    #Landmarks face
    #lms_face=np.array([[res.x,res.y] for res in lms_tuple.face_landmarks.landmark]).flatten() if lms_tuple.face_landmarks else np.zeros(shape=364)
    

    #Concateno, ya que el rellenar con 'ceros' igual me asegura de que se tendrán los vectores, así no se hayan mostrado en la cámara directamente
    landmarks_final=np.concatenate([lms_left_hand,lms_right_hand]) #[lms_left_hand,lms_right_hand] [:21, 21:]
    
    return landmarks_final.flatten() #[42,3]
    
