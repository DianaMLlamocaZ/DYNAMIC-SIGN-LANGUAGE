from custom_dataset import DatasetDL
from torch.utils.data import DataLoader
import torch

from utils_dataloader import add_padding
from model import Modelo

#Instancio un dataset y un dataloader para preparar los datos
dataset=DatasetDL("./dataset")
dataloader=DataLoader(dataset,batch_size=6,shuffle=True,collate_fn=add_padding)

#Instancio el modelo
modelo=Modelo(input_size=126,hidden_state_size=32,output_size=2)


#Hiperparámetros y losses
epocas=30
loss=torch.nn.CrossEntropyLoss()

#Optimizer
optim=torch.optim.Adam(modelo.parameters(),lr=0.001)

#Error por épocas
train_error_epoca=[]


#Entrenamiento
for epoca in range(epocas):
    error_epoca=0

    for batch in dataloader:
        kps,tg=batch
        kps=kps.to(torch.float32)
        
        #Predicción del modelo
        clase_pred=modelo(kps)

        #Loss
        error=loss(clase_pred,tg)

        #Actualización de parámetros
        optim.zero_grad()
        error.backward()
        optim.step()

        #Sumando error para promedio por época
        error_epoca+=error.item()

    print(f"Época: {epoca}. Error: {error_epoca/len(dataloader)}")
    train_error_epoca.append(error_epoca/len(dataloader))


#Guardo el modelo luego de entrenarlo
torch.save(modelo.state_dict(),"./modelo.pth")