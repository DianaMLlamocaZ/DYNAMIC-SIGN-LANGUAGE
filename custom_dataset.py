#Aquí defino el custom dataset
import torch
import os
import numpy as np


class DatasetDL(torch.utils.data.Dataset):
    def __init__(self,path_dataset):
        #print(os.listdir(path_dataset))
        
        self.kps_sample=[]
        self.target=[]

        self.clases={clase_str:clase_num for clase_num,clase_str in enumerate(os.listdir(path_dataset))}
        
        for clase in os.listdir(path_dataset):
            path_clase=f"./dataset/{clase}"
            for sample_clase in os.listdir(path_clase):
                #print(f"sample clase: {sample_clase}")
                sample=np.load(f"{path_clase}/{sample_clase}")
                #print("sample:",sample,sample.shape)
                self.kps_sample.append(sample)
                self.target.append(self.clases[clase])
        

    def __len__(self):
        return len(self.target)


    def __getitem__(self,index):
        sample=torch.tensor(self.kps_sample[index])
        tgt=torch.tensor(self.target[index])
        #print(f"targeeet: {tgt},{tgt.shape}")
        return sample,tgt