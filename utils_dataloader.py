import torch

#Función que añade el padding a las secuencias :)
def add_padding(batch):
    kps=[samples[0] for samples in batch]
    tgt=[samples[1] for samples in batch]

    #Padding
    max_length=max(len(kp) for kp in kps)

    padded_inputs=torch.stack([
        torch.cat([seq,torch.zeros(max_length-len(seq))]) 
        for seq in kps
    ])
    
    #Retorno los valores
    return padded_inputs,torch.tensor(tgt)