import torch
import numpy as np


class MnistFashionDataset(torch.utils.data.Dataset):
    """ Classe que representa o dataset do Mnist-Fashion\n
        params:\n
        `dir_dados`: local onde se encontra um arquivo CSV contendo as imagens e suas classificacoes"""

    def __init__(self, dir_dados):
        D = np.loadtxt(dir_dados, delimiter=',', dtype=int)
        self.len = len(D)
        self.imagens = torch.tensor(D[:, 1:], dtype=torch.float32)

        # transformando as classificacoes em vetores 1-hot
        # ex: classe = 3 -> vet = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
        a = D[:,  0]
        b = np.zeros((a.size, a.max()+1))
        b[np.arange(a.size),a] = 1
        a = b
        self.classes = torch.tensor(a, dtype=torch.float32)

    def __getitem__(self, index):
        return (self.imagens[index], self.classes[index])
    
    def __len__(self):
        return self.len