import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from termcolor import colored


class star_dataset(Dataset):

    def __init__(self, device: str, features: np.array, target: np.array):
        """
        Initialisation du DataLoader
        ---
        Paramètres :
        - device: appareil sur lequel stocker les tenseurs
        - features: features du jeu de données
        - target: target du jeu de données
        """
        self.device = device
        # récupération des séries spatio-temporelles
        self.features = torch.from_numpy(features).float().to(self.device)
        self.target = torch.from_numpy(target).float().to(self.device)

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, item):
        return self.features[item], self.target[item]


def get_data_loader(data: dict, batch_size: int, device: str, shuffle=False):
    """
    Chargement des jeux de données
    ---
    Paramètres :
    - data: dictionnaire contenant les différents jeux de données
    - batch_size: taille des batchs pour charger les données au modèle
    - device: appareil sur lequel stocker les tenseurs
    - shuffle: bool, si True alors les données sont chargées dans un ordre aléatoire
    """
    data_loader = dict()  # data_loader for [train, valid, test]
    for mode in ["train", "valid", "test"]:
        if mode in data.keys():
            feat_dict = data[mode]["features"]
            target = data[mode]["target"]
            dataset = star_dataset(device=device, features=feat_dict, target=target)
            data_loader[mode] = DataLoader(
                dataset=dataset, batch_size=batch_size, shuffle=shuffle
            )
    return data_loader
