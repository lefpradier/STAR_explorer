import numpy as np
import pickle as pkl
import pandas as pd
import torch
import captum
import main_stmgcn as mstmgcn
from preprocessor import Adj_Preprocessor
from captum.attr import Saliency, IntegratedGradients, KernelShap, FeatureAblation
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore")


def explain_star(x, input_adj, kernel_list, i, model, saturation=False):
    """
    Fonction d'explicabilité du modèle analysant les connexions du réseau STAR
    ---
    Paramètres :
    - x: torch.Tensor, données d'entrée
    - input_adj: np.array, matrice d'adjacence d'entrée
    - kernel_list: liste des kernels non modifiés
    - i: indice du nouveau kernel pour remplacement dans la kernel_list
    - model: classe STMGCN
    - saturation: bool, si True alors output calculé comme la saturation du réseau
    """

    #! FCT charge le model, les matrices adj
    def model_forward(adj, x, kernel_list, i, model, saturation):
        # refold adj
        n = int(np.sqrt(adj.shape[1]))
        adj = torch.squeeze(torch.reshape(adj, (adj.shape[0], n, n)))
        adj_preprocessor = Adj_Preprocessor(kernel_type="chebyshev", K=4)
        adj_mat = adj_preprocessor.process(adj)
        kernel_list[i] = adj_mat.to("cuda:0")
        out = model(x, kernel_list)
        if saturation:
            output = min(torch.sum(out) / 50000, 1)[None,]
        else:
            output = torch.squeeze(out)[:, None]
        return output

    #!dépliage matrice
    input_adj = torch.from_numpy(input_adj).float()
    n = input_adj.shape[0]
    input_adj = torch.reshape(input_adj, (1, n * n))
    ig = IntegratedGradients(model_forward)
    torch.backends.cudnn.enabled = False
    baselines = torch.zeros((1, n * n))
    mask = ig.attribute(
        input_adj,
        baselines=baselines,
        additional_forward_args=(x, kernel_list, i, model, saturation),
        internal_batch_size=1,
        n_steps=50,
    )
    adj_mask = np.reshape(mask.cpu().detach().numpy(), (n, n))
    return adj_mask


def explain_metro(x, input_adj, kernel_list, i, model, saturation=False):
    """
    Fonction d'explicabilité du modèle analysant les variables fonctionnelles de la métropole de Rennes
    ---
    Paramètres :
    - x: torch.Tensor, données d'entrée
    - input_adj: np.array, jeu de données non modifié des variables fonctionnelles de la métropole de Rennes
    - kernel_list: liste des kernels non modifiés
    - i: indice du nouveau kernel pour remplacement dans la kernel_list
    - model: classe STMGCN
    - saturation: bool, si True alors output calculé comme la saturation du réseau
    """

    #! FCT charge le model, les matrices adj
    def model_forward(input_eye, metro, x, kernel_list, i, model, saturation):
        # cancel out some variables
        metro = torch.matmul(metro, input_eye)
        # compute cosine similarity
        adj = F.cosine_similarity(metro[None, :, :], metro[:, None, :], dim=-1)
        adj_preprocessor = Adj_Preprocessor(kernel_type="chebyshev", K=4)
        adj_mat = adj_preprocessor.process(adj)
        kernel_list[i] = adj_mat.to("cuda:0")
        out = model(x, kernel_list)
        if saturation:
            output = min(torch.sum(out) / 50000, 1)[None,]
        else:
            output = torch.squeeze(out)[:, None]
        return output

    #!dépliage matrice
    metro = torch.from_numpy(input_adj).float()
    n = metro.shape[1]
    ig = IntegratedGradients(model_forward)
    torch.backends.cudnn.enabled = False
    input_eye = torch.eye(n).requires_grad_(True)
    baselines = input_eye * 0
    mask = ig.attribute(
        input_eye,
        baselines=baselines,
        additional_forward_args=(metro, x, kernel_list, i, model, saturation),
        internal_batch_size=1,
        n_steps=50,
    )
    adj_mask = np.diagonal(mask.cpu().detach().numpy())
    return adj_mask


def explain_time(x, kernel_list, model, saturation=False):
    """
    Fonction d'explicabilité du modèle analysant les connexions du réseau STAR
    ---
    Paramètres :
    - x: torch.Tensor, données d'entrée
    - kernel_list: liste des kernels
    - model: classe STMGCN
    - saturation: bool, si True alors output calculé comme la saturation du réseau
    """

    #! FCT charge le model, les matrices adj
    def model_forward(input_tensor, x, kernel_list, model, saturation=False):
        x = torch.matmul(input_tensor, torch.squeeze(x))[None, :, :, None]
        out = model(x, kernel_list)
        if saturation:
            output = min(torch.sum(out) / 50000, 1)[None,]
        else:
            output = torch.squeeze(out)[:, None]
        return output

    input_tensor = torch.eye(5).requires_grad_(True).to("cuda:0")
    #!dépliage matrice
    ig = IntegratedGradients(model_forward)
    torch.backends.cudnn.enabled = False
    baselines = input_tensor * 0
    mask = ig.attribute(
        input_tensor,
        baselines=baselines,
        additional_forward_args=(x, kernel_list, model, saturation),
        internal_batch_size=1,
        n_steps=50,
    )
    adj_mask = np.squeeze(np.diagonal(mask.cpu().detach().numpy()))
    return adj_mask
