import os
import torch
from torch import nn, optim
import stmgcn, dataloader, model_trainer, preprocessor
import pickle as pkl
import numpy as np
from termcolor import colored
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


def train_stmgcn(
    epoch=30,
    batch_size=64,
    learn_rate=2e-3,
    weight_decay=1e-4,
    M_adj=4,
    sta_kernel_config={"kernel_type": "chebyshev", "K": 2},
    loss_opt="MSE",
    device="cuda:0",
    model_dir="./output",
    drop_adj_list=[],
    verbose=True,
    early_stopper=10,
    idx_adj=3,
):
    """
    Création et entraînement d'un modèle ST-MGCN
    ---
    Paramètres :
    - epoch: int, nombre d'epochs
    - batch_size: int, taille des batchs
    - learn_rate: float, taux d'apprentissage
    - weight_decay: float, taux de diminution des poids
    - M_adj: int, nombre de matrices d'adjacence à utiliser dans le modèle
    - sta_kernel_config: dict, paramètres du kernel de convolution sur graphe
    - loss_opt: str, fonction de perte à optimiser
    - device: str, appareil sur lequel entraîner le modèle
    - model_dir: str, répertoire dans lequel enregistrer le modèle
    - drop_adj_list: list, liste des matrices d'adjacence à supprimer du modèle
    - verbose: bool, si True afficher les messages d'avancée du modèle
    - early_stopper: int, nombre d'epochs au dessus duquel on arrête l'entraînement s'il n'y a pas eu d'amélioration de la loss
    - idx_adj: int, indice des données à charger pour l'entraînement du modèle
    """
    torch.manual_seed(42)

    #!load data
    data = pkl.load(open("data%s.pkl" % str(idx_adj), "rb"))
    adj = pkl.load(open("adj_matrices%s.pkl" % str(idx_adj), "rb"))

    #!prepare static adjs
    sta_adj_list = list()
    for key in list(adj.keys()):
        if key not in drop_adj_list:
            n_nodes = adj[key].shape[0]
            adj_preprocessor = preprocessor.Adj_Preprocessor(**sta_kernel_config)
            adj_mat = torch.from_numpy(adj[key]).float()
            adj_mat = adj_preprocessor.process(adj_mat)
            sta_adj_list.append(adj_mat.to(device))
    assert len(sta_adj_list) == M_adj

    #!load data
    data_loader = dataloader.get_data_loader(
        data=data, batch_size=batch_size, device=device
    )

    #!model
    model = stmgcn.ST_MGCN(
        M=M_adj,
        seq_len=5,
        n_nodes=n_nodes,
        input_dim=1,
        lstm_hidden_dim=64,
        lstm_num_layers=3,
        gcn_hidden_dim=64,
        sta_kernel_config=sta_kernel_config,
        gconv_use_bias=True,
        gconv_activation=nn.ReLU,
    )
    model = model.to(device)

    if loss_opt == "MSE":
        loss = nn.MSELoss(reduction="mean")
    elif loss_opt == "MAE":
        loss = nn.L1Loss(reduction="mean")
    elif loss_opt == "Huber":
        loss = nn.SmoothL1Loss(reduction="mean")
    else:
        raise Exception("Unknown loss function.")
    optimizer = optim.Adam

    trainer = model_trainer.ModelTrainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr=learn_rate,
        wd=weight_decay,
        n_epochs=epoch,
    )

    os.makedirs(model_dir, exist_ok=True)

    #!training
    trainer.train(
        data_loader=data_loader,
        sta_adj_list=sta_adj_list,
        modes=["train", "valid"],
        model_dir=model_dir,
        verbose=verbose,
        early_stopper=early_stopper,
    )

    torch.cuda.empty_cache()

    #!evaluation
    mse, rmse, mae = trainer.test(
        data_loader=data_loader,
        sta_adj_list=sta_adj_list,
        modes=["valid"],
        model_dir=model_dir,
    )
    print(colored("training complete", "red"))
    return mse, rmse, mae


def train_stmgcn_external(
    epoch=30,
    batch_size=64,
    learn_rate=2e-3,
    weight_decay=1e-4,
    M_adj=3,
    sta_kernel_config={"kernel_type": "chebyshev", "K": 2},
    loss_opt="MSE",
    device="cuda:0",
    model_dir="./output",
    verbose=True,
    early_stopper=10,
):
    """
    Création et entraînement d'un modèle ST-MGCN sur le jeu de données externe
    ---
    Paramètres :
    - epoch: int, nombre d'epochs
    - batch_size: int, taille des batchs
    - learn_rate: float, taux d'apprentissage
    - weight_decay: float, taux de diminution des poids
    - M_adj: int, nombre de matrices d'adjacence à utiliser dans le modèle
    - sta_kernel_config: dict, paramètres du kernel de convolution sur graphe
    - loss_opt: str, fonction de perte à optimiser
    - device: str, appareil sur lequel entraîner le modèle
    - model_dir: str, répertoire dans lequel enregistrer le modèle
    - verbose: bool, si True afficher les messages d'avancée du modèle
    - early_stopper: int, nombre d'epochs au dessus duquel on arrête l'entraînement s'il n'y a pas eu d'amélioration de la loss
    """
    torch.manual_seed(42)

    #!load data
    data = pd.read_csv("data_external/data.csv")
    adj_list = [
        "data_external/weight_adj.csv",
        "data_external/weight_dis.csv",
        "data_external/weight_simi.csv",
    ]
    #! preprocess data
    data = data.stack().reset_index()
    data.columns = ["time", "patch", "demand"]
    idx_train = [i for i in range(int(0.75 * data["time"].nunique()))]
    train, valid = (
        data[data["time"].isin(idx_train)],
        data[~data["time"].isin(idx_train)],
    )
    temp = {"t-1": 1, "t-2": 2, "t-3": 3, "t-1j": 24, "t-7j": 24 * 7}
    for k in temp.keys():
        train[k] = train.groupby("patch")["demand"].shift(periods=temp[k])
        valid[k] = valid.groupby("patch")["demand"].shift(periods=temp[k])
    train.dropna(inplace=True)
    valid.dropna(inplace=True)
    my_data = {}
    #!TRAINING SET
    my_data["train"] = {}
    my_data["train"]["features"] = np.stack(
        train.groupby("time")
        .apply(lambda x: x[["t-7j", "t-1j", "t-3", "t-2", "t-1"]].to_numpy().T)
        .to_numpy()
    )[:, :, :, None]
    my_data["train"]["target"] = np.stack(
        train.groupby("time").apply(lambda x: x["demand"].to_numpy()).to_numpy()
    )[:, :, None]
    #!VALIDATION SET
    my_data["valid"] = {}
    my_data["valid"]["features"] = np.stack(
        valid.groupby("time")
        .apply(lambda x: x[["t-7j", "t-1j", "t-3", "t-2", "t-1"]].to_numpy().T)
        .to_numpy()
    )[:, :, :, None]
    my_data["valid"]["target"] = np.stack(
        valid.groupby("time").apply(lambda x: x["demand"].to_numpy()).to_numpy()
    )[:, :, None]

    #!prepare static adjs
    sta_adj_list = list()
    for key in adj_list:
        adj = pd.read_csv(key, names=["i" + str(i) for i in range(30)])
        adj = adj.to_numpy()
        n_nodes = adj.shape[0]
        adj_preprocessor = preprocessor.Adj_Preprocessor(**sta_kernel_config)
        adj_mat = torch.from_numpy(adj).float()
        adj_mat = adj_preprocessor.process(adj_mat)
        sta_adj_list.append(adj_mat.to(device))
    assert len(sta_adj_list) == M_adj  # ensure sta adj dim correct

    #!load data
    data_loader = dataloader.get_data_loader(
        data=my_data, batch_size=batch_size, device=device
    )

    #!model
    model = stmgcn.ST_MGCN(
        M=M_adj,
        seq_len=5,
        n_nodes=n_nodes,
        input_dim=1,
        lstm_hidden_dim=64,
        lstm_num_layers=3,
        gcn_hidden_dim=64,
        sta_kernel_config=sta_kernel_config,
        gconv_use_bias=True,
        gconv_activation=nn.ReLU,
    )
    model = model.to(device)

    if loss_opt == "MSE":
        loss = nn.MSELoss(reduction="mean")
    elif loss_opt == "MAE":
        loss = nn.L1Loss(reduction="mean")
    elif loss_opt == "Huber":
        loss = nn.SmoothL1Loss(reduction="mean")
    else:
        raise Exception("Unknown loss function.")
    optimizer = optim.Adam

    trainer = model_trainer.ModelTrainer(
        model=model,
        loss=loss,
        optimizer=optimizer,
        lr=learn_rate,
        wd=weight_decay,
        n_epochs=epoch,
    )

    os.makedirs(model_dir, exist_ok=True)

    #!training
    trainer.train(
        data_loader=data_loader,
        sta_adj_list=sta_adj_list,
        modes=["train", "valid"],
        model_dir=model_dir,
        verbose=verbose,
        early_stopper=early_stopper,
    )

    torch.cuda.empty_cache()

    #!evaluation
    mse, rmse, mae = trainer.test(
        data_loader=data_loader,
        sta_adj_list=sta_adj_list,
        modes=["valid"],
        model_dir=model_dir,
    )
    print(colored("training complete", "red"))
    return mse, rmse, mae


def load_stmgcn(
    path,
    M_adj=4,
    sta_kernel_config={"kernel_type": "chebyshev", "K": 2},
    device="cuda:0",
):
    """
    Chargement d'un modèle ST-MGCN sauvegardé
    ---
    Paramètres :
    - path: str, chemin où trouver le fichier de sauvegarde des poids
    - M_adj: int, nombre de matrices d'adjacence utilisées par le modèle
    - sta_kernel_config: dict, paramètres du kernel de convolution sur graphe
    - device: str, appareil sur lequel faire les calculs du modèle
    """
    model = stmgcn.ST_MGCN(
        M=M_adj,
        seq_len=5,
        n_nodes=431,
        input_dim=1,
        lstm_hidden_dim=64,
        lstm_num_layers=3,
        gcn_hidden_dim=64,
        sta_kernel_config=sta_kernel_config,
        gconv_use_bias=True,
        gconv_activation=nn.ReLU,
    )
    #!compile
    model = model.to(device)
    saved_checkpoint = torch.load(path)
    model.load_state_dict(saved_checkpoint["state_dict"])
    model.eval()
    print(colored("model loaded", "blue"))
    return model
