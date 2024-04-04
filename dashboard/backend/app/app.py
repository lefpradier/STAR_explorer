import torch
import json
import numpy as np
import stmgcn
import preprocessor as pp
import pickle as pkl
import pandas as pd
from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F


#! CHARGEMENT DU MODÈLE
model_file = "/opt/ml/ST_MGCN_best_model.pkl"
model = stmgcn.ST_MGCN(
    M=4,
    seq_len=5,
    n_nodes=431,
    input_dim=1,
    lstm_hidden_dim=64,
    lstm_num_layers=3,
    gcn_hidden_dim=64,
    sta_kernel_config={"kernel_type": "chebyshev", "K": 4},
    gconv_use_bias=True,
    gconv_activation=nn.ReLU,
)
saved_checkpoint = torch.load(model_file, map_location=torch.device("cpu"))
model.load_state_dict(saved_checkpoint["state_dict"])
model.eval()

#!DÉFINITION DES KERNELS DE CONVOLUTION
mat_file = "/opt/ml/adj_matrices2.pkl"
mats = pkl.load(open(mat_file, "rb"))
sta_adj_list = list()
for key in list(mats.keys()):
    adj_preprocessor = pp.Adj_Preprocessor(kernel_type="chebyshev", K=4)
    adj_mat = torch.from_numpy(mats[key]).float()
    adj_mat = adj_preprocessor.process(adj_mat)
    sta_adj_list.append(adj_mat)

#!CHARGEMENT DECONNEXION
disconnect = pkl.load(open("/opt/ml/carreaux_disconnect_function.pkl", "rb"))


#!INTEGRATION DES FCT (handler) et LES CONTEXTES
# La fonction prend *toujours* 2 arguments : event et context
# event : c'est le contenu de la requête HTTP
# context : c'est le contexte de la requête, pas nécessairement utile
def lambda_handler(event, context):
    # mise en forme des données
    # features
    input_features = np.asarray(json.loads(event["body"])["obs_seq"])
    input_features = np.float64(input_features)  # convert to float32
    input_features = np.expand_dims(np.asarray(input_features), 0)
    # conversion en tensor
    features = torch.from_numpy(input_features).float()
    # modification de la matrice de connectivité STAR si topo1 et topo2 ont été sélectionnés
    topo1 = json.loads(event["body"])["topo1"]
    topo2 = json.loads(event["body"])["topo2"]
    mat = deepcopy(mats["star_adj"])
    if topo1 is not None and topo2 is not None:
        to_disconnect = disconnect.loc[
            ((disconnect["c1"] == int(topo1)) & (disconnect["c2"] == int(topo2)))
            | ((disconnect["c1"] == int(topo2)) & (disconnect["c2"] == int(topo1))),
            "disconnect",
        ].values[0]
        for i1, i2 in to_disconnect:
            mat[i1, i2] = 0
        mat = torch.from_numpy(mat).float()
        adj_preprocessor = pp.Adj_Preprocessor(kernel_type="chebyshev", K=4)
        adj_mat = adj_preprocessor.process(mat)
        sta_adj_list[1] = adj_mat
    # prédiction
    predict = model(obs_seq=features, sta_adj_list=sta_adj_list)
    # nécessaire de transformation Tensor.tolist() pour passer à JSON
    return {"statusCode": 200, "body": json.dumps({"prediction": predict.tolist()})}
