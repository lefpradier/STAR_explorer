from torch import nn
import torch
import torch_geometric.nn as tgnn
from termcolor import colored


# /!\ n_nodes = nombre de carreaux


class CG_LSTM(nn.Module):
    """
    Classe Context-Gated LSTM
    """

    def __init__(
        self,
        seq_len: int,
        n_nodes: int,
        input_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        K: int,
        gconv_use_bias: bool,
        gconv_activation=nn.ReLU,
    ):
        """
        Paramètres:
        - seq_len: nombre de pas de temps sur les séries temporelles
        - n_nodes: nombre de carreaux dans les matrices d'adjacence
        - input_dim: nombre de neurones d'entrée
        - lstm_hidden_dim: nombre de neurones par couche cachée dans le LSTM
        - lstm_num_layers: nombre de couches cachées dans le LSTM
        - K: dimension de la convolution de graphe
        - gconv_use_bias: utilisation de biais pour la convolution de graphe
        - gconv_activation: fonction d'activation en sortie de la convolution de graphe
        """
        super().__init__()
        self.seq_len = seq_len
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers

        self.gconv_temporal_feats = GCN(
            K=K,
            input_dim=seq_len,
            hidden_dim=seq_len,
            bias=gconv_use_bias,
            activation=gconv_activation,
        )
        self.fc = nn.Linear(in_features=seq_len, out_features=seq_len, bias=True)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

    def forward(self, adj: torch.Tensor, obs_seq: torch.Tensor, hidden: tuple):
        """
        Context Gated LSTM:
        ---
        Paramètres :
        - adj: kernel pour la convolution de graphe
        - obs_seq: série temporelle sur le graphe
        - hidden: tuple, états cachés de la séquence
        """
        batch_size = obs_seq.shape[0]
        x_seq = obs_seq.sum(dim=-1)  # sum up feature dimension: default 1

        # channel-wise attention on timestep
        x_seq = x_seq.permute(0, 2, 1)
        x_seq_gconv = self.gconv_temporal_feats(A=adj, x=x_seq)
        x_hat = torch.add(x_seq, x_seq_gconv)  # eq. 6
        z_t = x_hat.sum(dim=1) / x_hat.shape[1]  # eq. 7
        s = torch.sigmoid(self.fc(torch.relu(self.fc(z_t))))  # eq. 8
        obs_seq_reweighted = torch.einsum("btnf,bt->btnf", [obs_seq, s])  # eq. 9

        # global-shared LSTM
        shared_seq = obs_seq_reweighted.permute(0, 2, 1, 3).reshape(
            batch_size * self.n_nodes, self.seq_len, self.input_dim
        )
        x, hidden = self.lstm(shared_seq, hidden)

        output = x[:, -1, :].reshape(batch_size, self.n_nodes, self.lstm_hidden_dim)
        return output, hidden

    def init_hidden(self, batch_size: int):
        """
        Initialisation des états cachés de la série temporelle
        ---
        Paramètres :
        - batch_size: taille des batchs
        """
        weight = next(self.parameters()).data
        hidden = (
            weight.new_zeros(
                self.lstm_num_layers, batch_size * self.n_nodes, self.lstm_hidden_dim
            ),
            weight.new_zeros(
                self.lstm_num_layers, batch_size * self.n_nodes, self.lstm_hidden_dim
            ),
        )
        return hidden


class ST_MGCN(nn.Module):
    """
    Classe Spatio-Temporal Multigraph Convolution Network
    """

    def __init__(
        self,
        M: int,
        seq_len: int,
        n_nodes: int,
        input_dim: int,
        lstm_hidden_dim: int,
        lstm_num_layers: int,
        gcn_hidden_dim: int,
        sta_kernel_config: dict,
        gconv_use_bias: bool,
        gconv_activation=nn.ReLU,
    ):
        """
        Paramètres :
        - M: nombre de matrices d'adjacence
        - seq_len: longueur de la série temporelle
        - n_nodes: nombre de carreaux dans le graphe
        - input_dim: nombre de neurones d'entrée
        - lstm_hidden_dim: nombre de neurones par couche cachée du LSTM
        - lstm_hidden_layers: nombre de couches cachées du LSTM
        - gcn_hidden_dim: nombre de neurones de la couche cachée du GCN
        - sta_kernel_config: paramètres de configuration des kernels de convolution de graphe
        - gconv_use_bias: utilisation du biais sur la convolution de graphe
        - gconv_activation: fonction d'activation du module GCN
        """
        super().__init__()
        self.M = M
        self.sta_K = self.get_support_K(sta_kernel_config)

        # initiate one pair of CG_LSTM & GCN for each adj input
        self.rnn_list, self.gcn_list = nn.ModuleList(), nn.ModuleList()
        for m in range(self.M):
            cglstm = CG_LSTM(
                seq_len=seq_len,
                n_nodes=n_nodes,
                input_dim=input_dim,
                lstm_hidden_dim=lstm_hidden_dim,
                lstm_num_layers=lstm_num_layers,
                K=self.sta_K,
                gconv_use_bias=gconv_use_bias,
                gconv_activation=gconv_activation,
            )
            self.rnn_list.append(cglstm)
            gcn = GCN(
                K=self.sta_K,
                input_dim=lstm_hidden_dim,
                hidden_dim=gcn_hidden_dim,
                bias=gconv_use_bias,
                activation=gconv_activation,
            )
            self.gcn_list.append(gcn)
        self.fc = nn.Linear(
            in_features=gcn_hidden_dim, out_features=input_dim, bias=True
        )

    @staticmethod
    def get_support_K(config: dict):
        if config["kernel_type"] == "localpool":
            assert config["K"] == 1
            K = 1
        elif config["kernel_type"] == "chebyshev":
            K = config["K"] + 1
        elif config["kernel_type"] == "random_walk_diffusion":
            K = config["K"] * 2 + 1
        else:
            raise ValueError(
                "Invalid kernel_type. Must be one of [chebyshev, localpool, random_walk_diffusion]."
            )
        return K

    def init_hidden_list(self, batch_size: int):
        """
        Initialisation des états cachés de la série temporelle
        ---
        Paramètres :
        - batch_size: taille des batchs
        """
        hidden_list = list()
        for m in range(self.M):
            hidden = self.rnn_list[m].init_hidden(batch_size)
            hidden_list.append(hidden)
        return hidden_list

    def forward(self, obs_seq: torch.Tensor, sta_adj_list: list):
        """
        Paramètres :
        - obs_seq: série temporelle
        - sta_adj_list: liste des kernels de convolution
        """
        assert len(sta_adj_list) == self.M
        batch_size = obs_seq.shape[0]
        hidden_list = self.init_hidden_list(batch_size)

        feat_list = list()
        for m in range(self.M):
            cg_rnn_out, hidden_list[m] = self.rnn_list[m](
                sta_adj_list[m], obs_seq, hidden_list[m]
            )
            gcn_out = self.gcn_list[m](sta_adj_list[m], cg_rnn_out)
            feat_list.append(gcn_out)
        feat_fusion = torch.sum(torch.stack(feat_list, dim=-1), dim=-1)  # aggregation

        output = self.fc(feat_fusion)
        return output


class GCN(nn.Module):
    """
    Classe Graph Convolution Network
    """

    def __init__(
        self, K: int, input_dim: int, hidden_dim: int, bias=True, activation=nn.ReLU
    ):
        """
        Paramètres :
        - K: nombre de valeurs propres à prendre sur la matrice laplacienne pour générer le kernel de convolution
        - input_dim: nombre de neurones d'entrée
        - hidden_dim: nombre de neurones de la couche cachée
        - bias: utilisation du biais pour la convolution
        - activation: fonction d'activation
        """
        super().__init__()
        self.K = K
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.activation = activation() if activation is not None else None
        self.init_params(n_supports=K)

    def init_params(self, n_supports: int, b_init=0):
        self.W = nn.Parameter(
            torch.empty(n_supports * self.input_dim, self.hidden_dim),
            requires_grad=True,
        )
        nn.init.xavier_normal_(self.W)
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.hidden_dim), requires_grad=True)
            nn.init.constant_(self.b, val=b_init)

    def forward(self, A: torch.Tensor, x: torch.Tensor):
        """
        Convolution de graphe
        ---
        Paramètres :
        - A: kernel de convolution
        - x: features
        """
        assert self.K == A.shape[0]

        support_list = list()
        for k in range(self.K):
            support = torch.einsum("ij,bjp->bip", [A[k, :, :], x])
            support_list.append(support)
        support_cat = torch.cat(support_list, dim=-1)

        output = torch.einsum("bip,pq->biq", [support_cat, self.W])
        if self.bias:
            output += self.b
        output = self.activation(output) if self.activation is not None else output
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"({self.K} * input {self.input_dim} -> hidden {self.hidden_dim})"
        )
