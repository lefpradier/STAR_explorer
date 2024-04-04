import torch
import numpy as np


class Adj_Preprocessor(object):
    def __init__(self, kernel_type: str, K: int):
        """
        Initialisation de la classe de préprocessing des matrices d'adjacence
        ---
        Paramètres :
        - kernel_type: choix parmi 'chebyshev', 'localpool', et 'random_walk_diffusion'
        - K: ordre du kernel, c'est-à-dire nombre de valeurs propres de la matrice laplacienne
        """
        self.kernel_type = kernel_type
        # chebyshev (Defferard NIPS'16)/localpool (Kipf ICLR'17)/random_walk_diffusion (Li ICLR'18)
        self.K = K if self.kernel_type != "localpool" else 1
        # max_chebyshev_polynomial_order (Defferard NIPS'16)/max_diffusion_step (Li ICLR'18)

    def process(self, adj: torch.Tensor):
        """
        Génération des kernels pour la convolution de graphe
        ---
        Paramètres :
        - adj: matrice d'adjacence de dimension (N, N)
        """
        kernel_list = list()

        if self.kernel_type in ["localpool", "chebyshev"]:  # spectral
            adj_norm = self.symmetric_normalize(adj)
            # adj_norm = self.random_walk_normalize(adj)     # for asymmetric normalization
            if self.kernel_type == "localpool":
                localpool = (
                    torch.eye(adj_norm.shape[0]) + adj_norm
                )  # same as add self-loop first
                kernel_list.append(localpool)

            else:  # chebyshev
                laplacian_norm = torch.eye(adj_norm.shape[0]) - adj_norm
                rescaled_laplacian = self.rescale_laplacian(laplacian_norm)
                kernel_list = self.compute_chebyshev_polynomials(
                    rescaled_laplacian, kernel_list
                )

        elif self.kernel_type == "random_walk_diffusion":  # spatial
            # diffuse k steps on transition matrix P
            P_forward = self.random_walk_normalize(adj)
            kernel_list = self.compute_chebyshev_polynomials(P_forward.T, kernel_list)
        else:
            raise ValueError(
                "Invalid kernel_type. Must be one of [chebyshev, localpool, random_walk_diffusion]."
            )

        kernels = torch.stack(kernel_list, dim=0)

        return kernels

    @staticmethod
    def random_walk_normalize(A):  # asymmetric
        d_inv = torch.pow(A.sum(dim=1), -1)  # OD matrix Ai,j sum on j (axis=1)
        d_inv[torch.isinf(d_inv)] = 0.0
        D = torch.diag(d_inv)
        A_norm = torch.mm(D, A)
        return A_norm

    @staticmethod
    def symmetric_normalize(A):
        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))
        A_norm = torch.mm(torch.mm(D, A), D)
        return A_norm

    @staticmethod
    def rescale_laplacian(L):
        # rescale laplacian to arccos range [-1,1] for input to Chebyshev polynomials of the first kind
        try:
            lambda_ = torch.eig(L)[0][:, 0]  # get the real parts of eigenvalues
            lambda_max = lambda_.max()  # get the largest eigenvalue
        except:
            lambda_max = 2
        L_rescale = (2 / lambda_max) * L - torch.eye(L.shape[0])
        return L_rescale

    def compute_chebyshev_polynomials(self, x, T_k):
        for k in range(self.K + 1):
            if k == 0:
                T_k.append(torch.eye(x.shape[0]))
            elif k == 1:
                T_k.append(x)
            else:
                T_k.append(2 * torch.mm(x, T_k[k - 1]) - T_k[k - 2])
        return T_k


def convert_matrix_to_edges(mat, device="cuda:0"):
    """
    Conversion d'une matrice d'adjacence en liste d'edges
    ---
    Paramètres :
    - mat: np.array de dimension (N,N)
    - device: str, localisation des tenseurs
    """
    n = mat.shape[0]
    edges = []
    weights = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if mat[i, j] > 0:
                edges.append([i, j])
                weights.append(mat[i, j])
    # convert to array
    edges = np.array(edges).T
    weights = np.array(weights)
    # convert to tensor object
    edges = torch.from_numpy(edges).to(device)
    weights = torch.from_numpy(weights).to(device)
    return edges, weights
