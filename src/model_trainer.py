from torch import nn
import torch
import time
import numpy as np
from termcolor import colored


class ModelTrainer(object):
    def __init__(
        self,
        model: nn.Module,
        loss: nn.Module,
        optimizer,
        lr: float,
        wd: float,
        n_epochs: int,
    ):
        """
        Initialisation du programme d'entraînement
        ---
        Paramètres :
        - model: classe STMGCN
        - loss: fonction de perte
        - optimizer: fonction d'optimisation
        - lr: taux d'apprentissage
        - wd: taux de dégradation des poids
        """
        self.model = model
        self.model_name = self.model.__class__.__name__
        self.criterion = loss
        self.optimizer = optimizer(
            params=self.model.parameters(), lr=lr, weight_decay=wd
        )
        self.n_epochs = n_epochs

    def train(
        self,
        data_loader: dict,
        sta_adj_list: list,
        modes: list,
        model_dir: str,
        early_stopper=10,
        verbose=True,
    ):
        """
        Entraînement du modèle
        ---
        Paramètres :
        - data_loader: objet DataLoader pour le chargement des données par batchs
        - sta_adj_list: liste des kernels de convolution
        - modes: list, choix des évaluations à faire parmi 'train', 'valid', et 'test"
        - model_dir: répertoire dans lequel sauvegarder le modèle
        - early_stopper: nombre d'epochs après lequel on arrête l'entraînement si aucune amélioration des performances
        - verbose: bool, si True alors afficher les mises à jour
        """
        checkpoint = {"epoch": 0, "state_dict": self.model.state_dict()}
        val_loss = np.inf
        init_early_stopper = early_stopper

        print("Training starts at: ", time.ctime())
        for epoch in range(1, self.n_epochs + 1):
            running_loss = {mode: 0.0 for mode in modes}
            for mode in modes:
                if mode == "train":
                    self.model.train()
                else:
                    self.model.eval()

                step = 0
                for x, y_true in data_loader[mode]:
                    with torch.set_grad_enabled(mode=mode == "train"):
                        y_pred = self.model(obs_seq=x, sta_adj_list=sta_adj_list)
                        loss = self.criterion(y_pred, y_true)
                        if mode == "train":
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                    running_loss[mode] += loss * y_true.shape[0]
                    step += y_true.shape[0]

                # epoch end
                if mode == "valid":
                    if running_loss[mode] / step <= val_loss:
                        if verbose:
                            print(
                                f"Epoch {epoch}, Val_loss drops from {val_loss:.5} to {running_loss[mode]/step:.5}. "
                                f"Update model checkpoint.."
                            )
                        val_loss = running_loss[mode] / step
                        checkpoint.update(
                            epoch=epoch, state_dict=self.model.state_dict()
                        )
                        torch.save(checkpoint, model_dir + "/ST_MGCN_best_model.pkl")
                        early_stopper = init_early_stopper
                    else:
                        if verbose:
                            print(
                                f"Epoch {epoch}, Val_loss does not improve from {val_loss:.5}."
                            )
                        early_stopper -= 1
                        if early_stopper == 0:
                            print(f"Early stopping at epoch {epoch}..")
                            return

        print("Training ends at: ", time.ctime())
        torch.save(checkpoint, model_dir + "/ST_MGCN_best_model.pkl")

    def test(self, data_loader: dict, sta_adj_list: list, modes: list, model_dir: str):
        """
        Test du modèle
        ---
        Paramètres :
        - data_loader: objet DataLoader pour le chargement des données par batchs
        - sta_adj_list: liste des kernels de convolution
        - modes: list, choix des évaluations à faire parmi 'valid' et 'test"
        - model_dir: répertoire dans lequel le modèle a été sauvegardé
        """
        saved_checkpoint = torch.load(model_dir + "/ST_MGCN_best_model.pkl")
        self.model.load_state_dict(saved_checkpoint["state_dict"])
        self.model.eval()

        print("Testing starts at: ", time.ctime())
        running_loss = {mode: 0.0 for mode in modes}
        for mode in modes:
            ground_truth, prediction = list(), list()
            for x, y_true in data_loader[mode]:
                with torch.no_grad():  # prevent keeping intermediate variable
                    y_pred = self.model(obs_seq=x, sta_adj_list=sta_adj_list)
                ground_truth.append(y_true.cpu().detach().numpy())
                prediction.append(y_pred.cpu().detach().numpy())

                loss = self.criterion(y_pred, y_true)
                running_loss[mode] += loss * y_true.shape[0]

            ground_truth = np.concatenate(ground_truth, axis=0)
            prediction = np.concatenate(prediction, axis=0)
            mse = self.MSE(prediction, ground_truth)
            rmse = self.RMSE(prediction, ground_truth)
            mae = self.MAE(prediction, ground_truth)
            mape = self.MAPE(prediction, ground_truth)
            print(colored(mode + " true MSE: " + str(mse), "red"))
            print(colored(mode + " true RMSE: " + str(rmse), "red"))
            print(colored(mode + " true MAE: " + str(mae), "red"))
            print(colored(mode + " true MAE: " + str(mape), "red"))
        print("Testing ends at: ", time.ctime())

        return mse, rmse, mae

    @staticmethod
    def MSE(y_pred: np.array, y_true: np.array):
        return np.mean(np.square(y_pred - y_true))

    @staticmethod
    def RMSE(y_pred: np.array, y_true: np.array):
        return np.sqrt(np.mean(np.square(y_pred - y_true)))

    @staticmethod
    def MAE(y_pred: np.array, y_true: np.array):
        return np.mean(np.abs(y_pred - y_true))

    @staticmethod
    def MAPE(y_pred: np.array, y_true: np.array, epsilon=1e-0):  # zero division
        return np.mean(np.abs(y_pred - y_true) / (y_true + epsilon))

    @staticmethod
    def PCC(y_pred: np.array, y_true: np.array):
        return np.corrcoef(y_pred.flatten(), y_true.flatten())[0, 1]
