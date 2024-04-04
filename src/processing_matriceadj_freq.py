import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from pyproj import Transformer
import sys
from termcolor import colored
import pickle as pkl
from sklearn.metrics import pairwise_distances


#! UTILS
def projection_carreau(df, coord="coordonnees"):
    """
    Projection des coordonnées latitude-longitude vers le système EPSG 3035 de l'INSEE
    ---
    Paramètres :
    - df: pd.DataFrame
    - coord: string, nom de la colonne contenant les coordonnées
    """
    df[["latitude", "longitude"]] = df[coord].str.split(", ", expand=True).astype(float)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3035")
    xy = transformer.transform(df["latitude"], df["longitude"])
    df["code_north"] = xy[0]
    df["code_east"] = xy[1]
    df["carreau_north"] = np.floor(df["code_north"] / 500) * 500
    df["carreau_east"] = np.floor(df["code_east"] / 500) * 500
    return df


def data_rennes(insee, arrets):
    """
    Importation des données fonctionnelles
    ---
    Paramètres :
    - insee: pd.DataFrame, contient les données fonctionnelles issues de l'INSEE
    - arrets: pd.DataFrame, contient les arrêts du réseau de transport STAR projetés sur carreaux de 500 mètres de côté
    """
    # Importation des données des commerces de la métropole de Rennes
    comm = pd.read_csv("data/commerces.csv", sep=";")
    comm = projection_carreau(comm)
    # Suppression des catégories de commerces avec moins de 50 représentants
    cat_comm = comm.groupby("type").size().reset_index()
    comm = comm[comm["type"].isin(cat_comm.loc[cat_comm[0] > 50, "type"])]
    comm = (
        comm.groupby(["carreau_north", "carreau_east", "type"])
        .size()
        .reset_index()
        .pivot(index=["carreau_north", "carreau_east"], columns=["type"])
        .fillna(0)
    )
    comm.columns = comm.columns.droplevel(0)
    comm = comm.reset_index()

    # Importation des données des écoles de la métropole de Rennes
    ecoles = pd.read_csv("data/ecoles.csv", sep=";")
    ecoles = projection_carreau(ecoles, "geo_point_2d")
    ecoles = (
        ecoles.groupby(["carreau_north", "carreau_east"])
        .agg({"nb_eleves": "sum"})
        .reset_index()
    )

    # Importation des données des parkings de la métropole de Rennes
    park = pd.read_csv("data/parkings.csv", sep=";")
    park = projection_carreau(park, "geo_point_2d")
    park = (
        park.groupby(["carreau_north", "carreau_east"])
        .agg({"nb_pl": "sum"})
        .reset_index()
    )

    # Concaténation des différents jeux de données
    rennes = pd.merge(park, ecoles, on=["carreau_north", "carreau_east"], how="outer")
    rennes = pd.merge(rennes, park, on=["carreau_north", "carreau_east"], how="outer")
    rennes = pd.merge(rennes, comm, on=["carreau_north", "carreau_east"], how="outer")
    rennes.fillna(0, inplace=True)
    rennes = pd.merge(rennes, insee, on=["carreau_north", "carreau_east"], how="left")
    rennes = rennes.dropna()
    carreaux_star = arrets[["carreau_north", "carreau_east"]].drop_duplicates(
        keep="first"
    )
    rennes = pd.merge(
        rennes, carreaux_star, on=["carreau_north", "carreau_east"], how="inner"
    )
    rennes["carreau_index"] = (
        rennes["carreau_north"].astype(str)
        + "N"
        + rennes["carreau_east"].astype(str)
        + "E"
    )
    return rennes


def import_demand_data(stations):
    """
    Importation des données de fréquentation du réseau STAR et holdout
    ---
    Paramètres :
    - stations: pd.DataFrame, correspondance entre stations du réseau et carreaux de 500 mètres
    """
    train = []
    valid = []
    test = []
    # Liste des fichiers de fréquentation
    freq_path = [
        f
        for f in os.listdir("data/frequentation")
        if os.path.isfile(os.path.join("data/frequentation", f))
        and not f.endswith(".zip")
    ]
    # Importation et traitement de chacun des fichiers
    for path in freq_path:
        freq = pd.read_csv(os.path.join("data/frequentation", path), sep=";")
        freq["DateTime"] = pd.to_datetime(
            freq["DateFreq"] + " " + freq["TrancheHoraire15mn"], dayfirst=True
        )
        freq["Frequentation"] = freq["Frequentation"].str.split(",").str[0].astype(int)
        # Association aux carreaux de 500 mètres de côté
        freq = freq.merge(stations, left_on="NomArret", right_on="nomarret")
        # regroupement des tranches horaires de 15 min en heures puis group en carreaux puis somme de freq
        # sur ces carreaux
        freq = (
            freq.groupby(
                [pd.Grouper(key="DateTime", axis=0, freq="H"), "carreau_index"]
            )
            .agg({"Frequentation": "sum"})
            .reset_index()
        )
        # holdout sur l'an 2022
        if np.any(freq["DateTime"].dt.year == 2022):
            train.append(freq)
        # valid sur 5 permiers mois de 2023
        elif np.any(freq["DateTime"].dt.month <= 5):
            valid.append(freq)
        # les autres mois de 2023 dans le test
        else:
            test.append(freq)
    # concaténation des diffents df issus des fichiers de freq
    train = pd.concat(train)
    valid = pd.concat(valid)
    test = pd.concat(test)

    # génération des combinaisons manquantes et remplissage par 0
    train = (
        train.pivot(index="carreau_index", columns="DateTime", values="Frequentation")
        .fillna(0)
        .stack()
        .to_frame()
        .reset_index()
        .rename(columns={0: "Frequentation"})
    )
    valid = (
        valid.pivot(index="carreau_index", columns="DateTime", values="Frequentation")
        .fillna(0)
        .stack()
        .to_frame()
        .reset_index()
        .rename(columns={0: "Frequentation"})
    )
    test = (
        test.pivot(index="carreau_index", columns="DateTime", values="Frequentation")
        .fillna(0)
        .stack()
        .to_frame()
        .reset_index()
        .rename(columns={0: "Frequentation"})
    )
    return train, valid, test


def vois_geo(carreaux, max_dist=np.inf):
    """
    Calcul de la matrice d'adjacence de voisinage géographique
    ---
    Paramètres :
    - carreaux: liste de carreaux
    - max_dist: float, distance maximale au dessus de laquelle on déconnecte deux carreaux
    """
    # Matrice carrée carreau-par-carreau
    N = len(carreaux)
    geo = np.zeros((N, N))
    # Boucle sur les carreaux
    for i in range(N - 1):
        for j in range(i + 1, N):
            # pour une paire de carreaux i, j :
            c1 = carreaux[i]
            c2 = carreaux[j]
            # extraction des coordonnées
            c1 = (float(c1.split("N")[0]), float(c1.split("N")[1].split("E")[0]))
            c2 = (float(c2.split("N")[0]), float(c2.split("N")[1].split("E")[0]))
            #! mesure du voisinage de c1 et c2 et normalisation par les distances des voisins directs
            # 500 distances voisins directs
            c = np.array([abs(c1[0] - c2[0]) / 500, abs(c1[1] - c2[1]) / 500])
            #!passage en distance euclidienne
            # floor permet de discretiser les distances courtes
            c = np.floor(np.sqrt(sum(c**2)))
            #!seuil au dessus duquel plus de voisinnage
            if c < max_dist:
                # donner le meme poids a tous les voisins du premier cercle
                #! distance euclidienne entre c1 et c2
                geo[i, j] = 1 / c**2
                geo[j, i] = 1 / c**2
        return geo


def vois_star(carreaux, parcours, geo, type_dist="discrete", max_dist=0):
    """
    Calcul de la matrice d'adjacence de connectivité sur le réseau STAR
    ---
    Paramètres :
    - carreaux: liste des carreaux de 500 mètres de côté
    - parcours: pd.DataFrame, parcours des lignes de bus et de métro du réseau STAR
    - geo: np.array, matrice d'adjacence de voisinage géographique
    - type_dist: string, si 'discrete' alors les distances sont calculées en 0/1,
                si 'continuous' alors elles sont calculées par l'inverse de la distance
    - max_dist: float, distance maximale au dessus de laquelle on déconnecte deux carreaux
    """
    # Matrice carrée carreau-par-carreau
    N = len(carreaux)
    connect = np.zeros((N, N))
    # Boucle sur les carreaux
    for i in range(N - 1):
        for j in range(i + 1, N):
            # calcul de connectivité pour les carreaux non-adjacents uniquement
            # on exclut les carreaux adjacents afin de ne pas reprendre l'information de la matrice de voisinage géographique
            if geo[i, j] < 1:
                # pour une paire de carreaux i, j :
                c1 = carreaux[i]
                c2 = carreaux[j]
                # y a-t-il des lignes communes ? sont-elles voisines ?
                lignes1 = parcours.loc[
                    parcours["carreau_index"] == c1, "idparcours"
                ].tolist()
                lignes2 = parcours.loc[
                    parcours["carreau_index"] == c2, "idparcours"
                ].tolist()
                #! connectedness: connectés par une ligne ou non
                if np.any(np.isin(lignes1, lignes2)):
                    lignes = [l for l in lignes1 if l in lignes2]
                    #!si oui, pour chaque ligne commune, quel nombre d'arrêts les séparent ?
                    chemins = []
                    #!pour chaque ligne commune
                    for l in lignes:
                        o1 = parcours.loc[
                            (parcours["carreau_index"] == c1)
                            & (parcours["idparcours"] == l),
                            "ordre",
                        ].tolist()
                        o2 = parcours.loc[
                            (parcours["carreau_index"] == c2)
                            & (parcours["idparcours"] == l),
                            "ordre",
                        ].tolist()
                        #!ajout de la distance pour chaque station des lignes concernées
                        chemins.extend([abs(oi1 - oi2) for oi1 in o1 for oi2 in o2])
                        #! CHOIX METHODE DE CONNEXION
                        # même ligne = voisins
                        if type_dist == "discrete" and min(chemins) < max_dist:
                            connect[i, j] = 1
                            connect[j, i] = 1
                        # même ligne = voisin mais pondéré par la distance
                        elif type_dist == "continuous" and min(chemins) < max_dist:
                            connect[i, j] = 1 / min(chemins)
                            connect[j, i] = 1 / min(chemins)
    return connect


def similarite_fonctionnelle(rennes, carreaux):
    """
    Calcul des matrices de similarité fonctionnelle
    ---
    Paramètres :
    - rennes: pd.DataFrame, contient les variables fonctionnelles par carreau
    - carreaux: liste des carreaux
    """
    rennes = rennes[rennes["carreau_index"].isin(carreaux)]
    rennes["carreau_index"] = pd.Categorical(
        rennes["carreau_index"], categories=carreaux
    )
    rennes.sort_values(by="carreau_index", inplace=True)
    rennes.index = rennes["carreau_index"]
    rennes_metro = rennes.drop(columns=[i for i in rennes.columns if "2019" in i])
    rennes_metro.drop(
        columns=["Code", "Libellé", "carreau_north", "carreau_east", "carreau_index"],
        inplace=True,
    )
    rennes_insee = rennes.drop(columns=[i for i in rennes.columns if "2019" not in i])

    sim_metro = 1 - pairwise_distances(rennes_metro, metric="cosine", n_jobs=-2)
    sim_insee = 1 - pairwise_distances(
        rennes_insee.dropna(), metric="braycurtis", n_jobs=-2
    )
    return sim_metro, sim_insee, rennes_metro


def sorting_carreaux(df, carreaux):
    """
    Sorting des jeux de données en fonction de l'ordre canonique des carreaux
    ---
    Paramètres :
    - df: pd.DataFrame
    - carreaux: liste des carreaux
    """
    df = df[df["carreau_index"].isin(carreaux)]
    df["carreau_index"] = pd.Categorical(df["carreau_index"], categories=carreaux)
    df.sort_values(by="carreau_index", inplace=True)
    return df


def time_features(df):
    """
    Création d'une série temporelle de 5 pas de temps pour chaque observation
    ---
    Paramètres :
    - df: pd.DataFrame
    """
    temp = {"t-1": 1, "t-2": 2, "t-3": 3, "t-1j": 24, "t-7j": 24 * 7}
    df.sort_values(by="DateTime", inplace=True)
    for k in temp.keys():
        df[k] = df.groupby("carreau_index")["Frequentation"].shift(periods=temp[k])
    df.dropna(inplace=True)
    return df


def time_series_stmgcn(df):
    """
    Conversion des données vers une forme exploitable par le modèle ST-MGCN
    ---
    Paramètres :
    - df: pd.DataFrame
    """
    out = {}
    out["features"] = (
        df.groupby("DateTime")
        .apply(lambda x: x[["t-7j", "t-1j", "t-3", "t-2", "t-1"]].to_numpy().T)
        .to_numpy()
    )
    out["target"] = (
        df.groupby("DateTime").apply(lambda x: x["Frequentation"].to_numpy()).to_numpy()
    )
    return out


#!IMPORTATION DES DONNÉES POUR LES MATRICES D'ADJACENCE
# Importation des données INSEE
insee = pd.read_csv("data/data_insee.csv", sep=";", header=2)
insee = insee[insee["Code"].str.contains("3035N")]
# Conversion du code en coordonnées nord et est
insee["carreau_north"] = (
    insee["Code"].apply(lambda st: st[st.find("N") + 1 : st.find("E")]).astype(int)
    * 1000
)
insee["carreau_east"] = insee["Code"].str.split("E").str[1].astype(int) * 1000
# Subdivision des carreaux de 1km de côté en carreaux de 500 mètres de côté
insee_list = []
for i in [0, 500]:
    for j in [0, 500]:
        alt_insee = insee.copy(deep=True)
        alt_insee["carreau_north"] += i
        alt_insee["carreau_east"] += j
        insee_list.append(alt_insee)
insee = pd.concat(insee_list)

# Importation des coordonnées des arrêts du réseau STAR
arrets_bus = pd.read_csv("data/arrets_bus.csv", sep=";")
arrets_metro = pd.read_csv("data/arrets_metro.csv", sep=";")
arrets = pd.concat([arrets_metro, arrets_bus])
arrets = projection_carreau(arrets)

# Importation des données fonctionnelles de la métropole de Rennes
rennes = data_rennes(insee, arrets)


#! RECONSTITUTION RÉSEAU BUS METRO
bus = pd.read_csv("data/parcours_bus.csv", sep=";")
metro = pd.read_csv("data/parcours_metro.csv", sep=";")
parcours = pd.concat([bus, metro])
# Agrégation des stations aux carreaux de 500 mètres
parcours = pd.merge(
    parcours,
    arrets[["id", "carreau_north", "carreau_east"]],
    left_on="idarret",
    right_on="id",
)
parcours["carreau_index"] = (
    parcours["carreau_north"].astype(str)
    + "N"
    + parcours["carreau_east"].astype(str)
    + "E"
)
stations = parcours[["nomarret", "carreau_index"]].drop_duplicates()

#!RÉCUPÉRATION DES DONNÉES DE FRÉQUENTATION
#!HOLDOUT
train, valid, test = import_demand_data(stations)


#!SOUS-ENSEMBLE DE CARREAUX OÙ LES DONNÉES SONT TOUTES PRÉSENTES
carreaux = parcours["carreau_index"].unique().tolist()
train_carreaux = train["carreau_index"].unique().tolist()
valid_carreaux = valid["carreau_index"].unique().tolist()
test_carreaux = test["carreau_index"].unique().tolist()
rennes_carreaux = rennes["carreau_index"].unique().tolist()
carreaux = [
    c
    for c in carreaux
    if c in train_carreaux
    and c in valid_carreaux
    and c in test_carreaux
    and c in rennes_carreaux
]

#!CALCUL DES MATRICES D'ADJACENCE
sim_metro, sim_insee, rennes_metro = similarite_fonctionnelle(rennes, carreaux)
geo = vois_geo(carreaux, max_dist=int(sys.argv[1]))
connect = vois_star(
    carreaux, parcours, geo, type_dist=sys.argv[2], max_dist=int(sys.argv[3])
)

#!SUPPRESSION DES CARREAUX DÉCONNECTÉS DES AUTRES
idx_to_drop = []
idx_to_drop.append(np.where(np.sum(sim_metro, axis=1) == 0))
idx_to_drop.append(np.where(np.sum(sim_insee, axis=1) == 0))
idx_to_drop.append(np.where(np.sum(geo, axis=1) == 0))
idx_to_drop.append(np.where(np.sum(connect, axis=1) == 0))
idx_to_drop = np.unique(np.concatenate(idx_to_drop, axis=None))
sim_metro = np.delete(np.delete(sim_metro, idx_to_drop, axis=0), idx_to_drop, axis=1)
sim_insee = np.delete(np.delete(sim_insee, idx_to_drop, axis=0), idx_to_drop, axis=1)
connect = np.delete(np.delete(connect, idx_to_drop, axis=0), idx_to_drop, axis=1)
geo = np.delete(np.delete(geo, idx_to_drop, axis=0), idx_to_drop, axis=1)
carreaux_to_drop = [carreaux[i] for i in idx_to_drop]
train = train[~train["carreau_index"].isin(carreaux_to_drop)]
valid = valid[~valid["carreau_index"].isin(carreaux_to_drop)]
test = test[~test["carreau_index"].isin(carreaux_to_drop)]
carreaux = [c for c in carreaux if c not in carreaux_to_drop]
# Exportation de la liste des carreaux
rennes_metro = rennes_metro.drop(index=carreaux_to_drop)
rennes_metro.to_csv("rennes_metro%s.csv" % sys.argv[4], index=False)
pkl.dump(carreaux, open("carreaux%s.pkl" % sys.argv[4], "wb"))

#! sorting des carreaux
train = sorting_carreaux(train, carreaux)
valid = sorting_carreaux(valid, carreaux)
test = sorting_carreaux(test, carreaux)

#!EXPORTATION DES MATRICES D'ADJACENCE
graphs = {
    "neighbor_adj": geo,
    "star_adj": connect,
    "insee_adj": sim_insee,
    "metro_adj": sim_metro,
}
pkl.dump(graphs, open("adj_matrices%s.pkl" % sys.argv[4], "wb"))

################################SERIES TEMPORELLES################################################

#!decoupage en tranche de temps pour le LSTM
#!shift permet de decaler en ligne (chercher la valeur juste avant par ex)
train = time_features(train)
valid = time_features(valid)
test = time_features(test)
pkl.dump({"train": train, "valid": valid, "test": test}, open("data_pandas.pkl", "wb"))

#!CONVERSION POUR INPUT ST-MGCN

data = {}
for idx, df in zip(["train", "valid", "test"], [train, valid, test]):
    data[idx] = time_series_stmgcn(df)
    N = data[idx]["features"].shape[0]
    M = data[idx]["features"][0].shape[1]
    data[idx]["features"] = np.concatenate(data[idx]["features"]).reshape(N, 5, M, 1)
    data[idx]["target"] = np.concatenate(data[idx]["target"]).reshape(N, M, 1)

pkl.dump(data, open("data%s.pkl" % sys.argv[4], "wb"))
