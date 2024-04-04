from pyproj import Transformer
import pickle as pkl
import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
import plotly.express as px
import shapely.geometry


def get_carreaux():
    """
    Chargement des coordonnées des carreaux
    """
    #!chargement de carreaux (idx et coord)
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326")
    carreaux_list = pkl.load(open("carreaux2.pkl", "rb"))
    carreaux = pd.DataFrame({"index": carreaux_list})
    #!séparation des coord en comp nord et est
    carreaux["north"] = carreaux["index"].str.split("N").str[0].astype(float)
    carreaux["east"] = (
        carreaux["index"].str.split("N").str[1].str.split("E").str[0].astype(float)
    )
    carreaux["northeast"] = (
        carreaux["north"].astype(int).astype(str)
        + "N "
        + carreaux["east"].astype(int).astype(str)
        + "E"
    )
    carreaux_list = carreaux["northeast"].tolist()
    #!passage des coor en lat et long
    latlong = transformer.transform(carreaux["north"], carreaux["east"])
    carreaux["north"] = latlong[0]
    carreaux["east"] = latlong[1]
    return carreaux, carreaux_list


def plot_network_go(carreaux, net, t, lignes, ligne):
    """
    Création de la carte de fréquentation observée
    ---
    Paramètres :
    - carreaux: pd.DataFrame, coordonnées des carreaux de 500 mètres de côté
    - net: np.array, demande à chaque pas de temps
    - lignes: dict, coordonnées des lignes de bus/métro
    - ligne: str, indice de la ligne de bus/métro à afficher
    """
    #!remplissage du dict des options graphiques
    lats = lignes[ligne]["lats"]
    lons = lignes[ligne]["lons"]
    names = lignes[ligne]["names"]
    cols = lignes[ligne]["cols"]
    #!creation du réseau et association avec la carte chargée
    minlat, maxlat = carreaux["north"].min(), carreaux["north"].max()
    minlon, maxlon = carreaux["east"].min(), carreaux["east"].max()
    medlat = np.mean([minlat, maxlat])
    medlon = np.mean([minlon, maxlon])
    dotsize = 0.5 + 9.5 * (net[t, :] - np.min(net[t, :])) / (
        np.max(net[t, :]) - np.min(net[t, :])
    )
    fig = px.scatter_mapbox(
        lon=carreaux["east"],
        lat=carreaux["north"],
        color=net[t, :],
        labels="Fréquentation",
        # symbol_sequence=["square"],
        size=dotsize,
        # color_continuous_scale=[[0, "#4D4D4D"], [0.5, "#FFB4B4"], [1, "#FF1717"]],
        color_continuous_scale=[[0, "#FFFFFF"], [0.5, "#FFB4B4"], [1, "#FF1717"]],
        range_color=[0, np.quantile(net, 0.95)],
    )
    fig.add_traces(
        list(
            px.line_mapbox(
                lat=lats, lon=lons, hover_name=names, color_discrete_sequence=cols
            ).select_traces()
        )
    )
    rangelat = maxlat - minlat
    rangelon = maxlon - minlon

    fig.update_mapboxes(
        bounds={
            "north": maxlat + 0.1 * rangelat,
            "south": minlat - 0.1 * rangelat,
            "east": maxlon + 0.1 * rangelon,
            "west": minlon - 0.1 * rangelon,
        }
    )
    fig.update_layout(
        {"plot_bgcolor": "#424D5C", "paper_bgcolor": "#424D5C"},
        height=500,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar_title_text="Fréquentation",
        font_color="white",
        title_font_color="white",
        mapbox_style="mapbox://styles/mapbox/navigation-night-v1",
        mapbox_accesstoken="pk.eyJ1IjoibGVmcHJhZGllciIsImEiOiJjbHNpc2FvMHowbjE4MmpwbmpwbXRhODM5In0.iyOanIJR0ZjfyUjXccvOPA",
    )
    fig.data = (fig.data[1], fig.data[0])
    return fig

def plot_saturation_go(net, t: int, ts: list, hours: list):
    """
    Création de la figure de saturation pour les données observées
    ---
    Paramètres :
    - net: np.array, demande à chaque pas de temps
    - t: int, heure à laquelle la prédiction est faite
    - ts: list, indices heures de la journée dans laquelle la prédiction est faite
    - hours: list, liste des heures
    """
    # remise en forme
    ##fenetre temporelle
    # bornage des fenetres temporelles
    saturation_ask = np.sum(net[ts, :], axis=1)
    # interpolate
    hours_new = np.linspace(np.min(hours), np.max(hours), 1000)
    spl = make_interp_spline(hours, saturation_ask, k=3)
    power_smooth = spl(hours_new)
    fig = px.scatter(
        x=hours_new,
        y=power_smooth,
        color=power_smooth,
        color_continuous_scale=[[0, "#FFFFFF"], [0.5, "#FFB4B4"], [1, "#FF1717"]],
        range_color=[0, 60000],
    )
    fig.add_vline(x=t, line_color="white", line_dash="dash")
    fig.update_layout(
        {"plot_bgcolor": "#424D5C", "paper_bgcolor": "#424D5C"},
        height=500,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        font_color="white",
        xaxis={"showgrid": False},
        yaxis={"showgrid": False},
        yaxis_range=[0, 60000],
        xaxis_title="Heure",
        yaxis_title="Fréquentation totale",
        coloraxis_colorbar_title_text="Fréquentation totale",
    )
    fig.update_xaxes(showline=True, linecolor="white")
    fig.update_yaxes(showline=True, linecolor="white")
    return fig


def plot_network_pred(carreaux, net, pred, lignes, ligne, topo1=None, topo2=None):
    """
    Création de la carte de fréquentation prédite
    ---
    Paramètres :
    - carreaux: pd.DataFrame, coordonnées des carreaux de 500 mètres de côté
    - net: np.array, demande à chaque pas de temps
    - pred: np.array, demande prédite
    - lignes: dict, coordonnées des lignes de bus/métro
    - ligne: str, indice de la ligne de bus/métro à afficher
    - topo1: str, première station entre lesquelles on souhaite supprimer une connexion
    - topo2: str, deuxième station entre lesquelles on souhaite supprimer une connexion
    """
    #!remplissage du dict des options graphiques
    lats = lignes[ligne]["lats"]
    lons = lignes[ligne]["lons"]
    names = lignes[ligne]["names"]
    cols = lignes[ligne]["cols"]
    #!creation du réseau et association avec la carte chargée
    minlat, maxlat = carreaux["north"].min(), carreaux["north"].max()
    minlon, maxlon = carreaux["east"].min(), carreaux["east"].max()
    medlat = np.mean([minlat, maxlat])
    medlon = np.mean([minlon, maxlon])
    dotsize = 0.5 + 9.5 * (pred - np.min(pred)) / (np.max(pred) - np.min(pred))
    fig = px.scatter_mapbox(
        lon=carreaux["east"],
        lat=carreaux["north"],
        color=pred,
        labels="Fréquentation",
        # symbol_sequence=["square"],
        size=dotsize,
        # color_continuous_scale=[[0, "#4D4D4D"], [0.5, "#FFB4B4"], [1, "#FF1717"]],
        color_continuous_scale=[[0, "#FFFFFF"], [0.5, "#FFB4B4"], [1, "#FF1717"]],
        range_color=[0, np.quantile(net, 0.95)],
    )
    fig.add_traces(
        list(
            px.line_mapbox(
                lat=lats, lon=lons, hover_name=names, color_discrete_sequence=cols
            ).select_traces()
        )
    )
    if topo1 is not None:
        fig.add_traces(
            px.scatter_mapbox(
                lon=carreaux.loc[carreaux["northeast"] == topo1, "east"],
                lat=carreaux.loc[carreaux["northeast"] == topo1, "north"],
            )
            .update_traces(marker_size=20, marker_color="yellow")
            .data
        )
        fig.data = (fig.data[1], fig.data[0], fig.data[2])
    if topo2 is not None:
        fig.add_traces(
            px.scatter_mapbox(
                lon=carreaux.loc[carreaux["northeast"] == topo2, "east"],
                lat=carreaux.loc[carreaux["northeast"] == topo2, "north"],
            )
            .update_traces(marker_size=20, marker_color="yellow")
            .data
        )
        fig.data = (fig.data[1], fig.data[0], fig.data[2], fig.data[3])
    rangelat = maxlat - minlat
    rangelon = maxlon - minlon
    fig.update_mapboxes(
        bounds={
            "north": maxlat + 0.1 * rangelat,
            "south": minlat - 0.1 * rangelat,
            "east": maxlon + 0.1 * rangelon,
            "west": minlon - 0.1 * rangelon,
        }
    )
    fig.update_layout(
        {"plot_bgcolor": "#424D5C", "paper_bgcolor": "#424D5C"},
        height=500,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        coloraxis_colorbar_title_text="Fréquentation",
        font_color="white",
        title_font_color="white",
        mapbox_style="mapbox://styles/mapbox/navigation-night-v1",
        mapbox_accesstoken="pk.eyJ1IjoibGVmcHJhZGllciIsImEiOiJjbHNpc2FvMHowbjE4MmpwbmpwbXRhODM5In0.iyOanIJR0ZjfyUjXccvOPA",
    )
    return fig


def plot_saturation_test(net, t: int, ts: list, hours: list, pred=None):
    """
    Création de la figure de saturation pour les données prédites
    ---
    Paramètres :
    - net: np.array, demande à chaque pas de temps
    - t: int, heure à laquelle la prédiction est faite
    - ts: list, indices des heures de la journée dans laquelle la prédiction est faite
    - hours: list, liste des heures
    - pred: float, demande totale prédite à l'heure demandée
    """
    # remise en forme
    ##fenetre temporelle
    # bornage des fenetres temporelles
    saturation_ask = np.sum(net[ts, :], axis=1)
    # interpolate
    hours_new = np.linspace(np.min(hours), np.max(hours), 1000)
    spl = make_interp_spline(hours, saturation_ask, k=3)
    power_smooth = spl(hours_new)
    fig = px.scatter(
        x=hours_new,
        y=power_smooth,
        color=power_smooth,
        color_continuous_scale=[[0, "#FFFFFF"], [0.5, "#FFB4B4"], [1, "#FF1717"]],
        range_color=[0, 60000],
    )

    fig.add_vline(x=t, line_color="white", line_dash="dash")
    fig.update_layout(
        {"plot_bgcolor": "#424D5C", "paper_bgcolor": "#424D5C"},
        height=500,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        font_color="white",
        xaxis={"showgrid": False},
        yaxis={"showgrid": False},
        yaxis_range=[0, 60000],
        xaxis_title="Heure",
        yaxis_title="Fréquentation totale",
        coloraxis_colorbar_title_text="Fréquentation totale",
    )
    fig.update_xaxes(showline=True, linecolor="white")
    fig.update_yaxes(showline=True, linecolor="white")
    if pred is not None:
        fig.add_traces(
            px.scatter(x=[t], y=[pred])
            .update_traces(marker_size=20, marker_color="yellow")
            .data
        )
    return fig

