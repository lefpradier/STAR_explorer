from dash import (
    Dash,
    html,
    dcc,
    dash_table,
)
from dash.dash_table.Format import Format, Scheme, Trim

import plotly.express as px
import dash_ag_grid as dag
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import pandas as pd

import dashboard_utils as du
import pickle as pkl
import numpy as np
import json
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform, State

import requests


#! CHARGEMENT DES DONNÉES
network_demand = pkl.load(open("demand_network.pkl", "rb"))
network_demand_test = pkl.load(open("demand_network_test.pkl", "rb"))
test = pkl.load(open("data_test.pkl", "rb"))
datetime = pkl.load(open("datetime.pkl", "rb"))
datetimetest = pkl.load(open("datetime_test.pkl", "rb"))
df_describe = pkl.load(open("df_describe.pkl", "rb"))
trace_lignes = pkl.load(open("lignes.pkl", "rb"))
carreaux, carreaux_list = du.get_carreaux()
#! MISE EN FORME DES LISTES DE JOURS ET HEURES DES DIFFÉRENTS CRÉNEAUX HORAIRES
jours = [i.split(" ")[0] for i in datetime]
jours_test = [i.split(" ")[0] for i in datetimetest]
hours = [int(i.split(" ")[1].split(":")[0]) for i in datetime]
hours_test = [int(i.split(" ")[1].split(":")[0]) for i in datetimetest]
#! MISE EN FORME DE LA LISTE DES OPTIONS DE LIGNES DE BUS/MÉTRO
lignes = [i for i in trace_lignes.keys()]
#! CHARGEMENT ET MISE EN FORME DES COMBINAISONS AUTORISÉES DE CARREAUX À DÉCONNECTER
# Ces combinaisons ont été testées au préalable de sorte à ce qu'aucune suppression de segment de ligne ne cause de discontinuité dans le réseau.
disc = pkl.load(open("carreaux_disconnect.pkl", "rb"))
disc_list = list(set(disc["c1"].tolist() + disc["c2"].tolist()))
disc_list.sort()
carreaux_disconnect = [carreaux_list[i] for i in disc_list]

#! MISE EN FORME DU TABLEAU
columns = [
    dict(id="Mesure", name=""),
    dict(id="Jour", name="Jour", type="numeric", format=Format(precision=2)),
    dict(
        id="Mois",
        name="Mois",
        type="numeric",
        format=Format(precision=2, scheme=Scheme.decimal_integer),
    ),
    dict(
        id="Année",
        name="Année",
        type="numeric",
        format=Format(precision=2, scheme=Scheme.decimal_integer),
    ),
    dict(
        id="Heure",
        name="Heure",
        type="numeric",
        format=Format(precision=2, scheme=Scheme.decimal_integer),
    ),
    dict(
        id="Latitude",
        name="Latitude",
        type="numeric",
        format=Format(precision=2, scheme=Scheme.fixed),
    ),
    dict(
        id="Longitude",
        name="Longitude",
        type="numeric",
        format=Format(precision=2, scheme=Scheme.fixed),
    ),
    dict(
        id="Fréquentation",
        name="Fréquentation",
        type="numeric",
        format=Format(precision=2, scheme=Scheme.decimal_integer),
    ),
]



#!CREATE APP TEMPLATE
app = DashProxy(
    __name__,
    external_stylesheets=[dbc.themes.SLATE],
    transforms=[MultiplexerTransform()],
)
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            "STAR : réseau de transports en commun de la métropole de Rennes",
                            className="mb-title",
                            style={"textAlign": "center"},
                        ),
                        dcc.Markdown(
                            id="global_desc",
                            children="""
                    - Ce dashboard présente la fréquentation des transports en commun de la métropole de Rennes (Bretagne, France), accessibles [ici](https://data.explore.star.fr/).  
                    - L'onglet **Data**  permet d'observer la fréquentation à chaque heure de l'année 2022. Ces données ont été utilisées pour entrainer un modèle *ST-MGCN* (voir [Geng et al., 2019](https://ojs.aaai.org/index.php/AAAI/article/view/4247)).  
                    - L'onglet **Simulation** permet d'explorer les prédictions du modèle déployé. Il permet de prédire la demande par créneau d'une heure sur des données sur la période allant de juin à octobre 2023. Il est également possible de simuler l'effet d'une perturbation du réseau de transport (en sélectionnant un tronçon à enlever).
                    """,
                            style={
                                "width": "100%",
                                "height": 100,
                                "color": "white",
                                "background-color": "#272B30",
                                "border-top": "transparent",
                                "border-bottom": "transparent",
                                "border-left": "transparent",
                                "border-right": "transparent",
                            },
                        ),
                    ],
                    xs=12,
                    sm=12,
                    md=12,
                    lg=12,
                )
            ]
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dcc.Tabs(
                            parent_className="custom-tabs",
                            className="custom-tabs-container",
                            style={"width": "100%"},
                            children=[
                                dcc.Tab(
                                    label="Data",
                                    className="custom-tab",
                                    selected_className="custom-tab--selected",
                                    children=[
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H3(
                                                            "Description des données",
                                                            className="mb-2",
                                                            style={"textAlign": "left"},
                                                        ),
                                                        dash_table.DataTable(
                                                            df_describe.to_dict(
                                                                "records"
                                                            ),
                                                            columns,
                                                            style_header={
                                                                "backgroundColor": "#000000",
                                                                "color": "white",
                                                                "fontWeight": "bold",
                                                            },
                                                            style_data={
                                                                "backgroundColor": "#272B30",
                                                                "color": "white",
                                                            },
                                                            style_data_conditional=[
                                                                {
                                                                    "if": {
                                                                        "column_id": "Mesure"
                                                                    },
                                                                    "fontWeight": "bold",
                                                                }
                                                            ],
                                                        ),
                                                        dcc.Markdown(
                                                            """
                                                        Cette table présente le jeu de données sur lequel le modèle ST-MGCN a été entraîné.
                                                        - Ce jeu de données contient 431 carreaux de 500 mètres de côté, définis par leur latitude et leur longitude.
                                                        - Il contient également 7247 créneaux de 1 heure, répartis entre le 8 janvier et le 31 décembre 2022.
                                                        - La fréquentation correspond au nombre de passagers ayant validé leur titre de transport durant un créneau dans les stations d'un carreau donné.
                                                            """,
                                                            style={
                                                                "width": "100%",
                                                                "height": 100,
                                                                "color": "white",
                                                                "background-color": "#272B30",
                                                                "border-top": "transparent",
                                                                "border-bottom": "transparent",
                                                                "border-left": "transparent",
                                                                "border-right": "transparent",
                                                            },
                                                        ),
                                                    ],
                                                    xs=12,
                                                    sm=12,
                                                    md=12,
                                                    lg=12,
                                                )
                                            ]
                                        ),
                                        html.Br(),
                                        html.Hr(
                                            style={
                                                "borderWidth": "0.3vh",
                                                "width": "100%",
                                                "borderColor": "#53917E",
                                                "borderStyle": "solid",
                                            }
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Créneau horaire",
                                                            className="label",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="time",
                                                            value=datetime[0],
                                                            clearable=False,
                                                            options=datetime,
                                                        ),
                                                    ],
                                                    xs=3,
                                                    sm=3,
                                                    md=3,
                                                    lg=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Ligne", className="label"
                                                        ),
                                                        dcc.Dropdown(
                                                            id="ligne",
                                                            value="74",
                                                            clearable=False,
                                                            options=lignes,
                                                        ),
                                                    ],
                                                    xs=3,
                                                    sm=3,
                                                    md=3,
                                                    lg=3,
                                                ),
                                            ]
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H3(
                                                            "Distribution spatiale de la demande",
                                                            className="mb-2",
                                                            style={
                                                                "textAlign": "center"
                                                            },
                                                        ),
                                                        dcc.Graph(id="network"),
                                                        dcc.Markdown(
                                                            id="textarea-example",
                                                            children="""
                                                            - Chaque bulle représente une surface de 500*500 mètres de la métropole de Rennes.  
                                                            - La couleur de chaque bulle correspond au niveau de demande (nombre de passagers ayant validés leurs titre de transport) sur l'heure sélectionnée.
                                                            """,
                                                            style={
                                                                "width": "100%",
                                                                "height": 100,
                                                                "color": "white",
                                                                "background-color": "#272B30",
                                                                "border-top": "transparent",
                                                                "border-bottom": "transparent",
                                                                "border-left": "transparent",
                                                                "border-right": "transparent",
                                                            },
                                                        ),
                                                    ],
                                                    xs=6,
                                                    sm=6,
                                                    md=6,
                                                    lg=6,
                                                ),
                                                # dbc.Col([html.Img(id="network")]),
                                                dbc.Col(
                                                    [
                                                        html.H3(
                                                            "Distribution temporelle de la demande",
                                                            className="mb-2",
                                                            style={
                                                                "textAlign": "center"
                                                            },
                                                        ),
                                                        dcc.Graph(id="saturation"),
                                                        dcc.Markdown(
                                                            id="textarea-example2",
                                                            children="""
                                                            - La couleur de la courbe pour la demande totale (nombre de passagers ayant validé leur titre de transport) sur le réseau entier au cours de la journée sélectionnée.  
                                                            - La barre verticale hachurée correspond à la demande globale à l'heure sélectionnée.
                                                            """,
                                                            style={
                                                                "width": "100%",
                                                                "height": 100,
                                                                "color": "white",
                                                                "background-color": "#272B30",
                                                                "border-top": "transparent",
                                                                "border-bottom": "transparent",
                                                                "border-left": "transparent",
                                                                "border-right": "transparent",
                                                            },
                                                        ),
                                                    ],
                                                    xs=6,
                                                    sm=6,
                                                    md=6,
                                                    lg=6,
                                                ),
                                            ],
                                            className="gx-5",
                                        ),
                                    ],
                                ),
                                dcc.Tab(
                                    label="Simulation",
                                    className="custom-tab",
                                    selected_className="custom-tab--selected",
                                    children=[
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Créneau horaire",
                                                            className="label",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="time2",
                                                            value=datetimetest[0],
                                                            clearable=False,
                                                            options=datetimetest,
                                                        ),
                                                    ],
                                                    xs=3,
                                                    sm=3,
                                                    md=3,
                                                    lg=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Ligne", className="label"
                                                        ),
                                                        dcc.Dropdown(
                                                            id="ligne2",
                                                            value="74",
                                                            clearable=False,
                                                            options=lignes,
                                                        ),
                                                    ],
                                                    xs=3,
                                                    sm=3,
                                                    md=3,
                                                    lg=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Enlever une connexion entre ... et ...",
                                                            className="label",
                                                        ),
                                                        dcc.Dropdown(
                                                            id="topology",
                                                            clearable=True,
                                                            options=carreaux_disconnect,
                                                        ),
                                                        dcc.Dropdown(
                                                            id="topology2",
                                                            clearable=True,
                                                        ),
                                                    ],
                                                    xs=3,
                                                    sm=3,
                                                    md=3,
                                                    lg=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Button(
                                                            id="submit",
                                                            n_clicks=0,
                                                            children="Simuler",
                                                        )
                                                    ],
                                                    xs=3,
                                                    sm=3,
                                                    md=3,
                                                    lg=3,
                                                ),
                                            ],
                                            align="center",
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.H3(
                                                            "Distribution spatiale de la demande",
                                                            className="mb-2",
                                                            style={
                                                                "textAlign": "center"
                                                            },
                                                        ),
                                                        dcc.Graph(id="network2"),
                                                        dcc.Store(id="values-to-plot"),
                                                        dcc.Markdown(
                                                            id="textarea-example-pred1",
                                                            children="""
                                                            - Chaque bulle représente une surface de 500*500 mètres de la métropole de la ville de Rennes.
                                                            - La couleur de chaque bulle correspond au niveau de demande (nombre de passagers ayant validé leur titre de transport) qui est **prédite par le modèle** sur l'heure sélectionnée.
                                                            """,
                                                            style={
                                                                "width": "100%",
                                                                "height": 100,
                                                                "color": "white",
                                                                "background-color": "#272B30",
                                                                "border-top": "transparent",
                                                                "border-bottom": "transparent",
                                                                "border-left": "transparent",
                                                                "border-right": "transparent",
                                                            },
                                                        ),
                                                    ],
                                                    xs=6,
                                                    sm=6,
                                                    md=6,
                                                    lg=6,
                                                ),
                                                # dbc.Col([html.Img(id="network")]),
                                                dbc.Col(
                                                    [
                                                        html.H3(
                                                            "Distribution temporelle de la demande",
                                                            className="mb-2",
                                                            style={
                                                                "textAlign": "center"
                                                            },
                                                        ),
                                                        dcc.Graph(id="saturation2"),
                                                        dcc.Markdown(
                                                            id="textarea-example-pred2",
                                                            children="""
                                                            - La couleur de la courbe de la demande totale (nombre de passagers ayant validé leur titre de transport) sur tout le réseau au jour sélectionné.
                                                            - La barre verticale hachurée représente la demande à l'heure sélectionnée.
                                                            - Le point jaune représente la demande totale **predite par le modèle** à l'heure sélectionnée.
                                                            """,
                                                            style={
                                                                "width": "100%",
                                                                "height": 100,
                                                                "color": "white",
                                                                "background-color": "#272B30",
                                                                "border-top": "transparent",
                                                                "border-bottom": "transparent",
                                                                "border-left": "transparent",
                                                                "border-right": "transparent",
                                                            },
                                                        ),
                                                    ],
                                                    xs=6,
                                                    sm=6,
                                                    md=6,
                                                    lg=6,
                                                ),
                                            ],
                                            className="gx-5",
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                    xs=12,
                    sm=12,
                    md=12,
                    lg=12,
                )
            ]
        ),
    ]
)


#!DEFINE CALLBACKS : interactive elements like figs
# Create interactivity between dropdown component and graph
@app.callback(
    Output("network", "figure"),
    Output("saturation", "figure"),
    Input("time", "value"),
    Input("ligne", "value"),
)
def plot_data(t, ligne):
    network = du.plot_network_go(
        carreaux, network_demand, datetime.index(t), trace_lignes, str(ligne)
    )
    jour = t.split(" ")[0]
    ts = [i for i in range(len(jours)) if jours[i] == jour]
    saturation = du.plot_saturation_go(
        network_demand, hours[datetime.index(t)], ts, [hours[i] for i in ts]
    )

    return network, saturation


@app.callback(
    Output("topology2", "options"),
    Output("topology2", "value"),
    Input("topology", "value"),
)
def topology_input(x):
    if x is not None:
        y = (
            disc.loc[disc["c1"] == carreaux_list.index(x), "c2"].tolist()
            + disc.loc[disc["c2"] == carreaux_list.index(x), "c1"].tolist()
        )
        y = list(set(y))
        y.sort()
        return [carreaux_list[i] for i in y if i != carreaux_list.index(x)], None
    else:
        return [], None


#! CALLBACK SIMULATION
# Create input route, post request, fetch prediction, and plot
@app.callback(
    Output("network2", "figure"),
    Output("saturation2", "figure"),
    Output("submit", "n_clicks"),
    Output("values-to-plot", "data"),
    Input("submit", "n_clicks"),
    State("time2", "value"),
    State("ligne2", "value"),
    State("topology", "value"),
    State("topology2", "value"),
)
def predict_and_plot(n_clicks, t, ligne, topo1, topo2):
    n_clicks += 1
    if n_clicks > 1:
        res = requests.post(
            url="https://46b50yi147.execute-api.eu-west-3.amazonaws.com/Prod/star_predict/",
            data=json.dumps(
                {
                    "obs_seq": test[datetimetest.index(t), :, :, :].tolist(),
                    "topo1": (
                        carreaux_list.index(topo1) if topo1 in carreaux_list else None
                    ),
                    "topo2": (
                        carreaux_list.index(topo2) if topo2 in carreaux_list else None
                    ),
                }
            ),
        )
        prediction = np.squeeze(np.array(json.loads(res.content)["prediction"]))
        network = du.plot_network_pred(
            carreaux,
            network_demand,
            np.where(prediction > 0, prediction, 0),
            trace_lignes,
            str(ligne),
        )
        jour = t.split(" ")[0]
        ts = [i for i in range(len(jours_test)) if jours_test[i] == jour]
        saturation = du.plot_saturation_test(
            network_demand_test,
            hours_test[datetimetest.index(t)],
            ts,
            [hours_test[i] for i in ts],
            np.where(prediction > 0, prediction, 0).sum(0),
        )
        return (
            network,
            saturation,
            n_clicks,
            json.dumps({"demand": np.where(prediction > 0, prediction, 0).tolist()}),
        )
    else:
        network = du.plot_network_pred(
            carreaux,
            network_demand,
            network_demand_test[datetimetest.index(t), :],
            trace_lignes,
            str(ligne),
        )
        jour = t.split(" ")[0]
        ts = [i for i in range(len(jours_test)) if jours_test[i] == jour]
        saturation = du.plot_saturation_test(
            network_demand_test,
            hours_test[datetimetest.index(t)],
            ts,
            [hours_test[i] for i in ts],
        )
        return (
            network,
            saturation,
            n_clicks,
            json.dumps(
                {"demand": network_demand_test[datetimetest.index(t), :].tolist()}
            ),
        )


#! CALLBACK SELECTOR SIMULATION
# Display selected points on the map
@app.callback(
    Output("network2", "figure"),
    Input("topology", "value"),
    Input("topology2", "value"),
    State("values-to-plot", "data"),
    State("ligne2", "value"),
)
def update_with_topology(topo1, topo2, data, ligne):
    demand = json.loads(data)["demand"]
    network = du.plot_network_pred(
        carreaux, network_demand, demand, trace_lignes, str(ligne), topo1, topo2
    )
    return network


#!EXECUTE APP
if __name__ == "__main__":
    app.run_server(debug=False, port=8050, host="0.0.0.0")
