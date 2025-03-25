import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os


# Chargement des données préparées
df_grouped = pd.read_csv("data_light.csv")



# Modèle de prédiction
X = df_grouped[['median_age', 'life_expectancy', 'human_development_index', 'gdp_per_capita', 'cardiovasc_death_rate']]
y = df_grouped['people_fully_vaccinated_per_hundred']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

df_grouped['prediction'] = model.predict(X)
df_grouped['absolute_error'] = abs(df_grouped['prediction'] - df_grouped['people_fully_vaccinated_per_hundred'])

# Initialisation de l'app Dash
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Vaccination Analytics Dashboard"

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Accueil", href="/")),
            dbc.NavItem(dbc.NavLink("Dataviz", href="/dataviz")),
            dbc.NavItem(dbc.NavLink("Exploration", href="/exploration")),
            dbc.NavItem(dbc.NavLink("Machine Learning", href="/ml")),
            dbc.NavItem(dbc.NavLink("Carte du Monde", href="/map")),
        ],
        brand="Dashboard Vaccination",
        color="dark",
        dark=True,
    ),
    html.Div(id='page-content')
])

# Layouts précis des pages
accueil_layout = html.Div([
    html.H1("Bienvenue sur le Dashboard Vaccination COVID-19", className="text-center"),
    html.P("Ce tableau de bord interactif explore les facteurs socio-économiques expliquant la couverture vaccinale mondiale.", className="text-center"),
    html.Img(src="assets\globe-6002422_1280.jpg", style={"width": "60%", "display": "block", "margin": "auto"})
])

viz_layout = html.Div([
    html.H3("Visualisation de la couverture vaccinale"),
    dcc.Dropdown(id="continent_filter", options=[{"label": c, "value": c} for c in df_grouped['continent'].unique()], multi=True),
    dcc.Graph(id="scatter_vaccination")
])

explore_layout = html.Div([
    html.H3("Exploration des variables"),
    dcc.Dropdown(id="x_variable", options=X.columns, value='median_age'),
    dcc.Graph(id="exploration_plot")
])

ml_layout = html.Div([
    html.H3("Prédiction manuelle de la couverture vaccinale", className="text-center"),
    html.P("Ce module vous permet de tester le modèle en entrant des variables socio-économiques.", className="text-center"),
    dbc.Row([
        dbc.Col([html.Label("Âge médian"), dcc.Input(id='input_median_age', type='number', value=35, className='form-control')]),
        dbc.Col([html.Label("Espérance de vie"), dcc.Input(id='input_life_expectancy', type='number', value=75, className='form-control')])
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([html.Label("Indice HDI"), dcc.Input(id='input_hdi', type='number', value=0.7, step=0.01, className='form-control')]),
        dbc.Col([html.Label("PIB par habitant"), dcc.Input(id='input_gdp', type='number', value=10000, className='form-control')])
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([html.Label("Mortalité cardio."), dcc.Input(id='input_cardio', type='number', value=200, className='form-control')])
    ], className="mb-3"),
    html.Div([
        html.Button("Prédire", id="predict_button", className="btn btn-primary"),
        html.Button("Sauvegarder", id="save_prediction", className="btn btn-success ms-2"),
        html.Button("Effacer historique", id="clear_history", className="btn btn-danger ms-2"),
        html.Button("Exporter CSV", id="export_button", className="btn btn-secondary ms-2"),
        dcc.Download(id="download_csv")
    ], className="text-center"),
    html.Div(id="prediction_output", className="text-center mt-4"),
    dcc.Store(id="prediction_history", data=[]),
    dcc.Graph(id="prediction_table")
])

# Callbacks gestion des pages (ajout de la carte du monde)
@callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/dataviz':
        return viz_layout
    elif pathname == '/exploration':
        return explore_layout
    elif pathname == '/ml':
        return ml_layout
    elif pathname == '/map':
        return html.Div([
            html.H3("Carte mondiale des prédictions", className="text-center mt-3"),
            dcc.RadioItems(
                id='map_choice',
                options=[
                    {'label': 'Prédiction', 'value': 'prediction'},
                    {'label': 'Réel', 'value': 'people_fully_vaccinated_per_hundred'},
                    {'label': 'Erreur Absolue', 'value': 'absolute_error'}
                ],
                value='prediction',
                inline=True,
                className='text-center'
            ),
            dcc.Graph(id='map_graph', style={'height': '80vh', 'width': '90%', 'margin': 'auto'}),
        ])
    return accueil_layout


   
@callback(
    Output('map_graph', 'figure'),
    Input('map_choice', 'value')
)
def update_map(map_choice):
    fig = px.choropleth(
        df_grouped,
        locations='iso_code',
        color=map_choice,
        hover_name='location',
        color_continuous_scale='Viridis',
        title=f"Carte Mondiale : {map_choice.replace('_', ' ').title()}",
        projection='natural earth'
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig

@callback(Output("scatter_vaccination", "figure"), Input("continent_filter", "value"))
def update_scatter(continent):
    dff = df_grouped.copy()
    if continent:
        dff = dff[dff['continent'].isin(continent)]
    fig = px.scatter(
        dff, x="gdp_per_capita", y="people_fully_vaccinated_per_hundred",
        color="continent", size="population", hover_name="location",
        title="Lien entre PIB/hab. et couverture vaccinale"
    )
    return fig
@callback(Output("exploration_plot", "figure"), Input("x_variable", "value"))
def update_explore(var):
    fig = px.scatter(
        df_grouped, x=var, y="people_fully_vaccinated_per_hundred",
        color="continent", hover_name="location", trendline="ols",
        title=f"Relation entre {var} et couverture vaccinale"
    )
    return fig




if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, host="0.0.0.0", port=port)

 

