import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
from sklearn.ensemble import RandomForestRegressor
import os
from sklearn.model_selection import train_test_split

# Chargement des données préparées
url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
df = pd.read_csv(url)
df = df[~df['location'].str.startswith("OWID_")]

features = [
    'location', 'continent', 'people_fully_vaccinated_per_hundred',
    'median_age', 'life_expectancy', 'human_development_index',
    'gdp_per_capita', 'cardiovasc_death_rate', 'iso_code', 'population'
]

df = df[features].dropna()
df_grouped = df.groupby(['location', 'continent', 'iso_code'], as_index=False).mean(numeric_only=True)

# Modèle de prédiction
X = df_grouped[['median_age', 'life_expectancy', 'human_development_index', 'gdp_per_capita', 'cardiovasc_death_rate']]
y = df_grouped['people_fully_vaccinated_per_hundred']
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)
df_grouped['prediction'] = model.predict(X)
df_grouped['absolute_error'] = abs(df_grouped['prediction'] - y)

saved_predictions = []

# Initialisation Dash
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    show_undo_redo=False
)

app.title = "Vaccination Analytics Dashboard"

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    dcc.Download(id="download_csv"),
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

# Layout Accueil
accueil_layout = html.Div([
    html.H1("Bienvenue sur le Dashboard Vaccination COVID-19", className="text-center mt-4"),
    html.P("Ce tableau de bord interactif explore les facteurs socio-économiques expliquant la couverture vaccinale mondiale.", className="text-center"),
    html.Img(src="assets\1.png", style={"width": "60%", "display": "block", "margin": "auto"})
])

# Layout Machine Learning
ml_layout = html.Div([
    html.H3("Prédiction manuelle de la couverture vaccinale", className="text-center mt-4"),
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
        html.Button("Prédire", id="predict_button", className="btn btn-primary me-2"),
        html.Button("Effacer", id="clear_button", className="btn btn-warning me-2"),
        html.Button("Exporter CSV", id="export_button", className="btn btn-info")
    ], className="text-center mt-2"),
    html.Div(id="prediction_output", className="text-center mt-4"),
    dcc.Graph(id="prediction_graph")
])

# Layout Dataviz
viz_layout = html.Div([
    html.H3("Visualisation de la couverture vaccinale", className="text-center mt-4"),
    dcc.Dropdown(id="continent_filter", options=[{"label": c, "value": c} for c in df_grouped['continent'].unique()], multi=True),
    dcc.Graph(id="scatter_vaccination")
])

# Layout Exploration
explore_layout = html.Div([
    html.H3("Exploration des variables", className="text-center mt-4"),
    dcc.Dropdown(id="x_variable", options=[{"label": col, "value": col} for col in X.columns], value='median_age'),
    dcc.Graph(id="exploration_plot")
])

# Layout Carte du Monde
map_layout = html.Div([
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

# Routing
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/ml':
        return ml_layout
    elif pathname == '/dataviz':
        return viz_layout
    elif pathname == '/exploration':
        return explore_layout
    elif pathname == '/map':
        return map_layout
    return accueil_layout

# Callback Machine Learning
@app.callback(
    [
        Output("prediction_output", "children"),
        Output("prediction_graph", "figure"),
        Output("input_median_age", "value"),
        Output("input_life_expectancy", "value"),
        Output("input_hdi", "value"),
        Output("input_gdp", "value"),
        Output("input_cardio", "value"),
        Output("download_csv", "data")
    ],
    [
        Input("predict_button", "n_clicks"),
        Input("clear_button", "n_clicks"),
        Input("export_button", "n_clicks")
    ],
    [
        State("input_median_age", "value"),
        State("input_life_expectancy", "value"),
        State("input_hdi", "value"),
        State("input_gdp", "value"),
        State("input_cardio", "value")
    ],
    prevent_initial_call=True
)
def handle_actions(predict_clicks, clear_clicks, export_clicks, median_age, life_exp, hdi, gdp, cardio):
    triggered_id = ctx.triggered_id
    fig = go.Figure()

    if triggered_id == "predict_button":
        input_data = pd.DataFrame([[median_age, life_exp, hdi, gdp, cardio]],
                                  columns=['median_age', 'life_expectancy', 'human_development_index', 'gdp_per_capita', 'cardiovasc_death_rate'])
        prediction = model.predict(input_data)[0]
        saved_predictions.append({
            "median_age": median_age,
            "life_expectancy": life_exp,
            "human_development_index": hdi,
            "gdp_per_capita": gdp,
            "cardiovasc_death_rate": cardio,
            "prediction": prediction
        })
        df_saved = pd.DataFrame(saved_predictions)
        fig = px.histogram(df_saved, x="prediction", nbins=20, title="Historique des prédictions")
        return html.Div([html.H4(f"Prédiction : {prediction:.2f}%")]), fig, no_update, no_update, no_update, no_update, no_update, None

    elif triggered_id == "clear_button":
        return "Champs réinitialisés.", go.Figure(), 35, 75, 0.7, 10000, 200, None

    elif triggered_id == "export_button":
        if saved_predictions:
            df_saved = pd.DataFrame(saved_predictions)
            return "Export en cours...", no_update, no_update, no_update, no_update, no_update, no_update, dcc.send_data_frame(df_saved.to_csv, "predictions.csv")
        else:
            return "Aucune donnée à exporter.", go.Figure(), no_update, no_update, no_update, no_update, no_update, None

    return no_update, no_update, no_update, no_update, no_update, no_update, no_update, None

# Callback Dataviz
@app.callback(Output("scatter_vaccination", "figure"), Input("continent_filter", "value"))
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

# Callback Exploration
@app.callback(Output("exploration_plot", "figure"), Input("x_variable", "value"))
def update_explore(var):
    fig = px.scatter(
        df_grouped, x=var, y="people_fully_vaccinated_per_hundred",
        color="continent", hover_name="location", trendline="ols",
        title=f"Relation entre {var} et couverture vaccinale"
    )
    return fig

# Callback Carte du Monde
@app.callback(Output('map_graph', 'figure'), Input('map_choice', 'value'))
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False  # ← important pour Railway
    )


