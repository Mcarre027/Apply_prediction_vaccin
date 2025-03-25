import pandas as pd

# Charger uniquement les colonnes utiles
url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
cols = [
    'location', 'continent', 'people_fully_vaccinated_per_hundred',
    'median_age', 'life_expectancy', 'human_development_index',
    'gdp_per_capita', 'cardiovasc_death_rate', 'iso_code', 'population'
]
df = pd.read_csv(url, usecols=cols)

# Nettoyage
df = df.dropna()
df = df[~df['location'].str.startswith("OWID_")]
df = df[df['people_fully_vaccinated_per_hundred'] > 0]

# Moyenne par pays, réduction à 70 pays max
df_grouped = df.groupby(['location', 'continent', 'iso_code'], as_index=False).mean(numeric_only=True)
df_light = df_grouped.sample(70, random_state=42)

# Sauvegarde
df_light.to_csv("data_light.csv", index=False)
print("✅ Fichier 'data_light.csv' généré.")
