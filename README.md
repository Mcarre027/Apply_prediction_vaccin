# 📊 Dashboard Vaccination COVID-19

Bienvenue dans ce projet de **Data Science & Visualisation interactive** !  
Cette application **Dash + Plotly** vous permet d’explorer la couverture vaccinale mondiale en fonction de **facteurs socio-économiques**, et de **prédire la couverture vaccinale attendue** grâce à un modèle de machine learning.

---

## 🌍 Aperçu des fonctionnalités

### 🏠 Page Accueil
- Introduction au projet et à ses objectifs.

### 📈 Page Dataviz
- Visualisation interactive des taux de vaccination par continent.
- Scatter plot entre PIB et couverture vaccinale.
- Filtrage dynamique.

### 🦢 Page Exploration
- Analyse du lien entre chaque variable explicative et la couverture vaccinale.
- Courbe de tendance via régression linéaire.

### 🤖 Page Machine Learning
- Modèle de prédiction basé sur un **Random Forest Regressor**.
- Visualisation de l’importance des variables.
- Formulaire interactif de prédiction avec jauge.
- Historique des prédictions comparables.
- Export de l’historique au format CSV.

### 🌐 Page Carte du Monde
- Carte choropleth interactive avec Plotly.
- Affichage des taux réels, prédits ou erreur absolue.
- Filtrage dynamique de l’erreur.

---

## ⚙️ Technologies utilisées

- **Python**
- **Dash** & **Plotly** pour l’application web et les graphiques
- **pandas** pour la manipulation de données
- **scikit-learn** pour le modèle de prédiction
- **Bootstrap (via Dash)** pour le design responsive

---

## 📁 Structure du projet

```
.
├── Prediction_couverture_vaccinale.py   # Fichier principal Dash
├── requirements.txt                     # Dépendances (facultatif)
├── README.md                            # Ce fichier
```

---

## 🚀 Lancer l'application en local

1. Clonez le repo :
```bash
git clone https://github.com/ton-utilisateur/dashboard-vaccination.git
cd dashboard-vaccination
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Lancez l’application :
```bash
python Prediction_couverture_vaccinale.py
```

4. Ouvrez votre navigateur sur [http://127.0.0.1:8050](http://127.0.0.1:8050)

---

## 📦 Données utilisées

Données issues de [Our World in Data – COVID-19 Dataset](https://github.com/owid/covid-19-data)

---

## ✅ Objectifs pédagogiques

- Feature engineering & modélisation
- Comparaison de modèles (Random Forest, Decision Tree…)
- Démonstration de visualisation interactive
- Interface utilisateur de type “simulateur”

---

## 📬 Auteur

Projet réalisé par **[Ton Nom]**, dans le cadre de ma formation de Data Analyst.  
[LinkedIn](https://www.linkedin.com/in/matthieu-carre-19a3092b8/) • [Email](carrehomebusiness@gmail.com)

---