# utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Dictionnaire de correction des noms d'équipes
EQUIPES_CORRECTIONS = {
    'Côte d’Ivoire': 'Ivory Coast',
    'DR Congo': 'Congo DR',
    'Congo': 'Congo DR',
    'Equatorial Guinea': 'Equatorial Guinea',
    'Cabo Verde': 'Cape Verde',
    'São Tomé and Príncipe': 'Sao Tome and Principe'
}

# Mapping des stages reached en scores numériques
STAGE_SCORES = {
    'Group': 0,
    'R16': 1,
    'QF': 2,
    'SF': 3,
    'Final': 4,
    'Champion': 5
}

def charger_donnees():
    """Charge toutes les données nécessaires"""
    print("📥 Chargement des données...")
    palmares = pd.read_csv('palmares.csv')
    historique = pd.read_csv('historique_rencontre.csv')
    forme = pd.read_csv('forme_equipe.csv')
    statistiques = pd.read_csv('statistique_equipe.csv')
    return palmares, historique, forme, statistiques

def corriger_noms_equipes(df, colonne_equipe):
    """Corrige les noms d'équipes pour uniformité"""
    if colonne_equipe in df.columns:
        df[colonne_equipe] = df[colonne_equipe].replace(EQUIPES_CORRECTIONS)
    return df