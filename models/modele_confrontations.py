# models/modele_confrontations.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

class ModeleConfrontationsDirectes:
    """Expert des matchs passés entre équipes"""
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.historique_complet = None
        
    def calculer_stats_confrontations(self, historique_df):
        """Calcule les statistiques des confrontations pour chaque paire d'équipes"""
        
        pairs_stats = []
        
        # Liste de toutes les équipes
        toutes_equipes = list(set(historique_df['home_team'].unique()) | set(historique_df['away_team'].unique()))
        
        for equipe_A in toutes_equipes:
            for equipe_B in toutes_equipes:
                if equipe_A != equipe_B:
                    # Filtrer les matchs entre ces deux équipes
                    matchs = historique_df[
                        ((historique_df['home_team'] == equipe_A) & (historique_df['away_team'] == equipe_B)) |
                        ((historique_df['home_team'] == equipe_B) & (historique_df['away_team'] == equipe_A))
                    ]
                    
                    if len(matchs) > 0:
                        stats = {
                            'equipe_A': equipe_A,
                            'equipe_B': equipe_B,
                            'nbr_matchs': len(matchs),
                            'victoires_A': 0,
                            'victoires_B': 0,
                            'nuls': 0,
                            'dernier_resultat': 'Draw',
                            'domination_A': 0.5
                        }
                        
                        for _, match in matchs.iterrows():
                            if match['home_team'] == equipe_A:
                                if match['vainqueur'] == equipe_A:
                                    stats['victoires_A'] += 1
                                elif match['vainqueur'] == equipe_B:
                                    stats['victoires_B'] += 1
                                else:
                                    stats['nuls'] += 1
                            else:  # equipe_B est home
                                if match['vainqueur'] == equipe_B:
                                    stats['victoires_A'] += 1
                                elif match['vainqueur'] == equipe_A:
                                    stats['victoires_B'] += 1
                                else:
                                    stats['nuls'] += 1
                        
                        # Calculer la domination
                        if stats['nbr_matchs'] > 0:
                            stats['domination_A'] = (stats['victoires_A'] + 0.5 * stats['nuls']) / stats['nbr_matchs']
                        
                        # Dernier résultat
                        dernier_match = matchs.sort_values('date').iloc[-1]
                        if dernier_match['vainqueur'] == equipe_A:
                            stats['dernier_resultat'] = 'Win_A'
                        elif dernier_match['vainqueur'] == equipe_B:
                            stats['dernier_resultat'] = 'Win_B'
                        else:
                            stats['dernier_resultat'] = 'Draw'
                        
                        pairs_stats.append(stats)
        
        return pd.DataFrame(pairs_stats)
    
    def entrainer(self, historique_df):
        """Entraîne le modèle sur les confrontations historiques"""
        # Préparer le dataset d'entraînement
        X_train = []
        y_train = []
        
        for _, match in historique_df.iterrows():
            equipe_A = match['home_team']
            equipe_B = match['away_team']
            
            # Calculer les features pour ce match
            matchs_precedents = historique_df[
                ((historique_df['home_team'] == equipe_A) & (historique_df['away_team'] == equipe_B)) |
                ((historique_df['home_team'] == equipe_B) & (historique_df['away_team'] == equipe_A))
            ]
            
            matchs_precedents = matchs_precedents[matchs_precedents['date'] < match['date']]
            
            if len(matchs_precedents) > 0:
                victoires_A = 0
                victoires_B = 0
                nuls = 0
                
                for _, prev_match in matchs_precedents.iterrows():
                    if prev_match['home_team'] == equipe_A:
                        if prev_match['vainqueur'] == equipe_A:
                            victoires_A += 1
                        elif prev_match['vainqueur'] == equipe_B:
                            victoires_B += 1
                        else:
                            nuls += 1
                    else:
                        if prev_match['vainqueur'] == equipe_B:
                            victoires_A += 1
                        elif prev_match['vainqueur'] == equipe_A:
                            victoires_B += 1
                        else:
                            nuls += 1
                
                total = victoires_A + victoires_B + nuls
                domination_A = (victoires_A + 0.5 * nuls) / total if total > 0 else 0.5
                
                # Features: domination historique + facteur terrain
                features = [
                    domination_A,
                    1 if match['neutral'] == 'True' else 0,
                    1  # Intercept
                ]
                
                X_train.append(features)
                
                # Target: 1 si A gagne, 0 si B gagne
                if match['vainqueur'] == equipe_A:
                    y_train.append(1)
                elif match['vainqueur'] == equipe_B:
                    y_train.append(0)
                else:
                    y_train.append(0.5)
        
        # Convertir en arrays et filtrer les nuls
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        mask = y_train != 0.5
        X_train = X_train[mask]
        y_train = y_train[mask].astype(int)
        
        if len(X_train) > 0:
            self.model.fit(X_train, y_train)
        
        # Stocker les stats complètes
        self.historique_complet = self.calculer_stats_confrontations(historique_df)
        
        return self
    
    def predire(self, equipe_A, equipe_B, neutral=True):
        """Prédit la probabilité que A batte B basé sur les confrontations directes"""
        if self.historique_complet is None:
            return 0.5
        
        # Chercher les stats de cette paire
        stats = self.historique_complet[
            ((self.historique_complet['equipe_A'] == equipe_A) & (self.historique_complet['equipe_B'] == equipe_B)) |
            ((self.historique_complet['equipe_A'] == equipe_B) & (self.historique_complet['equipe_B'] == equipe_A))
        ]
        
        if len(stats) == 0:
            return 0.5  # Pas d'historique
        
        # Prendre la bonne orientation
        if stats.iloc[0]['equipe_A'] == equipe_A:
            domination = stats.iloc[0]['domination_A']
        else:
            domination = 1 - stats.iloc[0]['domination_A']
        
        # Features pour la prédiction
        features = [
            domination,
            1 if neutral else 0,
            1
        ]
        
        proba = self.model.predict_proba([features])[0, 1]
        
        return proba