# systeme_prediction.py
import numpy as np
from models.modele_historique import ModeleHistoriqueCAN
from models.modele_confrontations import ModeleConfrontationsDirectes
from models.modele_forme import ModeleFormeActuelle
from models.meta_modele import MetaModeleFusion
from utils import charger_donnees, corriger_noms_equipes

class SystemePredictionCAN:
    """Système complet de prédiction multi-modèles"""
    
    def __init__(self):
        self.modele_historique = ModeleHistoriqueCAN()
        self.modele_confrontations = ModeleConfrontationsDirectes()
        self.modele_forme = ModeleFormeActuelle()
        self.meta_modele = MetaModeleFusion()
        
    def entrainer_complet(self, palmares, historique, forme, statistiques):
        """Entraîne tous les modèles"""
        print("🔄 Entraînement des modèles...")
        
        # Appliquer les corrections de noms
        palmares = corriger_noms_equipes(palmares, 'team')
        historique = corriger_noms_equipes(historique, 'home_team')
        historique = corriger_noms_equipes(historique, 'away_team')
        forme = corriger_noms_equipes(forme, 'team')
        
        # Entraîner chaque modèle
        self.modele_historique.entrainer(palmares, historique)
        self.modele_confrontations.entrainer(historique)
        self.modele_forme.entrainer(forme, statistiques, historique)
        
        print("✅ Modèles entraînés avec succès!")
        return self
    
    def predire_match(self, equipe_A, equipe_B, neutral=True):
        """Prédit la probabilité que A batte B"""
        
        # Obtenir les prédictions des 3 experts
        proba_historique = self.modele_historique.predire(equipe_A, equipe_B)
        proba_confrontations = self.modele_confrontations.predire(equipe_A, equipe_B, neutral)
        proba_forme = self.modele_forme.predire(equipe_A, equipe_B, neutral)
        
        # Fusionner avec le méta-modèle
        proba_finale = self.meta_modele.predire_avec_poids_fixes(
            proba_historique, proba_confrontations, proba_forme
        )
        
        return {
            'proba_finale': proba_finale,
            'details': {
                'historique_can': proba_historique,
                'confrontations_directes': proba_confrontations,
                'forme_actuelle': proba_forme
            }
        }
    
    def simuler_groupe(self, groupe_equipes):
        """Simule tous les matchs d'un groupe"""
        print(f"🎯 Simulation du groupe: {groupe_equipes}")
        
        points = {equipe: 0 for equipe in groupe_equipes}
        
        # Simuler tous les matchs du groupe
        for i, equipe_A in enumerate(groupe_equipes):
            for equipe_B in groupe_equipes[i+1:]:
                # Prédiction pour le match A vs B
                prediction = self.predire_match(equipe_A, equipe_B, neutral=True)
                proba_A = prediction['proba_finale']
                
                # Simuler le résultat
                if np.random.random() < proba_A:
                    points[equipe_A] += 3
                elif np.random.random() < 0.3:
                    points[equipe_A] += 1
                    points[equipe_B] += 1
                else:
                    points[equipe_B] += 3
        
        # Classement
        classement = sorted(points.items(), key=lambda x: x[1], reverse=True)
        
        print("📊 Classement simulé:")
        for equipe, pts in classement:
            print(f"  {equipe}: {pts} points")
        
        return classement