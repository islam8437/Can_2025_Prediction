# models/meta_modele.py
import numpy as np
from sklearn.linear_model import LogisticRegression

class MetaModeleFusion:
    """Combine les prédictions des 3 experts"""
    
    def __init__(self):
        self.meta_model = LogisticRegression(random_state=42)
        self.poids_fixes = None
    
    def entrainer(self, predictions_experts, resultats_reels):
        """
        predictions_experts: DataFrame avec colonnes ['proba_historique', 'proba_confrontations', 'proba_forme']
        resultats_reels: array avec résultats réels (1 si equipe_A gagne, 0 sinon)
        """
        X_meta = predictions_experts
        y_meta = resultats_reels
        
        if len(X_meta) > 0:
            self.meta_model.fit(X_meta, y_meta)
            self.poids_fixes = self.meta_model.coef_[0]
        
        return self
    
    def predire_avec_poids_fixes(self, proba_historique, proba_confrontations, proba_forme):
        """Utilise des poids fixes pour la fusion"""
        if self.poids_fixes is not None and len(self.poids_fixes) == 3:
            poids = self.poids_fixes / self.poids_fixes.sum()
            proba_finale = (
                poids[0] * proba_historique +
                poids[1] * proba_confrontations +
                poids[2] * proba_forme
            )
        else:
            # Poids par défaut basés sur l'expertise footballistique
            poids = [0.25, 0.35, 0.40]
            proba_finale = (
                poids[0] * proba_historique +
                poids[1] * proba_confrontations +
                poids[2] * proba_forme
            )
        
        return proba_finale
    
    def predire_avec_meta_modele(self, proba_historique, proba_confrontations, proba_forme):
        """Utilise le méta-modèle pour la fusion"""
        if hasattr(self.meta_model, 'predict_proba'):
            X = [[proba_historique, proba_confrontations, proba_forme]]
            proba_finale = self.meta_model.predict_proba(X)[0, 1]
        else:
            proba_finale = self.predire_avec_poids_fixes(proba_historique, proba_confrontations, proba_forme)
        
        return proba_finale