# main.py
from systeme_prediction import SystemePredictionCAN
from utils import charger_donnees

def main():
    print("=" * 60)
    print("⚽ SYSTÈME DE PRÉDICTION CAN - MULTI-MODÈLES ⚽")
    print("=" * 60)
    
    # Charger les données
    palmares, historique, forme, statistiques = charger_donnees()
    
    # Initialiser le système
    systeme = SystemePredictionCAN()
    
    # Entraîner tous les modèles
    systeme.entrainer_complet(palmares, historique, forme, statistiques)
    
    print("\n" + "=" * 60)
    print("🧪 TESTS DE PRÉDICTION")
    print("=" * 60)
    
    # Exemples de prédictions
    matchs_test = [
        ("Morocco", "Mali"),
        ("Egypt", "South Africa"),
        ("Nigeria", "Senegal"),
        ("Algeria", "Ivory Coast")
    ]
    
    for equipe_A, equipe_B in matchs_test:
        print(f"\n🔮 {equipe_A} vs {equipe_B}:")
        prediction = systeme.predire_match(equipe_A, equipe_B)
        
        print(f"   Probabilité finale: {prediction['proba_finale']:.2%}")
        print(f"   Détails:")
        print(f"     - Historique CAN: {prediction['details']['historique_can']:.2%}")
        print(f"     - Confrontations: {prediction['details']['confrontations_directes']:.2%}")
        print(f"     - Forme actuelle: {prediction['details']['forme_actuelle']:.2%}")
    
    print("\n" + "=" * 60)
    print("🏆 SIMULATION DE GROUPES CAN 2025")
    print("=" * 60)
    
    # Définir les groupes (exemple)
    groupes_can2025 = {
        'Groupe A': ['Morocco', 'Mali', 'Zambia', 'Comoros'],
        'Groupe B': ['Egypt', 'South Africa', 'Angola', 'Zimbabwe'],
        'Groupe C': ['Nigeria', 'Tunisia', 'Uganda', 'Tanzania'],
        'Groupe D': ['Senegal', 'DR Congo', 'Benin', 'Botswana'],
        'Groupe E': ['Algeria', 'Burkina Faso', 'Equatorial Guinea', 'Sudan'],
        'Groupe F': ['Ivory Coast', 'Cameroon', 'Gabon', 'Mozambique']
    }
    
    # Simuler chaque groupe
    for nom_groupe, equipes in groupes_can2025.items():
        print(f"\n{nom_groupe}:")
        systeme.simuler_groupe(equipes)

if __name__ == "__main__":
    main()