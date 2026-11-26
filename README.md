# 🫁 RadiologyAI

Application d'aide au diagnostic de la **pneumonie sur radiographie thoracique**, avec **validation médicale** intégrée.

**VGG16 (transfer learning) · Grad-CAM · Streamlit · Rapport PDF**

## Fonctionnalités
- Analyse d'une radiographie : probabilité de pneumonie + carte Grad-CAM des zones suspectes
- Validation par le médecin (valider / ne pas valider, diagnostic retenu, commentaire)
- Fiche résultat patient (VALIDÉ / NON VALIDÉ / EN ATTENTE) et rapport PDF avec recommandations
- Galerie normal vs pneumonie avec score de probabilité
- Performances : ROC, précision-rappel, F1, loss/accuracy, matrice de confusion

## Installation
```bash
git clone https://github.com/<votre-compte>/RadiologyAI.git
cd RadiologyAI
pip install -r requirements.txt
streamlit run app.py
```

## Structure
```
app.py                interface (6 pages, navigation horizontale)
core.py               modèle, Grad-CAM, recommandations, dossiers, PDF
assets.py             bannière poumon, radiographies de démonstration
export_artifacts.py   à lancer depuis le notebook pour exporter métriques et courbes
artifacts/model.keras modèle VGG16 entraîné
data/                 dossiers patients (local, non versionné)
```

## Métriques réelles
Dans le notebook, après l'évaluation :
```python
from export_artifacts import export
export(model, history, x_test, X_test_vgg, y_test)
```
Puis copiez les fichiers générés dans `artifacts/`. Sans eux, les pages Performances et Galerie affichent des données de démonstration.

## Avertissement
Outil de recherche et d'aide à la décision, **non certifié dispositif médical**. Il ne remplace pas le jugement d'un médecin. Données d'entraînement : Chest X-Ray Pneumonia (Kaggle). Ne publiez jamais de données patients réelles.
