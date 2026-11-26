"""Profil de l'auteur (modifiable). Le telephone n'est pas affiche par defaut : decommentez la ligne 'tel'."""
PROFILE = {
    "nom": "Ibrahima DIENG",
    "titre": "Junior Data Scientist & Analyste de Données",
    "sous_titre": "Statistique · Intelligence Artificielle",
    "lieu": "Thiès, Sénégal",
    "email": "ibrahimadieng997@gmail.com",
    # "tel": "+221 77 282 92 34",
    "github": "https://github.com/ibrahimadieng997-boop",
    "resume": ("Titulaire d'une licence en Statistique et Informatique Décisionnelle (UADB), je poursuis une formation en "
               "Data Science et Intelligence Artificielle au Dakar Institute of Technology (niveau équivalent Master 1). "
               "Je mobilise l'analyse statistique, Python, R, SQL, le Machine Learning et le Deep Learning pour transformer "
               "les données en informations exploitables, développer des modèles prédictifs et concevoir des tableaux de bord "
               "interactifs afin de soutenir la prise de décision."),
    "projets": [
        ("RadiologyAI", "Aide au diagnostic de la pneumonie sur radiographie thoracique : VGG16 en transfer learning, Grad-CAM, "
                        "validation médicale et rapport PDF (Python, TensorFlow/Keras, Streamlit)."),
        ("BreastCare AI", "Classification d'images échographiques mammaires en 3 classes (bénigne, maligne, normale) sur 436 images, "
                          "VGG16 en transfer learning, interface Streamlit."),
        ("Perceptron & MLP", "Prédiction de la progression du diabète (442 patients) avec perceptron simple et multicouche, "
                             "descente de gradient et rétropropagation."),
        ("Machine Learning", "Marketing bancaire (dépôt à terme), segmentation de patients, prédiction de l'état de véhicules (CoinAfrique)."),
    ],
    "competences": {
        "Langages": ["Python", "R", "SQL", "HTML/CSS", "Jupyter", "Google Colab"],
        "IA & Data": ["Machine Learning", "Deep Learning", "TensorFlow", "Keras", "scikit-learn", "NumPy", "Pandas"],
        "Visualisation": ["Power BI", "Tableau", "Excel", "Streamlit", "Seaborn", "Matplotlib"],
        "Données": ["PostgreSQL", "PowerAMC", "Web scraping (BeautifulSoup, Selenium)"],
    },
    "parcours": [
        ("2025 – 2026", "Certification en Data Science et IA (en cours)", "Dakar Institute of Technology"),
        ("2025 – 2026", "Certificat de Data Analyst", "Xarala Académie"),
        ("2022 – 2023", "Licence 3 Statistique et Informatique Décisionnelle", "Université Alioune Diop de Bambey"),
        ("2022 – 2023", "Attestation de travail", "Centre de Santé Mentale Dalal Xel, Thiès"),
    ],
    "langues": "Wolof · Français · Anglais",
    "engagement": "Membre actif de Mister Bane Center : actions humanitaires auprès des talibés (aide matérielle, accompagnement, journées solidaires).",
}
