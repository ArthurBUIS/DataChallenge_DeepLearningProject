The repo is organized as follows:


DataChallenge/
│
├── data/
│   ├── raw/                   # CSV bruts
│   ├── processed/             # datasets filtrés / transformés
│   └── README.md
│
├── notebooks/
│   ├── 00_exploration.ipynb
│   ├── 01_feature_analysis.ipynb
│   └── sandbox.ipynb
│
├── src/
│   ├── __init__.py
│
│   ├── data/
│   │   ├── dataset.py         # chargement + sélection de features
│   │   └── preprocessing.py   # feature engineering
│
│   ├── models/
│   │   ├── baseline.py        # modèle TF-IDF + LogReg
│   │   └── svm.py             # futur modèle SVM
│
│   ├── training/
│   │   ├── train.py           # boucle d'entraînement générique
│   │   └── evaluate.py
│
│   ├── inference/
│   │   └── predict.py
│
│   ├── config/
│   │   └── config.yaml
│
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── model_saving.py
│
├── models/                    # modèles sauvegardés (.joblib + .json)
│
├── submissions/
│
├── scripts/
│   ├── train_baseline.py
│   └── train_svm.py
│
├── requirements.txt
├── .gitignore
└── README.md
