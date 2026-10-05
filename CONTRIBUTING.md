# Contribuer à FastVideo

Merci pour les corrections, les mesures matérielles et les améliorations des adaptateurs.
Le projet conserve une app Python locale avec une fenêtre et un moteur d'inférence simple.

## Préparer l'environnement

Pour lancer l'app, suis le [guide d'installation](docs/installation.md).
Pour travailler sur les tests et le formatage uniquement :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Sous Linux, utilise `.venv/bin/python`.

## Vérifications avant une contribution

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Les tests automatiques n'utilisent ni webcam réelle, ni poids de modèle.
Une modification de l'adaptateur doit aussi être vérifiée par une génération réelle :

```powershell
.\.venv\Scripts\python.exe tools/verify_models.py --models lfm-450m --offline
```

Pour une nouvelle architecture, ajoute un choix dans `vision.py`, fixe le dépôt
et sa révision, puis mets à jour le catalogue et les docs. Vérifie la génération,
la précision GPU choisie, l'annulation et la libération de mémoire.

## Rapporter un problème ou des mesures

Indique le système, Python, les versions PyTorch/Transformers, le GPU, le modèle,
les options de lancement et les étapes pour reproduire. Pour AMD, précise ROCm
et l'architecture GPU. Les rapports peuvent être ouverts dans les
[issues du dépôt](https://github.com/lfpoulain/fastvideo/issues).

Les exemples publics devraient utiliser des images synthétiques ou des images
dont tu peux partager le contenu. Les poids, les caches et les rapports locaux
restent exclus de Git.
