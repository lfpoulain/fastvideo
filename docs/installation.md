# Installation

[← Retour au README](../README.md)

FastVideo utilise une fenêtre Tkinter et un moteur PyTorch local. Python 3.12 est
la version testée. Pour AMD, vérifie aussi la version de Python prise en charge par
les paquets ROCm sélectionnés.

## Installation automatique

Après `git clone https://github.com/lfpoulain/fastvideo.git` puis `cd fastvideo` :

```powershell
# Windows : installer puis ouvrir l'app
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1

# Préparer seulement le HX470
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Backend rocm -AmdArch gfx1150 -SetupOnly

# Voir le plan sans installer ni télécharger
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Backend rocm -AmdArch gfx1150 -DryRun
```

```bash
# Linux : installer puis ouvrir l'app
bash setup.sh

# Préparer seulement le HX470
bash setup.sh --backend rocm --amd-arch gfx1150 --setup-only

# Voir le plan sans installer ni télécharger
bash setup.sh --backend rocm --amd-arch gfx1150 --dry-run
```

Les scripts peuvent être lancés depuis un autre dossier et acceptent les chemins
contenant des espaces. Ils cherchent Python 3.12 avec Tkinter. S'il manque, ils
utilisent [uv](https://docs.astral.sh/uv/guides/install-python/) pour le télécharger
dans `.tools/`. Si uv manque aussi, ils téléchargent son installateur officiel
fixé à la version 0.12.23. Aucune activation du venv n'est nécessaire.
`-ExecutionPolicy Bypass` s'applique seulement au processus lancé.

| PowerShell | Bash | Effet |
| --- | --- | --- |
| `-Backend auto` | `--backend auto` | Détection par défaut ; réutilise d'abord un PyTorch GPU fonctionnel. |
| `-Backend rocm -AmdArch gfx1150` | `--backend rocm --amd-arch gfx1150` | Paquets AMD pour le HX470. |
| `-Backend cuda` | `--backend cuda` | Paquets NVIDIA CUDA 12.8. |
| `-Backend cpu` | `--backend cpu` | Paquets CPU, y compris sur une machine équipée d'un GPU. |
| `-Model lfm-450m` | `--model lfm-450m` | Modèle sélectionné à l'ouverture ; les huit choix restent disponibles. |
| `-SetupOnly` | `--setup-only` | Installation et vérification sans ouvrir la fenêtre. |
| `-DryRun` | `--dry-run` | Affichage du plan sans écrire de fichiers ni installer. |
| `-Venv 'autre-dossier'` | `--venv autre-dossier` | Utiliser un autre environnement au lieu de `.venv`. |
| `-AppArgs @('--interval', '0.5')` | `-- --interval 0.5` | Options supplémentaires transmises à `app.py`. |

Pour imposer un interpréteur, utilise `-Python 'C:\chemin\python.exe'` sous
PowerShell ou `FASTVIDEO_PYTHON=/chemin/python3.12 bash setup.sh` sous Bash.

Les tableaux `-AppArgs` se passent avec un appel PowerShell direct. Exemple :

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "& .\setup.ps1 -Model lfm-450m -AppArgs @('--interval', '0.5')"
```

La détection du HX470 / Radeon 890M choisit `gfx1150`. Les profils AMD explicites
disponibles sont `gfx1150`, `gfx1151`, `gfx1152`, `gfx1100` et `gfx1201`.
Pour une carte AMD dont l'architecture n'est pas reconnue, le script demande un
profil. Consulte la [matrice AMD](https://rocmdocs.amd.com/en/latest/install/rocm.html)
pour vérifier le matériel, le système et les pilotes avant l'installation.

Les scripts installent PyTorch avant les dépendances de l'app et contraignent
ensuite ses versions pour conserver la distribution CUDA / ROCm choisie.
Ils réutilisent un PyTorch compatible existant. Un GPU demandé mais indisponible
interrompt l'installation avec une erreur. Après une installation réussie,
les prochains lancements vérifient l'environnement puis évitent de relancer pip.
Une mise à jour de l'installateur ou des dépendances déclenche une nouvelle préparation.
Relancer après un téléchargement interrompu reprend la préparation.

Les pilotes, les prérequis ROCm et la session graphique restent à installer au
niveau du système. Sous Linux, si OpenCV signale une bibliothèque absente, installe
les paquets correspondants, par exemple `libgl1` et `libglib2.0-0` sur Ubuntu.
Les scripts n'installent pas les poids des huit modèles : seul le modèle utilisé
est téléchargé à sa première analyse.

La suite du guide permet une installation manuelle.

## Préparer le dossier

Sous Windows :

```powershell
git clone https://github.com/lfpoulain/fastvideo.git
cd fastvideo
python -m venv .venv
```

Sous Linux :

```bash
git clone https://github.com/lfpoulain/fastvideo.git
cd fastvideo
python3 -m venv .venv
```

Dans les commandes suivantes, remplace `.\.venv\Scripts\python.exe` par
`.venv/bin/python` sous Linux. Les commandes Windows utilisent directement
l'interpréteur du venv et ne nécessitent pas de modifier la politique PowerShell.

## AMD

### Ryzen AI 9 HX470 · ROCm

Utilise les pilotes, le système et les wheels PyTorch correspondant au HX470
(`gfx1150`) dans les [instructions AMD ROCm](https://rocmdocs.amd.com/en/latest/install/rocm.html).
Le [guide AMD PyTorch](https://developer.amd.com/playbooks/pytorch-rocm-llms/)
fournit les paquets pour cette architecture.

Après avoir suivi les prérequis AMD, exemple d'installation ROCm 10 :

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1150]==2.13.0+rocm10.0.0" "torchvision[device-gfx1150]==0.28.0+rocm10.0.0"
```

Vérifie que PyTorch voit le GPU :

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('ROCm:', torch.version.hip); print('GPU disponible:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`ROCm` doit indiquer une version HIP et `GPU disponible` doit être `True`.
PyTorch utilise également l'API `torch.cuda` pour les GPU AMD.

FastVideo sélectionne FP16 sur ROCm. Le moteur utilise les opérations PyTorch et
SDPA ; l'app n'impose pas de dépendance à FlashAttention ou bitsandbytes.
Le fonctionnement et la vitesse du HX470 restent à mesurer sur cette machine.

## NVIDIA

Utilise le [sélecteur officiel PyTorch](https://pytorch.org/get-started/locally/)
pour choisir la distribution CUDA compatible avec ton pilote.

La configuration testée sur RTX 4090 utilise PyTorch 2.11.0 avec CUDA 12.8 et
torchvision 0.26.0. Pour reproduire cette configuration :

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Le mode automatique choisit BF16 si le GPU le prend en charge, sinon FP16.

## CPU

Pour essayer l'app sur un PC sans GPU pris en charge :

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

L'app utilise FP32 sur CPU. La cadence des descriptions dépend alors du processeur
et du modèle choisi.

## Installer l'app

Après PyTorch, ajoute les dépendances de l'app :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Installe les dépendances dans le même venv que PyTorch, en conservant la distribution
GPU choisie. Transformers et timm sont fixés aux versions testées.

Sous Linux, Tk doit être disponible pour l'interpréteur choisi. Exemple Ubuntu avec
le Python système :

```bash
sudo apt install python3-tk
```

## Premier téléchargement et mode hors ligne

Le premier chargement d'un modèle télécharge ses fichiers dans `models/`, à côté
de `app.py`. Prévois plusieurs gigaoctets selon le modèle. Le dossier est exclu de Git.
L'interface reste réactive pendant le chargement.

Après un premier chargement réussi, tu peux demander une exécution hors ligne :

```powershell
.\.venv\Scripts\python.exe app.py --offline --model lfm-450m
```

Le mode hors ligne exige que tous les fichiers du modèle choisi soient déjà en cache.
