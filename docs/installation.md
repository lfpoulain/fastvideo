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

# Ryzen AI Max 385 / 395
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Backend rocm -AmdArch gfx1151 -SetupOnly

# Voir le plan sans installer ni télécharger
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -Backend rocm -AmdArch gfx1150 -DryRun
```

```bash
# Linux : installer puis ouvrir l'app
bash setup.sh

# Préparer seulement le HX470
bash setup.sh --backend rocm --amd-arch gfx1150 --setup-only

# Ryzen AI Max 385 / 395
bash setup.sh --backend rocm --amd-arch gfx1151 --setup-only

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
| `-Backend rocm -AmdArch gfx1150` | `--backend rocm --amd-arch gfx1150` | Paquets AMD pour HX370 / HX470 et GPU 890M / 880M. |
| `-Backend rocm -AmdArch gfx1151` | `--backend rocm --amd-arch gfx1151` | Paquets AMD pour Ryzen AI Max 385 / 395 et GPU 8050S / 8060S. |
| `-Backend cuda` | `--backend cuda` | Paquets NVIDIA CUDA 12.8. |
| `-Backend cpu` | `--backend cpu` | Paquets CPU, y compris sur une machine équipée d'un GPU. |
| `-Model lfm-450m` | `--model lfm-450m` | Modèle sélectionné à l'ouverture ; les douze choix restent disponibles. |
| `-SetupOnly` | `--setup-only` | Installation et vérification sans ouvrir la fenêtre. |
| `-DryRun` | `--dry-run` | Affichage du plan sans installer ; seul le journal de démarrage est écrit. |
| `-Venv 'autre-dossier'` | `--venv autre-dossier` | Utiliser un autre environnement au lieu de `.venv`. |
| `-LogFile 'mon-lancement.log'` | — | Choisir le fichier du journal PowerShell. |
| `-NoPause` | — | Ne pas attendre Entrée après une erreur lors d'un lancement interactif. |
| `-AppArgs @('--interval', '0.5')` | `-- --interval 0.5` | Options supplémentaires transmises à `app.py`. |

Pour imposer un interpréteur, utilise `-Python 'C:\chemin\python.exe'` sous
PowerShell ou `FASTVIDEO_PYTHON=/chemin/python3.12 bash setup.sh` sous Bash.

Les tableaux `-AppArgs` se passent avec un appel PowerShell direct. Exemple :

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "& .\setup.ps1 -Model lfm-450m -AppArgs @('--interval', '0.5')"
```

L'installation privilégie le target HIP / `rocminfo`, puis le nom du GPU, puis
celui du processeur. Elle ne fixe pas toutes les machines AMD au profil du HX470.

| Matériel détecté | Profil |
| --- | --- |
| HX370 / HX375 / HX470 / HX475, Ryzen AI 9 365 / 465, Radeon 890M / 880M | `gfx1150` |
| Ryzen AI Max 385 / 390 / 395 et variantes PRO, Radeon 8040S / 8050S / 8060S / 8065S | `gfx1151` |
| Ryzen AI 350 / 340 / 345 / 330 / 440 / 450, Radeon 860M / 820M | `gfx1152` |

Ces correspondances proviennent des [notes officielles AMD ROCm](https://rocmdocs.amd.com/en/develop/about/release-notes.html).
Le nom Radeon 840M seul reste ambigu entre plusieurs targets ; un target déjà
détecté par HIP ou le nom complet du processeur est nécessaire.
Les profils AMD explicites disponibles sont `gfx1150`, `gfx1151`, `gfx1152`,
`gfx1100` et `gfx1201`. La sélection GPU existante continue de fonctionner sur NVIDIA.
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
Les scripts n'installent pas les poids de tous les modèles : seul le modèle utilisé
est téléchargé à sa première analyse.

### Si la fenêtre se ferme trop vite

Chaque exécution des lanceurs crée un journal `logs/startup-<date>-<pid>.log`
dans le dépôt, dès le début du script, avant la recherche de Python. Il contient
le système, l'interpréteur choisi, les vérifications des candidats Python,
le runtime PyTorch existant, la détection du matériel, les commandes exécutées,
leurs sorties et erreurs, puis le code de sortie final. La sortie reste visible
dans le terminal. Les journaux restent locaux et sont exclus de Git.

Sous PowerShell, une erreur garde la console interactive ouverte jusqu'à Entrée.
Les exécutions avec entrée redirigée ne bloquent pas ; `-NoPause` désactive aussi
cette attente. Le journal reste disponible après fermeture de la fenêtre.
Si le dossier du dépôt ne permet pas d'écrire le journal, le lanceur utilise
`%TEMP%\FastVideo\logs` sous Windows, ou `${TMPDIR:-/tmp}/fastvideo-logs` sous Bash.
Le chemin exact est affiché au début et à la fin.

Sur le PC distant, ouvre un terminal PowerShell dans le dépôt puis lance :

```powershell
git pull
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1 -LogFile .\logs\startup-remote.log
```

Ce fichier choisi est remplacé à chaque exécution. Pour lire sa fin :

```powershell
Get-Content .\logs\startup-remote.log -Tail 100
```

Le journal du lanceur complète `logs/fastvideo.log`, écrit une fois l'interface
initialisée. Il capture aussi les erreurs d'importation et d'ouverture de Tk
émises par l'app lancée depuis le script. Un lancement direct de `app.py` ne crée
pas ce journal du lanceur. Si PowerShell refuse d'exécuter le script avant son
démarrage (politique d'exécution ou paramètres invalides), aucun journal ne peut
encore être créé ; copie alors le message du terminal. La commande ci-dessus
fixe la politique d'exécution pour ce processus.

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

### Ryzen AI HX370 / HX470 et Ryzen AI Max · ROCm

Utilise les pilotes, le système et les wheels PyTorch correspondant au GPU :
`gfx1150` pour HX370 / HX470, `gfx1151` pour Ryzen AI Max 385 / 395, dans les
[instructions AMD ROCm](https://rocmdocs.amd.com/en/latest/install/rocm.html).
Le [guide AMD PyTorch](https://developer.amd.com/playbooks/pytorch-rocm-llms/)
fournit les paquets pour cette architecture.

Après avoir suivi les prérequis AMD, exemple d'installation ROCm 10 :

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1150]==2.13.0+rocm10.0.0" "torchvision[device-gfx1150]==0.28.0+rocm10.0.0"
```

Pour Ryzen AI Max 385 / 395, remplace les deux extras `device-gfx1150` par
`device-gfx1151`. Il s'agit de l'iGPU Radeon ; l'app n'utilise pas le NPU.

Vérifie que PyTorch voit le GPU :

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('ROCm:', torch.version.hip); print('GPU disponible:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`ROCm` doit indiquer une version HIP et `GPU disponible` doit être `True`.
PyTorch utilise également l'API `torch.cuda` pour les GPU AMD.

FastVideo sélectionne FP16 sur ROCm. Le moteur utilise les opérations PyTorch et
SDPA ; l'app n'impose pas de dépendance à FlashAttention ou bitsandbytes.
Le fonctionnement et la vitesse restent à mesurer sur le matériel AMD utilisé.
Sur un iGPU, vérifie la mémoire accessible à PyTorch, particulièrement pour FastVLM 7B
et Moondream3. La quantité de RAM du PC n'est pas nécessairement la mémoire allouée au GPU.

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
