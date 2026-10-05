# Installation

[← Retour au README](../README.md)

FastVideo utilise une fenêtre Tkinter et un moteur PyTorch local. Python 3.12 est
la version testée. Pour AMD, vérifie aussi la version de Python prise en charge par
les paquets ROCm sélectionnés.

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
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
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
