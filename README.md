<p align="center">
  <img src="docs/assets/banner.svg" alt="FastVideo — Local vision. Live webcam." width="100%">
</p>

<p align="center">
  <a href="https://github.com/lfpoulain/fastvideo/actions/workflows/ci.yml"><img src="https://github.com/lfpoulain/fastvideo/actions/workflows/ci.yml/badge.svg" alt="Vérifications automatiques"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&amp;logoColor=white" alt="Python 3.12"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-8dddaf" alt="Licence MIT"></a>
  <img src="https://img.shields.io/badge/Inference-Local-224b3b" alt="Inférence locale">
</p>

# FastVideo

**Ta webcam, huit modèles vision et une fenêtre Python.**

FastVideo affiche la webcam en continu et décrit ce qu'elle voit avec un modèle exécuté
sur ton PC. Choisis le modèle dans le menu, adapte la consigne et compare les résultats.
L'app utilise Tkinter, OpenCV et PyTorch, avec détection automatique de **CUDA, ROCm ou CPU**.

**[Installation](docs/installation.md) · [Utilisation](docs/usage.md) · [Architecture](docs/architecture.md) · [Validation](docs/validation.md) · [Contribuer](CONTRIBUTING.md)**

## Ce que tu peux faire

- Afficher une webcam et obtenir des descriptions actualisées en français.
- Passer de SmolVLM2 à Qwen, MiniCPM, LFM ou FastVLM dans la même fenêtre.
- Modifier la consigne et la fréquence des analyses, puis mettre l'IA en pause.
- Comparer jusqu'à trois images récentes pour observer les changements visibles.
- Analyser une image locale depuis le terminal et afficher le temps d'inférence.
- Utiliser les modèles déjà téléchargés en mode hors ligne, avec `--offline`.

Les images sont traitées en mémoire sur le PC. Aucune clé API ni serveur à lancer.
Les poids sont téléchargés depuis les dépôts officiels au premier usage de chaque modèle.

## Démarrage rapide

Python **3.12** est recommandé. Installe d'abord les pilotes et le PyTorch adaptés à ton GPU.

<details open>
<summary><strong>Windows · PowerShell</strong></summary>

```powershell
git clone https://github.com/lfpoulain/fastvideo.git
cd fastvideo
python -m venv .venv
```

Choisis l'installation **[AMD ROCm / HX470](docs/installation.md#amd)**,
**[NVIDIA CUDA](docs/installation.md#nvidia)** ou **[CPU](docs/installation.md#cpu)**,
puis lance :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

</details>

<details>
<summary><strong>Linux · Terminal</strong></summary>

```bash
git clone https://github.com/lfpoulain/fastvideo.git
cd fastvideo
python3 -m venv .venv
```

Installe PyTorch pour ton GPU en suivant le [guide d'installation](docs/installation.md), puis :

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Tk doit être disponible pour la version de Python choisie. Sur Ubuntu, le paquet
`python3-tk` fournit Tk pour le Python système.

</details>

Dans la fenêtre : **choisir un modèle → ouvrir la webcam → analyser en direct**.

## Les huit modèles

| Modèle | Identifiant `--model` | Dépôt officiel |
| --- | --- | --- |
| **SmolVLM2 500M** · sélection par défaut | `smol` | [HuggingFaceTB](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) |
| **Qwen3.5 0,8B** | `qwen-0.8b` | [Qwen](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| **Qwen3.5 2B** | `qwen-2b` | [Qwen](https://huggingface.co/Qwen/Qwen3.5-2B) |
| **MiniCPM-V 4.6** | `minicpm` | [OpenBMB](https://huggingface.co/openbmb/MiniCPM-V-4.6) |
| **LFM2.5-VL 450M** | `lfm-450m` | [Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) |
| **LFM2.5-VL 1,6B** | `lfm-1.6b` | [Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B) |
| **LFM2.5-VL 3B** | `lfm-3b` | [Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-VL-3B) |
| **FastVLM 0,5B** | `fastvlm` | [Apple](https://huggingface.co/apple/FastVLM-0.5B) |

Les révisions sont fixées dans [`vision.py`](vision.py). Les sept premiers modèles utilisent
les architectures natives de Transformers. FastVLM utilise le code du dépôt officiel Apple,
chargé à une révision précise. FastVLM analyse une image par génération.

## Quelques commandes

```powershell
# LFM 450M, réponses courtes et intervalle de 0,5 seconde
.\.venv\Scripts\python.exe app.py --model lfm-450m --interval 0.5 --max-tokens 60

# Exiger l'accélération AMD ROCm
.\.venv\Scripts\python.exe app.py --device rocm

# Comparer trois images récentes
.\.venv\Scripts\python.exe app.py --frames 3

# Analyser une image locale
.\.venv\Scripts\python.exe app.py --model qwen-2b --image photo.jpg
```

Voir [toutes les options et le dépannage](docs/usage.md).

## Pensé pour une webcam

La capture, l'interface et l'inférence tournent séparément. L'app utilise les images
les plus récentes, limite leur taille à 640 × 480 et ne lance qu'une analyse à la fois.
Un seul modèle est chargé ; sa mémoire est libérée avant de passer au suivant.
Le mode ROCm utilise FP16 et l'attention SDPA de PyTorch.

La vidéo reste continue et les descriptions arrivent à la vitesse du modèle.
L'intervalle est le délai minimum entre les départs de deux analyses.

**Validation actuelle :** les huit modèles ont généré des réponses sur une RTX 4090
en BF16 et FP16. La vraie webcam, la pause et le changement de modèle ont été testés.
Le HX470 reste à valider sur matériel AMD. Voir le [protocole et les mesures](docs/validation.md).

## Licence

Le code de l'application est sous [licence MIT](LICENSE). Les poids et le code fourni
par les dépôts des modèles conservent leurs licences respectives, indiquées sur leurs
pages officielles. Les poids et les images de webcam ne sont pas inclus dans ce dépôt.
