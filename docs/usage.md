# Utilisation

[← Retour au README](../README.md)

## La fenêtre

1. Choisis un des douze modèles dans le menu.
2. Clique sur **Ouvrir la webcam**. Le numéro `0` désigne la caméra par défaut.
3. Clique sur **Analyser en direct** pour charger le modèle et commencer.
4. Modifie la consigne pour demander une description, un comptage ou la lecture d'un texte.
5. Clique sur **Mettre en pause** pour arrêter les nouvelles analyses.

Le panneau de droite affiche la dernière réponse, le GPU utilisé, la mémoire
allouée par PyTorch et l'heure de la réponse. La fermeture de la webcam arrête l'analyse.
Changer de modèle annule la génération en cours, puis libère le précédent modèle.
Un téléchargement ou un chargement initial doit se terminer avant le changement effectif.

### Boutons et réglages

| Commande | Effet |
| --- | --- |
| **Charger le modèle** | Prépare les fichiers et charge le modèle sans ouvrir la webcam. |
| **Moteur d’analyse** | Automatique GPU/CPU, ROCm, CUDA, CPU ou NPU AMD. Le changement libère le moteur précédent au prochain chargement. |
| **Installer FastFlowLM…** | Ouvre le guide d'installation du moteur et du pilote NPU AMD. |
| **Résolution envoyée à l’IA** | Change le plafond de dimensions à la prochaine analyse, sans réouvrir la webcam ni recharger le modèle. |
| **Attention ROCm expérimentale** | Autorise les kernels AMD expérimentaux ; à cocher avant le premier chargement. |
| **Analyser en direct / Mettre en pause** | Lance les analyses répétées ou arrête les nouvelles générations. |
| **Analyser une image** | Analyse ponctuellement la dernière image de la webcam, puis reste en pause. |
| **Copier** | Copie la dernière description dans le presse-papiers. |
| **Exporter…** | Enregistre le journal visible dans un fichier `.log` ou `.txt`. |
| **Effacer** | Vide le journal affiché ; le fichier persistant est conservé. |

Le panneau de gauche permet de régler l'intervalle, le nombre d'images et la
longueur maximale de la réponse pendant l'utilisation. Les nouvelles valeurs
s'appliquent à l'analyse suivante. Quatre consignes prêtes à l'emploi sont proposées,
et le texte reste librement modifiable. Le panneau défile si la fenêtre est réduite.

### Résolution et FPS de la webcam

Choisis **1920x1080** et **25 FPS** dans le panneau de gauche (valeurs par défaut).
Les choix vont de 640×480 à 3840×2160, avec une cadence demandée de 1 à 60 FPS.
Clique sur **Appliquer à la webcam** si elle est déjà ouverte : le flux est
réouvert avec les nouveaux réglages, le modèle chargé est conservé et l'analyse
en direct reprend si elle était active.

Le format **auto** essaie MJPG en HD pour réduire la bande passante USB.
Tu peux choisir explicitement `mjpg` ou `yuy2` selon ta webcam. Les modes réellement
disponibles dépendent du matériel et du pilote. Sous le flux et dans les logs,
FastVideo affiche la résolution reçue, les FPS annoncés par le pilote et le codec ;
le compteur **Caméra** affiche les FPS réellement mesurés. Un mode refusé est signalé.
Voir les [propriétés de capture OpenCV](https://docs.opencv.org/4.x/d4/d15/group__videoio__flags__base.html).

Le bouton **Agrandir** ouvre un aperçu séparé, redimensionnable. **F11** bascule
cet aperçu en plein écran et **Échap** le ferme. La case **Miroir** ne modifie que
l'affichage. Le flux reste en haute résolution ; les images destinées au modèle
sont réduites selon **Résolution envoyée à l’IA** : 256×256, 320×240, 640×480
(défaut), 960×540, 1280×720, 1920×1080 ou `original`. Les proportions sont conservées,
sans recadrage ni agrandissement. Ainsi, une source 1920×1080 avec un plafond
640×480 est envoyée en 640×360. Le choix s'applique aussi aux images historiques
et au mode CLI ; la résolution réelle envoyée est écrite dans le journal.

```powershell
.\.venv\Scripts\python.exe app.py --capture-resolution 1920x1080 --capture-fps 25
```

### Vitesse et téléchargement

- **Caméra** : FPS mesurés à partir des images reçues pendant les deux dernières secondes.
- **Dernière analyse** : durée du prétraitement et de la génération, avec synchronisation GPU.
- **Rythme observé** : réponses par minute calculées sur les dernières réponses, intervalle compris.
- **Moteur** : CUDA, ROCm, CPU ou NPU AMD ; le runtime figure sous la réponse.

Les FPS de la caméra sont indépendants de la vitesse du modèle. Le rythme apparaît
après deux réponses et repart à zéro lors d'une reprise ou d'un changement de modèle.
La mémoire affichée est l'allocation PyTorch, pas la totalité de la mémoire du GPU.

Le journal indique la vérification du cache, les téléchargements, le chargement,
les commandes et les temps de réponse. Pendant un transfert, la barre affiche les
octets traités, le pourcentage et le débit moyen en Mio/s ou Gio/s. Les fichiers
déjà en cache sont réutilisés. Le chargement en mémoire utilise une barre animée ;
il n'affiche pas de pourcentage inventé.

Les logs de l'interface sont aussi écrits dans `logs/fastvideo.log`, avec rotation
à 2 Mio et deux sauvegardes. Les images, les consignes et les descriptions ne sont
pas ajoutées automatiquement au journal. Le journal visible conserve au maximum
400 entrées et peut être exporté. Les erreurs détaillées restent dans le fichier.

Exemples de consignes :

> Décris les objets visibles en une phrase courte, en français.

> Compte les objets sur la table et indique leurs couleurs.

> Lis le texte visible dans l'image. Signale les mots illisibles.

## Options du terminal

```powershell
.\.venv\Scripts\python.exe app.py --help
```

| Option | Valeur par défaut | Utilité |
| --- | --- | --- |
| `--model` | `smol`, ou `qwen-0.8b` avec `--device npu` | Choix initial du modèle ; il reste modifiable dans la fenêtre. |
| `--camera` | `0` | Numéro de webcam, de 0 à 9. |
| `--capture-resolution` | `1920x1080` | Résolution demandée au flux webcam, de 640×480 à 3840×2160. |
| `--capture-fps` | `25` | FPS demandés à la webcam, de 1 à 60. |
| `--capture-format` | `auto` | `auto` (essaie MJPG en HD), `mjpg` ou `yuy2`. |
| `--analysis-resolution` | `640x480` | Plafond pour l'image IA : `256x256`, `320x240`, `640x480`, `960x540`, `1280x720`, `1920x1080`, `original`. |
| `--device` | `auto` | `auto`, `cuda`, `rocm`, `cpu` ou `npu`. |
| `--flm-path` | recherche automatique | Chemin de l'exécutable FastFlowLM si absent du PATH et des dossiers Windows usuels. |
| `--rocm-experimental-attention` | désactivé | Autorise au lancement les kernels d'attention ROCm expérimentaux. |
| `--interval` | `2` | Délai minimum entre les départs des analyses, de 0,5 à 30 secondes. |
| `--frames` | `1` | Jusqu'à 3 images récentes ; FastVLM et Moondream utilisent la dernière. |
| `--max-tokens` | `100` | Limite de génération, de 1 à 512 tokens. |
| `--offline` | désactivé | Utiliser seulement les fichiers déjà téléchargés. |
| `--image` | — | Analyser un fichier image sans ouvrir la fenêtre ni la webcam. |
| `--list-models` | — | Afficher les identifiants et les dépôts des modèles. |
| `--log-file` | `logs/fastvideo.log` dans le dépôt | Emplacement du journal persistant de l'interface. |

Sous Linux, utilise `.venv/bin/python` dans ces commandes.

### Une image locale

```powershell
.\.venv\Scripts\python.exe app.py --model minicpm --image photo.jpg
```

La commande imprime la description et le temps d'inférence. L'image est convertie
en RGB et réduite selon `--analysis-resolution`, avec conservation des proportions.

### NPU AMD

Choisis **NPU AMD · FastFlowLM** dans la fenêtre. Si nécessaire, clique sur
**Installer FastFlowLM…** et suis le guide officiel pour le moteur et le pilote.
Sous Windows, il faut Windows 11 et un NPU AMD XDNA 2 ; le guide demande un pilote
NPU au moins égal à 32.0.203.311. Utilise un pilote récent compatible avec ton PC.
Sous Linux, suis les prérequis XRT/amdxdna du guide FastFlowLM.

FastVideo démarre son propre processus FLM en arrière-plan, sur `127.0.0.1` et
un port privé. Il prépare les fichiers via `flm pull` et réutilise le cache FLM
séparé des poids PyTorch, sans téléchargement forcé. L'emplacement reste celui
configuré par FastFlowLM (`FLM_MODEL_PATH` s'il est défini). Le téléchargement et
ses indications de débit sont remontés dans le journal ; la préparation utilise
une barre animée. Aucun service cloud ni clé API n'est nécessaire pour l'analyse.
Le moteur est arrêté quand il est remplacé ou quand FastVideo est fermé.

Seuls **Qwen3.5 0,8B, 2B et 4B** sont proposés dans ce mode, en Q4 et avec le
raisonnement désactivé. MiniCPM-V, SmolVLM2, LFM2.5-VL, FastVLM et Moondream restent
sur GPU/CPU. L'app ne fournit pas de conversion NPU de modèles arbitraires.
L'analyse NPU reçoit uniquement la dernière image, même avec `--frames 3`.
La résolution IA est réglée dans FastVideo ; le redimensionnement automatique
supplémentaire de FLM est désactivé. Le contexte FLM est limité à 8192 tokens :
les grandes images peuvent dépasser ce budget. Commence en 320×240 ou 640×480.

Le mode `--offline` strict est refusé pour le NPU : FLM peut consulter des
métadonnées et sa propre version lors de la préparation. Les images et
l'inférence restent locales ; FastVideo n'enregistre pas les requêtes ni les
descriptions du serveur dans le journal. Aucun repli GPU silencieux n'est effectué.

```powershell
.\.venv\Scripts\python.exe app.py --device npu --model qwen-2b --analysis-resolution 320x240
.\.venv\Scripts\python.exe app.py --device npu --model qwen-0.8b --image photo.jpg
```

Sans `--model`, `--device npu` sélectionne Qwen3.5 0,8B. Le mode `auto` conserve
la sélection GPU/CPU ; il ne choisit pas le NPU implicitement.
Sources : [FastFlowLM Windows](https://fastflowlm.com/docs/install_win/),
[Linux](https://github.com/ROCm/FastFlowLM/blob/main/docs/linux-getting-started.md),
[catalogue Qwen](https://fastflowlm.com/docs/models/qwen/).

### Plusieurs images récentes

```powershell
.\.venv\Scripts\python.exe app.py --model smol --frames 3
```

L'app conserve un petit historique avec un échantillon par seconde et y ajoute
l'image actuelle. Elle transmet les images dans l'ordre chronologique.
Il peut y avoir moins d'images au démarrage, tant que l'historique se remplit.
Les trois FastVLM et Moondream3 reçoivent une seule image, même avec `--frames 3`.

### Les nouveaux modèles

```powershell
.\.venv\Scripts\python.exe app.py --model fastvlm-1.5b
.\.venv\Scripts\python.exe app.py --model fastvlm-7b
.\.venv\Scripts\python.exe app.py --model qwen-4b
.\.venv\Scripts\python.exe app.py --model moondream3 --max-tokens 60
```

Moondream3 utilise `query` avec le raisonnement désactivé, une température de zéro
et la limite de tokens demandée. L'annulation ferme son flux de réponse entre les
morceaux de texte. L'encodage initial d'une image se termine avant de s'arrêter.
Le mode hors ligne exige aussi le tokenizer `moondream/starmie-v1`, téléchargé et
mis en cache automatiquement lors du premier chargement connecté.

### Réduire la latence

Une seule image et une réponse courte réduisent le travail du modèle :

```powershell
.\.venv\Scripts\python.exe app.py --model lfm-450m --frames 1 --max-tokens 60 --interval 0.5
```

Si une analyse dure plus longtemps que l'intervalle, la suivante attend sa fin.
Les images intermédiaires ne sont pas mises en attente. La résolution d'analyse
est plafonnée et MiniCPM utilise une seule vue avec un downsampling de 16×.

## Dépannage

| Symptôme | Vérification |
| --- | --- |
| Webcam introuvable ou occupée | Ferme les autres apps utilisant la caméra, puis essaie un autre numéro. |
| L'app affiche `CPU` | Vérifie `torch.cuda.is_available()` et la distribution PyTorch installée. |
| `--device rocm` échoue | Vérifie que `torch.version.hip` contient une version et que le GPU est disponible. |
| Erreur en mode hors ligne | Charge une première fois ce modèle avec une connexion Internet. |
| Mémoire GPU insuffisante | Essaie SmolVLM2 ou LFM 450M, une image et une réponse courte ; ferme les autres tâches GPU. |
| `tkinter` absent sous Linux | Installe Tk pour la version de Python utilisée. |
| Description approximative ou mauvaise langue | Essaie une consigne plus précise ou un autre modèle. La qualité dépend du modèle et de la scène. |
| Avertissement ROCm SDPA expérimental | Essaie `--rocm-experimental-attention` au lancement, puis compare plusieurs analyses. |
| Avertissement `causal_conv1d` absent | Transformers utilise les opérations PyTorch de référence, fonctionnelles mais plus lentes. Voir les [prérequis officiels du kernel](https://github.com/Dao-AILab/causal-conv1d). |

### Attention ROCm expérimentale et convolutions LFM

Dans l'interface, coche **Attention ROCm expérimentale**, sous le choix du modèle,
avant de cliquer sur **Charger le modèle** ou de lancer une analyse. La case reprend
la valeur du flag CLI ou de la variable d'environnement au démarrage. Elle permet
aussi de désactiver cette valeur avant le chargement. Le choix vaut pour la session
courante et les changements sont consignés dans le journal.

Dès le premier chargement, la case est verrouillée, y compris lors d'un changement
de modèle : PyTorch peut mémoriser le réglage. Pour le modifier, ferme l'app,
relance-la puis choisis la case avant de charger un modèle. Ce réglage concerne
uniquement AMD avec ROCm ; il ne fournit pas les convolutions optimisées de LFM.

Tu peux aussi activer l'option en ligne de commande.
Ferme FastVideo, puis relance dans un nouveau processus :

```powershell
.\.venv\Scripts\python.exe app.py --device rocm --model lfm-3b --rocm-experimental-attention
```

Cette option définit `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` avant le chargement
du moteur. Elle autorise les kernels expérimentaux présents dans ton PyTorch ;
elle ne garantit ni leur sélection ni un gain de vitesse. PyTorch conserve la
sélection SDPA et ses autres chemins disponibles. Aucun paquet supplémentaire
n'est installé. Voir la [sélection des kernels dans PyTorch](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp).

La configuration est fixée au lancement : PyTorch peut mémoriser cette variable
lors du premier appel. Sans l'option, FastVideo conserve la variable d'environnement
existante. Pour revenir au mode précédent, ferme l'app et relance sans l'option ;
si tu as défini la variable toi-même dans PowerShell, retire-la aussi avec
`Remove-Item Env:TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL -ErrorAction SilentlyContinue`.
Les lanceurs acceptent également l'option :

```powershell
.\setup.ps1 -AppArgs @('--model', 'lfm-3b', '--rocm-experimental-attention')
```

```bash
bash setup.sh -- --model lfm-3b --rocm-experimental-attention
```

Le journal affiche PyTorch, HIP et l'architecture de la Radeon. « Autorisés »
décrit la configuration, pas une mesure du kernel effectivement exécuté.
Pour LFM, il indique aussi la disponibilité des kernels de convolution annoncée
par le runtime Transformers chargé. L'avertissement `causal_conv1d_update` est
indépendant : l'option d'attention ne le résout pas. Le repli PyTorch fonctionne.
Les dernières [releases officielles de causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases)
consultées ne proposent pas de wheel Windows/ROCm ; FastVideo n'en installe pas
automatiquement. Une compilation HIP demande une chaîne de build compatible et
une validation sur le GPU concerné. Pour réduire la latence sans cette extension,
essaie LFM 450M ou 1,6B, une image et une réponse courte.

Tu peux comparer le même modèle et la même image synthétique dans deux processus :

```powershell
.\.venv\Scripts\python.exe tools/verify_models.py --models lfm-3b --offline --output artifacts/rocm-default.json
.\.venv\Scripts\python.exe tools/verify_models.py --models lfm-3b --offline --rocm-experimental-attention --output artifacts/rocm-experimental.json
```

Ces rapports contiennent les diagnostics et les durées `cold_seconds` puis
`warm_seconds`. Vérifie aussi la réponse produite. Un premier passage plus lent
peut venir de l'initialisation des kernels ; compare surtout les passages suivants.

Les réponses des modèles peuvent contenir des erreurs. Les mesures publiées décrivent
une vérification d'exécution sur une image synthétique ; elles ne mesurent pas la
fiabilité sur toutes les scènes de webcam.
