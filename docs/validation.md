# Validation et mesures

[← Retour au README](../README.md)

Dernière validation matérielle : **5 octobre 2026**.

## Configuration testée

| Composant | Configuration |
| --- | --- |
| Système | Windows |
| GPU | NVIDIA GeForce RTX 4090, 24 Go |
| Python | 3.12.10 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.18.0 |
| Attention | SDPA |
| Précisions vérifiées | BF16 et FP16 |

## Vérifications effectuées

Les douze modèles ont été chargés depuis leurs révisions fixées et ont généré une
réponse non vide sur une image synthétique en BF16 et FP16. Les huit modèles natifs ont également
été utilisés avec deux images. L'arrêt avant génération et la libération de la
mémoire après changement de modèle ont été vérifiés.

La capture a été testée avec une vraie webcam. Les commandes Tkinter ont été
exercées depuis Python sur le catalogue initial : huit choix, analyse locale, pause, changement pendant
l'inférence, rejet d'un ancien résultat, fermeture et réouverture de la caméra.
L'interface sombre a ensuite été inspectée sur des captures de sa propre fenêtre,
à 1360 × 900 et 1140 × 780, avec un panneau de réglages défilant.

Le menu étendu a été instancié avec ses douze choix et Moondream sélectionné.
FastVLM 1,5B, FastVLM 7B, Qwen3.5 4B et Moondream3 ont également généré leurs réponses
en FP16 avec `--offline`, `HF_HUB_OFFLINE=1` et `TRANSFORMERS_OFFLINE=1`.
Moondream a aussi été vérifié hors ligne en BF16. Son tokenizer séparé est fixé
à une révision, et les tests couvrent la fermeture du flux lors d'une annulation.

La nouvelle interface a été exercée avec LFM 450M sur RTX 4090 : téléchargement
dans un cache vierge, progression réelle sur environ 860 Mio, chargement sans
webcam, analyse ponctuelle, analyses répétées, pause, copie du résultat et consigne
prédéfinie. Le contrôle a également rejeté une réponse d'une ancienne session.
L'image de démonstration est synthétique. Le chargement depuis le cache a été
vérifié lors d'un second passage.
L'ouverture et la fermeture d'une vraie webcam ont également été vérifiées avec
les nouveaux boutons et les FPS mesurés. L'export du journal et son effacement
dans l'interface ont réussi, tout en conservant le fichier de logs persistant.
La capture HD a ensuite été vérifiée : demande 1920×1080 à 25 FPS, résolution
reçue 1920×1080 et cadence annoncée 25 FPS. Une réouverture avec demande 1280×720
à 30 FPS a conservé le modèle ; le pilote ayant maintenu le flux en 1920×1080,
ce choix différent a été signalé. Ces cadences sont celles annoncées par le pilote,
pas une mesure de fluidité réelle. Les tests vérifient aussi que le flux reste
en Full HD et que les images destinées au modèle sont réduites.
L'aperçu agrandi, le miroir, F11 et Échap ont été exercés sur une image Full HD
synthétique. La reprise de l'analyse après réouverture a été vérifiée avec une
capture simulée en conservant l'objet du modèle chargé.
Un test Tkinter ouvre aussi le menu natif `ttk.Combobox` et déclenche la molette
au-dessus de son `popdown` : aucune exception ni défilement du panneau derrière
le menu. Le défilement normal du panneau est ensuite vérifié. Ce test est ignoré
si aucun affichage Tk n'est disponible, notamment sur un runner Linux sans écran.

Les tests couvrent les FPS de capture, la sélection exacte des shards de Moondream,
la réutilisation du tokenizer en cache, l'absence de requête de préparation hors
ligne et la reprise d'une image fraîche après un chargement lent.

## Repères observés sur RTX 4090

Image RGB 640 × 480 contenant un carré rouge, un cercle bleu et du texte.
Consigne : décrire les formes et leurs couleurs en une phrase en français.
Limite : 48 nouveaux tokens, génération déterministe, une image, BF16.
Le temps ci-dessous est celui de la seconde génération, après chargement et
première inférence. Il inclut le prétraitement et la génération ; le GPU est
synchronisé autour de la mesure.

| Modèle | Temps observé | Mémoire allouée après chargement |
| --- | ---: | ---: |
| SmolVLM2 500M | 0,72 s | 0,96 Gio |
| Qwen3.5 0,8B | 0,65 s | 1,63 Gio |
| Qwen3.5 2B | 1,01 s | 4,13 Gio |
| Qwen3.5 4B | 1,90 s | 8,56 Gio |
| MiniCPM-V 4.6 | 0,79 s | 2,47 Gio |
| LFM2.5-VL 450M | 0,23 s | 0,85 Gio |
| LFM2.5-VL 1,6B | 0,25 s | 2,99 Gio |
| LFM2.5-VL 3B | 0,32 s | 5,84 Gio |
| FastVLM 0,5B | 1,34 s | 1,18 Gio |
| FastVLM 1,5B | 1,33 s | 3,16 Gio |
| FastVLM 7B | 1,46 s | 14,54 Gio |
| Moondream3 Preview | 1,54 s | 17,29 Gio |

Il s'agit d'une mesure unique par modèle. Les réponses ont des longueurs différentes.
Les valeurs ne donnent ni un débit à longueur égale, ni un score de qualité.
La mémoire indique l'allocation PyTorch après chargement, hors pic d'inférence.
Les images réelles, la consigne et le nombre de tokens influencent la latence.
Moondream utilise le chemin SDPA non compilé et le raisonnement est désactivé.
Qwen utilise les opérations PyTorch de référence, sans `causal_conv1d` ni
`flash-linear-attention`. Ces mesures ne représentent pas leurs kernels spécialisés.

## Reproduire la vérification des modèles

Après installation des dépendances de l'app :

```powershell
.\.venv\Scripts\python.exe tools/verify_models.py
.\.venv\Scripts\python.exe tools/verify_models.py --precision fp16 --offline
```

Le script construit son image de test, vérifie les réponses et écrit un rapport JSON
dans `artifacts/`. Il peut télécharger plusieurs gigaoctets au premier passage.
Pour tester un seul modèle :

```powershell
.\.venv\Scripts\python.exe tools/verify_models.py --models lfm-450m
```

Pour reproduire les nouvelles entrées :

```powershell
.\.venv\Scripts\python.exe tools/verify_models.py --models fastvlm-1.5b fastvlm-7b qwen-4b moondream3
.\.venv\Scripts\python.exe tools/verify_models.py --models fastvlm-1.5b fastvlm-7b qwen-4b moondream3 --precision fp16 --offline
```

Le mode FP16 est choisi avant le chargement, puis le script vérifie la précision
réelle des paramètres. Moondream nécessite une conversion explicite des poids et
des crops, car son code officiel les initialise en BF16. Les buffers de position
sont reconstruits en pleine précision après cette conversion.

Les tests automatiques GitHub vérifient le code, le choix du runtime et la capture
avec une caméra simulée. Ils s'exécutent sur Windows et Linux sans charger les modèles.

## Scripts de premier clone

Les lanceurs `setup.ps1` et `setup.sh` partagent l'installateur Python.
La préparation complète a été exécutée sous Windows dans un dossier neuf avec
des espaces dans son chemin, avec Python absent du `PATH` : téléchargement local
de uv 0.12.23 et Python 3.12.15, création du venv, installation de PyTorch CPU
2.11.0+cpu et des dépendances, puis vérification réussie.
La préparation d'un environnement NVIDIA existant a conservé PyTorch CUDA 12.8.

Les tests de l'installateur couvrent HX370 / HX470, Ryzen AI Max 385 / 395, les GPU
Radeon correspondants, la priorité au target HIP, les choix de backend,
les contraintes qui préservent PyTorch, les erreurs de pilotes et de pip, ainsi
que le mode de simulation sans création de venv. La CI vérifie également les
scripts PowerShell sur Windows et Bash sur Linux avec les plans CPU et ROCm.
Les paquets ROCm ne sont pas installés sur les runners GitHub.

## AMD Ryzen AI

Le choix ROCm et ses erreurs de configuration ont été testés avec des runtimes simulés.
Les douze modèles ont aussi été exécutés en FP16 sur NVIDIA, la précision choisie pour ROCm.
La détection des familles AMD et les plans `gfx1150`, `gfx1151` et `gfx1152` sont testés.
Les kernels ROCm et les performances des HX370 / HX470 et Ryzen AI Max restent à valider
sur le matériel concerné. Un test FP16 sur NVIDIA ne valide pas les kernels AMD.
Le [guide d'installation](installation.md#amd) fournit le point de départ.
