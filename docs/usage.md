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
| **Analyser en direct / Mettre en pause** | Lance les analyses répétées ou arrête les nouvelles générations. |
| **Analyser une image** | Analyse ponctuellement la dernière image de la webcam, puis reste en pause. |
| **Copier** | Copie la dernière description dans le presse-papiers. |
| **Exporter…** | Enregistre le journal visible dans un fichier `.log` ou `.txt`. |
| **Effacer** | Vide le journal affiché ; le fichier persistant est conservé. |

Le panneau de gauche permet de régler l'intervalle, le nombre d'images et la
longueur maximale de la réponse pendant l'utilisation. Les nouvelles valeurs
s'appliquent à l'analyse suivante. Quatre consignes prêtes à l'emploi sont proposées,
et le texte reste librement modifiable. Le panneau défile si la fenêtre est réduite.

### Vitesse et téléchargement

- **Caméra** : FPS mesurés à partir des images reçues pendant les deux dernières secondes.
- **Dernière analyse** : durée du prétraitement et de la génération, avec synchronisation GPU.
- **Rythme observé** : réponses par minute calculées sur les dernières réponses, intervalle compris.
- **Moteur** : CUDA, ROCm ou CPU ; le nom du GPU figure sous la réponse.

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
| `--model` | `smol` | Choix initial du modèle ; il reste modifiable dans la fenêtre. |
| `--camera` | `0` | Numéro de webcam, de 0 à 9. |
| `--device` | `auto` | `auto`, `cuda`, `rocm` ou `cpu`. |
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
en RGB et réduite à 640 × 480 au maximum, avec conservation des proportions.

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

Les réponses des modèles peuvent contenir des erreurs. Les mesures publiées décrivent
une vérification d'exécution sur une image synthétique ; elles ne mesurent pas la
fiabilité sur toutes les scènes de webcam.
