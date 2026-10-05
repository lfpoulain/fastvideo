# Utilisation

[← Retour au README](../README.md)

## La fenêtre

1. Choisis un des huit modèles dans le menu.
2. Clique sur **Ouvrir la webcam**. Le numéro `0` désigne la caméra par défaut.
3. Clique sur **Analyser en direct** pour charger le modèle et commencer.
4. Modifie la consigne pour demander une description, un comptage ou la lecture d'un texte.
5. Clique sur **Mettre en pause** pour arrêter les nouvelles analyses.

Le panneau de droite affiche la dernière réponse, le GPU utilisé, le temps
d'inférence et l'heure de la réponse. La fermeture de la webcam arrête l'analyse.
Changer de modèle annule la génération en cours, puis libère le précédent modèle.
Un téléchargement ou un chargement initial doit se terminer avant le changement effectif.

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
| `--frames` | `1` | Jusqu'à 3 images récentes ; FastVLM utilise toujours la dernière. |
| `--max-tokens` | `100` | Limite de génération, de 1 à 512 tokens. |
| `--offline` | désactivé | Utiliser seulement les fichiers déjà téléchargés. |
| `--image` | — | Analyser un fichier image sans ouvrir la fenêtre ni la webcam. |
| `--list-models` | — | Afficher les identifiants et les dépôts des modèles. |

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
