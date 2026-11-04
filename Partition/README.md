# Partition commune

Le fichier [Group 1.csv](Group%201.csv) contient la partition finale adoptée avec les autres groupes. Elle est utilisée pour les branches audio, vidéo et texte afin de travailler sur les mêmes ensembles et de pouvoir combiner leurs prédictions.

| Ensemble | Films | Extraits |
|---|---:|---:|
| Apprentissage (`train`) | 250 | 2 207 |
| Développement (`dev`) | 50 | 442 |
| Test (`test`) | 50 | 439 |

Tous les extraits d'un même film restent dans un seul ensemble. Cette séparation limite les fuites d'information entre l'apprentissage et l'évaluation.

## Format du fichier

- `film_id` : identifiant du film
- `split` : ensemble auquel appartient le film (`train`, `dev` ou `test`)

