# Analyse des données

Cette première étape sert à comprendre le corpus, à vérifier les fichiers fournis et à préparer une proposition de partition. Nous cherchons notamment à savoir si les émotions et les films sont représentés de manière équilibrée, et quelles caractéristiques pourraient être utiles pour la suite du projet.

## Documents du dossier

- [Rapport d'analyse](Analyse_de_donnees.pdf) : synthèse des observations et proposition de partition finalement retenue par notre groupe.
- [Notebook d'analyse](ProjetMultimedia_Analyse_de_donnee.ipynb) : code, résultats et essais successifs de partition.

## 1. Vérification du corpus

Nous construisons une table contenant une ligne par extrait à partir des annotations JSON. Nous ajoutons l'identifiant du film, obtenu à partir de celui de l'extrait, pour pouvoir regrouper les scènes d'un même film.

Les contrôles portent sur les valeurs manquantes, les identifiants répétés, les durées, les descriptions et la présence des fichiers audio et vidéo associés.

L'archive contient **3 088 extraits provenant de 350 films**. Nous ne détectons aucune valeur manquante dans les annotations, aucun identifiant d'extrait répété, aucune durée nulle ou négative et aucun fichier audio ou vidéo attendu manquant.


## 2. Principales observations

### Répartition des émotions

Nous comptons les extraits par émotion pour repérer les classes les plus et les moins représentées.

![Nombre d'extraits par émotion](photo/repartition_emotions.png)

Les huit émotions sont présentes, mais leurs effectifs sont déséquilibrés. La colère (`anger`) compte **677 extraits**, contre **244** pour l'amusement (`amusement`).

Nous examinons aussi le nombre de films contenant chaque émotion : `anger` apparaît dans 223 films et `amusement` dans 96. On remarque qu'un même film peut contenir plusieurs émotions.

Ces observations nous conduisent à comparer les proportions d'émotions dans les partitions candidates et à tenir compte du déséquilibre lors de l'évaluation des modèles.

### Durées et nombre d'extraits par film

On regarde aussi la durée des extraits. On remarque que la majorité est courte, avec quelques extraits plus longs.

Aussi, les films fournissent entre **2 et 22 extraits**. Fixer le nombre de films par ensemble ne garantit donc pas des effectifs d'extraits proportionnels. Nous prenons aussi cet aspect en compte lors de la comparaison des partitions.

### Premières observations audio et vidéo

Nous comparons la luminosité et la richesse des couleurs selon les émotions. Leurs distributions se recouvrent largement.

Le rapport présente aussi une comparaison du niveau RMS des bandes sonores, avec un recouvrement important entre les classes. Le RMS dépend également du volume d'enregistrement, de la musique et des bruitages.
Les graphiques complémentaires et les exemples sont présentés dans le [rapport](Analyse_de_donnees.pdf).

## 3. Notre proposition de partition

### Principe

Nous conservons tous les extraits d'un même film dans un seul ensemble. Cela limite "la triche" pendant l'apprentissage.

Chaque candidat respecte les nombres de films : **250 pour l'apprentissage, 50 pour le développement et 50 pour le test**.

### Recherche des candidats

Nous construisons une table des effectifs par film et par émotion. Nous utilisons tous les effectifs du film, plutôt que de lui attribuer seulement son émotion majoritaire.

Nous comparons ensuite **50 000 répartitions**, obtenues avec les graines de 0 à 49 999. Pour chaque graine, nous mélangeons les films puis formons les trois ensembles.

Le critère principal est l'écart moyen absolu entre les proportions d'émotions de chaque ensemble et celles du corpus complet. Plus cet écart est faible, plus les distributions sont proches.

Le critère secondaire décrit l'équilibre du nombre d'extraits.

### Comparaison et proposition du groupe

Notre groupe propose finalement la **graine 17630**.

Nous vérifions que les nombres de films sont respectés, qu'aucun film n'est partagé entre les ensembles, que tous les extraits sont affectés et que les huit émotions sont présentes dans chaque ensemble.

Cette proposition se distingue de la partition commune finalement adoptée avec les autres groupes. La partition réellement utilisée pour les modèles est documentée dans [Partition](../Partition/).

## 4. Bilan pour la suite du projet

L'analyse montre un déséquilibre des émotions, des contributions différentes selon les films et des durées variables. Elle justifie une séparation par film et une vérification des distributions après partition.

Les premières caractéristiques audio et visuelles montrent des différences, mais aussi un fort recouvrement entre les émotions.
