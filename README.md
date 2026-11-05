# Projet multimédia — Reconnaissance des émotions

- VUKOSAVLJEVIC Petar
- TUGUES SORRIBES Lluc
- GHAFFAR Mohamed
- PECHARMAN Yoris

## 1. Présentation du projet

Ce projet, réalisé dans le cadre du Master 2 SID, consiste à reconnaître l'émotion associée à un extrait de film à partir de trois modalités : le texte, l'audio et la vidéo.

Nous utilisons un sous-ensemble de la partie Movie du corpus EmoVid, comprenant huit émotions : `amusement`, `awe`, `contentment`, `excitement`, `anger`, `disgust`, `fear` et `sadness`.

L'objectif est de construire un modèle pour chaque modalité, puis de combiner leurs prédictions dans un système multimédia. L'idée est d'exploiter des informations complémentaires : ce qui est dit, la manière dont la bande sonore s'exprime et ce qui apparaît à l'image.

Les consignes et la présentation des données sont disponibles dans le [sujet du projet](Data/Challenge2026-M2SID.pdf). Le dossier [Data](Data/) présente le corpus et les fichiers fournis.

## 2. Travail réalisé

**[Analyse des données](Analyse/)**

Nous avons commencé par explorer le corpus pour avoir une idée de ce à quoi ressemble le jeu de données, puis vérifier les annotations et les fichiers associés. Nous avons étudié la répartition des émotions, les durées, la contribution des films et quelques caractéristiques audio et visuelles. Cette étape a permis de mieux comprendre les données et de préparer notre proposition de partition. Les détails de l'analyse et de notre proposition de partition sont présentés dans le dossier [Analyse](Analyse/).

**[Partition commune](Partition/)**

La partition adoptée avec les autres groupes est utilisée pour toutes les modalités. Elle a été choisie pour sa bonne répartition des émotions entre les ensembles d'apprentissage, de développement et de test. Les extraits d'un même film restent dans un seul ensemble afin de limiter les fuites d'information entre l'apprentissage et l'évaluation. Une partition commune permet aussi de comparer et de combiner les prédictions sur les mêmes extraits. Le fichier de partition et son organisation sont présentés dans le dossier [Partition](Partition/).

**[Transcription de l'audio](Transctiption/)**

La transcription repose sur Silero VAD pour repérer les extraits contenant de la parole, puis sur Whisper pour les transcrire. L'idée était de construire une entrée textuelle à partir des dialogues, tout en évitant de transcrire systématiquement les passages sans parole détectée. Les détails de la détection de parole et de la transcription sont disponibles dans le dossier [Transctiption](Transctiption/).

**[Modèles texte](Texte/)**

Pour exploiter le contenu des transcriptions, un modèle de langue préentraîné a été adapté à nos huit émotions. Une expérience sur les descriptions des scènes a également été réalisée. Ces deux sources textuelles sont distinctes : les transcriptions représentent les paroles, tandis que les descriptions résument le contenu de la scène. Les méthodes et les expériences sur le texte sont détaillées dans le dossier [Texte](Texte/).

**[Modèles audio](Audio/)**

Nous avons étudié des caractéristiques acoustiques, puis comparé plusieurs représentations et classifieurs. Le modèle retenu repose sur wav2vec2, déjà adapté à la parole émotionnelle, puis partiellement ajusté sur nos extraits. Cette branche utilise directement la bande sonore, sans transcription ni description textuelle en entrée. La méthode retenue et les autres expériences audio sont détaillées dans le dossier [Audio](Audio/).

**[Modèles vidéo](Video/)**

La branche vidéo utilise CLIP pour extraire des caractéristiques visuelles à partir d'images réparties dans chaque extrait. Le modèle retenu résume ces représentations par leur moyenne et leur écart-type, puis utilise une régression logistique. L'idée est de réutiliser une représentation visuelle préentraînée et d'entraîner un classifieur simple pour les émotions du projet. La méthode retenue et les autres expériences vidéo sont détaillées dans le dossier [Video](Video/).

**[Fusion des modalités](Fusion/)**

Chaque branche produit un vecteur avec huit probabilités d'émotion par extrait. Nous avons comparé plusieurs règles de fusion, dont le vote, de prendre le maximum, la moyenne et la moyenne pondérée. Les prédictions sont reliées par l'identifiant de l'extrait, et les poids de fusion sont choisis sur le développement. L'objectif est de vérifier si les modalités se complètent et si leur combinaison améliore la reconnaissance des émotions. Les règles de combinaison et les expériences de fusion sont détaillées dans le dossier [Fusion](Fusion/).

Les modèles sont évalués principalement avec le F1 macro, qui donne le même poids à chaque émotion. La comparaison des branches et de leur fusion permet ensuite d'examiner les apports de chaque modalité, les erreurs restantes et les limites du système.

## 3. Analyse des résultats

### Comparaison des branches

| Source | F1 macro dev | F1 macro test | Extraits évalués sur test |
|---|---:|---:|---|
| Vidéo | 0,390 | 0,369 | 439, ensemble complet |
| Audio | 0,262 | 0,220 | 439, ensemble complet |
| Texte : transcriptions | 0,188 | 0,208 | 257, avec une transcription disponible |
| Texte : descriptions des scènes | 0,349 | 0,324 | 439, ensemble complet |
| **Fusion des quatre sources** | **0,444** | **0,409** | **439, ensemble complet** |

La **vidéo obtient le meilleur résultat parmi les sources prises séparément**. Sur ces extraits, les images semblent donc apporter des informations utiles pour reconnaître l’émotion de la scène. Les descriptions obtiennent également un résultat intéressant, elles résument les actions et le contenu visuel, ce qui peut aider à identifier l’ambiance de la scène.

Les résultats de l’audio et des transcriptions sont plus faibles. Une explication possible est que les dialogues sont parfois courts ou ambigus et que la bande sonore mélange les voix, la musique et les bruitages. L’émotion attendue ne correspond pas forcément à celle exprimée dans les paroles ou la voix seules.

### Apport de la fusion

La fusion retenue est une **moyenne pondérée**.

Sur test, le F1 macro passe de **0,369 pour la vidéo seule à 0,409 pour la fusion**, soit un gain d’environ **4,1 points de pourcentage** avec les valeurs non arrondies. L’accuracy passe de **41,0 % à 44,6 %**.

Cela montre que les sources peuvent se compléter, même lorsqu’elles ont des performances individuelles différentes. En revanche, leur donner le même poids ne suffit pas : la moyenne simple obtient seulement **0,365 sur test**. La pondération permet ici de conserver une place importante à la vidéo tout en utilisant les autres informations.

L’ajout des descriptions joue un rôle dans le gain observé. La fusion vidéo, audio et transcriptions atteint environ **0,375 sur test**, contre **0,409** avec les quatre sources. Le gain de la fusion complète par rapport à la vidéo est positif dans l’intervalle bootstrap apparié à 95 %, estimé entre **+0,7 et +7,3 points** en rééchantillonnant les films. Ce résultat reste lié à notre partition et aux modèles entraînés.

### Bilan et limites

Les résultats restent modestes. Plusieurs émotions sont encore confondues et les performances varient selon les classes. En vidéo, la peur est la mieux reconnue sur dev, alors que le dégoût et l’excitation sont plus difficiles. En audio, la colère est mieux reconnue que le contentement.

La taille du corpus et nos moyens de calcul ont limité les expériences. C'est possible que si on avait plus de données, on aurait des meilleurs résultats.

Enfin, les descriptions sont des résumés visuels fournis avec le corpus, et non des dialogues obtenus à partir de l’audio. Leur provenance et leur lien avec la production des annotations doivent être pris en compte pour interpréter leur apport.

Ce projet nous a permis de comparer plusieurs façons de représenter une scène et de montrer l’intérêt d’une combinaison pondérée sur notre test. Les analyses détaillées sont disponibles dans les dossiers [Audio](Audio/), [Video](Video/), [Texte](Texte/) et [Fusion](Fusion/).
