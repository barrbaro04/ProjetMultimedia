# Données du projet

## Corpus utilisé

Les données fournies pour le projet proviennent de la partie **Movie** du corpus [EmoVid](https://zane-zyqiu.github.io/EmoVid/). Nous cherchons à reconnaître l'émotion associée à chaque extrait de film à partir de la vidéo, de l'audio et du texte.

L'archive utilisée contient **3 088 extraits issus de 350 films**, répartis entre huit émotions : `amusement`, `awe`, `contentment`, `excitement`, `anger`, `disgust`, `fear` et `sadness`.

## Fichiers fournis

Chaque extrait possède trois fichiers associés :

| Fichier | Contenu |
|---|---|
| Vidéo `.mp4` | Images de la scène, en 1 280 × 720 pixels. |
| Audio `.mp3` | Bande sonore séparée, à 44,1 kHz en stéréo. Elle peut contenir des voix, de la musique et des bruitages. |
| Annotation `.json` | Identifiant, émotion, description de la scène et caractéristiques visuelle de l'extrait. |

Les fichiers sont regroupés par film dans les dossiers `video/`, `audio/` et `annotation/`. Ils partagent le même identifiant : par exemple, `00000_clip_001` désigne un extrait du film `00000`.

## Annotations

| Champ | Signification |
|---|---|
| `video_id` | Identifiant de l'extrait. |
| `emotion` | Émotion à prédire. |
| `caption` | Description textuelle de la scène. |
| `duration` | Durée de l'extrait, en secondes. |
| `brightness` | Luminosité, entre 0 et 1. |
| `colorfulness` | Richesse des couleurs, entre 0 et 1. |
| `hue` | Teinte |

La `caption` décrit la scène et ne correspond pas à une transcription des paroles. Les transcriptions sont produites séparément à partir de l'audio.

## Accès et documentation

Les fichiers audio et vidéo ne sont pas inclus dans ce dépôt. Le [sujet du projet](Challenge2026-M2SID.pdf) contient le lien d'accès aux données ainsi que les références du corpus.

L'analyse des données et notre proposition de partition sont présentées dans [Analyse](../Analyse/). La partition commune utilisée pour les modèles est disponible dans [Partition](../Partition/).
