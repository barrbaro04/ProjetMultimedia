# Modélisation et Classification de Texte (Émotions)

La partie textuelle de notre projet repose sur deux axes principaux : la **transcription** des dialogues et les **descriptions (captions)** des scènes.

Le principe général de cette modélisation consiste à charger un modèle pré-entraîné sur Hugging Face et à lui ajouter une « tête » (*head*) de classification personnalisée que nous entraînons sur nos données. Les clips sont classés en **8 émotions** : `amusement`, `awe`, `contentment`, `excitement`, `anger`, `disgust`, `fear`, `sadness`.

---

## 1. Choix de l'architecture d'optimisation

Avant d'entraîner les modèles, nous avons défini les composants fondamentaux pour piloter l'apprentissage :

* **L'optimiseur :** nous avons choisi **AdamW**, car il découple proprement la régularisation (*weight decay*) des gradients adaptatifs. Cela améliore la généralisation et limite l'overfitting.
* **La fonction de perte (*Loss*) :** l'entropie croisée (*Cross-Entropy Loss*), adaptée à la classification multi-classe. Elle pénalise fortement les prédictions erronées mais confiantes du modèle.
* **Les métriques d'évaluation :** l'exactitude (*Accuracy*) et le **F1-Macro**. L'accuracy peut avantager les classes majoritaires. Le F1-macro, lui, traite toutes les classes de la même manière : c'est une métrique plus robuste pour vérifier que le modèle ne néglige pas les émotions minoritaires de notre jeu de données.

---

## 2. Préparation des données

### Gestion des transcriptions vides
Une partie des clips ne contient aucune parole (le VAD ne détecte pas de voix, ou Whisper renvoie un texte vide). Entraîner le modèle sur ces clips n'a pas de sens : il devrait deviner une émotion à partir d'une chaîne vide, ce qui ajoute du bruit.

* **Train et dev :** nous **retirons ces clips** de l'entraînement et du dev (le dev sert à choisir le meilleur epoch). Il reste **1411 clips avec texte sur 2208** dans le train, et **281 sur 442** dans le dev.

### Tokenisation
Les textes sont tronqués à **128 tokens** (`truncation=True`), car très peu de phrases dépassent cette limite. Nous n'ajoutons pas de padding à la tokenisation : un `DataCollatorWithPadding` complète dynamiquement chaque batch, ce qui économise de la mémoire.

---

## 3. Itérations et entraînement sur les transcriptions

### Premier modèle : `roberta-base` (approche naïve)
Nous avons commencé par travailler uniquement sur les transcriptions avec un modèle `roberta-base` classique et les hyperparamètres par défaut.

* **Résultat :** le modèle peinait à généraliser et prédisait de manière écrasante les deux classes de différente valance majoritaires  (`anger` et `contentment`).

### Deuxième modèle : introduction du `WeightedTrainer`
Pour contrer le déséquilibre des classes, nous avons implémenté un `WeightedTrainer` personnalisé. Les poids sont calculés avec `compute_class_weight('balanced')` sur le train, puis injectés directement dans la fonction de perte : se tromper sur une émotion rare coûte alors plus cher que se tromper sur une émotion fréquente.

* **Résultat :** malgré cette adaptation, les performances ne se sont pas améliorées de façon significative et le modèle peinait toujours à apprendre efficacement.

### Troisième modèle : `j-hartmann/emotion-english-distilroberta-base`
Partant du constat que le modèle de base comprenait mal les nuances émotionnelles, nous avons opté pour le modèle spécialisé **`j-hartmann/emotion-english-distilroberta-base`**, déjà pré-entraîné sur la détection d'émotions.

Ce modèle a été pré-entraîné sur **7 émotions**, alors que notre tâche en compte **8**. Nous utilisons `ignore_mismatched_sizes=True` : la dernière couche est réinitialisée à 8 sorties, tandis que le reste du réseau conserve ses poids pré-entraînés.

En ajustant finement les hyperparamètres (taux d'apprentissage, gestion du *weight decay* avec AdamW), nous avons obtenu un **F1-macro sur le jeu de développement (*dev*) allant jusqu'à 0.32**.

---

## 4. Stratégie d'entraînement

* **Pondération des classes** : voir le `WeightedTrainer` ci-dessus, avec des poids calculés uniquement sur le train filtré (clips avec texte).
* **Early stopping** : l'entraînement s'arrête si le F1-macro du dev ne s'améliore plus pendant **3 epochs** consécutifs, et le modèle conserve le **meilleur epoch** (`load_best_model_at_end=True`). C'est aussi une forme de régularisation, car on évite de trop coller au train.

Un taux d'apprentissage plus faible limite le risque de « casser » les poids pré-entraînés lors du fine-tuning.

---

## 5. Évaluation et analyse des erreurs

Au-delà de l'accuracy et du F1-macro, nous avons ajouté des vérifications pour comprendre *comment* le modèle se trompe :

* **Matrice de confusion et rapport par classe** : pour voir quelles émotions sont confondues entre elles. [Ex. : `fear` et `sadness` sont souvent confondues ; `awe` est rarement prédite.]
* **Accuracy de la valence** : le modèle distingue-t-il au moins les émotions **positives** (`amusement`, `awe`, `contentment`, `excitement`) des **négatives** (`anger`, `disgust`, `fear`, `sadness`) ? Nous la comparons à une baseline qui prédit toujours la classe majoritaire. [Résultat : X % contre Y %.]
* **Top-k accuracy** (k = 1, 2, 3) : l'émotion correcte est-elle parmi les k plus probables ? Même avec un top-1 faible, un bon top-3 montre que le modèle est « presque juste », ce qui reste utile pour la fusion. [Résultats : top-1 = X, top-2 = Y, top-3 = Z.]
* **Courbes d'apprentissage** : loss train vs dev et métriques dev par epoch, pour détecter l'overfitting.

---

## 6. Modèle sur les descriptions (captions)

Nous avons appliqué la même architecture (`j-hartmann/emotion-english-distilroberta-base`, mêmes hyperparamètres, même pondération des classes) aux **descriptions des scènes**. Contrairement aux transcriptions, elles sont disponibles pour **tous les clips**, et leur rédaction est plus structurée que celle des dialogues.

[Comparer ici le F1-macro dev : descriptions = 0.348, transcriptions = 0.32, et commenter laquelle est la plus informative et pourquoi.]

---

## 7. Sortie pour la fusion multimodale

Chaque modèle texte exporte, pour le dev et le test, un CSV contenant le `clip_id` et les **8 probabilités** (softmax des logits), une colonne par émotion :

```
clip_id,amusement,awe,contentment,excitement,anger,disgust,fear,sadness
```

Exporter des probabilités et non seulement l'émotion prédite permet de pondérer les modalités selon leur confiance lors de la fusion avec le modèle vidéo. Fichiers produits : `probs_transcription_dev.csv`, `probs_transcription_test.csv` (et leurs équivalents `probs_description_*.csv` pour les captions).

---

## 8. Limites et pistes d'amélioration

* Le modèle est **uniquement anglais** : les transcriptions dans une autre langue (Whisper détecte la langue) sont mal comprises. On pourrait les traduire ou les filtrer par langue.
* Les transcriptions sont **courtes et parfois ambiguës** : un dialogue seul ne suffit pas toujours à déterminer l'émotion d'une scène (ton de la voix, image, musique). C'est la motivation de la fusion multimodale.
* Le F1-macro reste modeste (0.32) à cause du **déséquilibre des classes** et de la **subjectivité des annotations**. Pistes : augmentation de données, autres modèles (RoBERTa-large, DeBERTa), fine-tuning en deux temps.

---

## 9. Modèle retenu

Le modèle final pour la partie texte est **`j-hartmann/emotion-english-distilroberta-base`**, fine-tuné sur nos données.

| Caractéristique | Valeur |
|-----------------|--------|
| Modèle de base | `j-hartmann/emotion-english-distilroberta-base` (DistilRoBERTa, pré-entraîné sur 7 émotions) |
| Tête de classification | Réinitialisée à **8 classes** (`ignore_mismatched_sizes=True`) |
| Entrée | Transcriptions (clips avec texte) et descriptions (captions) |
| Longueur maximale | 128 tokens (troncature, padding dynamique par batch) |
| Optimiseur | AdamW |
| Fonction de perte | Cross-Entropy pondérée par classe (`WeightedTrainer`, poids `balanced`) |
| Learning rate | 3e-5 |
| Nombre d'epochs | 8 maximum |
| Batch size | 16 (train) / 32 (évaluation) |
| Weight decay | 0.01 |
| Précision | `fp16` si GPU disponible |
| Régularisation | Weight decay + early stopping (patience = 3) |
| Sélection du modèle | Meilleur F1-macro sur le dev (`load_best_model_at_end=True`) |
| Données d'entraînement | Train filtré : clips avec texte uniquement |
| **F1-macro dev** | **0.32** (transcriptions) / [X] (descriptions) |
| **F1-macro test** | [à compléter] |
| Sortie | Probabilités (softmax) des 8 émotions par clip, uniformes (1/8) si pas de texte |
