# Fusion multimodale : démarche, méthodologie et résultats

Nous combinons par **fusion tardive** les probabilités de quatre sources : vidéo, audio, transcription des dialogues et description textuelle de la scène. Six règles de combinaison, avec et sans masque de disponibilité, sont comparées ; le choix se fait sur le dev et le test n'est mesuré qu'une fois.

- Méthode retenue : **moyenne pondérée** (vidéo 0,6 ; audio 0,1 ; transcription 0,2 ; description 0,1), sans masque.
- **F1 macro de 40,9 % sur le test**, contre 36,9 % pour la vidéo seule : **+4,1 points, gain significatif** (IC 95 % [+0,7 ; +7,3] points).
- Les règles à poids égaux (vote, max, moyenne simple) restent sous la vidéo seule : la pondération est indispensable.
- C'est surtout la **description** qui fait la différence, avec un point de vigilance sur un possible biais commun avec les étiquettes (voir § 6).

## 1. Point de départ : ce que fournissent les branches

La fusion ne réentraîne aucun modèle : elle reprend les sorties produites par chaque membre du groupe. Chaque branche fournit, pour chaque clip, un vecteur de 8 probabilités (une par émotion de Mikels), identifié par son `video_id`. Toutes utilisent la même partition par film (`Group_1.csv`, repris dans `dataset.csv` : 250 / 50 / 50 films, soit 2 207 / 442 / 439 clips), si bien qu'aucun film n'apparaît dans deux ensembles.

| Branche | Modèle | Couverture | F1 dev | F1 test |
|---|---|---|---|---|
| Vidéo | CLIP ViT-L/14 gelé, 8 images, moyenne + écart-type, régression logistique | 100 % des clips | 0,390  | 0,369 |
| Audio | wav2vec2 pré-entraîné sur MSP-Podcast, fine-tuning partiel | 100 % (inutile sans parole) | 0,262 | 0,220 |
| Transcription | DistilRoBERTa émotion (j-hartmann) fine-tunée sur les dialogues Whisper | 64 % (clips parlés) | 0,188 | 0,208 |
| Description | même modèle, fine-tuné sur la description textuelle de chaque scène | 100 % des clips | 0,349  | 0,324  |
| Repère | hasard (8 classes équiprobables) | — | ≈ 12,5 % | ≈ 12,5 % |

*Tableau 1 : entrées de la fusion. Les scores vidéo et audio, recalculés à partir des fichiers des équipes, sont identiques à ceux annoncés. Les deux modèles texte ont été relancés pour exporter leurs probabilités ; avec environ 1 400 exemples d'entraînement, deux exécutions identiques peuvent différer de plusieurs points.*

Avant de fusionner, nous avons audité les fichiers des branches :

- les scores annoncés sont exacts et toutes les branches utilisent bien le même découpage ;
- un doublon de fichier (`00273_clip_005 (1)`) est écarté ;
- 17 transcriptions sont des hallucinations de Whisper (caractères japonais, persans…) et 68 font deux mots ou moins ;
- **l'audio est au niveau du hasard sur les clips sans parole** (F1 ≈ 15 %, contre ≈ 20–25 % avec parole) : c'est ce qui motive le masque (§ 3.2).

## 2. Démarche suivie

1. Audit des sorties de chaque branche et harmonisation du format (`video_id` + 8 colonnes, mêmes noms d'émotions).
2. Construction d'une table de référence des 3 088 clips à partir de `dataset.csv` (split, émotion, parole selon Silero VAD, transcription exploitable), avec vérification automatique de sa cohérence avec `Group_1.csv` et `vad_results.json`.
3. Première fusion audio + vidéo.
4. Ajout de la transcription (gain non significatif : 37,6 % contre 36,9 %), puis de la description, qui change nettement le résultat.
5. Ablation source par source et comparaison avec la littérature.

Tout est regroupé dans un notebook unique (`notebook_fusion.ipynb`) qui intègre automatiquement une source dès que ses fichiers sont présents.

## 3. Méthodologie

### 3.1 Une fusion tardive

On distingue classiquement trois familles de fusion :

- **précoce** : on concatène les représentations des modalités avant un classifieur unique ;
- **intermédiaire** : les modèles échangent de l'information pendant l'apprentissage (par exemple par attention croisée) ;
- **tardive** : chaque modèle reste indépendant et l'on combine leurs sorties.

Nous avons retenu la fusion tardive pour trois raisons : les branches n'exportent que des probabilités (pas d'embeddings) ; une source absente se gère simplement en la retirant du calcul ; et les règles n'ont que 0 à 3 paramètres, ce qui limite le surapprentissage avec environ 440 clips de dev.

Schéma : chaque source (vidéo, audio, transcription, description) produit 8 probabilités par clip → alignement par `video_id` (+ masque éventuel) → combinaison (vote, max, moyenne, moyenne pondérée) → émotion = argmax des probabilités fusionnées.

### 3.2 Masque de disponibilité

Le masque fait voter une source seulement lorsqu'elle est fiable pour le clip :

- l'**audio** si Silero VAD détecte de la parole (détection faite par l'équipe transcription) ;
- la **transcription** si elle est exploitable (au moins deux mots, moins de 30 % de caractères non latins).

La vidéo et la description existent pour tous les clips. Chaque règle est évaluée avec et sans masque ; c'est le dev qui tranche.

### 3.3 Règles de combinaison

Pour un clip, notons *p_m* le vecteur de probabilités de la source *m* et *M* l'ensemble des sources disponibles. La prédiction est l'émotion *k* qui maximise le score fusionné :

- **vote majoritaire** : chaque source vote pour argmax_k *p_m(k)* ; les égalités sont départagées par la probabilité moyenne ;
- **max** : score(k) = max sur *m* de *p_m(k)*, c'est-à-dire que la source la plus confiante décide ;
- **moyenne** : score(k) = moyenne des *p_m(k)* (l'idée de départ du groupe) ; variante **géométrique** : moyenne des log-probabilités ;
- **moyennes pondérées** : mêmes formules avec un poids *w_m* par source (somme = 1, renormalisée clip par clip selon les sources présentes). Avec quatre sources, les poids sont cherchés sur une grille au pas de 0,1 (220 combinaisons ; pas de 0,05 avec deux ou trois sources) pour limiter le surapprentissage du dev.

### 3.4 Protocole d'évaluation

Le protocole reprend celui de l'équipe vidéo.

- **Tous les choix** (règle, masque, poids) se font sur le **dev** ; le **test** n'est mesuré qu'à la fin.
- La mesure principale est le **F1 macro**, adapté au déséquilibre des classes (colère 22 %, amusement 8 %) ; on rapporte aussi l'exactitude et la valence (positif / négatif).
- L'incertitude est estimée par un **bootstrap sur les films** (1 000 tirages) : les clips d'un même film sont corrélés, rééchantillonner les clips sous-estimerait la variance.
- Le gain d'une méthode sur la vidéo seule est mesuré sur les *mêmes* tirages (comparaison appariée).

## 4. Résultats

### 4.1 Comparaison des méthodes

| Méthode | Masque | F1 dev | F1 test | IC 95 % test | Exactitude test |
|---|---|---|---|---|---|
| Vidéo seule | — | 39,0 | 36,9 | [30,8 ; 42,1] | 41,0 |
| Audio seule | — | 26,2 | 22,0 | [17,0 ; 26,0] | 24,1 |
| Description seule | — | 34,9 | 32,4 | [26,9 ; 36,6] | 32,8 |
| Vote majoritaire | sans | 41,0 | 35,7 | [29,6 ; 39,9] | 39,0 |
| Vote majoritaire | avec | 42,5 | 34,1 | [28,1 ; 38,4] | 37,1 |
| Max | sans | 36,6 | 35,0 | [29,4 ; 39,4] | 36,7 |
| Max | avec | 37,0 | 35,5 | [29,9 ; 39,6] | 37,1 |
| Moyenne | sans | 41,5 | 36,5 | [30,7 ; 40,3] | 39,4 |
| Moyenne | avec | 42,1 | 34,9 | [29,1 ; 39,0] | 37,8 |
| Moyenne géométrique | sans | 41,0 | 36,9 | [31,0 ; 41,6] | 40,5 |
| Moyenne géométrique | avec | 42,6 | 36,5 | [30,2 ; 41,3] | 39,9 |
| **Moyenne pondérée (retenue)** | **sans** | **44,4** | **40,9** | **[34,2 ; 46,8]** | **44,6** |
| Moyenne pondérée | avec | 44,2 | 40,1 | [33,4 ; 45,8] | 44,0 |
| Moyenne géométrique pondérée | sans | 44,0 | 38,3 | [32,6 ; 43,2] | 42,1 |
| Moyenne géométrique pondérée | avec | 44,2 | 39,2 | [33,0 ; 44,3] | 43,5 |

*Tableau 2 : toutes les combinaisons testées (en %). La méthode retenue est celle qui a le meilleur F1 sur le dev ; les scores de test sont donnés pour toutes, mais n'ont servi à aucun choix.*

**Lecture.** Les règles à poids égaux (vote, max, moyennes simples) restent **sous la vidéo seule ou à son niveau** : elles donnent autant de voix aux sources faibles (audio 22 %, transcription 21 %) qu'à la meilleure. Seules les **moyennes pondérées** la dépassent. La méthode retenue atteint **40,9 %** contre 36,9 % : un gain de **+4,1 points** dont l'intervalle de confiance apparié, de +0,7 à +7,3 points, ne contient pas zéro (la fusion fait mieux dans 99 % des tirages). L'exactitude passe de 41,0 % à 44,6 % et la valence de 68,6 % à 72,4 %. Sur le dev, le gain (+5,4 points) est optimiste puisque les poids y ont été réglés. Le masque n'apporte rien de net : avec masque, la même règle donne 44,2 % sur le dev (contre 44,4 % sans), on garde donc la version la plus simple.

### 4.2 Apport de chaque source (ablation)

Pour isoler l'effet de chaque source, nous refaisons la fusion pondérée en ne gardant qu'une partie d'entre elles (poids réglés sur le dev pour chaque combinaison).

| Sources | Poids (dev) | F1 test | Gain vs vidéo | IC 95 % du gain |
|---|---|---|---|---|
| Vidéo seule | — | 36,9 | — | — |
| V + A | 0,95 / 0,05 | 37,6 | +0,7 | [+0,0 ; +1,4] |
| V + T | 0,75 / 0,25 | 37,7 | +0,8 | [−1,1 ; +2,8] |
| V + D | 0,85 / 0,15 | 38,6 | +1,8 | [−0,7 ; +4,3] |
| V + A + T | 0,55 / 0,25 / 0,20 | 37,5 | +0,7 | [−2,1 ; +2,8] |
| V + A + D | 0,60 / 0,20 / 0,20 | 39,1 | +2,3 | [−1,7 ; +5,9] |
| V + T + D | 0,85 / 0,00 / 0,15 | 38,6 | +1,8 | [−0,7 ; +4,3] |
| **V + A + T + D** | **0,60 / 0,10 / 0,20 / 0,10** | **40,9** | **+4,1** | **[+0,7 ; +7,3]** |

*Tableau 3 : V = vidéo, A = audio, T = transcription, D = description. F1 et gains en points de pourcentage. Pour V + T + D, la transcription reçoit un poids nul sur le dev : le résultat est donc identique à V + D.*

Ajouter la seule **transcription** à la vidéo ne change presque rien (+0,8 point), comme l'audio. La **description** apporte davantage (+1,8 points ; +2,3 points en y ajoutant l'audio). Mais seule la combinaison des **quatre sources** dépasse la vidéo de façon significative (+4,1 points, IC 95 % [+0,7 ; +7,3]) : chacune corrige une partie des erreurs des autres, à condition d'être pondérée. Les poids retenus donnent la plus grande part à la vidéo (0,6), puis à la transcription (0,2), à l'audio et à la description (0,1 chacune). Ces poids mesurent la complémentarité avec la vidéo, pas la qualité de chaque source prise seule.

### 4.3 Ce que la fusion change

Par rapport à la vidéo seule, la fusion corrige **27 clips** et n'en casse que **11** sur 439. Toutes les émotions progressent ou restent stables :

| Émotion | F1 vidéo seule | F1 fusion | Écart |
|---|---|---|---|
| Amusement | 33,8 | 39,0 | +5,2 |
| Émerveillement | 31,9 | 39,5 | +7,6 |
| Contentement | 44,4 | 48,3 | +3,8 |
| Excitation | 15,2 | 19,5 | +4,3 |
| Colère | 52,6 | 58,8 | +6,2 |
| Dégoût | 27,8 | 32,9 | +5,1 |
| Peur | 52,0 | 51,9 | −0,0 |
| Tristesse | 37,2 | 37,5 | +0,3 |

*Tableau 4 : F1 par émotion sur le test (en %).*

Sur les 180 clips **sans parole**, le F1 passe de 28,4 % à 34,4 % : la transcription n'existe pas pour ces clips et la description est la seule source textuelle disponible ; sans elle, la fusion n'y atteint que 30,1 %. Sur les 259 clips avec parole, le F1 passe de 38,3 % à 40,4 %.

| Sur les clips avec transcription | Transcription | Description | Vidéo | Fusion des 4 |
|---|---|---|---|---|
| dev (281 clips) | 18,8 % | 24,9 % | 34,2 % | 37,5 % |
| test (257 clips) | 20,8 % | 32,8 % | 38,0 % | 40,1 % |

*Tableau 5 : comparaison à périmètre égal. La transcription n'existe que pour les clips parlés, on compare donc les sources sur ce sous-ensemble.*

## 5. Positionnement par rapport aux travaux existants

La démarche s'appuie sur l'état de l'art réalisé en amont du projet. Aucun travail publié ne traite exactement notre cas (8 classes de Mikels, extraits de films anglophones, étiquettes produites en majorité par un modèle vision-langage) ; le papier EmoVid ne publie d'ailleurs aucune baseline de classification.

| Travail | Ce qu'ils ont fait | Ce que nous en reprenons ou comparons |
|---|---|---|
| MER 2024 (Lian et al., 2024) | Encodeurs pré-entraînés gelés par modalité, moyenne temporelle, fusion par attention sur les vecteurs de modalité ; Silero VAD pour repérer la parole. Trimodal : +3,3 points (WAF) sur la meilleure modalité. | Même logique d'encodeurs gelés (CLIP côté vidéo) et même VAD. Faute d'embeddings, nos poids sont fixes et réglés sur le dev au lieu d'être appris par attention. Notre gain (+4,1 points) est du même ordre. |
| Soul AI, gagnant MER 2024 (Qi et al., 2024) | Modality dropout (p = 0,3) pendant l'entraînement de la fusion : +0,6 point. | Non applicable (notre fusion n'est pas entraînée) ; le masque joue un rôle voisin face aux sources manquantes. |
| Zhao, Wang & Wang, 2022 (IEMOCAP) | Fusion tardive par somme de logits 75,8 %, contre 74,6 % en fusion précoce par co-attention et 71,3 % en simple concaténation. | Justifie le choix d'une fusion tardive. Notre moyenne géométrique (somme de log-probabilités) est l'équivalent de leur somme de logits. |
| Li, Bell & Lai, 2024 | Les erreurs de transcription coûtent ≈ 10 points au texte seul, ≈ 4 points une fois l'audio ajouté. | Nos transcriptions Whisper de films sont bruitées : cohérent avec la faiblesse de la branche transcription. |
| Barański et al., 2025 | Whisper produit du texte sur 40 % des segments sans parole. | Motive le VAD en amont (équipe transcription) et notre filtre des transcriptions inexploitables. |
| VAANet (Zhao et al., 2020) | VideoEmotion-8 (8 classes, vidéos web) : la vision domine, l'audio n'ajoute que +0,8 à +0,9 point. | Même constat pour l'audio chez nous (+0,7 point) : sur ce type de contenu, la vidéo porte la décision. |
| MELD (Poria et al., 2019) | Extraits de *Friends* : le texte domine, la fusion gagne ≈ 3 points. | Contre-exemple : sur une série très dialoguée, les dialogues sont forts ; sur nos extraits de film, ils le sont peu. |
| EmoVid (2025) | Étiquettes produites en majorité par NVILA-Lite-2B, légendes par NVILA-8B-Video : risque de fuite si ces légendes servent d'entrée. | Point de vigilance pour notre source « description » (§ 6). |

*Tableau 6 : travaux de référence. Chiffres tels que rapportés par leurs auteurs ; les corpus et métriques diffèrent, on compare donc des tendances, pas des valeurs.*

Au sein du groupe, la fusion reprend directement : la partition par film de l'équipe analyse, la détection de parole (Silero VAD) et les transcriptions (Faster-Whisper) de l'équipe transcription, les probabilités des équipes vidéo et audio sans modification, les deux modèles de l'équipe texte, et le protocole d'évaluation de l'équipe vidéo (choix sans regarder le test, intervalles de confiance par bootstrap sur les films).

## 6. Limites et pistes

- **Risque de fuite par la description** : les descriptions sont générées par un modèle vision-langage à partir des images, et la majorité des étiquettes d'EmoVid l'a été par un modèle de la même famille (NVILA). Si les deux sont proches, une partie du gain peut refléter leurs biais communs plutôt que l'émotion réelle. Sans la description, la fusion (V + A + T) reste au niveau de la vidéo seule : il faut donc vérifier quel modèle a produit les descriptions.
- **Peu de données pour régler la fusion** : avec ≈ 440 clips de dev et 3 poids libres, le score du dev est optimiste ; seul le test fait foi.
- **Branches texte instables** : avec environ 1 400 exemples, deux entraînements identiques peuvent différer de plusieurs points.
- **Probabilités non calibrées** : les modèles n'ont pas la même calibration ; les poids en compensent une partie.
- **Pistes** : un méta-classifieur (stacking) entraîné sur des probabilités *out-of-fold* de toutes les branches ; une fusion par attention avec modality dropout à partir des embeddings, comme MER 2024 ; une calibration des probabilités (température) avant combinaison.

## Conclusion

La fusion tardive pondérée des quatre sources améliore significativement la vidéo seule (40,9 % contre 36,9 % de F1 macro sur le test, +4,1 points). Deux leçons : une moyenne à poids égaux ne fonctionne pas quand les sources sont de qualité inégale, il faut pondérer ; et ce sont les sources complémentaires, description en tête, qui permettent de dépasser la vidéo, l'audio et la transcription seuls n'apportant presque rien. Le gain lié à la description reste à confirmer au regard du risque de biais commun avec les étiquettes.

## Reproduire les résultats

Le notebook `notebook_fusion.ipynb` est fourni exécuté, avec toutes ses sorties. Pour le relancer (par exemple sur Colab), placer dans un même dossier :

- `dataset.csv` (référence du groupe : `video_id`, `label`, `split`, `caption`, `transcript`, `has_speech`) ;
- les probabilités dev et test de chaque branche : `probs_video_{dev,test}.csv`, `probabilites_audio_{dev,test}.csv`, `probs_transcription_{dev,test}.csv`, `probs_description_{dev,test}.csv` ;
- facultatif : `Group_1.csv` et `vad_results.json`, pour vérifier la cohérence du split et de la détection de parole.

Puis adapter `DATA_DIR` dans la première cellule. Les sorties (tableaux récapitulatifs, probabilités fusionnées, figures) sont écrites dans `outputs_fusion/`.

## Sources

- Lian et al., « MER 2024 », arXiv:2404.17113
- Qi et al., « Multimodal Emotion Recognition with Vision-language Prompting and Modality Dropout », arXiv:2409.07078
- Zhao, Wang & Wang, « Multi-level Fusion of Wav2vec 2.0 and BERT », arXiv:2207.04697
- Li, Bell & Lai, « Speech Emotion Recognition with ASR Transcripts », arXiv:2406.08353
- Zhao et al., « VAANet », arXiv:2003.00832
- Poria et al., « MELD », arXiv:1810.02508
- Barański et al., 2025, hallucinations de Whisper sur la non-parole
- EmoVid, arXiv:2511.11002
