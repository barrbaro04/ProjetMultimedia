# Pipeline de Transcription et Traitement des Données Audio

Après avoir fini l'analyse des données, nous avons voulu transcrire les données audio pour prédire les émotions à partir d'un modèle textuel. 

## 1. Première approche (Approche naïve)
La première idée était d'appliquer un modèle disponible sur Hugging Face déjà entraîné sur cette tâche sur tout l'ensemble de données. 
Donc, nous avons pris **Whisper-Small (OpenAI)** et nous avons transformé tous les fichiers audio en fichiers textuels. En regardant les résultats, nous nous sommes aperçus qu'il y avait des données aberrantes. Nous avons donc décidé d'implémenter un post-traitement suivant 2 règles :
* Réduire les mots ou groupes de mots répétés plus de 3 fois de suite (ex. remplacer "Yeah Yeah Yeah Yeah" par un seul "Yeah").
* Ne conserver que les caractères alphabétiques (avec accents), numériques et la ponctuation standard pour éliminer le *gibberish* généré sur le bruit pur.

Cette méthode nous a permis de transcrire la grande majorité des audios, mais de manière très lente, et nous avons détecté des textes retranscrits de manière erronée. 

## 2. Deuxième approche (Pipeline optimisé)
Suite aux limites de la première méthode, nous avons décidé d'implémenter une stratégie en deux étapes :

* **Premièrement :** Appliquer sur les données le modèle **Silero VAD** pour préclassifier les extraits audio qui contiennent de la parole et ceux qui n'en contiennent pas. Si l'extrait contient moins de 0,5 seconde de voix ou un taux de parole insuffisant, le clip est immédiatement classé comme `[non_speech]`.
* **Deuxièmement :** Pour les clips contenant de la parole, appliquer le modèle **Faster-Whisper** avec trois garde-fous natifs :
  * **`vad_filter=True`** : Filtre automatiquement les parties silencieuses ou purement musicales de l'audio pendant l'inférence.
  * **`no_speech_threshold=0.6`** : Si la probabilité qu'il n'y ait pas de parole est supérieure à 60 %, Whisper renvoie une chaîne vide au lieu d'inventer du texte.
  * **`compression_ratio_threshold=2.4`** : Mesure la répétition du texte généré. Les hallucinations en boucle (comme *"music playing music playing..."*) ont un ratio de compression très élevé : Whisper détecte la boucle et rejette automatiquement la séquence.

## Bilan des résultats
Sur les **3088 clips** au total, Silero VAD a détecté que **1117** ne contenaient pas de parole. 

Parmi les **1971 clips** avec de la parole, Faster-Whisper a réussi à en transcrire **1970** grâce aux trois garde-fous. Ainsi, sur environ deux tiers des données, nous disposons de transcriptions fiables prêtes pour la suite du pipeline d'entraînement. 
