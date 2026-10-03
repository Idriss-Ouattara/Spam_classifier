# Spam SMS Classifier : deux approches from scratch

Classifieur de spam SMS avec deux algorithmes implémentés à la main (sans scikit-learn) : une **régression logistique** et un **Naive Bayes**, entraînés et évalués sur le même jeu de données.

## Objectif

Ce projet fait partie d'une série d'implémentations from scratch (régression logistique, Naive Bayes, HMM, correcteur orthographique…) réalisées pour comprendre de l'intérieur les mécanismes des algorithmes classiques de NLP avant d'utiliser des librairies.

## Données

`Spam_SMS.csv` : **5 574 messages** en anglais, dont 4 827 `ham` (86,6 %) et 747 `spam` (13,4 %). Colonnes attendues : `Class` (`spam` / `ham`) et `Message`. Le jeu est **déséquilibré**, ce qui compte pour l'interprétation des résultats (voir plus bas).

## Prétraitement commun

- mise en minuscules ;
- découpage en mots sur les espaces ;
- suppression des stopwords anglais (NLTK) et des tokens de ponctuation isolés (`!`, `,`…) ; la ponctuation collée à un mot (`now!`) est conservée ;
- split **80 % / 20 %** train / test, dans l'ordre du fichier (4 459 / 1 115 messages).

## `naive_bayes.py` : Naive Bayes

- **Comptage** : fréquence de chaque mot par classe (`vocab_spam`, `vocab_ham`).
- **Lissage de Laplace** : `+1` sur chaque compte, pour qu'un mot absent d'une classe n'annule pas tout le score (zero probability problem).
- **Priors** : `log(n_classe / n_total)`, pour tenir compte du déséquilibre spam/ham au lieu de supposer un 50/50.
- **Log-probabilités** : somme des logs plutôt que produit, pour éviter l'underflow sur les messages longs.

```bash
python naive_bayes.py --data Spam_SMS.csv
```

Au premier lancement, NLTK télécharge la liste des stopwords (connexion internet nécessaire).

## `spam_classifier.py` : régression logistique

- **Vectorisation** : bag-of-words binaire (1 si le mot du vocabulaire apparaît dans le message, 0 sinon).
- **Modèle** : `z = w·x + b`, puis `sigmoid(z)` pour la probabilité.
- **Entraînement** : descente de gradient stochastique, poids mis à jour après chaque exemple.
- **Évaluation** : accuracy sur train et test (seuil à 0,5).

```bash
python spam_classifier.py --data Spam_SMS.csv --lr 0.1 --epochs 10
```

## Résultats

Évaluation sur les 1 115 messages de test (145 spams, 970 hams), avec les paramètres par défaut (`--lr 0.1 --epochs 10`, vocabulaire de 11 729 mots pour la régression logistique) :

| Mesure | Naive Bayes | Régression logistique |
|---|---|---|
| Accuracy (test) | 96,86 % | **98,12 %** |
| Accuracy (train) | non mesurée | 99,82 % |
| Précision (spam) | 83,1 % | **100 %** |
| Rappel (spam) | **95,2 %** | 85,5 % |
| Faux positifs (hams classés spam) | 28 | **0** |
| Faux négatifs (spams ratés) | **7** | 21 |

Référence « toujours ham » : 87,0 % d'accuracy. Un classifieur qui ne détecte aucun spam atteint déjà 87 %, donc l'accuracy seule est trompeuse sur ce jeu déséquilibré (13 % de spams) ; la précision et le rappel sur la classe spam sont plus parlants.

**Lecture des résultats.** Les deux modèles se trompent de façon opposée :
- la régression logistique ne classe **jamais** un message légitime en spam, mais laisse passer 14,5 % des spams ;
- le Naive Bayes détecte presque tous les spams, mais bloque à tort 28 messages légitimes.

Le meilleur choix dépend du coût de l'erreur : pour un filtre anti-spam grand public, bloquer un vrai message est en général plus grave que laisser passer un spam, ce qui favorise la régression logistique. L'écart entre accuracy d'entraînement (99,82 %) et de test (98,12 %) indique un léger surapprentissage.

## Installation

```bash
pip install -r requirements.txt
```

## Limites connues

- **Mots inconnus** (Naive Bayes) : un mot absent du vocabulaire d'entraînement n'est pas ignoré. Il reçoit `1 / (total_classe + vocabulaire)` dans chaque classe, une valeur légèrement plus élevée pour la classe qui a le moins de mots (le spam), ce qui crée un petit biais.
- **Split séquentiel** : pas de shuffle ni de stratification, donc le résultat dépend de l'ordre du fichier.
- **Tokenisation simple** : découpage sur les espaces, sans traitement de la ponctuation attachée aux mots.
- **Régression logistique** : descente de gradient stochastique pure (pas de vectorisation NumPy) : environ 1 min 20 pour 10 epochs sur 11 729 features (mesuré en environnement de test) ; pas de régularisation, avec un léger surapprentissage.
- Évaluation sur un seul split : pas de validation croisée.

## Pistes d'amélioration

- Ignorer les mots hors vocabulaire dans le Naive Bayes.
- Mélanger les données (shuffle) et utiliser une validation croisée.
- Ajuster le seuil de décision de la régression logistique (0,5 par défaut) pour regagner du rappel sur les spams, au prix de quelques faux positifs.
- Ajouter de la régularisation (L2) à la régression logistique.
- Vectoriser la régression logistique avec NumPy pour accélérer l'entraînement.


Projet personnel réalisé par Issouf Idriss Ouattara, étudiant en Master 2 NLP UGA
