Spam SMS Classifier — deux approches from scratch

Classifieur de spam SMS avec deux algorithmes implémentés à la main (sans scikit-learn) :
une **régression logistique** et un **Naive Bayes**, tous deux entraînés et évalués sur
le même jeu de données.

Objectif

Ce projet fait partie d'une série d'implémentations from scratch (régression logistique,
Naive Bayes, HMM, autocorrecteur...) réalisées pour comprendre en profondeur les
mécanismes internes des algorithmes de NLP classiques avant d'utiliser des librairies.

Prétraitement commun

Les deux scripts partagent le même prétraitement : mise en minuscules, suppression des
stopwords anglais (NLTK) et de la ponctuation, puis split 80/20 train/test.

"pam_classifier.py" — Régression logistique

- Vectorisation : bag-of-words binaire (1 si le mot du vocabulaire apparaît dans le
  message, 0 sinon).
- Modèle : 'z = w·x + b', 'sigmoid(z)' pour la probabilité.
- Entraînement : descente de gradient stochastique, mise à jour des poids après
  chaque exemple.
- Évaluation : accuracy sur train et test (seuil à 0.5).

"
python spam_classifier.py --data chemin/vers/Spam_SMS.csv --lr 0.1 --epochs 10
"

## `naive_bayes.py` — Naive Bayes

- Comptage : fréquence de chaque mot par classe ("vocab_spam", "vocab_ham").
- Lissage de Laplace : "+1" sur chaque compte pour éviter qu'un mot absent du
  vocabulaire d'une classe n'annule tout le score (zero probability problem).
- Priors : "log(n_classe / n_total)" pour tenir compte du déséquilibre spam/ham
  du dataset plutôt que de supposer un 50/50 implicite.
- Log-probabilités : les probabilités sont sommées en log plutôt que multipliées,
  pour éviter l'underflow numérique sur des messages longs.


python naive_bayes.py --Spam_SMS.csv


Le CSV attendu par les deux scripts doit contenir deux colonnes : 'Class' ('spam'/'ham')
et "Message".

Installation

pip install -r requirements.txt


Limites connues

- Descente de gradient stochastique pure (pas de vectorisation NumPy) : entraînement
  lent sur un vocabulaire large.
- Pas de régularisation ni de gestion fine des mots hors vocabulaire.
- Le split train/test est séquentiel (pas de shuffle ni de stratification).

Pistes d'amélioration

- Vectoriser les calculs avec NumPy pour accélérer l'entraînement de la régression logistique.
- Ajouter un shuffle + une validation croisée.
- Comparer précision/rappel des deux modèles (pas seulement l'accuracy globale).
