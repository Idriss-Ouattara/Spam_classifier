"""
Classifieur de spam SMS avec une régression logistique implémentée from scratch
(sans scikit-learn), sur une représentation bag-of-words binaire.
"""

import argparse
import string
import math
import nltk
import pandas as pd
from nltk.corpus import stopwords


def charger_donnees(chemin_csv):
    """
    Charge le CSV et nettoie chaque message : minuscules, suppression des
    stopwords anglais (mots très fréquents et peu informatifs comme "the",
    "is", "at") et de la ponctuation.

    Retourne une liste de tuples (label, texte_nettoye), ex :
    ("spam", "win free prize now")
    """
    nltk.download("stopwords", quiet=True)
    df = pd.read_csv(chemin_csv)
    stopwords_english = stopwords.words("english")

    donnees_propres = []
    for _, ligne in df.iterrows():
        texte = ligne["Message"].lower()
        label = ligne["Class"].lower()

        mots = texte.split()
        mots_propres = [
            mot for mot in mots
            if mot not in stopwords_english and mot not in string.punctuation
        ]
        texte_propre = " ".join(mots_propres)
        donnees_propres.append((label, texte_propre))

    return donnees_propres


def construire_vocabulaire(train):
    """
    Récupère l'ensemble des mots uniques présents dans le train (jamais dans
    le test, pour ne pas "tricher" en connaissant à l'avance le vocabulaire
    des données qu'on n'est pas censé avoir vues).

    Ce vocabulaire définit la taille et l'ordre des vecteurs qu'on va
    construire ensuite : si le vocabulaire contient 3500 mots, chaque message
    sera représenté par un vecteur de 3500 chiffres.
    """
    vocabulaire = set()
    for _, texte in train:
        for mot in texte.split():
            vocabulaire.add(mot)
    return list(vocabulaire)  # liste pour avoir un ordre fixe


def vectoriser(texte, vocabulaire):
    """
    Transforme un texte en un vecteur bag-of-words BINAIRE.

    Exemple concret avec un vocabulaire de 5 mots ["free", "win", "meeting",
    "money", "tomorrow"] :
    - texte = "win free money" -> [1, 1, 0, 1, 0]
      (1 pour "free" et "win" et "money" car présents, 0 pour "meeting" et
      "tomorrow" car absents)
    - texte = "tomorrow meeting" -> [0, 0, 1, 0, 1]

    Le vecteur ne dit PAS combien de fois un mot apparaît (contrairement à un
    bag-of-words classique basé sur des comptes), seulement s'il est présent
    (1) ou absent (0). C'est une simplification volontaire.
    """
    mots = set(texte.split())
    return [1 if mot in mots else 0 for mot in vocabulaire]


def vectoriser_dataset(donnees, vocabulaire):
    """
    Applique vectoriser() à tout un dataset et encode le label en 0/1 :
    spam -> 1, ham -> 0. C'est ce format numérique (X, y) qu'attend un
    modèle de classification.
    """
    X, y = [], []
    for label, texte in donnees:
        X.append(vectoriser(texte, vocabulaire))
        y.append(1 if label == "spam" else 0)
    return X, y


def sigmoid(z):
    """
    Écrase n'importe quel nombre réel (négatif ou positif, petit ou grand)
    dans l'intervalle (0, 1), pour en faire une probabilité.

    Quelques repères numériques :
    - sigmoid(0)   = 0.5   (incertitude totale, aucun signal)
    - sigmoid(5)   ≈ 0.99  (z très positif -> quasi certain que c'est spam)
    - sigmoid(-5)  ≈ 0.01  (z très négatif -> quasi certain que c'est ham)

    C'est cette fonction qui permet de transformer le score brut z (qui peut
    valoir n'importe quoi, ex: -12.4 ou 7.8) en quelque chose d'interprétable
    comme une probabilité.
    """
    return 1 / (1 + math.exp(-z))


def predict(x, poids, biais):
    """
    Calcule la probabilité que le message représenté par le vecteur x soit
    du spam.

    Étape 1 : score brut z = combinaison linéaire des features.
    z = biais + x[0]*poids[0] + x[1]*poids[1] + ... + x[n]*poids[n]

    Concrètement : x[i] vaut 0 ou 1 (mot absent/présent), donc x[i]*poids[i]
    ne compte QUE si le mot i est présent dans le message. Un mot très
    caractéristique du spam (ex: "free") aura un poids fortement positif ;
    un mot très caractéristique du ham (ex: "meeting") aura un poids
    fortement négatif. Le biais, lui, capture la tendance de base du modèle
    indépendamment du contenu du message (ex: si le dataset contient
    beaucoup plus de ham que de spam, le biais aura tendance à être négatif).

    Étape 2 : on passe z dans sigmoid() pour obtenir une probabilité.
    """
    z = biais
    for i in range(len(x)):
        z += x[i] * poids[i]
    return sigmoid(z)


def entrainer_regression(X, y, lr=0.1, epochs=10):
    """
    Apprend les poids et le biais par descente de gradient stochastique :
    on met à jour les paramètres après CHAQUE exemple (pas après avoir vu
    tout le dataset), ce qui rend l'apprentissage progressif et bruité mais
    rapide à mettre en oeuvre.

    Logique de la boucle, pour un exemple (x, label) donné :

    1. On prédit avec les poids actuels : y_pred = predict(x, poids, biais)
       (au tout début, poids et biais valent 0, donc y_pred = sigmoid(0) = 0.5
       pour TOUS les exemples : le modèle n'a encore rien appris.)

    2. On calcule l'erreur : erreur = y_pred - label
       - Si label=1 (spam) et y_pred=0.5 -> erreur = -0.5 (le modèle sous-estime)
       - Si label=0 (ham) et y_pred=0.5  -> erreur = +0.5 (le modèle surestime)
       Cette erreur est la dérivée de la fonction de coût (log loss) par
       rapport à z ; c'est ce qui indique dans quel sens et de combien
       corriger chaque poids.

    3. On ajuste chaque poids dans la direction qui réduit l'erreur :
       poids[i] -= lr * erreur * x[i]
       - x[i] vaut 0 ou 1 : si le mot i n'est pas dans le message (x[i]=0),
         son poids n'est PAS modifié pour cet exemple (0 * erreur = 0), ce
         qui est cohérent : un mot absent n'a aucune raison d'être ajusté sur
         cet exemple précis.
       - Si le mot i est présent (x[i]=1) et que le modèle a sous-estimé un
         spam (erreur négative), le poids AUGMENTE (-lr * erreur négative =
         positif) : la présence de ce mot rendra la prochaine prédiction plus
         proche de "spam".

    4. lr (learning rate) contrôle la taille du pas de correction. Trop grand
       (ex: 10), les poids oscillent sans se stabiliser. Trop petit (ex:
       0.0001), l'apprentissage est très lent et epochs=10 ne suffira peut-être
       pas à converger.

    On répète ce processus epochs fois sur l'ensemble du train : chaque
    epoch est un passage complet sur tous les exemples d'entraînement.
    """
    n_features = len(X[0])
    poids = [0.0] * n_features
    biais = 0.0

    for epoch in range(epochs):
        for x, label in zip(X, y):
            y_pred = predict(x, poids, biais)
            erreur = y_pred - label  # dérivée de la loss (log loss) par rapport à z

            for i in range(n_features):
                poids[i] -= lr * erreur * x[i]
            biais -= lr * erreur

        print(f"Epoch {epoch + 1}/{epochs} terminée")

    return poids, biais


def evaluer(X, y, poids, biais):
    """
    Mesure la performance du modèle : pour chaque exemple, on prédit une
    probabilité, on la seuille à 0.5 pour obtenir une classe (spam si
    prob >= 0.5, sinon ham), et on compare au vrai label.

    Le seuil 0.5 est un choix arbitraire mais standard : rien n'empêche de le
    déplacer (ex: 0.3) si on préfère être plus agressif à détecter le spam,
    quitte à avoir plus de faux positifs.

    Retourne l'accuracy en pourcentage : (nombre de bonnes prédictions / total) * 100
    """
    correct = 0
    for x, label in zip(X, y):
        prob = predict(x, poids, biais)
        prediction = 1 if prob >= 0.5 else 0
        if prediction == label:
            correct += 1
    return correct / len(y) * 100


def main():
    parser = argparse.ArgumentParser(description="Spam classifier from scratch")
    parser.add_argument(
        "--data", default="Spam_SMS.csv", help="Chemin vers le CSV (colonnes: Class, Message)"
    )
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    # 1. Chargement + nettoyage du texte
    donnees_propres = charger_donnees(args.data)
    print(f"{len(donnees_propres)} messages chargés")

    # 2. Split train/test (80% / 20%, split séquentiel, pas de shuffle)
    split = int(len(donnees_propres) * 0.8)
    train = donnees_propres[:split]
    test = donnees_propres[split:]
    print(f"Train : {len(train)} exemples")
    print(f"Test  : {len(test)} exemples")

    # 3. Vocabulaire construit UNIQUEMENT sur le train
    vocabulaire = construire_vocabulaire(train)
    print(f"Taille du vocabulaire : {len(vocabulaire)}")

    # 4. Vectorisation : chaque message -> un vecteur de 0/1
    X_train, y_train = vectoriser_dataset(train, vocabulaire)
    X_test, y_test = vectoriser_dataset(test, vocabulaire)
    print(f"X_train : {len(X_train)} exemples x {len(X_train[0])} features")
    print(f"X_test  : {len(X_test)} exemples x {len(X_test[0])} features")

    # 5. Entraînement : apprentissage des poids et du biais sur le train
    poids, biais = entrainer_regression(X_train, y_train, lr=args.lr, epochs=args.epochs)

    # 6. Évaluation : accuracy sur train (données vues) et test (données jamais vues)
    acc_train = evaluer(X_train, y_train, poids, biais)
    acc_test = evaluer(X_test, y_test, poids, biais)
    print(f"Accuracy Train : {acc_train:.2f}%")
    print(f"Accuracy Test  : {acc_test:.2f}%")


if __name__ == "__main__":
    main()
