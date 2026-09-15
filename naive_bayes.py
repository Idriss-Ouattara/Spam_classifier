"""
Classifieur de spam SMS avec un Naive Bayes implémenté from scratch (sans scikit-learn).

Utilise des log-probabilités (pour éviter l'underflow), le lissage de Laplace
(pour éviter les probabilités nulles sur les mots absents du vocabulaire d'une classe)
et les priors des classes (pour tenir compte du déséquilibre spam/ham du dataset).
"""

import argparse
import math
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords


def charger_donnees(chemin_csv):
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
        donnees_propres.append((texte_propre, label))

    return donnees_propres


def entrainer(train):
    vocab_spam = {}
    vocab_ham = {}
    n_spam = 0
    n_ham = 0

    for texte, label in train:
        mots = texte.lower().split()
        if label == "spam":
            n_spam += 1
        else:
            n_ham += 1

        for mot in mots:
            if label == "spam":
                vocab_spam[mot] = vocab_spam.get(mot, 0) + 1
            else:
                vocab_ham[mot] = vocab_ham.get(mot, 0) + 1

    total_spam = sum(vocab_spam.values())
    total_ham = sum(vocab_ham.values())

    # Priors : sans eux, le classifieur suppose implicitement un 50/50 spam/ham,
    # alors que le dataset est déséquilibré (beaucoup plus de ham que de spam).
    total = len(train)
    prior_spam = math.log(n_spam / total)
    prior_ham = math.log(n_ham / total)

    # Vocabulaire global pour un lissage de Laplace symétrique : les deux probabilités
    # (spam et ham) sont ainsi calculées sur la même base.
    vocab_global = len(set(vocab_spam) | set(vocab_ham))

    return vocab_spam, vocab_ham, total_spam, total_ham, prior_spam, prior_ham, vocab_global


def classifier(texte, vocab_spam, vocab_ham, total_spam, total_ham, prior_spam, prior_ham, vocab_global):
    mots = texte.lower().split()

    score_spam = prior_spam
    score_ham = prior_ham

    for mot in mots:
        # +1 = lissage de Laplace. Sans lui, un mot absent du vocabulaire d'une classe
        # donnerait une probabilité de 0, qui annulerait tout le score (produit ou somme
        # de logs à -inf) : le "zero probability problem" du Naive Bayes.
        p_spam = (vocab_spam.get(mot, 0) + 1) / (total_spam + vocab_global)
        p_ham = (vocab_ham.get(mot, 0) + 1) / (total_ham + vocab_global)

        score_spam += math.log(p_spam)
        score_ham += math.log(p_ham)

    return "spam" if score_spam > score_ham else "ham"


def evaluer(test, vocab_spam, vocab_ham, total_spam, total_ham, prior_spam, prior_ham, vocab_global):
    correct = 0
    for texte, label in test:
        prediction = classifier(
            texte, vocab_spam, vocab_ham, total_spam, total_ham, prior_spam, prior_ham, vocab_global
        )
        if prediction == label:
            correct += 1
    return correct / len(test)


def main():
    parser = argparse.ArgumentParser(description="Naive Bayes spam classifier from scratch")
    parser.add_argument(
        "--data", default="Spam_SMS.csv", help="Chemin vers le CSV (colonnes: Class, Message)"
    )
    args = parser.parse_args()

    donnees_propres = charger_donnees(args.data)

    split = int(len(donnees_propres) * 0.8)
    train = donnees_propres[:split]
    test = donnees_propres[split:]
    print(f"Train : {len(train)} exemples")
    print(f"Test  : {len(test)} exemples")

    vocab_spam, vocab_ham, total_spam, total_ham, prior_spam, prior_ham, vocab_global = entrainer(train)

    # Quelques tests manuels pour sanity-check
    exemples = [
        "free money win prize now",
        "tomorrow meeting will be at 6am",
    ]
    for exemple in exemples:
        pred = classifier(exemple, vocab_spam, vocab_ham, total_spam, total_ham, prior_spam, prior_ham, vocab_global)
        print(f"'{exemple}' -> {pred}")

    accuracy = evaluer(test, vocab_spam, vocab_ham, total_spam, total_ham, prior_spam, prior_ham, vocab_global)
    print(f"Accuracy : {accuracy:.2%}")


if __name__ == "__main__":
    main()
