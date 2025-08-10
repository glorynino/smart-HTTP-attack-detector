#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

import matplotlib.pyplot as plt

# -------------------------
#  CONFIG
# -------------------------
CSV_PATH = "csic2010_clean.csv"          # <-- place ton CSV ici
RANDOM_STATE = 42
TEST_SIZE = 0.2

TFIDF_MAX_FEATURES = 20000               # nombre max de features TF-IDF (ajuste si mémoire faible)
SVD_COMPONENTS = 200                     # réduction dimensionnelle après TF-IDF (transforme sparse->dense)
RF_ESTIMATORS = 200
OUTPUT_MODEL = "csic2010_rf_pipeline.joblib"
PLOT_ROC = True

# -------------------------
#  1) CHARGEMENT
# -------------------------
print("[1/7] Chargement du CSV...")
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Fichier non trouvé : {CSV_PATH}")

# on force le type chaîne pour éviter warnings/mélanges
df = pd.read_csv(CSV_PATH, dtype=str, low_memory=False)
print("shape:", df.shape)
print(df['label'].value_counts())

# convertir label en entier si ce n'est pas déjà le cas
df['label'] = df['label'].astype(int)

# -------------------------
# 2) CONSTRUCTION DU TEXTE
# -------------------------
print("[2/7] Construction d'une colonne 'text' à partir des colonnes textuelles...")
text_cols = df.select_dtypes(include=['object']).columns.tolist()
# supprimer 'label' si présent dans la liste (on ne veut pas l'utiliser comme feature)
text_cols = [c for c in text_cols if c not in ('label', 'index', 'Unnamed: 0')]

print("Colonnes textuelles détectées :", text_cols)
df[text_cols] = df[text_cols].fillna('')
# concatène les colonnes textuelles en une seule colonne 'text' (séparateur espace)
df['text'] = df[text_cols].agg(' '.join, axis=1)

# -------------------------
# 3) SPLIT train/test
# -------------------------
print("[3/7] Split train/test...")
X = df['text']
y = df['label'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print("Train:", X_train.shape[0], " Test:", X_test.shape[0])

# -------------------------
# 4) PIPELINE TF-IDF -> SVD -> RF
# -------------------------
print("[4/7] Construction du pipeline TF-IDF -> SVD -> RandomForest ...")
# Pourquoi char n-grams ? Les payloads HTTP contiennent souvent des patterns (séquences de caractères)
# utiles pour détecter injections / encodages. Tu peux tester aussi word n-grams (analyzer='word').
tfidf = TfidfVectorizer(
    analyzer='char_wb',          # 'char_wb' marche bien pour payloads/URL ; changer si besoin
    ngram_range=(3, 5),
    max_features=TFIDF_MAX_FEATURES
)

svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)

rf = RandomForestClassifier(
    n_estimators=RF_ESTIMATORS,
    n_jobs=-1,
    random_state=RANDOM_STATE,
    class_weight='balanced'     # utile si classes déséquilibrées
)

pipeline = Pipeline([
    ('tfidf', tfidf),
    ('svd', svd),
    ('clf', rf)
])

# -------------------------
# 5) ENTRAINEMENT
# -------------------------
print("[5/7] Entraînement du pipeline sur le train...")
pipeline.fit(X_train, y_train)
print("[OK] Entraînement terminé.")

# -------------------------
# 6) EVALUATION
# -------------------------
print("[6/7] Évaluation sur le jeu de test...")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification report :")
print(classification_report(y_test, y_pred, digits=4))
print("Confusion matrix :")
print(confusion_matrix(y_test, y_pred))

# Probabilités & ROC AUC (si implémenté)
if hasattr(pipeline.named_steps['clf'], "predict_proba"):
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, y_proba)
        print("ROC AUC:", auc)
        if PLOT_ROC:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure()
            plt.plot(fpr, tpr)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC curve")
            plt.grid(True)
            plt.savefig("roc_curve.png")
            print("ROC saved to roc_curve.png")
    except Exception as e:
        print("Impossible de calculer ROC AUC:", e)

# -------------------------
# 7) SAUVEGARDE DU MODELE
# -------------------------
print("[7/7] Sauvegarde du pipeline (TF-IDF + SVD + RF) ...")
joblib.dump(pipeline, OUTPUT_MODEL)
print("Pipeline sauvegardé sous :", OUTPUT_MODEL)

# Fonction utilitaire d'exemple pour prédiction rapide sur de nouvelles requêtes
def predict_texts(texts):
    """
    texts : list of strings
    retourne : preds, probs (probs peut être None si predict_proba absent)
    """
    preds = pipeline.predict(texts)
    probs = pipeline.predict_proba(texts)[:,1] if hasattr(pipeline.named_steps['clf'], "predict_proba") else None
    return preds, probs

if __name__ == "__main__":
    # petit test rapide
    exemple = [
        "GET /index.php?id=1",
        "GET /search?q=<script>alert(1)</script>",
        "POST /login.php username=admin' --&password=blah"
    ]
    preds, probs = predict_texts(exemple)
    for t, p, prob in zip(exemple, preds, probs if probs is not None else [None]*len(preds)):
        print(f"Exemple: {t}\n  => pred: {p}, prob: {prob}\n")
