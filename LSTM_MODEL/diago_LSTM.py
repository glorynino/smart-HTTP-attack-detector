import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# === 1. Charger les données ===
df = pd.read_csv("DATA/LSTM_csic2010_clean_v2.csv")
texts = df["text"].astype(str).tolist()
labels = df["label"].values

# === 2. Recharger le tokenizer utilisé à l'entraînement ===
with open("DATA/tokenizer_simple.pickle", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 200  # doit être identique à celui utilisé à l’entraînement
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len)

# === 3. Charger le modèle entraîné ===
model = keras.models.load_model("DATA/lstm_csic2010_SIMPLE_FIXED.h5")

# === 4. Prédictions ===
y_pred_prob = model.predict(X)
y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

# === 5. Évaluation ===
print("=== Résultats du diagnostic ===")
print(f"Accuracy:  {accuracy_score(labels, y_pred):.4f}")
print(f"Précision: {precision_score(labels, y_pred, zero_division=0):.4f}")
print(f"Rappel:    {recall_score(labels, y_pred, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(labels, y_pred, zero_division=0):.4f}")

print("\n=== Rapport de classification ===")
print(classification_report(labels, y_pred, zero_division=0))

print("\n=== Matrice de confusion ===")
print(confusion_matrix(labels, y_pred))
