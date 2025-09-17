import pickle
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# === Initialisation Flask ===
app = Flask(__name__)

# === Chargement du modèle LSTM ===
model = load_model("LSTM_MODEL\\lstm_csic2010_SIMPLE_FIXED.h5")

# === Chargement du tokenizer (.pickle) ===
with open("LSTM_MODEL\\tokenizer_simple.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

# === Paramètres ===
MAXLEN = 100  # longueur max des séquences (doit être la même que pendant l'entraînement)

# === Route principale ===
@app.route("/")
def home():
    return jsonify({"message": "🚀 API LSTM prête !"})

# === Route de prédiction ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupération du JSON envoyé
        data = request.get_json(force=True)
        texts = data.get("features", [])

        if not texts:
            return jsonify({"error": "Aucune requête reçue"}), 400

        # Transformation texte → séquences numériques
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=MAXLEN, padding="post")

        # Prédiction avec LSTM
        preds = model.predict(padded)
        results = (preds >= 0.5).astype(int).flatten().tolist()

        # Construction de la réponse
        output = []
        for txt, pred, prob in zip(texts, results, preds.flatten().tolist()):
            output.append({
                "request": txt,
                "prediction": int(pred),   # 0 = normal, 1 = malveillant
                "probability": float(prob) # probabilité brute du modèle
            })

        return jsonify(output)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === Lancement ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
