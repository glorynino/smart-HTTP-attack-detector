import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

def reentrainer_lstm_simple(data_path):
    """Version simplifiée sans métrique F1 problématique"""
    
    print("🔧" + "="*60)
    print("   RÉENTRAÎNEMENT LSTM - VERSION SIMPLIFIÉE")
    print("="*60)
    
    # 1. CHARGEMENT ET PRÉPARATION DONNÉES
    print("\n1️⃣ CHARGEMENT DONNÉES...")
    df = pd.read_csv(data_path)
    
    # Nettoyage
    df['text'] = df['text'].fillna('')
    df = df[df['text'].str.len() > 5]
    df = df[df['label'].isin([0, 1])]
    
    print(f"Dataset après nettoyage: {df.shape}")
    print(f"Distribution finale: {df['label'].value_counts()}")
    
    # 2. TOKENIZATION
    print("\n2️⃣ TOKENIZATION...")
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    
    # Analyse longueurs
    seq_lengths = [len(seq) for seq in sequences]
    max_length = min(150, int(np.percentile(seq_lengths, 95)))
    
    X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    y = df['label'].values
    
    print(f"Forme X: {X.shape}, max_length: {max_length}")
    
    # 3. SPLIT
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 4. CLASS WEIGHTS
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # 5. MODÈLE SIMPLIFIÉ
    print(f"\n3️⃣ CRÉATION MODÈLE...")
    keras.utils.set_random_seed(42)
    
    model = keras.Sequential([
        keras.layers.Embedding(input_dim=5000, output_dim=64, input_length=max_length),
        keras.layers.SpatialDropout1D(0.2),
        keras.layers.Bidirectional(keras.layers.LSTM(32, dropout=0.3, recurrent_dropout=0.3)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # 6. COMPILATION SIMPLE - SANS F1 !
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']  # Seulement accuracy pour éviter les problèmes
    )
    
    print("Architecture:")
    model.summary()
    
    # 7. CALLBACKS SIMPLES
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 8. ENTRAÎNEMENT
    print(f"\n🚀 ENTRAÎNEMENT...")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    # 9. ÉVALUATION MANUELLE AVEC SKLEARN
    print(f"\n📊 ÉVALUATION...")
    
    # Test sur validation d'abord
    y_val_pred_proba = model.predict(X_val, verbose=0)
    print(f"Probabilities Val - Min: {y_val_pred_proba.min():.4f}, Max: {y_val_pred_proba.max():.4f}")
    print(f"                  - Mean: {y_val_pred_proba.mean():.4f}, Std: {y_val_pred_proba.std():.4f}")
    
    # Test différents seuils sur validation
    print(f"\nTest seuils sur validation:")
    best_f1_val = 0
    best_threshold = 0.5
    
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_val_pred = (y_val_pred_proba >= threshold).astype(int).flatten()
        f1_val = f1_score(y_val, y_val_pred, zero_division=0)
        print(f"Seuil {threshold}: F1 = {f1_val:.3f}")
        
        if f1_val > best_f1_val:
            best_f1_val = f1_val
            best_threshold = threshold
    
    print(f"🎯 Meilleur seuil: {best_threshold} (F1 val: {best_f1_val:.3f})")
    
    # Test final sur données test
    y_test_pred_proba = model.predict(X_test, verbose=0)
    y_test_pred = (y_test_pred_proba >= best_threshold).astype(int).flatten()
    
    print(f"\n🏆 RÉSULTATS FINAUX (seuil {best_threshold}):")
    print(classification_report(y_test, y_test_pred, target_names=['Normal', 'Malveillant']))
    
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nMatrice de confusion:")
    print(cm)
    
    # Vérifier que le modèle n'est pas bloqué
    test_std = y_test_pred_proba.std()
    if test_std < 0.01:
        print(f"⚠️ ATTENTION: Std des probabilités = {test_std:.6f} - Modèle potentiellement bloqué!")
    else:
        print(f"✅ Std des probabilités = {test_std:.4f} - Modèle fonctionne bien!")
    
    # 10. GRAPHIQUES
    plt.figure(figsize=(15, 8))
    
    # Historique d'entraînement
    plt.subplot(2, 3, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Distribution probabilités
    plt.subplot(2, 3, 3)
    plt.hist(y_test_pred_proba.flatten(), bins=30, alpha=0.7, edgecolor='black')
    plt.title('Distribution probabilités test')
    plt.xlabel('Probabilité')
    plt.ylabel('Fréquence')
    plt.grid(True, alpha=0.3)
    
    # Probabilités par classe
    plt.subplot(2, 3, 4)
    normal_probs = y_test_pred_proba[y_test == 0]
    malveillant_probs = y_test_pred_proba[y_test == 1]
    
    plt.hist(normal_probs.flatten(), bins=20, alpha=0.6, label='Normal', color='blue')
    plt.hist(malveillant_probs.flatten(), bins=20, alpha=0.6, label='Malveillant', color='red')
    plt.title('Probabilités par vraie classe')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Matrice de confusion
    plt.subplot(2, 3, 5)
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Malveillant'],
                yticklabels=['Normal', 'Malveillant'])
    plt.title('Matrice de confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    
    # Courbe ROC simple
    plt.subplot(2, 3, 6)
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lstm_results_simplified.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 11. SAUVEGARDER
    model.save('lstm_csic2010_SIMPLE_FIXED.h5')
    
    import pickle
    with open('tokenizer_simple.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\n💾 SAUVEGARDÉ:")
    print(f"✅ Modèle: lstm_csic2010_SIMPLE_FIXED.h5")
    print(f"✅ Tokenizer: tokenizer_simple.pickle")
    print(f"✅ Seuil optimal: {best_threshold}")
    
    # Fonction de prédiction
    def predict_new_texts(texts):
        sequences = tokenizer.texts_to_sequences(texts)
        X_new = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
        probas = model.predict(X_new, verbose=0)
        predictions = (probas >= best_threshold).astype(int)
        return predictions, probas
    
    # Test rapide
    print(f"\n🧪 TEST RAPIDE:")
    test_texts = [
        "GET /index.html HTTP/1.1",
        "GET /admin' OR 1=1-- HTTP/1.1",
        "POST /login.php?id=1 UNION SELECT * FROM users--"
    ]
    
    preds, probs = predict_new_texts(test_texts)
    for i, text in enumerate(test_texts):
        label = "Malveillant" if preds[i][0] == 1 else "Normal"
        print(f"'{text[:50]}...' → {label} (prob: {probs[i][0]:.3f})")
    
    return model, tokenizer, best_threshold, predict_new_texts

if __name__ == "__main__":
    print("🚀 ENTRAÎNEMENT LSTM SIMPLIFIÉ...")
    print("📝 Cette version évite les problèmes de métriques F1")
    
    data_path = "DATA/LSTM_csic2010_clean_v2.csv"
    
    try:
        model, tokenizer, seuil, predict_func = reentrainer_lstm_simple(data_path)
        print("\n🎉 SUCCÈS! Le modèle est prêt!")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        print("Vérifiez le chemin du fichier de données.")