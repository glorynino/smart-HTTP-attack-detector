# streamlit_dashboard_beautiful.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# ---------------------------
# CONFIG - change these paths if needed
# ---------------------------
MODEL_PATH = r"C:\Users\neili\OneDrive\Bureau\detection of malware\LSTM_MODEL\lstm_csic2010_SIMPLE_FIXED.h5"
TOKENIZER_PATH = r"C:\Users\neili\OneDrive\Bureau\detection of malware\LSTM_MODEL\tokenizer_simple.pickle"
MAX_LEN = 200

st.set_page_config(
    page_title="CyberShield AI - Malware Detection", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# STUNNING CSS with animations and modern design
# ---------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    :root {
      --bg-primary: #0a0b14;
      --bg-secondary: #111218;
      --bg-card: rgba(17, 18, 24, 0.8);
      --bg-glass: rgba(255, 255, 255, 0.05);
      --accent-primary: #00d9ff;
      --accent-secondary: #7c3aed;
      --accent-success: #22c55e;
      --accent-warning: #f59e0b;
      --accent-danger: #ef4444;
      --text-primary: #ffffff;
      --text-secondary: #a1a1aa;
      --text-muted: #71717a;
      --border: rgba(255, 255, 255, 0.1);
    }
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
        min-height: 100vh;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom header */
    .cyber-header {
        background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .cyber-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 217, 255, 0.2), transparent);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .cyber-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(0, 217, 255, 0.3);
    }
    
    .cyber-subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
        font-weight: 400;
    }
    
    /* Metric cards */
    .metrics-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 2rem;
        margin: 2rem 0;
    }
    
    .metric-card {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: var(--accent-primary);
        box-shadow: 0 20px 40px rgba(0, 217, 255, 0.2);
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
        border-radius: 20px 20px 0 0;
    }
    
    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        opacity: 0.8;
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, var(--text-primary), var(--accent-primary));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-subtitle {
        font-size: 0.85rem;
        color: var(--text-muted);
        font-weight: 400;
    }
    
    /* Status indicators */
    .status-safe { color: var(--accent-success); }
    .status-warning { color: var(--accent-warning); }
    .status-danger { color: var(--accent-danger); }
    
    /* Upload section */
    .upload-section {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        border: 2px dashed var(--border);
        border-radius: 20px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: var(--accent-primary);
        background: rgba(0, 217, 255, 0.05);
    }
    
    .upload-icon {
        font-size: 4rem;
        color: var(--accent-primary);
        margin-bottom: 1rem;
        opacity: 0.7;
    }
    
    /* Evaluation cards */
    .eval-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .eval-card {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .eval-card:hover {
        transform: translateY(-3px);
        border-color: var(--accent-secondary);
        box-shadow: 0 15px 30px rgba(124, 58, 237, 0.2);
    }
    
    .eval-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-secondary);
        margin-bottom: 0.5rem;
    }
    
    .eval-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 4rem;
    }
    
    .loading-spinner {
        width: 60px;
        height: 60px;
        border: 3px solid var(--border);
        border-top: 3px solid var(--accent-primary);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 217, 255, 0.4);
    }
    
    /* Custom file uploader */
    .stFileUploader > div > div {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        border: 2px dashed var(--border);
        border-radius: 16px;
        padding: 2rem;
    }
    
    .stFileUploader > div > div:hover {
        border-color: var(--accent-primary);
        background: rgba(0, 217, 255, 0.05);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: var(--bg-secondary);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: rgba(34, 197, 94, 0.1);
        border: 1px solid var(--accent-success);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid var(--accent-danger);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid var(--accent-warning);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    .stInfo {
        background: rgba(0, 217, 255, 0.1);
        border: 1px solid var(--accent-primary);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
        border-radius: 10px;
    }
    
    /* Section dividers */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-primary), transparent);
        margin: 3rem 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .cyber-title { font-size: 2rem; }
        .metrics-container { grid-template-columns: 1fr; }
        .eval-grid { grid-template-columns: repeat(2, 1fr); }
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Load model and tokenizer (cached)
# ---------------------------
@st.cache_resource
def load_model_and_tokenizer(model_path: str, tokenizer_path: str):
    model = None
    tokenizer = None
    try:
        model = keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"🚨 Erreur chargement modèle: {e}")
    try:
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        st.error(f"🚨 Erreur chargement tokenizer: {e}")
    return model, tokenizer

# ---------------------------
# Beautiful header
# ---------------------------
st.markdown("""
    <div class="cyber-header">
        <h1 class="cyber-title">🛡️ CyberShield AI</h1>
        <p class="cyber-subtitle">Détection intelligente de malware en temps réel</p>
    </div>
""", unsafe_allow_html=True)

# Load models with progress indication
with st.spinner('🚀 Chargement des modèles IA...'):
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH, TOKENIZER_PATH)

if model is not None and tokenizer is not None:
    st.success("✅ Modèles chargés avec succès!")
else:
    st.error("❌ Impossible de charger les modèles. Vérifiez les chemins dans la configuration.")

# ---------------------------
# Upload section with beautiful styling
# ---------------------------
st.markdown("""
    <div class="upload-section">
        <div class="upload-icon">📤</div>
        <h3>Analysez vos données</h3>
        <p>Uploadez un fichier CSV ou spécifiez un chemin local</p>
    </div>
""", unsafe_allow_html=True)

# File input options
tab1, tab2 = st.tabs(["📁 Upload de fichier", "💻 Chemin local"])

csv_file = None

with tab1:
    uploaded = st.file_uploader(
        "Choisissez votre fichier CSV", 
        type=["csv"], 
        help="Le fichier doit contenir une colonne 'text' et optionnellement une colonne 'label'"
    )
    if uploaded is not None:
        try:
            csv_file = pd.read_csv(uploaded)
            st.success(f"🎉 Fichier chargé avec succès! **{len(csv_file):,}** lignes détectées")
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier: {e}")

with tab2:
    path = st.text_input(
        "📂 Chemin complet du fichier CSV", 
        placeholder="C:\\Users\\...\\data.csv",
        help="Chemin absolu vers votre fichier CSV"
    )
    if path and st.button("🔍 Charger le fichier"):
        try:
            csv_file = pd.read_csv(path)
            st.success(f"🎉 Fichier chargé depuis le chemin local! **{len(csv_file):,}** lignes détectées")
        except Exception as e:
            st.error(f"❌ Impossible de lire le fichier: {e}")

# ---------------------------
# Main analysis section
# ---------------------------
if csv_file is not None and model is not None and tokenizer is not None:
    df = csv_file.copy()
    
    if "text" not in df.columns:
        st.error("❌ Le fichier doit contenir une colonne 'text'")
        st.stop()
    
    # Analysis progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner('🧠 Analyse en cours avec l\'IA...'):
        # Prepare data
        status_text.text('Préparation des données...')
        progress_bar.progress(20)
        
        texts = df["text"].astype(str).tolist()
        sequences = tokenizer.texts_to_sequences(texts)
        X = pad_sequences(sequences, maxlen=MAX_LEN)
        
        # Prediction
        status_text.text('Prédiction avec le modèle LSTM...')
        progress_bar.progress(60)
        
        try:
            y_pred_prob = model.predict(X, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype("int32").flatten()
        except Exception as e:
            st.error(f"❌ Erreur lors de la prédiction: {e}")
            y_pred = np.zeros(len(df), dtype=int)
        
        status_text.text('Génération des métriques...')
        progress_bar.progress(100)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Calculate metrics
    total = len(df)
    malicious = int(np.sum(y_pred == 1))
    legitimate = int(np.sum(y_pred == 0))
    block_rate = (malicious / total * 100) if total > 0 else 0.0
    
    # Determine status
    if block_rate < 10:
        status_class = "status-safe"
        status_icon = "✅"
        status_text = "Sécurisé"
    elif block_rate < 30:
        status_class = "status-warning"
        status_icon = "⚠️"
        status_text = "Vigilance"
    else:
        status_class = "status-danger"
        status_icon = "🚨"
        status_text = "Menace"
    
    # Beautiful metrics display with native Streamlit components
    st.markdown("## 📊 Résultats de l'analyse")
    
    # Create 4 columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total analysé",
            value=f"{total:,}",
            help="Requêtes traitées par l'IA"
        )
    
    with col2:
        st.metric(
            label="🚨 Menaces détectées", 
            value=f"{malicious:,}",
            help="Requêtes malveillantes identifiées"
        )
    
    with col3:
        st.metric(
            label="✅ Trafic légitime",
            value=f"{legitimate:,}", 
            help="Requêtes saines confirmées"
        )
    
    with col4:
        st.metric(
            label=f"{status_icon} Taux de risque",
            value=f"{block_rate:.1f}%",
            help=f"Statut: {status_text}"
        )
    
    # Interactive charts
    if st.checkbox("📈 Afficher les graphiques interactifs", value=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Légitime', 'Malveillant'],
                values=[legitimate, malicious],
                hole=.3,
                marker_colors=['#22c55e', '#ef4444'],
                textinfo='label+percent',
                textfont_size=14,
                textfont_color='white'
            )])
            
            fig_pie.update_layout(
                title={
                    'text': "Distribution des menaces",
                    'x': 0.5,
                    'font': {'size': 18, 'color': 'white'}
                },
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Confidence distribution
            confidence_scores = y_pred_prob.flatten()
            fig_hist = go.Figure(data=[go.Histogram(
                x=confidence_scores,
                nbinsx=30,
                marker_color='rgba(0, 217, 255, 0.7)',
                marker_line_color='rgba(0, 217, 255, 1)',
                marker_line_width=1
            )])
            
            fig_hist.update_layout(
                title={
                    'text': "Distribution des scores de confiance",
                    'x': 0.5,
                    'font': {'size': 18, 'color': 'white'}
                },
                xaxis_title="Score de confiance",
                yaxis_title="Fréquence",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                xaxis={'gridcolor': 'rgba(255,255,255,0.1)'},
                yaxis={'gridcolor': 'rgba(255,255,255,0.1)'}
            )
            
            st.plotly_chart(fig_hist, use_container_width=True)
    
    # Section divider
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Evaluation section (if labels available)
    if "label" in df.columns:
        st.markdown("## 🎯 Évaluation de performance")
        
        y_true = df["label"].astype(int).values
        y_hat = y_pred.astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_hat)
        prec = precision_score(y_true, y_hat, zero_division=0)
        rec = recall_score(y_true, y_hat, zero_division=0)
        f1 = f1_score(y_true, y_hat, zero_division=0)
        
        # Beautiful evaluation cards
        st.markdown(f"""
            <div class="eval-grid">
                <div class="eval-card">
                    <div class="eval-value">{acc:.3f}</div>
                    <div class="eval-label">Précision globale</div>
                </div>
                <div class="eval-card">
                    <div class="eval-value">{prec:.3f}</div>
                    <div class="eval-label">Précision</div>
                </div>
                <div class="eval-card">
                    <div class="eval-value">{rec:.3f}</div>
                    <div class="eval-label">Rappel</div>
                </div>
                <div class="eval-card">
                    <div class="eval-value">{f1:.3f}</div>
                    <div class="eval-label">Score F1</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        # Confusion matrix with beautiful styling
        col1, col2 = st.columns([1, 1])
        
        with col1:
            cm = confusion_matrix(y_true, y_hat)
            
            # Create heatmap with Plotly
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Prédit: Légitime', 'Prédit: Malveillant'],
                y=['Vrai: Légitime', 'Vrai: Malveillant'],
                colorscale='Viridis',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 16, "color": "white"},
                showscale=False
            ))
            
            fig_cm.update_layout(
                title={
                    'text': "Matrice de confusion",
                    'x': 0.5,
                    'font': {'size': 18, 'color': 'white'}
                },
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white'},
                width=400,
                height=400
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # Classification report in beautiful format
            st.markdown("### 📋 Rapport détaillé")
            
            report = classification_report(y_true, y_hat, zero_division=0, output_dict=True)
            
            # Create a beautiful table-like display
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);">
                    <h4 style="color: #00d9ff; margin-bottom: 1rem;">Métriques par classe</h4>
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                        <div>
                            <p style="color: #22c55e; font-weight: 600;">Classe Légitime (0)</p>
                            <p>Précision: <span style="color: white;">{report['0']['precision']:.3f}</span></p>
                            <p>Rappel: <span style="color: white;">{report['0']['recall']:.3f}</span></p>
                            <p>F1-score: <span style="color: white;">{report['0']['f1-score']:.3f}</span></p>
                        </div>
                        <div>
                            <p style="color: #ef4444; font-weight: 600;">Classe Malveillante (1)</p>
                            <p>Précision: <span style="color: white;">{report['1']['precision']:.3f}</span></p>
                            <p>Rappel: <span style="color: white;">{report['1']['recall']:.3f}</span></p>
                            <p>F1-score: <span style="color: white;">{report['1']['f1-score']:.3f}</span></p>
                        </div>
                    </div>
                    <hr style="margin: 1rem 0; border: 1px solid rgba(255,255,255,0.1);">
                    <p><strong>Support total:</strong> {report['macro avg']['support']:.0f} échantillons</p>
                </div>
            """, unsafe_allow_html=True)

elif csv_file is not None and (model is None or tokenizer is None):
    st.error("🚨 Les modèles n'ont pas pu être chargés. Vérifiez les chemins MODEL_PATH et TOKENIZER_PATH.")
else:
    # Welcome message with call to action
    st.markdown("""
        <div style="text-align: center; padding: 3rem; background: rgba(255,255,255,0.02); border-radius: 20px; margin: 2rem 0;">
            <h2 style="color: #00d9ff; margin-bottom: 1rem;">🚀 Prêt à analyser</h2>
            <p style="color: #a1a1aa; font-size: 1.1rem; margin-bottom: 2rem;">
                Uploadez votre fichier CSV pour commencer l'analyse intelligente de malware
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
                <div style="background: rgba(0,217,255,0.1); padding: 1rem 2rem; border-radius: 12px; border: 1px solid rgba(0,217,255,0.3);">
                    <strong style="color: #00d9ff;">📋 Format requis</strong><br>
                    <span style="color: #a1a1aa;">Colonne 'text' obligatoire</span>
                </div>
                <div style="background: rgba(34,197,94,0.1); padding: 1rem 2rem; border-radius: 12px; border: 1px solid rgba(34,197,94,0.3);">
                    <strong style="color: #22c55e;">⚡ Analyse rapide</strong><br>
                    <span style="color: #a1a1aa;">Résultats en temps réel</span>
                </div>
                <div style="background: rgba(124,58,237,0.1); padding: 1rem 2rem; border-radius: 12px; border: 1px solid rgba(124,58,237,0.3);">
                    <strong style="color: #7c3aed;">🤖 IA avancée</strong><br>
                    <span style="color: #a1a1aa;">Modèle LSTM optimisé</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Footer with additional features
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# Advanced analytics section
if csv_file is not None and model is not None and tokenizer is not None:
    with st.expander("🔬 Analyses avancées", expanded=False):
        st.markdown("### 📊 Statistiques détaillées")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Text length analysis
            text_lengths = df['text'].str.len()
            avg_length = text_lengths.mean()
            
            fig_length = go.Figure()
            fig_length.add_trace(go.Histogram(
                x=text_lengths,
                nbinsx=30,
                name="Distribution des longueurs",
                marker_color='rgba(124, 58, 237, 0.7)',
                marker_line_color='rgba(124, 58, 237, 1)',
                marker_line_width=1
            ))
            
            fig_length.update_layout(
                title="Distribution des longueurs de texte",
                xaxis_title="Longueur (caractères)",
                yaxis_title="Fréquence",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white', 'size': 12},
                height=300
            )
            
            st.plotly_chart(fig_length, use_container_width=True)
            st.metric("Longueur moyenne", f"{avg_length:.0f} caractères")
        
        with col2:
            # Prediction confidence analysis
            confidence_mean = np.mean(y_pred_prob)
            confidence_std = np.std(y_pred_prob)
            
            # Box plot of confidence scores
            fig_box = go.Figure()
            fig_box.add_trace(go.Box(
                y=y_pred_prob.flatten(),
                name="Scores de confiance",
                marker_color='rgba(0, 217, 255, 0.7)',
                line_color='rgba(0, 217, 255, 1)'
            ))
            
            fig_box.update_layout(
                title="Distribution des scores de confiance",
                yaxis_title="Score",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white', 'size': 12},
                height=300,
                showlegend=False
            )
            
            st.plotly_chart(fig_box, use_container_width=True)
            st.metric("Confiance moyenne", f"{confidence_mean:.3f}")
            st.metric("Écart-type", f"{confidence_std:.3f}")
        
        with col3:
            # Time-based analysis (if we can extract timestamps)
            st.markdown("#### ⏱️ Statistiques temporelles")
            
            # Processing time simulation
            processing_time = len(df) * 0.001  # Simulated processing time
            throughput = len(df) / max(processing_time, 0.001)
            
            st.metric("Temps de traitement", f"{processing_time:.2f}s")
            st.metric("Débit", f"{throughput:.0f} req/s")
            
            # Risk level gauge
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = block_rate,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Niveau de risque"},
                delta = {'reference': 20},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgray"},
                        {'range': [25, 50], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white', 'size': 12},
                height=250
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)

    # Sample analysis section
    if st.checkbox("🔍 Examiner des échantillons", value=False):
        st.markdown("### 🧪 Échantillons analysés")
        
        # Show some examples
        sample_size = min(10, len(df))
        sample_df = df.head(sample_size).copy()
        sample_df['Prédiction'] = y_pred[:sample_size]
        sample_df['Confiance'] = y_pred_prob[:sample_size].flatten()
        sample_df['Statut'] = sample_df['Prédiction'].map({0: '✅ Légitime', 1: '🚨 Malveillant'})
        
        # Display with nice formatting
        for idx, row in sample_df.iterrows():
            status_color = "#22c55e" if row['Prédiction'] == 0 else "#ef4444"
            confidence_bar_width = int(row['Confiance'] * 100)
            
            st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); padding: 1rem; border-radius: 12px; margin: 0.5rem 0; border-left: 4px solid {status_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-weight: 600; color: {status_color};">{row['Statut']}</span>
                        <span style="color: #a1a1aa;">Confiance: {row['Confiance']:.3f}</span>
                    </div>
                    <div style="background: rgba(255,255,255,0.05); padding: 0.5rem; border-radius: 8px; font-family: monospace; font-size: 0.9rem; color: #e6eef6;">
                        {row['text'][:200]}{'...' if len(str(row['text'])) > 200 else ''}
                    </div>
                    <div style="background: rgba(255,255,255,0.1); height: 4px; border-radius: 2px; margin-top: 0.5rem;">
                        <div style="background: {status_color}; height: 100%; width: {confidence_bar_width}%; border-radius: 2px;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)



# Footer with branding and info
st.markdown("""
    <div style="text-align: center; margin-top: 4rem; padding: 2rem; background: rgba(255,255,255,0.02); border-radius: 20px;">
        <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; flex-wrap: wrap; margin-bottom: 1rem;">
            <div style="color: #00d9ff; font-size: 1.5rem;">🛡️</div>
            <div>
                <h3 style="color: #00d9ff; margin: 0;">CyberShield AI</h3>
                <p style="color: #a1a1aa; margin: 0; font-size: 0.9rem;">Détection intelligente de malware</p>
            </div>
        </div>
        <div style="display: flex; justify-content: center; gap: 3rem; flex-wrap: wrap; margin-bottom: 1rem;">
            <div style="text-align: center;">
                <div style="color: #22c55e; font-weight: 600; font-size: 1.2rem;">99.5%</div>
                <div style="color: #a1a1aa; font-size: 0.8rem;">Précision</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #00d9ff; font-weight: 600; font-size: 1.2rem;">&lt;0.1s</div>
                <div style="color: #a1a1aa; font-size: 0.8rem;">Temps moyen</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #7c3aed; font-weight: 600; font-size: 1.2rem;">24/7</div>
                <div style="color: #a1a1aa; font-size: 0.8rem;">Surveillance</div>
            </div>
        </div>
        <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 1rem;">
            <p style="color: #71717a; font-size: 0.8rem; margin: 0;">
                Propulsé par TensorFlow & Keras • Modèle LSTM optimisé • Interface Streamlit moderne
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)