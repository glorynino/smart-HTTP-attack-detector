import pandas as pd
import numpy as np
import unicodedata
import re
from urllib.parse import unquote, urlparse, parse_qs
import warnings

warnings.filterwarnings('ignore')

def normalize_col_name(c):
    c = str(c).strip().lower()
    c = re.sub(r'[\s\-]+', '_', c)            # remplace espaces/traits d'union par _
    c = re.sub(r'[^0-9a-z_]', '', c)          # retire ponctuations restantes
    return c

def guess_label_mapping(series):
    """Retourne une Series d'entiers (0/1) si possible, sinon NaN pour non-mappé."""
    s = series.astype(str).str.strip().str.lower()
    map_dict = {
        'norm': 0, 'normal': 0, 'legit': 0, 'legitimate': 0, 'benign': 0, '0': 0,
        'anom': 1, 'anomalous': 1, 'malicious': 1, 'malware': 1, '1': 1
    }
    mapped = s.map(map_dict)
    return mapped

def clean_csic_dataset(input_file, output_file="csic2010_clean_v2.csv",
                       drop_col_missing_thresh=0.9, drop_row_missing_thresh=0.8):
    print("[INFO] Chargement du dataset...")
    try:
        df = pd.read_csv(input_file, dtype=str, low_memory=False)
    except Exception as e:
        print(f"[ERREUR] Impossible de charger le fichier : {e}")
        return None

    print(f"[OK] Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # 1) Supprimer colonne index inutile
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
        print("[INFO] 'Unnamed: 0' supprimée (probable index exporté).")

    # 2) Normaliser noms de colonnes
    orig_cols = list(df.columns)
    df.columns = [normalize_col_name(c) for c in df.columns]
    print("[INFO] Colonnes normalisées :")
    print("  Avant:", orig_cols)
    print("  Après:", list(df.columns))

    # Corriger faute courante
    if 'lenght' in df.columns and 'length' not in df.columns:
        df.rename(columns={'lenght': 'length'}, inplace=True)
        print("[INFO] Colonne 'lenght' renommée en 'length'.")

    # 3) Détection colonne label
    label_col = None
    for cand in ['classification', 'label', 'classif', 'type']:
        if cand in df.columns:
            label_col = cand
            break

    if label_col:
        print(f"[INFO] Colonne d'étiquettes trouvée : '{label_col}'")
        df['label'] = guess_label_mapping(df[label_col])
        unmapped = df['label'].isna().sum()
        if unmapped > 0:
            print(f"[ATTENTION] {unmapped} valeurs d'étiquette non mappées -> à inspecter")
        else:
            print("[OK] Toutes les étiquettes mappées en 0/1.")
    else:
        print("[ATTENTION] Aucune colonne label trouvée.")

    initial_shape = df.shape

    # 4) Suppression doublons
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        df.drop_duplicates(inplace=True)
        print(f"[OK] {dup_count} doublons supprimés.")
    else:
        print("[OK] Aucun doublon trouvé.")

    # 5) Colonnes HTTP
    http_keywords = ['url', 'uri', 'path', 'query', 'method', 'user_agent', 'referer',
                     'host', 'content', 'payload', 'cookie', 'accept', 'accept_encoding',
                     'content_type', 'connection']
    http_cols = [c for c in df.columns if any(k in c for k in http_keywords)]
    print("[INFO] Colonnes HTTP identifiées :", http_cols)

    # 6) Nettoyage colonnes HTTP
    for col in df.columns:
        if col == 'label':
            continue
        df[col] = df[col].replace(['', 'NULL', 'null', 'None', 'none', 'nan', 'NaN'], np.nan)
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', np.nan)

    # Décodage URL + split
    for col in http_cols:
        if 'url' in col or 'uri' in col or col.endswith('_path'):
            def safe_unquote(x):
                try:
                    return unquote(x) if pd.notna(x) else x
                except:
                    return x
            decoded = df[col].apply(safe_unquote)
            df[col + '_decoded'] = decoded
            def split_url(u):
                try:
                    p = urlparse(u)
                    return p.path or '', p.query or ''
                except:
                    return '', ''
            df[[col + '_path', col + '_query']] = decoded.fillna('').apply(
                lambda x: pd.Series(split_url(x))
            )
            df[col + '_num_params'] = df[col + '_query'].apply(
                lambda q: len(parse_qs(q)) if q else 0
            )
            df[col + '_len'] = df[col].fillna('').str.len()

    # 7) Features heuristiques
    def contains_regex(col_series, pattern):
        return col_series.astype(str).str.contains(pattern, regex=True, na=False).astype(int)

    sqli_pat = r"(?i)(union\s+select|or\s+1=1|into\s+outfile|load_file\(|benchmark\(|sleep\(|--|/\*|\*/|\bexec\b)"
    xss_pat  = r"(?i)(<script|onerror=|onload=|javascript:|<img\b[^>]*on)"
    trav_pat = r"(\.\./|%2e%2e%2f)"

    if 'content' in df.columns:
        df['sqli_in_content'] = contains_regex(df['content'], sqli_pat)
        df['xss_in_content']  = contains_regex(df['content'], xss_pat)
        df['trav_in_content'] = contains_regex(df['content'], trav_pat)

    for c in [c for c in df.columns if c.endswith('_decoded')]:
        df[c.replace('_decoded', '_sqli')] = contains_regex(df[c], sqli_pat)
        df[c.replace('_decoded', '_xss')]  = contains_regex(df[c], xss_pat)
        df[c.replace('_decoded', '_trav')] = contains_regex(df[c], trav_pat)
        df[c.replace('_decoded', '_special_chars_count')] = df[c].fillna('').str.count(r'[^0-9a-zA-Z]')

    # 8) Colonnes trop manquantes
    col_missing_ratio = df.isnull().mean()
    to_drop_cols = col_missing_ratio[col_missing_ratio > drop_col_missing_thresh].index.tolist()
    if to_drop_cols:
        print(f"[ATTENTION] Colonnes trop manquantes supprimées : {to_drop_cols}")
        df.drop(columns=to_drop_cols, inplace=True)

    # 9) Lignes trop manquantes
    rows_to_drop_mask = df.isnull().sum(axis=1) > (len(df.columns) * drop_row_missing_thresh)
    if rows_to_drop_mask.sum() > 0:
        print(f"[ATTENTION] {rows_to_drop_mask.sum()} lignes très incomplètes (non supprimées).")

    final_shape = df.shape
    print(f"[INFO] Forme : {initial_shape} -> {final_shape}")

    # Distribution des labels
    if 'label' in df.columns:
        print("[INFO] Distribution 'label' (0=normal,1=malicious) :")
        print(df['label'].value_counts(dropna=False))

    # 🔹 Nouvelle étape : créer colonne 'text' pour LSTM
    if 'url_decoded' in df.columns and 'content' in df.columns:
        df['text'] = df['url_decoded'].fillna('') + " " + df['content'].fillna('')
    elif 'url_decoded' in df.columns:
        df['text'] = df['url_decoded'].fillna('')
    elif 'content' in df.columns:
        df['text'] = df['content'].fillna('')
    else:
        df['text'] = ''  # fallback

    print("[OK] Colonne 'text' créée pour LSTM.")

    # 10) Sauvegarde
    try:
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"[OK] Dataset nettoyé enregistré sous : {output_file}")
    except Exception as e:
        print(f"[ERREUR] Impossible de sauvegarder : {e}")

    return df

# UTILISATION
if __name__ == "__main__":
    cleaned = clean_csic_dataset("DATA/csic_database.csv")
    if cleaned is not None:
        print(cleaned[['label','text']].head())
