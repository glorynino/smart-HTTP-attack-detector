import pandas as pd
#charger le fichier 
df = pd.read_csv("output_http_csic_2010_weka_with_duplications_RAW-RFC2616_escd_v02_full.csv",
                 dtype=str,
                 low_memory = False
                 )
print(df.head())


#conversion des label 1

df['label'] = df['label'].map({'norm': 0, 'anom': 1})
print(df['label'].value_counts())  # distribution des classes

# === 2. Suppression des doublons ===
print("[INFO] Suppression des doublons...")
df.drop_duplicates(inplace=True)


# === 3 tout passer en minisucule et supprimer les accent ===
df.str.lower()
df.unicodedata.normalize('NFKC', df.str.capitalize())

df.str.replace(r'^\s*$', df.nan, regex=True)



# === 6. Sauvegarde du dataset propre ===
output_file = "csic2010_clean.csv"
df.to_csv(output_file, index=False)
print(f"[OK] Dataset nettoyé enregistré sous : {output_file}")