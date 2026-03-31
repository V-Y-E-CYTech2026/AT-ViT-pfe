import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

# --- 1. Chargement de la Configuration depuis le .env ---
load_dotenv()

DOSSIER_IMAGES = os.getenv("ORIGINAL_IMG_DIR")
BASE_DIR = os.getenv("BASE_DIR")
CSV_FILENAME = os.getenv("CSV_FILENAME")

# Sécurité : vérifier que les variables ont bien été trouvées
if not all([DOSSIER_IMAGES, BASE_DIR, CSV_FILENAME]):
    raise ValueError("Erreur : Vérifiez que ORIGINAL_IMG_DIR, BASE_DIR et CSV_FILENAME sont bien définis dans le fichier .env")

CHEMIN_SORTIE_CSV = os.path.join(BASE_DIR, CSV_FILENAME)

# --- 2. Dictionnaire de correspondance (Mapping multi-traits) ---
MAPPING_ESPECES = {
    "amborella":      {"acuminate_tips": 1, "feuille_base_aigue": 1, "thorns": 0},
    "castanea":       {"acuminate_tips": 1, "feuille_base_aigue": 1, "thorns": 0},
    "desmodium":      {"acuminate_tips": 1, "feuille_base_aigue": 1, "thorns": 0},
    "ulmus":          {"acuminate_tips": 1, "feuille_base_aigue": 1, "thorns": 0},
    "rubus":          {"acuminate_tips": 1, "feuille_base_aigue": 1, "thorns": 1},
    "litsea":         {"acuminate_tips": 1, "feuille_base_aigue": 1, "thorns": 0},
    "eugenia":        {"acuminate_tips": 1, "feuille_base_aigue": 1, "thorns": 0},
    "laurus":         {"acuminate_tips": 1, "feuille_base_aigue": 1, "thorns": 0},
    "convolvulaceae": {"acuminate_tips": 1, "feuille_base_aigue": 1, "thorns": 0},
    "magnolia":       {"acuminate_tips": 1, "feuille_base_aigue": 1, "thorns": 0},
    "monimiaceae":    {"acuminate_tips": 1, "feuille_base_aigue": 1, "thorns": 0}
}

def extraire_espece(nom_fichier):
    """Extrait 'amborella' depuis 'amborella2.jpg'"""
    nom_sans_ext = os.path.splitext(nom_fichier)[0]
    return re.sub(r'\d+$', '', nom_sans_ext).lower()

# --- 3. Construction des données ---
donnees = []

print(f"Lecture des images depuis : {DOSSIER_IMAGES}")

for fichier in os.listdir(DOSSIER_IMAGES):
    if fichier.lower().endswith(('.jpg', '.png', '.jpeg')):
        code_image = os.path.splitext(fichier)[0] # Ex: 'amborella2'
        espece = extraire_espece(fichier)         # Ex: 'amborella'
        
        traits = MAPPING_ESPECES.get(espece)
        
        if traits:
            ligne = {"code": code_image, "espece": espece}
            ligne.update(traits)
            donnees.append(ligne)
        else:
            print(f"Attention : Espèce '{espece}' non trouvée (fichier: {fichier})")

df = pd.DataFrame(donnees)
print(f"Total d'images prêtes : {len(df)}")

# --- 4. Séparation Train / Test ---
train_df, test_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42, 
    stratify=df["espece"]
)

train_df["train_test_set"] = "train"
test_df["train_test_set"] = "test"

# --- 5. Finalisation et Sauvegarde ---
df_final = pd.concat([train_df, test_df])

# Suppression de la colonne temporaire
df_final = df_final.drop(columns=["espece"])

# Sauvegarde au bon endroit grâce au .env
df_final.to_csv(CHEMIN_SORTIE_CSV, index=False)

print(f"\nFichier créé avec succès à cet emplacement : {CHEMIN_SORTIE_CSV}")
print("Aperçu :")
print(df_final.head())