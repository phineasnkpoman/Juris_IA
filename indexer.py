import os
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

DOSSIER_PDF = "textes_juridiques"

print("Chargement du modèle d'embeddings...")
embedder = SentenceTransformer("intfloat/multilingual-e5-base")

client_chroma = chromadb.PersistentClient(path="./base_juridique")
collection = client_chroma.get_or_create_collection(name="textes_ivoiriens")

def decouper_texte(texte, taille=800, chevauchement=100):
    morceaux = []
    debut = 0
    while debut < len(texte):
        fin = debut + taille
        morceaux.append(texte[debut:fin])
        debut += taille - chevauchement
    return morceaux

fichiers = [f for f in os.listdir(DOSSIER_PDF) if f.endswith(".pdf")]
print(f"{len(fichiers)} PDF trouvés.\n")

id_compteur = 0

for nom_fichier in fichiers:
    chemin = os.path.join(DOSSIER_PDF, nom_fichier)
    print(f"Traitement : {nom_fichier}")

    try:
        lecteur = PdfReader(chemin)
        texte_complet = ""
        for page in lecteur.pages:
            texte_complet += page.extract_text() + "\n"

        if len(texte_complet.strip()) < 50:
            print(f"  -> Ignoré (pas de texte extractible, peut-être scanné)")
            continue

        morceaux = decouper_texte(texte_complet)

        for morceau in morceaux:
            if len(morceau.strip()) < 30:
                continue
            embedding = embedder.encode(morceau).tolist()
            collection.add(
                ids=[f"doc_{id_compteur}"],
                embeddings=[embedding],
                documents=[morceau],
                metadatas=[{"source": nom_fichier}]
            )
            id_compteur += 1

        print(f"  -> {len(morceaux)} morceaux ajoutés")

    except Exception as e:
        print(f"  -> Erreur : {e}")

print(f"\nTerminé ! {id_compteur} morceaux au total indexés dans la base.")