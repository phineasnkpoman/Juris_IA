"""Exporte la base ChromaDB vers un index leger (dossier index/) utilise par le site en ligne.

A relancer apres chaque `python indexer.py` :  python exporter_index.py
Les vecteurs des extraits sont repris tels quels (calcules par le modele complet) ;
seul l'encodage des questions passe par le modele ONNX compresse.
"""
import json
import os

import chromadb
import numpy as np

collection = chromadb.PersistentClient(path="./base_juridique").get_collection(name="textes_ivoiriens")
donnees = collection.get(include=["documents", "metadatas", "embeddings"])

vecteurs = np.array(donnees["embeddings"], dtype=np.float32)
vecteurs /= np.linalg.norm(vecteurs, axis=1, keepdims=True)

os.makedirs("index", exist_ok=True)
np.save("index/embeddings.npy", vecteurs.astype(np.float16))
with open("index/morceaux.json", "w", encoding="utf-8") as f:
    json.dump({"documents": donnees["documents"], "sources": [m["source"] for m in donnees["metadatas"]]}, f, ensure_ascii=False)
print(f"{len(vecteurs)} morceaux exportes dans index/")
