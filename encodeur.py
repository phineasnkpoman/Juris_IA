"""Calcul des embeddings des questions avec une version ONNX allegee de intfloat/multilingual-e5-base.

Remplace sentence-transformers + PyTorch (plus de 1 Go de RAM) par onnxruntime (environ 150 Mo),
pour que le site tienne sur un hebergement gratuit. Le modele allege est fabrique par preparer_modele.py.
"""
import os

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

# Modele allege publie sur Hugging Face (utilise si le dossier local modele/ est absent, par exemple en ligne)
DEPOT_MODELE = "Phiineass/jurisia-e5-fr-onnx"
DOSSIER_LOCAL = "modele"


def chemin(fichier):
    local = os.path.join(DOSSIER_LOCAL, fichier)
    return local if os.path.exists(local) else hf_hub_download(DEPOT_MODELE, fichier)


class Encodeur:
    def __init__(self):
        self.tok = Tokenizer.from_file(chemin("tokenizer.json"))
        self.tok.enable_truncation(512)
        self.tok.enable_padding(pad_id=1, pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(chemin("model.onnx"), options, providers=["CPUExecutionProvider"])
        self.entrees = {e.name for e in self.session.get_inputs()}

    def encoder(self, textes):
        """Renvoie une matrice (n, 768) de vecteurs normalises."""
        lots = self.tok.encode_batch(list(textes))
        ids = np.array([l.ids for l in lots], dtype=np.int64)
        masque = np.array([l.attention_mask for l in lots], dtype=np.int64)
        flux = {"input_ids": ids, "attention_mask": masque}
        if "token_type_ids" in self.entrees:
            flux["token_type_ids"] = np.zeros_like(ids)
        etats = self.session.run(None, flux)[0].astype(np.float32)
        # Moyenne sur les vrais jetons (sans le remplissage), comme sentence-transformers
        vecteurs = (etats * masque[..., None]).sum(1) / masque.sum(1, keepdims=True)
        return vecteurs / np.linalg.norm(vecteurs, axis=1, keepdims=True)
