"""Fabrique une version allegee du modele d'embeddings, limitee au vocabulaire utile au francais.

Le modele multilingue d'origine connait 250 000 morceaux de mots (une centaine de langues), soit
l'essentiel de sa memoire. On garde ceux qui apparaissent dans les textes juridiques, plus les petits
morceaux en alphabet latin qui permettent de decouper n'importe quel mot. Le resultat va dans modele/.

Utilisation (en local, une seule fois) : python preparer_modele.py
puis, pour le publier :                 python preparer_modele.py --publier
"""
import copy
import json
import os
import re
import sys

import numpy as np
import onnx
from huggingface_hub import HfApi, hf_hub_download
from onnx import numpy_helper

from encodeur import DEPOT_MODELE as DEPOT_PUBLIE

DEPOT_MODELE = "Xenova/multilingual-e5-base"
FICHIER_MODELE = "onnx/model_quantized.onnx"

if "--publier" in sys.argv:
    api = HfApi()
    api.create_repo(DEPOT_PUBLIE, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path="modele", repo_id=DEPOT_PUBLIE, repo_type="model", commit_message="Modele allege JurisIA")
    print(f"Publie : https://huggingface.co/{DEPOT_PUBLIE}")
    sys.exit()

with open(hf_hub_download(DEPOT_MODELE, "tokenizer.json"), encoding="utf-8") as f:
    config_tok = json.load(f)
vocab = config_tok["model"]["vocab"]  # liste [morceau, score], l'indice est l'identifiant

from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_str(json.dumps(config_tok))
with open("index/morceaux.json", encoding="utf-8") as f:
    documents = json.load(f)["documents"]

garder = set(range(4))  # <s>, <pad>, </s>, <unk> gardent leurs identifiants
for lot in tok.encode_batch(documents):
    garder.update(lot.ids)
latin = re.compile(r"^▁?[A-Za-zÀ-ÿŒœ0-9'’.,;:!?()\-«»\"/%€$]{1,4}$")
garder.update(i for i, (morceau, _) in enumerate(vocab) if latin.match(morceau))
garder.add(len(vocab) - 1)  # <mask>, dernier identifiant
anciens = sorted(garder)
print(f"Vocabulaire : {len(vocab)} -> {len(anciens)} morceaux")

# Tokenizer reduit : memes morceaux et scores, renumerotes
nouveau_tok = copy.deepcopy(config_tok)
nouveau_tok["model"]["vocab"] = [vocab[i] for i in anciens]
nouvel_id = {ancien: nouveau for nouveau, ancien in enumerate(anciens)}
for jeton in nouveau_tok.get("added_tokens", []):
    jeton["id"] = nouvel_id[jeton["id"]]
pp = nouveau_tok.get("post_processor") or {}
for proc in [pp] + pp.get("processors", []):
    for spec in (proc.get("special_tokens") or {}).values():
        spec["ids"] = [nouvel_id[i] for i in spec["ids"]]

# Modele reduit : seules les lignes gardees de la table des mots
modele = onnx.load(hf_hub_download(DEPOT_MODELE, FICHIER_MODELE))
for init in modele.graph.initializer:
    if init.name == "embeddings.word_embeddings.weight_quantized":
        table = numpy_helper.to_array(init)
        init.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(table[anciens]), init.name))

os.makedirs("modele", exist_ok=True)
onnx.save(modele, "modele/model.onnx")
with open("modele/tokenizer.json", "w", encoding="utf-8") as f:
    json.dump(nouveau_tok, f, ensure_ascii=False)
print(f"Modele allege : {os.path.getsize('modele/model.onnx') >> 20} Mo dans modele/")
