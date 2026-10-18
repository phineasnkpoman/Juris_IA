# JurisIA ⚖️🦉

Chatbot qui rend le **droit ivoirien accessible à tous**. Il répond en français simple à partir de textes juridiques officiels (technique RAG), avec citation des sources, saisie vocale et lecture à voix haute.

## Fonctionnement
PDF juridiques → découpage en morceaux → embeddings (`intfloat/multilingual-e5-base`) → base ChromaDB → index léger (`index/`) → recherche des extraits pertinents → réponse générée par un modèle Groq.

Le site encode les questions avec une version ONNX allégée du modèle ([Phiineass/jurisia-e5-fr-onnx](https://huggingface.co/Phiineass/jurisia-e5-fr-onnx)) : il tourne avec environ 300 Mo de mémoire, sans PyTorch.

## Installation (Windows / PowerShell)
```powershell
git clone https://github.com/phineasnkpoman/Juris_IA.git
cd Juris_IA
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration
Copie `.env.example` en `.env` puis colle ta clé Groq (gratuite sur https://console.groq.com) :
```
GROQ_API_KEY=gsk_...
```

## Utilisation
Lance le site : `python app.py` puis ouvre http://127.0.0.1:5000

## Ajouter des textes de loi
1. Installe les outils d'indexation : `pip install -r requirements-indexation.txt`
2. Mets tes PDF dans `textes_juridiques/`
3. Construis la base : `python indexer.py`
4. Exporte l'index léger utilisé par le site : `python exporter_index.py`
5. Pousse le dossier `index/` sur GitHub : Render redéploie le site automatiquement.

## Mise en ligne (Render, gratuit)
Le fichier `render.yaml` décrit le service. Sur Render : **New → Blueprint**, choisis ce dépôt, puis renseigne la variable `GROQ_API_KEY`.

## Avertissement
JurisIA fournit une information générale et ne remplace pas un avocat, un notaire ou un huissier.
