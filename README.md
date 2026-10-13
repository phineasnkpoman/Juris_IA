# JurisIA ⚖️🦉

Chatbot qui rend le **droit ivoirien accessible à tous**. Il répond en français simple à partir de textes juridiques officiels (technique RAG), avec citation des sources, saisie vocale et lecture à voix haute.

## Fonctionnement
PDF juridiques → découpage en morceaux → embeddings (`intfloat/multilingual-e5-base`) → base ChromaDB → recherche des extraits pertinents → réponse générée par un modèle Groq.

## Installation (Windows / PowerShell)
```powershell
git clone https://github.com/TON-PSEUDO/jurisia.git
cd jurisia
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
1. Mets tes PDF dans `textes_juridiques/`
2. Construis la base : `python indexer.py`
3. Lance le site : `python app.py` puis ouvre http://127.0.0.1:5000

## Avertissement
JurisIA fournit une information générale et ne remplace pas un avocat, un notaire ou un huissier.
