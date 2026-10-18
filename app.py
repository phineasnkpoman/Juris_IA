import json
import os
import re

import numpy as np
from flask import Flask, render_template, request, jsonify
from groq import Groq

from encodeur import Encodeur

# --- Configuration ---
# La cle Groq est lue dans le fichier .env (jamais dans le code, pour ne pas la publier sur GitHub)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # sans python-dotenv, on utilise la variable d'environnement GROQ_API_KEY
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_KEY:
    raise SystemExit("Cle manquante : cree un fichier .env contenant GROQ_API_KEY=gsk_... (voir .env.example)")
# Modeles preferes, du plus souhaite au moins souhaite. Le premier disponible sur ta cle est utilise.
MODELES_PREFERES = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
]
EXCLUS = ("whisper", "tts", "guard", "orpheus", "embed", "playai", "safeguard", "compound")

app = Flask(__name__)

# --- Chargement des outils (une seule fois au demarrage) ---
print("Chargement des outils...")
encodeur = Encodeur()
# Index leger produit par exporter_index.py (vecteurs normalises + textes)
vecteurs_index = np.load("index/embeddings.npy").astype(np.float32)
with open("index/morceaux.json", encoding="utf-8") as f:
    morceaux_index = json.load(f)
# Les requetes partent directement chez Groq avec ta cle
client = Groq(api_key=GROQ_KEY)


def choisir_modele():
    """Demande a Groq quels modeles ta cle peut utiliser et prend le meilleur disponible."""
    force = os.environ.get("GROQ_MODEL")
    if force:
        return force
    try:
        ids = [m.id for m in client.models.list().data]
        ids_chat = [i for i in ids if not any(x in i.lower() for x in EXCLUS)]
        print("Modeles disponibles sur ta cle :", ", ".join(ids_chat) or "(aucun)")
        for prefere in MODELES_PREFERES:
            if prefere in ids_chat:
                return prefere
        if ids_chat:
            return ids_chat[0]
    except Exception as e:
        print("Impossible de lister les modeles Groq :", e)
    return MODELES_PREFERES[0]


MODEL = choisir_modele()
print("Modele utilise :", MODEL)
print("Pret !")


def rechercher_contexte(question, theme="", n_resultats=5):
    requete = f"{theme} : {question}" if theme else question
    vecteur_question = encodeur.encoder([requete])[0]
    # Vecteurs normalises : le produit scalaire est la similarite cosinus
    meilleurs = np.argsort(-(vecteurs_index @ vecteur_question))[:n_resultats]
    morceaux = [morceaux_index["documents"][i] for i in meilleurs]
    sources = [morceaux_index["sources"][i] for i in meilleurs]
    return morceaux, sources


def construire_prompt(morceaux, sources, theme=""):
    contexte = ""
    for i, (morceau, source) in enumerate(zip(morceaux, sources)):
        contexte += f"\n[Extrait {i+1} - Source: {source}]\n{morceau}\n"

    focus = f"\nLa question concerne le theme : {theme}.\n" if theme else ""

    return f"""Tu es JurisIA (la mascotte s'appelle Maitre Hibou), un assistant juridique specialise en droit ivoirien, destine au grand public.
Reponds en francais simple, sans jargon, avec des phrases courtes.

Il y a trois cas :
1. Salutation, remerciement, ou question sur toi et ce que tu sais faire : reponds naturellement et chaleureusement, SANS utiliser les extraits, puis invite la personne a poser sa question de droit.
2. Question sans rapport avec le droit (culture generale, explication, conseil, redaction, code, etc.) : reponds-y normalement et utilement avec tes propres connaissances, comme le ferait un assistant general, SANS utiliser les extraits. Ensuite, termine par une courte phrase qui explique poliment que tu es specialise en droit ivoirien et que tu peux aider sur ce sujet.
3. Question de droit : reponds UNIQUEMENT a partir des extraits de textes juridiques ci-dessous. Si les extraits ne permettent pas de repondre, dis-le clairement, n'invente rien, et suggere de reformuler la question ou de consulter un professionnel. Cite le nom du document sur lequel tu t'appuies. Quand c'est utile, structure la reponse en petites listes a puces. Termine par une courte phrase conseillant de consulter un professionnel (avocat, notaire, huissier) pour une decision importante.

IMPORTANT : commence ta reponse par [[OUI]] si elle s'appuie sur les extraits ci-dessous, ou par [[NON]] dans les autres cas (cas 1, cas 2, ou extraits insuffisants). Ce marqueur est retire avant l'affichage.
{focus}
EXTRAITS DE TEXTES JURIDIQUES :
{contexte}
"""


@app.route("/")
def accueil():
    return render_template("index.html")


@app.route("/demander", methods=["POST"])
def demander():
    donnees = request.get_json(silent=True) or {}
    question = (donnees.get("question") or "").strip()
    theme = (donnees.get("theme") or "").strip()[:80]
    historique = donnees.get("historique") or []

    if not question:
        return jsonify({"erreur": "Question vide."}), 400
    question = question[:1500]

    try:
        morceaux, sources = rechercher_contexte(question, theme)
        prompt_systeme = construire_prompt(morceaux, sources, theme)

        messages = [{"role": "system", "content": prompt_systeme}]
        # On garde quelques echanges recents pour que le bot suive la conversation
        for m in historique[-6:]:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                messages.append({"role": m["role"], "content": str(m.get("content", ""))[:3000]})
        messages.append({"role": "user", "content": question})

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=1500,  # marge : ce modele "reflechit" avant de repondre
        )
        reponse = (response.choices[0].message.content or "").strip()
        sources_uniques = list(dict.fromkeys(sources))  # sans doublons, ordre conserve

        # Le modele indique avec [[OUI]] / [[NON]] s'il s'est appuye sur les extraits.
        marqueur = re.match(r"\s*\[\[(OUI|NON)\]\]\s*", reponse, re.IGNORECASE)
        if marqueur:
            reponse = reponse[marqueur.end():]
            if marqueur.group(1).upper() == "NON":
                sources_uniques = []  # pas de sources affichees pour un simple echange ou un hors-sujet

        return jsonify({"reponse": reponse, "sources": sources_uniques})

    except Exception as e:
        print("Erreur /demander :", e)
        return jsonify({"erreur": "Le service est momentanement indisponible."}), 500


if __name__ == "__main__":
    app.run(debug=True)