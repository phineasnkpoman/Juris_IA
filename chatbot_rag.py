import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import chromadb
from sentence_transformers import SentenceTransformer

# Chargement des variables d'environnement depuis le fichier .env
load_dotenv()

# --- Configuration ---
TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # Récupéré de manière sécurisée

if not TOKEN:
    raise ValueError("Le token HUGGINGFACEHUB_API_TOKEN est introuvable dans le fichier .env")

MODEL = "meta-llama/Llama-3.3-70B-Instruct"  # ou celui qui fonctionnait chez toi

# --- Chargement des outils ---
print("Chargement des outils...")
embedder = SentenceTransformer("intfloat/multilingual-e5-base")
client_chroma = chromadb.PersistentClient(path="./base_juridique")
collection = client_chroma.get_collection(name="textes_ivoiriens")
client = InferenceClient(token=TOKEN)

def rechercher_contexte(question, n_resultats=5):
    embedding_question = embedder.encode(question).tolist()
    resultats = collection.query(
        query_embeddings=[embedding_question],
        n_results=n_resultats
    )
    morceaux = resultats["documents"][0]
    sources = [meta["source"] for meta in resultats["metadatas"][0]]
    return morceaux, sources

def construire_prompt(question, morceaux, sources):
    contexte = ""
    for i, (morceau, source) in enumerate(zip(morceaux, sources)):
        contexte += f"\n[Extrait {i+1} - Source: {source}]\n{morceau}\n"

    prompt_systeme = f"""Tu es un assistant juridique spécialisé en droit ivoirien.
Réponds UNIQUEMENT à partir des extraits de textes juridiques fournis ci-dessous.
Si les extraits ne contiennent pas l'information nécessaire, dis-le clairement, n'invente rien.
Cite toujours la source (nom du document) sur laquelle tu t'appuies.

EXTRAITS DE TEXTES JURIDIQUES :
{contexte}
"""
    return prompt_systeme

print("Chatbot JurisIA prêt ! Tape 'quit' pour arrêter.\n")

while True:
    question = input("Toi : ")
    if question.lower() == "quit":
        break

    morceaux, sources = rechercher_contexte(question)
    prompt_systeme = construire_prompt(question, morceaux, sources)

    messages = [
        {"role": "system", "content": prompt_systeme},
        {"role": "user", "content": question}
    ]

    response = client.chat_completion(
        model=MODEL,
        messages=messages,
        max_tokens=500
    )

    reponse = response.choices[0].message.content
    print(f"\nBot : {reponse}\n")
    print(f"(Sources consultées : {', '.join(set(sources))})\n")