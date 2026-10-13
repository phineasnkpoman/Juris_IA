import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Chargement des variables d'environnement depuis le fichier .env
load_dotenv()

# Récupération sécurisée du token depuis l'environnement
TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not TOKEN:
    raise ValueError("Le token HUGGINGFACEHUB_API_TOKEN est introuvable dans le fichier .env")

client = InferenceClient(token=TOKEN)

MODEL = "meta-llama/Llama-3.3-70B-Instruct"

history = [
    {"role": "system", "content": "Tu es un assistant sympathique et utile."}
]

print("Chatbot prêt (via l'API Hugging Face) ! Tape 'quit' pour arrêter.\n")

while True:
    user_input = input("Toi : ")
    if user_input.lower() == "quit":
        break

    history.append({"role": "user", "content": user_input})

    response = client.chat_completion(
        model=MODEL,
        messages=history,
        max_tokens=300
    )

    bot_reply = response.choices[0].message.content
    print(f"Bot : {bot_reply}\n")

    history.append({"role": "assistant", "content": bot_reply})