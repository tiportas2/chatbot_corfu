import os
import json

BASE_DIR = "historico_data"

def guardar_conversa(user_id, conversa_id, historico):
    pasta = os.path.join(BASE_DIR, f"user_{user_id}")
    os.makedirs(pasta, exist_ok=True)
    caminho = os.path.join(pasta, f"{conversa_id}.json")
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(historico, f, ensure_ascii=False, indent=2)

def carregar_conversa(user_id, conversa_id):
    caminho = os.path.join(BASE_DIR, f"user_{user_id}", f"{conversa_id}.json")
    if os.path.exists(caminho):
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def listar_conversas(user_id):
    pasta = os.path.join(BASE_DIR, f"user_{user_id}")
    if not os.path.exists(pasta):
        return []
    return [f.replace(".json", "") for f in os.listdir(pasta) if f.endswith(".json")]


def apagar_conversa(user_id, conversa_id):
    caminho = os.path.join(BASE_DIR, f"user_{user_id}", f"{conversa_id}.json")
    if os.path.exists(caminho):
        os.remove(caminho)

def apagar_todas_conversas(user_id):
    pasta = os.path.join(BASE_DIR, f"user_{user_id}")
    if os.path.exists(pasta):
        for ficheiro in os.listdir(pasta):
            if ficheiro.endswith(".json"):
                os.remove(os.path.join(pasta, ficheiro))
