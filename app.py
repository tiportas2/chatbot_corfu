import os
from dotenv import load_dotenv

# Carregar o ficheiro .env
load_dotenv()
print("✅ Endpoint:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("✅ Key:", os.getenv("AZURE_OPENAI_KEY")[:6] + "..." + os.getenv("AZURE_OPENAI_KEY")[-6:])
print("✅ Version:", os.getenv("AZURE_OPENAI_API_VERSION"))
print("✅ Deployment:", os.getenv("AZURE_OPENAI_DEPLOYMENT"))

import streamlit as st
# from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path
from PIL import Image

import unicodedata
import re

import base64
from io import BytesIO

# from streamlit_extras.cookies import get_cookie, set_cookie  ❌ (antigo e obsoleto)
from streamlit_cookies_manager import EncryptedCookieManager 

import uuid

import streamlit as st
from memoria import listar_conversas, carregar_conversa, guardar_conversa, apagar_conversa, apagar_todas_conversas

def gerar_nome_conversa_baseado_na_pergunta(pergunta):
    nome = pergunta.lower()
    nome = re.sub(r"[^\w\s]", "", nome)  # remove pontuação
    nome = "_".join(nome.split()[:5])    # usa até 5 palavras
    return nome


# Criar user_id único se ainda não existir
# 🍪 Ler ou criar user_id persistente
cookies = EncryptedCookieManager(password=os.getenv("COOKIE_SECRET", "default123"))

if not cookies.ready():
    st.stop()  # Aguarda cookies estarem prontos

if not cookies.get("user_id"):
    new_id = str(uuid.uuid4())
    cookies["user_id"] = new_id
    cookies.save()
    user_id = new_id
else:
    user_id = cookies.get("user_id")

st.session_state.user_id = user_id

if "confirmar_apagar_tudo" not in st.session_state:
    st.session_state.confirmar_apagar_tudo = False

# Inicializar ID da conversa
if "conversa_id" not in st.session_state:
    st.session_state.conversa_id = "conversa_1"

# Inicializar lista de conversas se necessário
if "conversas" not in st.session_state:
    st.session_state.conversas = {}


# Tentar carregar histórico do disco se ainda não houver
if "chat_history" not in st.session_state:
    historico_guardado = carregar_conversa(
        st.session_state.user_id,
        st.session_state.conversa_id
    )
    st.session_state.chat_history = historico_guardado or []

# Carregar lista de conversas
conversas_disponiveis = listar_conversas(st.session_state.user_id)
# Preencher dicionário com os nomes das conversas e respetivo conteúdo (se quiseres usar mais tarde)
for c in conversas_disponiveis:
    if c not in st.session_state.conversas:
        st.session_state.conversas[c] = carregar_conversa(st.session_state.user_id, c) 

# Garante que a conversa atual está na lista
if st.session_state.conversa_id not in conversas_disponiveis:
    conversas_disponiveis = [st.session_state.conversa_id] + conversas_disponiveis


# Sidebar interativa
with st.sidebar:
    st.markdown("## 🧠 Sessão")
    st.write(f"👤 User ID: `{st.session_state.user_id}`")
    st.write(f"💬 Conversa ID: `{st.session_state.conversa_id}`")

    # Seletor de conversas existentes
    conversa_escolhida = st.radio(
        "📁 Conversas:",
        conversas_disponiveis,
        index=conversas_disponiveis.index(st.session_state.conversa_id)
        if st.session_state.conversa_id in conversas_disponiveis else 0
    )

    # Trocar conversa
    if conversa_escolhida != st.session_state.conversa_id:
        st.session_state.conversa_id = conversa_escolhida
        st.session_state.chat_history = carregar_conversa(
            st.session_state.user_id,
            st.session_state.conversa_id
        )

    # Botão Nova conversa
    if st.button("➕ Nova conversa"):
        # Guardar conversa atual antes de trocar
        guardar_conversa(
            st.session_state.user_id,
            st.session_state.conversa_id,
            st.session_state.chat_history
        )

        # Gerar novo nome único (conversa_1, conversa_2, etc)
        idx = 1
        while f"conversa_{idx}" in conversas_disponiveis:
            idx += 1
        novo_nome = f"conversa_{idx}"

        # Atualizar sessão
        st.session_state.conversa_id = novo_nome
        st.session_state.chat_history = []

        # Guardar nova conversa
        guardar_conversa(st.session_state.user_id, novo_nome, [])

        # Reset da flag de sugestões
        st.session_state.primeira_interacao = True

        # Forçar refresh da app para atualizar lista
        st.rerun()

    # Botão Limpar conversa atual
    if st.button("🧼 Limpar conversa atual"):
        st.session_state.chat_history = []
        guardar_conversa(
            st.session_state.user_id,
            st.session_state.conversa_id,
            []
        )

    # Botão Apagar conversa atual
    if st.button("🗑️ Apagar conversa atual"):
        apagar_conversa(
            st.session_state.user_id,
            st.session_state.conversa_id
        )

        # Atualizar estado da aplicação
        conversas_restantes = listar_conversas(st.session_state.user_id)
        if conversas_restantes:
            nova_conversa = conversas_restantes[0]
            st.session_state.conversa_id = nova_conversa
            st.session_state.chat_history = carregar_conversa(
                st.session_state.user_id, nova_conversa
            )
        else:
            st.session_state.conversa_id = "conversa_1"
            st.session_state.chat_history = []
            guardar_conversa(st.session_state.user_id, "conversa_1", [])

        st.success("✅ Conversa apagada com sucesso.")
        st.rerun()

    # Botão Apagar todas as conversas
    with st.expander("⚠️ Opções avançadas"):
        st.markdown("### 🔥 Apagar todas as conversas")

        if st.button("❌ Quero apagar tudo"):
            st.session_state.confirmar_apagar_tudo = True

        if st.session_state.get("confirmar_apagar_tudo", False):
            st.warning("⚠️ Tens a certeza que queres apagar todas as conversas? Esta ação não pode ser revertida.")
            col1, col2 = st.columns([1, 2])

            with col1:
                if st.button("✅ Sim, apagar tudo agora"):
                    apagar_todas_conversas(st.session_state.user_id)
                    st.session_state.conversa_id = "conversa_1"
                    st.session_state.chat_history = []
                    guardar_conversa(st.session_state.user_id, "conversa_1", [])
                    st.success("🧨 Todas as conversas foram apagadas com sucesso.")
                    st.session_state.confirmar_apagar_tudo = False
                    st.rerun()

            with col2:
                if st.button("❌ Cancelar"):
                    st.session_state.confirmar_apagar_tudo = False



# Guardar sempre que houver histórico (podes mover para o fim do teu `chat handler`)
guardar_conversa(
    st.session_state.user_id,
    st.session_state.conversa_id,
    st.session_state.chat_history
)

from langchain.memory import ConversationBufferWindowMemory

# Inicializar memória de janela com as últimas 4 mensagens
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=4,  # número de interações (user+AI) a manter
        return_messages=True
    )

memory = st.session_state.memory


def img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


# ======================
# 1️⃣ Azure Configuração
# ======================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "ChatGPT")

llm = AzureChatOpenAI(
    temperature=0.7,
    model=AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    openai_api_type="azure",
    api_key=AZURE_OPENAI_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)

# ======================
# 2️⃣ Carregar documentos
# ======================
base_path = Path("rag_docs")
documentos = []

for folder in base_path.iterdir():
    if folder.is_dir():
        tipo = folder.name.lower()
        for file in folder.glob("*.txt"):
            with open(file, "r", encoding="utf-8") as f:
                conteudo = f.read()
            documentos.append({
                "nome": file.name,
                "conteudo": conteudo,
                "tipo": tipo,
                "id": file.stem.lower()
            })

# ======================
# 3️⃣ Criar chunks
# ======================
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

def dividir_em_chunks(documentos, splitter):
    chunks = []
    for doc in documentos:
        partes = splitter.split_text(doc["conteudo"])
        for i, parte in enumerate(partes):
            if len(parte.strip()) > 50:
                chunks.append({
                    "nome": doc["nome"],
                    "tipo": doc["tipo"],
                    "id": doc["id"],
                    "chunk_id": i,
                    "conteudo": parte
                })
    return chunks

document_chunks = dividir_em_chunks(documentos, splitter)

# ======================
# 4️⃣ Gerar embeddings e carregar base vetorial
# ======================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


docs_formatados = [
    Document(page_content=chunk["conteudo"], metadata={
        "nome": chunk["nome"],
        "tipo": chunk["tipo"],
        "id": chunk["id"],
        "chunk_id": chunk["chunk_id"]
    }) for chunk in document_chunks
]

db_path = "chroma_db"
def criar_ou_carregar_faiss(docs, embedding):
    db = FAISS.from_documents(documents=docs, embedding=embedding)
    return db

chroma_db = criar_ou_carregar_faiss(docs_formatados, embedding_model)

# ======================
# 5️⃣ Funções RAG
# ======================

# ======================
# 
# ======================
def normalizar(texto):
    texto = texto.lower()
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    return texto

def classificar_pergunta(pergunta):
    pergunta_norm = normalizar(pergunta)

    nomes_simples = {
        "vasco", "andre", "pato", "joao", "jota",
        "diogo", "rodrigo", "miranda", "ralph", "piloto", "paty"
    }

    nomes_compostos = {
        "monsieur ralph", "cara de ovo"
    }

    palavras_factuais = {
        "voo", "voos", "voar", "chegada", "chegar", "chegou",
        "partida", "partir", "data", "datas", "hora",
        "preco", "preço", "custos", "custo", "custaram",
        "valor", "pagamento", "quanto", "quantos",
        "aluguer", "seguro", "atividade", "atividades",
        "cruzeiro", "excursao", "passeio", "programa",
        "porto", "gruta", "bilhete", "confirmacao",
        "condutor", "condutores", "motorista", "carta"
    }

    palavras_pessoais = {
        "alcunha", "alcunhas", "apelido", "nome", "personagem", "quem é"
    }

    # Detetar nomes simples com regex (palavras isoladas)
    if any(re.search(rf'\b{nome}\b', pergunta_norm) for nome in nomes_simples):
        return "pessoal"

    # Detetar nomes compostos ou termos pessoais com match direto
    if any(term in pergunta_norm for term in nomes_compostos.union(palavras_pessoais)):
        return "pessoal"

    if any(p in pergunta_norm for p in palavras_factuais):
        return "factual"

    return "generica"


# ======================
# 
# ======================
def rerank_heuristico(resultados_com_score, pergunta):
    pergunta_lower = pergunta.lower()

    # Termos típicos de perguntas factuais
    termos_prioridade = [
        "voo", "voos", "ida", "regresso", "partida", "chegada",
        "escalas", "paris", "milao", "corfu", "aeroporto", "datas"
    ]

    # Nomes associados a participantes
    termos_pessoais = ["vasco", "andre", "pato", "joao", "jota", "diogo", "rodrigo", "miranda", "ralph"]

    # Inferir tipo de pergunta
    if any(t in pergunta_lower for t in termos_prioridade):
        tipo_esperado = "logistica"
    elif any(n in pergunta_lower for n in termos_pessoais):
        tipo_esperado = "participantes"
    else:
        tipo_esperado = None

    # Opcional: deteção de nome pessoal mencionado (para reforçar chunks certos)
    nome_mencionado = next((n for n in termos_pessoais if n in pergunta_lower), None)

    def pontuar(r, score):
        bonus = 0

        # Pontos por termos factuais no conteúdo
        bonus += 2 * sum(t in r.page_content.lower() for t in termos_prioridade)

        # Pontos por tipo correto (logistica ou participantes)
        if tipo_esperado and r.metadata.get("tipo") == tipo_esperado:
            bonus += 3

        # Pontos por id coincidir com nome mencionado
        if nome_mencionado and nome_mencionado in r.metadata.get("id", "").lower():
            bonus += 3

        return score - 0.1 * bonus

    return sorted(resultados_com_score, key=lambda r_score: pontuar(*r_score))
# ======================
# 
# ======================
def responder_factual(pergunta, resultados):
    memory = st.session_state.memory

    documentos = [r for r, _ in resultados]
    logistica_chunks = [r for r in documentos if r.metadata["tipo"] == "logistica"]
    termos_relevantes = ["voo", "seguro", "carro", "chegada", "partida", "corfu"]

    prioridade_chunks = [
        r for r in logistica_chunks
        if any(t in r.page_content.lower() for t in termos_relevantes)
    ]

    contexto = "\n\n".join(r.page_content for r in prioridade_chunks) if prioridade_chunks else "\n\n".join(r.page_content for r in logistica_chunks)

    # ✅ Novo system_message com fluidez e precisão
    system_message = (
        "Estás a responder como um assistente de viagem do grupo que vai a Corfu em agosto de 2025. "
        "A conversa deve ser fluída e natural — usa o histórico anterior para perceber o contexto e dar continuidade. "
        "As tuas respostas devem ser diretas, claras e baseadas exclusivamente no contexto fornecido. "
        "Não inventes dados. Se não houver informação suficiente, diz isso com simplicidade. "
        "Se a pergunta for vaga (ex: 'e o seguro?'), deves assumir que a conversa está em andamento e responder com base no que for mais relevante.\n\n"
        "Exemplos:\n"
        "- Pergunta: 'Qual é o seguro do carro?' → Resposta: 'O seguro é da Collinson, custa 55,80€ e cobre danos e roubo de 8 a 14 de agosto em Corfu.'\n"
        "- Pergunta: 'Quanto custaram os carros?' → Resposta: 'O aluguer dos carros foi de 117,40€ por pessoa.'\n"
        "- Pergunta: 'Quem são os condutores do carro?' → Resposta: 'Os condutores são Rodrigo Miranda e Rodrigo Pato, conforme indicado na apólice do seguro.'"
    )

    # ✅ Corrigir roles: transformar "human"/"ai" em "user"/"assistant"
    historico_formatado = []
    for m in memory.chat_memory.messages:
        if m.type == "human":
            historico_formatado.append({"role": "user", "content": m.content})
        elif m.type == "ai":
            historico_formatado.append({"role": "assistant", "content": m.content})

    # ✅ Construir prompt completo com histórico
    mensagens = (
        [{"role": "system", "content": system_message}]
        + historico_formatado
        + [{"role": "user", "content": f"Pergunta: {pergunta}\n\nContexto:\n{contexto}"}]
    )

    resposta = llm.invoke(mensagens).content

    # ✅ Atualizar memória com nova interação
    memory.chat_memory.add_user_message(pergunta)
    memory.chat_memory.add_ai_message(resposta)
    st.session_state.mensagens = mensagens

    return resposta



def responder_pessoal(pergunta, resultados):
    memory = st.session_state.memory

    documentos = [r for r, _ in resultados]
    participantes_chunks = [r for r in documentos if r.metadata["tipo"] == "participantes"]

    mapa_nomes = {
        "vasco": "vasco_machado",
        "andre": "andre_marques",
        "andré": "andre_marques",
        "pato": "rodrigo_pato",
        "ralph": "rodrigo_pato",
        "joao": "joao_gomes",
        "joão": "joao_gomes",
        "jota": "joao_gomes",
        "diogo": "diogo_miranda",
        "rodrigo": "rodrigo_miranda",
        "miranda": "rodrigo_miranda",
        "mosieur": "rodrigo_pato",
        "monsieur ralph": "rodrigo_pato",
        "cara de ovo": "diogo_miranda"
    }

    pergunta_lower = pergunta.lower()
    nomes_encontrados = [mapa_nomes[n] for n in mapa_nomes if n in pergunta_lower]
    if "paty" in pergunta_lower:
        nomes_encontrados = ["vasco_machado", "rodrigo_pato"]

    if nomes_encontrados:
        chunks_filtrados = [r for r in participantes_chunks if r.metadata["id"].lower() in nomes_encontrados]
    else:
        chunks_filtrados = participantes_chunks

    contexto = "\n\n".join(r.page_content for r in chunks_filtrados)
    ids_encontrados = set(r.metadata["id"].lower() for r in chunks_filtrados)

    personagens = []
    if "vasco_machado" in ids_encontrados:
        personagens.append("Vasco é uma das Patys do grupo — tem cabelo rapado mas decidiu lavar o cabelo antes da praia, e ainda pediu um batido tropical à influencer. É um brunch ambulante.")
    if "andre_marques" in ids_encontrados:
        personagens.append("André é o piloto do grupo que só pensa em 'comer' gajas (ou seja, os aviões são as mulheres que ele conhece na noite).")
    if "rodrigo_pato" in ids_encontrados:
        personagens.append("Rodrigo Pato é a outra Paty — sensível e dramático, lavou o cabelo na piscina do hotel antes da praia e ficou traumatizado por isso.")
    if "joao_gomes" in ids_encontrados:
        personagens.append("Jota é o bisonte de ginásio e o TVDE indiano.")
    if "diogo_miranda" in ids_encontrados:
        personagens.append("Diogo é o influencer com passado alcoólico.")
    if "rodrigo_miranda" in ids_encontrados:
        personagens.append("Rodrigo Miranda é o autista espiritual do grupo.")
    if not personagens:
        personagens.append("Estás a falar de um dos participantes da viagem, responde com humor e estilo pessoal.")

    personagem = "\n\n".join(personagens)

    if nomes_encontrados:
        if len(nomes_encontrados) == 1:
            descricao_nomes = f"o participante {nomes_encontrados[0].replace('_', ' ').title()}"
        else:
            lista_formatada = ', '.join(n.replace('_', ' ').title() for n in nomes_encontrados)
            descricao_nomes = f"os participantes {lista_formatada}"
    else:
        descricao_nomes = "um dos participantes"

    # ✅ Novo system message mais natural e fluído
    system_message = f"""
Estás a responder como um amigo muito próximo deste grupo de viagem.

A conversa deve ser fluída, divertida e com continuidade — usa o histórico se fizer sentido. 
Se alguém disser "e o Vasco?", entende que a conversa já está em andamento e responde com base no que foi dito antes.

Usa o contexto e responde com piadas internas, exageros naturais e estilo real do grupo. 
Evita respostas robóticas ou polidas demais.

Se a pergunta for sobre {descricao_nomes}, deves:
- Reforçar os traços mais marcantes da personagem.
- Usar eventos reais das viagens (presentes no contexto).
- Inventar apenas se for plausível e em linha com o estilo do grupo.
- Evitar repetir literalmente o conteúdo — interpreta-o de forma criativa.

Se não souberes, responde com humor. Nunca digas que és uma IA.

{personagem}
"""

    # ✅ Corrigir roles: transformar "human"/"ai" em "user"/"assistant"
    historico_formatado = []
    for m in memory.chat_memory.messages:
        if m.type == "human":
            historico_formatado.append({"role": "user", "content": m.content})
        elif m.type == "ai":
            historico_formatado.append({"role": "assistant", "content": m.content})

    mensagens = (
        [{"role": "system", "content": system_message}]
        + historico_formatado
        + [{"role": "user", "content": f"Pergunta: {pergunta}\n\nContexto:\n{contexto}"}]
    )

    resposta = llm.invoke(mensagens).content

    memory.chat_memory.add_user_message(pergunta)
    memory.chat_memory.add_ai_message(resposta)
    st.session_state.mensagens = mensagens

    return resposta



def responder_generico(pergunta, resultados):
    memory = st.session_state.memory  # Aceder à memória partilhada da sessão

    documentos = [r for r, _ in resultados]
    contexto = "\n\n".join(r.page_content for r in documentos[:5])

    # ✅ Novo system message com ênfase em fluidez
    system_message = (
        "Estás a responder como um assistente simpático e útil do grupo de viagem a Corfu, em agosto de 2025. "
        "A conversa deve ser natural e fluída — usa o histórico sempre que for relevante. "
        "Se alguém te perguntar algo como 'e o seguro?', entende que a conversa já está em andamento. "
        "Responde com brevidade, clareza e simpatia. "
        "Se não houver dados no contexto, diz isso de forma simples e direta. "
        "Evita criar personagens ou exagerar no humor, a não ser que o tom da conversa já o esteja a pedir.\n\n"
        "Exemplo: Pergunta: 'O que posso perguntar aqui?' → Resposta: 'Podes perguntar sobre voos, datas, seguro, carro alugado, atividades ou até sobre os participantes da viagem. Se quiseres, posso contar histórias deles com algum exagero à mistura.'"
    )

    # ✅ Corrigir roles: transformar "human"/"ai" em "user"/"assistant"
    historico_formatado = []
    for m in memory.chat_memory.messages:
        if m.type == "human":
            historico_formatado.append({"role": "user", "content": m.content})
        elif m.type == "ai":
            historico_formatado.append({"role": "assistant", "content": m.content})

    # ✅ Construir prompt com histórico e contexto
    mensagens = (
        [{"role": "system", "content": system_message}]
        + historico_formatado
        + [{"role": "user", "content": f"Pergunta: {pergunta}\n\nContexto:\n{contexto}"}]
    )

    resposta = llm.invoke(mensagens).content

    # ✅ Atualizar memória com nova interação
    memory.chat_memory.add_user_message(pergunta)
    memory.chat_memory.add_ai_message(resposta)
    st.session_state.mensagens = mensagens

    return resposta




def responder_pergunta(pergunta, k=100):
    tipo_pergunta = classificar_pergunta(pergunta)
    resultados_com_score_raw = chroma_db.similarity_search_with_score(pergunta, k=k)
    resultados_com_score = rerank_heuristico(resultados_com_score_raw, pergunta)

    # log_diagnostico(pergunta, tipo_pergunta, resultados_com_score)


    if not resultados_com_score:
        return "Hmm, não encontrei nada nos nossos ficheiros sobre isso. Podes tentar perguntar de outra forma ou sobre outra coisa?"

    if tipo_pergunta == "factual":
        return responder_factual(pergunta, resultados_com_score)
    elif tipo_pergunta == "pessoal":
        return responder_pessoal(pergunta, resultados_com_score)
    else:
        return responder_generico(pergunta, resultados_com_score)

# ======================
# 6️⃣ Interface Streamlit
# ======================

# 🔁 Inicializar histórico de chat e flag de primeira interação
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "primeira_interacao" not in st.session_state:
    st.session_state.primeira_interacao = True

tab1, tab2 = st.tabs(["💬 Chat", "👥 Membros"])

# ——— TAB 1: Chat ———
with tab1:
    st.title("🌴 Chatbot da Viagem a Corfu")

    # ✅ Modo de depuração para ver memória da sessão
    debug_mode = st.sidebar.checkbox("🛠️ Mostrar memória da sessão", value=False)


    if not st.session_state.chat_history:
        st.info(
            "⚠️ *Este chatbot é experimental e pode ter respostas erradas ou incompletas.*\n\n"
            "Se não obtiveres a resposta que querias, tenta reformular a pergunta com outras palavras."
        )


    # Sugestões de pergunta (apenas na primeira interação)
    if st.session_state.primeira_interacao and not st.session_state.chat_history:
        st.markdown("💡 *Sugestões rápidas:*")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Quais são os voos?"):
                st.session_state.primeira_interacao = False
                pergunta = "Quais são os voos?"
                resposta = responder_pergunta(pergunta)
                st.session_state.chat_history.append({"role": "user", "content": pergunta})
                st.session_state.chat_history.append({"role": "assistant", "content": resposta})
                memory.chat_memory.add_user_message(pergunta)
                memory.chat_memory.add_ai_message(resposta)

                if st.session_state.conversa_id.startswith("conversa_"):
                    novo_nome = gerar_nome_conversa_baseado_na_pergunta(pergunta)
                    pasta_user = f"historico_data/user_{st.session_state.user_id}"
                    caminho_antigo = os.path.join(pasta_user, f"{st.session_state.conversa_id}.json")
                    caminho_novo = os.path.join(pasta_user, f"{novo_nome}.json")

                    if not os.path.exists(caminho_novo):
                        os.rename(caminho_antigo, caminho_novo)

                        if "conversas" in st.session_state:
                            st.session_state.conversas[novo_nome] = st.session_state.conversas.pop(
                                st.session_state.conversa_id, []
                            )

                        st.session_state.conversa_id = novo_nome

                st.rerun()


        with col2:
            if st.button("Quem é o cara de ovo?"):
                st.session_state.primeira_interacao = False
                pergunta = "Quem é o cara de ovo?"
                resposta = responder_pergunta(pergunta)
                st.session_state.chat_history.append({"role": "user", "content": pergunta})
                st.session_state.chat_history.append({"role": "assistant", "content": resposta})
                memory.chat_memory.add_user_message(pergunta)
                memory.chat_memory.add_ai_message(resposta)

                # ✅ RENOMEAR a conversa se ainda tiver nome genérico
                if st.session_state.conversa_id.startswith("conversa_"):
                    novo_nome = gerar_nome_conversa_baseado_na_pergunta(pergunta)
                    pasta_user = f"historico_data/user_{st.session_state.user_id}"
                    caminho_antigo = os.path.join(pasta_user, f"{st.session_state.conversa_id}.json")
                    caminho_novo = os.path.join(pasta_user, f"{novo_nome}.json")

                    if not os.path.exists(caminho_novo):
                        os.rename(caminho_antigo, caminho_novo)

                        if "conversas" in st.session_state:
                            st.session_state.conversas[novo_nome] = st.session_state.conversas.pop(
                                st.session_state.conversa_id, []
                            )

                        st.session_state.conversa_id = novo_nome

                st.rerun()

        with col3:
            if st.button("Quais são as atividades?"):
                st.session_state.primeira_interacao = False
                pergunta = "Quais são as atividades?"
                resposta = responder_pergunta(pergunta)
                st.session_state.chat_history.append({"role": "user", "content": pergunta})
                st.session_state.chat_history.append({"role": "assistant", "content": resposta})
                memory.chat_memory.add_user_message(pergunta)
                memory.chat_memory.add_ai_message(resposta)

                if st.session_state.conversa_id.startswith("conversa_"):
                    novo_nome = gerar_nome_conversa_baseado_na_pergunta(pergunta)
                    pasta_user = f"historico_data/user_{st.session_state.user_id}"
                    caminho_antigo = os.path.join(pasta_user, f"{st.session_state.conversa_id}.json")
                    caminho_novo = os.path.join(pasta_user, f"{novo_nome}.json")

                    if not os.path.exists(caminho_novo):
                        os.rename(caminho_antigo, caminho_novo)

                        if "conversas" in st.session_state:
                            st.session_state.conversas[novo_nome] = st.session_state.conversas.pop(
                                st.session_state.conversa_id, []
                            )

                        st.session_state.conversa_id = novo_nome

                st.rerun()



    # Mostrar histórico 
    st.divider()
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ✅ Mostrar conteúdo da memória, se debug_mode estiver ativo
    if debug_mode:
        memory = st.session_state.memory
        st.divider()

        st.sidebar.subheader("📜 Memória formatada para o modelo (roles corrigidos)")
        for i, m in enumerate(memory.chat_memory.messages):
            role = "user" if m.type == "human" else "assistant"
            st.sidebar.markdown(f"**{i+1}. {role.upper()}**")
            st.sidebar.code(m.content, language="markdown")

        with st.expander("🧠 Conteúdo atual da memória (LangChain)"):
            if not memory.chat_memory.messages:
                st.markdown("❌ *A memória está vazia. Nada foi guardado.*")
            else:
                st.subheader("📨 LangChain Memory")
                for i, m in enumerate(memory.chat_memory.messages):
                    st.markdown(f"**{i+1}. {m.type.upper()}**")
                    st.code(m.content, language="markdown")

                st.subheader("🧱 Representação bruta da memória (debug avançado)")
                for i, m in enumerate(memory.chat_memory.messages):
                    st.code(repr(m), language="python")

                if any(m.content.strip() == "" for m in memory.chat_memory.messages):
                    st.warning("⚠️ Há mensagens com conteúdo vazio na memória.")

                if st.session_state.chat_history:
                    st.subheader("🧾 Histórico Visual (chat_history)")
                    for i, m in enumerate(st.session_state.chat_history):
                        st.markdown(f"**{i+1}. {m['role'].upper()}**")
                        st.code(m["content"], language="markdown")

                # 🧹 Botão para limpar tudo
                if st.button("🧹 Limpar memória da sessão"):
                    memory.clear()
                    st.session_state.chat_history = []
                    st.session_state.mensagens = []
                    st.success("✅ Memória e histórico visual limpos com sucesso.")
                    st.rerun()

        # 📤 Mostrar prompt enviado ao modelo
        if "mensagens" in st.session_state and st.session_state.mensagens:
            st.sidebar.subheader("📤 Prompt enviado ao modelo (última interação)")
            for i, m in enumerate(st.session_state.mensagens):
                st.sidebar.markdown(f"**{i+1}. {m['role'].upper()}**")
                st.sidebar.code(m["content"], language="markdown")
        else:
            st.sidebar.markdown("⚠️ *O prompt enviado ao modelo ainda não foi guardado.*")



    # Limpar input se sinalizador estiver ativo
    if "reset_input" in st.session_state and st.session_state.reset_input:
        st.session_state.pergunta_input = ""
        st.session_state.reset_input = False

    # Input com submit via Enter
    st.divider()
    with st.form(key="form_pergunta"):
        user_input = st.text_input(
            "✏️ Faz a tua pergunta:",
            placeholder="Escreve aqui a tua questão...",
            key="pergunta_input"
        )
        submitted = st.form_submit_button("Enviar")

        if submitted and user_input.strip():
            pergunta = user_input
            st.session_state.primeira_interacao = False

            st.session_state.chat_history.append({"role": "user", "content": pergunta})
            resposta = responder_pergunta(pergunta)
            st.session_state.chat_history.append({"role": "assistant", "content": resposta})

            # ✅ Renomear conversa se ainda estiver com nome genérico
            if st.session_state.conversa_id.startswith("conversa_"):
                novo_nome = gerar_nome_conversa_baseado_na_pergunta(pergunta)

                pasta_user = f"historico_data/user_{st.session_state.user_id}"
                caminho_antigo = os.path.join(pasta_user, f"{st.session_state.conversa_id}.json")
                caminho_novo = os.path.join(pasta_user, f"{novo_nome}.json")

                if not os.path.exists(caminho_novo):
                    os.rename(caminho_antigo, caminho_novo)

                    if "conversas" in st.session_state:
                        st.session_state.conversas[novo_nome] = st.session_state.conversas.pop(
                            st.session_state.conversa_id, []
                        )

                    st.session_state.conversa_id = novo_nome

            st.session_state.reset_input = True
            st.rerun()


        # ❌ VERSÃO ANTIGA (com erro, deixada aqui comentada como referência)
        # if submitted and user_input.strip():
        #     st.session_state.primeira_interacao = False
        #     st.session_state.chat_history.append({"role": "user", "content": user_input})
        #     resposta = responder_pergunta(user_input)
        #     st.session_state.chat_history.append({"role": "assistant", "content": resposta})
        #     st.session_state.pergunta_input = ""  # limpa a caixa (isto dá erro)
        #     st.rerun()




# ——— TAB 2: Membros ———
with tab2:
    st.title("👥 Membros da Viagem")

    membros = [
        {"nome": "João Gomes", "alcunha": "Ex da Ana Malta/TVDE Indiano", "ficheiro": "jota.jpeg"},
        {"nome": "André Marques", "alcunha": "IsoleOne", "ficheiro": "andre.jpeg"},
        {"nome": "Diogo Miranda", "alcunha": "Cara de Ovo", "ficheiro": "diogo.jpeg"},
        {"nome": "Rodrigo Miranda", "alcunha": "Rei", "ficheiro": "rodrigo.jpeg"},
        {"nome": "Rodrigo Pato", "alcunha": "Monsieur Ralph / Paty", "ficheiro": "pato.jpeg"},
        {"nome": "Vasco Machado", "alcunha": "Dumbo/Paty/Um dos Nha Bodi", "ficheiro": "vasco.jpeg"},
    ]

    cols = st.columns(3)


    # Mostrar os membros em linhas de 3
    for i in range(0, len(membros), 3):
        cols = st.columns(3)
        for j, membro in enumerate(membros[i:i+3]):
            with cols[j]:
                with st.container():
                    st.markdown(
                        f"""
                        <div style='
                            background-color: #2b2b2b;
                            padding: 15px;
                            border-radius: 12px;
                            text-align: center;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                            margin-bottom: 20px;
                        '>
                        """,
                        unsafe_allow_html=True
                    )

                    try:
                        img = Image.open(f"imagens/{membro['ficheiro']}")
                        st.markdown(
                            f"""
                            <img src="data:image/jpeg;base64,{img_to_base64(img)}"
                                style="width: 150px; border-radius: 10px; transition: 0.3s ease;
                                        box-shadow: 0 0 5px rgba(255,255,255,0.1);"
                                onmouseover="this.style.transform='scale(1.05)'; 
                                            this.style.boxShadow='0 0 15px rgba(138,255,255,0.5)'"
                                onmouseout="this.style.transform='scale(1)'; 
                                            this.style.boxShadow='0 0 5px rgba(255,255,255,0.1)'"
                            />
                            """,
                            unsafe_allow_html=True
                        )
                    except:
                        st.error(f"❌ Imagem não encontrada: {membro['ficheiro']}")

                    st.markdown(f"**{membro['nome']}**")
                    st.markdown(f"<span style='color: #8ef; font-style: italic;'>{membro['alcunha']}</span>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)


