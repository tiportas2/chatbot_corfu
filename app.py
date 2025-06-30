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
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path
from PIL import Image

import base64
from io import BytesIO

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
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
def classificar_pergunta(pergunta):
    nomes = ["vasco", "andré", "pato", "joão", "jota", "diogo", "rodrigo", "miranda", "ralph"]
    palavras_factuais = [
        "voo", "voos", "data", "datas", "partida", "chegada", "hora", "preço", "custos",
        "aluguer", "seguro", "atividade", "atividades", "cruzeiro", "excursão", "passeio", "programa"
    ]
    if any(nome in pergunta.lower() for nome in nomes):
        return "personalizada"
    if any(palavra in pergunta.lower() for palavra in palavras_factuais):
        return "factual"
    return "personalizada"

def responder_pergunta(pergunta, k=100):
    tipo_pergunta = classificar_pergunta(pergunta)
    resultados = chroma_db.similarity_search(pergunta, k=k)

    if not resultados:
        return "Hmm, não encontrei nada nos nossos ficheiros sobre isso. Podes tentar perguntar de outra forma ou sobre outra coisa?"

    participantes_chunks = [r for r in resultados if r.metadata["tipo"] == "participantes"]
    logistica_chunks = [r for r in resultados if r.metadata["tipo"] == "logistica"]

    if tipo_pergunta == "factual" or not participantes_chunks:
        termos_relevantes = ["voo", "seguro", "carro", "chegada", "partida", "corfu", "milão", "paris"]
        prioridade_chunks = [
            r for r in logistica_chunks
            if any(t in r.page_content.lower() for t in termos_relevantes)
        ]
        contexto = "\n\n".join(r.page_content for r in prioridade_chunks) if prioridade_chunks else "\n\n".join(r.page_content for r in logistica_chunks)
        system_message = (
            "Estás a responder como um assistente de viagem. Responde apenas com base nas informações fornecidas no contexto abaixo. "
            "Não faças suposições, não inventes companhias aéreas, não recomendes websites genéricos. "
            "Foca-te nos detalhes concretos da viagem (voos, datas, horas, companhias, preços, etc). "
            "Se o contexto não tiver a resposta, diz que não há informação suficiente."
        )
    else:
        nome_mencionado = None
        nomes_possiveis = ["vasco", "andré", "pato", "joão", "jota", "diogo", "rodrigo", "miranda", "ralph"]
        for nome in nomes_possiveis:
            if nome in pergunta.lower():
                nome_mencionado = nome
                break
        if nome_mencionado:
            chunks_filtrados = [r for r in participantes_chunks if nome_mencionado in r.metadata["id"].lower()]
        else:
            chunks_filtrados = participantes_chunks

        contexto = "\n\n".join(r.page_content for r in chunks_filtrados)
        nomes_encontrados = set(r.metadata["id"].lower() for r in chunks_filtrados)
        personagens = []
        if "vasco" in nomes_encontrados:
            personagens.append("Vasco é uma das patys do grupo...")
        if "andre" in nomes_encontrados:
            personagens.append("André é o piloto emocional do grupo...")
        if "pato" in nomes_encontrados:
            personagens.append("Rodrigo Pato é sensível...")
        if "joao" in nomes_encontrados or "jota" in nomes_encontrados:
            personagens.append("Jota é o bisonte de ginásio...")
        if "diogo" in nomes_encontrados:
            personagens.append("Diogo é o influencer com passado alcoólico...")
        if "rodrigo_miranda" in nomes_encontrados or "rodrigo" in nomes_encontrados:
            personagens.append("Rodrigo Miranda é o autista espiritual do grupo...")
        if not personagens:
            personagens.append("Estás a falar de um dos participantes da viagem, responde com humor e estilo pessoal.")

        personagem = "\n\n".join(personagens)
        system_message = (
            f"Estás a responder como um amigo próximo do grupo que vai a Corfu. "
            f"Usa piadas internas, linguagem do grupo e um tom leve, divertido e exagerado. "
            f"O teu estilo deve imitar o dos textos dos participantes: frases curtas, gíria, humor, absurdos e inside jokes. "
            f"Nunca respondas de forma neutra ou genérica. "
            f"Se mencionares alguém, usa expressões e acontecimentos reais descritos nos textos.\n\n{personagem}"
        )

    mensagens = st.session_state.chat_history.copy()
    mensagens.append({"role": "system", "content": system_message})
    mensagens.append({"role": "user", "content": f"A pergunta do utilizador é:\n{pergunta}\n\nAqui está o contexto útil:\n{contexto}"})
    resposta = llm.invoke(mensagens)
    # st.session_state.chat_history.append({"role": "assistant", "content": resposta.content})
    return resposta.content

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

    if not st.session_state.chat_history:
        st.info(
            "⚠️ *Este chatbot é experimental e pode ter respostas erradas ou incompletas.*\n"
            "Se não obtiveres a resposta que querias, tenta reformular a pergunta com outras palavras"
        )


    # Sugestões de pergunta (apenas na primeira interação)
    if st.session_state.primeira_interacao and not st.session_state.chat_history:
        st.markdown("💡 *Sugestões rápidas:*")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Quais são os voos?"):
                st.session_state.primeira_interacao = False
                st.session_state.chat_history.append({"role": "user", "content": "Quais são os voos?"})
                resposta = responder_pergunta("Quais são os voos?")
                st.session_state.chat_history.append({"role": "assistant", "content": resposta})
                st.rerun()

        with col2:
            if st.button("Quem é o cara de ovo?"):
                st.session_state.primeira_interacao = False
                st.session_state.chat_history.append({"role": "user", "content": "Quem é o cara de ovo?"})
                resposta = responder_pergunta("Quem é o cara de ovo?")
                st.session_state.chat_history.append({"role": "assistant", "content": resposta})
                st.rerun()

        with col3:
            if st.button("Quais são as atividades?"):
                st.session_state.primeira_interacao = False
                st.session_state.chat_history.append({"role": "user", "content": "Quais são as atividades?"})
                resposta = responder_pergunta("Quais são as atividades?")
                st.session_state.chat_history.append({"role": "assistant", "content": resposta})
                st.rerun()

    # Mostrar histórico 
    st.divider()
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    

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
            st.session_state.primeira_interacao = False
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            resposta = responder_pergunta(user_input)
            st.session_state.chat_history.append({"role": "assistant", "content": resposta})
            # st.session_state.pergunta_input = ""  # limpa a caixa
            st.experimental_rerun()





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


