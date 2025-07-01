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

import unicodedata
import re

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
    # Extrair só os documentos (ignorando os scores por agora)
    documentos = [r for r, _ in resultados]

    # Filtrar chunks do tipo 'logistica'
    logistica_chunks = [r for r in documentos if r.metadata["tipo"] == "logistica"]
    termos_relevantes = ["voo", "seguro", "carro", "chegada", "partida", "corfu"]

    # Dar prioridade aos chunks mais relevantes semanticamente
    prioridade_chunks = [
        r for r in logistica_chunks
        if any(t in r.page_content.lower() for t in termos_relevantes)
    ]

    # Se houver chunks com termos relevantes, usa esses; senão, usa todos os de logística
    contexto = "\n\n".join(r.page_content for r in prioridade_chunks) if prioridade_chunks else "\n\n".join(r.page_content for r in logistica_chunks)

    system_message = (
    "Estás a responder como um assistente de viagem. Responde apenas com base no contexto abaixo, que diz respeito à viagem a Corfu em agosto de 2025. "
    "Sê direto, factual e não inventes informação. Se não souberes, diz que não há dados suficientes.\n\n"
    "Exemplos:\n"
    "- Pergunta: 'Qual é o seguro do carro?' → Resposta: 'O seguro é da Collinson, custa 55,80€ e cobre danos e roubo de 8 a 14 de agosto em Corfu.'\n"
    "- Pergunta: 'Quanto custaram os carros?' → Resposta: 'O aluguer dos carros foi de 117,40€ por pessoa.'\n"
    "- Pergunta: 'Quem são os condutores do carro?' → Resposta: 'Os condutores são Rodrigo Miranda e Rodrigo Pato, conforme indicado na apólice do seguro.'"
    )

    mensagens = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Pergunta: {pergunta}\n\nContexto:\n{contexto}"}
    ]
    return llm.invoke(mensagens).content

def responder_pessoal(pergunta, resultados):
    # Extrair apenas os documentos
    documentos = [r for r, _ in resultados]

    participantes_chunks = [r for r in documentos if r.metadata["tipo"] == "participantes"]

    # Mapeamento robusto de nomes/alcunhas → id dos participantes
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

    # Verifica se há match com um ou mais nomes no mapa
    nomes_encontrados = [mapa_nomes[n] for n in mapa_nomes if n in pergunta_lower]

    # Tratamento especial para a alcunha ambígua "paty"
    if "paty" in pergunta_lower:
        nomes_encontrados = ["vasco_machado", "rodrigo_pato"]

    # Filtrar os chunks com base nos IDs encontrados
    if nomes_encontrados:
        chunks_filtrados = [r for r in participantes_chunks if r.metadata["id"].lower() in nomes_encontrados]
    else:
        chunks_filtrados = participantes_chunks

    contexto = "\n\n".join(r.page_content for r in chunks_filtrados)

    personagens = []
    ids_encontrados = set(r.metadata["id"].lower() for r in chunks_filtrados)

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

    # Gera a descrição de nomes mencionados para o prompt
    if nomes_encontrados:
        if len(nomes_encontrados) == 1:
            descricao_nomes = f"o participante {nomes_encontrados[0].replace('_', ' ').title()}"
        else:
            lista_formatada = ', '.join(n.replace('_', ' ').title() for n in nomes_encontrados)
            descricao_nomes = f"os participantes {lista_formatada}"
    else:
        descricao_nomes = "um dos participantes"

    system_message = f"""
Estás a responder como um amigo muito próximo deste grupo de viagem.

Responde com base no contexto abaixo, mantendo o tom divertido, exagerado e cheio de piadas internas. Usa o estilo real do grupo e evita parecer artificial ou polido demais.

Se a pergunta for sobre {descricao_nomes}, deves:
- Reforçar os traços mais marcantes da personagem.
- Usar eventos reais das viagens (presentes no contexto).
- Inventar apenas se for plausível e em linha com o estilo do grupo.
- Evitar repetir literalmente o conteúdo — interpreta-o de forma criativa.

Se não souberes, responde com humor e estilo. Nunca respondas como uma IA.

{personagem}
"""

    mensagens = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Pergunta: {pergunta}\n\nContexto:\n{contexto}"}
    ]
    return llm.invoke(mensagens).content


def responder_generico(pergunta, resultados):
    # Extrair os documentos (ignorando os scores)
    documentos = [r for r, _ in resultados]

    # Pega nos primeiros 5 documentos apenas
    contexto = "\n\n".join(r.page_content for r in documentos[:5])

    system_message = (
        "Estás a responder como um assistente do grupo de viagem a Corfu, em agosto de 2025. "
        "Dá respostas breves, úteis e simpáticas. "
        "Se não houver dados no contexto, diz isso de forma simples e direta. "
        "Não cries personagens nem faças humor desnecessário.\n\n"
        "Exemplo: Pergunta: 'O que posso perguntar aqui?' → Resposta: 'Podes perguntar sobre voos, datas, seguro, carro alugado, atividades ou até sobre os participantes da viagem. Se quiseres, posso contar histórias deles com algum exagero à mistura.'"
    )

    mensagens = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Pergunta: {pergunta}\n\nContexto:\n{contexto}"}
    ]
    return llm.invoke(mensagens).content

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

    if not st.session_state.chat_history:
        st.info(
            "⚠️ *Este chatbot é experimental e pode ter respostas erradas ou incompletas.*\n\n"
            "Se não obtiveres a resposta que querias, tenta reformular a pergunta com outras palavras.\n\n"
            "Este chatbot não tem memória, cada mensagem deve ser feita de forma independente tendo em conta que o contexto não será considerado."
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

        # ✅ NOVA VERSÃO (mantém o input limpo sem erro)
        if submitted and user_input.strip():
            pergunta = user_input  # guarda o valor antes de limpar
            st.session_state.primeira_interacao = False
            st.session_state.chat_history.append({"role": "user", "content": pergunta})
            resposta = responder_pergunta(pergunta)
            st.session_state.chat_history.append({"role": "assistant", "content": resposta})
            st.session_state.reset_input = True  # ativa sinalizador para limpar
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


