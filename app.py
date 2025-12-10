import streamlit as st
import google.generativeai as genai
import faiss
import pickle
import numpy as np
import json
import toml
import os
from pytubefix import YouTube
import xml.etree.ElementTree as ET

# --- CONFIGURAÇÃO INICIAL DA PÁGINA ---
st.set_page_config(
    page_title="MiudinhoAI v2.0",
    page_icon="🤖",
    layout="wide"
)

# --- 1. CONFIGURAÇÃO DE SEGURANÇA (API KEY) ---
GEMINI_API_KEY = None

# Tenta ler do Streamlit Cloud (Secrets)
try:
    if "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

# Se não achou, tenta ler localmente (ajuste o caminho se necessário)
if not GEMINI_API_KEY:
    # Caminho local de backup (apenas para seu uso no VSCode)
    CAMINHO_SECRETS_LOCAL = r"C:\Users\bruno\OneDrive\Projetos Python\14) MIUDINHO.AI\.streamlit\secrets.toml"
    try:
        if os.path.exists(CAMINHO_SECRETS_LOCAL):
            with open(CAMINHO_SECRETS_LOCAL, "r") as f:
                config = toml.load(f)
                GEMINI_API_KEY = config.get("GEMINI_API_KEY")
    except Exception:
        pass

if not GEMINI_API_KEY:
    st.error("❌ ERRO: Chave de API não encontrada.")
    st.info("Configure o arquivo .streamlit/secrets.toml para execução local ou adicione aos Secrets do Streamlit Cloud.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- MODELOS E CONSTANTES ---
# Usando o Flash para ser rápido na busca e rerank
MODELO_RERANK = 'gemini-2.5-flash' 
# Usando o Pro ou Flash para a resposta final (Flash é mais rápido, Pro é mais detalhado)
MODELO_RESPOSTA = 'gemini-2.5-flash' 
MODELO_EMBEDDING = 'models/text-embedding-004'

# Caminhos (Usando caminhos relativos para funcionar no Github/Cloud)
# Certifique-se que os arquivos estão na raiz ou na pasta correta no Git
# Se estiver rodando local e der erro, volte para o caminho absoluto.
FAISS_INDEX_FILE = 'banco_vetorial_gemini_srt_900.index'
CHUNKS_MAPPING_FILE = 'chunks_mapeamento_gemini_srt_900.pkl'
VIDEO_JSON_FILE = 'videos_miudinho_uberaba.json'

# --- FUNÇÕES DE CARREGAMENTO (CACHE) ---

@st.cache_resource
def load_faiss_index():
    """Carrega o índice FAISS e os metadados."""
    try:
        # Tenta carregar. Se não achar, tenta caminho absoluto (fallback local)
        if not os.path.exists(FAISS_INDEX_FILE):
             # Fallback para seu caminho local absoluto se o relativo falhar
             caminho_abs_index = r'C:\Users\bruno\OneDrive\Projetos Python\14) MIUDINHO.AI\banco_vetorial_gemini_srt_900.index'
             caminho_abs_pkl = r'C:\Users\bruno\OneDrive\Projetos Python\14) MIUDINHO.AI\chunks_mapeamento_gemini_srt_900.pkl'
             if os.path.exists(caminho_abs_index):
                 index = faiss.read_index(caminho_abs_index)
                 with open(caminho_abs_pkl, 'rb') as f:
                     metadata = pickle.load(f)
                 return index, metadata
        
        # Carregamento padrão (Cloud/Git)
        index = faiss.read_index(FAISS_INDEX_FILE)
        with open(CHUNKS_MAPPING_FILE, 'rb') as f:
            metadata = pickle.load(f)
        return index, metadata
        
    except Exception as e:
        st.error(f"Erro ao carregar banco de dados: {e}")
        return None, None

@st.cache_data
def load_video_data():
    """Carrega o JSON dos vídeos."""
    try:
        if os.path.exists(VIDEO_JSON_FILE):
            with open(VIDEO_JSON_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
             # Fallback local
             caminho_abs_json = r'C:\Users\bruno\OneDrive\Projetos Python\14) MIUDINHO.AI\videos_miudinho_uberaba.json'
             if os.path.exists(caminho_abs_json):
                with open(caminho_abs_json, 'r', encoding='utf-8') as f:
                    return json.load(f)
        return []
    except Exception:
        return []

# --- LÓGICA DE INTELIGÊNCIA (RAG + VIDEO) ---

def expand_query(user_query):
    """Gera variações da pergunta."""
    try:
        model = genai.GenerativeModel(MODELO_RERANK)
        prompt = f"""
        Gere 3 formas diferentes de perguntar: "{user_query}"
        Foque em sinônimos teológicos e palavras-chave relacionadas a estudos bíblicos.
        Retorne apenas as perguntas, uma por linha.
        """
        response = model.generate_content(prompt)
        variations = [line.strip() for line in response.text.split('\n') if line.strip()]
        return [user_query] + variations
    except:
        return [user_query]

def rerank_chunks(query, chunks, top_n=10):
    """
    Reordena os chunks para garantir relevância, mas mantém um número saudável (top_n=10)
    para não encurtar demais a resposta final.
    """
    if not chunks:
        return []
        
    # Monta texto para o Gemini avaliar
    candidatos_txt = ""
    for i, c in enumerate(chunks):
        candidatos_txt += f"ID_{i}: {c['text'][:400]}...\n\n" # Envia só o começo para economizar

    prompt = f"""
    Analise a pergunta: "{query}"
    Classifique os trechos abaixo por relevância para responder essa pergunta.
    
    TRECHOS:
    {candidatos_txt}
    
    Retorne APENAS os IDs dos {top_n} melhores, ordenados do mais relevante para o menos, separados por vírgula.
    Exemplo: ID_2, ID_0, ID_5
    """
    
    try:
        model = genai.GenerativeModel(MODELO_RERANK)
        response = model.generate_content(prompt)
        ids_str = response.text.replace("ID_", "").split(",")
        indices = []
        for x in ids_str:
            try:
                idx = int(x.strip())
                if 0 <= idx < len(chunks):
                    indices.append(idx)
            except:
                continue
        
        results = [chunks[i] for i in indices]
        # Se o rerank falhar ou retornar vazio, devolve os originais (fallback)
        return results if results else chunks[:top_n]
    except:
        return chunks[:top_n]

def get_video_transcript(url):
    """Pega a legenda do YouTube via Pytubefix."""
    try:
        yt = YouTube(url)
        # Tenta várias tags de idioma pt
        caption = None
        for lang in ['pt', 'pt-BR', 'a.pt']:
            if lang in yt.captions:
                caption = yt.captions[lang]
                break
        
        if not caption:
            return None
            
        xml_captions = caption.xml_captions
        root = ET.fromstring(xml_captions)
        lines = [elem.text for elem in root.iter('text') if elem.text]
        return " ".join(lines)
    except Exception as e:
        st.error(f"Erro ao obter legenda: {e}")
        return None

# --- INTERFACE PRINCIPAL ---

def main():
    st.title("🤖 MiudinhoAI - Central de Conhecimento")
    
    tab1, tab2 = st.tabs(["🔍 Busca Global (Acervo)", "🎬 Análise de Vídeo Individual"])
    
    # --- ABA 1: BUSCA GLOBAL ---
    with tab1:
        st.header("Pesquise em todo o canal")
        st.caption("O sistema busca o momento exato da fala nos vídeos.")
        
        index, metadata = load_faiss_index()
        query = st.text_input("Qual sua dúvida teológica ou curiosidade?", key="search_box")
        
        if st.button("Pesquisar no Acervo", type="primary"):
            if not index or not query:
                st.warning("Banco de dados não carregado ou busca vazia.")
            else:
                status = st.status("🕵️ Processando sua busca...", expanded=True)
                
                # 1. Expansão
                status.write("Expandindo termos da pesquisa...")
                queries = expand_query(query)
                
                # 2. Busca Vetorial (Pega bastante coisa para filtrar depois)
                status.write("Varrendo banco de dados...")
                chunk_results = []
                
                result = genai.embed_content(
                    model=MODELO_EMBEDDING,
                    content=queries,
                    task_type="retrieval_query"
                )

                embeddings = result['embedding']
                
                # Busca no FAISS
                D, I = index.search(np.array(embeddings), k=10) # 10 por variação
                
                # Deduplicação
                seen_indices = set()
                candidates = []
                for row in I:
                    for idx in row:
                        if idx != -1 and idx not in seen_indices:
                            seen_indices.add(idx)
                            if idx < len(metadata):
                                candidates.append(metadata[idx])
                
                # 3. Rerank (O Refinamento)
                status.write(f"Analisando {len(candidates)} trechos encontrados...")
                # Aumentei o top_n para 7 para garantir resposta longa
                final_chunks = rerank_chunks(query, candidates, top_n=7) 
                
                status.update(label="✅ Busca concluída!", state="complete", expanded=False)
                
                # 4. Geração da Resposta
                if final_chunks:
                    st.subheader("📝 Resposta Sintetizada")
                    
                    contexto = ""
                    for c in final_chunks:
                        contexto += f"Fonte: {c['source_file']}\nTexto: {c['text']}\n\n"
                    
                    prompt_resposta = f"""
                    Use os trechos abaixo para responder a pergunta: "{query}".
                    
                    TRECHOS:
                    {contexto}
                    
                    Instruções:
                    1. Seja DETALHADO e didático. Explique bem o conceito.
                    2. Se houver divergência nos trechos, mencione.
                    3. Cite o nome do arquivo fonte entre parênteses quando usar uma informação.
                    """
                    
                    with st.spinner("Escrevendo resposta..."):
                        model_resp = genai.GenerativeModel(MODELO_RESPOSTA)
                        res = model_resp.generate_content(prompt_resposta)
                        st.markdown(res.text)
                    
                    st.divider()
                    st.subheader("📺 Fontes Encontradas (Clique para assistir)")
                    
                    # Layout: Vídeo na Esquerda, Texto na Direita
                    for i, chunk in enumerate(final_chunks):
                        with st.expander(f"Fonte {i+1}: {chunk['source_file']} (Ver trecho)", expanded=True):
                            col_video, col_text = st.columns([1, 1.2]) # Ajuste de proporção
                            
                            with col_video:
                                url = chunk.get('url')
                                time = int(chunk.get('start_time', 0))
                                if url:
                                    st.video(url, start_time=time)
                                    st.caption(f"Inicia em: {time}s")
                                else:
                                    st.image("https://via.placeholder.com/300x169?text=Sem+URL")
                            
                            with col_text:
                                st.markdown("**Transcrição:**")
                                st.info(chunk['text'])
                else:
                    st.warning("Nenhum conteúdo relevante encontrado.")

    # --- ABA 2: ANÁLISE INDIVIDUAL (RESTAURADA ORIGINAL) ---
    with tab2:
        st.header("Analise um vídeo específico")
        st.info("Escolha um vídeo para resumo ou análise de expressões.")

        video_data = load_video_data()

        if video_data:
            video_titles = [video['titulo'] for video in video_data]
            selected_title = st.selectbox("Escolha o vídeo:", options=video_titles, key="video_selector")
            
            selected_video = next((v for v in video_data if v['titulo'] == selected_title), None)

            if selected_video:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.video(selected_video['url'])
                with col2:
                    st.subheader(selected_video['titulo'])
                    st.write(f"**Versículo:** *{selected_video.get('descricao', '')}*")
                
                st.divider()

                # --- AQUI ESTÁ A LÓGICA RESTAURADA ---
                action = st.radio(
                    "O que você gostaria de fazer com este vídeo?",
                    ("Análise de Expressões e Referências", "Resumo Inteligente do Vídeo"),
                    key="action_choice",
                    horizontal=True
                )

                if st.button("Analisar com Gemini", key="analyze_button", type="primary"):
                    with st.spinner("Baixando legendas e processando... 📜"):
                        transcript = get_video_transcript(selected_video['url'])

                    if transcript:
                        prompt_base = ""
                        
                        # Lógica condicional de prompts restaurada
                        if action == "Análise de Expressões e Referências":
                            prompt_base = f"""
                            Você é um assistente acadêmico de estudos bíblicos.
                            Analise a transcrição e o versículo.
                            Extraia e liste APENAS:
                            ### Palavras e Expressões em Análise
                            (Foco do "estudo miudinho").
                            ### Referências Bibliográficas
                            (Livros e autores citados). Se não houver, diga "Nenhuma referência explícita".
                            """
                        
                        elif action == "Resumo Inteligente do Vídeo":
                            prompt_base = f"""
                            Crie um resumo claro conectando a transcrição ao versículo.
                            ### Resumo da Análise
                            (2-3 parágrafos explicando o tema).
                            ### Tópicos Principais
                            (3-5 pontos centrais).
                            """

                        prompt_final = f"""
                        {prompt_base}
                        --- DADOS ---
                        Versículo: {selected_video.get('descricao', '')}
                        Transcrição: {transcript[:25000]}
                        """
                        
                        try:
                            model = genai.GenerativeModel(MODELO_RESPOSTA)
                            response = model.generate_content(prompt_final)
                            st.header("Resultado da Análise")
                            st.markdown(response.text)
                        except Exception as e:
                            st.error(f"Erro na API Gemini: {e}")
                    else:
                        st.error("Não foi possível obter legendas em Português para este vídeo.")

if __name__ == "__main__":
    main()