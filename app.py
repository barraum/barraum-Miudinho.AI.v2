import streamlit as st
import google.generativeai as genai
import faiss
import pickle
import numpy as np
import toml
import os

# --- CONFIGURAÇÕES E CONSTANTES ---
NOME_DO_PROJETO = "Miudinho.AI v2.0 (Rerank + Expansion)"
MODELO_EMBEDDING = 'models/text-embedding-004'
MODELO_GERACAO = 'gemini-2.5-flash'  # Modelo rápido e barato para Rerank e Resposta

# Caminhos (ajustados conforme sua estrutura de pastas na imagem)
CAMINHO_INDEX = 'banco_vetorial_gemini_srt_900.index'
CAMINHO_PKL = 'chunks_mapeamento_gemini_srt_900.pkl'

# Configuração da Página
st.set_page_config(page_title=NOME_DO_PROJETO, layout="wide")

# --- 1. CONFIGURAÇÃO DE SEGURANÇA (API KEY) ---
GEMINI_API_KEY = None

# Tenta ler do Streamlit Cloud (Secrets)
try:
    if "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    pass

# Se não achou, tenta ler localmente (.streamlit/secrets.toml) ou .env
if not GEMINI_API_KEY:
    # Tenta caminho relativo padrão do streamlit
    secrets_path = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(secrets_path):
        try:
            with open(secrets_path, "r") as f:
                config = toml.load(f)
                GEMINI_API_KEY = config.get("GEMINI_API_KEY")
        except Exception:
            pass

if not GEMINI_API_KEY:
    st.error("❌ ERRO: Chave de API não encontrada. Configure o arquivo .streamlit/secrets.toml")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# --- 2. FUNÇÕES DE CARREGAMENTO (CACHED) ---
@st.cache_resource
def carregar_dados():
    """Carrega o índice FAISS e os metadados PKL apenas uma vez."""
    if not os.path.exists(CAMINHO_INDEX) or not os.path.exists(CAMINHO_PKL):
        return None, None
    
    index = faiss.read_index(CAMINHO_INDEX)
    with open(CAMINHO_PKL, 'rb') as f:
        chunks = pickle.load(f)
    return index, chunks

# --- 3. FUNÇÕES DE INTELIGÊNCIA (EXPANSION & RERANK) ---

def expandir_consulta(pergunta_usuario):
    """Gera variações da pergunta para aumentar a chance de encontrar trechos."""
    prompt = f"""
    Você é um especialista em buscas semânticas para conteúdo bíblico.
    O usuário fez a pergunta: "{pergunta_usuario}"
    
    Gere 3 variações curtas dessa pergunta para buscar em um banco de dados vetorial.
    Pense em sinônimos teológicos ou formas diferentes de frasear.
    
    Retorne APENAS as perguntas separadas por quebra de linha. Nada mais.
    """
    try:
        model = genai.GenerativeModel(MODELO_GERACAO)
        response = model.generate_content(prompt)
        # Limpa e separa as linhas
        variacoes = [line.strip() for line in response.text.split('\n') if line.strip()]
        # Adiciona a pergunta original na lista
        return [pergunta_usuario] + variacoes[:3] 
    except Exception as e:
        print(f"Erro na expansão: {e}")
        return [pergunta_usuario]

def rerank_chunks(pergunta, candidatos, top_n=5):
    """
    Recebe uma lista grande de candidatos (chunks) e usa o Gemini para
    escolher os melhores e ordená-los por relevância real.
    """
    # Monta um texto numerado com os candidatos
    textos_candidatos = ""
    for i, chunk in enumerate(candidatos):
        # Limitamos o tamanho do texto para não estourar tokens desnecessariamente
        trecho_curto = chunk['text'][:600].replace('\n', ' ')
        textos_candidatos += f"ID_{i}: {trecho_curto}\n\n"

    prompt_rerank = f"""
    Analise a relevância dos trechos abaixo para responder à pergunta: "{pergunta}"
    
    TRECHOS CANDIDATOS:
    {textos_candidatos}
    
    TAREFA:
    Identifique quais desses trechos são REALMENTE úteis para responder à pergunta.
    Classifique os melhores (no máximo {top_n}).
    
    Retorne APENAS os IDs dos trechos escolhidos, em ordem de relevância (do melhor para o pior), separados por vírgula.
    Exemplo de saída: ID_3, ID_0, ID_12
    """
    
    try:
        model = genai.GenerativeModel(MODELO_GERACAO)
        response = model.generate_content(prompt_rerank)
        resposta_texto = response.text.strip()
        
        # Processa a resposta para pegar os índices
        indices_escolhidos = []
        partes = resposta_texto.replace("ID_", "").split(",")
        
        for p in partes:
            try:
                idx = int(p.strip())
                if 0 <= idx < len(candidatos):
                    indices_escolhidos.append(idx)
            except ValueError:
                continue
        
        # Reconstrói a lista de objetos baseada nos índices escolhidos
        chunks_reranked = [candidatos[i] for i in indices_escolhidos]
        
        # Se o modelo falhar em retornar IDs válidos, devolve os originais cortados
        if not chunks_reranked:
            return candidatos[:top_n]
            
        return chunks_reranked
        
    except Exception as e:
        print(f"Erro no Reranking: {e}")
        return candidatos[:top_n] # Fallback: retorna os primeiros do FAISS

# --- 4. FUNÇÃO DE BUSCA PRINCIPAL ---
def buscar_resposta(query_usuario, index, chunks_data):
    
    # 1. Expansão da Consulta
    with st.status("🧠 Analisando sua pergunta...", expanded=False) as status:
        status.write("Gerando variações para busca ampla...")
        queries_expandidas = expandir_consulta(query_usuario)
        status.write(f"Variações geradas: {queries_expandidas}")
        
        # 2. Busca Vetorial (FAISS) para cada variação
        status.write("Consultando banco de dados...")
        todos_indices = []
        
        model_emb = 'models/text-embedding-004'
        for q in queries_expandidas:
            vetor_pergunta = genai.embed_content(
                model=model_emb,
                content=q,
                task_type="RETRIEVAL_QUERY"
            )["embedding"]
            
            # Busca Top 15 para cada variação (queremos volume para filtrar depois)
            D, I = index.search(np.array([vetor_pergunta]), k=15)
            todos_indices.extend(I[0])
        
        # 3. Deduplicação (remove repetidos)
        indices_unicos = list(set(todos_indices))
        candidatos_iniciais = [chunks_data[i] for i in indices_unicos if i < len(chunks_data)]
        
        status.write(f"Encontrados {len(candidatos_iniciais)} trechos potenciais. Refinando...")

        # 4. Reranking (O Pulo do Gato 🐈)
        # Seleciona apenas os Top 5 melhores dentre os candidatos
        chunks_finais = rerank_chunks(query_usuario, candidatos_iniciais, top_n=5)
        
        status.update(label="✅ Busca e Análise concluídas!", state="complete", expanded=False)
    
    return chunks_finais

# --- 5. INTERFACE DO USUÁRIO ---
def main():
    st.title("🎥 " + NOME_DO_PROJETO)
    st.markdown("Busque por assuntos na base de vídeos e vá direto ao momento da fala.")
    
    index, chunks_data = carregar_dados()
    
    if index is None:
        st.error("⚠️ Banco de dados não encontrado! Rode o script 'criar_banco_vetores_srt.py' primeiro.")
        return

    query = st.text_input("O que você procura?", placeholder="Ex: O que foi falado sobre o filho pródigo?")
    
    if st.button("Pesquisar", type="primary"):
        if not query:
            st.warning("Digite algo para pesquisar.")
            return

        # Executa a busca inteligente
        chunks_relevantes = buscar_resposta(query, index, chunks_data)
        
        if not chunks_relevantes:
            st.warning("Nada encontrado. Tente reformular a pergunta.")
            return

        # Geração da Resposta Final com Gemini
        with st.spinner("Gerando resposta explicativa..."):
            contexto = ""
            for doc in chunks_relevantes:
                # Inclui metadados no contexto para o LLM saber a fonte
                info_fonte = f"[Vídeo: {doc.get('source_file', 'desc')}]"
                contexto += f"{info_fonte}\nConteúdo: {doc['text']}\n\n"
            
            prompt_final = f"""
            Você é um assistente útil e preciso. Use os trechos abaixo transcritos de vídeos para responder à pergunta do usuário.
            
            Pergunta: {query}
            
            Trechos encontrados:
            {contexto}
            
            Instruções:
            1. Responda de forma direta e explicativa.
            2. Se os trechos não responderem à pergunta, diga que não encontrou a informação.
            3. Cite o contexto se necessário.
            """
            
            model = genai.GenerativeModel(MODELO_GERACAO)
            resposta = model.generate_content(prompt_final)
            
        st.markdown("### Resposta:")
        st.write(resposta.text)
        
        st.divider()
        st.subheader("📺 Fontes e Vídeos")
        st.caption("Clique nos vídeos abaixo para assistir exatamente onde o assunto é abordado.")

        # Exibição dos Vídeos (Lógica de Timestamp mantida!)
        for i, chunk in enumerate(chunks_relevantes):
            with st.expander(f"Fonte {i+1}: ...{chunk['text'][:60]}...", expanded=True):
                col1, col2 = st.columns([1, 1.5])
                
                url_video = chunk.get('url')
                start_time = int(chunk.get('start_time', 0))
                
                with col1:
                    if url_video:
                        st.video(url_video, start_time=start_time)
                        st.caption(f"Começa em: {start_time}s")
                    else:
                        st.image("https://via.placeholder.com/300x169?text=Sem+Video", caption="URL não mapeada")
                
                with col2:
                    st.markdown(f"**Transcrição:**")
                    st.info(chunk['text'])
                    st.markdown(f"*Arquivo original: {chunk.get('source_file')}*")

if __name__ == "__main__":
    main()