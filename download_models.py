import os
import json
import logging
from huggingface_hub import snapshot_download, hf_hub_download, HfApi

# --- Configuração do Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)
logger = logging.getLogger("model_downloader")

# --- 1. Configuração dos Diretórios (do docker-compose.yml) ---

# Para onde o modelo de embedding (SentenceTransformer) deve ir
EMBED_CACHE_DIR = "./vllm_cache" #

# Para onde os modelos de chat (GGUF) devem ir
GGUF_MODELS_DIR = "./models"     #

# Cria os diretórios se não existirem
os.makedirs(EMBED_CACHE_DIR, exist_ok=True)
os.makedirs(GGUF_MODELS_DIR, exist_ok=True)

logger.info(f"Cache de Embeddings (CPU): {os.path.abspath(EMBED_CACHE_DIR)}")
logger.info(f"Modelos de Chat (GGUF):      {os.path.abspath(GGUF_MODELS_DIR)}")
logger.info("-" * 40)

# Use seu token HF do .env se existir
hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN", os.getenv("HF_TOKEN"))
api = HfApi(token=hf_token)

# --- 2. Download Automatizado do Modelo de Embedding ---

EMBED_MODEL_REPO_ID = "BAAI/bge-m3" # 

try:
    logger.info(f"Iniciando download do modelo de embedding: {EMBED_MODEL_REPO_ID}")
    snapshot_download(
        repo_id=EMBED_MODEL_REPO_ID,
        cache_dir=EMBED_CACHE_DIR, # Salva no cache da API
        token=hf_token,
        ignore_patterns=["*.onnx", "*.ot", "*.bin"], 
    )
    logger.info(f"SUCESSO: Modelo de embedding baixado para {EMBED_CACHE_DIR}")

except Exception as e:
    logger.error(f"FALHA ao baixar o modelo de embedding: {e}")
    logger.error("A API não funcionará sem este modelo.")

logger.info("-" * 40)

# --- 3. Download Automatizado dos Modelos GGUF ---
logger.info("--- Iniciando Download dos Modelos GGUF ---")

# Mapeamento dos modelos solicitados para os arquivos GGUF específicos
# { "Repo ID": "Arquivo GGUF dentro do repo" }
gguf_models_to_download = {
    "unsloth/Qwen3-4B-Thinking-2507-GGUF": "Qwen3-4B-Thinking-2507-Q6_K.gguf",
    "unsloth/Llama-3.2-1B-Instruct-GGUF": "Llama-3.2-1B-Instruct-Q5_K_M.gguf",
    "google/gemma-3-1b-it-qat-q4_0-gguf": "gemma-3-1b-it-q4_0.gguf",
    "google/gemma-3-4b-it-qat-q4_0-gguf": "gemma-3-4b-it-q4_0.gguf"
    # Adicionei suposições de arquivos Q4_K_M (quantização de 4 bits), que são comuns.
}


for repo_id, filename in gguf_models_to_download.items():
    try:
        logger.info(f"Baixando {filename} do repositório {repo_id}...")
        
        # Baixa o arquivo GGUF diretamente para o diretório ./models
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=GGUF_MODELS_DIR,
            local_dir_use_symlinks=False, # Garante que o arquivo seja copiado
            token=hf_token,
        )
        
        destination_path = os.path.join(GGUF_MODELS_DIR, filename)
        logger.info(f"SUCESSO: Modelo GGUF salvo em: {destination_path}")

    except Exception as e:
        logger.error(f"FALHA ao baixar {repo_id}/{filename}. Verifique se o nome do repositório e do arquivo estão corretos. Erro: {e}")

logger.info("-" * 40)
logger.info("--- Download de modelos concluído ---")
logger.info(f"Verifique se o seu arquivo .env (CANDIDATE_MODELS_LIST)")
logger.info(f"contém os NOMES DOS ARQUIVOS exatos que foram baixados para {GGUF_MODELS_DIR}:")
try:
    logger.info(f"Arquivos em ./models: {os.listdir(GGUF_MODELS_DIR)}")
except Exception as e:
    logger.error(f"Não foi possível listar arquivos em ./models: {e}")