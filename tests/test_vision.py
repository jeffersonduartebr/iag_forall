"""Módulo `tests/test_vision.py`: descreve responsabilidades e integrações deste arquivo."""

import requests
import base64
import os
import glob
import time
import io
from PIL import Image

# ==============================================================================
# ⚙️ CONFIGURAÇÃO
# ==============================================================================
API_URL = "http://localhost:8000/query"
IMAGE_DIR = "."  # Diretório atual (ou mude para "tests/assets")
MAX_IMAGE_SIZE = 384  # Redimensiona o maior lado para 768px (Ideal para VLMs)

# ==============================================================================
# 🛠️ FUNÇÕES AUXILIARES
# ==============================================================================

def encode_image_optimized(image_path):
    """
    Lê a imagem, converte para RGB, redimensiona se necessário e retorna Base64.
    Isso evita enviar imagens 4K que causam lentidão extrema ou erro de VRAM.
    """
    if not os.path.exists(image_path):
        print(f"❌ Erro: Arquivo {image_path} não encontrado.")
        return None

    try:
        with Image.open(image_path) as img:
            # 1. Converter para RGB (remove canal Alpha que quebra alguns modelos)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # 2. Redimensionar se for gigante (mantendo proporção)
            width, height = img.size
            if max(width, height) > MAX_IMAGE_SIZE:
                img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
                # print(f"   📉 Redimensionado de {width}x{height} para {img.size}")
            
            # 3. Converter para Base64
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=70) # JPEG leve
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
    except Exception as e:
        print(f"❌ Erro ao processar imagem {image_path}: {e}")
        return None

def send_request(image_path):
    """Envia uma única imagem para o Router."""
    filename = os.path.basename(image_path)
    print(f"\n{'='*60}")
    print(f"🖼️  Processando: {filename}")
    
    b64_string = encode_image_optimized(image_path)
    if not b64_string: return

    payload = {
        "query": "Descreva detalhadamente o que você vê nesta imagem.",
        "modality": "vision",   # Força o Router a olhar para a lista de visão
        "image_b64": b64_string,
        "max_tokens": 512,
        "model": "ollama/llava:7b",  # Força um modelo VLM específico
        "temperature": 0.1,     # Baixa temperatura para descrições factuais
        "use_cache": False,      # Força processamento novo para teste
        # DESLIGA O RAG EXPLICITAMENTE
        "enable_rag_for_answer": False, 
        "enable_rag_for_image": False
    }

    start = time.time()
    try:
        # Timeout generoso (180s) para casos de Cold Start do Ollama
        response = requests.post(API_URL, json=payload, timeout=180)
        latency = time.time() - start

        if response.status_code == 200:
            data = response.json()
            print(f"✅ SUCESSO em {latency:.2f}s")
            print(f"🤖 Modelo Escolhido: {data.get('model', 'N/A')}")
            
            # Se o router usar UQ, mostra a incerteza (se disponível no metadata)
            route_meta = data.get("route", {})
            if "uncertainty" in route_meta.get("objectives", {}):
                print(f"❓ Incerteza (UQ): {route_meta['objectives']['uncertainty']:.2f}")

            print("-" * 40)
            print(f"📝 Resposta:\n{data.get('answer', '').strip()}")
        else:
            print(f"❌ Erro {response.status_code}: {response.text}")

    except Exception as e:
        print(f"❌ Falha de conexão/Timeout: {e}")

# ==============================================================================
# 🚀 LOOP PRINCIPAL
# ==============================================================================
def main():
    # Busca extensões comuns (case insensitive para Windows/Linux)
    """Executa main."""
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.JPG', '*.PNG']
    files = []
    
    for p in patterns:
        files.extend(glob.glob(os.path.join(IMAGE_DIR, p)))
    
    # Remove duplicatas e ordena
    files = sorted(list(set(files)))

    if not files:
        print(f"⚠️ Nenhuma imagem encontrada no diretório: {os.path.abspath(IMAGE_DIR)}")
        return

    print(f"📂 Encontradas {len(files)} imagens. Iniciando bateria de testes...")

    for img_file in files:
        send_request(img_file)
        # Pequena pausa para dar fôlego à GPU entre requests
        time.sleep(5)

if __name__ == "__main__":
    # Verifica dependência
    try:
        import PIL
    except ImportError:
        print("⚠️ Biblioteca Pillow não instalada. Rodando: pip install pillow")
        os.system("pip install pillow")
        import PIL
        
    main()