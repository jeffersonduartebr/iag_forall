import requests
import base64
import json
import os
import glob
import time

# Configuração
API_URL = "http://localhost:8000/query"
# Diretório onde buscar as imagens (ponto . significa diretório atual)
IMAGE_DIR = "." 

def encode_image(image_path):
    """Lê o arquivo e converte para Base64."""
    if not os.path.exists(image_path):
        print(f"❌ Erro: Imagem {image_path} não encontrada.")
        return None
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ Erro ao ler {image_path}: {e}")
        return None

def process_image(image_path):
    """Envia uma única imagem para a API."""
    print(f"\n{'='*60}")
    print(f"🖼️  Processando arquivo: {image_path}")
    
    b64_string = encode_image(image_path)
    if not b64_string:
        return

    payload = {
        "query": "Descreva detalhadamente o que você vê nesta imagem. Se houver texto, transcreva-o.",
        "modality": "vision",  # Força o modo visão
        "image_b64": b64_string,
        "max_tokens": 512,
        "temperature": 0.2
    }

    print("🚀 Enviando requisição para o Router...")
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, json=payload, timeout=300) # Timeout alto para VLMs
        
        if response.status_code == 200:
            data = response.json()
            duration = time.time() - start_time
            
            print("✅ SUCESSO!")
            print(f"🤖 Modelo: {data.get('model', 'Desconhecido')}")
            print(f"⏱️  Latência Total: {duration:.2f}s")
            print(f"💰 Custo Est.: ${data.get('cost_per_1k', 0):.6f}")
            print("-" * 40)
            print("📝 RESPOSTA:")
            print(data.get('answer', 'Sem resposta textual.'))
        else:
            print(f"❌ Erro {response.status_code}: {response.text}")

    except Exception as e:
        print(f"❌ Falha na conexão ou timeout: {e}")

def main():
    # Busca arquivos com extensões de imagem comuns
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    
    for ext in extensions:
        # glob.glob busca arquivos que correspondem ao padrão
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        # Para case-insensitive no Windows, geralmente funciona, 
        # mas no Linux *.jpg não acha .JPG.
        # Adicionando versão uppercase para garantir:
        image_files.extend(glob.glob(os.path.join(IMAGE_DIR, ext.upper())))

    # Remove duplicatas e ordena
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print(f"⚠️ Nenhuma imagem ({', '.join(extensions)}) encontrada em '{IMAGE_DIR}'")
        return

    print(f"📂 Encontradas {len(image_files)} imagens. Iniciando processamento em série...")
    
    for img in image_files:
        process_image(img)
        # Pequena pausa para não sobrecarregar se for rodar localmente em CPU
        time.sleep(1) 

if __name__ == "__main__":
    main()