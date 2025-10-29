from locust import HttpUser, task, between
import random

# ==========================================================
# 500 CONSULTAS DIVERSIFICADAS — 20 TEMAS × 25 PERGUNTAS
# ==========================================================

QUERIES = [

    # ======================================================
    # 🏛️ História (25)
    # ======================================================
    {"query": "Explique as causas da Primeira Guerra Mundial."},
    {"query": "O que foi a Revolução Francesa e seus efeitos políticos?"},
    {"query": "Quem foi Napoleão Bonaparte e qual seu impacto histórico?"},
    {"query": "Resuma a Independência do Brasil."},
    {"query": "Explique o que foi o Iluminismo."},
    {"query": "O que motivou a Revolução Industrial na Inglaterra?"},
    {"query": "Quem foi Mahatma Gandhi e sua importância histórica?"},
    {"query": "O que foi o Muro de Berlim e quando caiu?"},
    {"query": "O que foi a Guerra Fria e seus principais eventos?"},
    {"query": "O que caracterizou a Era Vargas no Brasil?"},
    {"query": "Qual foi o papel da ONU após a Segunda Guerra Mundial?"},
    {"query": "Explique a Ditadura Militar no Brasil."},
    {"query": "Quem foi Getúlio Vargas?"},
    {"query": "O que foi o Tratado de Versalhes?"},
    {"query": "Explique a corrida espacial entre EUA e URSS."},
    {"query": "O que foi o Renascimento Cultural e suas características?"},
    {"query": "Quem foi Leonardo da Vinci e sua importância?"},
    {"query": "Explique as Cruzadas."},
    {"query": "O que foi o Feudalismo e suas relações sociais?"},
    {"query": "Qual a importância da Revolução Russa de 1917?"},
    {"query": "O que foi o Apartheid e quem foi Nelson Mandela?"},
    {"query": "Explique a Inquisição e seu papel histórico."},
    {"query": "O que foi o imperialismo europeu?"},
    {"query": "Quais foram os principais navegadores das Grandes Navegações?"},
    {"query": "O que foi a Revolução Cubana?"},

    # ======================================================
    # 💻 Programação (25)
    # ======================================================
    {"query": "Escreva um código Python que inverte uma string."},
    {"query": "Explique o conceito de função lambda."},
    {"query": "O que é uma classe em Python?"},
    {"query": "Como tratar exceções em Python?"},
    {"query": "Explique o conceito de herança na OOP."},
    {"query": "Escreva um programa que calcule o fatorial em C."},
    {"query": "Explique o que é recursão."},
    {"query": "Escreva um código JavaScript que soma elementos de uma lista."},
    {"query": "O que é uma API REST?"},
    {"query": "Explique o conceito de variável imutável."},
    {"query": "O que é um dicionário em Python?"},
    {"query": "Explique a diferença entre lista, tupla e conjunto."},
    {"query": "Como usar pandas para agrupar dados?"},
    {"query": "Escreva um SQL para selecionar alunos com nota > 8."},
    {"query": "Explique o que é um container Docker."},
    {"query": "Como criar uma API com FastAPI?"},
    {"query": "O que é asyncio em Python?"},
    {"query": "Explique a diferença entre process e thread."},
    {"query": "O que é um webhook?"},
    {"query": "Como criar um Dockerfile básico?"},
    {"query": "Escreva um script Python que faz scraping de um site."},
    {"query": "Explique a diferença entre GET e POST."},
    {"query": "O que é uma expressão regular e como usá-la?"},
    {"query": "Como usar o módulo logging em Python?"},
    {"query": "Explique o conceito de CI/CD."},

    # ======================================================
    # ⚛️ Física (25)
    # ======================================================
    {"query": "Explique a lei da gravitação universal."},
    {"query": "O que é a teoria da relatividade?"},
    {"query": "Explique o conceito de energia cinética."},
    {"query": "O que é movimento uniformemente acelerado?"},
    {"query": "Explique a diferença entre massa e peso."},
    {"query": "O que é um campo magnético?"},
    {"query": "Explique o conceito de entropia."},
    {"query": "O que é o princípio da inércia?"},
    {"query": "Explique o conceito de trabalho e potência."},
    {"query": "O que é a lei de Coulomb?"},
    {"query": "Explique a diferença entre corrente contínua e alternada."},
    {"query": "O que é o efeito fotoelétrico?"},
    {"query": "Explique o conceito de torque."},
    {"query": "O que é o momento linear?"},
    {"query": "Explique o conceito de impulso."},
    {"query": "O que é a força centrípeta?"},
    {"query": "Explique o conceito de ondas mecânicas."},
    {"query": "O que é o som e como ele se propaga?"},
    {"query": "Explique a diferença entre reflexão e refração da luz."},
    {"query": "O que é o espectro eletromagnético?"},
    {"query": "Explique o conceito de calor e temperatura."},
    {"query": "O que é o princípio de Arquimedes?"},
    {"query": "Explique o funcionamento de um transformador elétrico."},
    {"query": "O que é energia potencial gravitacional?"},
    {"query": "Explique o conceito de pressão e como é medida."},

    # ======================================================
    # 🧪 Química (25)
    # ======================================================
    {"query": "Explique o conceito de ligação covalente."},
    {"query": "O que é um átomo e suas partículas?"},
    {"query": "Explique a diferença entre ácido e base."},
    {"query": "O que é uma reação de oxirredução?"},
    {"query": "Explique a tabela periódica."},
    {"query": "O que é uma ligação iônica?"},
    {"query": "Explique o conceito de mol."},
    {"query": "O que é uma reação endotérmica?"},
    {"query": "Explique o conceito de catalisador."},
    {"query": "O que é pH e como calculá-lo?"},
    {"query": "Explique o conceito de concentração molar."},
    {"query": "O que é uma reação de neutralização?"},
    {"query": "Explique o conceito de eletrólito."},
    {"query": "O que é uma mistura homogênea?"},
    {"query": "Explique o conceito de polaridade."},
    {"query": "O que é uma reação de combustão?"},
    {"query": "Explique o conceito de estequiometria."},
    {"query": "O que é uma substância pura?"},
    {"query": "Explique o conceito de isotopia."},
    {"query": "O que é uma reação de dupla troca?"},
    {"query": "Explique o conceito de equilíbrio químico."},
    {"query": "O que é uma ligação metálica?"},
    {"query": "Explique o conceito de solubilidade."},
    {"query": "O que é o número atômico?"},
    {"query": "Explique a diferença entre elemento e composto químico."},

    # ======================================================
    # 📊 Matemática (25)
    # ======================================================
    {"query": "Explique o teorema de Pitágoras."},
    {"query": "O que é uma progressão aritmética?"},
    {"query": "Explique o conceito de função exponencial."},
    {"query": "O que é um número primo?"},
    {"query": "Explique o conceito de derivada."},
    {"query": "O que é uma integral?"},
    {"query": "Explique o que é limite em cálculo."},
    {"query": "O que é uma matriz e como multiplicar?"},
    {"query": "Explique o conceito de determinante."},
    {"query": "O que é uma função quadrática?"},
    {"query": "Explique o conceito de probabilidade."},
    {"query": "O que é média, mediana e moda?"},
    {"query": "Explique o conceito de vetor."},
    {"query": "O que é um logaritmo?"},
    {"query": "Explique o conceito de geometria analítica."},
    {"query": "O que é um gráfico de função linear?"},
    {"query": "Explique o conceito de seno e cosseno."},
    {"query": "O que é trigonometria?"},
    {"query": "Explique o conceito de desigualdade."},
    {"query": "O que é uma equação diferencial?"},
    {"query": "Explique o conceito de sequência numérica."},
    {"query": "O que é o teorema de Tales?"},
    {"query": "Explique o conceito de área e volume."},
    {"query": "O que é análise combinatória?"},
    {"query": "Explique o conceito de estatística descritiva."},

    # ======================================================
    # 🌾 Agronegócio e IoT (25)
    # ======================================================
    {"query": "Explique o conceito de automação agrícola."},
    {"query": "O que é um sensor de umidade do solo?"},
    {"query": "Explique como funciona um sistema de irrigação automática."},
    {"query": "O que é um ESP32 e como aplicá-lo na agricultura?"},
    {"query": "Explique o conceito de Internet das Coisas (IoT)."},
    {"query": "O que é um microcontrolador?"},
    {"query": "Explique como usar um sensor DHT22."},
    {"query": "O que é MQTT e como aplicá-lo em fazendas inteligentes?"},
    {"query": "Explique o conceito de automação em aviários."},
    {"query": "O que é um pressostato?"},
    {"query": "Explique o papel dos relés em automação rural."},
    {"query": "O que é uma bomba submersa e como controlá-la?"},
    {"query": "Explique o conceito de sensoriamento remoto."},
    {"query": "O que é uma válvula solenoide?"},
    {"query": "Explique como usar um RTC em um projeto IoT."},
    {"query": "O que é um servidor NTP e por que é importante?"},
    {"query": "Explique o conceito de fotoperíodo em galinhas poedeiras."},
    {"query": "O que é um sistema de climatização de aviário?"},
    {"query": "Explique como medir temperatura e umidade no campo."},
    {"query": "O que é uma fonte de alimentação estabilizada?"},
    {"query": "Explique como controlar motores com ESP32."},
    {"query": "O que é uma rede LoRaWAN?"},
    {"query": "Explique o conceito de telemetria agrícola."},
    {"query": "O que é um sensor de amônia?"},
    {"query": "Explique o conceito de manutenção preditiva no campo."},
]

# ==========================================================
# Classe Locust
# ==========================================================
class RouterUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://llm_router_api:8000"

    @task
    def send_query(self):
        q = random.choice(QUERIES)
        payload = {
            "query": q["query"],
            "enable_rag_for_answer": False,
            "max_tokens": 2048,
            "temperature": 0.3
        }
        self.client.post("/query", json=payload)
