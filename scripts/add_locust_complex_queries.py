#!/usr/bin/env python3
"""Insert one # Complexas (1) query per Locust QUERIES category."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tests" / "locustfile.py"

# (anchor_line_exact, complex_query_text)
INSERTIONS: list[tuple[str, str]] = [
    # --- 19 categorias emoji (após Novas 2) ---
    (
        '    {"query": "O que foi o \'Holocausto\' e sua importância para os direitos humanos?"},',
        "Sintetize, em narrativa analítica, como a Primeira Guerra Mundial, o Tratado de Versalhes, a crise de 1929 e o ascenso do totalitarismo formam uma cadeia causal que explica a Segunda Guerra Mundial; inclua três historiografias concorrentes e indique quais evidências as sustentam.",
    ),
    (
        '    {"query": "Como funciona o \'garbage collection\' em linguagens como Python ou Java?"},',
        "Projete, em Python, um sistema de roteamento de APIs com rate limiting, circuit breaker, retries exponenciais, observabilidade estruturada e testes de propriedade; explique trade-offs entre consistência, latência e custo em monólito vs microsserviços.",
    ),
    (
        '    {"query": "O que é \'spintronics\'?"},',
        "Derive conceitualmente como a relatividade geral altera a noção newtoniana de gravidade, discuta testes clássicos (déficit de Mercúrio, desvio da luz, ondas gravitacionais) e analise limites onde mecânica quântica e gravitação ainda não se unificam.",
    ),
    (
        '    {"query": "Explique o \'estado de transição\' em uma reação química."},',
        "Explique termodinâmica e cinética da Reação de Maillard em três escalas (molecular, culinária e metabólica), predizendo como temperatura, pH e umidade alteram perfil sensorial e potenciais produtos tóxicos.",
    ),
    (
        '    {"query": "O que é \'criptografia de chave pública\' (ex: RSA) e a matemática por trás dela?"},',
        "Resolva e interprete um problema de otimização multiobjetivo com restrições lineares e não lineares: formule o Lagrangiano, discuta condições KKT, compare solução analítica com método numérico e comente sensibilidade dos pesos na fronteira de Pareto.",
    ),
    (
        '    {"query": "Como a IA em \'edge devices\' pode detectar padrões anômalos (ex: doença em folhas) localmente?"},',
        "Desenhe uma arquitetura edge-to-cloud para monitoramento de safra com sensores de solo, visão em drones e modelos preditivos de pragas; avalie TCO, latência, segurança e impacto em produtividade versus agricultura convencional.",
    ),
    (
        '    {"query": "Analise a \'Guerra Fria\' sob uma perspectiva geopolítica de disputa por zonas de influência."},',
        "Analise a Amazônia como sistema socioecológico: conecte desmatamento, políticas públicas, cadeias globais de commodities e feedbacks climáticos, propondo indicadores espaciais para governança territorial.",
    ),
    (
        '    {"query": "Descreva a \'autenticação 802.1X\' para controle de acesso à rede (NAC)."},',
        "Compare arquiteturas de rede para data center híbrido (BGP, OSPF, EVPN-VXLAN, service mesh): discuta convergência, segurança zero-trust, observabilidade e continuidade sob ataque DDoS e falha de região.",
    ),
    (
        '    {"query": "O que é \'Word2Vec\' (Skip-gram, CBOW) e \'embeddings\' de palavras?"},',
        "Critique um pipeline de LLM em produção: dados, fine-tuning, avaliação humana, detecção de alucinação, custo por token, latência P95 e riscos de viés; proponha governança com métricas e gates de release.",
    ),
    (
        '    {"query": "Explique o que é \'PCR\' (Reação em Cadeia da Polimerase) e como é usada em diagnósticos."},',
        "Explique o ciclo central de carbono na célula (glicólise, ciclo de Krebs, cadeia respiratória) integrando bioquímica, regulação hormonal e implicações clínicas em doenças metabólicas.",
    ),
    (
        '    {"query": "Explique o que é \'Taxa Selic\' e como o \'COPOM\' a utiliza para controlar a inflação."},',
        "Modele o impacto de um choque monetário (alta de juros) em inflação, câmbio, dívida pública e mercado de trabalho em economia emergente; compare visões keynesiana, monetarista e estruturalista.",
    ),
    (
        '    {"query": "Discuta a \'Semana de Arte Moderna de 22\' e seu impacto na literatura."},',
        "Faça leitura comparativa de modernismo e pós-modernismo na literatura brasileira: analise voz narrativa, intertextualidade, política de linguagem e função social da obra, citando dois autores e um poema.",
    ),
    (
        '    {"query": "O que é \'apartheid\' e como a sociologia analisa a \'segregação racial\'?"},',
        "Discuta o dilema entre utilitarismo, deontologia e ética da virtude aplicado a decisões de IA em saúde pública; inclua argumentos de Rawls, Kant e Foucault sobre justiça, poder e responsabilidade institucional.",
    ),
    (
        '    {"query": "O que foi o \'Tropicália\' (Tropicalismo) no Brasil e sua fusão cultural?"},',
        "Analise uma obra barroca e uma instalação contemporânea: compare técnica, patronato, recepção crítica e papel político da arte; relacione com teoria estética (Burckhardt, Benjamin, Adorno).",
    ),
    (
        '    {"query": "O que é \'guerra fiscal\' entre os estados no Brasil?"},',
        "Avalie o tensionamento entre separação de poderes, ativismo judicial e accountability democrática em democracias contemporâneas; use casos hipotéticos de controle de constitucionalidade e impeachment.",
    ),
    (
        '    {"query": "Discuta a ascensão dos \'e-sports\' como fenômeno cultural e econômico."},',
        "Analise a estética do cinema de Christopher Nolan e de Glauber Rocha como duas respostas à crise narrativa contemporânea: montagem, tempo, ideologia e recepção crítica em contextos nacionais distintos.",
    ),
    (
        '    {"query": "Descreva o \'método de extração\' de café \'espresso\' vs \'coado\' (V60)."},',
        "Modele a ciência do café como sistema físico-químico: extração, temperatura, moagem, TDS e percepção sensorial; proponha protocolo experimental para otimizar xícara sem sacrificar consistência.",
    ),
    (
        '    {"query": "O que é \'turismo espacial\' e quais empresas estão liderando (SpaceX, Blue Origin)?"},',
        "Avalie a viabilidade de missões tripuladas a Marte até 2040: propulsão, radiação, suporte vital, custo e governança internacional; compare cenários otimista, base e pessimista.",
    ),
    (
        '    {"query": "Como a \'impressão 3D\' (Programação/Engenharia) está mudando a \'cadeia de suprimentos\' (Economia) e quais os desafios de \'propriedade intelectual\' (Direito) de \'blueprints\' digitais?"},',
        "Integre ética, direito, economia e engenharia em um caso de IA generativa em saúde: desde coleta de dados clínicos até responsabilidade civil por erro diagnóstico e impacto distributivo no SUS.",
    ),
    # --- Foco Meta ---
    (
        '    {"query": "Qual \'o\' \'futuro\' \'do\' \'trabalho\' (Economia/Sociologia) \'com\' \'a\' \'automação\' (IA/Programação)? \'Debata\' \'Renda\' \'Básica\' \'Universal\' (Economia/Filosofia) \'vs.\' \'requalificação\' (Educação/Economia). \'Como\' \'a\' \'sociedade\' (Sociologia) \'encontraria\' \'propósito\' (Filosofia/Psicologia) \'sem\' \'o\' \'emprego\' \'tradicional\' (História)?"},',
        "Construa um ensaio integrador sobre governança global em três frentes simultâneas — clima, IA e desigualdade — usando teoria dos jogos, direito internacional e economia política para propor mecanismos institucionais viáveis.",
    ),
    # --- Brasil (5 subcategorias) ---
    (
        '    {"query": "Discuta a \'transferência\' da \'corte portuguesa\' (História) para o \'Brasil\' (1808). Quais \'impactos\' \'culturais\' (Arte/Música), \'políticos\' (Política) e \'econômicos\' (Economia) \'irreversíveis\' (Filosofia) isso \'causou\' na \'colônia\', \'levando\' à \'Independência\' (História)?"},',
        "Reconstrua a formação do Brasil como processo de longa duração: escravidão, mineração, café, industrialização tardia e ditadura; identifique continuidades estruturais que explicam desigualdade contemporânea.",
    ),
    (
        '    {"query": "Analise a \'Bacia do Prata\' (Geografia/Geopolítica). Qual a \'importância\' \'histórica\' (História) (ex: Guerra do Paraguai) e \'econômica\' (Economia) (escoamento de safra, \'Hidrelétrica de Itaipu\') (Física) \'desta\' \'bacia hidrográfica\' (Geografia)?"},',
        "Modele o Brasil como sistema territorial: hidrologia, energia, logística, fronteiras e clima; proponha indicadores integrados para políticas de adaptação e competitividade regional.",
    ),
    (
        '    {"query": "O que é \'economia informal\' (Economia/Sociologia) no \'Brasil\'? Analise \'suas\' \'causas\' (alta carga tributária, burocracia) (Direito/Economia) e \'consequências\' (falta de direitos trabalhistas) (Direito/Saúde)."},',
        "Simule reformas tributária, previdenciária e cambial no Brasil em cenário de choque externo: efeitos em inflação, emprego, dívida e competitividade industrial em horizonte de dez anos.",
    ),
    (
        '    {"query": "Analise a \'diplomacia\' \'brasileira\' (Política/Geopolítica). Discuta a \'tradição\' (História) do \'Itamaraty\' (Direito) (multilateralismo, \'soft power\') (Sociologia) \'versus\' \'novas\' \'abordagens\' \'ideológicas\' (Filosofia) na \'política externa\' (Política)."},',
        "Avalie o presidencialismo de coalizão brasileiro à luz de crises institucionais recentes: STF, Congresso, Forças Armadas e mídias digitais; proponha reformas políticas com trade-offs explícitos.",
    ),
    (
        '    {"query": "Discuta \'Língua Brasileira de Sinais\' (LIBRAS) (Gramática/Sociologia). É \'apenas\' uma \'tradução\' (Gramática) do \'português\' ou uma \'língua\' \'completa\' (Filosofia) com \'gramática\' \'própria\' (Gramática)? \'Analise\' a \'luta\' (Política) da \'comunidade surda\' (Saúde/Sociologia) por \'direitos\' (Direito)."},',
        "Examine identidade nacional brasileira através de sincretismo religioso, música, esporte e desigualdade racial; discuta se o mito da democracia racial ainda estrutura políticas públicas contemporâneas.",
    ),
    # --- Regionais (antes dos blocos de 25 difíceis ou ao fim da seção) ---
    (
        '    {"query": "What is the \'Sun Belt\' and why has it experienced population growth?"},',
        "Sintetize a geopolítica da América do Norte no século XXI: USMCA, competição EUA-China, energia, migração e segurança cibernética; proponha três cenários estratégicos para a década de 2030.",
    ),
    (
        '    {"query": "Analyze the shift in the \'center of gravity\' within the EU towards the East (particularly Poland and the Baltic states) following the 2022 Ukraine war, and assess how this will challenge the traditional dominance of the Franco-German axis."},',
        "Avalie a integração europeia pós-2022 como sistema complexo: Ucrânia, energia, migração, direito da UE e defesa; identifique pontos de ruptura e mecanismos de resiliência institucional.",
    ),
    (
        '    {"query": "Prove the Central Limit Theorem using characteristic functions and Lévy\'s continuity theorem."},',
        "Construa um roteiro de demonstração que conecte análise real, álgebra linear, teoria da medida e otimização convexa para explicar por que problemas difíceis em ML frequentemente reduzem a geometria de alta dimensão.",
    ),
    (
        '    {"query": "Explain the challenge of the Kurdish quest for statehood in Turkey, Syria, Iraq, and Iran."},',
        "Analise a Ásia como tabuleiro multipolar: China, Índia, ASEAN, Oriente Médio e Coreia; integre economia, segurança, demografia e tecnologia em um mapa de riscos para os próximos quinze anos.",
    ),
    (
        '    {"query": "O que é o transplante de medula óssea?"},',
        "Desenhe um plano nacional de saúde para doenças crônicas e envelhecimento populacional: prevenção, telemedicina, IA diagnóstica, financiamento e equidade, com métricas de impacto em cinco anos.",
    ),
    (
        '    {"query": "Quais os desafios geopolíticos do Ártico com o derretimento do gelo?"},',
        "Modele mudanças climáticas como problema geográfico sistêmico: água, energia, cidades costeiras, migração e biodiversidade; proponha políticas adaptativas com análise custo-benefício regionalizada.",
    ),
]


def _format_block(anchor: str, query: str) -> str:
    escaped = query.replace("\\", "\\\\").replace('"', '\\"')
    return (
        f"{anchor}\n\n"
        f"    # Complexas (1)\n"
        f'    {{"query": "{escaped}"}},\n'
    )


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    added = 0
    skipped = 0
    missing: list[str] = []

    for anchor, query in INSERTIONS:
        if anchor not in text:
            missing.append(anchor[:80])
            continue
        pos = text.find(anchor)
        window = text[pos : pos + len(anchor) + 500]
        if "# Complexas (1)" in window:
            skipped += 1
            continue
        text = text.replace(anchor, _format_block(anchor, query), 1)
        added += 1

    PATH.write_text(text, encoding="utf-8")

    mod = ast.parse(PATH.read_text(encoding="utf-8"))
    count = None
    for node in mod.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "QUERIES":
                    count = len(node.value.elts)

    print(f"Added: {added}, skipped (already present): {skipped}")
    print(f"QUERIES count: {count}")
    if missing:
        print(f"Missing anchors: {len(missing)}", file=sys.stderr)
        for m in missing:
            print(f"  - {m}...", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
