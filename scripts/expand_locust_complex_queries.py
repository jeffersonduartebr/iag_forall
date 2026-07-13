#!/usr/bin/env python3
"""Expand each # Complexas (1) block to # Complexas (5) with four additional queries."""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tests" / "locustfile.py"

# Four additional complex queries per category (31 categories, in file order).
EXTRA_BY_CATEGORY: list[list[str]] = [
    # 1 História
    [
        "Compare revoluções atlânticas (EUA, França, Haiti): causas sociais, modelos de cidadania e legados para direitos civis contemporâneos.",
        "Analise o imperialismo europeu no século XIX como sistema econômico-militar: partilha da África, rivalidades e feedbacks na Primeira Guerra Mundial.",
        "Discuta memória histórica e revisionismo no debate sobre ditaduras latino-americanas: arquivos, comissões da verdade e narrativas públicas.",
        "Reconstrua a Guerra Fria no Terceiro Mundo: descolonização, golpes patrocinados, movimentos de não alinhamento e impacto em desenvolvimento.",
    ],
    # 2 Programação
    [
        "Desenhe uma API GraphQL com autenticação OAuth2, cache em camadas, paginação cursor-based e estratégia de versionamento sem breaking changes.",
        "Explique como implementar um compilador minimalista (lexer, parser, AST, codegen) e compare com interpretação JIT para uma linguagem educacional.",
        "Projete pipeline CI/CD com testes de contrato, análise estática, canary deploy e rollback automático; discuta SLOs e error budgets.",
        "Avalie trade-offs entre programação orientada a eventos, atores e CSP para sistemas distribuídos de alta concorrência.",
    ],
    # 3 Física
    [
        "Derive a equação de Schrödinger para partícula livre e interprete probabilidade, colapso e interpretações (Copenhague, many-worlds, pilot-wave).",
        "Explique radiação de corpo negro, efeito fotoelétrico e modelo de Bohr como passos históricos para a mecânica quântica.",
        "Analise entropia e segunda lei em sistemas abertos: máquinas térmicas, refrigeradores e limites de eficiência em engenharia energética.",
        "Discuta dualidade onda-partícula em interferência e emaranhamento quântico; por que não permite comunicação superluminal?",
    ],
    # 4 Química
    [
        "Modele cinética química de reações catalíticas heterogêneas em reatores industriais: etapas, velocidade aparente e deativação de catalisador.",
        "Explique equilíbrio químico, princípio de Le Chatelier e espontaneidade via energia livre de Gibbs em processos eletroquímicos.",
        "Compare polímeros termoplásticos e termofixos em propriedades mecânicas, reciclagem e aplicações em engenharia de materiais.",
        "Analise química verde: solventes alternativos, atom economy e avaliação de ciclo de vida em síntese farmacêutica.",
    ],
    # 5 Matemática
    [
        "Prove o teorema fundamental do cálculo e aplique-o para resolver problema de otimização com restrições em economia e engenharia.",
        "Explique espaços vetoriais, autovalores e decomposição espectral com aplicação em PCA e redução de dimensionalidade.",
        "Discuta incompletude, decidibilidade e complexidade computacional: P vs NP e impacto em criptografia e otimização.",
        "Modele cadeias de Markov e processos estocásticos para filas, fiabilidade de sistemas e simulação Monte Carlo.",
    ],
    # 6 Agronegócio e IoT
    [
        "Projete rede LoRaWAN para monitoramento de 10 mil hectares: topologia, duty cycle, energia e integração com ERP agrícola.",
        "Avalie agricultura de precisão com ML: calibração de sensores, drift de modelo e custo-benefício para pequenos produtores.",
        "Compare irrigação por gotejamento inteligente vs pivô central em cenários de escassez hídrica e tarifação dinâmica.",
        "Desenhe digital twin de fazenda: simulação de safra, clima, pragas e logística com indicadores ESG e risco climático.",
    ],
    # 7 Geografia
    [
        "Analise urbanização global e segregação espacial: gentrificação, periferias e políticas de moradia em metrópoles emergentes.",
        "Explique mudanças climáticas regionais via teleconexões (El Niño, AMOC) e impactos em agricultura, energia e migração.",
        "Discuta geopolítica de recursos hídricos transfronteiriços: bacias compartilhadas, tratados e conflitos latentes.",
        "Modele vulnerabilidade costeira a eventos extremos: elevação do nível do mar, seguros, adaptação e justiça espacial.",
    ],
    # 8 Redes
    [
        "Projete arquitetura multi-cloud com SD-WAN, segmentação microperímetro e resposta a incidentes em tempo real.",
        "Compare TCP vs QUIC vs HTTP/3 para APIs globais: latência, perda de pacotes, 0-RTT e implicações de segurança.",
        "Explique roteamento interdomínios (BGP), políticas de anúncio, hijacking e mitigação com RPKI/ROV.",
        "Desenhe observabilidade de rede com NetFlow, eBPF e tracing distribuído para diagnosticar gargalos em microsserviços.",
    ],
    # 9 IA
    [
        "Compare arquiteturas Transformer, Mamba e híbridas para contexto longo: custo, memória, latência e qualidade em RAG.",
        "Projete avaliação de LLM com golden sets, juízes LLM, métricas humanas e testes de regressão para releases seguros.",
        "Discuta alinhamento (RLHF, DPO, constitutional AI): trade-offs entre utilidade, segurança e censura indesejada.",
        "Analise riscos de prompt injection, data poisoning e model stealing em APIs de IA expostas publicamente.",
    ],
    # 10 Economia
    [
        "Modele inflação persistente em economia aberta: expectativas, indexação, câmbio e política monetária com metas duplas.",
        "Analise externalidades e impostos Pigouvianos vs sistemas cap-and-trade para descarbonização industrial.",
        "Explique crises financeiras via assimetria de informação, alavancagem e contagion em redes bancárias globais.",
        "Compare modelos de crescimento endógeno e institucionalismo: inovação, capital humano e qualidade de governança.",
    ],
    # 11 Biologia
    [
        "Explique síntese, expressão e regulação gênica (transcrição, epigenética, CRISPR) e implicações éticas em terapia gênica.",
        "Analise evolução por seleção natural, deriva genética e coevolução hospedeiro-patógeno com exemplos epidemiológicos.",
        "Discuta microbioma humano e eixo intestino-cérebro: evidências, limites da evidência e aplicações clínicas.",
        "Modele dinâmica populacional de espécies invasoras e estratégias de conservação em hotspots de biodiversidade.",
    ],
    # 12 Literatura
    [
        "Analise narratologia em romance policial e realismo mágico: focalização, tempo narrativo e pacto de leitura.",
        "Compare tradução literária literal vs poética usando um trecho multilíngue; discuta perda pragmática e cultural.",
        "Explique gêneros discursivos (ensaio, crônica, manifesto) e função social da literatura em regimes autoritários.",
        "Faça leitura intertextual entre poesia concreta e literatura digital: materialidade, código e leitor implicado.",
    ],
    # 13 Filosofia e Sociologia
    [
        "Discuta justiça distributiva (Rawls, Nozick, Sen): meritocracia, reparação histórica e capacidades em políticas públicas.",
        "Analise biopoder e governamentalidade em Foucault aplicados a vigilância digital e gestão de risco em saúde pública.",
        "Compare existencialismo, estoicismo e budismo na construção de sentido diante de incerteza existencial contemporânea.",
        "Explique teoria crítica da raça e interseccionalidade: estrutura social, experiência vivida e políticas antidiscriminação.",
    ],
    # 14 Arte e Música
    [
        "Analise harmonia funcional vs atonalidade em Schoenberg: contexto histórico, percepção auditiva e recepção crítica.",
        "Compare curadoria de museu tradicional e exposições imersivas digitais: autenticidade, acessibilidade e economia cultural.",
        "Discuta apropriação cultural vs hibridismo em música popular global: direitos, representação e mercado.",
        "Explique semiótica visual em propaganda e publicidade: cores, composição e construção de desejo.",
    ],
    # 15 Direito e Política
    [
        "Analise controle de constitucionalidade difuso vs concentrado: legitimidade democrática e estabilidade jurídica.",
        "Discuta direitos fundamentais em conflito (liberdade de expressão vs dignidade): teste de proporcionalidade e casos-limite.",
        "Explique federalismo fiscal e competências tributárias: guerra fiscal, pacto federativo e reformas estruturais.",
        "Avalie regulação de plataformas digitais: responsabilidade intermediária, moderação de conteúdo e transparência algorítmica.",
    ],
    # 16 Cinema
    [
        "Analise montagem soviética (Eisenstein) vs continuidade clássica de Hollywood: cognição, emoção e ideologia.",
        "Discuta representação de gênero e raça em blockbusters contemporâneos: estereótipos, contranarrativas e bilheteria.",
        "Explique economia política do streaming: dados de audiência, cancelamentos e impacto na diversidade autoral.",
        "Compare documentário observacional e docuficção: ética, verdade e manipulação narrativa.",
    ],
    # 17 Culinária
    [
        "Analise fermentação (pão, queijo, kombucha) em microbiologia, segurança alimentar e identidade cultural regional.",
        "Explique pairing molecular e neurogastronomia: por que certas combinações funcionam e limites do marketing gastronômico.",
        "Discuta sustentabilidade na gastronomia: pegada hídrica, desperdício e cardápios de baixo carbono em restaurantes.",
        "Modele cadeia do frio e HACCP em cozinha industrial: pontos críticos, auditoria e recall.",
    ],
    # 18 Astronomia
    [
        "Explique formação estelar, sequência principal e fim de vida de estrelas massivas (supernova, buraco negro).",
        "Analise métodos de detecção de exoplanetas (trânsito, velocidade radial) e critérios de habitabilidade.",
        "Discuta cosmologia observacional: radiação cósmica de fundo, energia escura e parâmetros do modelo ΛCDM.",
        "Avalie missões de astrobioogia e SETI: protocolos, falsificabilidade e implicações sociopolíticas de um contato.",
    ],
    # 19 Transversais
    [
        "Integre história, economia e direito na análise de sanções internacionais: eficácia, spillovers humanitários e alternativas diplomáticas.",
        "Discuta educação, IA e mercado de trabalho: currículos adaptativos, certificação e risco de automação de funções cognitivas.",
        "Analise energia nuclear vs renováveis em matriz elétrica: segurança, custo nivelado, descarte e geopolítica.",
        "Projete framework de avaliação de impacto social de tecnologias emergentes (blockchain, biotech, LLMs) com métricas multidisciplinares.",
    ],
    # 20 Foco Meta
    [
        "Desenvolva cenário prospectivo para AGI até 2040: capacidades, riscos existenciais, governança internacional e papel do setor privado.",
        "Analise colapso climático e migração forçada com modelagem de fluxos, direito internacional e pressão populista.",
        "Discuta epistemologia da ciência em era de big data: reprodutibilidade, p-hacking e papel de meta-análises.",
        "Proponha constituição digital hipotética: privacidade, soberania de dados, IA pública e participação cidadã.",
    ],
    # 21 Brasil História
    [
        "Releia Independência do Brasil como negociação de elites: escravidão, monarquia e continuidades coloniais.",
        "Analise ditadura militar brasileira: repressão, modernização econômica, memória e impunidade institucional.",
        "Discuta abolicionismo tardio e pós-abolição: liberdade formal, exclusão econômica e políticas de reparação.",
        "Compare movimentos sociais urbanos e rurais no Brasil republicano: MST, sindicatos, direitos e repressão.",
    ],
    # 22 Brasil Geografia
    [
        "Modele vulnerabilidade climática do Nordeste: secas, agricultura familiar e políticas de adaptação territorial.",
        "Analise metropolização brasileira: periferização, mobilidade, violência e planejamento urbano integrado.",
        "Discuta integração logística sul-americana: hidrovias, ferrovias, portos e competitividade do agronegócio.",
        "Explique dinâmica da Amazônia legal: fiscalização, garimpo, povos indígenas e soberania nacional.",
    ],
    # 23 Brasil Economia
    [
        "Simule impacto da reforma tributária sobre setores (indústria, serviços, agro) com elasticidades e arrecadação regional.",
        "Analise dívida pública brasileira: trajetória, risco fiscal, indexação e espaço para política anticíclica.",
        "Discuta inserção do Brasil em cadeias globais de valor: manufatura, commodities e política industrial verde.",
        "Avalie mercado de trabalho brasileiro: informalidade, plataformas digitais, qualificação e desigualdade regional.",
    ],
    # 24 Brasil Política
    [
        "Analise federalismo brasileiro em crises sanitárias: coordenação, competências e litígios federativos.",
        "Discuta reforma política: cláusula de barreira, financiamento de campanha e representatividade de minorias.",
        "Explique relação Executivo-Legislativo em coalizões frágeis: agendas, emendas e governabilidade.",
        "Avalie papel das mídias e redes sociais na polarização e qualidade do debate democrático no Brasil.",
    ],
    # 25 Brasil Cultura
    [
        "Analise funk, samba e rap como produção cultural periférica: estética, criminalização e economia criativa.",
        "Discuta políticas culturais (Lei Rouanet, editais) e concentração de recursos no Sudeste vs interiorização.",
        "Explique religiosidade brasileira contemporânea: pentecostalismo, pluralismo e influência política.",
        "Compare identidades regionais (Nordeste, Sul, Amazônia) na literatura, música e disputas simbólicas nacionais.",
    ],
    # 26 América do Norte
    [
        "Evaluate US-China strategic competition in semiconductors, AI, and rare earths with scenarios for allied reshoring.",
        "Analyze Indigenous treaty rights and resource extraction conflicts across Canada and the United States.",
        "Assess North American energy transition: shale, renewables, grid interconnection, and cross-border policy alignment.",
        "Deconstruct the political economy of Big Tech regulation (antitrust, Section 230, privacy) across US, Canada, and Mexico.",
    ],
    # 27 Europa
    [
        "Analyze EU enlargement fatigue versus security imperatives: Western Balkans, Ukraine, and institutional reform needs.",
        "Evaluate European energy security after Russian gas decoupling: LNG, nuclear, renewables, and industrial competitiveness.",
        "Discuss democratic backsliding in Hungary and Poland through EU legal mechanisms and national constitutional identity.",
        "Assess the future of the Eurozone banking union, fiscal capacity, and sovereign debt sustainability in southern Europe.",
    ],
    # 28 Matemática avançada
    [
        "Outline a proof strategy for the Prime Number Theorem and explain its connection to the Riemann zeta function.",
        "Discuss category theory as a unifying language for algebra and topology with examples of functors and natural transformations.",
        "Explain measure-theoretic probability foundations and why almost-sure convergence differs from convergence in probability.",
        "Analyze optimization on manifolds (Riemannian gradient descent) and its role in modern machine learning on constrained spaces.",
    ],
    # 29 Ásia
    [
        "Evaluate China's semiconductor self-sufficiency drive and its impact on Taiwan, South Korea, and global supply chains.",
        "Analyze India's demographic dividend versus jobless growth: education, manufacturing, and urban infrastructure gaps.",
        "Assess ASEAN centrality amid US-China rivalry in the South China Sea and Mekong water politics.",
        "Discuss Japan's aging society, immigration constraints, and monetary policy limits in a low-growth equilibrium.",
    ],
    # 30 Medicina
    [
        "Design a national antimicrobial stewardship program balancing hospital protocols, agriculture use, and pharmaceutical incentives.",
        "Analyze ethical frameworks for triage during pandemic surges: utilitarian allocation, equity, and legal liability.",
        "Evaluate real-world evidence vs randomized trials for approving adaptive therapies in oncology and rare diseases.",
        "Discuss global health equity in vaccine manufacturing: IP waivers, tech transfer hubs, and geopolitical dependencies.",
    ],
    # 31 Geografia regional
    [
        "Model planetary boundaries (Rockström) and regional safe operating spaces for water, land, and biodiversity in Brazil.",
        "Analyze logistics geography of global supply chains post-pandemic: nearshoring, chokepoints, and resilience metrics.",
        "Discuss political ecology of mining in the Global South: royalties, environmental justice, and state capture.",
        "Evaluate smart city narratives versus informal settlements: data governance, surveillance, and right to the city.",
    ],
]

BLOCK_RE = re.compile(
    r"    # Complexas \(1\)\n"
    r'(    \{"query": "(?:[^"\\]|\\.)*"\},)\n',
    re.MULTILINE,
)


def _escape_query(q: str) -> str:
    return q.replace("\\", "\\\\").replace('"', '\\"')


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    matches = list(BLOCK_RE.finditer(text))
    if len(matches) != len(EXTRA_BY_CATEGORY):
        raise SystemExit(
            f"Expected {len(EXTRA_BY_CATEGORY)} Complexas (1) blocks, found {len(matches)}"
        )

    offset = 0
    for idx, match in enumerate(matches):
        existing_line = match.group(1)
        extras = EXTRA_BY_CATEGORY[idx]
        lines = ["    # Complexas (5)", existing_line]
        for q in extras:
            lines.append(f'    {{"query": "{_escape_query(q)}"}},')
        replacement = "\n".join(lines) + "\n"

        start = match.start() + offset
        end = match.end() + offset
        text = text[:start] + replacement + text[end:]
        offset += len(replacement) - (match.end() - match.start())

    PATH.write_text(text, encoding="utf-8")

    mod = ast.parse(PATH.read_text(encoding="utf-8"))
    count = None
    for node in mod.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "QUERIES":
                    count = len(node.value.elts)

    complex_blocks = text.count("# Complexas (5)")
    print(f"Expanded blocks: {complex_blocks}")
    print(f"Added queries: {len(EXTRA_BY_CATEGORY) * 4}")
    print(f"QUERIES count: {count}")


if __name__ == "__main__":
    main()
