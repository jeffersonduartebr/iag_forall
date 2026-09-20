# Objective: Institutional anchor documents and the RAG-dependent items derived from the same source of truth.
"""The RAG partition: questions that no amount of pre-training can answer.

Documents and questions are generated from one list of facts, so the gold answer
cannot drift from the document text. A hand-written document plus hand-written
questions would eventually disagree, and a benchmark that disagrees with itself
measures nothing.

Three item kinds come out of here:

``anchored``
    the answer exists only in the institutional document;
``abstention``
    the question is about the same documents but the fact is deliberately absent,
    so the correct behaviour is to decline — this is what separates a model that
    retrieves from one that invents;
``twin``
    the same shape of question, answerable from universal technical knowledge,
    which is the control that tells a retrieval failure apart from ignorance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

INSTITUTION = "Instituto Tecnico do Vale (ITV)"


@dataclass
class Fact:
    """One institutional fact: a line in a document and a question about it."""

    key: str
    doc: str
    section: str
    statement: str
    question: str
    gold: Any
    answer_type: str = "exact"
    grader: str = "exact"
    steps: int = 1
    discipline: str = "projeto_bd"
    tolerance: Optional[float] = None


@dataclass
class Twin:
    """A parametric-knowledge counterpart: same shape, no document required."""

    key: str
    question: str
    gold: Any
    discipline: str
    answer_type: str = "exact"
    grader: str = "exact"
    steps: int = 1
    tolerance: Optional[float] = None


DOC_TITLES = {
    "reg_avaliacao": "Regulamento de Avaliacao Academica",
    "norma_bd": "Norma Interna de Padronizacao de Esquemas de Banco de Dados",
    "politica_hardware": "Politica de Aquisicao e Montagem de Equipamentos",
}


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------

FACTS: List[Fact] = [
    # -- Regulamento de avaliacao -------------------------------------------
    Fact(
        key="nota_minima",
        doc="reg_avaliacao",
        section="Art. 4",
        statement="A media final minima para aprovacao direta e 6,5 (seis virgula cinco).",
        question="Segundo o Regulamento de Avaliacao Academica do ITV, qual e a media final minima para aprovacao direta?",
        gold=6.5,
        answer_type="numeric",
        grader="numeric",
        tolerance=1e-6,
        discipline="matematica",
    ),
    Fact(
        key="frequencia_minima",
        doc="reg_avaliacao",
        section="Art. 5",
        statement="E exigida frequencia minima de 72% da carga horaria de cada componente curricular.",
        question="Qual e a frequencia minima exigida por componente curricular, segundo o regulamento do ITV?",
        gold=72.0,
        answer_type="numeric",
        grader="numeric",
        tolerance=1e-6,
        discipline="matematica",
    ),
    Fact(
        key="peso_recuperacao",
        doc="reg_avaliacao",
        section="Art. 7",
        statement=(
            "Na avaliacao de recuperacao, a nota obtida compoe a media final com peso 0,4, "
            "combinada a media do semestre com peso 0,6."
        ),
        question="No ITV, com que peso a nota da avaliacao de recuperacao entra na media final?",
        gold=0.4,
        answer_type="numeric",
        grader="numeric",
        tolerance=1e-6,
        discipline="matematica",
    ),
    Fact(
        key="prazo_revisao",
        doc="reg_avaliacao",
        section="Art. 11",
        statement="O pedido de revisao de nota deve ser protocolado em ate 5 dias uteis apos a divulgacao.",
        question="Em quantos dias uteis apos a divulgacao o aluno do ITV pode pedir revisao de nota?",
        gold=5.0,
        answer_type="numeric",
        grader="numeric",
        tolerance=1e-6,
        discipline="matematica",
    ),
    Fact(
        key="formulario_revisao",
        doc="reg_avaliacao",
        section="Art. 11",
        statement="O pedido de revisao usa o formulario ITV-AV-08, disponivel na secretaria academica.",
        question="Qual e o codigo do formulario de pedido de revisao de nota do ITV?",
        gold=["ITV-AV-08", "ITVAV08"],
        discipline="matematica",
    ),
    Fact(
        key="prazo_lancamento",
        doc="reg_avaliacao",
        section="Art. 9",
        statement="As notas devem ser lancadas no sistema academico em ate 10 dias corridos apos a aplicacao.",
        question="Qual e o prazo, em dias corridos, para lancamento de notas no sistema academico do ITV?",
        gold=10.0,
        answer_type="numeric",
        grader="numeric",
        tolerance=1e-6,
        discipline="matematica",
    ),
    Fact(
        key="media_recuperacao_minima",
        doc="reg_avaliacao",
        section="Art. 8",
        statement="Apos a recuperacao, a media final minima para aprovacao e 5,0.",
        question="Qual e a media final minima para aprovacao apos a recuperacao, segundo o regulamento do ITV?",
        gold=5.0,
        answer_type="numeric",
        grader="numeric",
        tolerance=1e-6,
        discipline="matematica",
    ),
    # -- Norma de banco de dados --------------------------------------------
    Fact(
        key="prefixo_tabela",
        doc="norma_bd",
        section="Secao 2.1",
        statement="Toda tabela transacional deve usar o prefixo tb_ e nome no singular, em minusculas.",
        question="Segundo a Norma Interna de Padronizacao de Esquemas do ITV, qual prefixo deve ser usado em tabelas transacionais?",
        gold=["tb_", "tb"],
    ),
    Fact(
        key="tipo_chave",
        doc="norma_bd",
        section="Secao 2.3",
        statement="Chaves primarias substitutas devem ser do tipo BIGINT sem sinal, com incremento automatico.",
        question="Qual tipo de dado a norma do ITV exige para chaves primarias substitutas?",
        gold=["BIGINT", "BIG INT"],
    ),
    Fact(
        key="campos_auditoria",
        doc="norma_bd",
        section="Secao 3.2",
        statement=(
            "Toda tabela deve conter os campos de auditoria criado_em, atualizado_em e criado_por, "
            "sendo os dois primeiros obrigatoriamente NOT NULL."
        ),
        question="Quantos campos de auditoria a norma de esquemas do ITV torna obrigatorios em toda tabela?",
        gold=3.0,
        answer_type="numeric",
        grader="numeric",
        tolerance=1e-6,
    ),
    Fact(
        key="retencao_log",
        doc="norma_bd",
        section="Secao 5.1",
        statement="Os registros da tabela de log de auditoria devem ser retidos por 18 meses antes do expurgo.",
        question="Por quantos meses o ITV exige a retencao dos registros de log de auditoria?",
        gold=18.0,
        answer_type="numeric",
        grader="numeric",
        tolerance=1e-6,
    ),
    Fact(
        key="forma_normal_minima",
        doc="norma_bd",
        section="Secao 2.5",
        statement=(
            "Esquemas novos devem ser entregues no minimo na 3FN; desnormalizacoes exigem "
            "justificativa formal de desempenho aprovada pelo comite de dados."
        ),
        question="Qual e a forma normal minima exigida pela norma do ITV para esquemas novos?",
        gold=["3FN", "3NF", "terceira forma normal"],
    ),
    Fact(
        key="limite_indices",
        doc="norma_bd",
        section="Secao 4.2",
        statement="Cada tabela pode ter no maximo 6 indices secundarios sem aprovacao do comite de dados.",
        question="Quantos indices secundarios o ITV permite por tabela sem aprovacao do comite de dados?",
        gold=6.0,
        answer_type="numeric",
        grader="numeric",
        tolerance=1e-6,
    ),
    Fact(
        key="nomenclatura_fk",
        doc="norma_bd",
        section="Secao 2.4",
        statement="Chaves estrangeiras devem ser nomeadas no padrao fk_<tabela_origem>_<tabela_destino>.",
        question="Qual e o padrao de nomenclatura de chaves estrangeiras definido pela norma do ITV?",
        gold=["fk_<tabela_origem>_<tabela_destino>", "fk_"],
    ),
    Fact(
        key="janela_manutencao",
        doc="norma_bd",
        section="Secao 6.1",
        statement="A janela de manutencao programada dos bancos de producao ocorre aos domingos, das 2h as 5h.",
        question="Em que dia da semana ocorre a janela de manutencao programada dos bancos de producao do ITV?",
        gold=["domingo", "domingos"],
    ),
    # -- Politica de hardware ------------------------------------------------
    Fact(
        key="potencia_bancada",
        doc="politica_hardware",
        section="Art. 3",
        statement="Cada bancada de laboratorio admite consumo eletrico total de ate 850 W.",
        question="Qual e o consumo eletrico maximo, em watts, admitido por bancada de laboratorio no ITV?",
        gold=850.0,
        answer_type="numeric",
        grader="numeric",
        tolerance=1e-6,
        discipline="organizacao_computadores",
    ),
    Fact(
        key="certificacao_fonte",
        doc="politica_hardware",
        section="Art. 4",
        statement="As fontes adquiridas devem possuir certificacao minima 80 Plus Bronze.",
        question="Qual e a certificacao minima exigida pelo ITV para fontes de alimentacao adquiridas?",
        gold=["80 Plus Bronze", "80+ Bronze", "Bronze"],
        discipline="organizacao_computadores",
    ),
    Fact(
        key="tdp_maximo",
        doc="politica_hardware",
        section="Art. 5",
        statement="Processadores adquiridos para laboratorios de ensino nao podem exceder TDP de 125 W.",
        question="Qual e o TDP maximo, em watts, permitido para processadores de laboratorio de ensino no ITV?",
        gold=125.0,
        answer_type="numeric",
        grader="numeric",
        tolerance=1e-6,
        discipline="organizacao_computadores",
    ),
    Fact(
        key="garantia_minima",
        doc="politica_hardware",
        section="Art. 7",
        statement="Equipamentos devem ser adquiridos com garantia on-site minima de 36 meses.",
        question="Qual e a garantia on-site minima, em meses, exigida pelo ITV na aquisicao de equipamentos?",
        gold=36.0,
        answer_type="numeric",
        grader="numeric",
        tolerance=1e-6,
        discipline="organizacao_computadores",
    ),
    Fact(
        key="ciclo_substituicao",
        doc="politica_hardware",
        section="Art. 8",
        statement="O ciclo de substituicao do parque de maquinas de ensino e de 5 anos.",
        question="De quantos anos e o ciclo de substituicao do parque de maquinas de ensino do ITV?",
        gold=5.0,
        answer_type="numeric",
        grader="numeric",
        tolerance=1e-6,
        discipline="organizacao_computadores",
    ),
    Fact(
        key="memoria_minima",
        doc="politica_hardware",
        section="Art. 6",
        statement="A configuracao minima homologada para estacoes de ensino e de 16 GB de memoria RAM.",
        question="Qual e a memoria RAM minima homologada pelo ITV para estacoes de ensino, em GB?",
        gold=16.0,
        answer_type="numeric",
        grader="numeric",
        tolerance=1e-6,
        discipline="organizacao_computadores",
    ),
    Fact(
        key="codigo_descarte",
        doc="politica_hardware",
        section="Art. 12",
        statement="O descarte de componentes segue o procedimento ITV-PO-23, com registro no inventario.",
        question="Qual e o codigo do procedimento de descarte de componentes do ITV?",
        gold=["ITV-PO-23", "ITVPO23"],
        discipline="organizacao_computadores",
    ),
    Fact(
        key="prazo_chamado",
        doc="politica_hardware",
        section="Art. 10",
        statement="Chamados de manutencao corretiva devem ser atendidos em ate 48 horas uteis.",
        question="Em quantas horas uteis o ITV exige o atendimento de chamados de manutencao corretiva?",
        gold=48.0,
        answer_type="numeric",
        grader="numeric",
        tolerance=1e-6,
        discipline="organizacao_computadores",
    ),
]


#: Questions about the same documents whose answer is deliberately absent from them.
ABSENT_QUESTIONS: List[Dict[str, Any]] = [
    {
        "key": "ausente_segunda_chamada",
        "doc": "reg_avaliacao",
        "question": "Segundo o Regulamento de Avaliacao Academica do ITV, qual e o prazo para solicitar prova de segunda chamada?",
        "discipline": "matematica",
    },
    {
        "key": "ausente_peso_seminario",
        "doc": "reg_avaliacao",
        "question": "Qual peso o regulamento do ITV atribui a apresentacao de seminarios na composicao da media?",
        "discipline": "matematica",
    },
    {
        "key": "ausente_colunas_max",
        "doc": "norma_bd",
        "question": "Qual e o numero maximo de colunas por tabela permitido pela norma de esquemas do ITV?",
        "discipline": "projeto_bd",
    },
    {
        "key": "ausente_sgbd_homologado",
        "doc": "norma_bd",
        "question": "Qual versao de SGBD a norma do ITV homologa para ambientes de producao?",
        "discipline": "projeto_bd",
    },
    {
        "key": "ausente_marca_gpu",
        "doc": "politica_hardware",
        "question": "Qual marca de placa de video a politica de aquisicao do ITV determina como padrao?",
        "discipline": "organizacao_computadores",
    },
    {
        "key": "ausente_temperatura",
        "doc": "politica_hardware",
        "question": "Qual e a temperatura maxima de operacao admitida pela politica do ITV para os laboratorios?",
        "discipline": "organizacao_computadores",
    },
]


#: Parametric controls: same question shape, universal answer, no document.
TWINS: List[Twin] = [
    Twin("twin_truncate", "No padrao SQL, qual comando remove todas as linhas de uma tabela preservando a sua estrutura? Responda apenas com o comando.", ["TRUNCATE"], "projeto_bd"),
    Twin("twin_acid", "Na sigla ACID, o que significa a letra D? Responda apenas com a palavra.", ["durabilidade", "durability"], "projeto_bd"),
    Twin("twin_pk_null", "No padrao SQL, uma coluna declarada como chave primaria pode conter valores NULL? Responda apenas SIM ou NAO.", ["NAO", "nao"], "projeto_bd"),
    Twin("twin_join_default", "Em SQL, qual tipo de juncao e aplicado quando se escreve apenas JOIN entre duas tabelas? Responda apenas com o nome.", ["INNER JOIN", "INNER", "interna"], "projeto_bd"),
    Twin("twin_3fn", "Qual forma normal elimina dependencias transitivas entre atributos nao primos? Responda apenas com o nome.", ["3FN", "3NF", "terceira forma normal"], "projeto_bd"),
    Twin("twin_having", "Em SQL, qual clausula filtra grupos apos a agregacao? Responda apenas com a palavra-chave.", ["HAVING"], "projeto_bd"),
    Twin("twin_indice_btree", "Qual estrutura de dados e usada pela maioria dos indices de proposito geral em SGBDs relacionais? Responda apenas com o nome.", ["B-tree", "arvore B", "B+tree", "B+ tree", "arvore B+"], "projeto_bd"),
    Twin("twin_ddl", "A que subconjunto da linguagem SQL pertence o comando CREATE TABLE? Responda apenas com a sigla.", ["DDL"], "projeto_bd"),
    Twin("twin_normal_1fn", "Qual forma normal exige que todos os atributos sejam atomicos? Responda apenas com o nome.", ["1FN", "1NF", "primeira forma normal"], "projeto_bd"),
    Twin("twin_rollback", "Qual comando SQL desfaz as alteracoes de uma transacao ainda nao confirmada? Responda apenas com o comando.", ["ROLLBACK"], "projeto_bd"),
    Twin("twin_bit_byte", "Quantos bits formam um byte? Responda apenas com o numero.", 8.0, "organizacao_computadores", "numeric", "numeric", 1, 1e-6),
    Twin("twin_von_neumann", "Na arquitetura de von Neumann, instrucoes e dados compartilham a mesma memoria? Responda apenas SIM ou NAO.", ["SIM", "sim"], "organizacao_computadores"),
    Twin("twin_ula", "Qual unidade do processador executa as operacoes aritmeticas e logicas? Responda apenas com o nome ou a sigla.", ["ULA", "ALU", "unidade logica e aritmetica"], "organizacao_computadores"),
    Twin("twin_volatil", "A memoria RAM e volatil ou nao volatil? Responda apenas com uma palavra.", ["volatil"], "organizacao_computadores"),
    Twin("twin_cache_nivel", "Qual nivel de cache e o mais proximo do nucleo do processador? Responda apenas com a sigla.", ["L1"], "organizacao_computadores"),
    Twin("twin_barramento", "Qual barramento transporta os dados entre processador e memoria: de dados, de enderecos ou de controle? Responda apenas com o nome.", ["de dados", "dados"], "organizacao_computadores"),
    Twin("twin_risc", "O que significa a letra R na sigla RISC? Responda apenas com a palavra em ingles.", ["reduced"], "organizacao_computadores"),
    Twin("twin_ssd_movel", "Um SSD possui partes moveis? Responda apenas SIM ou NAO.", ["NAO", "nao"], "organizacao_computadores"),
    Twin("twin_hex_ff", "Qual e o valor decimal do numero hexadecimal FF? Responda apenas com o numero.", 255.0, "organizacao_computadores", "numeric", "numeric", 1, 1e-6),
    Twin("twin_paridade", "Quantos discos de paridade um arranjo RAID 5 utiliza? Responda apenas com o numero.", 1.0, "organizacao_computadores", "numeric", "numeric", 1, 1e-6),
    Twin("twin_pi", "Qual e o valor de pi arredondado a duas casas decimais? Responda apenas com o numero.", 3.14, "matematica", "numeric", "numeric", 1, 1e-3),
    Twin("twin_primo", "O numero 1 e considerado primo? Responda apenas SIM ou NAO.", ["NAO", "nao"], "matematica"),
    Twin("twin_soma_angulos", "Qual e a soma dos angulos internos de um triangulo, em graus? Responda apenas com o numero.", 180.0, "matematica", "numeric", "numeric", 1, 1e-6),
    Twin("twin_raiz_quadrada", "Qual e a raiz quadrada de 144? Responda apenas com o numero.", 12.0, "matematica", "numeric", "numeric", 1, 1e-6),
    Twin("twin_fatorial", "Qual e o valor de 5!? Responda apenas com o numero.", 120.0, "matematica", "numeric", "numeric", 1, 1e-6),
    Twin("twin_log", "Qual e o valor de log na base 2 de 64? Responda apenas com o numero.", 6.0, "matematica", "numeric", "numeric", 1, 1e-6),
    Twin("twin_probabilidade_max", "Qual e o maior valor que uma probabilidade pode assumir? Responda apenas com o numero.", 1.0, "matematica", "numeric", "numeric", 1, 1e-6),
    Twin("twin_mediana", "Qual medida estatistica divide um conjunto ordenado em duas metades iguais? Responda apenas com o nome.", ["mediana"], "matematica"),
    Twin("twin_derivada_constante", "Qual e a derivada de uma funcao constante? Responda apenas com o numero.", 0.0, "matematica", "numeric", "numeric", 1, 1e-6),
    Twin("twin_pitagoras", "Em um triangulo retangulo de catetos 3 e 4, qual e o comprimento da hipotenusa? Responda apenas com o numero.", 5.0, "matematica", "numeric", "numeric", 1, 1e-6),
]


# ---------------------------------------------------------------------------
# Document rendering
# ---------------------------------------------------------------------------


def render_documents() -> Dict[str, str]:
    """Build the markdown of each anchor document from the facts themselves."""
    grouped: Dict[str, List[Fact]] = {}
    for fact in FACTS:
        grouped.setdefault(fact.doc, []).append(fact)

    documents: Dict[str, str] = {}
    for doc_id, facts in grouped.items():
        lines = [
            f"# {DOC_TITLES[doc_id]}",
            "",
            f"{INSTITUTION} - documento interno. Versao 1.0.",
            "",
            "Este documento e ficticio e existe apenas para avaliar a dependencia de recuperacao "
            "de contexto (RAG). Os valores aqui definidos nao correspondem a nenhuma instituicao real.",
            "",
        ]
        for fact in facts:
            lines.append(f"## {fact.section}")
            lines.append("")
            lines.append(fact.statement)
            lines.append("")
        documents[doc_id] = "\n".join(lines)
    return documents
