# Objective: Parametric generators of database design items with algorithmically computed gold answers.
"""Database design generators.

Normalization items are usually ungradable because they are asked as "explain".
Asked instead as "which is the highest normal form of R given these
dependencies", they have one decidable answer, computed here by
:mod:`fd_theory`. SQL items are graded by running the candidate query against
:mod:`sql_fixture` and comparing result sets, so any correct formulation counts.
"""

from __future__ import annotations

from typing import Any, List

from .basegen import BaseSpec, generator
from .fd_theory import FD, candidate_keys, closure, format_fds, highest_normal_form, is_lossless
from .sql_fixture import SCHEMA_DESCRIPTION

DISC = "projeto_bd"

TOL_EXACT = 1e-6

SQL_PREAMBLE = (
    "Considere o esquema relacional abaixo:\n\n"
    f"{SCHEMA_DESCRIPTION}\n\n"
)


def _fd(left: str, right: str) -> FD:
    """Build a functional dependency from two attribute strings."""
    return frozenset(left), frozenset(right)


# ---------------------------------------------------------------------------
# One logical step
# ---------------------------------------------------------------------------


@generator(DISC)
def cross_join_cardinality(rng: Any) -> BaseSpec:
    """Rows produced by a cartesian product: one multiplication."""
    left = rng.choice([7, 12, 18, 25, 40])
    right = rng.choice([3, 6, 9, 15, 22])
    return BaseSpec(
        topic="produto-cartesiano",
        query=(
            f"Uma tabela A tem {left} linhas e uma tabela B tem {right} linhas. Quantas linhas "
            "retorna a consulta SELECT * FROM A CROSS JOIN B? Responda apenas com o numero."
        ),
        answer_type="numeric",
        grader="numeric",
        gold=float(left * right),
        steps=1,
        tolerance=TOL_EXACT,
        tags=["algebra-relacional"],
    )


@generator(DISC)
def count_with_nulls(rng: Any) -> BaseSpec:
    """COUNT(coluna) versus COUNT(*): one distinction, one subtraction."""
    rows = rng.choice([40, 60, 85, 120, 200])
    nulls = rng.choice([3, 7, 12, 19, 25])
    return BaseSpec(
        topic="semantica-de-null",
        query=(
            f"Uma tabela tem {rows} linhas, das quais {nulls} possuem a coluna nota igual a NULL. "
            "Quantas linhas retorna SELECT COUNT(nota) FROM tabela? Responda apenas com o numero."
        ),
        answer_type="numeric",
        grader="numeric",
        gold=float(rows - nulls),
        steps=1,
        tolerance=TOL_EXACT,
        tags=["sql", "null"],
    )


@generator(DISC)
def violation_first_normal_form(rng: Any) -> BaseSpec:
    """Name the normal form violated by a described anomaly."""
    scenario, gold = rng.choice(
        [
            (
                "uma coluna telefones que armazena varios numeros separados por virgula",
                ["1FN", "primeira forma normal", "1NF"],
            ),
            (
                "um atributo nao primo que depende apenas de parte de uma chave primaria composta",
                ["2FN", "segunda forma normal", "2NF"],
            ),
            (
                "um atributo nao primo que depende de outro atributo nao primo",
                ["3FN", "terceira forma normal", "3NF"],
            ),
        ]
    )
    return BaseSpec(
        topic="violacao-de-forma-normal",
        query=(
            f"Uma tabela apresenta {scenario}. Qual forma normal e violada? "
            "Responda apenas com o nome da forma normal."
        ),
        answer_type="exact",
        grader="exact",
        gold=gold,
        steps=1,
        tags=["normalizacao"],
    )


@generator(DISC)
def simple_sql_count(rng: Any) -> BaseSpec:
    """A single-table aggregate with one filter."""
    year = rng.choice([2022, 2023, 2024])
    return BaseSpec(
        topic="sql-agregacao-simples",
        query=(
            SQL_PREAMBLE
            + f"Escreva uma consulta SQL que retorne a quantidade de alunos que ingressaram em {year}."
        ),
        answer_type="sql",
        grader="sql",
        gold=f"SELECT COUNT(*) FROM aluno WHERE ano_ingresso = {year}",
        steps=1,
        tags=["sql"],
    )


@generator(DISC)
def trivial_closure(rng: Any) -> BaseSpec:
    """Closure of a single attribute under one dependency: one application."""
    left, right = rng.choice([("A", "B"), ("C", "D"), ("X", "Y"), ("P", "Q")])
    return BaseSpec(
        topic="fecho-trivial",
        query=(
            f"Em uma relacao com os atributos {left} e {right} e a unica dependencia funcional "
            f"{left} -> {right}, qual e o fecho de {{{left}}}? "
            "Responda apenas com o conjunto de atributos."
        ),
        answer_type="set",
        grader="set",
        gold=sorted([left, right]),
        steps=1,
        tags=["dependencias-funcionais"],
    )


# ---------------------------------------------------------------------------
# Multi-step
# ---------------------------------------------------------------------------

#: Dependency sets drawn for the normalization items, each over A..E.
_FD_SETS: List[List[FD]] = [
    [_fd("A", "BC"), _fd("B", "D"), _fd("D", "E")],
    [_fd("AB", "C"), _fd("C", "D"), _fd("D", "B")],
    [_fd("AB", "CD"), _fd("A", "E"), _fd("E", "C")],
    [_fd("A", "B"), _fd("B", "C"), _fd("C", "A")],
    [_fd("AB", "C"), _fd("A", "D"), _fd("D", "E")],
    [_fd("A", "BCDE")],
    [_fd("AB", "CDE"), _fd("C", "D")],
    [_fd("A", "B"), _fd("BC", "D"), _fd("D", "AE")],
]


@generator(DISC)
def attribute_closure(rng: Any) -> BaseSpec:
    """Closure of a set of attributes: repeated application until a fixed point."""
    fds = rng.choice(_FD_SETS)
    all_attrs = frozenset().union(*[left | right for left, right in fds])
    start = frozenset(rng.sample(sorted(all_attrs), rng.choice([1, 2])))
    gold = sorted(closure(start, fds))
    start_text = "".join(sorted(start))
    return BaseSpec(
        topic="fecho-de-atributos",
        query=(
            f"Considere a relacao R({', '.join(sorted(all_attrs))}) com as dependencias funcionais: "
            f"{format_fds(fds)}. Qual e o fecho de {{{', '.join(sorted(start))}}}? "
            "Responda apenas com o conjunto de atributos."
        ),
        query_dense=f"R({''.join(sorted(all_attrs))}), F = {{{format_fds(fds)}}}. {start_text}+ = ?",
        steps_dense=3,
        answer_type="set",
        grader="set",
        gold=gold,
        steps=3,
        tags=["dependencias-funcionais", "normalizacao"],
    )


@generator(DISC)
def single_candidate_key(rng: Any) -> BaseSpec:
    """Find the candidate key, drawn only from schemas where it is unique."""
    options = []
    for fds in _FD_SETS:
        all_attrs = frozenset().union(*[left | right for left, right in fds])
        keys = candidate_keys(all_attrs, fds)
        if len(keys) == 1:
            options.append((fds, all_attrs, keys[0]))
    fds, all_attrs, key = rng.choice(options)
    return BaseSpec(
        topic="chave-candidata",
        query=(
            f"Considere a relacao R({', '.join(sorted(all_attrs))}) com as dependencias funcionais: "
            f"{format_fds(fds)}. Qual e a unica chave candidata de R? "
            "Responda apenas com o conjunto de atributos."
        ),
        query_dense=f"R({''.join(sorted(all_attrs))}), F = {{{format_fds(fds)}}}. Chave candidata?",
        steps_dense=4,
        answer_type="set",
        grader="set",
        gold=sorted(key),
        steps=4,
        tags=["dependencias-funcionais", "chaves"],
    )


@generator(DISC)
def normal_form_diagnosis(rng: Any) -> BaseSpec:
    """Determine the highest normal form the schema already satisfies."""
    fds = rng.choice(_FD_SETS)
    all_attrs = frozenset().union(*[left | right for left, right in fds])
    form = highest_normal_form(all_attrs, fds)
    accepted = {
        "BCNF": ["BCNF", "FNBC", "Boyce-Codd"],
        "3NF": ["3FN", "3NF", "terceira forma normal"],
        "2NF": ["2FN", "2NF", "segunda forma normal"],
        "1NF": ["1FN", "1NF", "primeira forma normal"],
    }[form]
    return BaseSpec(
        topic="diagnostico-de-forma-normal",
        query=(
            f"Considere a relacao R({', '.join(sorted(all_attrs))}), com atributos atomicos, "
            f"e as dependencias funcionais: {format_fds(fds)}. Qual e a forma normal mais alta "
            "que R satisfaz? Responda apenas com o nome da forma normal."
        ),
        query_dense=f"R({''.join(sorted(all_attrs))}), F = {{{format_fds(fds)}}}. Forma normal mais alta?",
        steps_dense=4,
        answer_type="exact",
        grader="exact",
        gold=accepted,
        steps=4,
        tags=["normalizacao"],
    )


@generator(DISC)
def lossless_decomposition(rng: Any) -> BaseSpec:
    """Decide whether a binary decomposition preserves the join."""
    fds = rng.choice(_FD_SETS)
    all_attrs = frozenset().union(*[left | right for left, right in fds])
    ordered = sorted(all_attrs)
    cut = rng.randint(1, len(ordered) - 1)
    left = frozenset(ordered[: cut + 1])
    right = frozenset(ordered[cut:])
    verdict = is_lossless(all_attrs, left, right, fds)
    gold = ["SIM", "sem perda", "lossless"] if verdict else ["NAO", "com perda", "lossy"]
    return BaseSpec(
        topic="decomposicao-sem-perdas",
        query=(
            f"Considere R({', '.join(ordered)}) com as dependencias funcionais: {format_fds(fds)}. "
            f"A decomposicao em R1({', '.join(sorted(left))}) e R2({', '.join(sorted(right))}) "
            "e uma decomposicao sem perdas? Responda apenas SIM ou NAO."
        ),
        query_dense=(
            f"R({''.join(ordered)}), F = {{{format_fds(fds)}}}. "
            f"R1={''.join(sorted(left))}, R2={''.join(sorted(right))} preserva juncao? SIM/NAO"
        ),
        steps_dense=3,
        answer_type="exact",
        grader="exact",
        gold=gold,
        steps=3,
        tags=["normalizacao", "decomposicao"],
    )


#: Gold queries that exercise the traps built into the fixture (NULLs, empty sides).
_SQL_TASKS = [
    (
        "sql-left-join",
        "Escreva uma consulta SQL que retorne o nome de todos os alunos que nao possuem "
        "nenhuma matricula registrada.",
        "SELECT a.nome FROM aluno a LEFT JOIN matricula m ON m.aluno_id = a.id "
        "WHERE m.aluno_id IS NULL",
        3,
    ),
    (
        "sql-disciplina-vazia",
        "Escreva uma consulta SQL que retorne o codigo das disciplinas em que nenhum aluno "
        "se matriculou.",
        "SELECT d.codigo FROM disciplina d LEFT JOIN matricula m ON m.disciplina_id = d.id "
        "WHERE m.disciplina_id IS NULL",
        3,
    ),
    (
        "sql-media-having",
        "Escreva uma consulta SQL que retorne o codigo de cada disciplina e a media das notas "
        "lancadas, considerando apenas disciplinas cuja media seja maior ou igual a 7.",
        "SELECT d.codigo, AVG(m.nota) FROM disciplina d JOIN matricula m ON m.disciplina_id = d.id "
        "WHERE m.nota IS NOT NULL GROUP BY d.codigo HAVING AVG(m.nota) >= 7",
        4,
    ),
    (
        "sql-sem-orientador",
        "Escreva uma consulta SQL que retorne o nome dos alunos que nao possuem orientador "
        "atribuido.",
        "SELECT nome FROM aluno WHERE orientador_id IS NULL",
        2,
    ),
    (
        "sql-departamento-agregado",
        "Escreva uma consulta SQL que retorne o nome de cada departamento e a quantidade de "
        "disciplinas que ele oferece, incluindo departamentos sem disciplinas.",
        "SELECT dep.nome, COUNT(d.id) FROM departamento dep "
        "LEFT JOIN disciplina d ON d.departamento_id = dep.id GROUP BY dep.nome",
        4,
    ),
    (
        "sql-top-aluno",
        "Escreva uma consulta SQL que retorne o nome do aluno com a maior media de notas, "
        "desconsiderando matriculas sem nota lancada.",
        "SELECT a.nome FROM aluno a JOIN matricula m ON m.aluno_id = a.id WHERE m.nota IS NOT NULL "
        "GROUP BY a.id, a.nome ORDER BY AVG(m.nota) DESC LIMIT 1",
        4,
    ),
]


@generator(DISC)
def sql_query_task(rng: Any) -> BaseSpec:
    """A SQL task graded by executing the candidate query against the fixture."""
    topic, task, gold_sql, steps = rng.choice(_SQL_TASKS)
    return BaseSpec(
        topic=topic,
        query=SQL_PREAMBLE + task,
        query_dense=f"{SCHEMA_DESCRIPTION}\n\n{task.split('Escreva uma consulta SQL que retorne ')[-1]}",
        steps_dense=max(steps, 3),
        answer_type="sql",
        grader="sql",
        gold=gold_sql,
        steps=steps,
        tags=["sql"],
    )


@generator(DISC)
def conceptual_modeling(rng: Any) -> BaseSpec:
    """Open-ended modelling item, graded by rubric — the only kind that needs a judge."""
    domain, entities = rng.choice(
        [
            ("uma biblioteca universitaria com emprestimos e reservas", "acervo, exemplar, usuario, emprestimo"),
            ("uma clinica com agendamentos e prontuarios", "paciente, profissional, agendamento, atendimento"),
            ("um laboratorio de informatica com reserva de maquinas", "maquina, sala, reserva, responsavel"),
            ("uma escola tecnica com turmas e avaliacoes", "turma, disciplina, avaliacao, matricula"),
        ]
    )
    return BaseSpec(
        topic="modelagem-conceitual",
        query=(
            f"Projete o modelo conceitual (entidades, atributos e relacionamentos com cardinalidades) "
            f"para {domain}. Justifique as chaves primarias escolhidas e indique um ponto do modelo "
            "em que a normalizacao ate a 3FN altera o desenho."
        ),
        answer_type="rubric",
        grader="rubric",
        gold=None,
        steps=4,
        rubric=[
            f"Identifica as entidades centrais do dominio ({entities})",
            "Atribui cardinalidades coerentes a cada relacionamento",
            "Justifica cada chave primaria em termos de identificacao unica",
            "Resolve ao menos um relacionamento N:N com entidade associativa",
            "Aponta uma dependencia transitiva concreta e o efeito da 3FN sobre ela",
        ],
        tags=["modelagem", "normalizacao"],
    )
