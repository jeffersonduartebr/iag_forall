# Objective: Deterministic transformations that derive verbosity traps and semantic outliers from canonical items.
"""Transformations that change the *form* of a question and never its answer.

That invariant is the whole point of the paired design: because the gold value
is shared with the canonical item, any accuracy gap between a base item and its
variant is caused by the wording alone. Two rules keep the invariant true:

* no transformation ever touches a token containing a digit, so no given value
  and no unit can be corrupted;
* the padding is generated from topic-neutral narrative fragments that state no
  constraint, so nothing in it can be mistaken for a given.

Every transformation is seeded, so rebuilding the corpus reproduces it exactly.
"""

from __future__ import annotations

import random
import re
from typing import Callable, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Verbosity padding
# ---------------------------------------------------------------------------

#: Institutional narrative that sets a scene and constrains nothing.
_PREAMBLES: Dict[str, List[str]] = {
    "matematica": [
        "A coordenacao do curso reuniu-se na ultima quinta-feira para revisar o plano de ensino do "
        "semestre. Depois de uma discussao longa sobre a ordem dos topicos, ficou decidido que a "
        "lista de exercicios seria distribuida em formato impresso, ainda que boa parte da turma "
        "prefira o material digital. O professor responsavel comentou que, em turmas anteriores, "
        "os alunos costumavam resolver os exercicios em duplas, o que gerava discussoes produtivas "
        "mas atrasava a correcao. Houve tambem um debate sobre o horario de atendimento, que "
        "permanecera as tercas a tarde, na sala do bloco central, salvo em semanas de prova.",
        "Durante a semana academica, um grupo de estudantes organizou uma oficina de resolucao de "
        "problemas. A oficina aconteceu no auditorio menor, que tem capacidade reduzida e por isso "
        "exigiu inscricao previa. Os organizadores relataram que a procura superou a expectativa e "
        "que pretendem repetir a atividade no proximo semestre, talvez com transmissao online. Um "
        "dos monitores observou que a maior dificuldade dos participantes nao estava nas contas, "
        "mas na leitura atenta do enunciado, tema que rendeu uma mesa-redonda no dia seguinte.",
    ],
    "organizacao_computadores": [
        "O laboratorio de informatica passou por uma reforma recente que incluiu a troca do piso "
        "elevado e a revisao completa da climatizacao. A equipe tecnica aproveitou a parada para "
        "reorganizar o cabeamento, que estava identificado de forma inconsistente desde a montagem "
        "original. Segundo o responsavel, a maior parte do tempo foi gasta com documentacao, e nao "
        "com a parte fisica. Ficou combinado que cada bancada tera uma etiqueta com o numero do "
        "ponto de rede, e que o inventario sera revisado no inicio de cada semestre letivo.",
        "A instituicao publicou um edital para renovacao do parque de maquinas. O processo previa "
        "visita tecnica, apresentacao de amostras e um periodo de testes em ambiente controlado. "
        "Durante a fase de testes, a equipe discutiu preferencias de gabinete, ruido dos "
        "ventiladores e facilidade de manutencao, pontos que costumam ser ignorados na comparacao "
        "puramente tecnica. O relatorio final recomendou padronizar os modelos por laboratorio, "
        "para simplificar o estoque de pecas de reposicao ao longo dos anos seguintes.",
    ],
    "projeto_bd": [
        "A equipe de sistemas realizou uma reuniao de alinhamento com a secretaria academica para "
        "entender o fluxo de matricula. A conversa durou mais do que o previsto porque cada setor "
        "descrevia o mesmo processo com um vocabulario diferente. Ficou evidente que parte dos "
        "problemas relatados vinha de planilhas paralelas mantidas por cada coordenacao, e nao do "
        "sistema em si. Ao final, decidiu-se documentar o glossario antes de qualquer alteracao, e "
        "marcar uma segunda reuniao para validar os termos com os usuarios que operam o balcao.",
        "Durante a migracao do servidor, a equipe manteve o ambiente antigo em modo somente leitura "
        "por duas semanas, como precaucao. O plano de retorno foi escrito e testado, embora nunca "
        "tenha sido necessario aciona-lo. Os relatorios gerenciais passaram a ser emitidos pela "
        "manha, o que reduziu a concorrencia com o uso interativo. A equipe observou que a maior "
        "queixa dos usuarios nao era o tempo de resposta, mas a falta de aviso previo sobre janelas "
        "de manutencao, ponto incorporado ao procedimento operacional padrao.",
    ],
}

#: Closing noise. Says nothing that constrains the answer.
_EPILOGUES = [
    "Vale lembrar que o calendario academico pode sofrer ajustes e que eventuais mudancas serao "
    "comunicadas pelos canais oficiais da instituicao.",
    "Observacao administrativa: este material foi revisado pela coordenacao e substitui as versoes "
    "anteriores distribuidas em sala.",
    "Registre-se ainda que a bibliografia complementar permanece disponivel na biblioteca, com "
    "exemplares para consulta local.",
]


#: The padding must dominate the statement, not merely precede it. Below roughly
#: five times the question's own length the "trap" is just a preamble, and a
#: model that reads carelessly still lands on the answer by accident.
PAD_RATIO = 4.2
PAD_MIN_WORDS = 150
PAD_MAX_WORDS = 400


def pad_with_distractors(query: str, discipline: str, rng: random.Random) -> Tuple[str, int]:
    """Bury a one-step question in irrelevant narrative.

    Paragraphs are added until the noise is several times the size of the
    question itself, so the trap scales with the statement instead of being a
    fixed preamble that a long question would outweigh.

    Returns the padded statement and the number of distractor tokens added, which
    the corpus records so the trap's strength can be correlated with accuracy.
    """
    own = _PREAMBLES.get(discipline, _PREAMBLES["matematica"])
    others = [p for key, group in sorted(_PREAMBLES.items()) if key != discipline for p in group]
    pool = list(own)
    rng.shuffle(pool)
    extra = list(others)
    rng.shuffle(extra)
    pool += extra

    target = max(PAD_MIN_WORDS, int(len(query.split()) * PAD_RATIO))
    chosen: List[str] = []
    added = 0
    for paragraph in pool:
        if added >= target or added >= PAD_MAX_WORDS:
            break
        chosen.append(paragraph)
        added += len(paragraph.split())

    epilogue = rng.choice(_EPILOGUES)
    added += len(epilogue.split())
    body = "\n\n".join(chosen)
    padded = f"{body}\n\nFeita essa contextualizacao, resolva o seguinte: {query}\n\n{epilogue}"
    return padded, added


# ---------------------------------------------------------------------------
# Semantic perturbation
# ---------------------------------------------------------------------------

_HAS_DIGIT = re.compile(r"\d")

_ACCENTS = str.maketrans("áàâãéêíóôõúçÁÀÂÃÉÊÍÓÔÕÚÇ", "aaaaeeioooucAAAAEEIOOOUC")

_SLANG_OPENERS = [
    "eai professor, seguinte:",
    "fala, preciso de uma ajuda rapida aqui ó:",
    "mano, travei nessa questao, olha so:",
    "bom dia! desculpa incomodar mas to com uma duvida besta:",
]

_SLANG_CLOSERS = [
    "vlw pela ajuda!!",
    "me explica rapidinho pfvr",
    "so o resultado ta bom, obrigado",
    "responde ai qnd puder",
]

_EN_CONNECTIVES = {
    "Qual e": "What is",
    "Quanto e": "How much is",
    "Quantos": "How many",
    "Quantas": "How many",
    "Calcule": "Compute",
    "Resolva": "Solve",
    "Considere": "Consider",
    "Escreva": "Write",
    "Responda apenas com o numero": "Answer with the number only",
    "Responda apenas com o valor numerico": "Answer with the numeric value only",
    "Responda apenas com o conjunto de atributos": "Answer with the attribute set only",
    "arredondado a duas casas decimais": "rounded to two decimal places",
    "arredondado a quatro casas decimais": "rounded to four decimal places",
}


def _typo_word(word: str, rng: random.Random) -> str:
    """Introduce one plausible typing error into an alphabetic word."""
    if len(word) < 5 or _HAS_DIGIT.search(word):
        return word
    kind = rng.choice(("swap", "drop", "double"))
    pos = rng.randrange(1, len(word) - 1)
    if kind == "swap":
        return word[:pos] + word[pos + 1] + word[pos] + word[pos + 2 :]
    if kind == "drop":
        return word[:pos] + word[pos + 1 :]
    return word[:pos] + word[pos] + word[pos:]


def _apply_typos(query: str, rng: random.Random) -> str:
    """Corrupt roughly one word in six, never touching tokens with digits."""
    words = query.split(" ")
    return " ".join(_typo_word(w, rng) if rng.random() < 0.17 else w for w in words)


def _apply_slang(query: str, rng: random.Random) -> str:
    """Rewrite as a hurried chat message: no accents, no capitals, colloquial framing."""
    body = query.translate(_ACCENTS).lower()
    return f"{rng.choice(_SLANG_OPENERS)} {body} {rng.choice(_SLANG_CLOSERS)}"


def _apply_code_switch(query: str, rng: random.Random) -> str:
    """Switch the scaffolding to English and keep the technical content in Portuguese."""
    out = query
    for pt, en in _EN_CONNECTIVES.items():
        out = out.replace(pt, en)
    return f"Hey, quick question (answer in Portuguese, please): {out}"


def _apply_reorder(query: str, rng: random.Random) -> str:
    """Put the question first and the givens after it, inverting the usual order."""
    sentences = [s.strip() for s in re.split(r"(?<=[.?])\s+", query) if s.strip()]
    if len(sentences) < 2:
        return f"Responda ao final. {query}"
    question = next((s for s in reversed(sentences) if "?" in s), sentences[-1])
    rest = [s for s in sentences if s is not question]
    return f"{question} Os dados sao estes, depois da pergunta: " + " ".join(rest)


def _apply_noisy_channel(query: str, rng: random.Random) -> str:
    """Simulate a bad transcription: stray marks and collapsed spaces between words."""
    words = query.split(" ")
    out: List[str] = []
    for word in words:
        if not _HAS_DIGIT.search(word) and rng.random() < 0.12:
            word = word + rng.choice(("..", "~", "*", "__"))
        out.append(word)
    text = " ".join(out)
    return re.sub(r"(\w{4,}) (\w{4,})", lambda m: f"{m.group(1)}{m.group(2)}" if rng.random() < 0.1 else m.group(0), text)


PERTURBATIONS: Dict[str, Callable[[str, random.Random], str]] = {
    "typos": _apply_typos,
    "slang": _apply_slang,
    "code_switch": _apply_code_switch,
    "reorder": _apply_reorder,
    "noisy_channel": _apply_noisy_channel,
}


def perturb_semantics(query: str, rng: random.Random, strategy: str | None = None) -> Tuple[str, str]:
    """Push a statement off the distribution the router's centroids were built on.

    Returns the perturbed text and the strategy name, which the corpus records so
    the effect on u(q) can be attributed to a specific kind of noise.

    ``typos`` and ``noisy_channel`` are probabilistic and can leave a short
    statement untouched, which would silently produce an outlier identical to its
    base. Strategies are therefore tried until the text actually changes.
    """
    order = [strategy] if strategy else []
    remaining = sorted(PERTURBATIONS)
    rng.shuffle(remaining)
    order += [name for name in remaining if name != strategy]

    for name in order:
        candidate = PERTURBATIONS[name](query, rng)
        if candidate.strip() != query.strip():
            return candidate, name
    # Unreachable in practice: slang and code_switch always prepend text.
    return f"{rng.choice(_SLANG_OPENERS)} {query}", "slang"
