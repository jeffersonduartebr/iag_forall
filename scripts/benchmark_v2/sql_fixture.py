# Objective: Fixed in-memory SQLite schema and seed data used to grade SQL items by result set.
"""The schema every SQL item in the corpus is written against.

It is small on purpose but deliberately not trivial: a nullable foreign key
(``aluno.orientador_id``), a student with no enrollments and a discipline with
no students, so ``LEFT JOIN`` versus ``INNER JOIN`` and ``COUNT(*)`` versus
``COUNT(col)`` produce different answers. Those are exactly the mistakes a
weaker model makes, and a fixture without them cannot detect any of them.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Iterator

SCHEMA_SQL = """
CREATE TABLE departamento (
    id          INTEGER PRIMARY KEY,
    nome        TEXT NOT NULL UNIQUE,
    orcamento   REAL NOT NULL
);

CREATE TABLE professor (
    id              INTEGER PRIMARY KEY,
    nome            TEXT NOT NULL,
    departamento_id INTEGER NOT NULL REFERENCES departamento(id),
    salario         REAL NOT NULL
);

CREATE TABLE aluno (
    id            INTEGER PRIMARY KEY,
    nome          TEXT NOT NULL,
    ano_ingresso  INTEGER NOT NULL,
    orientador_id INTEGER REFERENCES professor(id)
);

CREATE TABLE disciplina (
    id              INTEGER PRIMARY KEY,
    codigo          TEXT NOT NULL UNIQUE,
    nome            TEXT NOT NULL,
    creditos        INTEGER NOT NULL,
    departamento_id INTEGER NOT NULL REFERENCES departamento(id)
);

CREATE TABLE matricula (
    aluno_id      INTEGER NOT NULL REFERENCES aluno(id),
    disciplina_id INTEGER NOT NULL REFERENCES disciplina(id),
    semestre      TEXT NOT NULL,
    nota          REAL,
    PRIMARY KEY (aluno_id, disciplina_id, semestre)
);
"""

SEED_SQL = """
INSERT INTO departamento (id, nome, orcamento) VALUES
    (1, 'Computacao', 500000.0),
    (2, 'Matematica', 300000.0),
    (3, 'Fisica', 250000.0);

INSERT INTO professor (id, nome, departamento_id, salario) VALUES
    (1, 'Ada Lovelace',    1, 12000.0),
    (2, 'Alan Turing',     1, 14000.0),
    (3, 'Emmy Noether',    2, 13000.0),
    (4, 'Marie Curie',     3,  9000.0);

-- Bruno has no advisor; Elisa has no advisor and no enrollments.
INSERT INTO aluno (id, nome, ano_ingresso, orientador_id) VALUES
    (1, 'Ana Souza',     2022, 1),
    (2, 'Bruno Lima',    2022, NULL),
    (3, 'Carla Dias',    2023, 2),
    (4, 'Diego Alves',   2023, 3),
    (5, 'Elisa Rocha',   2024, NULL);

-- ORG301 has no enrollments at all.
INSERT INTO disciplina (id, codigo, nome, creditos, departamento_id) VALUES
    (1, 'BD101',  'Banco de Dados I',              4, 1),
    (2, 'ORG201', 'Organizacao de Computadores',   4, 1),
    (3, 'MAT101', 'Calculo I',                     6, 2),
    (4, 'MAT202', 'Algebra Linear',                4, 2),
    (5, 'ORG301', 'Arquiteturas Paralelas',        4, 1);

-- Two enrollments have a NULL grade (in progress).
INSERT INTO matricula (aluno_id, disciplina_id, semestre, nota) VALUES
    (1, 1, '2024.1', 9.0),
    (1, 3, '2024.1', 7.5),
    (1, 2, '2024.2', 8.0),
    (2, 1, '2024.1', 6.0),
    (2, 3, '2024.1', 4.5),
    (3, 1, '2024.2', 10.0),
    (3, 4, '2024.2', NULL),
    (4, 3, '2024.1', 5.5),
    (4, 4, '2024.2', 8.5),
    (4, 2, '2024.2', NULL);
"""

#: Human-readable schema, embedded verbatim in every SQL item statement.
SCHEMA_DESCRIPTION = """departamento(id, nome, orcamento)
professor(id, nome, departamento_id -> departamento.id, salario)
aluno(id, nome, ano_ingresso, orientador_id -> professor.id, pode ser NULL)
disciplina(id, codigo, nome, creditos, departamento_id -> departamento.id)
matricula(aluno_id -> aluno.id, disciplina_id -> disciplina.id, semestre, nota, pode ser NULL)"""


def build_connection() -> sqlite3.Connection:
    """In-memory database with the schema and the seed rows applied."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_SQL)
    conn.executescript(SEED_SQL)
    conn.commit()
    return conn


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    """Context-managed fixture connection, closed on exit."""
    conn = build_connection()
    try:
        yield conn
    finally:
        conn.close()
