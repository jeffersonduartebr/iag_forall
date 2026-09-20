# Objective: Parametric generators of mathematics items with computed gold answers.
"""Mathematics generators.

Every gold value here is *computed* from the drawn parameters, so the statement
and the answer can never drift apart. Statements that do not resolve to an
integer say how to round, both to make grading fair and because following that
instruction is itself part of what we measure.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

from .basegen import BaseSpec, fmt, generator

DISC = "matematica"

#: Integer answers must match exactly; real answers get a 0.1% relative window.
TOL_EXACT = 1e-6
TOL_REAL = 1e-3

ROUND2 = "Responda apenas com o valor numerico, arredondado a duas casas decimais."
ROUND4 = "Responda apenas com o valor numerico, arredondado a quatro casas decimais."


# ---------------------------------------------------------------------------
# One logical step: the raw material for the verbosity traps
# ---------------------------------------------------------------------------


@generator(DISC)
def percent_of(rng: Any) -> BaseSpec:
    """Single application of a percentage."""
    pct = rng.choice([12, 15, 18, 23, 27, 34, 42, 56, 68, 73])
    total = rng.choice([180, 240, 320, 480, 650, 720, 950, 1250, 1840, 2400])
    gold = round(total * pct / 100, 2)
    return BaseSpec(
        topic="porcentagem",
        query=f"Quanto e {pct}% de {total}? {ROUND2}",
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=1,
        tolerance=TOL_REAL,
        tags=["aritmetica"],
    )


@generator(DISC)
def linear_equation(rng: Any) -> BaseSpec:
    """Solve ``ax + b = c`` for x, drawn so the root is an integer."""
    a = rng.choice([3, 4, 6, 7, 8, 9, 11, 12])
    root = rng.randint(-12, 25)
    b = rng.randint(-40, 40)
    c = a * root + b
    sign = "+" if b >= 0 else "-"
    return BaseSpec(
        topic="equacao-linear",
        query=f"Resolva a equacao {a}x {sign} {abs(b)} = {c}. Responda apenas com o valor de x.",
        answer_type="numeric",
        grader="numeric",
        gold=float(root),
        steps=1,
        tolerance=TOL_EXACT,
        tags=["algebra"],
    )


@generator(DISC)
def circle_area(rng: Any) -> BaseSpec:
    """Area of a circle from its radius: one formula, one substitution."""
    radius = rng.choice([3, 5, 7, 9, 12, 15, 18, 21])
    gold = round(math.pi * radius**2, 2)
    return BaseSpec(
        topic="geometria-plana",
        query=f"Uma circunferencia tem raio {radius} cm. Qual e a sua area em cm2? Use pi = 3,14159. {ROUND2}",
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=1,
        tolerance=TOL_REAL,
        tags=["geometria"],
    )


@generator(DISC)
def arithmetic_mean(rng: Any) -> BaseSpec:
    """Mean of a small sample."""
    values = [rng.randint(20, 95) for _ in range(rng.choice([5, 6, 7]))]
    gold = round(sum(values) / len(values), 2)
    listing = ", ".join(str(v) for v in values)
    return BaseSpec(
        topic="media-aritmetica",
        query=f"Calcule a media aritmetica dos valores: {listing}. {ROUND2}",
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=1,
        tolerance=TOL_REAL,
        tags=["estatistica"],
    )


@generator(DISC)
def gcd_pair(rng: Any) -> BaseSpec:
    """Greatest common divisor of two numbers built from shared factors."""
    base = rng.choice([6, 8, 12, 14, 18, 21, 24])
    x, y = base * rng.randint(2, 9), base * rng.randint(2, 9)
    while x == y:
        y = base * rng.randint(2, 9)
    gold = math.gcd(x, y)
    return BaseSpec(
        topic="mdc",
        query=f"Qual e o maximo divisor comum entre {x} e {y}? Responda apenas com o numero.",
        answer_type="numeric",
        grader="numeric",
        gold=float(gold),
        steps=1,
        tolerance=TOL_EXACT,
        tags=["teoria-dos-numeros"],
    )


@generator(DISC)
def monomial_derivative(rng: Any) -> BaseSpec:
    """Derivative of a monomial evaluated at a point: one rule, one substitution."""
    coef = rng.randint(2, 9)
    power = rng.randint(2, 5)
    point = rng.randint(2, 6)
    gold = coef * power * point ** (power - 1)
    return BaseSpec(
        topic="derivada",
        query=(
            f"Seja f(x) = {coef}x^{power}. Qual e o valor de f'({point})? Responda apenas com o numero."
        ),
        answer_type="numeric",
        grader="numeric",
        gold=float(gold),
        steps=1,
        tolerance=TOL_EXACT,
        tags=["calculo"],
    )


@generator(DISC)
def single_probability(rng: Any) -> BaseSpec:
    """Probability of one draw from an urn."""
    red = rng.randint(3, 12)
    blue = rng.randint(4, 15)
    green = rng.randint(2, 9)
    total = red + blue + green
    gold = round(red / total, 4)
    return BaseSpec(
        topic="probabilidade-simples",
        query=(
            f"Uma urna contem {red} bolas vermelhas, {blue} azuis e {green} verdes. "
            f"Retirando uma bola ao acaso, qual e a probabilidade de ser vermelha? {ROUND4}"
        ),
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=1,
        tolerance=TOL_REAL,
        tags=["probabilidade"],
    )


@generator(DISC)
def simple_interest(rng: Any) -> BaseSpec:
    """One application of the simple-interest formula."""
    principal = rng.choice([1200, 2500, 3800, 5000, 7400, 9600])
    rate = rng.choice([1.5, 2.0, 2.5, 3.0, 4.0])
    months = rng.choice([6, 9, 12, 18, 24])
    gold = round(principal * (rate / 100) * months, 2)
    return BaseSpec(
        topic="juros-simples",
        query=(
            f"Um capital de R$ {principal} e aplicado a juros simples de {fmt(rate)}% ao mes "
            f"durante {months} meses. Qual e o valor dos juros? {ROUND2}"
        ),
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=1,
        tolerance=TOL_REAL,
        tags=["matematica-financeira"],
    )


# ---------------------------------------------------------------------------
# Multi-step: each carries a dense phrasing that withholds the scaffolding
# ---------------------------------------------------------------------------


@generator(DISC)
def bayes_diagnostic(rng: Any) -> BaseSpec:
    """Posterior probability of a positive test: prevalence, sensitivity, specificity."""
    prevalence = rng.choice([0.01, 0.02, 0.04, 0.05, 0.08])
    sensitivity = rng.choice([0.90, 0.92, 0.95, 0.97, 0.99])
    specificity = rng.choice([0.85, 0.90, 0.93, 0.96, 0.98])
    true_pos = prevalence * sensitivity
    false_pos = (1 - prevalence) * (1 - specificity)
    gold = round(true_pos / (true_pos + false_pos), 4)
    pv, sn, sp = fmt(prevalence * 100), fmt(sensitivity * 100), fmt(specificity * 100)
    return BaseSpec(
        topic="bayes",
        query=(
            f"Uma doenca atinge {pv}% da populacao. Um teste tem sensibilidade de {sn}% "
            f"e especificidade de {sp}%. Uma pessoa escolhida ao acaso testa positivo. "
            f"Qual e a probabilidade de ela realmente ter a doenca? {ROUND4}"
        ),
        query_dense=(
            f"Prevalencia {pv}%, sensibilidade {sn}%, especificidade {sp}%. "
            f"Valor preditivo positivo? {ROUND4}"
        ),
        steps_dense=4,
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=4,
        tolerance=TOL_REAL,
        tags=["probabilidade", "inferencia"],
    )


@generator(DISC)
def linear_system(rng: Any) -> BaseSpec:
    """2x2 system with an integer solution, asked for one variable."""
    x, y = rng.randint(-9, 14), rng.randint(-9, 14)
    a, b = rng.randint(2, 9), rng.randint(2, 9)
    c, d = rng.randint(2, 9), rng.randint(2, 9)
    while a * d - b * c == 0:
        d = rng.randint(2, 9)
    e, f = a * x + b * y, c * x + d * y
    return BaseSpec(
        topic="sistema-linear",
        query=(
            f"Resolva o sistema:\n{a}x + {b}y = {e}\n{c}x + {d}y = {f}\n"
            "Responda apenas com o valor de x."
        ),
        query_dense=f"Sistema [[{a},{b}],[{c},{d}]] · [x,y] = [{e},{f}]. Valor de x?",
        steps_dense=3,
        answer_type="numeric",
        grader="numeric",
        gold=float(x),
        steps=3,
        tolerance=TOL_EXACT,
        tags=["algebra-linear"],
    )


@generator(DISC)
def compound_vs_simple(rng: Any) -> BaseSpec:
    """Difference between compound and simple interest: two models then a subtraction."""
    principal = rng.choice([2000, 3500, 5000, 8000, 12000])
    rate = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0]) / 100
    periods = rng.choice([6, 8, 10, 12])
    compound = principal * ((1 + rate) ** periods - 1)
    simple = principal * rate * periods
    gold = round(compound - simple, 2)
    return BaseSpec(
        topic="juros-compostos",
        query=(
            f"Um capital de R$ {principal} e aplicado por {periods} meses a uma taxa de "
            f"{fmt(rate * 100)}% ao mes. Quanto os juros compostos superam os juros simples "
            f"no mesmo periodo? {ROUND2}"
        ),
        query_dense=(
            f"R$ {principal}, {fmt(rate * 100)}% a.m., {periods} meses. "
            f"Diferenca entre o regime composto e o simples? {ROUND2}"
        ),
        steps_dense=4,
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=4,
        tolerance=TOL_REAL,
        tags=["matematica-financeira"],
    )


@generator(DISC)
def modular_exponentiation(rng: Any) -> BaseSpec:
    """Remainder of a large power: needs modular reduction, not brute force."""
    base = rng.randint(3, 19)
    exponent = rng.choice([45, 67, 83, 101, 128, 155, 201])
    modulus = rng.choice([7, 11, 13, 17, 19, 23, 29])
    gold = pow(base, exponent, modulus)
    return BaseSpec(
        topic="aritmetica-modular",
        query=(
            f"Qual e o resto da divisao de {base}^{exponent} por {modulus}? "
            "Responda apenas com o numero."
        ),
        query_dense=f"{base}^{exponent} mod {modulus} = ?",
        steps_dense=3,
        answer_type="numeric",
        grader="numeric",
        gold=float(gold),
        steps=3,
        tolerance=TOL_EXACT,
        tags=["teoria-dos-numeros"],
    )


@generator(DISC)
def committee_counting(rng: Any) -> BaseSpec:
    """Count committees under a constraint: complement counting."""
    men = rng.randint(5, 9)
    women = rng.randint(4, 8)
    size = rng.randint(3, 5)
    total = math.comb(men + women, size)
    no_women = math.comb(men, size) if men >= size else 0
    gold = total - no_women
    return BaseSpec(
        topic="combinatoria",
        query=(
            f"Um grupo tem {men} homens e {women} mulheres. Quantas comissoes de {size} pessoas "
            "podem ser formadas contendo pelo menos uma mulher? Responda apenas com o numero."
        ),
        query_dense=f"{men}H + {women}M, comissoes de {size} com ao menos uma mulher. Quantas?",
        steps_dense=3,
        answer_type="numeric",
        grader="numeric",
        gold=float(gold),
        steps=3,
        tolerance=TOL_EXACT,
        tags=["combinatoria"],
    )


@generator(DISC)
def work_rate(rng: Any) -> BaseSpec:
    """Two workers with different rates: combined time as an exact fraction."""
    hours_a = rng.choice([4, 6, 8, 9, 12])
    hours_b = rng.choice([3, 5, 7, 10, 15])
    while hours_b == hours_a:
        hours_b = rng.choice([3, 5, 7, 10, 15])
    combined = Fraction(1, 1) / (Fraction(1, hours_a) + Fraction(1, hours_b))
    gold = round(float(combined), 4)
    return BaseSpec(
        topic="taxas-de-trabalho",
        query=(
            f"Uma maquina executa uma tarefa em {hours_a} horas e outra executa a mesma tarefa "
            f"em {hours_b} horas. Trabalhando simultaneamente, em quantas horas concluem a tarefa? {ROUND4}"
        ),
        query_dense=f"Taxas 1/{hours_a} e 1/{hours_b} por hora, somadas. Tempo total da tarefa? {ROUND4}",
        steps_dense=3,
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=3,
        tolerance=TOL_REAL,
        tags=["proporcionalidade"],
    )


@generator(DISC)
def parabola_optimum(rng: Any) -> BaseSpec:
    """Maximise a quadratic revenue model: vertex, then evaluate."""
    price0 = rng.choice([40, 50, 60, 80, 100])
    demand0 = rng.choice([200, 300, 400, 500])
    drop = rng.choice([4, 5, 8, 10])
    # receita(x) = (price0 + x) * (demand0 - drop * x), com x inteiro e nao negativo.
    # A busca direta evita o caso em que o vertice cai em x < 0 e o maximo esta na borda.
    gold = float(max((price0 + x) * (demand0 - drop * x) for x in range(0, demand0 // drop + 1)))
    return BaseSpec(
        topic="otimizacao-quadratica",
        query=(
            f"Um produto vendido a R$ {price0} tem demanda de {demand0} unidades. Cada real de "
            f"aumento no preco reduz a demanda em {drop} unidades. Considerando apenas aumentos "
            "inteiros, qual e a receita maxima possivel? Responda apenas com o numero."
        ),
        query_dense=(
            f"R(x) = ({price0}+x)({demand0}-{drop}x), x inteiro. Valor maximo de R?"
        ),
        steps_dense=4,
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=4,
        tolerance=TOL_EXACT,
        tags=["funcoes", "otimizacao"],
    )


@generator(DISC)
def geometric_series(rng: Any) -> BaseSpec:
    """Sum of a finite geometric series, then a ratio against the first term."""
    first = rng.choice([2, 3, 5, 7, 10])
    ratio = rng.choice([2, 3])
    terms = rng.choice([6, 7, 8, 9, 10])
    total = first * (ratio**terms - 1) / (ratio - 1)
    gold = round(total / first, 4)
    return BaseSpec(
        topic="progressao-geometrica",
        query=(
            f"Uma progressao geometrica tem primeiro termo {first} e razao {ratio}. "
            f"Qual e a razao entre a soma dos {terms} primeiros termos e o primeiro termo? {ROUND4}"
        ),
        query_dense=f"PG: a1={first}, q={ratio}, n={terms}. Quanto vale S{terms}/a1? {ROUND4}",
        steps_dense=3,
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=3,
        tolerance=TOL_REAL,
        tags=["sequencias"],
    )
