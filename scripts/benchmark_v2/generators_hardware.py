# Objective: Parametric generators of computer organization and assembly items with computed gold answers.
"""Computer organization and hardware assembly generators.

The multi-step items here are the classic chained calculations of the field —
address splitting, AMAT, Amdahl, CPI, effective bandwidth — where a single
wrong intermediate value poisons the result. That property is what makes them
discriminative: a model cannot bluff its way to the right number.
"""

from __future__ import annotations

import math
from typing import Any

from .basegen import BaseSpec, fmt, generator

DISC = "organizacao_computadores"

TOL_EXACT = 1e-6
TOL_REAL = 1e-3

ROUND2 = "Responda apenas com o valor numerico, arredondado a duas casas decimais."


# ---------------------------------------------------------------------------
# One logical step
# ---------------------------------------------------------------------------


@generator(DISC)
def binary_prefix_conversion(rng: Any) -> BaseSpec:
    """Convert between binary multiples: one multiplication by a power of 1024."""
    amount = rng.choice([2, 3, 4, 6, 8, 12, 16, 24, 32])
    gold = float(amount * 1024)
    return BaseSpec(
        topic="unidades-binarias",
        query=f"Quantos MiB existem em {amount} GiB? Responda apenas com o numero.",
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=1,
        tolerance=TOL_EXACT,
        tags=["representacao"],
    )


@generator(DISC)
def block_offset_bits(rng: Any) -> BaseSpec:
    """Offset bits of a cache block: a single base-2 logarithm."""
    block = rng.choice([16, 32, 64, 128, 256])
    gold = float(int(math.log2(block)))
    return BaseSpec(
        topic="cache-offset",
        query=(
            f"Uma cache usa blocos de {block} bytes. Quantos bits do endereco sao usados "
            "como deslocamento dentro do bloco? Responda apenas com o numero."
        ),
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=1,
        tolerance=TOL_EXACT,
        tags=["memoria-cache"],
    )


@generator(DISC)
def address_lines(rng: Any) -> BaseSpec:
    """Address lines needed to reach a given number of words."""
    bits = rng.choice([10, 12, 14, 16, 18, 20, 22])
    words = 2**bits
    return BaseSpec(
        topic="barramento-de-endereco",
        query=(
            f"Uma memoria possui {words} palavras enderecaveis. Quantas linhas de endereco "
            "sao necessarias? Responda apenas com o numero."
        ),
        answer_type="numeric",
        grader="numeric",
        gold=float(bits),
        steps=1,
        tolerance=TOL_EXACT,
        tags=["memoria"],
    )


@generator(DISC)
def clock_period(rng: Any) -> BaseSpec:
    """Clock period in nanoseconds from the frequency."""
    ghz = rng.choice([1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.5])
    gold = round(1 / ghz, 4)
    return BaseSpec(
        topic="periodo-de-clock",
        query=(
            f"Um processador opera a {fmt(ghz, 1)} GHz. Qual e o periodo de clock em nanossegundos? "
            "Responda apenas com o valor numerico, arredondado a quatro casas decimais."
        ),
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=1,
        tolerance=TOL_REAL,
        tags=["desempenho"],
    )


@generator(DISC)
def raid_usable_capacity(rng: Any) -> BaseSpec:
    """Usable capacity of a RAID array: one level, one formula."""
    disks = rng.choice([4, 6, 8, 10, 12])
    size = rng.choice([2, 4, 6, 8, 12])
    level, usable = rng.choice(
        [("RAID 0", disks), ("RAID 10", disks // 2), ("RAID 5", disks - 1), ("RAID 6", disks - 2)]
    )
    gold = float(usable * size)
    return BaseSpec(
        topic="raid",
        query=(
            f"Um arranjo {level} e montado com {disks} discos de {size} TB cada. "
            "Qual e a capacidade util em TB? Responda apenas com o numero."
        ),
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=1,
        tolerance=TOL_EXACT,
        tags=["armazenamento"],
    )


@generator(DISC)
def total_ram(rng: Any) -> BaseSpec:
    """Installed memory from module count and size."""
    modules = rng.choice([2, 4, 8])
    size = rng.choice([4, 8, 16, 32])
    return BaseSpec(
        topic="capacidade-de-memoria",
        query=(
            f"Uma placa-mae recebe {modules} modulos de {size} GB cada. "
            "Qual e a memoria total instalada em GB? Responda apenas com o numero."
        ),
        answer_type="numeric",
        grader="numeric",
        gold=float(modules * size),
        steps=1,
        tolerance=TOL_EXACT,
        tags=["montagem"],
    )


@generator(DISC)
def transfer_time(rng: Any) -> BaseSpec:
    """Transfer time from size and sustained bandwidth."""
    size_gb = rng.choice([8, 16, 24, 40, 64, 120])
    rate = rng.choice([50, 120, 200, 400, 550])
    gold = round(size_gb * 1000 / rate, 2)
    return BaseSpec(
        topic="taxa-de-transferencia",
        query=(
            f"Um arquivo de {size_gb} GB e copiado para um disco com taxa sustentada de "
            f"{rate} MB/s. Considerando 1 GB = 1000 MB, quantos segundos leva a copia? {ROUND2}"
        ),
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=1,
        tolerance=TOL_REAL,
        tags=["armazenamento", "desempenho"],
    )


@generator(DISC)
def psu_total_draw(rng: Any) -> BaseSpec:
    """Total nominal draw: a sum of component ratings."""
    cpu = rng.choice([65, 95, 105, 125, 170])
    gpu = rng.choice([75, 160, 220, 285, 350])
    rest = rng.choice([40, 55, 70, 85])
    return BaseSpec(
        topic="consumo-eletrico",
        query=(
            f"Uma montagem tem CPU de {cpu} W, GPU de {gpu} W e demais componentes somando {rest} W. "
            "Qual e o consumo nominal total em watts? Responda apenas com o numero."
        ),
        answer_type="numeric",
        grader="numeric",
        gold=float(cpu + gpu + rest),
        steps=1,
        tolerance=TOL_EXACT,
        tags=["montagem", "fonte"],
    )


# ---------------------------------------------------------------------------
# Multi-step
# ---------------------------------------------------------------------------


@generator(DISC)
def cache_address_split(rng: Any) -> BaseSpec:
    """Tag bits of a set-associative cache: offset, sets, index, then subtraction."""
    addr_bits = rng.choice([32, 32, 40, 48])
    block = rng.choice([32, 64, 128])
    ways = rng.choice([1, 2, 4, 8])
    sets = rng.choice([64, 128, 256, 512, 1024])
    cache_kib = sets * ways * block // 1024
    offset = int(math.log2(block))
    index = int(math.log2(sets))
    gold = float(addr_bits - offset - index)
    assoc = "mapeamento direto" if ways == 1 else f"associativa por conjuntos de {ways} vias"
    return BaseSpec(
        topic="cache-endereco",
        query=(
            f"Uma cache de {cache_kib} KiB, {assoc}, usa blocos de {block} bytes em uma arquitetura "
            f"de {addr_bits} bits. Quantos bits do endereco compoem o campo de tag? "
            "Responda apenas com o numero."
        ),
        query_dense=(
            f"Cache {cache_kib} KiB, {ways} via(s), bloco {block} B, endereco {addr_bits} bits. Bits de tag?"
        ),
        steps_dense=4,
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=4,
        tolerance=TOL_EXACT,
        tags=["memoria-cache", "arquitetura"],
    )


@generator(DISC)
def amat_two_levels(rng: Any) -> BaseSpec:
    """Average memory access time across two cache levels."""
    t1 = rng.choice([1, 2, 3])
    m1 = rng.choice([0.02, 0.04, 0.05, 0.08, 0.10])
    t2 = rng.choice([8, 10, 12, 15, 20])
    m2 = rng.choice([0.10, 0.20, 0.25, 0.40, 0.50])
    penalty = rng.choice([80, 100, 150, 200, 300])
    gold = round(t1 + m1 * (t2 + m2 * penalty), 4)
    return BaseSpec(
        topic="amat",
        query=(
            f"Um sistema tem cache L1 com tempo de acerto de {t1} ciclo(s) e taxa de falhas de "
            f"{fmt(m1 * 100)}%. As falhas de L1 vao para a L2, com tempo de acesso de {t2} ciclos "
            f"e taxa de falhas de {fmt(m2 * 100)}%. Uma falha na L2 custa {penalty} ciclos. "
            "Qual e o tempo medio de acesso a memoria, em ciclos? "
            "Responda apenas com o valor numerico, arredondado a quatro casas decimais."
        ),
        query_dense=(
            f"L1: {t1}c, miss {fmt(m1 * 100)}%. L2: {t2}c, miss {fmt(m2 * 100)}%. "
            f"Penalidade {penalty}c. AMAT em ciclos?"
        ),
        steps_dense=4,
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=4,
        tolerance=TOL_REAL,
        tags=["memoria-cache", "desempenho"],
    )


@generator(DISC)
def amdahl_speedup(rng: Any) -> BaseSpec:
    """Speedup under Amdahl's law for a given parallel fraction and core count."""
    parallel = rng.choice([0.50, 0.60, 0.75, 0.80, 0.90, 0.95])
    cores = rng.choice([2, 4, 8, 16, 32, 64])
    gold = round(1 / ((1 - parallel) + parallel / cores), 4)
    return BaseSpec(
        topic="lei-de-amdahl",
        query=(
            f"Um programa tem {fmt(parallel * 100)}% do tempo de execucao paralelizavel. "
            f"Executando-o em {cores} nucleos, qual e o ganho de desempenho segundo a lei de Amdahl? "
            "Responda apenas com o valor numerico, arredondado a quatro casas decimais."
        ),
        query_dense=f"Amdahl: p={fmt(parallel)}, n={cores}. Speedup?",
        steps_dense=3,
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=3,
        tolerance=TOL_REAL,
        tags=["paralelismo", "desempenho"],
    )


@generator(DISC)
def cpi_execution_time(rng: Any) -> BaseSpec:
    """Execution time from an instruction mix, per-class CPI and clock rate."""
    freq = [rng.choice([40, 45, 50]), rng.choice([20, 25, 30]), 0]
    freq[2] = 100 - freq[0] - freq[1]
    cpi = [1, rng.choice([2, 3, 4]), rng.choice([5, 6, 8])]
    instructions = rng.choice([1, 2, 4, 5]) * 10**9
    ghz = rng.choice([2.0, 2.5, 3.0, 3.5])
    mean_cpi = sum(f / 100 * c for f, c in zip(freq, cpi))
    gold = round(instructions * mean_cpi / (ghz * 10**9), 4)
    return BaseSpec(
        topic="cpi",
        query=(
            f"Um programa com {instructions // 10**9} bilhao(oes) de instrucoes tem a seguinte "
            f"distribuicao: {freq[0]}% aritmeticas (CPI {cpi[0]}), {freq[1]}% de acesso a memoria "
            f"(CPI {cpi[1]}) e {freq[2]}% de desvio (CPI {cpi[2]}). O processador opera a "
            f"{fmt(ghz, 1)} GHz. Qual e o tempo de execucao em segundos? "
            "Responda apenas com o valor numerico, arredondado a quatro casas decimais."
        ),
        query_dense=(
            f"Mix {freq[0]}/{freq[1]}/{freq[2]}%, CPI {cpi[0]}/{cpi[1]}/{cpi[2]}, "
            f"{instructions // 10**9}e9 instrucoes, {fmt(ghz, 1)} GHz. Tempo de execucao (s)?"
        ),
        steps_dense=4,
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=4,
        tolerance=TOL_REAL,
        tags=["desempenho", "arquitetura"],
    )


@generator(DISC)
def memory_bandwidth(rng: Any) -> BaseSpec:
    """Theoretical peak bandwidth of a multi-channel memory configuration."""
    transfers = rng.choice([2400, 2666, 3200, 3600, 4800, 5600])
    channels = rng.choice([1, 2, 4])
    width_bits = 64
    gold = round(transfers * 10**6 * (width_bits / 8) * channels / 10**9, 2)
    return BaseSpec(
        topic="largura-de-banda",
        query=(
            f"Um sistema usa memoria DDR a {transfers} MT/s, com barramento de {width_bits} bits "
            f"por canal e {channels} canal(is) ativo(s). Qual e a largura de banda teorica de pico "
            f"em GB/s (1 GB = 10^9 bytes)? {ROUND2}"
        ),
        query_dense=f"{transfers} MT/s, {width_bits} bits, {channels} canal(is). Banda de pico em GB/s?",
        steps_dense=3,
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=3,
        tolerance=TOL_REAL,
        tags=["memoria", "desempenho"],
    )


@generator(DISC)
def psu_sizing_with_headroom(rng: Any) -> BaseSpec:
    """Size a power supply: sum the draw, add headroom, correct for efficiency."""
    cpu = rng.choice([65, 95, 125, 170])
    gpu = rng.choice([160, 220, 285, 350, 450])
    rest = rng.choice([45, 60, 75, 90])
    headroom = rng.choice([20, 25, 30])
    efficiency = rng.choice([0.80, 0.85, 0.90])
    total = cpu + gpu + rest
    gold = round(total * (1 + headroom / 100) / efficiency, 2)
    return BaseSpec(
        topic="dimensionamento-de-fonte",
        query=(
            f"Uma montagem consome {cpu} W na CPU, {gpu} W na GPU e {rest} W nos demais componentes. "
            f"Considerando uma margem de seguranca de {headroom}% sobre o consumo total e uma fonte "
            f"com {fmt(efficiency * 100)}% de eficiencia, qual e a potencia de entrada minima em watts? {ROUND2}"
        ),
        query_dense=(
            f"{cpu}+{gpu}+{rest} W, margem {headroom}%, eficiencia {fmt(efficiency * 100)}%. "
            "Potencia de entrada minima (W)?"
        ),
        steps_dense=4,
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=4,
        tolerance=TOL_REAL,
        tags=["montagem", "fonte"],
    )


@generator(DISC)
def pipeline_speedup(rng: Any) -> BaseSpec:
    """Effective pipeline speedup once stall cycles are accounted for."""
    stages = rng.choice([5, 6, 8, 10, 12])
    stall_rate = rng.choice([0.10, 0.15, 0.20, 0.25, 0.30])
    stall_cycles = rng.choice([1, 2, 3])
    gold = round(stages / (1 + stall_rate * stall_cycles), 4)
    return BaseSpec(
        topic="pipeline",
        query=(
            f"Um pipeline de {stages} estagios sofre paradas em {fmt(stall_rate * 100)}% das "
            f"instrucoes, cada uma custando {stall_cycles} ciclo(s) adicional(is). "
            "Qual e o ganho efetivo em relacao a execucao nao segmentada? "
            "Responda apenas com o valor numerico, arredondado a quatro casas decimais."
        ),
        query_dense=(
            f"Pipeline {stages} estagios, {fmt(stall_rate * 100)}% de stalls de "
            f"{stall_cycles} ciclo(s). Speedup efetivo?"
        ),
        steps_dense=3,
        answer_type="numeric",
        grader="numeric",
        gold=gold,
        steps=3,
        tolerance=TOL_REAL,
        tags=["arquitetura", "desempenho"],
    )
