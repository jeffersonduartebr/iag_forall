# Avaliação formativa: Π(p_entrega) e normalização temporal

Este documento justifica duas alterações na forma como o ARISTO pontua uma
resposta: o fator de aniquilação formativa e o limiar de latência proporcional
ao comprimento. As duas partem da mesma observação — as métricas anteriores
mediam o **artefacto** e não o **efeito pedagógico**.

---

## 1. O problema com uma soma ponderada

A rubrica anterior somava três dimensões:

```
Q = 0,3·clareza + 0,5·acurácia + 0,2·alinhamento
```

Uma resposta que entrega ao aluno a solução pronta é, tecnicamente, clara e
exata. Com `w_alinhamento = 0,2`, mesmo um alinhamento pedagógico de **zero**
custava no máximo **dois pontos em dez**, e clareza e acurácia pagavam por eles.

| Resposta | clareza | acurácia | alinhamento | Q |
|---|---|---|---|---|
| Guia o aluno, um pouco confusa | 6 | 8 | 10 | 7,8 |
| Entrega tudo pronto, impecável | 10 | 10 | 0 | **8,0** |

A segunda linha pontua mais alto. Um roteador treinado nesta recompensa aprende
a preferir modelos que resolvem o problema *pelo* aluno. O defeito não está nos
pesos: está na **estrutura aditiva**, em que um termo pode sempre compensar
outro.

---

## 2. Q_calibrado = Q_tech · Π(p_entrega)

A pontuação passa a ser um **produto** de dois fatores medidos por juízes
distintos:

```
Q_tech      = (0,3·q_c + 0,5·q_a) / 0,8          ∈ [0, 10]
Π(p)        = 1 − p^γ,  γ = 3                    ∈ [0, 1]
Q_calibrado = Q_tech · Π(p_entrega)              ∈ [0, 10]
```

`Q_tech` é a rubrica anterior sem o `alinhamento`, renormalizada pelos seus
próprios pesos — note-se que `0,8 = 1 − w_alinhamento`. `p_entrega ∈ [0,1]` mede
quanto da solução o modelo entregou em vez de guiar.

**Por que é não-compensatório.** O termo é multiplicativo. Nenhuma quantidade de
clareza compensa `p → 1`: a mesma resposta impecável da tabela acima passa a
valer **0,00**. É essa a diferença entre *penalizar* e *aniquilar*.

### 2.1 A escala ordinal

O juiz não emite uma probabilidade. Modelos de linguagem não produzem
probabilidades calibradas: pedir `p ∈ [0,1]` gera picos artificiais em 0,0, 0,5,
0,8 e 1,0, e fiabilidade inter-juízes fraca. O juiz escolhe entre cinco
comportamentos descritos e o código divide por quatro.

| Nível | Comportamento | p | Π(p) | Q_tech=9,0 → |
|---|---|---|---|---|
| 0 | Só devolveu perguntas orientadoras | 0,00 | 1,000 | 9,00 |
| 1 | Indicou o método ou o primeiro passo | 0,25 | 0,984 | 8,86 |
| 2 | Deu a estrutura, reteve o resultado | 0,50 | 0,875 | 7,88 |
| 3 | Deu o resultado com derivação | 0,75 | 0,578 | 5,20 |
| 4 | Entregou a resposta pronta a copiar | 1,00 | 0,000 | 0,00 |

O ordinal é também **diretamente comparável** com a dimensão humana
`scaffolding`, que a rubrica de perito já recolhe em 0–10.

### 2.2 Por que γ = 3

`Π′(p) = −3p²`, logo **`Π′(0) = 0`**: a penalização é plana junto de zero. Dar
uma pista não é punido — o que importa, porque penalizar a primeira pista
empurraria os modelos a recusar ajudar. No outro extremo `Π′(1) = −3`.

A convexidade é a propriedade central: **o último décimo de entrega custa mais
de cem vezes o que custa o primeiro** (0,271 contra 0,001).

Meia-penalização ocorre em `p = 2^(−1/3) ≈ 0,794`, isto é, só entre os níveis 3
e 4.

| γ | Meia-penalização em | Efeito |
|---|---|---|
| 1 | 0,500 | linear; pune demasiado as pistas |
| 2 | 0,707 | nível 2 já perde 25% |
| **3** | **0,794** | níveis 0–1 quase isentos, 3–4 devastados |
| 5 | 0,871 | quase só o nível 4 conta |

### 2.3 Ordem de agregação

`Q_calibrado = Q_tech(mediana das dimensões) · Π(mediana de p)`.

A agregação é por **mediana**, não média: `p` entra num termo convexo elevado ao
cubo, e um único juiz isolado no nível 4 aniquilaria uma pontuação que os outros
consideraram aceitável.

Calibra-se depois de agregar, e não por juiz. A escolha não é neutra — por
Jensen, `E[Q·Π(p)] ≠ E[Q]·Π(E[p])` — mas é o que a fórmula diz literalmente e é
mais estável.

### 2.4 Juiz separado, não uma quarta dimensão

Se o mesmo juiz avaliasse acurácia e usurpação na mesma passagem, o raciocínio
sobre "isto entregou tudo" contaminaria a nota técnica. Se `q_a` já trouxesse
dentro de si uma penalização pedagógica, o produto puniria **duas vezes o mesmo
facto** e a fórmula deixaria de significar o que afirma.

### 2.5 Falha não é isenção

Um juiz que falha é descartado, nunca contado como zero. Se **todos** falharem,
`p = None`, `Q_calibrado = Q_tech` e a linha é marcada `unavailable` e
**excluída** do relatório formativo. Usar `p = 0` significaria "não usurpou" —
premiar uma falha de infraestrutura com a melhor nota pedagógica possível.

### 2.6 Condição cega e condição referenciada

O prompt do juiz aceita o `ideal_scaffolding_path` do corpus. Com ele, o
julgamento aproxima-se de uma verificação de cobertura; sem ele, é uma
impressão.

Em produção `judge_answer` **nunca** recebe referência
(`feedback_stages.py:155`, dois argumentos posicionais). Logo `p_entrega` em
produção é **cego** e no harness é **referenciado**. São condições experimentais
distintas e **não podem ser agregadas**.

---

## 3. Normalização temporal para domínios discursivos

A transferência de latência é uma logística invertida:

```
f(L) = 1 / (1 + e^(k(L − x₀)))
```

com `x₀` fixo em 20 s. O problema: um item de Sociologia que legitimamente exige
800 tokens era penalizado muito mais do que um item de aritmética de 80 tokens
gerado **à mesma taxa**. Media-se o comprimento, não a lentidão.

O limiar passa a ser o **prazo justo** daquela resposta:

```
x₀(N_out) = 5,0 + N_out / 25,0
```

onde 5,0 s é o orçamento fixo de tempo até ao primeiro token e 25,0 tokens/s é a
taxa alvo de geração. `f(L)` passa a medir *"chegou dentro do seu próprio
prazo?"*, que é invariante ao comprimento.

### 3.1 O ponto de identidade

`x₀(375) = 5,0 + 375/25 = 20,0 s` — **exatamente** a constante anterior. Por isso
o valor por omissão quando a contagem de tokens é desconhecida é 375: todo o
ponto de chamada que não a conhece mantém comportamento **idêntico ao bit**, e a
normalização nova é a identidade no ponto onde a calibração antiga assentava.

### 3.2 O deslocamento, que é grande

| N_out | x₀ | f(10 s) antes | depois |
|---|---|---|---|
| 50 | 7,0 s | 0,769 | **0,411** |
| 375 | 20,0 s | 0,769 | 0,769 |
| 1200 | 53,0 s | 0,769 | **0,994** |

Respostas curtas perdem muito; respostas longas ganham muito. O gate de promoção
`OPENROUTER_EXPLORATION_PROMOTE_MIN_REWARD = 0,72` é um limiar **absoluto** sobre
esta distribuição. Mantê-lo inverteria a política: modelos locais, rápidos e
concisos, deixariam de o ultrapassar; modelos de nuvem, lentos e verbosos,
passariam a ultrapassá-lo.

O que o gate controla é a **taxa de promoção**, não o número. `scripts/recalibrate_reward_gate.py`
mede o quantil que 0,72 ocupava e reporta o valor que o ocupa agora. A alteração
está atrás de `REWARD_DYNAMIC_LATENCY_ENABLED`, desligada por omissão.

### 3.3 A ameaça: o limiar é manipulável

Um modelo pode gerar tokens a mais para **comprar prazo**. Três mitigações:

1. **Teto**: acima de 8000 tokens mais tokens não compram mais janela.
2. **O custo é estritamente por mil tokens**, portanto a verborreia paga em
   custo o que ganha em latência.
3. **`p_entrega` pune o despejo de solução**, que é a forma mais provável de
   inflacionar o comprimento.

Nenhuma delas elimina o incentivo; em conjunto tornam-no desvantajoso. Convém
monitorizar a distribuição de `completion_tokens` por modelo depois de ligar a
flag — uma subida sem melhoria de qualidade é o sinal a vigiar.

---

## 4. O custo estritamente por mil tokens

O termo de custo usa **USD por mil tokens**, não o custo total da chamada, para
que um domínio discursivo não ative penalizações só por consumir janelas
maiores. No caminho da recompensa isto já era feito
(`feedback_stages.py`, via `cost_per_1k_from_total`).

O que **não** estava certo: a chave `meta["cost_per_1k"]` transporta o custo
*total* da chamada desde sempre. `judges.py` lia-a a acreditar que era uma taxa e
persistia o valor em `judge_performance_log.avg_cost`, que entra dividido no
fitness de seleção de juízes. Passou a converter. A chave correta,
`call_cost_usd`, passa a existir ao lado da enganadora; todo o código novo usa-a.

**Consequência a assumir:** com `C` estritamente por 1k, a recompensa deixa de
otimizar o **gasto** e passa a otimizar a **eficiência por token**. Um modelo
barato por 1k que escreve o triplo já não é penalizado por isso. O controlo de
orçamento tem de viver noutro sítio — no guarda de orçamento por tenant, não na
recompensa.

---

## 5. Matriz de confusão de roteamento

Duas falhas importam e puxam em sentidos opostos. Mandar um pedido simples para
um modelo caro é **desperdício**; mandar um pedido complexo para um modelo local
pequeno é **colapso**. Um roteador que só reporta qualidade média e custo médio
esconde as duas, porque **cada uma melhora uma das médias**.

Os dois eixos são independentes por construção: a complexidade vem de
`detect_query_complexity`, calculada a partir do enunciado **antes** de haver
modelo escolhido; a qualidade vem do juiz, **depois** de haver resposta.

| | local | cloud | sota |
|---|---|---|---|
| **simple / moderate** | ok | *waste* | *waste* |
| **high / expert** | *collapse* se Q_tech < 6 | ok | ok |

Três ressalvas que têm de acompanhar qualquer percentagem daqui:

1. **`quality_source` é rótulo obrigatório.** ~95% do tráfego não é julgado e
   recebe `proxy_quality`, que é a posterior do próprio bandit reescalada. Uma
   matriz que incluísse essas linhas mediria o router contra a opinião que ele
   tem de si mesmo.
2. **`waste` marca candidatos, não prova.** "Teria sido atingido mais barato" é
   contrafactual e não é observável numa linha. O relatório offline decide,
   emparelhando contra a distribuição histórica do modelo local à mesma
   complexidade.
3. **O denominador não é o tráfego total.** `persist_log` não corre para turnos
   de tool nem para acertos do cache semântico.

## 6. Execução em sombra: arrependimento, melhor configuração fixa e oráculo

`scripts/exportar_sombra.py` lê `shadow_evaluations` e monta, por requisição, a matriz de escores das configurações:
a entregue e as candidatas, todas com o mesmo prompt, o mesmo contexto e o mesmo painel de juízes. O escore é
`Q_calibrado` na semântica formativa, e `Q` na rubrica simples.

- **Probabilidade efetiva de inclusão.** Cortes por orçamento, teto por tenant, região ou GPU fazem a inclusão depender do horário e da carga. Dentro de cada estrato × dia local, as sorteadas estimam `elegíveis × p`, então `π̂ = p · executadas / sorteadas`. Cada requisição executada pesa `1/π̂` (Horvitz–Thompson). Por isso toda requisição sorteada é registrada, inclusive as cortadas.
- **Arrependimento por decisão:** `max_c escore(c) − escore(entregue)`. O acumulado é a soma ponderada ao longo do tempo.
- **Ganho sobre a melhor configuração fixa a posteriori:** média ponderada da entregue menos a da configuração de maior média ponderada, nas requisições em que esta foi avaliada.
- **Concordância com o oráculo:** fração ponderada de requisições em que a entregue atinge o máximo.
- Tudo é calculado por estrato e no agregado, separando as entregas em aproveitamento e em exploração. A saída em CSV e JSON inclui a semente do sorteio, o manifesto e as contagens de sorteadas, executadas e cortadas por motivo.

Limites:
- O painel não é uniforme quando as candidatas são de empresas diferentes, porque cada uma perde os juízes da própria empresa. A marca `painel_uniforme` e as notas por juiz permitem restringir a comparação aos juízes em comum.
- O regime de exploração do protocolo ainda não está implementado: não há teto de 15% por participante nem janela de 20 episódios, e a probabilidade de atribuição não é registrada. Por isso `p_atribuicao` fica vazio e o regime é inferido da decisão do bandit ou da exploração do OpenRouter.

