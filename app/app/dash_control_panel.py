"""
dash_control_panel.py
----------------------------------------------------
Painel de Controle do Router Core
 - Aba 1: Controle e parâmetros do sistema
 - Aba 2: Histórico e métricas com gráficos interativos
----------------------------------------------------
Integra Redis, Banco de Dados (settings_current, settings_history, query_logs)
e coleta métricas para visualização em tempo real.
"""

import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import pandas as pd
import sqlalchemy
from sqlalchemy import text
import redis
import os
import json
from datetime import datetime
from dotenv import load_dotenv

# =====================================================
# 🔧 Inicialização e conexões
# =====================================================
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "mariadb")
DB_USER = os.getenv("DB_USER", "router_user")
DB_PASS = os.getenv("DB_PASS", "router_pass")
DB_NAME = os.getenv("DB_NAME", "routerdb")
DB_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"

engine = sqlalchemy.create_engine(DB_URL, pool_pre_ping=True)
r = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379, db=0, decode_responses=True)

app = dash.Dash(__name__, title="Router Core Dashboard", suppress_callback_exceptions=True)
server = app.server

# =====================================================
# 🔹 Utilitários auxiliares
# =====================================================
def get_db_value(key):
    try:
        with engine.connect() as conn:
            res = conn.execute(text("SELECT svalue FROM settings_current WHERE sk = :k"), {"k": key}).fetchone()
            return json.loads(res[0]) if res else None
    except Exception:
        return None


def get_env_value(key, default=None):
    val = os.getenv(key, default)
    try:
        return json.loads(val)
    except Exception:
        return val


def get_config():
    """Obtém configuração atual (prioridade Redis → BD → .env)."""
    return {
        "OLLAMA_MODEL": r.get("router:ollama_model") or get_db_value("OLLAMA_MODEL") or get_env_value("OLLAMA_MODEL", "ollama/gemma3:4b-it-qat"),
        "TEMPERATURE": float(r.get("router:temperature") or get_db_value("TEMPERATURE") or 0.4),
        "MAX_TOKENS": int(r.get("router:max_tokens") or get_db_value("MAX_TOKENS") or 512),
        "OLLAMA_MAX_PARALLEL": int(r.get("router:ollama_max_parallel") or get_db_value("OLLAMA_MAX_PARALLEL") or 2),
    }


def persist_change(key, value, actor="dash_panel"):
    """Atualiza Redis e banco, registrando histórico."""
    try:
        r.set(f"router:{key.lower()}", value)
        sval = json.dumps(value)
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO settings_current (sk, svalue)
                VALUES (:k, :v)
                ON DUPLICATE KEY UPDATE svalue=:v, updated_at=NOW()
            """), {"k": key, "v": sval})
            conn.execute(text("""
                INSERT INTO settings_history (sk, svalue, source, actor)
                VALUES (:k, :v, 'dash', :a)
            """), {"k": key, "v": sval, "a": actor})
    except Exception as e:
        print(f"[dash_control_panel] Erro ao persistir {key}: {e}")


# =====================================================
# 🎨 Layout principal (Tabs)
# =====================================================
app.layout = html.Div([
    html.H2("🧠 Painel de Controle - Router Core", style={"textAlign": "center"}),
    html.Hr(),

    dcc.Tabs(
        id="tabs",
        value="tab-config",
        colors={"border": "#ccc", "primary": "#0b5394", "background": "#e6f2ff"},
        children=[
            dcc.Tab(label="⚙️ Configurações", value="tab-config"),
            dcc.Tab(label="📈 Histórico e Métricas", value="tab-metrics"),
        ]
    ),
    html.Div(id="tabs-content")
])

# =====================================================
# 🧩 Aba 1 — Controle e parâmetros
# =====================================================
def render_config_tab():
    config = get_config()
    return html.Div([
        html.Br(),
        html.H4("⚙️ Parâmetros de Execução"),
        html.Label("Modelo ativo"),
        dcc.Dropdown(
            id="model-dropdown",
            options=[
                {"label": "Gemma 3 4B QAT", "value": "ollama/gemma3:4b-it-qat"},
                {"label": "Granite 1B", "value": "ollama/granite4:1b"},
                {"label": "DeepSeek R1 1.5B", "value": "ollama/deepseek-r1:1.5b"},
                {"label": "OpenAI GPT-5", "value": "openai/gpt-5"},
            ],
            value=config["OLLAMA_MODEL"],
            clearable=False,
            style={"width": "50%"}
        ),
        html.Br(),

        html.Label("Temperatura"),
        dcc.Slider(0, 1, 0.05, value=config["TEMPERATURE"], id="temp-slider", tooltip={"placement": "bottom"}),
        html.Div(id="temp-value", style={"marginBottom": "10px"}),

        html.Label("Máx. Tokens por resposta"),
        dcc.Input(id="max-tokens", type="number", value=config["MAX_TOKENS"], min=64, max=8192, step=64),
        html.Br(), html.Br(),

        html.Label("Execuções paralelas (OLLAMA_MAX_PARALLEL)"),
        dcc.Slider(1, 4, 1, value=config["OLLAMA_MAX_PARALLEL"], id="parallel-slider", marks={i: str(i) for i in range(1, 5)}),
        html.Br(),

        html.Button("Salvar alterações", id="save-btn", n_clicks=0,
                    style={"background": "#2b7", "color": "white", "padding": "8px 16px"}),

        html.Div(id="save-msg", style={"marginTop": "15px", "color": "green"}),

        html.Hr(),
        html.H4("📊 Status e Monitoramento"),
        html.Div(id="status-info"),
        dcc.Interval(id="interval-refresh", interval=10_000, n_intervals=0)
    ])

# =====================================================
# 📈 Aba 2 — Histórico e Métricas
# =====================================================
def render_metrics_tab():
    return html.Div([
        html.Br(),
        html.H4("📈 Histórico de Configurações e Custos"),
        dcc.Dropdown(
            id="metric-type",
            options=[
                {"label": "Alterações de Parâmetros", "value": "settings"},
                {"label": "Custos por Modelo (LLM)", "value": "costs"},
                {"label": "Desempenho do Cache", "value": "cache"},
            ],
            value="settings",
            style={"width": "40%"}
        ),
        html.Br(),
        dcc.Graph(id="metric-graph", style={"height": "600px"}),
        dcc.Interval(id="interval-metrics", interval=30_000, n_intervals=0)
    ])

# =====================================================
# Callbacks — Tabs
# =====================================================
@app.callback(Output("tabs-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-config":
        return render_config_tab()
    return render_metrics_tab()

# =====================================================
# Callbacks — Aba Configurações
# =====================================================
@app.callback(
    Output("save-msg", "children"),
    Input("save-btn", "n_clicks"),
    State("model-dropdown", "value"),
    State("temp-slider", "value"),
    State("max-tokens", "value"),
    State("parallel-slider", "value"),
)
def save_changes(n, model, temp, tokens, parallel):
    if n > 0:
        persist_change("OLLAMA_MODEL", model)
        persist_change("TEMPERATURE", temp)
        persist_change("MAX_TOKENS", tokens)
        persist_change("OLLAMA_MAX_PARALLEL", parallel)
        return "✅ Configurações atualizadas com sucesso!"
    return ""

@app.callback(Output("status-info", "children"), Input("interval-refresh", "n_intervals"))
def refresh_status(_):
    """Exibe resumo operacional."""
    try:
        with engine.connect() as conn:
            total_queries = conn.execute(text("SELECT COUNT(*) FROM query_logs")).scalar() if conn.dialect.has_table(conn, "query_logs") else 0
            last_update = conn.execute(text("SELECT MAX(updated_at) FROM settings_current")).scalar()
        hits = r.get("semantic_cache_hits_total") or 0
        misses = r.get("semantic_cache_misses_total") or 0
        return html.Div([
            html.P(f"🧠 Consultas registradas: {total_queries}"),
            html.P(f"📅 Última atualização de configuração: {last_update or 'N/A'}"),
            html.P(f"💾 Cache Hits: {hits} | Misses: {misses}"),
            html.P(f"⏰ Atualizado em: {datetime.now().strftime('%H:%M:%S')}"),
        ])
    except Exception as e:
        return html.P(f"Erro ao carregar status: {e}")

# =====================================================
# Callbacks — Aba Métricas
# =====================================================
@app.callback(
    Output("metric-graph", "figure"),
    Input("metric-type", "value"),
    Input("interval-metrics", "n_intervals")
)
def update_metrics_graph(metric_type, _):
    try:
        with engine.connect() as conn:
            if metric_type == "settings":
                df = pd.read_sql(text("""
                    SELECT sk AS parametro, svalue, updated_at
                    FROM settings_history
                    WHERE updated_at > NOW() - INTERVAL 7 DAY
                """), conn)
                df["updated_at"] = pd.to_datetime(df["updated_at"])
                fig = px.scatter(df, x="updated_at", y="svalue", color="parametro",
                                 title="Alterações recentes de parâmetros (últimos 7 dias)")
                return fig

            elif metric_type == "costs":
                df = pd.read_sql(text("""
                    SELECT model, SUM(cost_usd) AS custo_total, DATE(created_at) AS data
                    FROM model_costs
                    GROUP BY model, DATE(created_at)
                    ORDER BY data DESC
                """), conn)
                fig = px.bar(df, x="data", y="custo_total", color="model",
                             title="Custos acumulados por modelo")
                return fig

            elif metric_type == "cache":
                hits = int(r.get("semantic_cache_hits_total") or 0)
                misses = int(r.get("semantic_cache_misses_total") or 0)
                df = pd.DataFrame({
                    "Tipo": ["Hits", "Misses"],
                    "Valor": [hits, misses]
                })
                fig = px.pie(df, names="Tipo", values="Valor", title="Desempenho do Cache")
                return fig

    except Exception as e:
        return px.scatter(title=f"Erro ao carregar métricas: {e}")

# =====================================================
# Execução
# =====================================================
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
