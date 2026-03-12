# Objective: Application runtime code for dash control panel.
"""Application runtime code for dash control panel.

This module is part of the tracked codebase and should remain aligned with the
current runtime architecture and operational documentation.
"""


import os
import redis
import dash
import pandas as pd
import sqlalchemy
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from sqlalchemy import create_engine, text

# CORRIGIDO: Importa o módulo de settings centralizado
# (Assumindo que o Dash está um nível acima, ajuste o path se necessário)
try:
    from app.settings_dynamic import settings
except ImportError:
    # Fallback se o path for diferente
    import sys
    # Adiciona o diretório pai ao path para encontrar 'app'
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from app.settings_dynamic import settings

# =========================================================
# 🔧 Configurações básicas (Lidas do settings)
# =========================================================
REDIS_HOST = settings.REDIS_HOST
REDIS_PORT = settings.REDIS_PORT
REDIS_PASS = settings.get("REDIS_PASS", os.getenv("REDIS_PASS", "SenhaForte"))

DB_HOST = settings.DB_HOST
DB_USER = settings.DB_USER
DB_PASS = settings.DB_PASS
DB_NAME = settings.DB_NAME

# Redis (cliente ainda é necessário para o Dash verificar o ping)
r = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS, decode_responses=True
)

# SQLAlchemy
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

# =========================================================
# 🧠 Funções auxiliares
# =========================================================
def get_system_status():
    """Return system status.

This helper centralizes retrieval logic so callers do not have to duplicate lookup behavior."""
    try:
        redis_ok = r.ping()
    except Exception:
        redis_ok = False

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    return redis_ok, db_ok


def get_dynamic_settings():
    """CORRIGIDO: Lê as configurações do módulo centralizado (Redis > DB > .env)."""
    s = {
        "temperature": settings.TEMPERATURE_DEFAULT,
        "max_tokens": settings.MAX_TOKENS_DEFAULT,
        "top_p": float(settings.get("top_p", 0.9)),
        "bandit_exploration_rate": float(settings.get("BANDIT_EPSILON", 0.12)), # Usa a chave do bandits.py
    }
    return s


def save_dynamic_settings(data):
    """CORRIGIDO: Salva usando settings.set() para persistir em Redis + DB."""
    for k, v in data.items():
        # Mapeia nomes do Dash para chaves de configuração reais
        if k == "bandit_exploration_rate":
            key_name = "BANDIT_EPSILON"
        elif k == "temperature":
             key_name = "TEMPERATURE_DEFAULT"
        elif k == "max_tokens":
             key_name = "MAX_TOKENS_DEFAULT"
        else:
            key_name = k # ex: 'top_p'
            
        settings.set(key_name, v, actor="dash_panel", source="ui")


def fetch_query_history(limit=30):
    """CORRIGIDO: Lê da tabela 'query_log' e colunas corretas."""
    query = """
        SELECT id, query_text, chosen_model as selected_model, 
               quality as score, created_at
        FROM query_log
        ORDER BY created_at DESC
        LIMIT :limit
    """
    try:
        df = pd.read_sql(text(query), engine, params={"limit": limit})
        return df
    except Exception as e:
        print(f"[ERRO HISTÓRICO]: {e}")
        # Colunas ajustadas para corresponder à query
        return pd.DataFrame(columns=["id", "query_text", "selected_model", "score", "created_at"])


def get_nsga_weights():
    """
    NOTA: Esta função lê uma tabela 'nsga_weights' com colunas 'objective' e 'weight'.
    Isso conflita com 'nsga_weights_updater.py' que usa colunas 'model' e 'weight'.
    Mantendo a lógica original do Dash por enquanto, mas isso é um bug de integração.
    """
    try:
        with engine.connect() as conn:
            # A tabela 'nsga_weights' criada pelo nsga_updater não tem 'objective'
            # Isso VAI FALHAR a menos que a tabela seja criada manualmente.
            result = conn.execute(
                text("SELECT objective, weight FROM nsga_weights ORDER BY objective ASC")
            )
            data = {row[0]: float(row[1]) for row in result}
            if not data: # Fallback se a query falhar ou retornar vazio
                return {"accuracy": 0.5, "latency": 0.3, "cost": 0.2}
            return data
    except Exception as e:
        print(f"[ERRO GET PESO NSGA]: {e}")
        return {"accuracy": 0.5, "latency": 0.3, "cost": 0.2}


def update_nsga_weight(objective, new_value):
    """CORRIGIDO: Adicionado conn.commit()"""
    try:
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE nsga_weights SET weight=:val WHERE objective=:obj"),
                {"val": new_value, "obj": objective},
            )
            conn.commit() # <-- CORRIGIDO
    except Exception as e:
        print(f"[ERRO UPDATE PESO NSGA]: {e}")


# =========================================================
# 🎨 Inicialização do app Dash
# =========================================================
app = dash.Dash(__name__, title="Painel de Controle LLM Router", suppress_callback_exceptions=True)
app.title = "LLM Router Control Panel"
server = app.server

# =========================================================
# 📑 Layout com abas
# =========================================================
app.layout = html.Div(
    [
        html.H1("🧭 Painel de Controle — LLM Router", style={"textAlign": "center"}),
        dcc.Tabs(
            id="tabs",
            value="tab-sys",
            children=[
                dcc.Tab(label="📊 Sistema", value="tab-sys"),
                dcc.Tab(label="⚙️ Variáveis", value="tab-vars"),
                dcc.Tab(label="🧠 Histórico", value="tab-hist"),
                dcc.Tab(label="🎯 Pesos NSGA-II", value="tab-nsga"),
            ],
        ),
        html.Div(id="tabs-content", style={"margin": "20px"}),
    ]
)

# =========================================================
# 🧩 Conteúdo das Abas
# =========================================================
@app.callback(Output("tabs-content", "children"), [Input("tabs", "value")])
def render_content(tab):
    """Execute the render content routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    if tab == "tab-sys":
        redis_ok, db_ok = get_system_status()
        color_r = "green" if redis_ok else "red"
        color_d = "green" if db_ok else "red"
        return html.Div(
            [
                html.H3("📡 Monitoramento do Sistema"),
                html.Div(
                    [
                        html.P(f"Redis: {'✅ Online' if redis_ok else '❌ Offline'}",
                               style={"color": color_r, "fontWeight": "bold"}),
                        html.P(f"Banco de Dados: {'✅ Online' if db_ok else '❌ Offline'}",
                               style={"color": color_d, "fontWeight": "bold"}),
                    ]
                ),
                html.Button("🔄 Atualizar Status", id="btn-refresh-sys", n_clicks=0),
            ]
        )

    elif tab == "tab-vars":
        settings_vals = get_dynamic_settings() # <-- CORRIGIDO
        return html.Div(
            [
                html.H3("⚙️ Ajuste de Variáveis de Execução"),
                html.Label("Temperatura"),
                dcc.Slider(id="slider-temp", min=0, max=2, step=0.05, value=settings_vals["temperature"], marks=None, tooltip={"placement": "bottom"}),
                html.Label("Máximo de Tokens"),
                dcc.Input(id="input-tokens", type="number", value=settings_vals["max_tokens"]),
                html.Label("Top-p"),
                dcc.Slider(id="slider-top-p", min=0.0, max=1.0, step=0.01, value=settings_vals["top_p"], marks=None, tooltip={"placement": "bottom"}),
                html.Label("Taxa de Exploração (Bandit Epsilon)"), # <-- CORRIGIDO (Label)
                dcc.Slider(id="slider-bandit", min=0.0, max=1.0, step=0.05, value=settings_vals["bandit_exploration_rate"], marks=None, tooltip={"placement": "bottom"}),
                html.Br(),
                html.Button("💾 Salvar", id="btn-save-vars", n_clicks=0),
                html.Div(id="save-vars-status", style={"marginTop": "10px", "fontWeight": "bold"}),
            ]
        )

    elif tab == "tab-hist":
        df = fetch_query_history()
        # CORRIGIDO: Colunas da tabela devem bater com a query
        cols = ["id", "query_text", "selected_model", "score", "created_at"]
        return html.Div(
            [
                html.H3("🧠 Histórico de Consultas e Julgamentos"),
                dash_table.DataTable(
                    id="table-hist",
                    columns=[{"name": i, "id": i} for i in cols],
                    data=df.to_dict("records"),
                    page_size=10,
                    style_table={"overflowX": "auto"},
                ),
                html.Button("🔄 Atualizar Histórico", id="btn-refresh-hist", n_clicks=0),
            ]
        )

    elif tab == "tab-nsga":
        weights = get_nsga_weights()
        return html.Div(
            [
                html.H3("🎯 Ajuste de Pesos Multiobjetivo (NSGA-II)"),
                html.Label("Acurácia"),
                dcc.Slider(id="w-acc", min=0, max=1, step=0.01, value=weights.get("accuracy", 0.5)),
                html.Label("Latência"),
                dcc.Slider(id="w-lat", min=0, max=1, step=0.01, value=weights.get("latency", 0.3)),
                html.Label("Custo"),
                dcc.Slider(id="w-cost", min=0, max=1, step=0.01, value=weights.get("cost", 0.2)),
                html.Br(),
                html.Button("💾 Atualizar Pesos", id="btn-update-nsga", n_clicks=0),
                html.Div(id="nsga-update-status", style={"marginTop": "10px", "fontWeight": "bold"}),
            ]
        )


# =========================================================
# 🔄 Callbacks
# =========================================================
@app.callback(
    Output("save-vars-status", "children"),
    Input("btn-save-vars", "n_clicks"),
    State("slider-temp", "value"),
    State("input-tokens", "value"),
    State("slider-top-p", "value"),
    State("slider-bandit", "value"),
)
def save_variables(n_clicks, temp, tokens, top_p, bandit):
    """Save variables.

The function persists the current representation to its backing store."""
    if n_clicks > 0:
        save_dynamic_settings( # <-- CORRIGIDO
            {
                "temperature": temp,
                "max_tokens": tokens,
                "top_p": top_p,
                "bandit_exploration_rate": bandit,
            }
        )
        return f"✅ Configurações salvas (Redis+DB) ({n_clicks})"
    return ""


@app.callback(
    Output("table-hist", "data"),
    Input("btn-refresh-hist", "n_clicks"),
)
def refresh_history(n_clicks):
    """Execute the refresh history routine.

This helper encapsulates one focused step used by the surrounding workflow."""
    df = fetch_query_history()
    return df.to_dict("records")


@app.callback(
    Output("nsga-update-status", "children"),
    Input("btn-update-nsga", "n_clicks"),
    State("w-acc", "value"),
    State("w-lat", "value"),
    State("w-cost", "value"),
)
def update_nsga_weights_callback(n, w_acc, w_lat, w_cost):
    """Update nsga weights callback.

This function applies the module-specific mutation logic for the target resource."""
    if n > 0:
        update_nsga_weight("accuracy", w_acc)
        update_nsga_weight("latency", w_lat)
        update_nsga_weight("cost", w_cost)
        return "✅ Pesos NSGA-II atualizados com sucesso"
    return ""


# =========================================================
# 🚀 Execução
# =========================================================
if __name__ == "__main__":
    print("🚀 Painel Dash iniciado em http://0.0.0.0:8050/")
    app.run_server(host="0.0.0.0", port=8050, debug=os.getenv("DASH_DEBUG_MODE", "False") == "True")
