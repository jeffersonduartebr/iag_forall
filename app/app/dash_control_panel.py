import os
import redis
import dash
import pandas as pd
import sqlalchemy
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from sqlalchemy import create_engine, text

# =========================================================
# 🔧 Configurações básicas
# =========================================================
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASS = os.getenv("REDIS_PASS", "SenhaForte")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "router_user")
DB_PASS = os.getenv("DB_PASS", "router_pass")
DB_NAME = os.getenv("DB_NAME", "routerdb")

# Redis
r = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, password=REDIS_PASS, decode_responses=True
)

# SQLAlchemy
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

# =========================================================
# 🧠 Funções auxiliares
# =========================================================
def get_system_status():
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


def get_settings_from_redis():
    settings = {
        "temperature": float(r.get("temperature") or 0.7),
        "max_tokens": int(r.get("max_tokens") or 2048),
        "top_p": float(r.get("top_p") or 0.9),
        "bandit_exploration_rate": float(r.get("bandit_exploration_rate") or 0.2),
    }
    return settings


def save_settings_to_redis(data):
    for k, v in data.items():
        r.set(k, v)


def fetch_query_history(limit=30):
    query = """
        SELECT id, query_text, selected_model, judge_model, score, created_at
        FROM query_logs
        ORDER BY created_at DESC
        LIMIT :limit
    """
    try:
        df = pd.read_sql(text(query), engine, params={"limit": limit})
        return df
    except Exception as e:
        print(f"[ERRO HISTÓRICO]: {e}")
        return pd.DataFrame(columns=["id", "query_text", "selected_model", "judge_model", "score", "created_at"])


def get_nsga_weights():
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT objective, weight FROM nsga_weights ORDER BY objective ASC")
            )
            return {row[0]: float(row[1]) for row in result}
    except Exception:
        return {"accuracy": 0.5, "latency": 0.3, "cost": 0.2}


def update_nsga_weight(objective, new_value):
    try:
        with engine.connect() as conn:
            conn.execute(
                text("UPDATE nsga_weights SET weight=:val WHERE objective=:obj"),
                {"val": new_value, "obj": objective},
            )
            conn.commit()
    except Exception as e:
        print(f"[ERRO PESO NSGA]: {e}")


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
        settings = get_settings_from_redis()
        return html.Div(
            [
                html.H3("⚙️ Ajuste de Variáveis de Execução"),
                html.Label("Temperatura"),
                dcc.Slider(id="slider-temp", min=0, max=2, step=0.05, value=settings["temperature"], marks=None, tooltip={"placement": "bottom"}),
                html.Label("Máximo de Tokens"),
                dcc.Input(id="input-tokens", type="number", value=settings["max_tokens"]),
                html.Label("Top-p"),
                dcc.Slider(id="slider-top-p", min=0.0, max=1.0, step=0.01, value=settings["top_p"], marks=None, tooltip={"placement": "bottom"}),
                html.Label("Taxa de Exploração (Bandit)"),
                dcc.Slider(id="slider-bandit", min=0.0, max=1.0, step=0.05, value=settings["bandit_exploration_rate"], marks=None, tooltip={"placement": "bottom"}),
                html.Br(),
                html.Button("💾 Salvar", id="btn-save-vars", n_clicks=0),
                html.Div(id="save-vars-status", style={"marginTop": "10px", "fontWeight": "bold"}),
            ]
        )

    elif tab == "tab-hist":
        df = fetch_query_history()
        return html.Div(
            [
                html.H3("🧠 Histórico de Consultas e Julgamentos"),
                dash_table.DataTable(
                    id="table-hist",
                    columns=[{"name": i, "id": i} for i in df.columns],
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
    if n_clicks > 0:
        save_settings_to_redis(
            {
                "temperature": temp,
                "max_tokens": tokens,
                "top_p": top_p,
                "bandit_exploration_rate": bandit,
            }
        )
        return f"✅ Configurações salvas com sucesso ({n_clicks})"
    return ""


@app.callback(
    Output("table-hist", "data"),
    Input("btn-refresh-hist", "n_clicks"),
)
def refresh_history(n_clicks):
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
