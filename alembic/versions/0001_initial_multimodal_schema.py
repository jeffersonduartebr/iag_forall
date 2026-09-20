"""Initial Multimodal Core Schema for brand-new database"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# Alembic identifiers
revision = "0001_initial_multimodal_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # =========================================================
    # EMA HISTORY
    # =========================================================
    op.create_table(
        "ema_history",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("modality", mysql.ENUM("text", "vision", "multimodal"), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("ema_latency", sa.Float, nullable=False),
        sa.Column("ema_cost", sa.Float, nullable=False),
        sa.Column("ema_quality", sa.Float, nullable=False),
        sa.Column("ema_alignment", sa.Float, nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP"),
                  server_onupdate=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("model", "modality", name="uniq_model_modality"),
        sa.Index("idx_ema_modality", "modality"),
        sa.Index("idx_ema_model", "model"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    op.create_table(
        "ema_history_log",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("modality", mysql.ENUM("text", "vision", "multimodal"), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("ema_latency", sa.Float, nullable=False),
        sa.Column("ema_cost", sa.Float, nullable=False),
        sa.Column("ema_quality", sa.Float, nullable=False),
        sa.Column("ema_alignment", sa.Float, nullable=False),
        sa.Column("update_num", sa.Integer, nullable=False),
        sa.Column("created_at", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Index("idx_ema_log_model_created_at", "model", "created_at"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # =========================================================
    # BANDIT CONTEXT STATS
    # =========================================================
    op.create_table(
        "bandit_context_stats",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("context_label", sa.String(255), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("avg_reward", sa.Float, server_default="0"),
        sa.Column("count", sa.Integer, server_default="0"),
        sa.Column("var", sa.Float, server_default="0"),
        sa.Column("M2", sa.Float, server_default="0"),
        sa.Column(
            "last_update",
            sa.TIMESTAMP,
            server_default=sa.text("CURRENT_TIMESTAMP"),
            server_onupdate=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint("context_label", "model", name="uq_ctx_model"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # =========================================================
    # QUERY LOG — MULTIMODAL
    # =========================================================
    op.create_table(
        "query_log",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("query_text", sa.Text),
        sa.Column("chosen_model", sa.String(255), nullable=False),
        sa.Column("modality", sa.String(32), server_default="text"),
        sa.Column("image_provided", sa.Integer, server_default="0"),
        sa.Column("answer", sa.Text),
        sa.Column("image_output_b64", sa.Text),
        sa.Column("query_embedding", sa.LargeBinary),
        sa.Column("answer_embedding", sa.LargeBinary),
        sa.Column("quality", sa.Float),
        sa.Column("latency_s", sa.Float),
        sa.Column("cost_per_1k", sa.Float),
        sa.Column("reward", sa.Float),
        sa.Column("context_label", sa.String(64)),
        sa.Column("raw_payload", sa.Text),
        sa.Column("created_at", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Index("idx_query_log_created_at", "created_at"),
        sa.Index("idx_query_log_model", "chosen_model"),
        sa.Index("idx_query_log_modality", "modality"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # =========================================================
    # SEMANTIC CACHE
    # =========================================================
    op.create_table(
        "semantic_cache",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("query_hash", sa.String(64), unique=True),
        sa.Column("modality", mysql.ENUM("text", "vision", "multimodal"), nullable=False,
                  server_default="text"),
        sa.Column("query_text", sa.Text),
        sa.Column("context_category", sa.String(50)),
        sa.Column("answer", sa.Text),
        sa.Column("model_used", sa.String(255)),
        sa.Column("embedding", sa.LargeBinary),
        sa.Column("created_at", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Index("idx_cache_modality", "modality"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # =========================================================
    # JUDGES
    # =========================================================
    op.create_table(
        "judge_logs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("created_at", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("query", sa.Text),
        sa.Column("answer", sa.Text),
        sa.Column("judge_model", sa.String(255)),
        sa.Column("score_before", sa.Float),
        sa.Column("fallback_model", sa.String(255)),
        sa.Column("score_after", sa.Float),
        sa.Column("event_type", sa.String(50)),
        sa.Column("modality", sa.String(32), server_default="text"),
        sa.Column("image_hash", sa.String(128)),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    op.create_table(
        "judge_performance_log",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("judge_model", sa.String(255), nullable=False),
        sa.Column("avg_score", sa.Float, server_default="0"),
        sa.Column("avg_latency", sa.Float, server_default="0"),
        sa.Column("avg_cost", sa.Float, server_default="0"),
        sa.Column("consistency", sa.Float, server_default="0"),
        sa.Column("fitness", sa.Float, server_default="0"),
        sa.Column("window_start", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("window_end", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("created_at", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Index("idx_judge_model", "judge_model"),
        sa.Index("idx_window_end", "window_end"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # =========================================================
    # MODEL METRICS
    # =========================================================
    op.create_table(
        "model_metrics",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("modality", mysql.ENUM("text", "vision", "multimodal"), nullable=False),
        sa.Column("latency_ms", sa.Float, server_default="0"),
        sa.Column("cost_usd", sa.Float, server_default="0"),
        sa.Column("quality_score", sa.Float, server_default="0"),
        sa.Column("fitness", sa.Float, server_default="0"),
        sa.Column("generation", sa.Integer, server_default="0"),
        sa.Column("timestamp", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP"),
                  server_onupdate=sa.text("CURRENT_TIMESTAMP")),
        sa.Index("idx_model_timestamp", "model", "timestamp"),
        sa.Index("idx_metrics_modality", "modality"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # =========================================================
    # NSGA-II
    # =========================================================
    op.create_table(
        "nsga_weights",
        sa.Column("modality", mysql.ENUM("text", "vision", "multimodal"), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("weight", sa.Float, nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP"),
                  server_onupdate=sa.text("CURRENT_TIMESTAMP")),
        sa.PrimaryKeyConstraint("modality", "model", name="pk_nsga_weights"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    op.create_table(
        "nsga_params",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("N_pop", sa.Integer, nullable=False),
        sa.Column("N_gen", sa.Integer, nullable=False),
        sa.Column("cxpb", sa.Float, nullable=False),
        sa.Column("mutpb", sa.Float, nullable=False),
        sa.Column("eta_c", sa.Float, nullable=False),
        sa.Column("eta_m", sa.Float, nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP"),
                  server_onupdate=sa.text("CURRENT_TIMESTAMP")),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    op.create_table(
        "nsga_meta_results",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("timestamp", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("trial_id", sa.Integer, nullable=False),
        sa.Column("N_pop", sa.Integer, nullable=False),
        sa.Column("N_gen", sa.Integer, nullable=False),
        sa.Column("cxpb", sa.Float, nullable=False),
        sa.Column("mutpb", sa.Float, nullable=False),
        sa.Column("eta_c", sa.Float, nullable=False),
        sa.Column("eta_m", sa.Float, nullable=False),
        sa.Column("eff_mean", sa.Float, nullable=False),
        sa.Column("eff_std", sa.Float, nullable=False),
        sa.Index("idx_trial_id", "trial_id"),
        sa.Index("idx_eff", "eff_mean", "eff_std"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # =========================================================
    # MODEL REGISTRY
    # =========================================================
    op.create_table(
        "model_registry",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("modality", mysql.ENUM("text", "vision", "multimodal"), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("max_context", sa.Integer, server_default="4096"),
        sa.Column("cost_per_1k", sa.Float, server_default="0"),
        sa.Column("is_active", sa.Boolean, server_default=sa.text("1")),
        sa.UniqueConstraint("modality", "model", name="uniq_mod_model"),
        sa.Index("idx_registry_modality", "modality"),
        sa.Index("idx_provider", "provider"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # =========================================================
    # SETTINGS DYNAMIC CACHE
    # =========================================================
    op.create_table(
        "settings_dynamic_cache",
        sa.Column("key_name", sa.String(255), primary_key=True),
        sa.Column("value_json", mysql.JSON, nullable=False),
        sa.Column("updated_at", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP"),
                  server_onupdate=sa.text("CURRENT_TIMESTAMP")),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # =========================================================
    # CHROMA COLLECTIONS INDEX
    # =========================================================
    op.create_table(
        "chroma_collections_index",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("modality", mysql.ENUM("text", "vision", "multimodal"), nullable=False),
        sa.Column("collection_name", sa.String(255), nullable=False),
        sa.Column("embedding_dim", sa.Integer, nullable=False),
        sa.Column("last_count", sa.Integer, server_default="0"),
        sa.Column("updated_at", sa.TIMESTAMP, server_default=sa.text("CURRENT_TIMESTAMP"),
                  server_onupdate=sa.text("CURRENT_TIMESTAMP")),
        sa.UniqueConstraint("collection_name", "modality", name="uniq_collection_mod"),
        sa.Index("idx_chroma_modality", "modality"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )


def downgrade():

    # DROP TABLES
    op.drop_table("chroma_collections_index")
    op.drop_table("settings_dynamic_cache")
    op.drop_table("model_registry")
    op.drop_table("nsga_meta_results")
    op.drop_table("nsga_params")
    op.drop_table("nsga_weights")
    op.drop_table("model_metrics")
    op.drop_table("judge_performance_log")
    op.drop_table("judge_logs")
    op.drop_table("semantic_cache")
    op.drop_table("query_log")
    op.drop_table("bandit_context_stats")
    op.drop_table("ema_history_log")
    op.drop_table("ema_history")
