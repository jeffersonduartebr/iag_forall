import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../api/client";
import { EmptyState, ErrorBanner, LoadingBlock, PageHeader, StatusBadge } from "./ui";
import { useToast } from "../context/ToastContext";
import { ExplorationScatterChart } from "./ExplorationScatterChart";

export type ExplorationStatus = {
  enabled: boolean;
  setting_enabled?: boolean;
  openrouter_configured?: boolean;
  config: {
    mode?: string;
    rate: number;
    max_per_day: number;
    max_usd_per_day?: number;
    max_price_prompt_per_1k: number;
    max_price_completion_per_1k: number;
    promote_min_samples: number;
    promote_min_reward: number;
    promote_max_latency_s?: number;
    promote_max_cost_usd_per_1k?: number;
    auto_promote_enabled?: boolean;
    adaptive_rate_enabled?: boolean;
    shadow_compare_rate?: number;
    provider_allowlist: string[];
    pool_size: number;
  };
  usage: {
    explorations_today: number;
    remaining_today: number;
    usd_spent_today?: number;
    usd_remaining_today?: number | null;
  };
  models: Array<{
    model: string;
    count: number;
    mean_reward: number;
    mean_latency_s?: number;
    mean_observed_usd_per_1k?: number;
    catalog_usd_per_1k?: { prompt_usd_per_1k: number; completion_usd_per_1k: number };
    catalog_blended_usd_per_1k?: number;
    cost_tier?: string;
    cost_vs_pool_ratio?: number;
    promotion_blockers?: string[];
    promotion_passed?: string[];
    failure_rate?: number;
    auto_promoted_at?: number;
    blocklisted?: boolean;
    shadow_delta_quality_mean?: number;
    promotable?: boolean;
  }>;
  suggestions: Array<{
    model: string;
    mean_reward: number;
    mean_observed_usd_per_1k?: number;
    count: number;
  }>;
  cost_benchmark?: { pool_median_observed_usd_per_1k?: number | null };
  auto_promoted?: Array<{ model: string; promoted_at: number }>;
  blocklist?: string[];
};

function formatUsdPer1k(value?: number | null): string {
  if (value == null || Number.isNaN(value)) return "—";
  if (value < 0.001) return `$${value.toFixed(6)}/1K`;
  if (value < 0.01) return `$${value.toFixed(5)}/1K`;
  return `$${value.toFixed(4)}/1K`;
}

function costTierLabel(tier?: string): string {
  if (tier === "mais_barato") return "mais barato";
  if (tier === "mais_caro") return "mais caro";
  if (tier === "similar") return "similar";
  return "—";
}

function promotionBlockerLabel(code: string): string {
  const map: Record<string, string> = {
    amostras_insuficientes: "poucas amostras",
    reward_baixo: "reward baixo",
    latencia_alta: "latência alta",
    custo_alto: "custo alto",
    taxa_falha_alta: "muitas falhas",
  };
  return map[code] || code;
}

export type ExplorationUpdate = {
  enabled?: boolean;
  mode?: string;
  rate?: number;
  max_per_day?: number;
  auto_promote_enabled?: boolean;
  max_price_prompt_per_1k?: number;
  max_price_completion_per_1k?: number;
  promote_min_samples?: number;
  promote_min_reward?: number;
  provider_allowlist?: string[];
  pool_size?: number;
};

type Props = {
  onAddCandidate?: (model: string) => void;
};

type CredentialsStatus = {
  configured: boolean;
  masked_key: string;
};

export function OpenRouterExplorationPanel({ onAddCandidate }: Props) {
  const qc = useQueryClient();
  const toast = useToast();

  const credentials = useQuery({
    queryKey: ["openrouter-credentials"],
    queryFn: () => apiFetch<CredentialsStatus>("/admin/models/openrouter/credentials"),
  });

  const status = useQuery({
    queryKey: ["openrouter-exploration"],
    queryFn: () => apiFetch<ExplorationStatus>("/admin/models/openrouter/exploration"),
    refetchInterval: 10000,
  });

  const [enabled, setEnabled] = useState(false);
  const [ratePct, setRatePct] = useState(10);
  const [maxPerDay, setMaxPerDay] = useState(100);
  const [apiKey, setApiKey] = useState("");
  const [mode, setMode] = useState("balanced");

  const modes = useQuery({
    queryKey: ["openrouter-exploration-modes"],
    queryFn: () => apiFetch<{ modes: string[] }>("/admin/models/openrouter/exploration/modes"),
  });

  useEffect(() => {
    if (!status.data) return;
    setEnabled(status.data.setting_enabled ?? status.data.enabled);
    setRatePct(Math.round((status.data.config.rate || 0) * 100));
    setMaxPerDay(status.data.config.max_per_day || 100);
    setMode(status.data.config.mode || "balanced");
  }, [status.data]);

  const saveCredentials = useMutation({
    mutationFn: (key: string) =>
      apiFetch("/admin/models/openrouter/credentials", {
        method: "PUT",
        body: JSON.stringify({ api_key: key }),
      }),
    onSuccess: () => {
      setApiKey("");
      qc.invalidateQueries({ queryKey: ["openrouter-credentials"] });
      qc.invalidateQueries({ queryKey: ["openrouter-exploration"] });
      toast.push("API key OpenRouter salva.", "success");
    },
    onError: (err) => {
      toast.push(err instanceof Error ? err.message : "Falha ao salvar API key", "error");
    },
  });

  const blockModel = useMutation({
    mutationFn: (model: string) =>
      apiFetch("/admin/models/openrouter/exploration/blocklist", {
        method: "POST",
        body: JSON.stringify({ model }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["openrouter-exploration"] });
      toast.push("Modelo adicionado à blocklist.", "success");
    },
  });

  const setModePreset = useMutation({
    mutationFn: (nextMode: string) =>
      apiFetch("/admin/models/openrouter/exploration/mode", {
        method: "PUT",
        body: JSON.stringify({ mode: nextMode }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["openrouter-exploration"] });
      toast.push("Modo de exploração aplicado.", "success");
    },
  });

  const save = useMutation({
    mutationFn: (payload: ExplorationUpdate) =>
      apiFetch("/admin/models/openrouter/exploration", {
        method: "PUT",
        body: JSON.stringify(payload),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["openrouter-exploration"] });
      toast.push("Exploração OpenRouter atualizada.", "success");
    },
    onError: (err) => {
      toast.push(err instanceof Error ? err.message : "Falha ao salvar exploração", "error");
    },
  });

  const onSubmitCredentials = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = apiKey.trim();
    if (trimmed.length < 8) {
      toast.push("Informe uma API key válida (mín. 8 caracteres).", "error");
      return;
    }
    saveCredentials.mutate(trimmed);
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    save.mutate({
      enabled,
      rate: ratePct / 100,
      max_per_day: maxPerDay,
    });
  };

  const toggleNow = (next: boolean) => {
    setEnabled(next);
    save.mutate({ enabled: next, rate: ratePct / 100, max_per_day: maxPerDay });
  };

  const explored = status.data?.models || [];
  const suggestions = status.data?.suggestions || [];

  return (
    <section className="exploration-section" aria-labelledby="openrouter-exploration-title">
      <PageHeader
        title="Exploração OpenRouter"
        description="Descobre modelos OpenRouter fora dos candidatos, mede qualidade/custo/latência e promove automaticamente os que passam nos critérios multi-objetivo."
        actions={
          status.data ? (
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
              <StatusBadge
                label={status.data.enabled ? "ativa" : "inativa"}
                tone={status.data.enabled ? "ok" : "neutral"}
              />
              {status.data.config.auto_promote_enabled ? (
                <StatusBadge label="auto-promote ON" tone="ok" />
              ) : (
                <StatusBadge label="auto-promote OFF" tone="warn" />
              )}
            </div>
          ) : null
        }
      />

      {status.isError ? (
        <ErrorBanner message="Falha ao carregar exploração OpenRouter." onRetry={() => status.refetch()} />
      ) : null}

      {status.isLoading ? <LoadingBlock label="Carregando exploração..." /> : null}

      <form className="form-card exploration-form" onSubmit={onSubmitCredentials}>
        <h3 className="subsection-title" style={{ marginTop: 0 }}>
          Credenciais OpenRouter
        </h3>
        <p className="page-description" style={{ marginBottom: "0.75rem" }}>
          {credentials.data?.configured
            ? `Configurada: ${credentials.data.masked_key || "••••••••"}`
            : "Nenhuma API key configurada — exploração e modelos openrouter/ ficam indisponíveis."}
        </p>
        <div className="exploration-controls">
          <label className="field" style={{ flex: 1, minWidth: 280 }}>
            API key
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-or-v1-..."
              autoComplete="off"
            />
          </label>
          <button type="submit" className="btn-sm" disabled={saveCredentials.isPending}>
            {saveCredentials.isPending ? "Salvando..." : "Salvar API key"}
          </button>
        </div>
      </form>

      {status.data?.setting_enabled && !status.data?.openrouter_configured && !credentials.data?.configured && (
        <div className="banner banner-info" style={{ marginBottom: "1rem" }}>
          Exploração ligada nas configurações, mas a API key OpenRouter ainda não foi configurada (use o formulário acima ou{" "}
          <code>OPENROUTER_API_KEY</code> no <code>.env</code>).
        </div>
      )}

      <form className="form-card exploration-form" onSubmit={onSubmit}>
        <div className="exploration-controls">
          <label className="toggle-field">
            <span>Ativar exploração</span>
            <button
              type="button"
              role="switch"
              aria-checked={enabled}
              className={enabled ? "toggle on" : "toggle"}
              onClick={() => toggleNow(!enabled)}
            >
              <span className="toggle-thumb" />
            </button>
          </label>

          <label className="field">
            Modo
            <select
              value={mode}
              onChange={(e) => {
                const next = e.target.value;
                setMode(next);
                setModePreset.mutate(next);
              }}
            >
              {(modes.data?.modes || ["conservative", "balanced", "aggressive", "cost_hunt"]).map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>

          <label className="field">
            Taxa de exploração (%)
            <input
              type="number"
              min={0}
              max={100}
              step={1}
              value={ratePct}
              onChange={(e) => {
                setRatePct(Number(e.target.value));
              }}
            />
          </label>

          <label className="field">
            Máx. por dia
            <input
              type="number"
              min={1}
              max={10000}
              value={maxPerDay}
              onChange={(e) => {
                setMaxPerDay(Number(e.target.value));
              }}
            />
          </label>

          <button type="submit" className="btn-sm" disabled={save.isPending}>
            {save.isPending ? "Salvando..." : "Salvar configuração"}
          </button>
        </div>

        {status.data ? (
          <p className="card-hint" style={{ marginTop: "0.75rem" }}>
            Hoje: {status.data.usage.explorations_today} explorações · restam {status.data.usage.remaining_today}
            {status.data.usage.usd_spent_today != null
              ? ` · $${status.data.usage.usd_spent_today.toFixed(4)} gastos`
              : ""}
            {status.data.usage.usd_remaining_today != null
              ? ` · $${status.data.usage.usd_remaining_today.toFixed(4)} restantes`
              : ""}{" "}
            · pool {status.data.config.pool_size} · modo {status.data.config.mode || "balanced"} · providers:{" "}
            {(status.data.config.provider_allowlist || []).join(", ") || "todos"}
            {status.data.cost_benchmark?.pool_median_observed_usd_per_1k != null
              ? ` · mediana custo observado: ${formatUsdPer1k(status.data.cost_benchmark.pool_median_observed_usd_per_1k)}`
              : ""}
          </p>
        ) : null}
      </form>

      {(status.data?.auto_promoted || []).length > 0 ? (
        <>
          <h3 className="subsection-title">Promovidos automaticamente</h3>
          <div className="chip-row">
            {(status.data?.auto_promoted || []).map((p) => (
              <span key={`${p.model}-${p.promoted_at}`} className="chip chip-ok">
                <code>{p.model}</code>
              </span>
            ))}
          </div>
        </>
      ) : null}

      {explored.length > 0 ? (
        <>
          <h3 className="subsection-title">Mapa custo × reward</h3>
          <ExplorationScatterChart models={explored} />
        </>
      ) : null}

      {suggestions.length > 0 ? (
        <>
          <h3 id="openrouter-exploration-title" className="subsection-title">
            Sugestões de promoção
          </h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Modelo</th>
                  <th>Reward médio</th>
                  <th>Custo observado ($/1K)</th>
                  <th>Catálogo ($/1K in/out)</th>
                  <th>Comparação</th>
                  <th>Amostras</th>
                  <th>Ação</th>
                </tr>
              </thead>
              <tbody>
                {suggestions.map((s) => (
                  <tr key={s.model}>
                    <td>
                      <code>{s.model}</code>
                    </td>
                    <td>{s.mean_reward.toFixed(3)}</td>
                    <td>{formatUsdPer1k(s.mean_observed_usd_per_1k)}</td>
                    <td>—</td>
                    <td>—</td>
                    <td>{s.count}</td>
                    <td>
                      {onAddCandidate ? (
                        <button type="button" className="secondary btn-sm" onClick={() => onAddCandidate(s.model)}>
                          Adicionar aos candidatos
                        </button>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : null}

      <h3 className="subsection-title">Modelos em exploração</h3>
      {explored.length === 0 ? (
        <EmptyState
          title="Nenhuma exploração registrada ainda"
          description="Ative a exploração e aguarde tráfego de texto para acumular estatísticas."
        />
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Modelo</th>
                <th>Amostras</th>
                <th>Reward médio</th>
                <th>Latência média</th>
                <th>Custo observado ($/1K)</th>
                <th>Catálogo ($/1K in/out)</th>
                <th>Comparação</th>
                <th>Status</th>
                <th>Ação</th>
              </tr>
            </thead>
            <tbody>
              {explored.map((m) => (
                <tr key={m.model}>
                  <td>
                    <code>{m.model}</code>
                  </td>
                  <td>{m.count}</td>
                  <td>{m.mean_reward?.toFixed(3) ?? "—"}</td>
                  <td>{m.mean_latency_s != null ? `${m.mean_latency_s.toFixed(2)}s` : "—"}</td>
                  <td>{formatUsdPer1k(m.mean_observed_usd_per_1k)}</td>
                  <td>
                    {m.catalog_usd_per_1k
                      ? `${formatUsdPer1k(m.catalog_usd_per_1k.prompt_usd_per_1k)} / ${formatUsdPer1k(m.catalog_usd_per_1k.completion_usd_per_1k)}`
                      : "—"}
                  </td>
                  <td>
                    {m.cost_tier ? (
                      <StatusBadge
                        label={costTierLabel(m.cost_tier)}
                        tone={m.cost_tier === "mais_barato" ? "ok" : m.cost_tier === "mais_caro" ? "warn" : "neutral"}
                      />
                    ) : (
                      "—"
                    )}
                  </td>
                  <td>
                    {m.promotable ? (
                      <StatusBadge label="promovível" tone="ok" />
                    ) : m.auto_promoted_at ? (
                      <StatusBadge label="promovido" tone="ok" />
                    ) : (
                      <StatusBadge
                        label={
                          m.promotion_blockers?.length
                            ? m.promotion_blockers.map(promotionBlockerLabel).join(", ")
                            : "coletando"
                        }
                        tone="info"
                      />
                    )}
                  </td>
                  <td>
                    <button
                      type="button"
                      className="secondary btn-sm"
                      disabled={blockModel.isPending}
                      onClick={() => blockModel.mutate(m.model)}
                    >
                      Descartar
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
