import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { apiFetch } from "../api/client";
import { Card, ErrorBanner, LoadingBlock, PageHeader, StatusBadge } from "../components/ui";

type RoiSummary = {
  query_count: number;
  actual_cost_usd: number;
  baseline_cost_usd: number;
  savings_usd: number;
  savings_pct: number;
  acceptable_queries?: number;
  cost_per_acceptable_actual_usd?: number | null;
  cost_per_acceptable_baseline_usd?: number | null;
  projected_monthly_savings_usd?: number;
};

type RoiReport = {
  tenant_id?: string | null;
  period_days: number;
  baseline_model: string;
  quality_threshold: number;
  insufficient_data: boolean;
  summary: RoiSummary;
  daily_series: Array<{
    date: string;
    queries: number;
    actual_cost_usd: number;
    baseline_cost_usd: number;
    savings_usd: number;
  }>;
  model_breakdown: Array<{
    model: string;
    count: number;
    actual_cost_usd: number;
    baseline_cost_usd: number;
    savings_usd: number;
    quality_mean: number;
  }>;
  methodology?: Record<string, unknown>;
  disclaimer?: string;
};

const PERIODS = [
  { label: "7 dias", value: 7 },
  { label: "30 dias", value: 30 },
  { label: "90 dias", value: 90 },
];

function money(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  return `$${v.toFixed(2)}`;
}

function tooltipMoney(value: unknown): [string, string] {
  const n = typeof value === "number" ? value : Number(value ?? 0);
  return [`$${(Number.isFinite(n) ? n : 0).toFixed(4)}`, ""];
}

export function RoiPage() {
  const [tenantId, setTenantId] = useState("");
  const [days, setDays] = useState(30);
  const [baselineModel, setBaselineModel] = useState("");

  const budgets = useQuery({
    queryKey: ["roi-budgets"],
    queryFn: () => apiFetch<{ items: Array<{ tenant_id: string }> }>("/admin/budgets"),
  });

  const usage = useQuery({
    queryKey: ["roi-usage"],
    queryFn: () => apiFetch<{ tenants?: Record<string, unknown> }>("/admin/quotas/usage"),
  });

  const tenantOptions = useMemo(() => {
    const ids = new Set<string>();
    (budgets.data?.items || []).forEach((b) => ids.add(b.tenant_id));
    Object.keys(usage.data?.tenants || {}).forEach((id) => ids.add(id));
    return Array.from(ids).sort();
  }, [budgets.data, usage.data]);

  const roi = useQuery({
    queryKey: ["dashboard-roi", tenantId, days, baselineModel],
    queryFn: () => {
      const params = new URLSearchParams({ days: String(days) });
      if (tenantId.trim()) params.set("tenant_id", tenantId.trim());
      if (baselineModel.trim()) params.set("baseline_model", baselineModel.trim());
      return apiFetch<RoiReport>(`/admin/dashboard/roi?${params}`);
    },
    refetchInterval: 60000,
  });

  const cumulative = useMemo(() => {
    let acc = 0;
    return (roi.data?.daily_series || []).map((row) => {
      acc += row.savings_usd;
      return { ...row, cumulative_savings_usd: acc };
    });
  }, [roi.data]);

  const report = roi.data;
  const summary = report?.summary;

  return (
    <div className="roi-page">
      <PageHeader
        title="ROI — Economia vs baseline"
        description="Prova quanto o roteador economiza em relação a um modelo premium fixo. Use em pilotos e propostas comerciais."
      />

      <div className="form-card form-row roi-filters">
        <label className="field">
          Tenant (opcional)
          <select value={tenantId} onChange={(e) => setTenantId(e.target.value)}>
            <option value="">Todos os tenants</option>
            {tenantOptions.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Período
          <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
            {PERIODS.map((p) => (
              <option key={p.value} value={p.value}>
                {p.label}
              </option>
            ))}
          </select>
        </label>
        <label className="field">
          Baseline
          <input
            placeholder="openai/gpt-4o"
            value={baselineModel}
            onChange={(e) => setBaselineModel(e.target.value)}
          />
        </label>
        <button type="button" className="secondary" onClick={() => roi.refetch()} disabled={roi.isFetching}>
          Atualizar
        </button>
      </div>

      {roi.isLoading ? <LoadingBlock label="Calculando economia..." /> : null}
      {roi.isError ? <ErrorBanner message="Falha ao carregar relatório ROI." onRetry={() => roi.refetch()} /> : null}

      {report?.insufficient_data ? (
        <div className="chart-placeholder">
          Sem dados em <code>query_log</code> no período. Execute consultas via API para gerar o relatório de economia.
        </div>
      ) : null}

      {summary && !report?.insufficient_data ? (
        <>
          <div className="card-grid roi-cards">
            <Card
              title="Economia no período"
              value={money(summary.savings_usd)}
              hint={`${summary.savings_pct.toFixed(1)}% vs ${report.baseline_model}`}
              badge={<StatusBadge label="ROI" tone={summary.savings_pct > 0 ? "ok" : "warn"} />}
            />
            <Card title="Custo real (router)" value={money(summary.actual_cost_usd)} hint={`${summary.query_count} consultas`} />
            <Card title="Custo baseline" value={money(summary.baseline_cost_usd)} hint="Contrafactual premium" />
            <Card
              title="Projeção mensal"
              value={money(summary.projected_monthly_savings_usd)}
              hint="Extrapolação linear"
            />
            <Card
              title="Custo / resposta útil"
              value={money(summary.cost_per_acceptable_actual_usd)}
              hint={`Baseline: ${money(summary.cost_per_acceptable_baseline_usd)} · limiar ${report.quality_threshold}`}
            />
          </div>

          <div className="chart-box">
            <h3>Custo diário: router vs baseline</h3>
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={report.daily_series}>
                <CartesianGrid stroke="#243041" strokeDasharray="3 3" />
                <XAxis dataKey="date" stroke="#8fa3bf" fontSize={11} />
                <YAxis stroke="#8fa3bf" fontSize={11} tickFormatter={(v) => `$${v}`} />
                <Tooltip
                  contentStyle={{ background: "#121820", border: "1px solid #2a3648", borderRadius: 8 }}
                  formatter={tooltipMoney}
                />
                <Legend />
                <Area type="monotone" dataKey="baseline_cost_usd" name="Baseline" stroke="#f87171" fill="#f8717133" />
                <Area type="monotone" dataKey="actual_cost_usd" name="Router" stroke="#34d399" fill="#34d39933" />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-box">
            <h3>Economia acumulada</h3>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={cumulative}>
                <CartesianGrid stroke="#243041" strokeDasharray="3 3" />
                <XAxis dataKey="date" stroke="#8fa3bf" fontSize={11} />
                <YAxis stroke="#8fa3bf" fontSize={11} tickFormatter={(v) => `$${v}`} />
                <Tooltip
                  contentStyle={{ background: "#121820", border: "1px solid #2a3648", borderRadius: 8 }}
                  formatter={(value) => [tooltipMoney(value)[0], "Acumulado"]}
                />
                <Bar dataKey="cumulative_savings_usd" fill="#60a5fa" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="table-wrap">
            <h3>Economia por modelo escolhido</h3>
            <table>
              <thead>
                <tr>
                  <th>Modelo</th>
                  <th>Consultas</th>
                  <th>Custo real</th>
                  <th>Baseline</th>
                  <th>Economia</th>
                  <th>Qualidade média</th>
                </tr>
              </thead>
              <tbody>
                {report.model_breakdown.map((row) => (
                  <tr key={row.model}>
                    <td>{row.model}</td>
                    <td>{row.count}</td>
                    <td>{money(row.actual_cost_usd)}</td>
                    <td>{money(row.baseline_cost_usd)}</td>
                    <td>{money(row.savings_usd)}</td>
                    <td>{row.quality_mean.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {report.methodology ? (
            <details className="roi-methodology">
              <summary>Metodologia e parâmetros</summary>
              <dl className="roi-methodology-grid">
                <dt>Baseline</dt>
                <dd>
                  {String(report.methodology.baseline_model || report.baseline_model)} —{" "}
                  {String(report.methodology.baseline_description || "contrafactual premium")}
                </dd>
                <dt>Tokens</dt>
                <dd>{String(report.methodology.token_estimation || "chars / 4")}</dd>
                <dt>Resposta aceitável</dt>
                <dd>{String(report.methodology.acceptable_definition || `quality ≥ ${report.quality_threshold}`)}</dd>
                <dt>Visão (surcharge)</dt>
                <dd>
                  ${String(report.methodology.vision_surcharge_usd ?? "0.004")} / requisição multimodal
                </dd>
                <dt>Env</dt>
                <dd>{(report.methodology.env_overrides as string[] | undefined)?.join(", ") || "ROI_BASELINE_MODEL, ROI_QUALITY_THRESHOLD"}</dd>
              </dl>
            </details>
          ) : null}

          {report.disclaimer ? <p className="roi-disclaimer muted">{report.disclaimer}</p> : null}
        </>
      ) : null}
    </div>
  );
}
