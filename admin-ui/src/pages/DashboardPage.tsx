import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from "recharts";
import { apiFetch } from "../api/client";
import { Card, ErrorBanner, LoadingBlock, PageHeader, StatusBadge } from "../components/ui";
import { ExpertKappaPanel } from "../components/ExpertKappaPanel";
import { RoiSummaryPanel } from "../components/RoiSummaryPanel";

type Summary = {
  health?: { status?: string };
  circuit_breakers?: { open?: string[]; total?: number };
  prometheus_available?: boolean;
  usage?: { tenants?: Record<string, unknown> };
};

function extractSeries(data: unknown, key: string): { t: number; v: number }[] {
  const series = (data as { series?: Record<string, { result?: Array<{ values?: [number, string][] }> }> })?.series?.[key];
  const result = series?.result?.[0]?.values || [];
  return result.map(([t, v]) => ({ t: t * 1000, v: parseFloat(v) || 0 }));
}

function formatRelativeTime(ms: number): string {
  const delta = Math.max(0, Math.floor((Date.now() - ms) / 1000));
  if (delta < 5) return "agora";
  if (delta < 60) return `há ${delta}s`;
  return `há ${Math.floor(delta / 60)}min`;
}

function healthTone(status?: string): "ok" | "warn" | "error" | "neutral" {
  const s = (status || "").toLowerCase();
  if (s.includes("ok") || s.includes("healthy") || s.includes("up")) return "ok";
  if (s.includes("degrad")) return "warn";
  if (s.includes("fail") || s.includes("down")) return "error";
  return "neutral";
}

export function DashboardPage() {
  const summary = useQuery({
    queryKey: ["dashboard-summary"],
    queryFn: () => apiFetch<Summary>("/admin/dashboard/summary"),
    refetchInterval: 5000,
  });

  const series = useQuery({
    queryKey: ["dashboard-series"],
    queryFn: () => apiFetch<unknown>("/admin/dashboard/series?window_s=1800&step=5s"),
    refetchInterval: 5000,
  });

  const qps = useMemo(() => extractSeries(series.data, "qps"), [series.data]);
  const latency = useMemo(() => extractSeries(series.data, "latency_p95"), [series.data]);

  const openBreakers = summary.data?.circuit_breakers?.open || [];
  const tenantCount = Object.keys(summary.data?.usage?.tenants || {}).length;
  const lastUpdate = Math.max(summary.dataUpdatedAt || 0, series.dataUpdatedAt || 0);

  const isLoading = summary.isLoading || series.isLoading;
  const hasError = summary.isError || series.isError;

  return (
    <div>
      <PageHeader
        title="Dashboard"
        description="Visão operacional do roteador com atualização automática a cada 5 segundos."
        actions={
          lastUpdate ? (
            <span className="badge badge-info">Atualizado {formatRelativeTime(lastUpdate)}</span>
          ) : null
        }
      />

      {hasError ? (
        <ErrorBanner
          message="Não foi possível carregar métricas do dashboard."
          onRetry={() => {
            summary.refetch();
            series.refetch();
          }}
        />
      ) : null}

      {isLoading ? <LoadingBlock label="Carregando métricas..." /> : null}

      <div className="card-grid">
        <Card
          title="Health"
          value={summary.data?.health?.status || "—"}
          badge={<StatusBadge label={summary.data?.health?.status || "—"} tone={healthTone(summary.data?.health?.status)} />}
        />
        <Card
          title="Circuit breakers"
          value={openBreakers.length}
          hint={openBreakers.length ? openBreakers.slice(0, 3).join(", ") : "Nenhum aberto"}
          badge={
            <StatusBadge
              label={openBreakers.length ? "atenção" : "ok"}
              tone={openBreakers.length ? "warn" : "ok"}
            />
          }
        />
        <Card
          title="Prometheus"
          value={summary.data?.prometheus_available ? "Conectado" : "Indisponível"}
          badge={
            <StatusBadge
              label={summary.data?.prometheus_available ? "OK" : "Off"}
              tone={summary.data?.prometheus_available ? "ok" : "neutral"}
            />
          }
        />
        <Card title="Tenants ativos" value={tenantCount} hint="Com uso registrado no período" />
        <Card title="Atualização" value="5s" hint={lastUpdate ? `Última: ${formatRelativeTime(lastUpdate)}` : "Aguardando dados"} />
      </div>

      <div className="chart-box">
        <h3>QPS (aprox.)</h3>
        {series.isLoading ? (
          <div className="chart-placeholder">
            <LoadingBlock label="Carregando série..." />
          </div>
        ) : qps.length === 0 ? (
          <div className="chart-placeholder">Sem dados de QPS no intervalo selecionado.</div>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={qps}>
              <CartesianGrid stroke="#243041" strokeDasharray="3 3" />
              <XAxis dataKey="t" tickFormatter={(v) => new Date(v).toLocaleTimeString()} stroke="#8fa3bf" fontSize={11} />
              <YAxis stroke="#8fa3bf" fontSize={11} />
              <Tooltip
                contentStyle={{ background: "#121820", border: "1px solid #2a3648", borderRadius: 8 }}
                labelFormatter={(v) => new Date(Number(v)).toLocaleString()}
              />
              <Line type="monotone" dataKey="v" stroke="#60a5fa" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="chart-box">
        <h3>Latência p95 (s)</h3>
        {series.isLoading ? (
          <div className="chart-placeholder">
            <LoadingBlock label="Carregando série..." />
          </div>
        ) : latency.length === 0 ? (
          <div className="chart-placeholder">Sem dados de latência no intervalo selecionado.</div>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={latency}>
              <CartesianGrid stroke="#243041" strokeDasharray="3 3" />
              <XAxis dataKey="t" tickFormatter={(v) => new Date(v).toLocaleTimeString()} stroke="#8fa3bf" fontSize={11} />
              <YAxis stroke="#8fa3bf" fontSize={11} />
              <Tooltip
                contentStyle={{ background: "#121820", border: "1px solid #2a3648", borderRadius: 8 }}
                labelFormatter={(v) => new Date(Number(v)).toLocaleString()}
              />
              <Line type="monotone" dataKey="v" stroke="#34d399" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <RoiSummaryPanel />
      <ExpertKappaPanel />
    </div>
  );
}
