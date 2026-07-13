import { Link } from "react-router-dom";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { apiFetch } from "../api/client";
import { Card, ErrorBanner, LoadingBlock } from "../components/ui";

type RoiSummary = {
  insufficient_data: boolean;
  baseline_model: string;
  summary: {
    savings_usd: number;
    savings_pct: number;
    actual_cost_usd: number;
    projected_monthly_savings_usd?: number;
    query_count: number;
  };
  daily_series?: Array<{ date: string; savings_usd: number }>;
};

export function RoiSummaryPanel() {
  const roi = useQuery({
    queryKey: ["dashboard-roi-summary"],
    queryFn: () => apiFetch<RoiSummary>("/admin/dashboard/roi?days=30"),
    refetchInterval: 60000,
  });

  const sparkline = useMemo(
    () =>
      (roi.data?.daily_series || []).map((row) => ({
        date: row.date.slice(5),
        savings: row.savings_usd,
      })),
    [roi.data]
  );

  if (roi.isLoading) return <LoadingBlock label="Calculando ROI..." />;
  if (roi.isError) return <ErrorBanner message="ROI indisponível." onRetry={() => roi.refetch()} />;
  if (!roi.data || roi.data.insufficient_data) {
    return (
      <section className="roi-summary-panel">
        <div className="panel-section-head">
          <h3>Economia vs baseline</h3>
          <Link to="/roi" className="btn-sm secondary">
            Configurar
          </Link>
        </div>
        <p className="muted">Sem consultas no período — gere tráfego na API para ver economia projetada.</p>
      </section>
    );
  }

  const s = roi.data.summary;
  return (
    <section className="roi-summary-panel">
      <div className="panel-section-head">
        <div>
          <h3>Economia vs {roi.data.baseline_model}</h3>
          <p className="page-description">Últimos 30 dias · {s.query_count} consultas</p>
        </div>
        <Link to="/roi" className="btn-sm secondary">
          Ver ROI completo
        </Link>
      </div>
      <div className="card-grid kappa-cards">
        <Card title="Economia" value={`$${s.savings_usd.toFixed(2)}`} hint={`${s.savings_pct.toFixed(1)}%`} />
        <Card title="Gasto router" value={`$${s.actual_cost_usd.toFixed(2)}`} />
        <Card title="Projeção / mês" value={`$${(s.projected_monthly_savings_usd ?? 0).toFixed(2)}`} />
      </div>
      {sparkline.length > 1 ? (
        <div className="chart-box roi-sparkline">
          <h4>Economia diária (USD)</h4>
          <ResponsiveContainer width="100%" height={120}>
            <AreaChart data={sparkline}>
              <XAxis dataKey="date" stroke="#8fa3bf" fontSize={10} />
              <YAxis stroke="#8fa3bf" fontSize={10} tickFormatter={(v) => `$${v}`} width={48} />
              <Tooltip
                contentStyle={{ background: "#121820", border: "1px solid #2a3648", borderRadius: 8 }}
                formatter={(value) => [`$${Number(value ?? 0).toFixed(4)}`, "Economia"]}
              />
              <Area type="monotone" dataKey="savings" stroke="#60a5fa" fill="#60a5fa33" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      ) : null}
    </section>
  );
}
