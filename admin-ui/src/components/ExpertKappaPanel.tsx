import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Link } from "react-router-dom";
import { apiFetch } from "../api/client";
import { Card, ErrorBanner, LoadingBlock, StatusBadge } from "../components/ui";

type KappaTheme = {
  theme_id: string;
  theme_title: string;
  kappa: number | null;
  n: number;
  observed_agreement?: number | null;
};

type ExpertKappaDashboard = {
  global_kappa: number | null;
  global_n: number;
  mean_absolute_error?: number | null;
  total_assessments: number;
  active_experts: number;
  themes_reviewed: number;
  by_theme: KappaTheme[];
  insufficient_data: boolean;
};

function kappaTone(kappa: number | null | undefined): "ok" | "warn" | "error" | "neutral" {
  if (kappa == null) return "neutral";
  if (kappa >= 0.6) return "ok";
  if (kappa >= 0.4) return "warn";
  return "error";
}

function formatKappa(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toFixed(3);
}

export function ExpertKappaPanel() {
  const kappa = useQuery({
    queryKey: ["dashboard-expert-kappa"],
    queryFn: () => apiFetch<ExpertKappaDashboard>("/admin/dashboard/expert-kappa"),
    refetchInterval: 30000,
  });

  const chartData = useMemo(
    () =>
      (kappa.data?.by_theme || [])
        .filter((row) => row.kappa != null && row.n > 0)
        .map((row) => ({
          theme: row.theme_title.length > 18 ? `${row.theme_title.slice(0, 16)}…` : row.theme_title,
          kappa: Number(row.kappa),
          n: row.n,
        })),
    [kappa.data]
  );

  if (kappa.isLoading) {
    return <LoadingBlock label="Carregando concordância judge × especialista..." />;
  }

  if (kappa.isError) {
    return (
      <ErrorBanner
        message="Não foi possível carregar métricas κ por tema."
        onRetry={() => kappa.refetch()}
      />
    );
  }

  const data = kappa.data;
  if (!data) return null;

  return (
    <section className="expert-kappa-panel">
      <div className="panel-section-head">
        <div>
          <h3>Concordância Judge × Especialista (κ)</h3>
          <p className="page-description">
            Cohen&apos;s kappa por tema — quanto maior, melhor o alinhamento entre avaliação automática e humana.
          </p>
        </div>
        <Link to="/experts" className="btn-sm secondary">
          Gerenciar especialistas
        </Link>
      </div>

      <div className="card-grid kappa-cards">
        <Card
          title="κ global"
          value={formatKappa(data.global_kappa)}
          hint={`${data.global_n} pares com judge`}
          badge={
            <StatusBadge
              label={data.insufficient_data ? "dados insuficientes" : kappaTone(data.global_kappa) === "ok" ? "bom" : "atenção"}
              tone={kappaTone(data.global_kappa)}
            />
          }
        />
        <Card title="Avaliações" value={data.total_assessments} hint={`${data.active_experts} especialista(s)`} />
        <Card title="Temas revisados" value={data.themes_reviewed} />
        <Card
          title="MAE médio"
          value={data.mean_absolute_error != null ? data.mean_absolute_error.toFixed(2) : "—"}
          hint="Diferença média |humano − judge|"
        />
      </div>

      {data.by_theme.length === 0 ? (
        <div className="chart-placeholder">
          Nenhuma avaliação de especialista ainda. Convide revisores pelo{" "}
          <Link to="/expert/login">portal dedicado</Link>.
        </div>
      ) : (
        <>
          {chartData.length > 0 ? (
            <div className="chart-box">
              <h4>κ por tema</h4>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 16 }}>
                  <CartesianGrid stroke="#243041" strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 1]} stroke="#8fa3bf" fontSize={11} />
                  <YAxis type="category" dataKey="theme" width={120} stroke="#8fa3bf" fontSize={11} />
                  <Tooltip
                    contentStyle={{ background: "#121820", border: "1px solid #2a3648", borderRadius: 8 }}
                    formatter={(value) => [Number(value ?? 0).toFixed(3), "κ"]}
                  />
                  <Bar dataKey="kappa" fill="#a78bfa" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : null}

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Tema</th>
                  <th>κ</th>
                  <th>Pares (n)</th>
                  <th>Acordo observado</th>
                </tr>
              </thead>
              <tbody>
                {data.by_theme.map((row) => (
                  <tr key={row.theme_id}>
                    <td>
                      <strong>{row.theme_title}</strong>
                      <div className="muted">{row.theme_id}</div>
                    </td>
                    <td>
                      <StatusBadge label={formatKappa(row.kappa)} tone={kappaTone(row.kappa)} />
                    </td>
                    <td>{row.n}</td>
                    <td>{row.observed_agreement != null ? row.observed_agreement.toFixed(3) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
