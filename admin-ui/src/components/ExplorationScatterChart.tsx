import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import type { ExplorationStatus } from "./OpenRouterExplorationPanel";

type Point = {
  model: string;
  reward: number;
  cost: number;
  samples: number;
};

type Props = {
  models: ExplorationStatus["models"];
};

export function ExplorationScatterChart({ models }: Props) {
  const points: Point[] = models
    .filter((m) => m.mean_observed_usd_per_1k != null && m.mean_reward != null)
    .map((m) => ({
      model: m.model.replace("openrouter/", ""),
      reward: m.mean_reward,
      cost: m.mean_observed_usd_per_1k ?? 0,
      samples: m.count,
    }));

  if (points.length === 0) {
    return <p className="card-hint">Sem dados suficientes para o gráfico custo × reward.</p>;
  }

  return (
    <div className="exploration-scatter-wrap">
      <ResponsiveContainer width="100%" height={280}>
        <ScatterChart margin={{ top: 12, right: 16, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            type="number"
            dataKey="cost"
            name="Custo"
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            label={{ value: "Custo observado ($/1K)", position: "insideBottom", offset: -2, fill: "var(--muted)" }}
          />
          <YAxis
            type="number"
            dataKey="reward"
            name="Reward"
            domain={[0, 1]}
            tick={{ fill: "var(--muted)", fontSize: 11 }}
            label={{ value: "Reward médio", angle: -90, position: "insideLeft", fill: "var(--muted)" }}
          />
          <ZAxis type="number" dataKey="samples" range={[40, 400]} />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const p = payload[0].payload as Point;
              return (
                <div className="chart-tooltip">
                  <strong>{p.model}</strong>
                  <div>Reward: {p.reward.toFixed(3)}</div>
                  <div>Custo: ${p.cost.toFixed(6)}/1K</div>
                  <div>Amostras: {p.samples}</div>
                </div>
              );
            }}
          />
          <Scatter name="Modelos" data={points} fill="var(--accent)" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
