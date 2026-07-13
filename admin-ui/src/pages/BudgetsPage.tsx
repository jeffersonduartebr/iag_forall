import { useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../api/client";
import { EmptyState, ErrorBanner, LoadingBlock, PageHeader, StatusBadge } from "../components/ui";
import { useToast } from "../context/ToastContext";

type Budget = {
  tenant_id: string;
  daily_usd_limit: number;
  monthly_usd_limit: number;
  enabled: boolean | number;
};

export function BudgetsPage() {
  const qc = useQueryClient();
  const toast = useToast();
  const [tenantId, setTenantId] = useState("");
  const [daily, setDaily] = useState("10");
  const [monthly, setMonthly] = useState("100");
  const [enabled, setEnabled] = useState(true);
  const [selectedTenant, setSelectedTenant] = useState<string | null>(null);

  const budgets = useQuery({
    queryKey: ["budgets"],
    queryFn: () => apiFetch<{ items: Budget[] }>("/admin/budgets"),
  });

  const usage = useQuery({
    queryKey: ["usage"],
    queryFn: () => apiFetch<{ tenants?: Record<string, { cost_usd?: number; requests?: number }> }>("/admin/quotas/usage"),
    refetchInterval: 10000,
  });

  const save = useMutation({
    mutationFn: () =>
      apiFetch(`/admin/budgets/${encodeURIComponent(tenantId)}`, {
        method: "PUT",
        body: JSON.stringify({
          daily_usd_limit: parseFloat(daily),
          monthly_usd_limit: parseFloat(monthly),
          enabled,
        }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["budgets"] });
      qc.invalidateQueries({ queryKey: ["usage"] });
      toast.push(`Orçamento salvo para ${tenantId}`, "success");
      setSelectedTenant(tenantId);
    },
    onError: (err) => {
      toast.push(err instanceof Error ? err.message : "Falha ao salvar orçamento", "error");
    },
  });

  const loadBudget = (b: Budget) => {
    setTenantId(b.tenant_id);
    setDaily(String(b.daily_usd_limit));
    setMonthly(String(b.monthly_usd_limit));
    setEnabled(Boolean(b.enabled));
    setSelectedTenant(b.tenant_id);
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!tenantId.trim()) {
      toast.push("Informe o tenant_id.", "error");
      return;
    }
    if (Number.isNaN(parseFloat(daily)) || Number.isNaN(parseFloat(monthly))) {
      toast.push("Limites diário e mensal devem ser numéricos.", "error");
      return;
    }
    save.mutate();
  };

  const items = budgets.data?.items || [];

  return (
    <div>
      <PageHeader
        title="Orçamentos por tenant"
        description="Defina limites de custo diário e mensal. Clique em uma linha da tabela para editar."
      />

      {budgets.isError ? (
        <ErrorBanner message="Falha ao carregar orçamentos." onRetry={() => budgets.refetch()} />
      ) : null}

      <form className="form-card" onSubmit={onSubmit}>
        <div className="form-row">
          <label className="field">
            Tenant ID
            <input placeholder="ex: acme-corp" value={tenantId} onChange={(e) => setTenantId(e.target.value)} />
          </label>
          <label className="field">
            Limite diário (USD)
            <input type="number" min="0" step="0.01" value={daily} onChange={(e) => setDaily(e.target.value)} />
          </label>
          <label className="field">
            Limite mensal (USD)
            <input type="number" min="0" step="0.01" value={monthly} onChange={(e) => setMonthly(e.target.value)} />
          </label>
          <label className="checkbox-field">
            <input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />
            Ativo
          </label>
          <button type="submit" disabled={save.isPending}>
            {save.isPending ? "Salvando..." : selectedTenant ? "Atualizar" : "Salvar"}
          </button>
        </div>
      </form>

      {budgets.isLoading ? <LoadingBlock label="Carregando orçamentos..." /> : null}

      {!budgets.isLoading && items.length === 0 ? (
        <EmptyState title="Nenhum orçamento configurado" description="Use o formulário acima para criar o primeiro limite por tenant." />
      ) : null}

      {items.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Tenant</th>
                <th>Daily (USD)</th>
                <th>Monthly (USD)</th>
                <th>Status</th>
                <th>Uso (USD)</th>
                <th>Requisições</th>
              </tr>
            </thead>
            <tbody>
              {items.map((b) => {
                const u = usage.data?.tenants?.[b.tenant_id];
                const isSelected = selectedTenant === b.tenant_id;
                return (
                  <tr
                    key={b.tenant_id}
                    className={isSelected ? "clickable selected" : "clickable"}
                    onClick={() => loadBudget(b)}
                    title="Clique para editar"
                  >
                    <td>{b.tenant_id}</td>
                    <td>{b.daily_usd_limit}</td>
                    <td>{b.monthly_usd_limit}</td>
                    <td>
                      <StatusBadge label={b.enabled ? "ativo" : "inativo"} tone={b.enabled ? "ok" : "neutral"} />
                    </td>
                    <td>{u?.cost_usd != null ? u.cost_usd.toFixed(4) : "—"}</td>
                    <td>{u?.requests ?? "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
