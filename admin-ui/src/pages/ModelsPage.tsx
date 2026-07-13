import { useEffect, useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../api/client";
import { EmptyState, ErrorBanner, LoadingBlock, PageHeader, StatusBadge } from "../components/ui";
import { OpenRouterExplorationPanel } from "../components/OpenRouterExplorationPanel";
import { useToast } from "../context/ToastContext";

type ModelsResponse = {
  candidates: { text: string[]; vision: string[]; multimodal: string[] };
};

type HealthItem = {
  model: string;
  circuit_state: string;
  temporarily_unavailable: boolean;
  configured: boolean;
};

type CandidateTab = "text" | "vision" | "multimodal";

const TAB_LABELS: Record<CandidateTab, string> = {
  text: "Texto",
  vision: "Visão",
  multimodal: "Multimodal",
};

function circuitTone(state: string): "ok" | "warn" | "error" | "neutral" {
  const s = state.toLowerCase();
  if (s === "closed" || s === "healthy") return "ok";
  if (s === "half-open" || s === "half_open") return "warn";
  if (s === "open") return "error";
  return "neutral";
}

export function ModelsPage() {
  const qc = useQueryClient();
  const toast = useToast();
  const [tab, setTab] = useState<CandidateTab>("text");
  const [lists, setLists] = useState<Record<CandidateTab, string>>({
    text: "",
    vision: "",
    multimodal: "",
  });

  const models = useQuery({
    queryKey: ["models"],
    queryFn: () => apiFetch<ModelsResponse>("/admin/models"),
  });

  const health = useQuery({
    queryKey: ["models-health"],
    queryFn: () => apiFetch<{ items: HealthItem[] }>("/admin/models/health"),
    refetchInterval: 5000,
  });

  useEffect(() => {
    if (!models.data?.candidates) return;
    setLists({
      text: models.data.candidates.text.join("\n"),
      vision: models.data.candidates.vision.join("\n"),
      multimodal: models.data.candidates.multimodal.join("\n"),
    });
  }, [models.data]);

  const save = useMutation({
    mutationFn: () => {
      const parse = (raw: string) => raw.split("\n").map((s) => s.trim()).filter(Boolean);
      return apiFetch("/admin/models/candidates", {
        method: "PUT",
        body: JSON.stringify({
          text: parse(lists.text),
          vision: parse(lists.vision),
          multimodal: parse(lists.multimodal),
        }),
      });
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["models"] });
      qc.invalidateQueries({ queryKey: ["models-health"] });
      toast.push("Lista de candidatos aplicada sem restart.", "success");
    },
    onError: (err) => {
      toast.push(err instanceof Error ? err.message : "Falha ao salvar modelos", "error");
    },
  });

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    save.mutate();
  };

  const addCandidateFromExploration = (model: string) => {
    const current = lists.text
      .split("\n")
      .map((s) => s.trim())
      .filter(Boolean);
    if (current.includes(model)) {
      toast.push(`${model} já está na lista de texto.`, "info");
      return;
    }
    setLists((prev) => ({
      ...prev,
      text: [...current, model].join("\n"),
    }));
    setTab("text");
    toast.push(`${model} adicionado à lista de texto — clique em Aplicar para persistir.`, "success");
  };

  const healthItems = health.data?.items || [];
  const modelCount = useMemo(
    () => lists[tab].split("\n").map((s) => s.trim()).filter(Boolean).length,
    [lists, tab],
  );

  return (
    <div>
      <OpenRouterExplorationPanel onAddCandidate={addCandidateFromExploration} />

      <PageHeader
        title="Modelos candidatos"
        description="Gerencie listas de roteamento por modalidade. Um modelo por linha (ex: ollama/phi4:latest, openrouter/openai/gpt-4o-mini)."
      />

      {models.isError ? (
        <ErrorBanner message="Falha ao carregar modelos." onRetry={() => models.refetch()} />
      ) : null}

      <form className="form-card" onSubmit={onSubmit}>
        <div className="tabs" role="tablist" aria-label="Modalidades">
          {(Object.keys(TAB_LABELS) as CandidateTab[]).map((key) => (
            <button
              key={key}
              type="button"
              role="tab"
              aria-selected={tab === key}
              className={tab === key ? "tab active" : "tab"}
              onClick={() => setTab(key)}
            >
              {TAB_LABELS[key]}
            </button>
          ))}
        </div>
        <p className="page-description" style={{ marginBottom: "0.75rem" }}>
          {modelCount} modelo(s) na lista de {TAB_LABELS[tab].toLowerCase()}.
        </p>
        {models.isLoading ? (
          <LoadingBlock label="Carregando candidatos..." />
        ) : (
          <textarea
            className="chip-input"
            value={lists[tab]}
            onChange={(e) => setLists((prev) => ({ ...prev, [tab]: e.target.value }))}
            aria-label={`Modelos ${TAB_LABELS[tab]}`}
          />
        )}
        <button type="submit" disabled={save.isPending || models.isLoading} style={{ marginTop: "0.75rem" }}>
          {save.isPending ? "Aplicando..." : "Aplicar sem restart"}
        </button>
      </form>

      <PageHeader title="Saúde dos modelos" description="Estado de circuit breaker e disponibilidade por modelo candidato." />

      {health.isError ? (
        <ErrorBanner message="Falha ao carregar saúde dos modelos." onRetry={() => health.refetch()} />
      ) : null}

      {health.isLoading ? <LoadingBlock label="Carregando saúde..." /> : null}

      {!health.isLoading && healthItems.length === 0 ? (
        <EmptyState title="Nenhum modelo candidato" description="Configure candidatos acima para monitorar saúde." />
      ) : null}

      {healthItems.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Modelo</th>
                <th>Circuit</th>
                <th>Configurado</th>
                <th>Indisponível</th>
              </tr>
            </thead>
            <tbody>
              {healthItems.map((h) => (
                <tr key={h.model}>
                  <td>
                    <code>{h.model}</code>
                  </td>
                  <td>
                    <StatusBadge label={h.circuit_state} tone={circuitTone(h.circuit_state)} />
                  </td>
                  <td>
                    <StatusBadge label={h.configured ? "sim" : "não"} tone={h.configured ? "ok" : "warn"} />
                  </td>
                  <td>
                    <StatusBadge
                      label={h.temporarily_unavailable ? "sim" : "não"}
                      tone={h.temporarily_unavailable ? "error" : "ok"}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
