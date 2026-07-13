import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../api/client";
import { EmptyState, ErrorBanner, LoadingBlock, PageHeader } from "../components/ui";
import { useToast } from "../context/ToastContext";

type SettingMeta = {
  mutability?: string;
  domain?: string;
  description?: string;
};

type Catalog = { settings: Record<string, SettingMeta> };
type Snapshot = Record<string, unknown>;

const PRIORITY_KEYS = [
  "NSGA_W_QUALITY",
  "NSGA_W_LATENCY",
  "NSGA_W_COST",
  "BANDIT_EPSILON",
  "UNCERTAINTY_THRESHOLD",
  "CANDIDATE_MODELS_LIST",
  "CANDIDATE_VISION_MODELS_LIST",
  "CANDIDATE_MULTIMODAL_MODELS_LIST",
];

export function SettingsPage() {
  const qc = useQueryClient();
  const toast = useToast();
  const catalog = useQuery({
    queryKey: ["settings-catalog"],
    queryFn: () => apiFetch<Catalog>("/admin/settings/catalog"),
  });
  const snapshot = useQuery({
    queryKey: ["settings-snapshot"],
    queryFn: () => apiFetch<Snapshot>("/admin/settings"),
  });

  const [selectedKey, setSelectedKey] = useState("");
  const [value, setValue] = useState("");
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");

  const runtimeKeys = useMemo(() => {
    const keys = Object.entries(catalog.data?.settings || {})
      .filter(([, meta]) => meta.mutability === "runtime_safe")
      .map(([key]) => key);
    const ordered = [
      ...PRIORITY_KEYS.filter((k) => keys.includes(k)),
      ...keys.filter((k) => !PRIORITY_KEYS.includes(k)).sort(),
    ];
    return ordered;
  }, [catalog.data]);

  const filteredKeys = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return runtimeKeys;
    return runtimeKeys.filter((k) => k.toLowerCase().includes(q));
  }, [runtimeKeys, search]);

  const selectedMeta = selectedKey ? catalog.data?.settings?.[selectedKey] : undefined;

  const save = useMutation({
    mutationFn: (payload: { key: string; raw: string }) => {
      let parsed: unknown = payload.raw;
      const trimmed = payload.raw.trim();
      if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
        try {
          parsed = JSON.parse(trimmed);
        } catch {
          throw new Error("JSON inválido para este valor.");
        }
      }
      return apiFetch("/admin/settings", {
        method: "PUT",
        body: JSON.stringify({ settings: { [payload.key]: parsed } }),
      });
    },
    onSuccess: () => {
      setError("");
      qc.invalidateQueries({ queryKey: ["settings-snapshot"] });
      toast.push(`Configuração ${selectedKey} aplicada.`, "success");
    },
    onError: (err) => {
      const msg = err instanceof Error ? err.message : "Falha ao salvar";
      setError(msg);
      toast.push(msg, "error");
    },
  });

  const onSelect = (key: string) => {
    setSelectedKey(key);
    setError("");
    const current = snapshot.data?.[key];
    if (current === undefined || current === null) {
      setValue("");
      return;
    }
    setValue(typeof current === "string" ? current : JSON.stringify(current));
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!selectedKey) return;
    save.mutate({ key: selectedKey, raw: value });
  };

  return (
    <div>
      <PageHeader
        title="Configurações runtime"
        description="Alterações aplicadas sem restart. Apenas chaves marcadas como runtime_safe no catálogo."
      />

      {(catalog.isError || snapshot.isError) && (
        <ErrorBanner
          message="Falha ao carregar configurações."
          onRetry={() => {
            catalog.refetch();
            snapshot.refetch();
          }}
        />
      )}

      <div className="settings-grid">
        <div>
          <input
            className="settings-search"
            placeholder="Buscar chave..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label="Buscar configuração"
          />
          {catalog.isLoading ? (
            <LoadingBlock label="Carregando catálogo..." />
          ) : filteredKeys.length === 0 ? (
            <EmptyState title="Nenhuma chave encontrada" description="Ajuste o filtro de busca." />
          ) : (
            <div className="settings-list" role="listbox" aria-label="Chaves runtime">
              {filteredKeys.map((key) => (
                <button
                  key={key}
                  type="button"
                  role="option"
                  aria-selected={selectedKey === key}
                  title={catalog.data?.settings?.[key]?.description || key}
                  className={selectedKey === key ? "settings-key active" : "settings-key"}
                  onClick={() => onSelect(key)}
                >
                  {key}
                </button>
              ))}
            </div>
          )}
        </div>

        <form className="settings-editor form-card" onSubmit={onSubmit}>
          {selectedKey && selectedMeta ? (
            <dl className="settings-meta">
              {selectedMeta.domain ? (
                <>
                  <dt>Domínio</dt>
                  <dd>{selectedMeta.domain}</dd>
                </>
              ) : null}
              {selectedMeta.description ? (
                <>
                  <dt>Descrição</dt>
                  <dd>{selectedMeta.description}</dd>
                </>
              ) : null}
              <dt>Mutabilidade</dt>
              <dd>{selectedMeta.mutability || "—"}</dd>
            </dl>
          ) : null}

          <label>
            Chave
            <input value={selectedKey} readOnly placeholder="Selecione uma chave na lista" />
          </label>
          <label>
            Valor
            <textarea
              className="chip-input"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              placeholder='Ex: 0.5 ou ["ollama/phi4:latest"]'
              disabled={!selectedKey}
            />
          </label>
          {error ? (
            <div className="error-text" role="alert">
              {error}
            </div>
          ) : null}
          <button type="submit" disabled={!selectedKey || save.isPending}>
            {save.isPending ? "Salvando..." : "Aplicar"}
          </button>
        </form>
      </div>
    </div>
  );
}
