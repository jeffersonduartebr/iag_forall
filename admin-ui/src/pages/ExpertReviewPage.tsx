import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../api/client";
import { EmptyState, ErrorBanner, LoadingBlock, PageHeader, StatusBadge } from "../components/ui";
import { useToast } from "../context/ToastContext";

type Theme = { id: string; title: string; target_count: number };
type ExpertProfile = {
  user_id: string;
  display_name?: string;
  phone?: string;
  theme_ids: string[];
  credentials_note?: string;
};
type ReviewItem = {
  source: string;
  benchmark_id: string;
  theme: string;
  query_text: string;
  answer?: string;
  reference?: string;
  judge_quality?: number;
  eval_run_id?: string;
  split?: string;
  difficulty?: string;
};
type Rubric = {
  factual_correctness: number;
  task_completion: number;
  clarity_structure: number;
  scaffolding: number;
  audience_fit: number;
  misconception_handling: number;
};

const DEFAULT_RUBRIC: Rubric = {
  factual_correctness: 7,
  task_completion: 7,
  clarity_structure: 7,
  scaffolding: 7,
  audience_fit: 7,
  misconception_handling: 7,
};

export function ExpertReviewPage() {
  const qc = useQueryClient();
  const toast = useToast();
  const [selectedThemes, setSelectedThemes] = useState<string[]>([]);
  const [displayName, setDisplayName] = useState("");
  const [credentialsNote, setCredentialsNote] = useState("");
  const [setupDone, setSetupDone] = useState(false);
  const [evalRunId, setEvalRunId] = useState("");
  const [currentItem, setCurrentItem] = useState<ReviewItem | null>(null);
  const [answer, setAnswer] = useState("");
  const [judgeQuality, setJudgeQuality] = useState<number | null>(null);
  const [qualityScore, setQualityScore] = useState(7);
  const [rubric, setRubric] = useState<Rubric>(DEFAULT_RUBRIC);
  const [notes, setNotes] = useState("");

  const themes = useQuery({
    queryKey: ["expert-themes"],
    queryFn: () => apiFetch<{ items: Theme[] }>("/admin/experts/themes"),
  });

  const profile = useQuery({
    queryKey: ["expert-profile"],
    queryFn: () => apiFetch<ExpertProfile>("/admin/experts/profile"),
  });

  useEffect(() => {
    if (profile.data) {
      setSelectedThemes(profile.data.theme_ids || []);
      setDisplayName(profile.data.display_name || profile.data.user_id || "");
      setCredentialsNote(profile.data.credentials_note || "");
      if ((profile.data.theme_ids || []).length > 0) {
        setSetupDone(true);
      }
    }
  }, [profile.data]);

  const saveProfile = useMutation({
    mutationFn: () =>
      apiFetch<ExpertProfile>("/admin/experts/profile", {
        method: "PUT",
        body: JSON.stringify({
          display_name: displayName,
          theme_ids: selectedThemes,
          credentials_note: credentialsNote,
        }),
      }),
    onSuccess: (data) => {
      qc.setQueryData(["expert-profile"], data);
      setSetupDone(true);
      toast.push("Áreas de atuação salvas.", "success");
    },
    onError: (err) => toast.push(err instanceof Error ? err.message : "Falha ao salvar perfil", "error"),
  });

  const fetchNext = useMutation({
    mutationFn: () => {
      const params = new URLSearchParams({ split: "held_out" });
      if (evalRunId.trim()) params.set("eval_run_id", evalRunId.trim());
      return apiFetch<{ status: string; item: ReviewItem | null }>(`/admin/experts/next-item?${params}`);
    },
    onSuccess: (data) => {
      const item = data.item;
      setCurrentItem(item);
      setAnswer(item?.answer || "");
      setJudgeQuality(item?.judge_quality ?? null);
      setQualityScore(7);
      setRubric(DEFAULT_RUBRIC);
      setNotes("");
      if (!item) toast.push("Nenhum item pendente nas suas áreas.", "info");
    },
    onError: (err) => toast.push(err instanceof Error ? err.message : "Falha ao buscar item", "error"),
  });

  const previewAnswer = useMutation({
    mutationFn: () =>
      apiFetch<{ answer: string; judge_quality?: number; model?: string }>("/admin/experts/preview-answer", {
        method: "POST",
        body: JSON.stringify({
          query: currentItem?.query_text,
          theme: currentItem?.theme,
          benchmark_id: currentItem?.benchmark_id,
        }),
      }),
    onSuccess: (data) => {
      setAnswer(data.answer || "");
      setJudgeQuality(data.judge_quality ?? null);
      toast.push(`Resposta obtida (${data.model || "router"}).`, "success");
    },
    onError: (err) => toast.push(err instanceof Error ? err.message : "Falha ao obter resposta", "error"),
  });

  const submitAssessment = useMutation({
    mutationFn: () => {
      if (!currentItem) throw new Error("Nenhum item selecionado");
      if (!answer.trim()) throw new Error("Obtenha ou informe a resposta do sistema");
      return apiFetch("/admin/experts/assessments", {
        method: "POST",
        body: JSON.stringify({
          benchmark_id: currentItem.benchmark_id,
          theme: currentItem.theme,
          query_text: currentItem.query_text,
          answer,
          reference: currentItem.reference,
          eval_run_id: currentItem.eval_run_id || evalRunId.trim() || null,
          judge_quality: judgeQuality,
          quality_score: qualityScore,
          rubric,
          notes: notes || null,
        }),
      });
    },
    onSuccess: () => {
      toast.push("Análise registrada.", "success");
      fetchNext.mutate();
    },
    onError: (err) => toast.push(err instanceof Error ? err.message : "Falha ao enviar análise", "error"),
  });

  const themeMap = useMemo(() => {
    const map = new Map<string, Theme>();
    (themes.data?.items || []).forEach((t) => map.set(t.id, t));
    return map;
  }, [themes.data]);

  const toggleTheme = (id: string) => {
    setSelectedThemes((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  };

  if (profile.isLoading || themes.isLoading) {
    return <LoadingBlock label="Carregando portal de especialistas..." />;
  }

  if (profile.isError || themes.isError) {
    return <ErrorBanner message="Falha ao carregar dados do portal." onRetry={() => { profile.refetch(); themes.refetch(); }} />;
  }

  if (!setupDone) {
    return (
      <div>
        <PageHeader
          title="Portal do Especialista"
          description="Selecione suas áreas de atuação no catálogo acadêmico. Você só receberá consultas dos temas escolhidos."
        />
        <form
          className="form-card expert-setup"
          onSubmit={(e) => {
            e.preventDefault();
            if (!selectedThemes.length) {
              toast.push("Selecione ao menos uma área.", "error");
              return;
            }
            saveProfile.mutate();
          }}
        >
          <label className="field">
            Nome de exibição
            <input value={displayName} onChange={(e) => setDisplayName(e.target.value)} placeholder="Dr. Maria Silva" />
          </label>
          <label className="field">
            Credenciais / nota (opcional)
            <input value={credentialsNote} onChange={(e) => setCredentialsNote(e.target.value)} placeholder="Doutorado em História — UFRJ" />
          </label>
          <fieldset className="theme-grid">
            <legend>Áreas de atuação</legend>
            {(themes.data?.items || []).map((theme) => (
              <label key={theme.id} className="checkbox-field theme-chip">
                <input type="checkbox" checked={selectedThemes.includes(theme.id)} onChange={() => toggleTheme(theme.id)} />
                <span>
                  <strong>{theme.title}</strong>
                  <small>{theme.id} · {theme.target_count} itens</small>
                </span>
              </label>
            ))}
          </fieldset>
          <button type="submit" disabled={saveProfile.isPending}>
            {saveProfile.isPending ? "Salvando..." : "Começar revisão"}
          </button>
        </form>
      </div>
    );
  }

  return (
    <div className="expert-review-page">
      <PageHeader
        title="Revisão por consulta"
        description={`${displayName || profile.data?.display_name || profile.data?.user_id}${profile.data?.phone ? ` · ${profile.data.phone}` : ""} · ${selectedThemes.length} área(s) · split held-out`}
        actions={
          <button type="button" className="secondary" onClick={() => setSetupDone(false)}>
            Alterar áreas
          </button>
        }
      />

      <div className="form-card form-row expert-toolbar">
        <label className="field">
          Eval run (opcional)
          <input
            placeholder="eval_... — revisar respostas de um run"
            value={evalRunId}
            onChange={(e) => setEvalRunId(e.target.value)}
          />
        </label>
        <button type="button" onClick={() => fetchNext.mutate()} disabled={fetchNext.isPending}>
          {fetchNext.isPending ? "Buscando..." : "Próxima consulta"}
        </button>
      </div>

      {!currentItem ? (
        <EmptyState
          title="Nenhuma consulta carregada"
          description="Clique em “Próxima consulta” para receber um item do held-out nas suas áreas."
        />
      ) : (
        <div className="expert-review-grid">
          <section className="panel-card">
            <div className="panel-head">
              <h3>Consulta</h3>
              <StatusBadge label={currentItem.theme} tone="neutral" />
              {currentItem.split ? <StatusBadge label={currentItem.split} tone="ok" /> : null}
            </div>
            <p className="query-block">{currentItem.query_text}</p>
            {currentItem.reference ? (
              <details className="reference-box">
                <summary>Referência (gabarito)</summary>
                <p>{currentItem.reference}</p>
              </details>
            ) : null}
            <div className="meta-row">
              <span>ID: {currentItem.benchmark_id}</span>
              {currentItem.difficulty ? <span>Dificuldade: {currentItem.difficulty}</span> : null}
              {themeMap.get(currentItem.theme)?.title ? <span>{themeMap.get(currentItem.theme)?.title}</span> : null}
            </div>
          </section>

          <section className="panel-card">
            <div className="panel-head">
              <h3>Resposta do sistema</h3>
              {!answer ? (
                <button type="button" className="btn-sm" onClick={() => previewAnswer.mutate()} disabled={previewAnswer.isPending}>
                  {previewAnswer.isPending ? "Gerando..." : "Obter resposta"}
                </button>
              ) : null}
            </div>
            <textarea
              className="answer-area"
              rows={10}
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              placeholder="Clique em Obter resposta ou cole a resposta a ser avaliada."
            />
            {judgeQuality != null ? (
              <p className="muted">Nota do judge automático: <strong>{judgeQuality.toFixed(1)}</strong>/10</p>
            ) : null}
          </section>

          <section className="panel-card rubric-panel">
            <h3>Sua análise</h3>
            <label className="field">
              Nota global (0–10)
              <input
                type="number"
                min={0}
                max={10}
                step={0.5}
                value={qualityScore}
                onChange={(e) => setQualityScore(parseFloat(e.target.value))}
              />
            </label>
            {(Object.keys(rubric) as Array<keyof Rubric>).map((key) => (
              <label key={key} className="field rubric-field">
                {key.replace(/_/g, " ")}
                <input
                  type="range"
                  min={0}
                  max={10}
                  step={0.5}
                  value={rubric[key]}
                  onChange={(e) => setRubric((prev) => ({ ...prev, [key]: parseFloat(e.target.value) }))}
                />
                <span>{rubric[key].toFixed(1)}</span>
              </label>
            ))}
            <label className="field">
              Observações
              <textarea rows={3} value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Justificativa, erros observados, sugestões..." />
            </label>
            <button type="button" onClick={() => submitAssessment.mutate()} disabled={submitAssessment.isPending}>
              {submitAssessment.isPending ? "Enviando..." : "Emitir análise e próxima"}
            </button>
          </section>
        </div>
      )}
    </div>
  );
}
