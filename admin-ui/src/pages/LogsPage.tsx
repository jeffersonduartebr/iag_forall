import { useEffect, useMemo, useRef, useState } from "react";
import { getToken } from "../api/client";
import { EmptyState, PageHeader, StatusBadge } from "../components/ui";

type LogEntry = {
  level?: string;
  event?: string;
  message?: string;
  container?: string;
  timestamp?: number;
};

type ConnectionState = "connecting" | "live" | "reconnecting";

async function consumeLogStream(
  token: string,
  signal: AbortSignal,
  onEntry: (entry: LogEntry) => void,
): Promise<void> {
  const resp = await fetch("/admin/logs/stream", {
    headers: { Authorization: `Bearer ${token}` },
    signal,
  });
  if (!resp.ok || !resp.body) {
    throw new Error(`stream HTTP ${resp.status}`);
  }
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (!signal.aborted) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const part of parts) {
      const line = part.replace(/^data:\s*/, "").trim();
      if (!line) continue;
      try {
        onEntry(JSON.parse(line) as LogEntry);
      } catch {
        /* ignore malformed */
      }
    }
  }
}

export function LogsPage() {
  const [lines, setLines] = useState<LogEntry[]>([]);
  const [paused, setPaused] = useState(false);
  const [filter, setFilter] = useState("");
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const panelRef = useRef<HTMLDivElement>(null);
  const pausedRef = useRef(paused);

  useEffect(() => {
    pausedRef.current = paused;
  }, [paused]);

  useEffect(() => {
    const token = getToken();
    if (!token) return;

    const controller = new AbortController();
    let active = true;
    let backoffMs = 1000;

    const pushEntry = (entry: LogEntry) => {
      if (pausedRef.current) return;
      setLines((prev) => [...prev.slice(-500), entry]);
    };

    const connect = async () => {
      setConnection("connecting");
      while (active && !controller.signal.aborted) {
        try {
          setConnection("live");
          await consumeLogStream(token, controller.signal, pushEntry);
          if (!active) break;
          setConnection("reconnecting");
          pushEntry({ level: "warning", event: "stream_disconnected", message: "Reconectando..." });
        } catch {
          if (!active || controller.signal.aborted) break;
          setConnection("reconnecting");
          pushEntry({
            level: "warning",
            event: "stream_error",
            message: `Reconectando em ${backoffMs / 1000}s...`,
          });
          await new Promise((r) => setTimeout(r, backoffMs));
          backoffMs = Math.min(backoffMs * 2, 15000);
          continue;
        }
        backoffMs = 1000;
      }
    };

    connect();

    return () => {
      active = false;
      controller.abort();
    };
  }, []);

  const filteredLines = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return lines;
    return lines.filter((line) => {
      const text = `${line.container || ""} ${line.event || ""} ${line.message || ""}`.toLowerCase();
      return text.includes(q);
    });
  }, [lines, filter]);

  useEffect(() => {
    if (paused) return;
    const el = panelRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [filteredLines, paused]);

  const connectionBadge = (
    <StatusBadge
      label={
        connection === "live" ? "ao vivo" : connection === "connecting" ? "conectando" : "reconectando"
      }
      tone={connection === "live" ? "ok" : "warn"}
    />
  );

  return (
    <div>
      <PageHeader
        title="Logs em tempo real"
        description="Stream de eventos da aplicação com reconexão automática."
        actions={connectionBadge}
      />

      <div className="logs-toolbar">
        <input
          className="settings-search"
          style={{ maxWidth: 280, marginBottom: 0 }}
          placeholder="Filtrar logs..."
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          aria-label="Filtrar logs"
        />
        <button type="button" className="secondary btn-sm" onClick={() => setPaused((p) => !p)}>
          {paused ? "Retomar" : "Pausar"}
        </button>
        <button type="button" className="secondary btn-sm" onClick={() => setLines([])}>
          Limpar
        </button>
        <span className="logs-status">
          {filteredLines.length} linha(s)
          {filter ? ` (filtrado de ${lines.length})` : ""}
        </span>
      </div>

      <div
        className="logs-panel"
        ref={panelRef}
        role="log"
        aria-live={paused ? "off" : "polite"}
        aria-relevant="additions"
        aria-busy={connection === "connecting"}
      >
        {connection === "connecting" && lines.length === 0 ? (
          <div className="empty-state" style={{ border: "none", background: "transparent" }}>
            <strong>Conectando ao stream...</strong>
          </div>
        ) : null}

        {filteredLines.length === 0 && connection !== "connecting" ? (
          <EmptyState
            title={filter ? "Nenhum log corresponde ao filtro" : "Aguardando eventos"}
            description={paused ? "Stream pausado." : "Os logs aparecerão aqui em tempo real."}
          />
        ) : null}

        {filteredLines.map((line, idx) => (
          <div key={idx} className={`log-line ${line.level || ""}`}>
            [{line.container || "app"}] {line.event || line.message}
          </div>
        ))}
      </div>
    </div>
  );
}
