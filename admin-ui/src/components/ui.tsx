import type { ReactNode } from "react";

export function PageHeader({
  title,
  description,
  actions,
}: {
  title: string;
  description?: string;
  actions?: ReactNode;
}) {
  return (
    <header className="page-header">
      <div>
        <h2>{title}</h2>
        {description ? <p className="page-description">{description}</p> : null}
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </header>
  );
}

export function LoadingBlock({ label = "Carregando..." }: { label?: string }) {
  return (
    <div className="loading-block" role="status" aria-busy="true">
      <div className="spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

export function ErrorBanner({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="banner banner-error" role="alert">
      <span>{message}</span>
      {onRetry ? (
        <button type="button" className="secondary btn-sm" onClick={onRetry}>
          Tentar novamente
        </button>
      ) : null}
    </div>
  );
}

export function EmptyState({ title, description }: { title: string; description?: string }) {
  return (
    <div className="empty-state">
      <strong>{title}</strong>
      {description ? <p>{description}</p> : null}
    </div>
  );
}

export function StatusBadge({
  label,
  tone = "neutral",
}: {
  label: string;
  tone?: "ok" | "warn" | "error" | "neutral" | "info";
}) {
  return <span className={`badge badge-${tone}`}>{label}</span>;
}

export function Card({
  title,
  value,
  hint,
  badge,
}: {
  title: string;
  value: ReactNode;
  hint?: string;
  badge?: ReactNode;
}) {
  return (
    <div className="card">
      <div className="card-top">
        <h3>{title}</h3>
        {badge}
      </div>
      <div className="value">{value}</div>
      {hint ? <p className="card-hint">{hint}</p> : null}
    </div>
  );
}
