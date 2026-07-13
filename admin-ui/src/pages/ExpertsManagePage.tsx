import { useState } from "react";
import type { FormEvent } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "../api/client";
import { EmptyState, ErrorBanner, LoadingBlock, PageHeader, StatusBadge } from "../components/ui";
import { useToast } from "../context/ToastContext";

type ExpertAccount = {
  id: number;
  email: string;
  display_name: string;
  phone?: string | null;
  enabled: boolean | number;
  theme_ids?: string[];
  created_at?: string;
};

export function ExpertsManagePage() {
  const qc = useQueryClient();
  const toast = useToast();
  const [displayName, setDisplayName] = useState("");
  const [email, setEmail] = useState("");
  const [phone, setPhone] = useState("");
  const [password, setPassword] = useState("");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [editEnabled, setEditEnabled] = useState(true);

  const accounts = useQuery({
    queryKey: ["expert-accounts"],
    queryFn: () => apiFetch<{ items: ExpertAccount[] }>("/admin/experts/accounts"),
  });

  const createAccount = useMutation({
    mutationFn: () =>
      apiFetch("/admin/experts/accounts", {
        method: "POST",
        body: JSON.stringify({
          display_name: displayName.trim(),
          email: email.trim(),
          phone: phone.trim() || null,
          password,
        }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["expert-accounts"] });
      toast.push(`Especialista ${email} cadastrado.`, "success");
      setDisplayName("");
      setEmail("");
      setPhone("");
      setPassword("");
      setSelectedId(null);
    },
    onError: (err) => toast.push(err instanceof Error ? err.message : "Falha ao cadastrar", "error"),
  });

  const updateAccount = useMutation({
    mutationFn: () =>
      apiFetch(`/admin/experts/accounts/${selectedId}`, {
        method: "PUT",
        body: JSON.stringify({
          display_name: displayName.trim(),
          phone: phone.trim() || null,
          password: password.trim() || undefined,
          enabled: editEnabled,
        }),
      }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["expert-accounts"] });
      toast.push("Especialista atualizado.", "success");
      setPassword("");
    },
    onError: (err) => toast.push(err instanceof Error ? err.message : "Falha ao atualizar", "error"),
  });

  const loadAccount = (account: ExpertAccount) => {
    setSelectedId(account.id);
    setDisplayName(account.display_name);
    setEmail(account.email);
    setPhone(account.phone || "");
    setPassword("");
    setEditEnabled(Boolean(account.enabled));
  };

  const resetForm = () => {
    setSelectedId(null);
    setDisplayName("");
    setEmail("");
    setPhone("");
    setPassword("");
    setEditEnabled(true);
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!displayName.trim() || !email.trim()) {
      toast.push("Nome e e-mail são obrigatórios.", "error");
      return;
    }
    if (selectedId) {
      updateAccount.mutate();
      return;
    }
    if (password.length < 8) {
      toast.push("Senha inicial deve ter pelo menos 8 caracteres.", "error");
      return;
    }
    createAccount.mutate();
  };

  const items = accounts.data?.items || [];

  return (
    <div>
      <PageHeader
        title="Especialistas"
        description="Cadastre revisores humanos. Eles acessam o portal em /expert/login com e-mail e senha, escolhem a área e avaliam consultas."
        actions={
          <Link to="/expert/login" className="btn-sm secondary" target="_blank" rel="noreferrer">
            Abrir portal do especialista
          </Link>
        }
      />

      {accounts.isError ? (
        <ErrorBanner message="Falha ao carregar especialistas." onRetry={() => accounts.refetch()} />
      ) : null}

      <form className="form-card" onSubmit={onSubmit}>
        <div className="form-row">
          <label className="field">
            Nome completo
            <input placeholder="Dr. Maria Silva" value={displayName} onChange={(e) => setDisplayName(e.target.value)} required />
          </label>
          <label className="field">
            E-mail (login)
            <input
              type="email"
              placeholder="maria@universidade.edu"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              disabled={selectedId != null}
            />
          </label>
          <label className="field">
            Telefone
            <input placeholder="+55 11 99999-9999" value={phone} onChange={(e) => setPhone(e.target.value)} />
          </label>
          <label className="field">
            {selectedId ? "Nova senha (opcional)" : "Senha inicial"}
            <input
              type="password"
              placeholder={selectedId ? "Deixe vazio para manter" : "mínimo 8 caracteres"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required={!selectedId}
              minLength={selectedId ? 0 : 8}
            />
          </label>
          {selectedId ? (
            <label className="checkbox-field">
              <input type="checkbox" checked={editEnabled} onChange={(e) => setEditEnabled(e.target.checked)} />
              Conta ativa
            </label>
          ) : null}
          <button type="submit" disabled={createAccount.isPending || updateAccount.isPending}>
            {createAccount.isPending || updateAccount.isPending
              ? "Salvando..."
              : selectedId
                ? "Atualizar"
                : "Cadastrar especialista"}
          </button>
          {selectedId ? (
            <button type="button" className="secondary" onClick={resetForm}>
              Novo cadastro
            </button>
          ) : null}
        </div>
      </form>

      {accounts.isLoading ? <LoadingBlock label="Carregando especialistas..." /> : null}

      {!accounts.isLoading && items.length === 0 ? (
        <EmptyState
          title="Nenhum especialista cadastrado"
          description="Use o formulário acima para registrar o primeiro revisor. Ele receberá e-mail e senha para acessar o portal."
        />
      ) : null}

      {items.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Nome</th>
                <th>E-mail</th>
                <th>Telefone</th>
                <th>Áreas</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {items.map((account) => {
                const isSelected = selectedId === account.id;
                return (
                  <tr
                    key={account.id}
                    className={isSelected ? "clickable selected" : "clickable"}
                    onClick={() => loadAccount(account)}
                    title="Clique para editar"
                  >
                    <td>{account.display_name}</td>
                    <td>{account.email}</td>
                    <td>{account.phone || "—"}</td>
                    <td>{(account.theme_ids || []).length || "—"}</td>
                    <td>
                      <StatusBadge
                        label={account.enabled ? "ativo" : "inativo"}
                        tone={account.enabled ? "ok" : "neutral"}
                      />
                    </td>
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
