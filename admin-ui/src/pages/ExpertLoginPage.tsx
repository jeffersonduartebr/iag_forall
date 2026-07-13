import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { expertLogin, getExpertToken, setExpertToken } from "../api/client";

export function ExpertLoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (getExpertToken()) {
      navigate("/expert", { replace: true });
    }
  }, [navigate]);

  if (getExpertToken()) {
    return <Navigate to="/expert" replace />;
  }

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!email.trim() || !password) {
      setError("Informe e-mail e senha.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const res = await expertLogin(email.trim(), password);
      setExpertToken(res.access_token);
      navigate("/expert");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha no login");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page expert-login-page">
      <form className="login-form" onSubmit={onSubmit}>
        <h1>Portal do Especialista</h1>
        <p className="login-subtitle">Use o e-mail e a senha fornecidos pelo administrador</p>
        <label>
          E-mail
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="username"
            placeholder="seu.email@instituicao.edu"
            required
          />
        </label>
        <label>
          Senha
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
            required
          />
        </label>
        {error ? (
          <div className="error-text" role="alert">
            {error}
          </div>
        ) : null}
        <button type="submit" disabled={loading}>
          {loading ? "Entrando..." : "Entrar"}
        </button>
        <p className="login-footer-link">
          <Link to="/login">Acesso administrativo</Link>
        </p>
      </form>
    </div>
  );
}
