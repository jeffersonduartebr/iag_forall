import { useEffect, useState } from "react";
import type { FormEvent } from "react";
import { Link, Navigate, useNavigate } from "react-router-dom";
import { getAdminToken, login, setAdminToken } from "../api/client";

export function LoginPage() {
  const navigate = useNavigate();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (getAdminToken()) {
      navigate("/", { replace: true });
    }
  }, [navigate]);

  if (getAdminToken()) {
    return <Navigate to="/" replace />;
  }

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password) {
      setError("Informe usuário e senha.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const res = await login(username.trim(), password);
      setAdminToken(res.access_token);
      navigate("/");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Falha no login");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <form className="login-form" onSubmit={onSubmit}>
        <h1>Console Admin</h1>
        <p className="login-subtitle">Gestão do roteador LLM multiobjetivo</p>
        <label>
          Usuário
          <input
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoComplete="username"
            placeholder="seu.usuario"
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
          <Link to="/expert/login">Sou especialista — portal de revisão</Link>
        </p>
      </form>
    </div>
  );
}
