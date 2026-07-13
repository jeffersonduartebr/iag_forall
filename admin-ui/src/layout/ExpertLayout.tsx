import { Outlet, useNavigate } from "react-router-dom";

export function ExpertLayout() {
  const navigate = useNavigate();

  const logout = () => {
    localStorage.removeItem("iag_expert_token");
    navigate("/expert/login");
  };

  return (
    <div className="layout expert-layout">
      <aside className="sidebar expert-sidebar" aria-label="Portal do especialista">
        <div className="sidebar-brand">
          <h1>Revisão Especializada</h1>
          <p>Gold standard humano</p>
        </div>
        <nav aria-label="Seções">
          <span className="nav-active-item">
            <span className="nav-icon" aria-hidden="true">
              ✎
            </span>
            Análise por consulta
          </span>
        </nav>
        <div className="sidebar-footer">
          <button type="button" className="secondary" onClick={logout}>
            Sair
          </button>
        </div>
      </aside>
      <main className="content expert-content" aria-label="Área de revisão">
        <Outlet />
      </main>
    </div>
  );
}
