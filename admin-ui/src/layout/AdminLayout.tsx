import { useState } from "react";
import { NavLink, Outlet, useNavigate } from "react-router-dom";

const ADMIN_TOKEN_KEY = "iag_admin_token";

const NAV_ITEMS: Array<{ to: string; end?: boolean; label: string; icon: string }> = [
  { to: "/", end: true, label: "Dashboard", icon: "▣" },
  { to: "/logs", label: "Logs", icon: "☰" },
  { to: "/budgets", label: "Orçamentos", icon: "$" },
  { to: "/models", label: "Modelos", icon: "◇" },
  { to: "/roi", label: "ROI / Economia", icon: "◈" },
  { to: "/experts", label: "Especialistas", icon: "✎" },
  { to: "/settings", label: "Configurações", icon: "⚙" },
];

export function AdminLayout() {
  const navigate = useNavigate();
  const [menuOpen, setMenuOpen] = useState(false);

  const logout = () => {
    localStorage.removeItem(ADMIN_TOKEN_KEY);
    navigate("/login");
  };

  const closeMenu = () => setMenuOpen(false);

  return (
    <div className="layout">
      <div
        className={menuOpen ? "sidebar-overlay visible" : "sidebar-overlay"}
        onClick={closeMenu}
        aria-hidden="true"
      />
      <aside className={menuOpen ? "sidebar open" : "sidebar"} aria-label="Navegação principal">
        <div className="sidebar-brand">
          <h1>IAG Router Admin</h1>
          <p>Console operacional</p>
        </div>
        <nav aria-label="Seções">
          {NAV_ITEMS.map((item) => (
            <NavLink key={item.to} to={item.to} end={item.end} onClick={closeMenu}>
              <span className="nav-icon" aria-hidden="true">
                {item.icon}
              </span>
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <button type="button" className="secondary" onClick={logout}>
            Sair
          </button>
        </div>
      </aside>
      <div>
        <div className="mobile-topbar">
          <button type="button" className="btn-ghost btn-sm" onClick={() => setMenuOpen((v) => !v)} aria-expanded={menuOpen}>
            Menu
          </button>
          <strong>IAG Router Admin</strong>
          <span />
        </div>
        <main className="content" aria-label="Conteúdo principal">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
