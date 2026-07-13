export type Portal = "admin" | "expert";

const PORTAL_KEY = "iag_portal";
const ADMIN_TOKEN_KEY = "iag_admin_token";
const EXPERT_TOKEN_KEY = "iag_expert_token";

export function getPortal(): Portal {
  const value = localStorage.getItem(PORTAL_KEY);
  return value === "expert" ? "expert" : "admin";
}

export function setPortal(portal: Portal): void {
  localStorage.setItem(PORTAL_KEY, portal);
}

export function getToken(): string | null {
  return localStorage.getItem(getPortal() === "expert" ? EXPERT_TOKEN_KEY : ADMIN_TOKEN_KEY);
}

export function getAdminToken(): string | null {
  return localStorage.getItem(ADMIN_TOKEN_KEY);
}

export function getExpertToken(): string | null {
  return localStorage.getItem(EXPERT_TOKEN_KEY);
}

export function setAdminToken(token: string): void {
  localStorage.setItem(ADMIN_TOKEN_KEY, token);
  setPortal("admin");
}

export function setExpertToken(token: string): void {
  localStorage.setItem(EXPERT_TOKEN_KEY, token);
  setPortal("expert");
}

/** @deprecated use setAdminToken */
export function setToken(token: string): void {
  setAdminToken(token);
}

export function clearToken(): void {
  const portal = getPortal();
  if (portal === "expert") {
    localStorage.removeItem(EXPERT_TOKEN_KEY);
  } else {
    localStorage.removeItem(ADMIN_TOKEN_KEY);
  }
  localStorage.removeItem(PORTAL_KEY);
}

export function clearAllSessions(): void {
  localStorage.removeItem(ADMIN_TOKEN_KEY);
  localStorage.removeItem(EXPERT_TOKEN_KEY);
  localStorage.removeItem(PORTAL_KEY);
}

function loginRedirectPath(): string {
  return getPortal() === "expert" ? "/expert/login" : "/login";
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers = new Headers(init.headers || {});
  if (!headers.has("Content-Type") && init.body) {
    headers.set("Content-Type", "application/json");
  }
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  const resp = await fetch(path, { ...init, headers });
  if (resp.status === 401) {
    if (getPortal() === "expert") {
      localStorage.removeItem(EXPERT_TOKEN_KEY);
    } else {
      localStorage.removeItem(ADMIN_TOKEN_KEY);
    }
    window.location.href = loginRedirectPath();
    throw new Error("Não autorizado");
  }
  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(detail || `HTTP ${resp.status}`);
  }
  if (resp.status === 204) {
    return undefined as T;
  }
  return resp.json() as Promise<T>;
}

export async function login(username: string, password: string) {
  return apiFetch<{ access_token: string; username: string; portal?: string }>("/admin/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export async function expertLogin(email: string, password: string) {
  const resp = await fetch("/admin/auth/expert-login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(detail || `HTTP ${resp.status}`);
  }
  return resp.json() as Promise<{ access_token: string; username: string; portal: string }>;
}
