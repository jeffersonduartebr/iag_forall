import { Navigate, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AdminLayout } from "./layout/AdminLayout";
import { LoginPage } from "./pages/LoginPage";
import { DashboardPage } from "./pages/DashboardPage";
import { LogsPage } from "./pages/LogsPage";
import { BudgetsPage } from "./pages/BudgetsPage";
import { ModelsPage } from "./pages/ModelsPage";
import { SettingsPage } from "./pages/SettingsPage";
import { ExpertReviewPage } from "./pages/ExpertReviewPage";
import { ExpertsManagePage } from "./pages/ExpertsManagePage";
import { RoiPage } from "./pages/RoiPage";
import { ExpertLoginPage } from "./pages/ExpertLoginPage";
import { ExpertLayout } from "./layout/ExpertLayout";
import { getAdminToken, getExpertToken } from "./api/client";
import { ToastProvider } from "./context/ToastContext";
import "./styles.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      staleTime: 3000,
      refetchOnWindowFocus: false,
    },
  },
});

function RequireAuth({ children }: { children: React.ReactNode }) {
  if (!getAdminToken()) {
    return <Navigate to="/login" replace />;
  }
  return <>{children}</>;
}

function RequireExpertAuth({ children }: { children: React.ReactNode }) {
  if (!getExpertToken()) {
    return <Navigate to="/expert/login" replace />;
  }
  return <>{children}</>;
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/expert/login" element={<ExpertLoginPage />} />
          <Route
            path="/expert"
            element={
              <RequireExpertAuth>
                <ExpertLayout />
              </RequireExpertAuth>
            }
          >
            <Route index element={<ExpertReviewPage />} />
          </Route>
          <Route
            path="/"
            element={
              <RequireAuth>
                <AdminLayout />
              </RequireAuth>
            }
          >
            <Route index element={<DashboardPage />} />
            <Route path="logs" element={<LogsPage />} />
            <Route path="budgets" element={<BudgetsPage />} />
            <Route path="models" element={<ModelsPage />} />
            <Route path="settings" element={<SettingsPage />} />
            <Route path="experts" element={<ExpertsManagePage />} />
            <Route path="roi" element={<RoiPage />} />
          </Route>
        </Routes>
      </ToastProvider>
    </QueryClientProvider>
  );
}
