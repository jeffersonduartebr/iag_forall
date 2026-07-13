import { test, expect } from "@playwright/test";

const username = process.env.ADMIN_UI_USERNAME || "jefferson.silva";
const password = process.env.ADMIN_UI_PASSWORD || "abc@123";

test.describe("Admin console", () => {
  test("login and navigate main sections", async ({ page }) => {
    await page.goto("/login");
    await page.getByLabel("Usuário").fill(username);
    await page.getByLabel("Senha").fill(password);
    await page.getByRole("button", { name: "Entrar" }).click();

    await expect(page).toHaveURL(/\/?$/);
    await expect(page.getByRole("heading", { name: "Dashboard" })).toBeVisible();

    await page.getByRole("link", { name: "Logs" }).click();
    await expect(page.getByRole("heading", { name: "Logs em tempo real" })).toBeVisible();

    await page.getByRole("link", { name: "Orçamentos" }).click();
    await expect(page.getByRole("heading", { name: "Orçamentos por tenant" })).toBeVisible();

    await page.getByRole("link", { name: "Modelos" }).click();
    await expect(page.getByRole("heading", { name: "Exploração OpenRouter" })).toBeVisible();
    await expect(page.getByRole("heading", { name: "Modelos candidatos" })).toBeVisible();

    await page.getByRole("link", { name: "Configurações" }).click();
    await expect(page.getByRole("heading", { name: "Configurações runtime" })).toBeVisible();
  });

  test("dashboard auto-refresh area visible", async ({ page }) => {
    await page.goto("/login");
    await page.getByLabel("Usuário").fill(username);
    await page.getByLabel("Senha").fill(password);
    await page.getByRole("button", { name: "Entrar" }).click();
    await expect(page.getByText("Atualização")).toBeVisible();
    await expect(page.getByText("5s")).toBeVisible();
    await expect(page.getByText("QPS (aprox.)")).toBeVisible();
  });
});
