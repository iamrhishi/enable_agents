// @ts-check
const { test, expect } = require('@playwright/test');
const { loginAsNewUser, deleteTestUser } = require('./helpers/auth');
const { createTestProject } = require('./helpers/project');

test.describe('Executive Assistant', () => {
  /** @type {string} */
  let token;
  /** @type {string} */
  let projectId;

  test.beforeEach(async ({ page, request }) => {
    ({ token } = await loginAsNewUser(page, request, 'execassist'));
    projectId = await createTestProject(request, token, `Executive Assistant Test ${Date.now()}`);
    await page.goto(`/executive-assistant?project=${projectId}`);
  });

  test.afterEach(async ({ request }) => {
    if (token) await deleteTestUser(request, token);
  });

  test('should create a task and see it on the board', async ({ page }) => {
    const taskTitle = `E2E test task ${Date.now()}`;
    const input = page.locator('input[placeholder="Add a new task..."]');
    await input.fill(taskTitle);
    await input.press('Enter');

    await expect(page.locator(`.kanban-card:has-text("${taskTitle}")`)).toBeVisible({ timeout: 10000 });
  });

  test('should add a stakeholder', async ({ page }) => {
    await page.click('.module-tab:has-text("Team")');
    await page.click('button:has-text("+ Person")');
    await page.fill('input[placeholder="John Smith"]', 'E2E Stakeholder');
    await page.fill('input[placeholder="john@company.com"]', 'e2e.stakeholder@example.com');
    await page.click('button:has-text("Add Person")');

    await expect(page.locator(':has-text("E2E Stakeholder")').last()).toBeVisible({ timeout: 10000 });
  });
});
