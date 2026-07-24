// @ts-check
const { test, expect } = require('@playwright/test');
const { loginAsNewUser, deleteTestUser } = require('./helpers/auth');

test.describe('Agents Navigation', () => {
  /** @type {string} */
  let token;

  test.beforeEach(async ({ page, request }) => {
    // Real backend auth - a fake token would 401 on every API call now
    // that the backend verifies signed session tokens on every route.
    ({ token } = await loginAsNewUser(page, request, 'agentsnav'));
  });

  test.afterEach(async ({ request }) => {
    if (token) await deleteTestUser(request, token);
  });

  test('should display agents assembly page', async ({ page }) => {
    await page.goto('/agents');
    await expect(page.locator('h1, h2')).toContainText(/agent/i);
  });

  test('should show agent cards', async ({ page }) => {
    await page.goto('/agents');
    const cards = page.locator('.module-card, .carousel-3d-card, .agent-card, [data-testid="agent-card"]');
    await expect(cards.first()).toBeVisible();
  });

  test('should have Market Research agent', async ({ page }) => {
    await page.goto('/agents');
    await expect(page.locator(':has-text("Market Research")').last()).toBeVisible();
  });

  test('should have Content Marketing agent', async ({ page }) => {
    await page.goto('/agents');
    await expect(page.locator(':has-text("Content Marketing")').last()).toBeVisible();
  });

  test('should navigate to Market Research agent', async ({ page }) => {
    await page.goto('/agents');

    const tryBtn = page.locator('.module-card:has-text("Market Research") button:has-text("Try")');
    if (await tryBtn.isVisible()) {
      await tryBtn.click();
      await expect(page).toHaveURL(/market-research|requirements/);
    }
  });

  test('should have business/technical tabs', async ({ page }) => {
    await page.goto('/agents');
    const tabs = page.locator('.tab, [role="tab"]');
    await expect(tabs.first()).toBeVisible();
  });
});

test.describe('Settings', () => {
  /** @type {string} */
  let token;

  test.beforeEach(async ({ page, request }) => {
    ({ token } = await loginAsNewUser(page, request, 'settingsnav'));
  });

  test.afterEach(async ({ request }) => {
    if (token) await deleteTestUser(request, token);
  });

  test('should navigate to settings page', async ({ page }) => {
    await page.goto('/settings');
    await expect(page.locator('h1').first()).toContainText(/setting/i);
  });

  test('should display AI settings section', async ({ page }) => {
    await page.goto('/settings');
    await page.click('.nav-item:has-text("AI Providers")');
    await expect(page.locator(':has-text("OpenAI API Key")').last()).toBeVisible();
  });

  test('should display connectors section', async ({ page }) => {
    await page.goto('/settings');
    await page.click('.nav-item:has-text("Data Connectors")');
    await expect(page.locator(':has-text("Search API Key")').last()).toBeVisible();
  });

  test('should have save button', async ({ page }) => {
    await page.goto('/settings');
    const saveBtn = page.locator('button:has-text("Save")');
    await expect(saveBtn.first()).toBeVisible();
  });
});
