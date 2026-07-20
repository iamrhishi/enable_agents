// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Agents Navigation', () => {
  test.beforeEach(async ({ page }) => {
    // Set up auth state
    await page.goto('/login');
    await page.evaluate(() => {
      localStorage.setItem('sessionToken', 'test-token');
      localStorage.setItem('userEmail', 'test@example.com');
    });
  });

  test('should display agents assembly page', async ({ page }) => {
    await page.goto('/agents');
    await expect(page.locator('h1, h2')).toContainText(/agent/i);
  });

  test('should show agent cards', async ({ page }) => {
    await page.goto('/agents');
    const cards = page.locator('.module-card, .agent-card, [data-testid="agent-card"]');
    await expect(cards.first()).toBeVisible();
  });

  test('should have Market Research agent', async ({ page }) => {
    await page.goto('/agents');
    await expect(page.locator(':has-text("Market Research")')).toBeVisible();
  });

  test('should have Content Marketing agent', async ({ page }) => {
    await page.goto('/agents');
    await expect(page.locator(':has-text("Content Marketing")')).toBeVisible();
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
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await page.evaluate(() => {
      localStorage.setItem('sessionToken', 'test-token');
      localStorage.setItem('userEmail', 'test@example.com');
    });
  });

  test('should navigate to settings page', async ({ page }) => {
    await page.goto('/settings');
    await expect(page.locator('h1, h2')).toContainText(/setting/i);
  });

  test('should display AI settings section', async ({ page }) => {
    await page.goto('/settings');
    await expect(page.locator(':has-text("AI"), :has-text("API")')).toBeVisible();
  });

  test('should display connectors section', async ({ page }) => {
    await page.goto('/settings');
    await expect(page.locator(':has-text("Connector")')).toBeVisible();
  });

  test('should have save button', async ({ page }) => {
    await page.goto('/settings');
    const saveBtn = page.locator('button:has-text("Save")');
    await expect(saveBtn.first()).toBeVisible();
  });
});
