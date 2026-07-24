// @ts-check
const { test, expect } = require('@playwright/test');
const { loginAsNewUser, deleteTestUser } = require('./helpers/auth');

test.describe('Settings - API Keys', () => {
  /** @type {string} */
  let token;

  test.beforeEach(async ({ page, request }) => {
    ({ token } = await loginAsNewUser(page, request, 'settingskeys'));
    await page.goto('/settings');
  });

  test.afterEach(async ({ request }) => {
    if (token) await deleteTestUser(request, token);
  });

  test('should save, show as configured, and delete an API key', async ({ page }) => {
    await page.click('.nav-item:has-text("AI Providers")');

    const row = page.locator('.setting-item:has-text("OpenAI API Key")');
    await expect(row).toBeVisible({ timeout: 10000 });

    await row.locator('input').fill('sk-e2e-test-fake-key-12345');
    await row.locator('button:has-text("Save")').click();

    await expect(row.locator('.configured-badge')).toBeVisible({ timeout: 10000 });

    await row.locator('button:has-text("Delete")').click();
    // Confirm the "Delete X? This action cannot be undone." dialog.
    await page.click('[aria-modal="true"] button:has-text("Delete")');

    await expect(row.locator('.configured-badge')).not.toBeVisible({ timeout: 10000 });
  });
});
