// @ts-check
const { test, expect } = require('@playwright/test');
const { loginAsNewUser, deleteTestUser } = require('./helpers/auth');

test.describe('Demo Mode', () => {
  /** @type {string} */
  let token;

  test.beforeEach(async ({ page, request }) => {
    ({ token } = await loginAsNewUser(page, request, 'demomode'));
  });

  test.afterEach(async ({ request }) => {
    if (token) await deleteTestUser(request, token);
  });

  test('should show demo/live toggle in Settings > Preferences', async ({ page }) => {
    await page.goto('/settings');
    await page.click('.nav-item:has-text("Preferences")');
    await expect(page.locator('.mode-toggle-large')).toBeVisible();
  });

  test('should toggle to demo mode', async ({ page }) => {
    await page.goto('/settings');
    await page.click('.nav-item:has-text("Preferences")');

    const toggle = page.locator('.mode-toggle-large');
    await expect(toggle).toBeVisible();
    const wasLive = (await toggle.getAttribute('class'))?.includes('--live');
    await toggle.click();

    await expect(toggle).toHaveClass(wasLive ? /--demo/ : /--live/);
  });

  test('should persist mode selection', async ({ page }) => {
    await page.goto('/agents');

    // Set to demo mode
    await page.evaluate(() => {
      localStorage.setItem('enableAgentsMode', 'demo');
    });

    // Reload and check
    await page.reload();
    const mode = await page.evaluate(() => localStorage.getItem('enableAgentsMode'));
    expect(mode).toBe('demo');
  });

  test('demo mode should show sample data in Executive Assistant', async ({ page }) => {
    await page.evaluate(() => {
      localStorage.setItem('enableAgentsMode', 'demo');
    });

    // ProjectGate requires a ?project= param regardless of demo mode - the
    // id itself is never validated against the backend in demo mode.
    await page.goto('/executive-assistant?project=demo-project-1');

    // Should show demo tasks
    await expect(page.locator('.kanban-card').first()).toBeVisible({ timeout: 5000 });
  });
});
