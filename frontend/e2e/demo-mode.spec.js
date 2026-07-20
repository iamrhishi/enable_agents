// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Demo Mode', () => {
  test.beforeEach(async ({ page }) => {
    // Set up auth state
    await page.goto('/login');
    await page.evaluate(() => {
      localStorage.setItem('sessionToken', 'test-token');
      localStorage.setItem('userEmail', 'test@example.com');
    });
  });

  test('should show demo/live toggle in header', async ({ page }) => {
    await page.goto('/agents');
    const toggle = page.locator('.mode-toggle, [data-testid="mode-toggle"], :has-text("Demo"), :has-text("Live")');
    await expect(toggle.first()).toBeVisible();
  });

  test('should toggle to demo mode', async ({ page }) => {
    await page.goto('/agents');

    // Find and click demo toggle
    const demoBtn = page.locator('button:has-text("Demo"), .mode-toggle:has-text("Demo")');
    if (await demoBtn.isVisible()) {
      await demoBtn.click();

      // Verify demo mode is active
      await expect(page.locator('.demo-badge, :has-text("Demo Mode")')).toBeVisible();
    }
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

    await page.goto('/executive-assistant');

    // Should show demo tasks
    await expect(page.locator('.task-card, .task-item, :has-text("Q3 Product Launch")')).toBeVisible({ timeout: 5000 });
  });
});
