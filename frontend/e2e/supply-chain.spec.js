// @ts-check
const { test, expect } = require('@playwright/test');
const { loginAsNewUser, deleteTestUser } = require('./helpers/auth');
const { createTestProject } = require('./helpers/project');

test.describe('Supply Chain', () => {
  /** @type {string} */
  let token;
  /** @type {string} */
  let projectId;

  test.beforeEach(async ({ page, request }) => {
    ({ token } = await loginAsNewUser(page, request, 'supplychain'));
    projectId = await createTestProject(request, token, `Supply Chain Test ${Date.now()}`);
    await page.goto(`/supply-chain-agent?project=${projectId}`);
  });

  test.afterEach(async ({ request }) => {
    if (token) await deleteTestUser(request, token);
  });

  test('should add a supplier and submit an audit', async ({ page }) => {
    await page.click('button:has-text("+ Add Supplier")');
    await page.fill('input[placeholder*="Precision Components" i]', 'E2E Test Supplier Co');
    await page.click('.audit-modal button:has-text("Add Supplier")');

    const card = page.locator('.supplier-card:has-text("E2E Test Supplier Co")');
    await expect(card).toBeVisible({ timeout: 10000 });

    await card.locator('button:has-text("Start Audit")').click();
    // Fill every score input with a passing value.
    const scoreInputs = page.locator('.audit-modal input[type="number"]');
    const count = await scoreInputs.count();
    for (let i = 0; i < count; i++) {
      await scoreInputs.nth(i).fill('90');
    }
    await page.click('button:has-text("Submit Audit")');

    await expect(page.locator('.supplier-card:has-text("E2E Test Supplier Co") .status-badge')).toContainText(
      /passed/i,
      { timeout: 10000 }
    );
  });
});
