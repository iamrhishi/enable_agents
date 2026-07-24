// @ts-check
const { test, expect } = require('@playwright/test');
const { loginAsNewUser, deleteTestUser } = require('./helpers/auth');

test.describe('Team', () => {
  /** @type {string} */
  let token;

  test.beforeEach(async ({ page, request }) => {
    ({ token } = await loginAsNewUser(page, request, 'team'));
  });

  test.afterEach(async ({ request }) => {
    if (token) await deleteTestUser(request, token);
  });

  test('should load the team page and show the current user as a member', async ({ page }) => {
    await page.goto('/team');
    await expect(page.locator('h1')).toContainText(/team/i);
    // A brand-new user auto-creates a team with themselves as owner.
    await expect(page.locator(':has-text("Owner")').last()).toBeVisible({ timeout: 10000 });
  });
});
