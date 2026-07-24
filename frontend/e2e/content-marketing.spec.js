// @ts-check
const { test, expect } = require('@playwright/test');
const { loginAsNewUser, deleteTestUser } = require('./helpers/auth');
const { createTestProject } = require('./helpers/project');

test.describe('Content Marketing', () => {
  /** @type {string} */
  let token;
  /** @type {string} */
  let projectId;

  test.beforeEach(async ({ page, request }) => {
    ({ token } = await loginAsNewUser(page, request, 'contentmkt'));
    projectId = await createTestProject(request, token, `Content Marketing Test ${Date.now()}`);
    await page.goto(`/content-marketing?project=${projectId}`);
  });

  test.afterEach(async ({ request }) => {
    if (token) await deleteTestUser(request, token);
  });

  test('should upload a document and generate content', async ({ page }) => {
    // Project resolution (creating/reusing the internal CM project) happens
    // async on load - give it a moment before the upload input is usable.
    await page.waitForTimeout(1500);

    await page.setInputFiles('input[type="file"]', {
      name: 'notes.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from(
        'Our product is a B2B SaaS platform for supply chain visibility. Target audience: procurement managers at mid-size manufacturers.'
      ),
    });

    // Upload triggers a step change to content generation once it succeeds.
    await expect(page.locator('button:has-text("Generate Content")')).toBeVisible({ timeout: 15000 });

    await page.click('button:has-text("Generate Content")');

    // Real LLM call - give it real time to respond.
    await expect(page.locator('h2, h3, h4').filter({ hasText: /Generated Content/i })).toBeVisible({
      timeout: 45000,
    });
  });
});
