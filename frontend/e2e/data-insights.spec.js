// @ts-check
const { test, expect } = require('@playwright/test');
const { loginAsNewUser, deleteTestUser } = require('./helpers/auth');
const { createTestProject } = require('./helpers/project');

test.describe('Data Insights (Document Intelligence)', () => {
  /** @type {string} */
  let token;
  /** @type {string} */
  let projectId;

  test.beforeEach(async ({ page, request }) => {
    ({ token } = await loginAsNewUser(page, request, 'datainsights'));
    projectId = await createTestProject(request, token, `Data Insights Test ${Date.now()}`);
    await page.goto(`/data-insights?project=${projectId}`);
  });

  test.afterEach(async ({ request }) => {
    if (token) await deleteTestUser(request, token);
  });

  test('should upload a document, process it, and chat about it', async ({ page }) => {
    await page.setInputFiles('#file-input', {
      name: 'e2e-notes.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from(
        'Enable Agents is a platform for AI-powered business workflows. This test document mentions the keyword "unicorn-marker-42" exactly once.'
      ),
    });
    await page.click('button.di-upload-btn:has-text("Upload & Process")');

    const docCard = page.locator('.di-doc-card:has-text("e2e-notes.txt")');
    await expect(docCard).toBeVisible({ timeout: 10000 });

    // Real background processing (extraction + embeddings) - poll for completion.
    await expect(docCard.locator('.di-doc-status--completed')).toBeVisible({ timeout: 60000 });

    await docCard.click();
    await page.click('button:has-text("Ask AI"), .di-tab:has-text("Ask AI")');

    const chatInput = page.locator('textarea[placeholder="Ask anything about this document..."]');
    await expect(chatInput).toBeVisible({ timeout: 10000 });
    await chatInput.fill('What keyword does this document mention?');
    await chatInput.press('Enter');

    // Real LLM + vector search round trip.
    await expect(page.locator('.di-chat-view')).toContainText(/unicorn|keyword/i, {
      timeout: 30000,
    });
  });
});
