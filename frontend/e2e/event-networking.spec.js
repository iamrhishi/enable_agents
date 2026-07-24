// @ts-check
const { test, expect } = require('@playwright/test');
const { loginAsNewUser, deleteTestUser } = require('./helpers/auth');
const { createTestProject } = require('./helpers/project');

test.describe('Event Networking', () => {
  /** @type {string} */
  let token;
  /** @type {string} */
  let projectId;

  test.beforeEach(async ({ page, request }) => {
    ({ token } = await loginAsNewUser(page, request, 'eventnet'));
    projectId = await createTestProject(request, token, `Event Networking Test ${Date.now()}`);
    await page.goto(`/event-networking?project=${projectId}`);
  });

  test.afterEach(async ({ request }) => {
    if (token) await deleteTestUser(request, token);
  });

  test('should create an event and see it in the list', async ({ page }) => {
    await page.click('button:has-text("+ New Event")');
    await page.fill('input[placeholder*="Tech Summit" i]', 'E2E Test Conference');
    await page.click('button:has-text("Create Event")');

    await expect(page.locator('.event-card:has-text("E2E Test Conference")')).toBeVisible({ timeout: 10000 });
  });

  test('should upload attendees and get recommendations', async ({ page }) => {
    await page.click('button:has-text("+ New Event")');
    await page.fill('input[placeholder*="Tech Summit" i]', 'E2E Attendees Event');
    await page.click('button:has-text("Create Event")');
    await expect(page.locator('.event-card:has-text("E2E Attendees Event")')).toBeVisible({ timeout: 10000 });
    await page.click('.event-card:has-text("E2E Attendees Event")');

    // Selecting an event switches to the Contacts tab automatically.
    await page.click('button:has-text("+ Import Contacts")');
    await page.fill('textarea.csv-input', 'Jane Doe,jane@example.com,Acme Inc,CTO,AI;Cloud');
    await page.click('button:has-text("Import")');

    await expect(page.locator(':has-text("Jane Doe")').last()).toBeVisible({ timeout: 10000 });
  });
});
