// @ts-check
const { test, expect } = require('@playwright/test');
const { loginAsNewUser, deleteTestUser } = require('./helpers/auth');

test.describe('Projects', () => {
  /** @type {string} */
  let token;

  test.beforeEach(async ({ page, request }) => {
    ({ token } = await loginAsNewUser(page, request, 'projects'));
  });

  test.afterEach(async ({ request }) => {
    if (token) await deleteTestUser(request, token);
  });

  test('should navigate to projects page', async ({ page }) => {
    await page.goto('/projects');
    await expect(page.locator('h1')).toContainText(/project/i);
  });

  test('should display create project button', async ({ page }) => {
    await page.goto('/projects');
    const createBtn = page.locator('.btn-create, .empty-state-card .btn-primary').first();
    await expect(createBtn).toBeVisible();
  });

  test('should open project creation modal', async ({ page }) => {
    await page.goto('/projects');
    const createBtn = page.locator('.btn-create, .empty-state-card .btn-primary').first();
    await createBtn.click();

    // Modal should appear
    const modal = page.locator('[role="dialog"], .modal, .project-modal');
    await expect(modal).toBeVisible();
  });

  test('project form should have name field', async ({ page }) => {
    await page.goto('/projects');
    const createBtn = page.locator('.btn-create, .empty-state-card .btn-primary').first();
    await createBtn.click();

    const nameInput = page.locator('input[placeholder*="Marketing Campaign" i]');
    await expect(nameInput).toBeVisible();
  });

  test('should create a new project', async ({ page }) => {
    await page.goto('/projects');
    const createBtn = page.locator('.btn-create, .empty-state-card .btn-primary').first();
    await createBtn.click();

    const nameInput = page.locator('input[placeholder*="Marketing Campaign" i]');
    await nameInput.fill('Test Project ' + Date.now());

    const submitBtn = page.locator('button[type="submit"], button:has-text("Save"), button:has-text("Create")');
    await submitBtn.last().click();

    // Should show success or new project in list
    await expect(page.locator('.project-card:has-text("Test Project")')).toBeVisible({ timeout: 5000 });
  });
});
