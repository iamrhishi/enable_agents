// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Projects', () => {
  test.beforeEach(async ({ page }) => {
    // Set up auth state
    await page.goto('/login');
    await page.evaluate(() => {
      localStorage.setItem('sessionToken', 'test-token');
      localStorage.setItem('userEmail', 'test@example.com');
    });
  });

  test('should navigate to projects page', async ({ page }) => {
    await page.goto('/projects');
    await expect(page.locator('h1')).toContainText(/project/i);
  });

  test('should display create project button', async ({ page }) => {
    await page.goto('/projects');
    const createBtn = page.locator('button:has-text("Create"), button:has-text("New Project")');
    await expect(createBtn).toBeVisible();
  });

  test('should open project creation modal', async ({ page }) => {
    await page.goto('/projects');
    const createBtn = page.locator('button:has-text("Create"), button:has-text("New Project")');
    await createBtn.click();

    // Modal should appear
    const modal = page.locator('[role="dialog"], .modal, .project-modal');
    await expect(modal).toBeVisible();
  });

  test('project form should have name field', async ({ page }) => {
    await page.goto('/projects');
    const createBtn = page.locator('button:has-text("Create"), button:has-text("New Project")');
    await createBtn.click();

    const nameInput = page.locator('input[name="name"], input[placeholder*="name" i]');
    await expect(nameInput).toBeVisible();
  });

  test('should create a new project', async ({ page }) => {
    await page.goto('/projects');
    const createBtn = page.locator('button:has-text("Create"), button:has-text("New Project")');
    await createBtn.click();

    const nameInput = page.locator('input[name="name"], input[placeholder*="name" i]');
    await nameInput.fill('Test Project ' + Date.now());

    const submitBtn = page.locator('button[type="submit"], button:has-text("Save"), button:has-text("Create")');
    await submitBtn.last().click();

    // Should show success or new project in list
    await expect(
      page.locator('.project-card, .project-item, .toast-success, :has-text("Test Project")')
    ).toBeVisible({ timeout: 5000 });
  });
});
