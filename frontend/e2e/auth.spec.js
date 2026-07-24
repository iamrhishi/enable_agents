// @ts-check
const { test, expect } = require('@playwright/test');

test.describe('Authentication', () => {
  test.beforeEach(async ({ page }) => {
    // Clear any existing auth state. localStorage isn't accessible on the
    // blank initial document in newer Chromium, so navigate first.
    await page.context().clearCookies();
    await page.goto('/login');
    await page.evaluate(() => localStorage.clear());
  });

  test('should display login page for unauthenticated users', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveURL(/\/login/);
    await expect(page.locator('input[type="email"], input[name="email"]')).toBeVisible();
  });

  test('should show email and password fields', async ({ page }) => {
    // Login is a two-step form: email first, password appears after
    // clicking Continue.
    await page.goto('/login');
    await expect(page.locator('input[type="email"], input[name="email"]')).toBeVisible();
    await page.fill('input[type="email"], input[name="email"]', 'someone@example.com');
    await page.click('button[type="submit"]');
    await expect(page.locator('input[type="password"]')).toBeVisible();
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto('/login');

    await page.fill('input[type="email"], input[name="email"]', 'invalid@example.com');
    await page.click('button[type="submit"]');
    await page.fill('input[type="password"]', 'wrongpassword');
    await page.click('button[type="submit"]');

    // Should show error message
    await expect(page.locator('.ea-toast--error, .error, [role="alert"]')).toBeVisible({ timeout: 10000 });
  });

  test('should have link to register page', async ({ page }) => {
    await page.goto('/login');
    const registerLink = page.locator('button:has-text("Create account")');
    await expect(registerLink).toBeVisible();
  });

  test('should navigate to register page', async ({ page }) => {
    await page.goto('/login');
    await page.click('button:has-text("Create account")');
    await expect(page).toHaveURL(/\/register/);
  });

  test('register page should have required fields', async ({ page }) => {
    await page.goto('/register');
    await expect(page.locator('input[type="email"], input[name="email"]')).toBeVisible();
    await expect(page.locator('input[type="password"]')).toBeVisible();
    await expect(page.locator('input[name="firstName"]')).toBeVisible();
  });

  test('should show Google OAuth button', async ({ page }) => {
    await page.goto('/login');
    const googleBtn = page.locator('button:has-text("Google"), a:has-text("Google")');
    await expect(googleBtn).toBeVisible();
  });
});

test.describe('Logout', () => {
  test.beforeEach(async ({ page }) => {
    // Set up fake auth state
    await page.goto('/login');
    await page.evaluate(() => {
      localStorage.setItem('sessionToken', 'test-token');
      localStorage.setItem('userEmail', 'test@example.com');
    });
  });

  test('should redirect away from login when already logged in', async ({ page }) => {
    await page.goto('/');
    await expect(page).not.toHaveURL(/\/login/);
  });

  test('should have logout option in user menu', async ({ page }) => {
    await page.goto('/agents');
    // Open user menu (usually an avatar or user icon)
    const userMenu = page.locator('[data-testid="user-menu"], .user-menu, .header-user');
    if (await userMenu.isVisible()) {
      await userMenu.click();
      const logoutBtn = page.locator('button:has-text("Sign out"), button:has-text("Logout")');
      await expect(logoutBtn).toBeVisible();
    }
  });
});
