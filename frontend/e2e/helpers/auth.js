// @ts-check
/**
 * Real-auth helper for e2e tests.
 *
 * The backend requires a genuine signed session token on every API route
 * (see backend/core/auth.py) - a fake localStorage token is no longer
 * enough to make authenticated API calls succeed. This helper registers
 * (or logs in) a real throwaway user directly against the backend API and
 * returns a token that will actually pass verification, then seeds
 * localStorage so the app treats the browser session as logged in too.
 */

const API_BASE_URL = process.env.API_BASE_URL || 'http://localhost:8000';

function uniqueEmail(prefix = 'e2e') {
  return `${prefix}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}@enableyou.co`;
}

/**
 * Register a brand-new throwaway user via the real API.
 * @param {import('@playwright/test').APIRequestContext} request
 * @param {string} [prefix]
 * @returns {Promise<{ email: string, token: string }>}
 */
async function registerTestUser(request, prefix = 'e2e') {
  const email = uniqueEmail(prefix);
  const password = 'TestPass123!';

  const res = await request.post(`${API_BASE_URL}/register`, {
    data: { email, password, username: email },
  });
  if (!res.ok()) {
    throw new Error(`registerTestUser failed: ${res.status()} ${await res.text()}`);
  }
  const body = await res.json();
  return { email, token: body.session_token };
}

/**
 * Delete a throwaway user via the real API (best-effort cleanup).
 * @param {import('@playwright/test').APIRequestContext} request
 * @param {string} token
 */
async function deleteTestUser(request, token) {
  try {
    await request.delete(`${API_BASE_URL}/api/account`, {
      headers: { Authorization: `Bearer ${token}` },
    });
  } catch {
    // Best-effort cleanup - a failure here shouldn't fail the test itself.
  }
}

/**
 * Log a Playwright page in as a real, freshly-registered user by seeding
 * localStorage with a genuine backend-issued session token.
 * @param {import('@playwright/test').Page} page
 * @param {import('@playwright/test').APIRequestContext} request
 * @param {string} [prefix]
 * @returns {Promise<{ email: string, token: string }>}
 */
async function loginAsNewUser(page, request, prefix = 'e2e') {
  const { email, token } = await registerTestUser(request, prefix);
  await page.goto('/login');
  await page.evaluate(({ token, email }) => {
    localStorage.setItem('sessionToken', token);
    localStorage.setItem('userEmail', email);
    localStorage.setItem('firstName', email.split('@')[0]);
    localStorage.setItem('enableAgentsMode', 'live');
  }, { token, email });
  return { email, token };
}

module.exports = { API_BASE_URL, uniqueEmail, registerTestUser, deleteTestUser, loginAsNewUser };
