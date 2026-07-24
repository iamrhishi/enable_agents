// @ts-check
/**
 * Real project creation helper for e2e tests. Most agent pages are
 * project-scoped (?project=<id> in the URL) and now enforce that the
 * caller's session actually owns/has team access to that project id
 * (see backend/core/auth.py: user_can_access_project) - so tests must
 * create a real project via the API rather than making one up.
 */

const { API_BASE_URL } = require('./auth');

/**
 * @param {import('@playwright/test').APIRequestContext} request
 * @param {string} token
 * @param {string} [name]
 * @returns {Promise<string>} the created project's id
 */
async function createTestProject(request, token, name = `E2E Project ${Date.now()}`) {
  const res = await request.post(`${API_BASE_URL}/api/projects`, {
    headers: { Authorization: `Bearer ${token}` },
    data: { name, description: 'Created by e2e test suite' },
  });
  if (!res.ok()) {
    throw new Error(`createTestProject failed: ${res.status()} ${await res.text()}`);
  }
  const body = await res.json();
  return body.project.id;
}

module.exports = { createTestProject };
