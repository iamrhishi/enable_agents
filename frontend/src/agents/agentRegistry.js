/**
 * Agent Registry
 *
 * Fetches the list of enabled agents from /api/v1/agents and provides
 * helpers used by AgentsAssembly to render the correct card and route.
 *
 * Each backend agent manifest must include:
 *   id, name, description, enabled, routes_prefix
 *
 * Front-end extras (icon, price, status, route) are maintained here as an
 * overlay map so the backend stays UI-agnostic.
 */

import { API_CONFIG } from '../config/apiConfig';

/** Static metadata keyed by agent id — add an entry when creating a new agent. */
const AGENT_UI_META = {
  content_marketing: {
    icon: '/assets/icons/bullhorn.png',
    price: '$49/month',
    status: 'in-progress',
    route: '/content-marketing',
    tab: 'business',
    keywords: ['content marketing', 'content creation', 'marketing strategy', 'SEO'],
    industries: ['retail', 'technology', 'media', 'education', 'ecommerce'],
  },
  // Add more agents here as they are migrated to the backend registry
};

/**
 * Fetch agent list from the backend registry endpoint.
 * Falls back to an empty array if the endpoint is unavailable.
 *
 * @returns {Promise<Array>} List of agent objects (manifest + ui meta merged)
 */
export async function fetchAgents() {
  try {
    const res = await fetch(`${API_CONFIG.API_URL}/api/v1/agents/`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const agents = await res.json();
    return agents
      .filter((a) => a.enabled)
      .map((a) => ({ ...a, ...(AGENT_UI_META[a.id] || {}) }));
  } catch (err) {
    console.warn('[agentRegistry] Could not fetch agents from backend:', err.message);
    return [];
  }
}

/**
 * Toggle an agent's enabled state via the backend API.
 *
 * @param {string} agentId
 * @param {boolean} enabled
 * @returns {Promise<object>} Updated agent manifest
 */
export async function toggleAgent(agentId, enabled) {
  const res = await fetch(`${API_CONFIG.API_URL}/api/v1/agents/${agentId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled }),
  });
  if (!res.ok) throw new Error(`Toggle failed: HTTP ${res.status}`);
  return res.json();
}
